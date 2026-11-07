/* A small RV32IM + Zba + Zbb + Zbs + Zalrsc interpreter, built as a host DLL and driven from
 * Python (rvhost.py). It counts every retired instruction by mnemonic and by PC.
 * rv_run() returns on ecall (1), illegal instruction (2), ebreak (3), the step
 * limit (4), a memory fault (5) or a failed unit check (6); the host handles the ecall
 * and resumes.
 *
 * circuits.md additions:
 * - a token model: every retired instruction is charged the tokens of its record in the
 *   Qwenputer trace format (chipformer/rv/spec.py: value records of 1-3 tokens by size, SB/SH
 *   4, taken jumps J, an R refresh after 63 commits without a jump);
 * - exact dedicated units ("circuits", circ_units.h): macro-uops that fire at fixed ROM PCs
 *   of this fx build, with the architectural effect of the original instructions, and a
 *   verify mode that also steps the original code from the same state and compares;
 * - poison checks: registers / dead stack bytes a unit leaves different from the original
 *   must be written before they are read (counted as violations otherwise).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define MEMSZ 0x08000000u

enum {
  LUI, AUIPC, JAL, JALR, BEQ, BNE, BLT, BGE, BLTU, BGEU, LB, LH, LW, LBU, LHU, SB, SH, SW,
  ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI, ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA,
  OR, AND, FENCE, ECALL, EBREAK, MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU, SH1ADD, SH2ADD,
  SH3ADD, LRW, SCW, CSR,
  /* Zbb / Zbs (speed-iter: candidate ISA extensions; immediate forms count with their R form) */
  ANDN, ORN, XNOR, CLZ, CTZ, CPOP, SEXTB, SEXTH, ZEXTH, MAX, MAXU, MIN, MINU, ROL, ROR, ORCB, REV8,
  BCLR, BEXT, BINV, BSET, NOPS
};

static uint8_t* M;
static uint32_t X[32];
static uint32_t PC;
static uint64_t icount;
static uint64_t ops[NOPS];
static uint64_t misaligned;
static uint32_t* pcc;
static uint32_t* mpc; /* misaligned accesses by PC */
static uint32_t text_lo, text_hi;
static uint32_t resv = 0xffffffffu;
static uint32_t fault_addr;
/* Profiling extras (rv_set_prof): mnemonic pairs and triples (all, and pairs where the
 * second instruction reads the first one's rd), bytes loaded/stored per PC, and
 * misaligned accesses that would fault on an aligned-only core. */
static int prof;
static uint64_t pair[NOPS * NOPS], pair_dep[NOPS * NOPS], trip[NOPS * NOPS * NOPS];
static int k1 = 0, k2 = 0;
static uint32_t prev_rd = 0;
static uint32_t* pcbytes; /* bytes loaded + stored, per PC */
/* Inclusive cost per function entry (prof): a call (JAL/JALR linking ra) pushes the
 * target, `ret` pops it; only the outermost activation of a function is counted. */
static uint64_t* incl;
static uint32_t* incl_depth;
#define CSTK 65536
static uint32_t cs_entry[CSTK];
static uint64_t cs_start[CSTK];
static int csp;
/* Call sites of one watched function (rv_set_watch): calls and the sum of a2 (memcpy's n). */
static uint32_t watch_pc = 0xffffffffu;
static uint32_t* wcalls;
static uint64_t* wbytes;
#define mt_icount_now icount
static int nfz; /* read-only ranges a unit bakes in (circ_loops.h) */
static void frozen_store_check(uint32_t a, uint32_t n);
#include "memtrace.h"

/* ---- token model (spec.py's record format) ---- */
static uint64_t tokens;      /* tokens of the trace so far */
static uint32_t tn;          /* commits since the last J / R (the R refresh comes at 63) */
static uint32_t* tokpc;      /* tokens charged per PC */
static int shadow;           /* stepping the original code for a unit check: no profiling */
static inline uint32_t vlen(uint32_t v) {
  int32_t s = (int32_t)v;
  return (s >= -(1 << 14) && s < (1 << 14)) ? 1 : (s >= -(1 << 29) && s < (1 << 29)) ? 2 : 3;
}
static inline uint32_t nib_used(uint32_t v) { uint32_t k = 0; for (int i = 0; i < 8; i++) if (v >> (4 * i)) k++; return k; }
static inline uint32_t div_k0(int sgn, uint32_t a, uint32_t b) { /* ref.div_start */
  if (b == 0) return 0;
  uint32_t da = nib_used(a), db = nib_used((sgn && (b >> 31)) ? ~b : b);
  int k = 7 - (int)da + (int)db;
  return k < 0 ? 0 : k > 8 ? 8 : (uint32_t)k;
}

/* ---- store log (unit checks) and poison ---- */
typedef struct { uint32_t a; uint8_t old; } logent_t;
static logent_t* lg;
static size_t lgn, lgcap;
static int log_on;
static inline void log_byte(uint32_t a) {
  if (lgn == lgcap) { lgcap = lgcap ? 2 * lgcap : 1 << 16; lg = realloc(lg, lgcap * sizeof(logent_t)); }
  lg[lgn].a = a; lg[lgn].old = M[a]; lgn++;
}
static inline void log_store(uint32_t a, uint32_t n) { if (log_on) for (uint32_t i = 0; i < n; i++) log_byte(a + i); }
#define PSTK_LO 0x07800000u /* native C stack region [PSTK_LO, MEMSZ): dead below sp at a return */
#define WSTK_HI 0x00100000u /* wasm shadow stack [0, 1 MB): dead below __stack_pointer at a return */
static uint32_t pois;        /* registers whose value differs from the original's (dead at a unit exit) */
static uint8_t* pbits;       /* bitmap: bytes that differ from the original's (dead at a unit exit) */
static uint32_t npmem;
static uint64_t viol_reg, viol_mem;
static uint32_t viol_pc, viol_what;
#define PBIT(a) (pbits[(a) >> 3] >> ((a) & 7) & 1)
static inline void preg(uint32_t r) {
  if (r && (pois >> r & 1)) { if (!viol_reg++) { viol_pc = PC; viol_what = r; } }
}
static inline void pload(uint32_t a, uint32_t n) {
  if (npmem) for (uint32_t i = 0; i < n; i++) if (a + i < MEMSZ && PBIT(a + i)) {
    if (!viol_mem++) { viol_pc = PC; viol_what = a + i; } }
}
static inline void pstore(uint32_t a, uint32_t n) {
  if (npmem) for (uint32_t i = 0; i < n; i++) if (a + i < MEMSZ && PBIT(a + i)) { pbits[(a + i) >> 3] &= (uint8_t)~(1u << ((a + i) & 7)); npmem--; }
}
static inline void ppoison(uint32_t a) { if (!PBIT(a)) { pbits[a >> 3] |= (uint8_t)(1u << (a & 7)); npmem++; } }
EXPORT uint64_t rv_tokens(void) { return tokens; }
EXPORT uint32_t* rv_tokpc(void) { return tokpc; }
EXPORT void rv_violations(uint64_t* out) { out[0] = viol_reg; out[1] = viol_mem; out[2] = viol_pc; out[3] = viol_what; out[4] = pois; out[5] = npmem; }

EXPORT uint8_t* rv_mem(void) { return M; }
EXPORT uint32_t* rv_regs(void) { return X; }
EXPORT uint32_t rv_get_pc(void) { return PC; }
EXPORT void rv_set_pc(uint32_t pc) { PC = pc; }
EXPORT uint64_t rv_icount(void) { return icount; }
EXPORT uint64_t* rv_ops(void) { return ops; }
EXPORT uint64_t rv_misaligned(void) { return misaligned; }
EXPORT uint32_t* rv_pccounts(void) { return pcc; }
EXPORT uint32_t* rv_mispc(void) { return mpc; }
EXPORT uint32_t rv_fault_addr(void) { return fault_addr; }
EXPORT int rv_nops(void) { return NOPS; }
EXPORT void rv_set_prof(int on) { prof = on; }
EXPORT uint64_t* rv_pairs(void) { return pair; }
EXPORT uint64_t* rv_pairs_dep(void) { return pair_dep; }
EXPORT uint64_t* rv_trips(void) { return trip; }
EXPORT uint32_t* rv_pcbytes(void) { return pcbytes; }
EXPORT uint64_t* rv_incl(void) { return incl; }
EXPORT void rv_set_watch(uint32_t pc) { watch_pc = pc; }
EXPORT uint32_t* rv_wcalls(void) { return wcalls; }
EXPORT uint64_t* rv_wbytes(void) { return wbytes; }

EXPORT int rv_init(uint32_t lo, uint32_t hi) {
  M = calloc(1, MEMSZ);
  text_lo = lo;
  text_hi = hi;
  pcc = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  mpc = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  pcbytes = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  incl = calloc((hi - lo) / 4 + 1, sizeof(uint64_t));
  incl_depth = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  wcalls = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  wbytes = calloc((hi - lo) / 4 + 1, sizeof(uint64_t));
  tokpc = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  pbits = calloc(MEMSZ / 8, 1);
  return M && tokpc && pbits && pcc && mpc && pcbytes && incl && incl_depth ? 0 : -1;
}

static inline int ok(uint32_t a, uint32_t n) { return a < MEMSZ && a + n <= MEMSZ; }
static inline uint32_t rotl(uint32_t a, uint32_t n) { n &= 31; return n ? (a << n) | (a >> (32 - n)) : a; }
static inline uint32_t rotr(uint32_t a, uint32_t n) { n &= 31; return n ? (a >> n) | (a << (32 - n)) : a; }

/* One instruction (0: retired; else rv_run's return code, PC not advanced). */
static int step(void) {
  {
    if (!ok(PC, 4) || (PC & 3)) { fault_addr = PC; return 5; }
    uint32_t in;
    memcpy(&in, M + PC, 4);
    uint32_t op = in & 0x7f, rd = (in >> 7) & 31, f3 = (in >> 12) & 7, rs1 = (in >> 15) & 31,
             rs2 = (in >> 20) & 31, f7 = in >> 25;
    uint32_t a = X[rs1], b = X[rs2], next = PC + 4, v = 0;
    int32_t immi = (int32_t)in >> 20;
    int k = -1;
    int wr = 1;
    uint32_t tref = 0;
    int jt = 0;
    if (tn == 63) { tref = 1; tn = 0; }
    if (pois) {
      if (op == 0x67 || op == 0x03 || op == 0x13 || op == 0x63 || op == 0x23 || op == 0x33 || op == 0x2f) preg(rs1);
      if (op == 0x63 || op == 0x23 || op == 0x33 || op == 0x2f) preg(rs2);
      if (in == 0x00000073u) { /* what the OS reads (rvhost.py): a7, then the call's arguments */
        preg(17);
        if (X[17] >= 0x400 || X[17] == 96) { preg(10); preg(11); }
        else if (X[17] == 93 || X[17] == 94 || X[17] == 97) preg(10);
      }
    }
    switch (op) {
      case 0x37: k = LUI; v = in & 0xfffff000u; break;
      case 0x17: k = AUIPC; v = PC + (in & 0xfffff000u); break;
      case 0x6f: {
        k = JAL;
        int32_t im = (((int32_t)in >> 31) << 20) | (((in >> 12) & 0xff) << 12) | (((in >> 20) & 1) << 11) |
                     (((in >> 21) & 0x3ff) << 1);
        v = next;
        next = PC + im;
        break;
      }
      case 0x67: k = JALR; v = next; next = (a + immi) & ~1u; break;
      case 0x63: {
        int32_t im = (((int32_t)in >> 31) << 12) | (((in >> 7) & 1) << 11) | (((in >> 25) & 0x3f) << 5) |
                     (((in >> 8) & 0xf) << 1);
        int t;
        switch (f3) {
          case 0: k = BEQ; t = a == b; break;
          case 1: k = BNE; t = a != b; break;
          case 4: k = BLT; t = (int32_t)a < (int32_t)b; break;
          case 5: k = BGE; t = (int32_t)a >= (int32_t)b; break;
          case 6: k = BLTU; t = a < b; break;
          case 7: k = BGEU; t = a >= b; break;
          default: return 2;
        }
        if (t) { next = PC + im; jt = 1; }
        wr = 0;
        break;
      }
      case 0x03: {
        uint32_t ad = a + immi;
        uint32_t n = (f3 & 3) == 0 ? 1 : (f3 & 3) == 1 ? 2 : 4;
        if (!ok(ad, n)) { fault_addr = ad; return 5; }
        if (ad & (n - 1)) { misaligned++; if (PC >= text_lo && PC < text_hi) mpc[(PC - text_lo) >> 2]++; }
        pload(ad, n);
        mt_load(ad, n);
        if (prof && PC >= text_lo && PC < text_hi) pcbytes[(PC - text_lo) >> 2] += n;
        switch (f3) {
          case 0: k = LB; v = (uint32_t)(int32_t)(int8_t)M[ad]; break;
          case 1: { int16_t h; memcpy(&h, M + ad, 2); k = LH; v = (uint32_t)(int32_t)h; break; }
          case 2: k = LW; memcpy(&v, M + ad, 4); break;
          case 4: k = LBU; v = M[ad]; break;
          case 5: { uint16_t h; memcpy(&h, M + ad, 2); k = LHU; v = h; break; }
          default: return 2;
        }
        break;
      }
      case 0x23: {
        int32_t im = (((int32_t)in >> 25) << 5) | ((in >> 7) & 31);
        uint32_t ad = a + im;
        uint32_t n = f3 == 0 ? 1 : f3 == 1 ? 2 : 4;
        if (f3 > 2) return 2;
        if (!ok(ad, n)) { fault_addr = ad; return 5; }
        if (ad & (n - 1)) { misaligned++; if (PC >= text_lo && PC < text_hi) mpc[(PC - text_lo) >> 2]++; }
        k = f3 == 0 ? SB : f3 == 1 ? SH : SW;
        log_store(ad, n);
        pstore(ad, n);
        if (nfz) frozen_store_check(ad, n);
        memcpy(M + ad, &b, n);
        mt_store(ad, n);
        if (prof && PC >= text_lo && PC < text_hi) pcbytes[(PC - text_lo) >> 2] += n;
        if ((ad & ~3u) == resv) resv = 0xffffffffu;
        wr = 0;
        break;
      }
      case 0x13:
        switch (f3) {
          case 0: k = ADDI; v = a + immi; break;
          case 2: k = SLTI; v = (int32_t)a < immi; break;
          case 3: k = SLTIU; v = a < (uint32_t)immi; break;
          case 4: k = XORI; v = a ^ immi; break;
          case 6: k = ORI; v = a | immi; break;
          case 7: k = ANDI; v = a & immi; break;
          case 1:
            if (f7 == 0) { k = SLLI; v = a << rs2; }
            else if (f7 == 0x30 && rs2 == 0) { k = CLZ; v = a ? (uint32_t)__builtin_clz(a) : 32; }
            else if (f7 == 0x30 && rs2 == 1) { k = CTZ; v = a ? (uint32_t)__builtin_ctz(a) : 32; }
            else if (f7 == 0x30 && rs2 == 2) { k = CPOP; v = (uint32_t)__builtin_popcount(a); }
            else if (f7 == 0x30 && rs2 == 4) { k = SEXTB; v = (uint32_t)(int32_t)(int8_t)a; }
            else if (f7 == 0x30 && rs2 == 5) { k = SEXTH; v = (uint32_t)(int32_t)(int16_t)a; }
            else if (f7 == 0x24) { k = BCLR; v = a & ~(1u << rs2); }
            else if (f7 == 0x34) { k = BINV; v = a ^ (1u << rs2); }
            else if (f7 == 0x14) { k = BSET; v = a | (1u << rs2); }
            else return 2;
            break;
          case 5:
            if (f7 == 0) { k = SRLI; v = a >> rs2; }
            else if (f7 == 0x20) { k = SRAI; v = (uint32_t)((int32_t)a >> rs2); }
            else if (f7 == 0x30) { k = ROR; v = rotr(a, rs2); }
            else if (f7 == 0x14 && rs2 == 7) { k = ORCB; v = 0; for (int i = 0; i < 32; i += 8) if ((a >> i) & 0xff) v |= 0xffu << i; }
            else if (f7 == 0x34 && rs2 == 24) { k = REV8; v = __builtin_bswap32(a); }
            else if (f7 == 0x24) { k = BEXT; v = (a >> rs2) & 1; }
            else return 2;
            break;
        }
        break;
      case 0x33:
        if (f7 == 0) {
          switch (f3) {
            case 0: k = ADD; v = a + b; break;
            case 1: k = SLL; v = a << (b & 31); break;
            case 2: k = SLT; v = (int32_t)a < (int32_t)b; break;
            case 3: k = SLTU; v = a < b; break;
            case 4: k = XOR; v = a ^ b; break;
            case 5: k = SRL; v = a >> (b & 31); break;
            case 6: k = OR; v = a | b; break;
            case 7: k = AND; v = a & b; break;
          }
        } else if (f7 == 0x20) {
          if (f3 == 0) { k = SUB; v = a - b; }
          else if (f3 == 5) { k = SRA; v = (uint32_t)((int32_t)a >> (b & 31)); }
          else if (f3 == 7) { k = ANDN; v = a & ~b; }
          else if (f3 == 6) { k = ORN; v = a | ~b; }
          else if (f3 == 4) { k = XNOR; v = ~(a ^ b); }
          else return 2;
        } else if (f7 == 0x05) {
          if (f3 == 4) { k = MIN; v = (int32_t)a < (int32_t)b ? a : b; }
          else if (f3 == 5) { k = MINU; v = a < b ? a : b; }
          else if (f3 == 6) { k = MAX; v = (int32_t)a > (int32_t)b ? a : b; }
          else if (f3 == 7) { k = MAXU; v = a > b ? a : b; }
          else return 2;
        } else if (f7 == 0x30) {
          if (f3 == 1) { k = ROL; v = rotl(a, b); }
          else if (f3 == 5) { k = ROR; v = rotr(a, b); }
          else return 2;
        } else if (f7 == 0x04 && f3 == 4 && rs2 == 0) {
          k = ZEXTH; v = a & 0xffff;
        } else if (f7 == 0x24) {
          if (f3 == 1) { k = BCLR; v = a & ~(1u << (b & 31)); }
          else if (f3 == 5) { k = BEXT; v = (a >> (b & 31)) & 1; }
          else return 2;
        } else if (f7 == 0x34 && f3 == 1) {
          k = BINV; v = a ^ (1u << (b & 31));
        } else if (f7 == 0x14 && f3 == 1) {
          k = BSET; v = a | (1u << (b & 31));
        } else if (f7 == 1) {
          switch (f3) {
            case 0: k = MUL; v = a * b; break;
            case 1: k = MULH; v = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32); break;
            case 2: k = MULHSU; v = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(uint64_t)b) >> 32); break;
            case 3: k = MULHU; v = (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); break;
            case 4: k = DIV;
              v = b == 0 ? 0xffffffffu : (a == 0x80000000u && b == 0xffffffffu) ? a : (uint32_t)((int32_t)a / (int32_t)b);
              break;
            case 5: k = DIVU; v = b == 0 ? 0xffffffffu : a / b; break;
            case 6: k = REM;
              v = b == 0 ? a : (a == 0x80000000u && b == 0xffffffffu) ? 0 : (uint32_t)((int32_t)a % (int32_t)b);
              break;
            case 7: k = REMU; v = b == 0 ? a : a % b; break;
          }
        } else if (f7 == 0x10) {
          if (f3 == 2) { k = SH1ADD; v = (a << 1) + b; }
          else if (f3 == 4) { k = SH2ADD; v = (a << 2) + b; }
          else if (f3 == 6) { k = SH3ADD; v = (a << 3) + b; }
          else return 2;
        } else return 2;
        break;
      case 0x0f: k = FENCE; wr = 0; break;
      case 0x73:
        if (in == 0x00000073u) {
          ops[ECALL]++; icount++; tokens += 3 + tref; tn++; /* the call, and a0's value from the OS */
          if (!shadow && PC >= text_lo && PC < text_hi) tokpc[(PC - text_lo) >> 2] += 3 + tref;
          return 1;
        }
        if (in == 0x00100073u) { ops[EBREAK]++; icount++; return 3; }
        k = CSR; v = 0; break; /* reads as zero, writes ignored */
      case 0x2f: {
        if (f3 != 2) return 2;
        uint32_t f5 = in >> 27;
        if (!ok(a, 4) || (a & 3)) { fault_addr = a; return 5; }
        if (f5 == 2) { k = LRW; pload(a, 4); memcpy(&v, M + a, 4); resv = a; mt_load(a, 4); }
        else if (f5 == 3) {
          k = SCW;
          if (resv == a) { log_store(a, 4); pstore(a, 4); if (nfz) frozen_store_check(a, 4); memcpy(M + a, &b, 4); mt_store(a, 4); v = 0; } else v = 1;
          resv = 0xffffffffu;
        } else return 2;
        break;
      }
      default: return 2;
    }
    if (wr && rd) { X[rd] = v; pois &= ~(1u << rd); }
    ops[k]++;
    { /* the record's tokens */
      uint32_t tk;
      if (k == JAL || k == JALR || jt) { tk = 1; tn = 0; }
      else {
        if (op == 0x63 || k == SW || k == FENCE) tk = 1;
        else if (k == SB || k == SH) tk = 4;
        else if (k == SCW) tk = (rd ? vlen(v) : 1) + 1;
        else if (k >= DIV && k <= REMU) tk = rd ? 1 + (8 - div_k0(k == DIV || k == REM, a, b)) + vlen(v) : 1;
        else tk = rd ? vlen(v) : 1;
        tn++;
      }
      tokens += tk + tref;
      if (!shadow && PC >= text_lo && PC < text_hi) tokpc[(PC - text_lo) >> 2] += tk + tref;
    }
    if (shadow) { icount++; PC = next; return 0; }
    if (prof && (k == JAL || k == JALR)) {
      if (rd == 1 && next >= text_lo && next < text_hi && csp < CSTK) {
        uint32_t e = (next - text_lo) >> 2;
        if (next == watch_pc && PC >= text_lo && PC < text_hi) {
          wcalls[(PC - text_lo) >> 2]++;
          wbytes[(PC - text_lo) >> 2] += X[12];
        }
        cs_entry[csp] = e;
        cs_start[csp++] = icount;
        incl_depth[e]++;
      } else if (k == JALR && rd == 0 && rs1 == 1 && csp > 0) {
        uint32_t e = cs_entry[--csp];
        if (incl_depth[e] == 1) incl[e] += icount + 1 - cs_start[csp];
        incl_depth[e]--;
      }
    }
    if (prof) {
      pair[k1 * NOPS + k]++;
      if (prev_rd && op != 0x37 && op != 0x17 && op != 0x6f && (prev_rd == rs1 || (prev_rd == rs2 && (op == 0x33 || op == 0x23 || op == 0x63)))) pair_dep[k1 * NOPS + k]++;
      trip[(k2 * NOPS + k1) * NOPS + k]++;
      k2 = k1;
      k1 = k;
      prev_rd = (wr && op != 0x63 && op != 0x23) ? rd : 0;
    }
    if (PC >= text_lo && PC < text_hi) pcc[(PC - text_lo) >> 2]++;
    icount++;
    PC = next;
  }
  return 0;
}

#include "circ_units.h"

EXPORT int rv_run(uint64_t limit) {
  uint64_t end = icount + limit;
  while (icount < end) {
    if (umode && PC >= text_lo && PC < text_hi && tmap[(PC - text_lo) >> 2]) {
      int r = unit_dispatch(tmap[(PC - text_lo) >> 2] - 1);
      if (r < 0) return 6;
      if (r > 0) continue;
    }
    int rc = step();
    if (rc) return rc;
  }
  return 4;
}
