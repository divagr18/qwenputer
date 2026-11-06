/* A small RV32IM + Zba + Zalrsc interpreter, built as a host DLL and driven from
 * Python (rvhost.py). It counts every retired instruction by mnemonic and by PC.
 * rv_run() returns on ecall (1), illegal instruction (2), ebreak (3), the step
 * limit (4) or a memory fault (5); the host handles the ecall and resumes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define MEMSZ 0x04000000u

enum {
  LUI, AUIPC, JAL, JALR, BEQ, BNE, BLT, BGE, BLTU, BGEU, LB, LH, LW, LBU, LHU, SB, SH, SW,
  ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI, ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA,
  OR, AND, FENCE, ECALL, EBREAK, MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU, SH1ADD, SH2ADD,
  SH3ADD, LRW, SCW, CSR, NOPS
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
#define mt_icount_now icount
#include "memtrace.h"

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

EXPORT int rv_init(uint32_t lo, uint32_t hi) {
  M = calloc(1, MEMSZ);
  text_lo = lo;
  text_hi = hi;
  pcc = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  mpc = calloc((hi - lo) / 4 + 1, sizeof(uint32_t));
  return M && pcc && mpc ? 0 : -1;
}

static inline int ok(uint32_t a, uint32_t n) { return a < MEMSZ && a + n <= MEMSZ; }

EXPORT int rv_run(uint64_t limit) {
  uint64_t end = icount + limit;
  while (icount < end) {
    if (!ok(PC, 4) || (PC & 3)) { fault_addr = PC; return 5; }
    uint32_t in;
    memcpy(&in, M + PC, 4);
    uint32_t op = in & 0x7f, rd = (in >> 7) & 31, f3 = (in >> 12) & 7, rs1 = (in >> 15) & 31,
             rs2 = (in >> 20) & 31, f7 = in >> 25;
    uint32_t a = X[rs1], b = X[rs2], next = PC + 4, v = 0;
    int32_t immi = (int32_t)in >> 20;
    int k = -1;
    int wr = 1;
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
        if (t) next = PC + im;
        wr = 0;
        break;
      }
      case 0x03: {
        uint32_t ad = a + immi;
        uint32_t n = (f3 & 3) == 0 ? 1 : (f3 & 3) == 1 ? 2 : 4;
        if (!ok(ad, n)) { fault_addr = ad; return 5; }
        if (ad & (n - 1)) { misaligned++; if (PC >= text_lo && PC < text_hi) mpc[(PC - text_lo) >> 2]++; }
        mt_load(ad, n);
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
        memcpy(M + ad, &b, n);
        mt_store(ad, n);
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
          case 1: if (f7) return 2; k = SLLI; v = a << rs2; break;
          case 5:
            if (f7 == 0) { k = SRLI; v = a >> rs2; }
            else if (f7 == 0x20) { k = SRAI; v = (uint32_t)((int32_t)a >> rs2); }
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
          else return 2;
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
        if (in == 0x00000073u) { ops[ECALL]++; icount++; return 1; }
        if (in == 0x00100073u) { ops[EBREAK]++; icount++; return 3; }
        k = CSR; v = 0; break; /* reads as zero, writes ignored */
      case 0x2f: {
        if (f3 != 2) return 2;
        uint32_t f5 = in >> 27;
        if (!ok(a, 4) || (a & 3)) { fault_addr = a; return 5; }
        if (f5 == 2) { k = LRW; memcpy(&v, M + a, 4); resv = a; mt_load(a, 4); }
        else if (f5 == 3) {
          k = SCW;
          if (resv == a) { memcpy(M + a, &b, 4); mt_store(a, 4); v = 0; } else v = 1;
          resv = 0xffffffffu;
        } else return 2;
        break;
      }
      default: return 2;
    }
    if (wr && rd) X[rd] = v;
    ops[k]++;
    if (PC >= text_lo && PC < text_hi) pcc[(PC - text_lo) >> 2]++;
    icount++;
    PC = next;
  }
  return 4;
}
