/* Exact dedicated units ("circuits") for fx's hot code: macro-uops that fire at fixed ROM PCs
 * of one fx build (a CPU specialized to its ROM) and have exactly the architectural effect of
 * the original instructions there, then resume at the exit PC. Included by rv32emu.c.
 *
 * A unit's effect covers memory bytes, every register live at its exit, and the PC. Registers
 * dead at the exit (static liveness on the binary, from fx/circ/units.py, or the calling
 * convention after a return) are left as they were; bytes of the native stack below sp at a
 * function return are dead too. Fallback paths (allocation, a full buffer, errors, inputs a
 * unit does not cover) decline or exit to the original code, which then runs as compiled.
 *
 * Modes (rv_units_mode): 0 off, 1 on, 2 verify. Verify runs the unit, records its state, undoes
 * it, steps the original code from the same state to the same exit, and compares PC, every
 * register not dead at that exit and every byte either side wrote (dead stack bytes excepted);
 * then it continues from the unit's state and poisons the registers / dead stack bytes that
 * differ, so a later read of one is counted as a violation (rv_violations).
 *
 * Tokens: each unit charges the tokens its records emit under circuits.md's record formats
 * (16-bit payloads, records of <= 64 data tokens, re-dispatch for long runs); the original
 * code's tokens come from step()'s per-instruction model.
 */

enum {
  U_NONE, U_MEMMOVE, U_MEMCPY, U_MEMSET, U_EQL, U_TOKEST, U_JSCAN, U_JESC, U_ASCII, U_MASKP,
  U_MEMCHR, U_BCOPY, U_SHA, U_WYHASH, U_MOVE_IR, U_EQL_IR, U_TOK2, U_CHR2, U_ASC2, U_SHA2, U_NKINDS
};
#define MAXT 64
#define DMAX 64 /* data tokens per record (records stay <= 80 tokens with their commits) */
typedef struct {
  uint32_t pc;
  int kind, on;
  uint32_t p[16];    /* unit parameters (exit PCs, constants), set by units.py from the binary */
  uint32_t dead[4];  /* per exit index: registers dead there */
  uint64_t fires, declines, utok, rinstr, rtok, bytes, fails;
} trig_t;
typedef struct {
  uint32_t exit_pc;
  int xi;           /* exit index (dead[xi]) */
  int creg;         /* verify: stop at exit_pc when X[creg] == cval (-1: the first time) */
  uint32_t cval;
  int fnret;        /* a function return: stack bytes below sp are dead */
  uint32_t wsp;     /* a wasm2c function's return: shadow-stack bytes below this are dead (0: none) */
  uint64_t data;    /* data tokens of the unit's records (split into records of DMAX) */
  uint64_t extra;   /* other tokens (stores of committed values) */
  uint32_t free_commit; /* live registers whose commit rides the exit J (no value tokens) */
  uint64_t tok, bytes; /* all tokens (set by unit_exec), bytes processed */
} ures_t;

static trig_t T[MAXT];
static int nT;
static uint8_t* tmap; /* per text word: trigger index + 1 */
static int umode;
static uint32_t ufail[8]; /* the first failed check: trigger, what, detail, unit value, orig value, steps */
static uint8_t* wmark;    /* bitmap over memory: bytes the unit wrote (verify) */
static uint8_t* omark;    /* bytes the original wrote (verify) */

EXPORT void rv_units_mode(int m) { umode = m; }
EXPORT int rv_unit_add(uint32_t pc, int kind, const uint32_t* p, int np, const uint32_t* dead, int nd) {
  if (nT >= MAXT || pc < text_lo || pc >= text_hi || (pc & 3)) return -1;
  if (!tmap) tmap = calloc((text_hi - text_lo) / 4 + 1, 1);
  if (!wmark) wmark = calloc(MEMSZ / 8, 1);
  if (!omark) omark = calloc(MEMSZ / 8, 1);
  trig_t* t = &T[nT];
  memset(t, 0, sizeof *t);
  t->pc = pc;
  t->kind = kind;
  t->on = 1;
  for (int i = 0; i < np && i < 16; i++) t->p[i] = p[i];
  for (int i = 0; i < nd && i < 4; i++) t->dead[i] = dead[i];
  tmap[(pc - text_lo) >> 2] = (uint8_t)(nT + 1);
  return nT++;
}
EXPORT void rv_unit_enable(int i, int on) { if (i >= 0 && i < nT) T[i].on = on; }
EXPORT void rv_unit_stats(int i, uint64_t* out) {
  trig_t* t = &T[i];
  out[0] = t->fires; out[1] = t->declines; out[2] = t->utok; out[3] = t->rinstr; out[4] = t->rtok;
  out[5] = t->bytes; out[6] = t->fails;
}
EXPORT uint32_t* rv_unit_fail(void) { return ufail; }

/* ---- memory access for units (bounds-checked by the callers; stores logged) ---- */
static inline int inmem(uint32_t a, uint32_t n) { return a < MEMSZ && n <= MEMSZ - a; }
static inline uint8_t rd8(uint32_t a) { return M[a]; }
static inline uint32_t rd32(uint32_t a) { uint32_t v; memcpy(&v, M + a, 4); return v; }
static void frozen_store_check(uint32_t a, uint32_t n);
static inline void wr8(uint32_t a, uint8_t v) { if (log_on) log_byte(a); if (nfz) frozen_store_check(a, 1); M[a] = v; }
static inline void wr32(uint32_t a, uint32_t v) { if (log_on) { for (int i = 0; i < 4; i++) log_byte(a + i); } if (nfz) frozen_store_check(a, 4); memcpy(M + a, &v, 4); }

/* ---- token helpers ---- */
static inline uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
static inline uint32_t words_spanned(uint32_t a, uint32_t n) { return n ? ((a + n - 1) >> 2) - (a >> 2) + 1 : 0; }
/* data tokens split into records of DMAX; each extra record costs `cont` (its re-dispatch) */
static inline uint64_t records(uint64_t data, uint64_t cont) { return data + (data > DMAX ? (ceil_div(data, DMAX) - 1) * cont : 0); }

/* ---- function-level units (entry PC; exit = return to ra) ---- */
static inline void fn_exit(ures_t* r) {
  r->exit_pc = X[1];
  r->xi = 0;
  r->creg = 2;
  r->cval = X[2];
  r->fnret = 1;
}

/* the bytes of a copy as memmove leaves them (rt.c's copy_fwd / copy_bwd give the same result
 * whenever memmove picks them, see circuits.md) */
static void move_bytes(uint32_t d, uint32_t s, uint32_t n) {
  if (d < s || d >= s + n) for (uint32_t i = 0; i < n; i++) wr8(d + i, rd8(s + i));
  else for (uint32_t i = n; i-- > 0;) wr8(d + i, rd8(s + i));
}
/* data tokens of a streaming copy: 2 per aligned source word (16-bit halves), 2 per partial
 * destination word (its merged value through the stream) */
static uint64_t move_data(uint32_t d, uint32_t s, uint32_t n) {
  if (n == 0 || d == s) return 0;
  uint32_t S = words_spanned(s, n), D = words_spanned(d, n), P;
  if (D == 1) P = (n == 4 && !(d & 3)) ? 0 : 1;
  else P = ((d & 3) != 0) + (((d + n) & 3) != 0);
  return 2ull * S + 2ull * P;
}

static int u_memmove(trig_t* t, ures_t* r, int is_memcpy) {
  (void)t;
  uint32_t d = X[10], s = X[11], n = X[12];
  if (n && d != s) {
    if (!inmem(d, n) || !inmem(s, n) || !inmem(s & ~3u, n + 8)) return 0;
    if (is_memcpy && d > s && d < s + n) return 0; /* overlapping memcpy: copy_fwd's own order */
    /* the copy may not touch the stack frames the original would create */
    if ((d < X[2] && d + n > X[2] - 64) || (s < X[2] && s + n + 4 > X[2] - 64)) return 0;
    move_bytes(d, s, n);
  }
  r->data = move_data(d, s, n);
  r->bytes = n;
  fn_exit(r);
  return 1;
}

/* memset: the fill word comes from the register (one SW-like token per full word), a partial
 * word's merged value through the stream (2 tokens) */
static int u_memset(trig_t* t, ures_t* r) {
  (void)t;
  uint32_t d = X[10], c = X[11] & 0xff, n = X[12];
  if (n) {
    if (!inmem(d, n)) return 0;
    if (d < X[2] && d + n > X[2] - 64) return 0;
    for (uint32_t i = 0; i < n; i++) wr8(d + i, (uint8_t)c);
  }
  if (n) {
    uint32_t D = words_spanned(d, n), P;
    if (D == 1) P = (n == 4 && !(d & 3)) ? 0 : 1;
    else P = ((d & 3) != 0) + (((d + n) & 3) != 0);
    r->data = (uint64_t)(D - P) + 2ull * P;
  }
  r->bytes = n;
  fn_exit(r);
  return 1;
}

/* mem.eqlBytes(a.ptr a1, a.len a2, b.ptr a3, b.len a4) -> a0. Operand a rides the stream in
 * 16-bit halves of its aligned words; b is read through the RAM port at each position. */
static int u_eql(trig_t* t, ures_t* r) {
  (void)t;
  uint32_t ap = X[11], n = X[12], bp = X[13], bn = X[14];
  uint32_t res;
  if (n != bn) res = 0;
  else if (n == 0 || ap == bp) res = 1;
  else {
    if (!inmem(ap, n) || !inmem(bp, n)) return 0;
    uint32_t m = 0;
    while (m < n && rd8(ap + m) == rd8(bp + m)) m++;
    res = m == n;
    uint32_t upto = m == n ? n : m + 1;               /* bytes compared */
    uint32_t wa = words_spanned(ap, upto) + 1;        /* + the word the funnel needs */
    uint32_t wt = words_spanned(ap, n);
    if (wa > wt) wa = wt;
    r->data = 2ull * wa;
    r->bytes = upto;
  }
  X[10] = res;
  fn_exit(r);
  return 1;
}

/* crypto.sha2.Sha2x32(..., 256).round(instance a0, state a1, block a2): one SHA-256
 * compression of the 64-byte block into the 8-word state (fully unrolled in fx's build). The
 * original leaves W0..63 in its wasm shadow-stack frame at __stack_pointer - 0x100 (fx reads
 * such stale frame bytes later, so the unit writes them too: they ride the W tokens). Record
 * (estimate of the streaming unit): W0..15 through the stream (2 tokens each), W16..63 (2
 * each), the new a and e of every round (4), the state's 8 words written back (2 each). */
static const uint32_t K256[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static inline uint32_t rr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static int u_sha(trig_t* t, ures_t* r) {
  (void)t;
  uint32_t inst = X[10], st = X[11], blk = X[12];
  if (!inmem(inst, 12) || (st & 3) || !inmem(st, 32) || !inmem(blk, 64)) return 0;
  uint32_t wsp = rd32(inst + 8);
  if (wsp < 0x100 || wsp > WSTK_HI) return 0;
  if (st < blk + 64 && blk < st + 32) return 0;
  if ((st < wsp && st + 32 > wsp - 0x100) || (blk < wsp && blk + 64 > wsp - 0x100)) return 0;
  uint32_t W[64], h[8];
  for (int i = 0; i < 16; i++)
    W[i] = (uint32_t)rd8(blk + 4 * i) << 24 | (uint32_t)rd8(blk + 4 * i + 1) << 16 | (uint32_t)rd8(blk + 4 * i + 2) << 8 | rd8(blk + 4 * i + 3);
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rr(W[i - 15], 7) ^ rr(W[i - 15], 18) ^ (W[i - 15] >> 3);
    uint32_t s1 = rr(W[i - 2], 17) ^ rr(W[i - 2], 19) ^ (W[i - 2] >> 10);
    W[i] = W[i - 16] + s0 + W[i - 7] + s1;
  }
  for (int i = 0; i < 8; i++) h[i] = rd32(st + 4 * i);
  for (int i = 0; i < 64; i++) wr32(wsp - 0x100 + 4 * i, W[i]);
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = hh + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + W[i];
    uint32_t t2 = (rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  for (int i = 0; i < 8; i++) wr32(st + 4 * i, h[i]);
  r->data = 32 + 96 + 64 * 4 + 16;
  r->bytes = 64;
  fn_exit(r);
  r->wsp = wsp;
  return 1;
}

/* hash.wyhash.Wyhash.hash(0, input) (std/hash/wyhash.zig, seed 0 folded in fx's build):
 * (instance a0, ptr a1, len a2) -> u64 in a0 (low) / a1 (high). The original leaves 64-bit
 * products in its 0x60-byte shadow-stack frame (wsp = __stack_pointer): the final mix's halves
 * at wsp - 0x60, final2's mum at wsp - 0x50, the last final1-loop mix at wsp - 0x40, the last
 * round's three mixes at wsp - 0x30 (k = 2) .. wsp - 0x10 (k = 0); the unit writes the same
 * bytes (each is a product already in the stream). Record (estimate): the input's aligned words
 * through the stream (2 tokens each) and 20 tokens per 64 x 64 -> 128-bit multiply (four
 * 32 x 32 products through the core's multiplier columns, the 128-bit sum in 16-bit tokens). */
static const uint64_t WYS[4] = {0xa0761d6478bd642full, 0xe7037ed1a0b428dbull, 0x8ebc6af09c88c6e3ull, 0x589965cc75374cc3ull};
static inline uint64_t rdle(uint32_t a, int n) { uint64_t v = 0; for (int i = n - 1; i >= 0; i--) v = v << 8 | rd8(a + (uint32_t)i); return v; }
static inline void wmum(uint64_t* a, uint64_t* b, uint32_t* nm) {
  unsigned __int128 x = (unsigned __int128)*a * *b;
  *a = (uint64_t)x; *b = (uint64_t)(x >> 64); (*nm)++;
}
static inline uint64_t wmix(uint64_t a, uint64_t b, uint32_t* nm) { wmum(&a, &b, nm); return a ^ b; }
static uint64_t wlast[2]; /* the halves of the latest mix */
static inline uint64_t wmixl(uint64_t a, uint64_t b, uint32_t* nm) { wmum(&a, &b, nm); wlast[0] = a; wlast[1] = b; return a ^ b; }
static inline void wr64(uint32_t a, uint64_t v) { wr32(a, (uint32_t)v); wr32(a + 4, (uint32_t)(v >> 32)); }
static int u_wyhash(trig_t* t, ures_t* r) {
  (void)t;
  uint32_t inst = X[10], p = X[11], n = X[12], nm = 0, dummy = 0;
  if (!inmem(inst, 12) || !inmem(p, n)) return 0;
  uint32_t wsp = rd32(inst + 8);
  if (wsp < 0x60 || wsp > WSTK_HI) return 0;
  if (p < wsp && p + n > wsp - 0x60) return 0; /* the input inside the frame */
  uint64_t s0 = 0 ^ wmix(0 ^ WYS[0], WYS[1], &dummy); /* a constant (seed 0) */
  uint64_t st[3] = {s0, s0, s0}, a, b, rnd[3][2], f1[2];
  int did_round = 0, did_f1 = 0;
  if (n <= 16) {
    if (n >= 4) {
      uint32_t end = n - 4, q = (n >> 3) << 2;
      a = rdle(p, 4) << 32 | rdle(p + q, 4);
      b = rdle(p + end, 4) << 32 | rdle(p + end - q, 4);
    } else if (n > 0) {
      a = (uint64_t)rd8(p) << 16 | (uint64_t)rd8(p + (n >> 1)) << 8 | rd8(p + n - 1);
      b = 0;
    } else a = b = 0;
  } else {
    uint32_t i = 0;
    if (n >= 48) {
      while (i + 48 < n) {
        for (int k = 0; k < 3; k++) {
          st[k] = wmixl(rdle(p + i + 16 * k, 8) ^ WYS[k + 1], rdle(p + i + 16 * k + 8, 8) ^ st[k], &nm);
          rnd[k][0] = wlast[0]; rnd[k][1] = wlast[1];
        }
        did_round = 1;
        i += 48;
      }
      st[0] ^= st[1] ^ st[2];
    }
    uint32_t m = n - i, j = 0;
    while (j + 16 < m) {
      st[0] = wmixl(rdle(p + i + j, 8) ^ WYS[1], rdle(p + i + j + 8, 8) ^ st[0], &nm);
      f1[0] = wlast[0]; f1[1] = wlast[1]; did_f1 = 1;
      j += 16;
    }
    a = rdle(p + n - 16, 8);
    b = rdle(p + n - 8, 8);
  }
  a ^= WYS[1];
  b ^= st[0];
  wmum(&a, &b, &nm);
  uint64_t h = wmixl(a ^ WYS[0] ^ (uint64_t)n, b ^ WYS[1], &nm);
  wr64(wsp - 0x60, wlast[0]); wr64(wsp - 0x58, wlast[1]);
  wr64(wsp - 0x50, a); wr64(wsp - 0x48, b);
  if (did_f1) { wr64(wsp - 0x40, f1[0]); wr64(wsp - 0x38, f1[1]); }
  if (did_round) for (int k = 0; k < 3; k++) { wr64(wsp - 0x10 - 0x10 * k, rnd[k][0]); wr64(wsp - 0x08 - 0x10 * k, rnd[k][1]); }
  X[10] = (uint32_t)h;
  X[11] = (uint32_t)(h >> 32);
  r->data = 2ull * words_spanned(p, n) + 20ull * nm;
  r->bytes = n;
  fn_exit(r);
  r->wsp = wsp;
  return 1;
}

/* ---- the units built as circuits in chipformer/rv/core.py (their exact record formats) ----
 * Both sit on a `beq` whose taken target is the function's `ret`: the ROM word keeps the beq's
 * decoding plus a macro flag, the circuit fires when its condition holds and otherwise runs the
 * beq as compiled; a fired record ends with J to the beq's target, where the original `ret`
 * runs. p[0] = that target. */

/* memmove's entry `beq a0, a1, ret`: d, s, n 4-aligned, 4 <= n <= 84, d != s, d < s or
 * d - s >= n (a forward word copy is then memmove's result). Record: MP, then per word
 * CW(lo16) CW(hi16) CWW (the CWW commits the destination word from the two payloads), J. */
static int u_move_ir(trig_t* t, ures_t* r) {
  uint32_t d = X[10], s = X[11], n = X[12];
  if (((d | s | n) & 3) || n < 4 || n > 84 || d == s) return 0;
  if (!(d < s || d - s >= n)) return 0;
  if (!inmem(d, n) || !inmem(s, n)) return 0;
  if ((d < X[2] && d + n > X[2] - 64) || (s < X[2] && s + n > X[2] - 64)) return 0;
  move_bytes(d, s, n);
  r->data = 1 + 3ull * (n / 4);
  r->bytes = n;
  r->exit_pc = t->p[0]; r->xi = 0;
  r->creg = 2; r->cval = X[2]; r->fnret = 1;
  return 1;
}

/* mem.eqlBytes' `beq a1, a3, ret` (after its length and zero checks; a0 = 1 there): a1 != a3,
 * 1 <= n (a2) <= 31. Record: MP, then per byte EB(a's byte) EC(equal?) up to the first
 * mismatch, then J (to the beq's target, a `ret`), which also commits a0 = the result; then the
 * `ret` (one J). */
static int u_eql_ir(trig_t* t, ures_t* r) {
  uint32_t ap = X[11], n = X[12], bp = X[13];
  if (ap == bp || n < 1 || n > 31 || !inmem(ap, n) || !inmem(bp, n)) return 0;
  uint32_t m = 0;
  while (m < n && rd8(ap + m) == rd8(bp + m)) m++;
  X[10] = m == n;
  r->data = 1 + 2ull * (m == n ? n : m + 1);
  r->free_commit = 1u << 10;
  r->bytes = m == n ? n : m + 1;
  /* the record's J goes to the beq's target, a `ret`; past this beq the original builds a frame
   * and returns through other `ret`s, so the state is compared after the return: the unit
   * covers that `ret` too (its J token) */
  r->extra = 1;
  fn_exit(r);
  (void)t;
  return 1;
}

#include "circ_loops.h"

static int unit_run(trig_t* t, ures_t* r) {
  switch (t->kind) {
    case U_MEMMOVE: return u_memmove(t, r, 0);
    case U_MEMCPY: return u_memmove(t, r, 1);
    case U_MEMSET: return u_memset(t, r);
    case U_EQL: return u_eql(t, r);
    case U_SHA: return u_sha(t, r);
    case U_SHA2: return (X[12] & 3) ? 0 : u_sha(t, r); /* as built: the message words aligned */
    case U_WYHASH: return u_wyhash(t, r);
    case U_MOVE_IR: return u_move_ir(t, r);
    case U_EQL_IR: return u_eql_ir(t, r);
    default: return loop_unit_run(t, r);
  }
}

/* A unit's run: registers dead at its exit keep their old values (the circuit never writes
 * them); every live register it changes is committed (a value record); the last token is a J
 * to the exit PC. Records hold DMAX data tokens; each re-dispatch commits the changed live
 * registers again and costs a J. */
static int unit_exec(trig_t* t, ures_t* r) {
  uint32_t X0[32];
  memcpy(X0, X, sizeof X0);
  if (!unit_run(t, r)) return 0;
  uint32_t dead = t->dead[r->xi];
  uint64_t C = 0;
  for (int i = 1; i < 32; i++) {
    if (dead >> i & 1) X[i] = X0[i];
    else if (X[i] != X0[i] && !(r->free_commit >> i & 1)) C += vlen(X[i]);
  }
  r->tok = records(r->data, C + 1) + r->extra + C + 1;
  return 1;
}

/* ---- verify: unit vs the original code from the same state ---- */
static uint64_t ops_save[NOPS];

static int fail(int ti, uint32_t what, uint32_t detail, uint32_t uv, uint32_t ov, uint32_t steps) {
  if (!ufail[1]) {
    ufail[0] = (uint32_t)ti; ufail[1] = what; ufail[2] = detail; ufail[3] = uv; ufail[4] = ov;
    ufail[5] = steps; ufail[6] = T[ti].pc;
  }
  T[ti].fails++;
  return -1;
}

static int unit_verify(int ti, uint64_t* steps_out) {
  trig_t* t = &T[ti];
  uint32_t X0[32], XU[32], pc0 = PC;
  memcpy(X0, X, sizeof X0);
  uint64_t tok0 = tokens, ic0 = icount;
  uint32_t tn0 = tn;
  ures_t r = {0};
  r.creg = -1;
  lgn = 0;
  log_on = 1;
  if (!unit_exec(t, &r)) {
    log_on = 0;
    if (lgn || memcmp(X0, X, sizeof X0)) return fail(ti, 7, 0, 0, 0, 0); /* a decline changed state */
    return 0;
  }
  memcpy(XU, X, sizeof XU);
  /* the unit's writes (address, final byte), then undo them */
  size_t nu = lgn;
  logent_t* uw = malloc((nu ? nu : 1) * sizeof(logent_t));
  for (size_t i = 0; i < nu; i++) { uw[i].a = lg[i].a; uw[i].old = M[lg[i].a]; wmark[lg[i].a >> 3] |= (uint8_t)(1u << (lg[i].a & 7)); }
  for (size_t i = nu; i-- > 0;) M[lg[i].a] = lg[i].old;
  memcpy(X, X0, sizeof X0);
  PC = pc0;
  /* step the original */
  lgn = 0;
  shadow = 1;
  memcpy(ops_save, ops, sizeof ops);
  uint64_t steps = 0;
  int rc = 0;
  for (;;) {
    if (PC == r.exit_pc && (r.creg < 0 || X[r.creg] == r.cval) && steps) break;
    if (steps > 50000000ull) { rc = 4; break; }
    rc = step();
    if (rc) break;
    steps++;
  }
  shadow = 0;
  log_on = 0;
  memcpy(ops, ops_save, sizeof ops);
  uint64_t otok = tokens - tok0;
  int bad = 0;
  if (rc) bad = fail(ti, 5, (uint32_t)rc, PC, 0, (uint32_t)steps);
  else if (PC != r.exit_pc) bad = fail(ti, 1, 0, r.exit_pc, PC, (uint32_t)steps);
  uint32_t dead = t->dead[r.xi];
  for (int i = 1; i < 32 && !bad; i++)
    if (!(dead >> i & 1) && XU[i] != X[i]) bad = fail(ti, 2, (uint32_t)i, XU[i], X[i], (uint32_t)steps);
  uint32_t sp_exit = XU[2];
#define DEADB(a) (r.fnret && (((a) >= PSTK_LO && (a) < sp_exit) || ((a) < WSTK_HI && (a) < r.wsp)))
#define UWR(a) (wmark[(a) >> 3] >> ((a) & 7) & 1)
#define OWR(a) (omark[(a) >> 3] >> ((a) & 7) & 1)
  /* the original's first write of each address holds the initial byte */
  for (size_t i = 0; i < lgn; i++) {
    uint32_t a = lg[i].a;
    if (OWR(a)) { lg[i].a |= 0x80000000u; continue; } /* a later write: not the initial value */
    omark[a >> 3] |= (uint8_t)(1u << (a & 7));
  }
  /* bytes the unit wrote: its final byte against the original's final byte (M now) */
  for (size_t i = 0; i < nu && !bad; i++) {
    uint32_t a = uw[i].a;
    if (uw[i].old != M[a] && !DEADB(a)) bad = fail(ti, 3, a, uw[i].old, M[a], (uint32_t)steps);
  }
  /* bytes only the original wrote: the initial byte (the unit's) against its final byte */
  for (size_t i = 0; i < lgn && !bad; i++) {
    if (lg[i].a >> 31) continue;
    uint32_t a = lg[i].a;
    if (!UWR(a) && lg[i].old != M[a] && !DEADB(a)) bad = fail(ti, 3, a, lg[i].old, M[a], (uint32_t)steps);
  }
  /* continue from the unit's state; poison what differs (dead registers, dead stack bytes) */
  for (int i = 1; i < 32; i++) if (XU[i] != X[i]) pois |= 1u << i;
  for (size_t i = 0; i < nu; i++) if (DEADB(uw[i].a) && uw[i].old != M[uw[i].a]) ppoison(uw[i].a);
  for (size_t i = 0; i < lgn; i++) {
    if (lg[i].a >> 31) continue;
    uint32_t a = lg[i].a;
    if (DEADB(a) && !UWR(a) && lg[i].old != M[a]) ppoison(a);
  }
  for (size_t i = lgn; i-- > 0;) {
    uint32_t a = lg[i].a & 0x7fffffffu;
    M[a] = lg[i].old;
    omark[a >> 3] = 0;
  }
  for (size_t i = 0; i < nu; i++) { M[uw[i].a] = uw[i].old; wmark[uw[i].a >> 3] = 0; }
#undef DEADB
#undef UWR
#undef OWR
  free(uw);
  memcpy(X, XU, sizeof XU);
  PC = r.exit_pc;
  tokens = tok0 + r.tok;
  tn = 0;
  icount = ic0;
  (void)tn0;
  t->fires++;
  t->utok += r.tok;
  t->rinstr += steps;
  t->rtok += otok;
  t->bytes += r.bytes;
  if (steps_out) { steps_out[0] = steps; steps_out[1] = otok; steps_out[2] = r.tok; }
  return bad ? -1 : 1;
}

static int unit_dispatch(int ti) {
  trig_t* t = &T[ti];
  if (!t->on) return 0;
  if (umode == 2) {
    int v = unit_verify(ti, 0);
    if (v == 0) t->declines++;
    return v;
  }
  ures_t r = {0};
  r.creg = -1;
  if (!unit_exec(t, &r)) { t->declines++; return 0; }
  PC = r.exit_pc;
  tokens += r.tok;
  tn = 0;
  t->fires++;
  t->utok += r.tok;
  t->bytes += r.bytes;
  return 1;
}

/* fuzzing: one checked firing at the current PC (which must be a trigger); out = steps,
 * original tokens, unit tokens. Returns 1 checked equal, 0 declined, -1 mismatch. */
EXPORT int rv_unit_check(uint64_t* out) {
  if (!tmap || PC < text_lo || PC >= text_hi || !tmap[(PC - text_lo) >> 2]) return -2;
  ufail[1] = 0;
  return unit_verify(tmap[(PC - text_lo) >> 2] - 1, out);
}
EXPORT void rv_clear_poison(void) {
  pois = 0;
  if (plist_over) memset(pbits, 0, MEMSZ / 8);
  else for (uint32_t i = 0; i < nplist; i++) pbits[plist[i] >> 3] = 0;
  nplist = 0;
  plist_over = 0;
  npmem = 0;
  viol_reg = viol_mem = 0;
}
