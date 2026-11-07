/* Loop units: macro-uops at loop heads inside fx's own compiled functions (circ_units.h).
 * Each one is written against the instruction words of its region (fx/circ/units.py has the
 * signatures; circuits.md the code): register numbers are the compiler's choice in that code,
 * and units.py only installs a unit where those exact words appear. p[] = the exit PCs (in the
 * order below), then the unit's constants. A unit either declines (no state change) or runs
 * to one of its exits; registers dead at that exit are put back by unit_exec. */

#define R_T0 5
#define R_S0 8
#define R_A0 10
#define R_A1 11
#define R_A2 12
#define R_A3 13
#define R_A4 14
#define R_A5 15
#define R_A6 16
#define R_A7 17
#define R_S3 19
#define R_S4 20
#define R_S5 21
#define R_S6 22
#define R_S7 23
#define R_S8 24
#define R_S10 26
#define R_S11 27
#define R_T4 29
#define R_T5 30

/* ---- read-only constants a unit bakes in (store-watched for the run) ---- */
#define MAXFZ 32
static uint32_t fz_lo[MAXFZ], fz_hi[MAXFZ];
static int nfz;
static uint64_t fz_viol;
static void freeze(uint32_t lo, uint32_t hi) {
  for (int i = 0; i < nfz; i++) if (fz_lo[i] == lo && fz_hi[i] == hi) return;
  if (nfz < MAXFZ) { fz_lo[nfz] = lo; fz_hi[nfz] = hi; nfz++; }
}
static void frozen_store_check(uint32_t a, uint32_t n) {
  for (int i = 0; i < nfz; i++) if (a < fz_hi[i] && a + n > fz_lo[i]) fz_viol++;
}
EXPORT uint64_t rv_frozen_violations(void) { return fz_viol; }

static inline uint32_t sext8(uint32_t b) { return (uint32_t)(int32_t)(int8_t)(uint8_t)b; }

/* measureProviderRequest's whitespace-run token estimate (prompt_context.zig, inlined twice):
 * for each byte at s5 (a6 bytes): a byte in 9..13 or ' ' adds ceil(run / 4) to the saturating
 * u64 total a0:a1 (hi:lo) and clears the run a3:a2; any other byte increments the saturating
 * run. Constants a4 = 5, a5 = 0x20 (checked). Exit p[0] after the last byte.
 * Record: 2 tokens per aligned input word (the run and the total after it, 16 bits each); the
 * unit declines unless both stay below 2^16. */
static int u_tokest(trig_t* t, ures_t* r) {
  uint32_t p = X[R_S5], n = X[R_A6];
  if (X[R_A4] != 5 || X[R_A5] != 0x20 || n == 0 || !inmem(p, n)) return 0;
  uint64_t run = (uint64_t)X[R_A3] << 32 | X[R_A2], tot = (uint64_t)X[R_A0] << 32 | X[R_A1];
  uint32_t a7 = X[R_A7], t0 = X[R_T0];
  for (uint32_t i = 0; i < n; i++) {
    uint32_t b = rd8(p + i);
    if (b - 9 < 5 || b == 0x20) {
      uint64_t q = (run >> 2) + ((run & 3) != 0);
      uint64_t s = tot + q;
      uint32_t ov = s < tot;
      a7 = (uint32_t)s;
      t0 = 0u - ov;
      tot = ov ? ~0ull : s;
      run = 0;
    } else {
      run = run + 1;
      a7 = run == 0 ? 0xffffffffu : 0; /* the saturation mask */
      if (run == 0) run = ~0ull;
    }
    if (run >= 65536 || tot >= 65536) return 0;
  }
  X[R_A2] = (uint32_t)run; X[R_A3] = (uint32_t)(run >> 32);
  X[R_A1] = (uint32_t)tot; X[R_A0] = (uint32_t)(tot >> 32);
  X[R_A7] = a7; X[R_T0] = t0;
  X[R_S5] = p + n; X[R_A6] = 0;
  r->exit_pc = t->p[0]; r->xi = 0;
  r->data = 2ull * words_spanned(p, n);
  r->bytes = n;
  return 1;
}

/* json.Scanner.next's string loop: a3 += 1; store a3 at 0x24(a2); stop at a4; the byte at
 * a0 + a3 continues the loop when it is in the plain set p[1..8] (from the binary's jump
 * table, units.py). The unit runs the plain iterations and exits at the head p[0] before the
 * first other one. Record: 1 token per aligned word read (the 16-bit cursor after it), and the
 * cursor's store. */
static int u_jscan(trig_t* t, ures_t* r) {
  uint32_t i0 = X[R_A3], end = X[R_A4], buf = X[R_A0], cur = X[R_A2] + 0x24;
  if ((cur & 3) || !inmem(cur, 4)) return 0;
  uint32_t k = 0, stop_byte = 0;
  for (;;) {
    uint32_t idx = i0 + k + 1;
    if (idx == end) break;
    if (!inmem(buf + idx, 1)) return 0;
    uint32_t b = rd8(buf + idx);
    if (!(t->p[1 + (b >> 5)] >> (b & 31) & 1)) { stop_byte = 1; break; }
    k++;
    if (k == 65535) break;
  }
  if (k == 0) return 0;
  uint32_t lo = buf + i0 + 1, hi = buf + i0 + k + stop_byte; /* bytes read: [lo, hi] */
  if (cur <= hi && cur + 4 > lo) return 0; /* the cursor would land in the scanned bytes */
  X[R_A3] = i0 + k;
  wr32(cur, i0 + k);
  r->exit_pc = t->p[0]; r->xi = 0;
  r->creg = R_A3; r->cval = i0 + k; /* the head again, after k iterations */
  r->data = words_spanned(lo, hi - lo + 1);
  r->extra = 1; /* the cursor's SW */
  r->bytes = k;
  return 1;
}

/* json.Stringify.encodeJsonString's run of bytes that need no escape: s8 += 1; stop at the
 * length at -0x3c(s0); the byte b at s5 + s8 leaves the loop when b == s3; when b != s7 it
 * continues if s7 < b (path A), or b & 0x7e == 0x20 (path B: a2 = a3 = 0x20), or
 * (b + 0xdd) & 0xff <= 0x38 (path C: a2 = that, a3 = 0x38); else it leaves. fx keeps '"' and
 * '\' in s3 / s7 in either order (the unit requires both below 256: the circuit compares the
 * byte with two register bytes). s11 = s5 + s8 and a1 = b on every path. Exit at the head. */
static int u_jesc(trig_t* t, ures_t* r) {
  uint32_t fr = X[R_S0] - 0x3c, q3 = X[R_S3], q7 = X[R_S7];
  if (q3 > 255 || q7 > 255 || (fr & 3) || !inmem(fr, 4)) return 0;
  uint32_t len = rd32(fr), i0 = X[R_S8], base = X[R_S5];
  uint32_t k = 0, stop_byte = 0, b, a1 = X[R_A1], a2 = X[R_A2], a3 = X[R_A3];
  for (;;) {
    uint32_t idx = i0 + k + 1;
    if (idx == len) break;
    if (!inmem(base + idx, 1)) return 0;
    b = rd8(base + idx);
    uint32_t n2 = a2, n3 = a3;
    int plain = 0;
    if (b != q3 && b != q7) {
      if (q7 < b) plain = 1;
      else {
        n2 = b & 0x7e; n3 = 0x20;
        if (n2 == 0x20) plain = 1;
        else { n2 = (b + 0xdd) & 0xff; n3 = 0x38; plain = n2 <= 0x38; }
      }
    }
    if (!plain) { stop_byte = 1; break; }
    a1 = b; a2 = n2; a3 = n3;
    k++;
    if (k == 65535) break;
  }
  if (k == 0) return 0;
  X[R_S8] = i0 + k;
  X[R_S11] = i0 + k + base;
  X[R_A1] = a1; X[R_A2] = a2; X[R_A3] = a3;
  r->exit_pc = t->p[0]; r->xi = 0;
  r->creg = R_S8; r->cval = i0 + k;
  r->data = words_spanned(base + i0 + 1, k + stop_byte);
  r->bytes = k;
  return 1;
}

/* json.Stringify.write's ASCII run: t5 = 1; a0 += 1; a0 >= s3 exits to p[0]; t4 = s4 + a0;
 * t5 = the byte, sign-extended; a non-negative one loops, a negative one exits to p[1]. */
static int u_ascii(trig_t* t, ures_t* r) {
  uint32_t i0 = X[R_A0], i = i0, bound = X[R_S3], base = X[R_S4];
  uint32_t t4 = X[R_T4], t5;
  uint32_t k = 0, xi;
  for (;;) {
    t5 = 1;
    i = i + 1;
    if (i >= bound) { xi = 0; break; }
    t4 = i + base;
    if (!inmem(t4, 1)) return 0;
    t5 = sext8(rd8(t4));
    k++;
    if ((int32_t)t5 < 0) { xi = 1; break; }
    if (k > 65535) return 0;
  }
  X[R_A0] = i; X[R_T4] = t4; X[R_T5] = t5;
  r->exit_pc = t->p[xi]; r->xi = (int)xi;
  r->data = words_spanned(base + i0 + 1, k);
  r->bytes = k;
  return 1;
}

/* text_utils.maskSecrets: findSecretSpan's loop over the 12 secret_prefixes at one position
 * (text_utils.zig:844): skip a prefix when the byte before the position is an assignment-key
 * character (s6 == 0: there is one), or when the prefix would not end inside the text; else
 * compare it with eqlIgnoreCase. Exits p[0] (none matched; s11 = 12) or p[1] (prefix s11
 * matched). The table (s5 == p[2]) and the prefixes are fixed data: the circuit compares the
 * text against them as constants, so their bytes are store-watched. Record: one token per
 * aligned text word the compares need (a 12-bit still-matching mask after it). */
static int u_maskp(trig_t* t, ures_t* r) {
  uint32_t tab = X[R_S5];
  if (tab != t->p[2] || X[R_S4] != 0x5f || X[R_A7] != 10 || !inmem(tab, 96)) return 0;
  uint32_t s6 = X[R_S6], s8 = X[R_S8], s3 = X[R_S3], s10 = X[R_S10];
  uint32_t a0 = X[R_A0], a1 = X[R_A1], a2 = X[R_A2], a3 = X[R_A3], a4 = X[R_A4], a5 = X[R_A5], a6 = X[R_A6];
  uint32_t k, lo = s10, hi = s10 + 1; /* text bytes read: [lo, hi) */
  int matched = 0;
  for (int i = 0; i < 12; i++) {
    uint32_t pp = rd32(tab + 8 * i), pl = rd32(tab + 8 * i + 4);
    if (pl == 0 || pl > 64 || !inmem(pp, pl)) return 0;
    freeze(pp, pp + pl);
  }
  freeze(tab, tab + 96);
  for (k = 0; k < 12; k++) {
    a0 = rd32(tab + 8 * k);
    a1 = rd32(tab + 8 * k + 4);
    if (s6 == 0) {
      if (!inmem(s10 - 1, 1)) return 0;
      a2 = rd8(s10 - 1);
      if (s10 - 1 < lo) lo = s10 - 1;
      if (a2 == 0x5f) continue;
      a3 = (a2 + 0xd0) & 0xfe;
      if (a3 < 10) continue;
      a2 = ((a2 & 0xdf) + 0xbf) & 0xfe;
      a3 = 0x1a;
      if (a2 < 0x1a) continue;
    }
    a4 = a1 + s8;
    if (a4 >= s3) continue;
    a2 = s10;
    int ne = 0;
    do {
      if (!inmem(a2, 1) || !inmem(a0, 1)) return 0;
      a3 = rd8(a2);
      a5 = rd8(a0);
      if (a2 + 1 > hi) hi = a2 + 1;
      a6 = (uint32_t)(((a3 + 0xbf) & 0xfe) < 0x1a) << 5;
      a3 |= a6;
      a6 = (uint32_t)(((a5 + 0xbf) & 0xfe) < 0x1a) << 5;
      a5 |= a6;
      if (a3 != a5) { ne = 1; break; }
      a2++; a1--; a0++;
    } while (a1 != 0);
    if (!ne) { matched = 1; break; }
  }
  if (!matched) { k = 12; a0 = 12; }
  X[R_S11] = k;
  X[R_A0] = a0; X[R_A1] = a1; X[R_A2] = a2; X[R_A3] = a3; X[R_A4] = a4; X[R_A5] = a5; X[R_A6] = a6;
  r->xi = matched;
  r->exit_pc = t->p[matched];
  r->data = words_spanned(lo, hi - lo);
  r->bytes = hi - lo;
  return 1;
}

/* acp.jsonrpc.writeOneLine's search for a byte (a0) in a1[0..a2): do { a3 = *a1; a3 == a0
 * exits p[0]; a2 -= 1; a1 += 1 } while (a2); then p[1]. Record: 1 token per aligned word. */
static int u_memchr(trig_t* t, ures_t* r) {
  uint32_t p = X[R_A1], n = X[R_A2], c = X[R_A0];
  if (n == 0 || n > 65535 || !inmem(p, n)) return 0;
  uint32_t i, a3 = X[R_A3], xi = 1;
  for (i = 0; i < n; i++) {
    a3 = rd8(p + i);
    if (a3 == c) { xi = 0; break; }
  }
  uint32_t seen = xi == 0 ? i + 1 : n;
  X[R_A3] = a3;
  X[R_A1] = p + (xi == 0 ? i : n);
  X[R_A2] = n - (xi == 0 ? i : n);
  r->exit_pc = t->p[xi]; r->xi = (int)xi;
  r->data = words_spanned(p, seen);
  r->bytes = seen;
  return 1;
}

/* c.malloc.realloc's byte copy: do { a3 = *s11++; *a2++ = a3 } while (s11 != a1). A copy whose
 * destination starts inside its source (the forward byte order matters there) declines. */
static int u_bcopy(trig_t* t, ures_t* r) {
  uint32_t s = X[R_S11], d = X[R_A2], n = X[R_A1] - s;
  if (n == 0 || !inmem(s, n) || !inmem(d, n)) return 0;
  if (d > s && d < s + n + 4) return 0;
  if (d < X[2] && d + n > X[2] - 64) return 0;
  move_bytes(d, s, n);
  X[R_A3] = rd8(d + n - 1);
  X[R_S11] = s + n;
  X[R_A2] = d + n;
  r->exit_pc = t->p[0]; r->xi = 0;
  r->data = move_data(d, s, n);
  r->bytes = n;
  return 1;
}

static int loop_unit_run(trig_t* t, ures_t* r) {
  switch (t->kind) {
    case U_TOKEST: return u_tokest(t, r);
    case U_JSCAN: return u_jscan(t, r);
    case U_JESC: return u_jesc(t, r);
    case U_ASCII: return u_ascii(t, r);
    case U_MASKP: return u_maskp(t, r);
    case U_MEMCHR: return u_memchr(t, r);
    case U_BCOPY: return u_bcopy(t, r);
  }
  return 0;
}
