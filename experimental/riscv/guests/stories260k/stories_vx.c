// stories260K on the RV core with the VX vector unit (chipformer/rv/vx.py, docs/vx.md): stories6.c's
// model with the loops as vector ops. The unit has its own memories (vector registers, the int8
// activation vector VA, the K / V caches, the weights), so the program only names registers and
// rows; every op's configuration travels in its record.
// - matvecs: VMAT over 64-lane row-chunks, out = acc * S_r; the input's scale F is folded into
//   whatever uses the output (a quantize factor, GLU's input scale, the residual add's factor)
// - quantize: VQ, a 16-bit window of each value times a 15-bit factor, the window from the
//   producer's max |v| (VB tokens keep it; VGET reads it and its bit length)
// - residual adds, SiLU * b and RoPE: VEW; the embedding row: computed here, VPUT in
// - attention: q per head by VQ, scores and p.V by VMATR over the caches, softmax by VEXP
// - 1/rms from VRED's sum of squares of VA, softmax sums by VRED
#include <stdint.h>
#include "rt.h"
#include "weights_vx.h"

#ifndef NTOK
#define NTOK 8
#endif
#define W 32       // cache slots: slot 0 holds position 0, slots 1..31 a ring of the latest
#define OFFQ 3     // VA chunks: xq in 0..2, the head's query in 3, probabilities in 4
#define OFFP 4
// vector registers: bank A 0..7, bank B 8..15 (an op reads at most one of each)
enum { RX = 0, RQ = 1, RK = 2, RH = 3 /* 3..5 */, RSC = 6, RXB = 8, RV = 9, RG = 10 /* 10..12 */,
       RO = 13 };

#define VOP(f3, cfg, a, b) \
    __asm__ volatile(".insn r 0x5B, " #f3 ", 0, %0, %1, %2" ::"r"(cfg), "r"(a), "r"(b) : "memory")
static inline int32_t vget(int32_t sel) {
    int32_t r;
    __asm__ volatile(".insn r 0x5B, 5, 0, %0, %1, zero" : "=r"(r) : "r"(sel) : "memory");
    return r;
}
static inline uint32_t mulhu(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
static inline int32_t mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }

static void vmat(int dst, int doff, int n, int k, int mode, int keep, int chunk, int base) {
    VOP(0, dst | n << 4 | k << 11 | mode << 13 | keep << 14 | chunk << 15, base | doff << 13, 0);
}
static void vmatr(int dst, int eoff, int n, int mode, int keep, int off, int base, int32_t s) {
    VOP(1, dst | n << 4 | mode << 10 | keep << 11 | off << 12, base | eoff << 10, s);
}
static void vq(int src, int eoff, int n, int dst, int sh, int off, int32_t r) {
    VOP(2, src | eoff << 4 | n << 10 | dst << 17 | sh << 19, off, r);
}
static void vew(int a, int b, int eoff, int n, int op, int keep, int sh, int so, int shb, int32_t p) {
    VOP(3, a | b << 4 | n << 8 | op << 15 | keep << 17 | eoff << 18, sh | (so & 63) << 5 | shb << 11, p);
}
static void vexp(int src, int n, int off, int ksh, int32_t m, int32_t c) {
    VOP(4, src | n << 4 | off << 11 | ksh << 20, m, c);
}
static void vred(int n, int mode, int off) { VOP(6, n | mode << 7 | off << 8, 0, 0); }
static void vput(int target, int32_t lo, int32_t hi) { VOP(7, target, lo, hi); }

static const uint8_t BL8[256] = {
#define B4(n) n, n, n, n
#define B16(n) B4(n), B4(n), B4(n), B4(n)
    0, 1, 2, 2, B4(3), B4(4), B4(4), B16(5), B16(6), B16(6), B16(7), B16(7), B16(7), B16(7),
    B16(8), B16(8), B16(8), B16(8), B16(8), B16(8), B16(8), B16(8)};
static int bitlen(uint32_t u) {
    if (u >> 16) return u >> 24 ? 24 + BL8[u >> 24] : 16 + BL8[u >> 16];
    return u >> 8 ? 8 + BL8[u >> 8] : BL8[u];
}
// the latest tracker's window (VGET 3: the bit length of its max)
static int vwin(void) { int b = vget(3); return b > 15 ? b - 15 : 0; }
static int32_t vmax_bl(int *bl) { *bl = vget(3); return vget(0); }
// a factor f as r * 2^sr with r < 2^15; returns sr
static int fwin(uint32_t f, int32_t *r) {
    int b = bitlen(f), sr = b > 15 ? b - 15 : 0;
    *r = (int32_t)(f >> sr);
    return sr;
}

static uint32_t KQ[NL], VQF[NL], KC[NL];  // 127 * 2^40 / KMAX, / VMAX; KMAX * 47075 >> 10

// VM[src][..n] -> VA[off..] = round(v * 127 / amax): a 16-bit window of v times r / 2^15 (bl: the
// bit length of amax)
static void quantize(int src, int n, int32_t amax, int bl, int off) {
    if (amax < 256) amax = 256, bl = 9;
    int sh = bl > 11 ? bl - 11 : 0;
    vq(src, 0, n, 0, sh, off, (127 << 15) / (amax >> sh));
}

// the window and factor of a quantize by m * 2^sh / 2^25 (m = mulhu(F, 127 * 2^40 / max)):
// returns sh, with *r in [2^11, 2^12) where m allows
static int cache_q(uint32_t m, int32_t *r) {
    int sh = 37 - bitlen(m);
    if (sh < 0) sh = 0;
    if (sh > 31) sh = 31;
    *r = (int32_t)(sh >= 25 ? m << (sh - 25) : m >> (25 - sh));
    return sh;
}

// F = 2^36 / sqrt(sum of squares of VA[0..n)) (stories6.c's norm_f)
static int32_t norm_f(int n) {
    vred(n, 0, 0);
    uint32_t m = (uint32_t)vget(1) << 10;  // r = floor(sqrt(m)): Newton from 2^ceil(bits / 2)
    uint32_t r = 1u << ((bitlen(m) + 1) >> 1), t;
    while ((t = (r + m / r) >> 1) < r) r = t;
    return (int32_t)((0x80000000u / r) << 10);
}

static int32_t plain_f(int32_t amax) { return (int32_t)(((uint32_t)amax << 9) / 127 << 8); }

// rows of a 64-input matrix into registers dst.. (64 rows a register, 16 an op); returns the
// window of the outputs
static int matmul(int dst, int base, int rows) {
    for (int r = 0; r < rows; r += 16)
        vmat(dst + r / 64, r % 64, rows - r < 16 ? rows - r : 16, 1, 0, r > 0, 0, base + r);
    return vwin();
}

// x += mulh(o, f) for the 64 rows o of a matvec in RO (sh: their window); the max |x| is tracked
static void residual(int sh, uint32_t f) {
    int32_t r;
    int sr = fwin(f, &r);
    for (int i = 0; i < DIM; i += 32) vew(RX, RO, i, 32, 0, i > 0, sh, sh + sr - 32, 0, r);
}

static int32_t qmax[NH], xbmax;
static int qbl[NH];

// the two heads of KV group g over the n filled slots; q's true values are mulh(q, f)
static void attend(int l, int g, int n, int32_t f) {
    int row = (4 * l + g) * W;
    for (int hh = 0; hh < 2; hh++) {
        int h = 2 * g + hh, bl = qbl[h];
        int32_t araw = qmax[h], amax = mulh(araw, f);
        if (amax < 256) {  // stories6.c floors the query's max at 256
            amax = 256;
            araw = (int32_t)((256ull << 32) / (uint32_t)f);
            bl = bitlen((uint32_t)araw);
        }
        int sh = bl > 11 ? bl - 11 : 0;
        vq(RQ, HS * h, HS, 0, sh, 64 * OFFQ, (127 << 15) / (araw >> sh));
        // ca = amax * KC >> 16 (stories6.c's exp-table scale), as its top 16 bits
        uint32_t lo = (uint32_t)amax * KC[l], hi = mulhu((uint32_t)amax, KC[l]);
        int32_t ca = (int32_t)(hi << 16 | lo >> 16), c, ksh;
        if (hi >> 15) {  // ca >= 2^31: stories6.c's int32 ca is negative there, a hard max
            c = 65535, ksh = 0;
        } else {
            int k1 = bitlen((uint32_t)ca) - 16;
            if (k1 < 1) k1 = 1;
            c = ca >> k1, ksh = 32 - k1;
        }
        vmatr(RSC, 0, n, 0, 0, OFFQ, row, 0);  // scores over the K cache, the signed max for VEXP
        vexp(RSC, n, 64 * OFFP, ksh, vget(0), c);
        vred(n, 1, OFFP);
        int32_t r = (VMAX[l] << 12) / (127 * vget(1));
        // o (Q16) = sum p v8 * VMAX / 127 / sum p; max |o| over the heads
        vmatr(RXB, HS * h, n, 1, 0, OFFP, row, r);
        int32_t om = vget(0);
        if (om > xbmax) xbmax = om;
    }
}

static int forward(int tok, int pos) {
    // the embedding row, into RX a pair at a time
    int32_t s = S_emb[tok], vmax = 0;
    const int8_t *e = Q_emb + tok * DIM;
    for (int i = 0; i < DIM; i += 2) {
        int32_t a = (int32_t)(((int64_t)e[i] * s) >> 16), b = (int32_t)(((int64_t)e[i + 1] * s) >> 16);
        int32_t aa = a < 0 ? -a : a, ab = b < 0 ? -b : b;
        if (aa > vmax) vmax = aa;
        if (ab > vmax) vmax = ab;
        vput(RX * 32 + i / 2, a, b);
    }
    int bl = bitlen((uint32_t)vmax);
    const int16_t *rp = ROPE + pos * HS;
    for (int k = 0; k < HS / 2; k++) vput(512 + k, rp[2 * k], rp[2 * k + 1]);
    for (int l = 0; l < NL; l++) {
        quantize(RX, DIM, vmax, bl, 0);
        int32_t f = norm_f(DIM);
        int aq = matmul(RQ, WB_wq[l], DIM);
        int ak = matmul(RK, WB_wk[l], KVD);
        matmul(RV, WB_wv[l], KVD);
        for (int h = 0; h < NH; h++) {  // RoPE per head, keeping each head's max for its quantize
            vew(RQ, 0, HS * h, HS, 2, 0, aq, 0, 0, 0);
            qmax[h] = vmax_bl(&qbl[h]);
        }
        vew(RK, 0, 0, KVD, 2, 0, ak, 0, 0, 0);
        int slot = pos == 0 ? 0 : 1 + (pos - 1) % (W - 1), n = pos + 1 < W ? pos + 1 : W;
        int32_t r;
        int sh = cache_q(mulhu((uint32_t)f, KQ[l]), &r);
        vq(RK, 0, KVD, 1, sh, 4 * l * W + slot, r);
        sh = cache_q(mulhu((uint32_t)f, VQF[l]), &r);
        vq(RV, 0, KVD, 2, sh, 4 * l * W + slot, r);
        xbmax = 0;
        for (int g = 0; g < NKV; g++) attend(l, g, n, f);
        vmax = xbmax, bl = bitlen((uint32_t)xbmax);
        quantize(RXB, DIM, vmax, bl, 0);
        if (vmax < 256) vmax = 256;
        int ao = matmul(RO, WB_wo[l], DIM);
        residual(ao, (uint32_t)plain_f(vmax));
        vmax = vmax_bl(&bl);
        quantize(RX, DIM, vmax, bl, 0);
        f = norm_f(DIM);
        int sha = matmul(RH, WB_w1[l], HID);
        int shb = matmul(RG, WB_w3[l], HID);
        int32_t ra;
        int sra = fwin((uint32_t)f, &ra);
        // hb = silu(true a) * (raw b >> shb) >> 8: 2^(36 - shb) / F of the true hb
        for (int i = 0; i < HID; i += 16)
            vew(RH + i / 64, RG + i / 64, i % 64, HID - i < 16 ? HID - i : 16, 1, i > 0, sha,
                sha + sra - 32, shb, ra);
        vmax = vmax_bl(&bl);
        if (vmax < 256) vmax = 256, bl = 9;
        int shq = bl > 11 ? bl - 11 : 0;
        int32_t rq = (127 << 15) / (vmax >> shq);
        for (int i = 0; i < HID; i += 64) vq(RH + i / 64, 0, HID - i < 64 ? HID - i : 64, 0, shq, i, rq);
        for (int r2 = 0; r2 < DIM; r2 += 8) vmat(RO, r2, 8, 3, 0, r2 > 0, 0, WB_w2[l] + r2);
        int a2 = vwin();
        uint32_t fp = mulhu((uint32_t)plain_f(vmax), (uint32_t)f);  // * F / 2^(36 - shb)
        fp = shb >= 4 ? fp << (shb - 4) : fp >> (4 - shb);
        residual(a2, fp);
        vmax = vmax_bl(&bl);
    }
    // the classifier: the final norm only rescales the logits, so argmax skips it
    quantize(RX, DIM, vmax, bl, 0);
    for (int r = 0; r < VOCAB; r += 16) vmat(0, 0, 16, 1, 1, r > 0, 0, WB_CLS + r);
    return vget(2) - WB_CLS;
}

int main(void) {
    for (int l = 0; l < NL; l++) {
        KQ[l] = (uint32_t)((127ull << 40) / (uint32_t)KMAX[l]);
        VQF[l] = (uint32_t)((127ull << 40) / (uint32_t)VMAX[l]);
        KC[l] = (uint32_t)(((uint64_t)KMAX[l] * 47075) >> 10);
    }
    int tok = 1;
    for (int pos = 0; pos < NTOK && pos < MAXPOS; pos++) {
        int nxt = forward(tok, pos);
        int a = VOFF[nxt], b = VOFF[nxt + 1];
        if (tok == 1 && VSTR[a] == ' ') a++;
        for (int i = a; i < b; i++) rv_putchar((char)VSTR[i]);
        tok = nxt;
    }
    rv_putchar('\n');
    return 0;
}
