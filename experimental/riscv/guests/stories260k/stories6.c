// stories260K as a program on the chipformer RV32 machine with VDOT / WB (v4):
// - int4 weights in the WROM (RMSNorm gains folded in), matvecs in the vd2.S kernels: 4 rows per
//   WB, the rescale fused (out = mulh(acc * S'_r, F), F = the input vector's scale incl. 1/rms)
// - RMSNorm is only 1/rms (int8 quantization is scale-invariant, so the norm never touches x)
// - SiLU from a table, RoPE in 32 bits, quantize with mulh rounding
// - attention (v6): int8 K and transposed V caches (per-layer scales), a window of the first token
//   plus the latest 31, scores and p.V with SMAQA, softmax from an int8 exp table
#include <stdint.h>
#include "rt.h"
#include "weights_vd2.h"

#ifndef NTOK
#define NTOK 8
#endif
#define W 32  // cache slots: slot 0 holds position 0, slots 1..31 a ring of the latest
#define SMAQA(acc, a, b) __asm__ volatile(".insn r 0x0B, 7, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))

static int32_t x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], hb[HID + 4], hb2[HID + 4];
static int32_t sc0[W], sc1[W], logits[VOCAB];
static int8_t pa[W] __attribute__((aligned(4))), pb[W] __attribute__((aligned(4)));
static int8_t xq[192] __attribute__((aligned(4)));  // zero past n
const int8_t *vk_x;  // the kernels' activations
static int8_t kq[NL][W][KVD] __attribute__((aligned(4)));  // K, int8 (scale KMAX / 127)
static int8_t vt[NL][KVD][W] __attribute__((aligned(4)));  // V transposed: 4 slots a word

void vk_8_init_scaled(int32_t *out, int wb, int rows, const int32_t *S, int32_t F);
void vk_8_init_raw(int32_t *out, int wb, int rows, const int32_t *S, int32_t F);
void vk_8_acc_raw(int32_t *out, int wb, int rows, const int32_t *S, int32_t F);
void vk_6_acc_scaled(int32_t *out, int wb, int rows, const int32_t *S, int32_t F);

static inline int32_t mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }

static int32_t vmax;  // the latest producer loop's max |v| (they track it, so quantize need not scan)
#define TRACK(v) do { int32_t a_ = (v) < 0 ? -(v) : (v); if (a_ > vmax) vmax = a_; } while (0)

// v (Q16) -> xq = round(v * 127 / amax) (amax: the producer's vmax); returns amax
static int32_t quantize(const int32_t *v, int n, int32_t amax) {
    if (amax < 256) amax = 256;
    int32_t r = 0x7F000000 / amax;  // 127 * 2^24 / amax
    int32_t h = amax / 254;          // half a step: mulh then rounds to nearest
#pragma GCC unroll 4
    for (int i = 0; i < n; i++) xq[i] = (int8_t)mulh((v[i] + h) << 8, r);
    return amax;
}

// F = 2^33 / (127 rms) * amax for the vector just quantized: with |xq| <= 127 from v * 127 / amax,
// rms(v) = amax / 127 * sqrt(sum xq^2 / 64), so F = 2^36 / sqrt(sum xq^2) (amax cancels)
static int32_t norm_f(void) {
    int32_t ss = 0;
#pragma GCC unroll 8
    for (int i = 0; i < DIM; i++) ss += xq[i] * xq[i];
    uint32_t m = (uint32_t)ss << 10, r = 0, b = 1u << 30;  // isqrt: 32 sqrt(ss)
    while (b > m) b >>= 2;
    while (b) {
        if (m >= r + b) { m -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (int32_t)((0x80000000u / r) << 10);  // 2^41 / (32 sqrt(ss))
}

// F for an input without a norm: amax / 127 / 2^16 * 2^33
static int32_t plain_f(int32_t amax) { return (int32_t)(((uint32_t)amax << 9) / 127 << 8); }

// out[r] (Q16) = W_r . v for the quantized v: S'_r = s_r * 2^15, f = the input's scale * 2^33
static void matmul(int32_t *out, int wb, const int32_t *S, int n, int rows, int32_t f) {
    vk_x = xq;
    if (n == DIM) vk_8_init_scaled(out, wb, rows, S, (int32_t)f);
    else {
        int l = wb;  // (n == HID: wb is the layer)
        vk_8_init_raw(out, WB_w2c0[l], rows, 0, 0);
        vk_x = xq + 64;
        vk_8_acc_raw(out, WB_w2c1[l], rows, 0, 0);
        vk_x = xq + 128;
        vk_6_acc_scaled(out, WB_w2c2[l], rows, S, (int32_t)f);
    }
}

static void rope(int32_t *v, int pos, int n) {
    const int16_t *t = ROPE + pos * HS;
    for (int i = 0; i < n; i += 2) {
        int32_t c = t[i % HS], s = t[i % HS + 1], a = v[i] >> 6, b = v[i + 1] >> 6;
        v[i] = (a * c - b * s) >> 8;
        v[i + 1] = (a * s + b * c) >> 8;
    }
}

static inline int8_t q8_of(int32_t v, int32_t r, int32_t h) {  // round(v * 127 / max), clamped
    int32_t t = mulh((v + (v < 0 ? -h : h)) << 8, r);
    return (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t);
}

static inline uint32_t pack4(const int8_t *b) {
    return (uint8_t)b[0] | (uint8_t)b[1] << 8 | (uint8_t)b[2] << 16 | (uint32_t)(uint8_t)b[3] << 24;
}

// the two heads of KV group g over the n filled slots: scores (SMAQA), int8 softmax, p.V (SMAQA)
static void attend(int l, int g, int n) {
    uint32_t qw[2][2];
    int32_t ca[2];
    for (int hh = 0; hh < 2; hh++) {  // each head's query: int8 with its own scale
        const int32_t *qv = q + (2 * g + hh) * HS;
        int32_t amax = 256;
        for (int j = 0; j < HS; j++) { int32_t a = qv[j] < 0 ? -qv[j] : qv[j]; if (a > amax) amax = a; }
        int32_t r = 0x7F000000 / amax, hf = amax / 254;
        int8_t b[HS];
        for (int j = 0; j < HS; j++) b[j] = q8_of(qv[j], r, hf);
        qw[hh][0] = pack4(b);
        qw[hh][1] = pack4(b + 4);
        // exp-table index per unit of (max dot - dot), * 2^32: 32 / sqrt(8) (amax / 127 / 2^16)
        // (KMAX / 127 / 2^16) * 2^32 = amax KMAX / 1425.6 = (amax KMAX >> 10) * 47075 >> 16
        ca[hh] = (int32_t)(((((int64_t)amax * KMAX[l]) >> 10) * 47075) >> 16);
    }
    int32_t m0 = INT32_MIN, m1 = INT32_MIN;
    for (int t = 0; t < n; t++) {
        const uint32_t *kw = (const uint32_t *)&kq[l][t][g * HS];
        uint32_t k0 = kw[0], k1 = kw[1];
        int32_t s0 = 0, s1 = 0;
        SMAQA(s0, qw[0][0], k0); SMAQA(s0, qw[0][1], k1);
        SMAQA(s1, qw[1][0], k0); SMAQA(s1, qw[1][1], k1);
        sc0[t] = s0; sc1[t] = s1;
        if (s0 > m0) m0 = s0;
        if (s1 > m1) m1 = s1;
    }
    int32_t sum0 = 0, sum1 = 0;
    for (int t = 0; t < n; t++) {
        uint32_t i0 = (uint32_t)mulh(m0 - sc0[t], ca[0]), i1 = (uint32_t)mulh(m1 - sc1[t], ca[1]);
        int32_t p0 = i0 < 256 ? P8[i0] : 0, p1 = i1 < 256 ? P8[i1] : 0;
        pa[t] = (int8_t)p0; pb[t] = (int8_t)p1;
        sum0 += p0; sum1 += p1;
    }
    int nw = (n + 3) >> 2;
    for (int t = n; t < 4 * nw; t++) { pa[t] = 0; pb[t] = 0; }
    int32_t o0[HS] = {0}, o1[HS] = {0};
    for (int w = 0; w < nw; w++) {
        uint32_t p0 = ((const uint32_t *)pa)[w], p1 = ((const uint32_t *)pb)[w];
#pragma GCC unroll 8
        for (int j = 0; j < HS; j++) {
            uint32_t vw = ((const uint32_t *)vt[l][g * HS + j])[w];
            SMAQA(o0[j], p0, vw);
            SMAQA(o1[j], p1, vw);
        }
    }
    // o (Q16) = sum p v8 * VMAX / 127 / sum p
    int32_t r0 = (VMAX[l] << 12) / (127 * sum0), r1 = (VMAX[l] << 12) / (127 * sum1);
    int32_t *x0 = xb + 2 * g * HS, *x1 = x0 + HS;
    for (int j = 0; j < HS; j++) {
        x0[j] = (int32_t)(((int64_t)o0[j] * r0) >> 12);
        x1[j] = (int32_t)(((int64_t)o1[j] * r1) >> 12);
        TRACK(x0[j]); TRACK(x1[j]);
    }
}

static int forward(int tok, int pos) {
    vmax = 0;
    for (int i = 0; i < DIM; i++) {
        x[i] = (int32_t)(((int64_t)Q_emb[tok * DIM + i] * S_emb[tok]) >> 16);
        TRACK(x[i]);
    }
    for (int l = 0; l < NL; l++) {
        quantize(x, DIM, vmax);
        int32_t f = norm_f();
        matmul(q, WB_wq[l], S_wq + l * DIM, DIM, DIM, f);
        matmul(kk, WB_wk[l], S_wk + l * KVD, DIM, KVD, f);
        matmul(vv, WB_wv[l], S_wv + l * KVD, DIM, KVD, f);
        rope(q, pos, DIM);
        rope(kk, pos, KVD);
        int slot = pos == 0 ? 0 : 1 + (pos - 1) % (W - 1), n = pos + 1 < W ? pos + 1 : W;
        int32_t kr = 0x7F000000 / KMAX[l], kh = KMAX[l] / 254;
        int32_t vr = 0x7F000000 / VMAX[l], vh = VMAX[l] / 254;
        for (int i = 0; i < KVD; i++) { kq[l][slot][i] = q8_of(kk[i], kr, kh); vt[l][i][slot] = q8_of(vv[i], vr, vh); }
        vmax = 0;
        for (int g = 0; g < NKV; g++) attend(l, g, n);
        matmul(q, WB_wo[l], S_wo + l * DIM, DIM, DIM, plain_f(quantize(xb, DIM, vmax)));
        vmax = 0;
        for (int i = 0; i < DIM; i++) { x[i] += q[i]; TRACK(x[i]); }
        quantize(x, DIM, vmax);
        f = norm_f();
        matmul(hb, WB_w1[l], S_w1 + l * HID, DIM, HID, f);
        matmul(hb2, WB_w3[l], S_w3 + l * HID, DIM, HID, f);
        vmax = 0;
        for (int i = 0; i < HID; i++) {  // silu(a) * b: table SILU (Q12) at a / 32
            int32_t a = hb[i], s;
            if (a >= 8 << 16) s = a >> 4;
            else if (a < -(8 << 16)) s = 0;
            else s = SILU[(a >> 11) + 256];
            hb[i] = (s * (hb2[i] >> 8)) >> 4;  // Q12 * Q8 -> Q16
            TRACK(hb[i]);
        }
        matmul(q, l, S_w2 + l * DIM, HID, DIM, plain_f(quantize(hb, HID, vmax)));
        vmax = 0;
        for (int i = 0; i < DIM; i++) { x[i] += q[i]; TRACK(x[i]); }
    }
    // the classifier: the final norm only rescales the logits, so argmax skips it
    quantize(x, DIM, vmax);
    vk_x = xq;
    vk_8_init_scaled(logits, WB_CLS, VOCAB, S_cls, 1 << 30);
    int best = 0;
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int main(void) {
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
