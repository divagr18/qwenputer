// stories260K as a program on the chipformer RV32IM machine: integer only, int8 weights
// (per-row scale) in the DROM, int8 activations per matmul (per-vector scale), Q16 residual,
// int16 Q8 KV cache in RAM.  -DMM=0 naive matmul, -DMM=1 4-row blocked + unrolled.
#include <stdint.h>
#include "rt.h"
#include "weights.h"

#ifndef NTOK
#define NTOK 8
#endif
#ifndef CACHEPOS
#define CACHEPOS 32
#endif
#ifndef MM
#define MM 0
#endif

static int32_t x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], hb[HID], hb2[HID], att[CACHEPOS];
static int32_t logits[VOCAB];
int8_t xq[HID];  // global: dot8.S reads it
static int16_t kc[NL][CACHEPOS][KVD], vc[NL][CACHEPOS][KVD];

// v (Q16) -> xq (int8), returns amax (Q16)
static int32_t quantize(const int32_t *v, int n) {
    int32_t amax = 0;
    for (int i = 0; i < n; i++) {
        int32_t a = v[i] < 0 ? -v[i] : v[i];
        if (a > amax) amax = a;
    }
    if (!amax) amax = 1;
    int32_t r = (127 << 24) / amax;
    for (int i = 0; i < n; i++) xq[i] = (int8_t)((v[i] * r + (1 << 23)) >> 24);
    return amax;
}

static inline int32_t rescale(int32_t acc, int32_t amax, int32_t m) {
    int32_t t = (int32_t)(((int64_t)acc * amax) >> 16);
    return (int32_t)(((int64_t)t * m) >> 16);
}

#if MM == 0
static void matmul(int32_t *out, const int8_t *w, const int32_t *M, int n, int d, int32_t amax) {
    for (int r = 0; r < d; r++) {
        const int8_t *wr = w + r * n;
        int32_t acc = 0;
        for (int i = 0; i < n; i++) acc += xq[i] * wr[i];
        out[r] = rescale(acc, amax, M[r]);
    }
}
#elif MM == 2
// 8 rows at a time, n fixed at compile time and fully unrolled: every weight load is an
// immediate offset from one row-block pointer (no pointer updates inside), each activation byte
// is loaded once for 8 products
#define MMKERNEL(NAME, N)                                                                     \
static void NAME(int32_t *out, const int8_t *w, const int32_t *M, int d, int32_t amax) {      \
    for (int r = 0; r < d; r += 8, w += 8 * N) {                                              \
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;               \
        _Pragma("GCC unroll 200")                                                             \
        for (int i = 0; i < N; i++) {                                                         \
            int32_t xi = xq[i];                                                               \
            a0 += xi * w[i]; a1 += xi * w[N + i]; a2 += xi * w[2 * N + i];                    \
            a3 += xi * w[3 * N + i]; a4 += xi * w[4 * N + i]; a5 += xi * w[5 * N + i];        \
            a6 += xi * w[6 * N + i]; a7 += xi * w[7 * N + i];                                 \
        }                                                                                     \
        int32_t acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};                                    \
        for (int k = 0; k < 8 && r + k < d; k++) out[r + k] = rescale(acc[k], amax, M[r + k]); \
    }                                                                                         \
}
MMKERNEL(mm_dim, DIM)
MMKERNEL(mm_hid, HID)
static void matmul(int32_t *out, const int8_t *w, const int32_t *M, int n, int d, int32_t amax) {
    if (n == DIM) mm_dim(out, w, M, d, amax); else mm_hid(out, w, M, d, amax);
}
#elif MM == 3
// hand-scheduled assembly kernels (dot8.S, from gen_asm.py): 8 rows per call
void dot8_64(const int8_t *w, int32_t *acc);
void dot8_172(const int8_t *w, int32_t *acc);
static void matmul(int32_t *out, const int8_t *w, const int32_t *M, int n, int d, int32_t amax) {
    int32_t acc[8];
    int r = 0;
    for (; r + 8 <= d; r += 8, w += 8 * n) {
        if (n == DIM) dot8_64(w, acc); else dot8_172(w, acc);
        for (int k = 0; k < 8; k++) out[r + k] = rescale(acc[k], amax, M[r + k]);
    }
    for (; r < d; r++, w += n) {  // tail rows (HID = 172 = 21 * 8 + 4)
        int32_t a = 0;
        for (int i = 0; i < n; i++) a += xq[i] * w[i];
        out[r] = rescale(a, amax, M[r]);
    }
}
#else
// 4 rows at a time: each activation byte is loaded once for 4 products; 4-way unrolled
static void matmul(int32_t *out, const int8_t *w, const int32_t *M, int n, int d, int32_t amax) {
    for (int r = 0; r < d; r += 4) {
        const int8_t *w0 = w + r * n, *w1 = w0 + n, *w2 = w1 + n, *w3 = w2 + n;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i = 0; i < n; i += 4) {
#define STEP(k) { int32_t xi = xq[i + k]; a0 += xi * w0[i + k]; a1 += xi * w1[i + k]; \
                  a2 += xi * w2[i + k]; a3 += xi * w3[i + k]; }
            STEP(0) STEP(1) STEP(2) STEP(3)
        }
        out[r] = rescale(a0, amax, M[r]);
        out[r + 1] = rescale(a1, amax, M[r + 1]);
        out[r + 2] = rescale(a2, amax, M[r + 2]);
        out[r + 3] = rescale(a3, amax, M[r + 3]);
    }
}
#endif

static uint32_t isqrt64(uint64_t v) {
    uint64_t r = 0, b = (uint64_t)1 << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (uint32_t)r;
}

static void rmsnorm(int32_t *o, const int32_t *v, const int32_t *w) {
    int64_t ss = 0;
    for (int i = 0; i < DIM; i++) ss += (int64_t)v[i] * v[i];
    uint32_t r = isqrt64((uint64_t)(ss / DIM) + 42950);  // Q16 rms (eps 1e-5)
    int64_t inv = ((int64_t)1 << 48) / r;                  // Q32 1/rms
    for (int i = 0; i < DIM; i++) {
        int32_t t = (int32_t)(((int64_t)v[i] * w[i]) >> 16);  // Q16
        o[i] = (int32_t)((t * inv) >> 32);
    }
}

// e^-d for d >= 0 (Q16) -> Q15
static int32_t exp_neg(int32_t d) {
    int32_t y = (int32_t)(((int64_t)d * 94548) >> 16);  // d * log2(e)
    int32_t k = y >> 16;
    if (k >= 15) return 0;
    return EXP2[(y >> 8) & 255] >> k;
}

static void rope(int32_t *v, int pos, int n) {
    const int16_t *t = ROPE + pos * HS;
    for (int i = 0; i < n; i += 2) {
        int32_t c = t[i % HS], s = t[i % HS + 1], a = v[i], b = v[i + 1];
        v[i] = (int32_t)(((int64_t)a * c - (int64_t)b * s) >> 14);
        v[i + 1] = (int32_t)(((int64_t)a * s + (int64_t)b * c) >> 14);
    }
}

static inline int16_t to_q8(int32_t v) {
    v >>= 8;
    return (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
}

static int forward(int tok, int pos) {
    for (int i = 0; i < DIM; i++)
        x[i] = (int32_t)(((int64_t)Q_emb[tok * DIM + i] * M_emb[tok] * 127) >> 16);
    for (int l = 0; l < NL; l++) {
        rmsnorm(xb, x, R_rms_att + l * DIM);
        int32_t am = quantize(xb, DIM);
        matmul(q, Q_wq + l * DIM * DIM, M_wq + l * DIM, DIM, DIM, am);
        matmul(kk, Q_wk + l * KVD * DIM, M_wk + l * KVD, DIM, KVD, am);
        matmul(vv, Q_wv + l * KVD * DIM, M_wv + l * KVD, DIM, KVD, am);
        rope(q, pos, DIM);
        rope(kk, pos, KVD);
        for (int i = 0; i < KVD; i++) { kc[l][pos][i] = to_q8(kk[i]); vc[l][pos][i] = to_q8(vv[i]); }
        for (int h = 0; h < NH; h++) {
            int g = h / (NH / NKV);
            int32_t q8[HS], mx = INT32_MIN;
            for (int j = 0; j < HS; j++) q8[j] = q[h * HS + j] >> 8;
            for (int t = 0; t <= pos; t++) {
                const int16_t *kr = kc[l][t] + g * HS;
                int32_t s = 0;
                for (int j = 0; j < HS; j++) s += q8[j] * kr[j];
                s = (int32_t)(((int64_t)s * 23170) >> 16);  // / sqrt(8)
                att[t] = s;
                if (s > mx) mx = s;
            }
            int32_t sum = 0;
            for (int t = 0; t <= pos; t++) { att[t] = exp_neg(mx - att[t]); sum += att[t]; }
            for (int t = 0; t <= pos; t++) att[t] = (att[t] << 15) / sum;
            for (int j = 0; j < HS; j++) {
                int32_t o = 0;
                for (int t = 0; t <= pos; t++) o += att[t] * vc[l][t][g * HS + j];
                xb[h * HS + j] = o >> 7;  // Q23 -> Q16
            }
        }
        am = quantize(xb, DIM);
        matmul(q, Q_wo + l * DIM * DIM, M_wo + l * DIM, DIM, DIM, am);
        for (int i = 0; i < DIM; i++) x[i] += q[i];
        rmsnorm(xb, x, R_rms_ffn + l * DIM);
        am = quantize(xb, DIM);
        matmul(hb, Q_w1 + l * HID * DIM, M_w1 + l * HID, DIM, HID, am);
        matmul(hb2, Q_w3 + l * HID * DIM, M_w3 + l * HID, DIM, HID, am);
        for (int i = 0; i < HID; i++) {
            int32_t a = hb[i], sg;
            if (a >= 0) { int32_t e = exp_neg(a); sg = (1 << 30) / (32768 + e); }
            else { int32_t e = exp_neg(-a); sg = (e << 15) / (32768 + e); }
            int32_t si = (int32_t)(((int64_t)a * sg) >> 15);
            hb[i] = (int32_t)(((int64_t)si * hb2[i]) >> 16);
        }
        am = quantize(hb, HID);
        matmul(q, Q_w2 + l * DIM * HID, M_w2 + l * DIM, HID, DIM, am);
        for (int i = 0; i < DIM; i++) x[i] += q[i];
    }
    rmsnorm(xb, x, R_rms_fin);
    int32_t am = quantize(xb, DIM);
    matmul(logits, Q_emb, M_emb, DIM, VOCAB, am);
    int best = 0;
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int main(void) {
    int tok = 1;
    for (int pos = 0; pos < NTOK && pos < CACHEPOS; pos++) {
        int nxt = forward(tok, pos);
        int a = VOFF[nxt], b = VOFF[nxt + 1];
        if (tok == 1 && VSTR[a] == ' ') a++;
        for (int i = a; i < b; i++) rv_putchar((char)VSTR[i]);
        tok = nxt;
    }
    rv_putchar('\n');
    return 0;
}
