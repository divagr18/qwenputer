"""Executor v4: the whole circuit step as one persistent CUDA kernel, the chip-token loop on the GPU.

Same arithmetic as tritonexec.Executor2 (the IR's semantics in fp32 over the circuit's weights,
every value exact), organized for latency instead of launches:

  phases    the circuit's nodes (heads and MLP nodes) are list-scheduled by the dims they read
            and write: a node runs in the first phase after every lower-level node that writes a
            dim it reads, and no earlier than any lower- or same-level node that reads a dim it
            writes. Its inputs there equal its inputs at its own level, on every dim it reads, at
            every step (so its conv history too). The 20 sublayers of the rv-q4p2 circuit become
            13 phases: both ROM sublayers run in one, and both RAM sublayers, the data ROM and
            the text buffer in another (they read addresses computed below them).
  one kernel  a cooperative grid of one block per SM and a grid barrier (an arrival counter)
            after each phase. Each block keeps its own copy of the residual in shared memory. A
            phase's outputs go (atomic adds) into that phase's own buffer; at the next phase's
            start each block adds the dims the phase writes and the block reads into its copy.
  the loop  after the last phase every block decodes the output code into the next token itself
            (the same table as core.decode_code) and loads that token's embedding, so the next
            step starts without a host round trip. A launch runs until the predicted token is
            not a chip token (text, nothing, or a budget), so RvChat calls the GPU once per chip
            burst, not once per token.
  work      per phase and block: every sparse row (MLP a / b, head q / k / v / z / a / beta, over
            the residual and the block's lag history) is cut into pieces of at most PMAX terms;
            the pieces are packed onto the block's threads (thread-major term slots, balanced by
            count; the next phase's slots load into registers while this phase finishes), each
            thread's partial sums go to shared memory, and a row is the sum of its pieces. Then
            each MLP unit adds coef * a * relu(b) to its output dim, and each value head does
            Executor2's decay, one-row write, at-most-QHOT-hot read, ballast norm and gated
            output. The kernel is generated per circuit: each phase's code has its sizes built in.

Exact zeros used (each is exact, see docs/fx/exec-v4.md): an MLP unit whose coef * a * relu(b) is
0 adds nothing (Executor2 adds +-0); a value head whose query is all 0 reads no row and adds
nothing (its read is the zero row, so its output is exactly 0); empty rows are not stored; a
block skips the output dims it never reads.

    ex = Exec4(c, text=T, v0=chip_token_offset)    # c: core.build(...)
    core.snapshot(c, machine); ex.load_init(); ex.reset()
    toks, keep_first, keep_last = ex.run(rows, mode=Exec4.CHIP, max_steps=...)
"""

from __future__ import annotations

import ctypes
import heapq
from collections import defaultdict

import numpy as np
import torch

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, Head

PMAX = 8  # terms per piece
NT = 512  # threads per block
ND = 16  # table words per phase (Exec4._build)


# --- the schedule ---------------------------------------------------------------------------
def node_reads(n) -> set:
    r = set()
    if isinstance(n, FFN):
        for _, a, b in n.terms:
            r |= set(a.terms) | set(b.terms)
        return r
    for chans in (n.q, n.k):
        for ch in chans.values():
            r |= set(ch.expr.terms)
    for vd in n.vheads():
        for ch in vd.v.values():
            r |= set(ch.expr.terms)
        r |= set(vd.a.terms)
        if vd.beta is not None:
            r |= set(vd.beta.terms)
        for e in vd.z.values():
            r |= set(e.terms)
    return r


def node_writes(n) -> set:
    if isinstance(n, FFN):
        return {n.out}
    return {d for vd in n.vheads() for _, d, _ in vd.out}


def schedule(nodes) -> list[int]:
    """Phase (0-based) of every node, in the nodes' level order (see the module doc)."""
    phase: list[int] = []
    writers, readers = defaultdict(list), defaultdict(list)
    for i, n in enumerate(nodes):
        R, W = node_reads(n), node_writes(n)
        p = 0
        for d in R:
            for w in writers[d]:
                if nodes[w].level < n.level:
                    p = max(p, phase[w] + 1)
        for d in W:
            for r in readers[d]:
                if nodes[r].level <= n.level:
                    p = max(p, phase[r])
        phase.append(p)
        for d in R:
            readers[d].append(i)
        for d in W:
            writers[d].append(i)
    return phase


def levels_as_phases(nodes) -> list[int]:
    lv = sorted({n.level for n in nodes})
    return [lv.index(n.level) for n in nodes]


def check_schedule(nodes, phase) -> None:
    """Every node's input in its phase equals its input at its level, on every dim it reads: no
    lower-level writer of such a dim runs in its phase or later, and no writer at its level or
    above runs before its phase."""
    for i, n in enumerate(nodes):
        R = node_reads(n)
        for j, w in enumerate(nodes):
            if not (node_writes(w) & R):
                continue
            if w.level < n.level:
                assert phase[j] < phase[i], (n.name, w.name)
            else:
                assert phase[j] >= phase[i], (n.name, w.name)


# --- the CUDA kernel ------------------------------------------------------------------------
HEAD = r"""
#define D @D@
#define DP @DP@
#define NB @NB@
#define NT @NT@
#define NPH @NPH@
#define HMAX @HMAX@
#define PARTMAX @PARTMAX@
#define BLOBMAX @BLOBMAX@
#define JM @JM@
#define QM @QM@
#define NKEEP @NKEEP@
#define NCODE @NCODE@
#define NFAM @NFAM@
#define V0 @V0@
#define ND 16
#define G (NT / 128)
#define DUMMY ((PARTMAX - 1) << 16)  // the padding slots' piece
#define ZB (PARTMAX - 129)  // 128 piece sums that stay 0

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ long long gtime() {
    long long v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}

// a grid barrier: an arrival counter, monotonic within a launch (the host zeroes it)
__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned& target) {
    __syncthreads();
    target += NB;
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (ld_acquire(bar) < target) { }
        __threadfence();
    }
    __syncthreads();
}

// (value, index) order: the larger value first, the smaller index on ties
__device__ __forceinline__ bool before(float v, int i, float v2, int i2) {
    return v > v2 || (v == v2 && i < i2);
}

// the top QH of a sorted list a (QH entries) and an entry (v, i), in place
template <int QH>
__device__ __forceinline__ void top_insert(float* av, int* ai, float v, int i) {
#pragma unroll
    for (int k = 0; k < QH; k++) {
        if (before(v, i, av[k], ai[k])) {
            float tv = av[k]; int ti = ai[k];
            av[k] = v; ai[k] = i;
            v = tv; i = ti;
        }
    }
}

__device__ __forceinline__ void warp_argmax(const unsigned* vb, int c0, float& v, int& i) {
    unsigned lm = max(max(vb[0], vb[1]), max(vb[2], vb[3]));
    const unsigned m = __reduce_max_sync(0xffffffffu, lm);
    unsigned li = 0xffffffffu;
#pragma unroll
    for (int c = 3; c >= 0; c--)
        if (vb[c] == m) li = c0 + c;
    i = (int)__reduce_min_sync(0xffffffffu, li);
    v = __uint_as_float(m);
}

template <int QH>
__device__ __forceinline__ void warp_top(float* av, int* ai) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float bv[QH]; int bi[QH];
#pragma unroll
        for (int k = 0; k < QH; k++) {
            bv[k] = __shfl_xor_sync(0xffffffffu, av[k], o);
            bi[k] = __shfl_xor_sync(0xffffffffu, ai[k], o);
        }
#pragma unroll
        for (int k = 0; k < QH; k++) top_insert<QH>(av, ai, bv[k], bi[k]);
    }
}

// one value head per warp, 4 channels per lane (Executor2's _head_kernel; see the module doc).
// part[]: the rows' values (a head's q / k / v / z rows are 128 consecutive rows; a negative
// row is the zero block ZB)
template <int QH>
__device__ __forceinline__ void heads(const int* d, const int* blob, const float* part,
                                      float* S, float* Ycur) {
    const int lane = threadIdx.x & 31;
    const int c0 = lane * 4;
    const int nvh = d[4];
    for (int vi = threadIdx.x >> 5; vi < nvh; vi += NT / 32) {
        const int* r = blob + d[5] + 16 * vi;
        const int rq = (r[0] < 0 ? ZB : r[0]) + c0, rk = (r[1] < 0 ? ZB : r[1]) + c0;
        const int rv = (r[2] < 0 ? ZB : r[2]) + c0, rz = (r[3] < 0 ? ZB : r[3]) + c0;
        float q4[4], k4[4], v4[4], z4[4];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            q4[c] = fmaxf(part[rq + c], 0.0f);
            k4[c] = fmaxf(part[rk + c], 0.0f);
            v4[c] = fmaxf(part[rv + c], 0.0f);
            z4[c] = fmaxf(part[rz + c], 0.0f);
        }
        const float bv = part[r[5] < 0 ? ZB : r[5]] + __int_as_float(r[11]);
        float* Sb = S + r[7];
        const int nr = r[6];
        if (nr > 0) {
            const float av = part[r[4] < 0 ? ZB : r[4]];
            const float fac = av == 0.0f ? 1.0f : (av == 110.0f ? 0.5f : 0.0f);
            if (fac != 1.0f)
                for (int rr = 0; rr < nr; rr++) {
                    float4* row = (float4*)(Sb + rr * 128) + lane;
                    float4 x = *row;
                    x.x = x.x * fac; x.y = x.y * fac; x.z = x.z * fac; x.w = x.w * fac;
                    *row = x;
                }
        }
        // the key's row (top 1) and the query's QH first rows: Executor2's argmax rounds (the
        // largest value, the smallest index on ties; a chosen row is zeroed for the next round)
        unsigned kb[4], qb[4];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            kb[c] = __float_as_uint(k4[c]) & 0x7fffffffu;
            qb[c] = __float_as_uint(q4[c]) & 0x7fffffffu;
        }
        float kmv[1]; int kmi[1];
        float qtv[QH]; int qti[QH];
        warp_argmax(kb, c0, kmv[0], kmi[0]);
#pragma unroll
        for (int k = 0; k < QH; k++) {
            warp_argmax(qb, c0, qtv[k], qti[k]);
#pragma unroll
            for (int c = 0; c < 4; c++)
                if (c0 + c == qti[k]) qb[c] = 0u;
        }
        // each lane decays, writes and reads only its own 4 channels, so the read of a row this
        // step wrote is that write (forwarded); a head without keys or decay never writes
        // its state in a launch (read through the read-only path)
        const bool wrote = kmv[0] > 0.0f && bv == 1.0f;
        const float4 vw = make_float4(v4[0], v4[1], v4[2], v4[3]);
        if (wrote) *((float4*)(Sb + kmi[0] * 128) + lane) = vw;
        const int qhot = r[10];
        const bool ro = r[1] < 0 && nr == 0;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < QH; k++) {
            if (k < qhot && qtv[k] > 0.0f) {
                const float4* row = (const float4*)(Sb + qti[k] * 128) + lane;
                const float4 x = (wrote && qti[k] == kmi[0]) ? vw : (ro ? __ldg(row) : *row);
                o.x += x.x; o.y += x.y; o.z += x.z; o.w += x.w;
            }
        }
        const float o0 = __shfl_sync(0xffffffffu, o.x, 0), o1 = __shfl_sync(0xffffffffu, o.y, 0);
        const float ref = o0 * o0 + o1 * o1;
        if (ref > 0.0f) {
            const float norm = __fsqrt_rn(__fdiv_rn(ref, 128.0f));
            const float ov[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const float wv = __int_as_float(blob[r[8] + c0 + c]);
                if (wv != 0.0f) {
                    const float cc = __fdiv_rn(ov[c], norm) * z4[c] * wv;
                    if (cc != 0.0f) atomicAdd(Ycur + blob[r[9] + c0 + c], cc);
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(NT, 1) exec4(
    const int* __restrict__ blob_g, const int* __restrict__ blob_off,
    const int2* __restrict__ slots, float* S, float* Y, float* hist_g,
    const float* __restrict__ forced, const int4* __restrict__ lut,
    const int* __restrict__ emb_dim, const float* __restrict__ emb_val,
    const int* __restrict__ keep, long long* out_tok, float* out_keep, float* dbg_x,
    unsigned* bar, int* status, long long* prof, int nforced, int max_steps, int mode, int dbg,
    int prof_step)
{
    extern __shared__ __align__(16) float sm[];
    float* sx = sm;                        // the residual [DP], then the lag history [HMAX]
    float* part = sm + DP + HMAX;          // piece sums
    int* blob = (int*)(part + PARTMAX);    // this block's tables (see Exec4._build)
    __shared__ int4 s_ctl;

    const int b = blockIdx.x, t = threadIdx.x;
    unsigned target = 0;
    {
        const int o0 = blob_off[b], n = blob_off[b + 1] - o0;
        for (int i = t; i < n; i += NT) blob[i] = blob_g[o0 + i];
    }
    for (int i = t; i < HMAX; i += NT) sx[DP + i] = hist_g[b * HMAX + i];
    for (int i = t; i < DP; i += NT) sx[i] = i < D ? forced[i] : 0.0f;
    for (int i = t; i < PARTMAX; i += NT) part[i] = 0.0f;
    __syncthreads();

    int2 U[JM];  // this phase's term slots, loaded during the previous phase
    {
        const int A = blob[0];
        const int2* sp = slots + blob[1] + t;
#pragma unroll
        for (int j = 0; j < @J0@; j++) U[j] = t < A ? __ldg(sp + j * A) : make_int2(DUMMY, 0);
    }
    int step = 0;
    while (true) {
@PHASES@
        // the step's last outputs, then the output code into the next token (every block)
        const bool prd = step == prof_step && t == 0;
        long long* ppd = prof + (b * NPH + NPH - 1) * 16;
        if (prd) ppd[10] = clock64();
        {
            const int* d = blob + NPH * ND;
            const float* Yp = Y + (NPH - 1) * DP;
            const int* yl = blob + d[12];
            for (int i = t; i < d[11]; i += NT) { const int dm = yl[i]; sx[dm] += __ldcg(Yp + dm); }
            __syncthreads();
        }
        if (t == 0) {
            int fid = 0, p = 0, any = 0;
            for (int i = 0; i < NCODE; i++) {
                const float v = sx[keep[i]];
                any |= v != 0.0f;
                const int bit = v > 0.0f;
                if (i < NFAM) fid |= bit << i; else p |= bit << (i - NFAM);
            }
            s_ctl = any ? __ldg(lut + (fid * 65536 + p)) : make_int4(-1, -1, 0, 0);
        }
        __syncthreads();
        if (prd) ppd[11] = clock64();
        const int4 ctl = s_ctl;  // (model id, embedding row, its entries e0 .. e1)
        if (b == 0) {
            if (t == 0) out_tok[step] = ctl.x;
            if (t < NKEEP) {
                const float v = sx[keep[t]];
                out_keep[NKEEP + t] = v;
                if (step == 0) out_keep[t] = v;
            }
            if (dbg) for (int i = t; i < D; i += NT) dbg_x[(long long)step * D + i] = sx[i];
        }
        step++;
        bool more;
        if (step < nforced) more = true;
        else if (step >= max_steps || ctl.y < 0 || mode == 0) more = false;
        else if (mode == 1) more = ctl.x >= V0;
        else more = true;
        __syncthreads();
        if (!more) break;
        if (step < nforced) {
            for (int i = t; i < D; i += NT) sx[i] = forced[(long long)step * D + i];
        } else {
            for (int i = t; i < D; i += NT) sx[i] = 0.0f;
            __syncthreads();
            for (int e = ctl.z + t; e < ctl.w; e += NT) sx[emb_dim[e]] = emb_val[e];
        }
        __syncthreads();
        if (prd) ppd[12] = clock64();
    }
    for (int i = t; i < HMAX; i += NT) hist_g[b * HMAX + i] = sx[DP + i];
    if (b == 0 && t == 0) status[0] = step;
}
"""

PHASE = r"""
        {   // ---- phase @P@
            const int* d = blob + @P@ * ND;
            float* Ycur = Y + @P@ * DP;
            const bool pr = step == prof_step && t == 0;
            long long* pp = prof + (b * NPH + @P@) * 16;
            if (pr) { pp[0] = gtime(); pp[1] = clock64(); }
@YADD@
            if (pr) pp[2] = clock64();
            {   // phase @ZP@'s buffer: read at the start of phase @ZP1@, next written in phase @ZP@
                float* Yz = Y + @ZP@ * DP;
                const int* zl = blob + d[14];
                for (int i = t; i < d[13]; i += NT) Yz[zl[i]] = 0.0f;
            }
            float* pc = part;  // this phase's piece sums
            {
                float acc = 0.0f;
                int prev = -1;
#pragma unroll
                for (int j = 0; j < @J@; j++) {
                    const int slot = (int)((unsigned)U[j].x >> 16);
                    if (slot != prev) {
                        if (prev >= 0) pc[prev] = acc;
                        acc = 0.0f;
                        prev = slot;
                    }
                    acc += __int_as_float(U[j].y) * sx[U[j].x & 0xffff];
                }
                if (prev >= 0) pc[prev] = acc;
                if (pr) { pp[3] = clock64(); pp[4] = pp[3]; }
            }
            __syncthreads();
            if (pr) pp[5] = clock64();
@FIX@
@LAGS@
            if (pr) pp[8] = clock64();
@UNITS@
            if (pr) pp[9] = clock64();
@HEADS@
            if (pr) { pp[6] = clock64(); pp[7] = gtime(); }
            // the grid barrier; the next phase's term slots load while it waits
            __syncthreads();
            target += NB;
            if (t == 0) {
                __threadfence();
                atomicAdd(bar, 1u);
            }
            {
                const int* dn = blob + @PN@ * ND;
                const int A = dn[0];
                const int2* sp = slots + dn[1] + t;
#pragma unroll
                for (int j = 0; j < @JN@; j++) U[j] = t < A ? __ldg(sp + j * A) : make_int2(DUMMY, 0);
            }
            if (t == 0) {
                while (ld_acquire(bar) < target) { }
                __threadfence();
            }
            __syncthreads();
        }
"""

YADD = r"""
            {   // phase @P1@'s outputs (the dims this block reads) into the residual
                const float* Yp = Y + @P1@ * DP;
                const int* yl = blob + d[12];
                for (int i = t; i < d[11]; i += NT) { const int dm = yl[i]; sx[dm] += __ldcg(Yp + dm); }
                __syncthreads();
            }"""

LAGS = r"""
            {   // the lag history moves on (only the gathers above read it)
                const int* ld = blob + d[7];
                for (int s = t; s < d[6]; s += NT) {
                    float* h = sx + DP + 3 * (d[8] + s);
                    h[2] = h[1];
                    h[1] = h[0];
                    h[0] = sx[ld[s]];
                }
            }"""

FIX = r"""
            {   // a row of several pieces: its first piece's sum + the others, in order; an
                // empty row of a head's 128: 0
                const int2* fx = (const int2*)(blob + d[15]);
                for (int i = t; i < d[9]; i += NT) {
                    const int2 e = fx[i];
                    const int r = e.x & 0xffff, n = e.x >> 16;
                    float x[12];
#pragma unroll
                    for (int k = 0; k < 12; k++) x[k] = k < n ? pc[e.y + k] : 0.0f;
                    float v = n ? pc[r] : 0.0f;
#pragma unroll
                    for (int k = 0; k < 12; k++)
                        if (k < n) v += x[k];
                    for (int k = 12; k < n; k++) v += pc[e.y + k];
                    pc[r] = v;
                }
                __syncthreads();
            }"""

UNITS = r"""
            {   // MLP units: coef * a * relu(b) into the output dim (nothing when it is 0)
                const int* un = blob + d[3];
                const int hw = min(d[4], NT / 32 - 1) * 32;  // the first warps take the heads
                for (int u = t - hw; u < d[2]; u += NT - hw) {
                    if (u < 0) break;
                    const int4 e = *(const int4*)(un + 4 * u);
                    const float a = pc[e.x < 0 ? ZB : e.x];
                    const float bb = pc[e.y < 0 ? ZB : e.y];
                    const float val = __int_as_float(e.w) * a * fmaxf(bb, 0.0f);
                    if (val != 0.0f) atomicAdd(Ycur + e.z, val);
                }
            }"""

HEADS = r"""
            heads<@QH@>(d, blob, pc, S, Ycur);"""


def gen_src(consts: dict, phases: list[dict]) -> str:
    """The kernel's source for one circuit plan: one code block per phase, its sizes built in."""
    nph = len(phases)
    blocks = []
    for p, ph in enumerate(phases):
        nxt = phases[(p + 1) % nph]
        code = PHASE
        code = code.replace("@YADD@", YADD.replace("@P1@", str(p - 1)) if p > 0 else "")
        code = code.replace("@LAGS@", LAGS if ph["lags"] else "")
        code = code.replace("@FIX@", FIX if ph["fix"] else "")
        code = code.replace("@UNITS@", UNITS if ph["units"] else "")
        code = code.replace("@HEADS@", HEADS.replace("@QH@", str(ph["qh"])) if ph["qh"] else "")
        zp = (p - 2) % nph
        for k, v in {"@P@": p, "@ZP@": zp, "@ZP1@": (zp + 1) % nph, "@J@": ph["J"],
                     "@JN@": nxt["J"], "@PN@": (p + 1) % nph}.items():
            code = code.replace(k, str(v))
        blocks.append(code)
    src = HEAD.replace("@PHASES@", "".join(blocks))
    for k, v in consts.items():
        src = src.replace(f"@{k}@", str(v))
    src = src.replace("@J0@", str(phases[0]["J"]))
    return src


class _Kernel:
    """The NVRTC-compiled kernel, launched cooperatively through the driver API."""

    def __init__(self, src: str, smem: int):
        from torch.cuda import _utils as U

        self.k = torch.cuda._compile_kernel(src, "exec4")
        self.k.set_shared_memory_config(smem)
        self.smem = smem
        self.lib = U._get_gpu_runtime_library()

    def occupancy(self, nt: int) -> int:
        n = ctypes.c_int(0)
        f = self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
        assert f(ctypes.byref(n), self.k.func, nt, self.smem) == 0
        return n.value

    def args(self, args):
        """A launch's argument array (kept: pass it to launch_args with new values set)."""
        refs, c_args = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = ctypes.c_void_p(a.data_ptr())
            elif isinstance(a, int):
                v = ctypes.c_int(a)
            else:
                raise TypeError(type(a))
            refs.append(v)
            c_args.append(ctypes.cast(ctypes.byref(v), ctypes.c_void_p))
        return refs, (ctypes.c_void_p * len(c_args))(*c_args)

    def launch_args(self, grid: int, block: int, arr, stream=None):
        stream = stream or torch.cuda.current_stream()
        err = self.lib.cuLaunchCooperativeKernel(self.k.func, grid, 1, 1, block, 1, 1, self.smem,
                                                 ctypes.c_void_p(stream.cuda_stream), arr)
        if err != 0:
            raise RuntimeError(f"cuLaunchCooperativeKernel failed: {err}")

    def launch(self, grid: int, block: int, args, stream=None):
        refs, c_args = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = ctypes.c_void_p(a.data_ptr())
            elif isinstance(a, int):
                v = ctypes.c_int(a)
            else:
                raise TypeError(type(a))
            refs.append(v)
            c_args.append(ctypes.cast(ctypes.byref(v), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(c_args))(*c_args)
        stream = stream or torch.cuda.current_stream()
        err = self.lib.cuLaunchCooperativeKernel(self.k.func, grid, 1, 1, block, 1, 1, self.smem,
                                                 ctypes.c_void_p(stream.cuda_stream), arr)
        if err != 0:
            raise RuntimeError(f"cuLaunchCooperativeKernel failed: {err}")


class Exec4:
    """The circuit ``c`` as one persistent kernel (see the module doc). One conversation row."""

    FORCED, CHIP, OWN = 0, 1, 2  # run modes: the forced rows only; then chip tokens; then any

    def __init__(self, c, device="cuda", nb: int | None = None, text=None, v0: int = 0,
                 tables=None, extra_layers: int = 0, flatten: bool = True):
        from chipformer.rv import core

        self.c, self.dev = c, device
        D = len(c.dims)
        self.D, self.DP = D, (D + 3) // 4 * 4
        assert self.DP < 1 << 14
        props = torch.cuda.get_device_properties(device)
        self.NB = nb or props.multi_processor_count
        nodes = sorted(c.nodes, key=lambda n: n.level)  # stable: the levels' own order
        if extra_layers:
            nodes = _with_extra_layers(c, nodes, extra_layers)
        self.nodes = nodes
        # flatten=False: one phase per level (a sublayer each, as Executor2 runs them)
        self.phase = schedule(nodes) if flatten else levels_as_phases(nodes)
        check_schedule(nodes, self.phase)
        self.NPH = max(self.phase) + 1
        # Executor2's per-level QHOT (the max over a level's heads applies to all of them)
        qhot_lv = defaultdict(lambda: 0)
        for n in nodes:
            if isinstance(n, Head):
                qhot_lv[n.level] = max(qhot_lv[n.level], getattr(n, "qhot", 2))
        self.keep_ir = list(c.keep)
        self.NKEEP = len(self.keep_ir)
        self.NCODE, self.NFAM = core.NFAM + core.NPAY, core.NFAM
        assert self.NCODE <= self.NKEEP <= NT
        self._build(nodes, qhot_lv)
        self.v0 = v0
        self.text = text
        self._tables(tables)
        self.JM = max(1, max(ph["J"] for ph in self.phases))
        QM = max(1, max(ph["qh"] for ph in self.phases))
        consts = {"D": D, "DP": self.DP, "NB": self.NB, "NT": NT, "NPH": self.NPH,
                  "HMAX": self.HMAX, "PARTMAX": self.PARTMAX, "BLOBMAX": self.BLOBMAX,
                  "JM": self.JM, "QM": QM, "NKEEP": self.NKEEP, "NCODE": self.NCODE,
                  "NFAM": self.NFAM, "V0": v0}
        self.src = gen_src(consts, self.phases)
        G = NT // 128
        self.smem = 4 * (self.DP + self.HMAX + self.PARTMAX + self.BLOBMAX)
        self.kern = _Kernel(self.src, self.smem)
        occ = self.kern.occupancy(NT)
        assert occ * props.multi_processor_count >= self.NB, (occ, self.NB)
        self.keep_t = torch.tensor(self.keep_ir, dtype=torch.int32, device=device)
        self._alloc_state()

    def _alloc_state(self):
        """A conversation row's own buffers (the plan, tables and kernel are shared)."""
        dz = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device=self.dev)
        self.S = dz(max(len(self.vheads), 1) * HD * HD)
        self.Ybar = dz(self.NPH * self.DP + 4)  # the outputs' buffers, then the counter
        self.Y = self.Ybar[: self.NPH * self.DP]
        self.bar = self.Ybar[self.NPH * self.DP:].view(torch.int32)
        self.hist = dz(self.NB * self.HMAX)
        self.status = dz(1, torch.int32)
        self.out_keep = dz(2 * self.NKEEP)
        self._out_tok = dz(1, torch.int64)
        self._dbg = dz(1)
        self.prof = dz(self.NB * self.NPH * 16, torch.int64)
        self.tok_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)

    def fork(self) -> "Exec4":
        """Another row: the same plan and kernel, its own state."""
        import copy

        other = copy.copy(self)
        other._alloc_state()
        other._args_key = None
        return other

    # --- the per-phase, per-block work ------------------------------------------------------
    def _build(self, nodes, qhot_lv):
        """Per block, a table blob (copied to shared memory at launch):

          [phase p < NPH: ND words]  J (unused), 0, n units, units at, n vheads, vheads at,
                               n lag dims, lag dims at, first lag slot, rows at, the most QHOT,
                               n Y dims, Y dims at, n zero dims, zero dims at, 0
          [NPH: ND words]      the last phase's Y dims (fields 11, 12)
          rows                 pstart | pcount << 16 (the row's pieces in the piece sums)
          units                (a row, b row, out dim, coef bits)
          value heads          (q row 0, k row 0, v row 0, z row 0, a row, b row, decay rows,
                               state offset, output weights at, output dims at, QHOT, bconst
                               bits, 0..) ; -1: an empty row (block)
          output weights / dims  128 each per value head
          lag dims, Y dims (the previous phase's outputs this block reads), zero dims

        and in global memory the term slots: per phase p, NB x J_p x NT int2 (slot j of thread
        t of block b at sb_p + (b * J_p + j) * NT + t): (smem index | piece << 16, weight)."""
        D, NB, DP, NPH = self.D, self.NB, self.DP, self.NPH
        f32 = np.float32
        fbits = lambda x: int(np.float32(x).view(np.int32))

        def lagged(ch):
            return {lag * D + d: float(cf) * tw for lag, tw in enumerate(ch.taps) if tw
                    for d, cf in ch.expr.terms.items()}

        def flat(e):
            return {d: float(cf) for d, cf in e.terms.items()}

        self.vheads = []  # (node index, value head) per state block
        groups = defaultdict(list)  # phase -> [(cost, kind, payload)]
        writes = [set() for _ in range(NPH)]  # the dims each phase writes
        for i, n in enumerate(nodes):
            p = self.phase[i]
            writes[p] |= node_writes(n)
            if isinstance(n, FFN):
                for cf, a, b in n.terms:
                    ra, rb = flat(a), flat(b)
                    groups[p].append((len(ra) + len(rb) + 2, "u",
                                      {"a": ra, "b": rb, "coef": float(cf), "out": n.out}))
                continue
            qrows = [lagged(n.q[s]) if s in n.q else {} for s in range(HD)]
            krows = [lagged(n.k[s]) if s in n.k else {} for s in range(HD)]
            reach = set(n.k) | {r for x in (n.init or []) for r in x}
            vhs = []
            for x, vd in enumerate(n.vheads()):
                od, w_ = np.zeros(HD, np.int32), np.zeros(HD, f32)
                for col, dd, w in vd.out:
                    od[col], w_[col] = dd, w
                vg = len(self.vheads)
                self.vheads.append((i, x))
                vhs.append({"v": [lagged(vd.v[col]) if col in vd.v else {} for col in range(HD)],
                            "z": [flat(vd.z[col]) if col in vd.z else {} for col in range(HD)],
                            "a": flat(vd.a), "b": {} if vd.beta is None else flat(vd.beta),
                            "bconst": 1.0 if vd.beta is None else 0.0,
                            "nrows": max(reach) + 1 if vd.a.terms and reach else 0,
                            "sofs": vg * HD * HD, "qhot": qhot_lv[n.level], "od": od, "ow": w_})
            cost = sum(len(r) for r in qrows + krows) + sum(
                sum(len(r) for r in vh["v"] + vh["z"]) + len(vh["a"]) + len(vh["b"]) + 400
                for vh in vhs)
            groups[p].append((cost, "h", {"q": qrows, "k": krows, "vhs": vhs}))
        self.writes = writes

        blobs = [[0] * (ND * (NPH + 1)) for _ in range(NB)]
        need = [set(self.keep_ir) for _ in range(NB)]  # the dims each block reads
        need[0] = set(range(D))  # block 0 also writes the whole residual (dbg)
        bp_pieces = {}  # (phase, block) -> (bins of pieces, J)
        hist_slots = [0] * NB
        self.PARTMAX = 0
        self.phase_stats = []
        self.phases = []
        for p in range(NPH):
            # assign: head groups first (fewest value heads, then least work), units by LPT
            items = groups[p]
            blk = [[] for _ in range(NB)]
            load, nheads = [0] * NB, [0] * NB
            for cost, kind, it in sorted((x for x in items if x[1] == "h"), key=lambda x: -x[0]):
                bi = min(range(NB), key=lambda k: (nheads[k], load[k]))
                blk[bi].append((kind, it))
                load[bi] += cost
                nheads[bi] += len(it["vhs"])
            heap = [(load[k], k) for k in range(NB)]
            heapq.heapify(heap)
            for cost, kind, it in sorted((x for x in items if x[1] == "u"), key=lambda x: -x[0]):
                ld_, bi = heapq.heappop(heap)
                blk[bi].append((kind, it))
                heapq.heappush(heap, (ld_ + cost, bi))
            stat = {"phase": p, "items": len(items), "terms": 0, "J": 0, "heads": 0,
                    "writes": len(writes[p])}
            has_units = has_lags = has_fix = False
            qh = 0
            for bi in range(NB):
                rows: list[dict] = []

                def add_row(r):
                    if not r:
                        return -1
                    rows.append(r)
                    return len(rows) - 1

                def add_rows128(rs):
                    if not any(rs):
                        return -1
                    first = len(rows)
                    rows.extend(rs)
                    return first

                units, vhr = [], []
                for kind, it in blk[bi]:
                    if kind == "u":
                        units.append((add_row(it["a"]), add_row(it["b"]), it["out"], it["coef"]))
                    else:
                        q0, k0 = add_rows128(it["q"]), add_rows128(it["k"])
                        for vh in it["vhs"]:
                            vhr.append((q0, k0, add_rows128(vh["v"]), add_rows128(vh["z"]),
                                        add_row(vh["a"]), add_row(vh["b"]), vh))
                for r in rows:
                    need[bi] |= {key % D for key in r}
                lags = sorted({key % D for r in rows for key in r if key >= D})
                base = hist_slots[bi]
                slot = {dd: base + s for s, dd in enumerate(lags)}
                hist_slots[bi] += len(lags)

                def smem_idx(key):
                    lag, dd = divmod(key, D)
                    return dd if lag == 0 else DP + 3 * slot[dd] + lag - 1

                # row r's value is piece sum r: its first piece's; its further pieces sum into
                # slots after the rows and a fix-up adds them; an empty row is zeroed there
                nr_ = len(rows)
                pieces, fix = [], []
                nextra = 0
                for r_i, r in enumerate(rows):
                    terms = [(smem_idx(k), np.float32(v)) for k, v in r.items() if v]
                    chunks = [terms[a:a + PMAX] for a in range(0, len(terms), PMAX)]
                    if not chunks:
                        fix.append((r_i, 0, 0))
                        continue
                    pieces.append((r_i, chunks[0]))
                    if len(chunks) > 1:
                        fix.append((r_i, len(chunks) - 1, nr_ + nextra))
                        for ch_ in chunks[1:]:
                            pieces.append((nr_ + nextra, ch_))
                            nextra += 1
                npart = nr_ + nextra
                self.PARTMAX = max(self.PARTMAX, npart + 129)
                assert npart + 129 < 1 << 15
                J = max([len(x) for _, x in pieces], default=0)
                bp_pieces[p, bi] = pieces
                # this block-phase's tables
                bl = blobs[bi]
                bl += [0] * (-len(bl) % 4)  # units are read as int4
                units_at = len(bl)
                for ra, rb, out, coef in units:
                    bl += [ra, rb, out, fbits(coef)]
                vh_at = len(bl)
                bl += [0] * (16 * len(vhr))
                for x, (q0, k0, v0_, z0, ar, br, vh) in enumerate(vhr):
                    ow_at = len(bl)
                    bl += [fbits(w) for w in vh["ow"]]
                    od_at = len(bl)
                    bl += [int(dd) for dd in vh["od"]]
                    bl[vh_at + 16 * x: vh_at + 16 * x + 12] = [
                        q0, k0, v0_, z0, ar, br, vh["nrows"], vh["sofs"], ow_at, od_at,
                        vh["qhot"], fbits(vh["bconst"])]
                bl += [0] * (-len(bl) % 2)  # fix-ups are read as int2
                fix_at = len(bl)
                for r_i, n_, e0 in fix:
                    bl += [r_i | (n_ << 16), e0]
                lag_at = len(bl)
                bl += lags
                qhb = max([vh["qhot"] for *_, vh in vhr], default=0)
                bl[ND * p + 2: ND * p + 11] = [len(units), units_at, len(vhr), vh_at,
                                               len(lags), lag_at, base, len(fix), qhb]
                bl[ND * p + 15] = fix_at
                has_units |= bool(units)
                has_lags |= bool(lags)
                has_fix |= bool(fix)
                qh = max(qh, qhb)
                stat["terms"] += sum(len(x) for _, x in pieces)
                stat["J"] = max(stat["J"], J)
                stat["heads"] += len(vhr)
            self.phase_stats.append(stat)
            self.phases.append({"J": stat["J"], "units": has_units, "lags": has_lags, "qh": qh,
                                "fix": has_fix})
        self.PARTMAX += -self.PARTMAX % 4  # (the last piece sum is the padding's, the 128
        # before it stay 0)
        # Y dims: at phase p's start, phase p - 1's writes that the block reads (entry NPH: the
        # last phase's, read before the decode); zero dims: phase (p - 2)'s writes, split
        for bi in range(NB):
            bl = blobs[bi]
            for p in range(1, NPH + 1):
                yl = sorted(writes[p - 1] & need[bi])
                bl[ND * p + 11: ND * p + 13] = [len(yl), len(bl)]
                bl += yl
            for p in range(NPH):
                zw = sorted(writes[(p - 2) % NPH])
                per = -(-len(zw) // NB)
                zl = zw[bi * per:(bi + 1) * per]
                bl[ND * p + 13: ND * p + 15] = [len(zl), len(bl)]
                bl += zl
        self.need = need
        # the term slots: per block-phase the pieces go first-fit-decreasing onto as few threads
        # as hold them at the phase's J terms each (J: its longest piece, or more if a block
        # has more than NT * J terms); thread t's slot j at base + j * A + t (A threads)
        T = []
        sb = 0
        for p in range(NPH):
            J = max(max([len(x) for _, x in bp_pieces[p, bi]], default=0) for bi in range(NB))
            J = max(J, max(-(-sum(len(x) for _, x in bp_pieces[p, bi]) // NT)
                           for bi in range(NB)))
            while True:
                packs = []
                for bi in range(NB):
                    pieces = bp_pieces[p, bi]
                    bins = []
                    by_room = defaultdict(list)  # room left -> open bins (best fit, smallest)
                    for pi in sorted(range(len(pieces)), key=lambda x: -len(pieces[x][1])):
                        L = len(pieces[pi][1])
                        r = next((r for r in range(L, J + 1) if by_room[r]), None)
                        if r is None:
                            bins.append([])
                            k, r = len(bins) - 1, J
                        else:
                            k = by_room[r].pop()
                        bins[k].append(pi)
                        if r - L > 0:
                            by_room[r - L].append(k)
                    packs.append(bins)
                if max(len(x) for x in packs) <= NT:
                    break
                J += 1
            self.phases[p]["J"] = J
            self.phase_stats[p]["J"] = J
            for bi in range(NB):
                pieces, bins = bp_pieces[p, bi], packs[bi]
                A = len(bins)
                arr = np.zeros((J, A, 2), np.int32)
                arr[:, :, 0] = (self.PARTMAX - 1) << 16  # padding: the dummy piece
                for k, bn in enumerate(bins):
                    j = 0
                    for pi in sorted(bn):
                        home, terms = pieces[pi]
                        for idx, wv in terms:
                            assert idx < 1 << 16
                            arr[j, k, 0] = np.uint32(idx | (home << 16)).view(np.int32)
                            arr[j, k, 1] = np.float32(wv).view(np.int32)
                            j += 1
                blobs[bi][ND * p: ND * p + 2] = [A, sb]
                T.append(arr.reshape(-1, 2))
                sb += J * A
        self.HMAX = 3 * max(max(hist_slots), 1)
        self.HMAX += -self.HMAX % 4
        for bl in blobs:
            bl += [0] * (-len(bl) % 4)
        self.BLOBMAX = max(len(bl) for bl in blobs)
        self.n_slots = sb
        off = np.concatenate([[0], np.cumsum([len(bl) for bl in blobs])]).astype(np.int32)
        dv = self.dev
        self.blob = torch.tensor(np.concatenate([np.asarray(bl, np.int64) for bl in blobs])
                                 .astype(np.int32), device=dv)
        self.blob_off = torch.tensor(off, device=dv)
        self.slots = torch.tensor(np.concatenate(T) if T else np.zeros((1, 2), np.int32),
                                  device=dv).contiguous()

    # --- decode and embedding tables ----------------------------------------------------------
    def _tables(self, tables):
        """lut[fid << 16 | p] = (model id, embedding row, its entries e0, e1): core.decode_code's
        token as a model id (chip v0 + t, text the Qwen id; -2: none) and its embedding row
        (chip tokens, then the printable text tokens and the newline the circuit echoes; -1:
        none). emb: the rows of Executor2.embed, sparse (every entry that is not +0.0)."""
        if tables is None:
            tables = build_tables(self.c, self.text, self.v0)
        self.tables = tables
        lut_id, lut_row, off, dims, vals = tables
        e0 = np.where(lut_row >= 0, off[np.maximum(lut_row, 0)], 0)
        e1 = np.where(lut_row >= 0, off[np.maximum(lut_row, 0) + 1], 0)
        lut = np.stack([lut_id, lut_row, e0, e1], 1).astype(np.int32)
        t = lambda a: torch.tensor(np.ascontiguousarray(a), device=self.dev)
        self.lut = t(lut.reshape(-1))
        self.emb_dim, self.emb_val = t(dims), t(vals)

    # --- state ------------------------------------------------------------------------------
    def load_init(self):
        """Every value head's initial state (the snapshot now in the graph), as Executor2."""
        buf = np.zeros(self.S.numel(), np.float32)
        for vg, (i, x) in enumerate(self.vheads):
            init = self.nodes[i].init or []
            if x < len(init):
                base = vg * HD * HD
                for r, vec in init[x].items():
                    buf[base + r * HD: base + (r + 1) * HD] = vec
        self.S.copy_(torch.from_numpy(buf))

    def zero_state(self):
        self.S.zero_()

    def reset(self):
        """Fresh lag history."""
        self.hist.zero_()

    def embed(self, toks):
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in sorted(self.c.token_fns):
            out[:, d] = [float(self.c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    # --- a launch ---------------------------------------------------------------------------
    def launch(self, rows: torch.Tensor, mode: int = 1, max_steps: int = 1, dbg: bool = False,
               prof_step: int = -1):
        """Enqueue a launch (see run); returns nothing."""
        n = rows.shape[0]
        max_steps = max(max_steps, n)
        assert rows.is_contiguous() and rows.shape[1] == self.D and rows.dtype == torch.float32
        if self._out_tok.numel() < max_steps:
            self._out_tok = torch.zeros(max_steps, dtype=torch.int64, device=self.dev)
        if dbg and self._dbg.numel() < max_steps * self.D:
            self._dbg = torch.zeros(max_steps * self.D, device=self.dev)
        self.Ybar.zero_()
        key = (self._out_tok.data_ptr(), self._dbg.data_ptr())
        if getattr(self, "_args_key", None) != key:
            self._args = self.kern.args([
                self.blob, self.blob_off, self.slots, self.S, self.Y, self.hist, rows, self.lut,
                self.emb_dim, self.emb_val, self.keep_t, self._out_tok, self.out_keep, self._dbg,
                self.bar, self.status, self.prof, n, max_steps, mode, int(dbg), prof_step])
            self._args_key = key
        refs, arr = self._args
        refs[6].value = rows.data_ptr()
        refs[17].value, refs[18].value, refs[19].value = n, max_steps, mode
        refs[20].value, refs[21].value = int(dbg), prof_step
        self.kern.launch_args(self.NB, NT, arr)

    def run(self, rows: torch.Tensor, mode: int = 1, max_steps: int = 1, dbg: bool = False,
            prof_step: int = -1, keep: bool = True):
        """Steps from the forced input rows [n, D] (embeddings), then (mode CHIP) while the
        predicted token is a chip token, or (mode OWN) any token the circuit can embed, up to
        max_steps steps in all. Returns (predicted model ids per step [-1: no code, -2: no
        token], keep after the first step, keep after the last step); with dbg, also the
        residual after every step."""
        self.launch(rows, mode, max_steps, dbg, prof_step)
        k = int(self.status.item())
        toks = self._out_tok[:k].cpu().numpy()
        if keep is False:
            return toks
        kf, kl = self.out_keep[:self.NKEEP].clone(), self.out_keep[self.NKEEP:].clone()
        if dbg:
            return toks, kf, kl, self._dbg[:k * self.D].view(k, self.D).double().cpu().numpy()
        return toks, kf, kl


def build_tables(c, text, v0: int):
    """See Exec4._tables. The embedding rows are Executor2.embed's, stored sparse; the decode
    table is core.decode_code over every (family id, payload) code."""
    from chipformer.rv import bridge as BR
    from chipformer.rv import core
    from chipformer.rv import spec as S

    nf = 1 << core.NFAM
    lut_id = np.full(nf << 16, -2, np.int32)
    lut_row = np.full(nf << 16, -1, np.int32)
    echo = sorted(set(text.printable) | {text.NL}) if text is not None else []
    echo_row = {q: S.VOCAB + i for i, q in enumerate(echo)}
    for fid in range(nf):
        for p in range(1 << 16):
            t = core.decode_code(fid, p, text)
            if t < 0:
                continue
            if S.is_text(t):
                qid = t - S.TEXT_TOKEN
                lut_id[fid << 16 | p] = qid
                lut_row[fid << 16 | p] = echo_row.get(qid, -1)
            else:
                lut_id[fid << 16 | p] = v0 + t
                lut_row[fid << 16 | p] = t
    toks = list(range(S.VOCAB)) + [BR.text_token(q) for q in echo]
    feat = sorted(c.token_fns)
    D = len(c.dims)
    cols = {}
    for d in feat:
        cols[d] = np.array([float(c.feature_value(d, t)) for t in toks], np.float32)
    off = np.zeros(len(toks) + 1, np.int64)
    ent_d, ent_v = [], []
    # ONE is 1.0 in every row (Executor2.embed); a feature dim may overwrite it
    per_tok_d = [[] for _ in toks]
    per_tok_v = [[] for _ in toks]
    for t in range(len(toks)):
        if ONE not in cols:
            per_tok_d[t].append(ONE)
            per_tok_v[t].append(np.float32(1.0))
    for d in feat:
        col = cols[d]
        nz = np.nonzero((col != 0) | np.signbit(col))[0]
        for t in nz:
            per_tok_d[t].append(d)
            per_tok_v[t].append(col[t])
    for t in range(len(toks)):
        off[t + 1] = off[t] + len(per_tok_d[t])
        ent_d += per_tok_d[t]
        ent_v += per_tok_v[t]
    assert off[-1] < 2**31 and D < 2**31
    return (lut_id, lut_row, off.astype(np.int32), np.asarray(ent_d, np.int32),
            np.asarray(ent_v, np.float32))


def _with_extra_layers(c, nodes, n_layers):
    """For the scaling measurement only: n_layers further RAM layers, each a copy of the 16 heads
    of the first RAM layer (the same weights, so the same queries, keys and row reads; their
    outputs go nowhere), at its own new level just above that layer, as further word-slot
    layers of a bigger RAM would sit."""
    import copy

    ram = sorted((n for n in nodes if isinstance(n, Head) and n.name.startswith("ram")),
                 key=lambda n: n.level)
    first = [n for n in ram if n.level == ram[0].level]
    extra = []
    for k in range(n_layers):
        for src in first:
            h = copy.copy(src)
            h.name = f"x{k}.{src.name}"
            h.level = src.level + (k + 1) / (n_layers + 1)  # between it and the next level
            h.out = []
            h.vx = [copy.copy(v) for v in src.vx]
            for v in h.vx:
                v.out = []
            extra.append(h)
    return sorted(nodes + extra, key=lambda n: n.level)


def cached_tables(c, text, v0: int):
    """build_tables, cached in ~/.cache/chipformer under a hash of everything they depend on
    (the circuit and token sources, the tokenizer's printable tokens, v0)."""
    import hashlib
    from pathlib import Path

    here = Path(__file__).parent
    h = hashlib.sha1()
    import inspect

    for f in ("core.py", "spec.py", "bridge.py"):
        h.update((here / f).read_bytes())
    h.update(inspect.getsource(build_tables).encode())
    h.update(repr((v0, len(c.dims), list(text.printable) if text else None,
                   text.NL if text else None)).encode())
    path = Path.home() / ".cache" / "chipformer" / f"exec4-tables-{h.hexdigest()[:16]}.npz"
    if path.exists():
        z = np.load(path)
        return tuple(z[k] for k in ("lut_id", "lut_row", "off", "dims", "vals"))
    t = build_tables(c, text, v0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, lut_id=t[0], lut_row=t[1], off=t[2], dims=t[3], vals=t[4])
    return t
