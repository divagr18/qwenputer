"""The attention kernels a KV-RAM read can meet, on CPU with real dtypes.

q, k, v: bf16 [B, H, L, E] (one kv head per q head here), mask: bool [L, N] (True = attend) or None.
- eager: transformers' eager_attention_forward (bf16 matmul -> bf16 scores, fp32 softmax, bf16 P).
- sdpa_math: torch SDPBackend.MATH (upcasts bf16 to fp32 internally in torch >= 2.5).
- sdpa_flash_cpu: torch SDPBackend.FLASH_ATTENTION on CPU (blocked online softmax, fp32 scores).
- fa2: a replica of FlashAttention-2's forward numerics (GPU kernel, not runnable here):
  S = QK^T fp32; per key block of Bc: m = running max; P = exp2(fma(S, scale*log2e, -m*scale*log2e))
  (flush-to-zero below 2^-126); l = l*exp2((m_old-m)*c) + rowsum(P); acc = acc*exp2(...) + bf16(P) @ V;
  O = bf16(acc / l).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SCALE = 1 / 16


def eager(q, k, v, mask=None, scale=SCALE):
    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

    class M:
        num_key_value_groups = 1
        training = False
    am = None
    if mask is not None:
        am = torch.zeros(mask.shape, dtype=q.dtype).masked_fill(~mask, torch.finfo(q.dtype).min)
    o, _ = eager_attention_forward(M, q, k, v, am, scaling=scale)
    return o.transpose(1, 2)  # back to [B, H, L, E]


def sdpa(q, k, v, mask=None, backend="math", scale=SCALE):
    b = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}[backend]
    with sdpa_kernel([b]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def scores_fp32(q, k, scale=SCALE, chunk=1 << 18):
    """q.k * scale exactly as the math backend forms it (bf16 -> fp32, q and k each times sqrt(scale)),
    chunked over keys. q: [L, E], k: [N, E] -> fp32 [L, N]."""
    s = math.sqrt(scale)
    qf = q.float() * s
    out = torch.empty(q.shape[0], k.shape[0], dtype=torch.float32)
    for i in range(0, k.shape[0], chunk):
        out[:, i:i + chunk] = qf @ (k[i:i + chunk].float() * s).T
    return out


def _round32(x64):
    return x64.to(torch.float32)


def _exp2_ftz(x32):
    y = torch.exp2(x32.double()).float()
    return torch.where(y < 2.0 ** -126, torch.zeros_like(y), y)


def fa2(q, k, v, mask=None, scale=SCALE, Bc=128, chunk=1 << 16):
    """FlashAttention-2 forward numerics for one head. q [L, E], k [N, E], v [N, Ev] bf16,
    mask bool [L, N]. Returns bf16 [L, Ev] and the fp32 scores' row max for diagnostics."""
    L, N = q.shape[0], k.shape[0]
    c = torch.tensor(scale * math.log2(math.e), dtype=torch.float32)  # softmax_scale_log2, rounded to fp32
    m = torch.full((L,), -math.inf, dtype=torch.float32)
    l = torch.zeros(L, dtype=torch.float32)
    acc = torch.zeros(L, v.shape[1], dtype=torch.float32)
    qf = q.float()
    for c0 in range(0, N, chunk):
        S = qf @ k[c0:c0 + chunk].float().T  # fp32 accumulate (exact in the designed regime)
        if mask is not None:
            S = S.masked_fill(~mask[:, c0:c0 + chunk], -math.inf)
        nb = (S.shape[1] + Bc - 1) // Bc
        pad = nb * Bc - S.shape[1]
        if pad:
            S = torch.cat([S, torch.full((L, pad), -math.inf)], 1)
        Sb = S.view(L, nb, Bc)
        bmax = Sb.amax(2)  # [L, nb]
        run = torch.cummax(torch.cat([m[:, None], bmax], 1), 1).values[:, 1:]  # running max after each block
        vv = v[c0:c0 + chunk].float()
        if pad:
            vv = torch.cat([vv, torch.zeros(pad, vv.shape[1])], 0)
        vb = vv.view(nb, Bc, -1)
        for i in range(L):
            # only blocks that change the max or carry nonzero P matter; P = 0 elsewhere (ftz)
            mi = run[i]
            ms = (mi * c)  # fp32 product, rounded
            # P for all blocks: fma(S, c, -ms) = S*c - ms in exact arithmetic, rounded once
            arg = _round32(Sb[i].double() * c.double() - ms.double()[:, None])
            P = _exp2_ftz(arg)
            P = torch.where(torch.isinf(Sb[i]), torch.zeros_like(P), P)
            live = (P.sum(1) > 0) | (torch.diff(torch.cat([m[i:i + 1], mi])) != 0)
            prev = m[i]
            for j in torch.nonzero(live).flatten().tolist():
                newm = mi[j]
                if torch.isinf(prev):
                    sc = torch.tensor(0.0)
                else:
                    d = (prev - newm).float()  # fp32 subtract, then fp32 multiply (FA2 softmax_rescale_o)
                    sc = _exp2_ftz(d * c)
                l[i] = l[i] * sc + P[j].sum()
                acc[i] = acc[i] * sc + P[j].to(torch.bfloat16).float() @ vb[j]
                prev = newm
            m[i] = mi[-1] if not torch.isinf(mi[-1]) else m[i]
    inv = torch.where(l == 0, torch.ones_like(l), 1.0 / l)  # FA2: acc_o *= 1/sum
    out = (acc * inv[:, None]).to(torch.bfloat16)
    return out


def math_replica(q, k, v, mask=None, scale=SCALE, chunk=1 << 18):
    """torch's MATH backend for bf16 inputs (upcast to fp32, q and k times sqrt(scale), softmax in
    fp32, P @ V in fp32, cast to bf16), chunked over keys so a 1.5M-entry log fits in memory.
    Checked bit-identical to sdpa(..., 'math') on smaller logs (study.py check_replica)."""
    S = scores_fp32(q, k, scale, chunk)
    if mask is not None:
        S.masked_fill_(~mask, -math.inf)
    S = torch.softmax(S, -1)
    out = torch.zeros(q.shape[0], v.shape[1], dtype=torch.float32)
    for i in range(0, k.shape[0], chunk):
        out += S[:, i:i + chunk] @ v[i:i + chunk].float()
    return out.to(torch.bfloat16)
