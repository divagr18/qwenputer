"""int4 weights for stories260K by quantization-aware self-distillation.

Teacher: the float model. Data: its own samples (no dataset or tokenizer needed). Student: int4
per-row fake-quant (learnable scales, straight-through rounding) on every matmul incl. an untied
classifier, int8 per-vector fake-quant on matmul inputs (as the C code). Loss: KL(teacher || student).

python qat.py [steps]  ->  qat_int4.npz (int4 codes + row scales), eval printed
"""
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
raw = open(f"{HERE}/stories260K.bin", "rb").read()
dim, hid, nl, nh, nkv, vocab, seqlen = struct.unpack("7i", raw[:28])
hs, kvd = dim // nh, dim // nh * nkv
f = np.frombuffer(raw[28:], dtype=np.float32).copy()
o = 0


def take(*shape):
    global o
    n = int(np.prod(shape))
    a = torch.tensor(f[o:o + n].reshape(shape))
    o += n
    return a


P0 = {"emb": take(vocab, dim), "ga": take(nl, dim), "wq": take(nl, dim, dim), "wk": take(nl, kvd, dim),
      "wv": take(nl, kvd, dim), "wo": take(nl, dim, dim), "gf": take(nl, dim), "w1": take(nl, hid, dim),
      "w2": take(nl, dim, hid), "w3": take(nl, hid, dim), "gfin": take(dim)}
# fold the RMSNorm gains into the matrices that read them (the deployed form: the C code's norm
# is then a single 1/rms in the output scale); the untied classifier gets the final gain
P0["cls"] = P0["emb"] * P0["gfin"]
for k, g in (("wq", "ga"), ("wk", "ga"), ("wv", "ga"), ("w1", "gf"), ("w3", "gf")):
    P0[k] = P0[k] * P0[g][:, None, :]
P0["ga"], P0["gf"], P0["gfin"] = torch.ones(nl, dim), torch.ones(nl, dim), torch.ones(dim)
MATS = ("cls", "wq", "wk", "wv", "wo", "w1", "w2", "w3")
L = 256
pos = torch.arange(L)
fr = 1.0 / 10000 ** (torch.arange(0, hs, 2).float() / hs)
ang = pos[:, None] * fr[None]
COS, SIN = torch.cos(ang), torch.sin(ang)


def rope(x):  # x (..., T, nheads, hs)
    T = x.shape[-3]
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = COS[:T, None], SIN[:T, None]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def fq_w(w, s):  # int4 per row, s (rows, 1)
    return torch.clamp(ste_round(w / s), -8, 7) * s


def fq_a(x):  # int8 per vector (last dim)
    a = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return torch.clamp(ste_round(x / a), -127, 127) * a


def forward(P, toks, S=None):
    """S: dict of row scales -> quantized student; None -> float teacher."""
    def W(k, l=None):
        w = P[k] if l is None else P[k][l]
        if S is None:
            return w
        return fq_w(w, S[k] if l is None else S[k][l])

    def lin(x, k, l=None):
        return (fq_a(x) if S is not None else x) @ W(k, l).T

    B, T = toks.shape
    x = P["emb"][toks]
    mask = torch.full((T, T), float("-inf")).triu(1)
    for l in range(nl):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["ga"][l]
        q = rope(lin(h, "wq", l).view(B, T, nh, hs))
        k = rope(lin(h, "wk", l).view(B, T, nkv, hs))
        v = lin(h, "wv", l).view(B, T, nkv, hs)
        k, v = k.repeat_interleave(nh // nkv, 2), v.repeat_interleave(nh // nkv, 2)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k) / hs ** 0.5 + mask
        a = torch.einsum("bhqk,bkhd->bqhd", att.softmax(-1), v).reshape(B, T, dim)
        x = x + lin(a, "wo", l)
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["gf"][l]
        x = x + lin(F.silu(lin(h, "w1", l)) * lin(h, "w3", l), "w2", l)
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["gfin"]
    return lin(h, "cls")


@torch.no_grad()
def sample(n, T=1.0, seed=0, L_=L):
    """Teacher samples with a KV cache (float weights)."""
    g = torch.Generator().manual_seed(seed)
    P = P0
    toks = torch.ones(n, 1, dtype=torch.long)
    kc = [torch.zeros(n, 0, nkv, hs) for _ in range(nl)]
    vc = [torch.zeros(n, 0, nkv, hs) for _ in range(nl)]
    for t in range(L_ - 1):
        x = P["emb"][toks[:, -1]]
        c, s_ = COS[t], SIN[t]
        for l in range(nl):
            h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["ga"][l]
            q = (h @ P["wq"][l].T).view(n, nh, hs)
            k = (h @ P["wk"][l].T).view(n, nkv, hs)
            v = (h @ P["wv"][l].T).view(n, nkv, hs)
            rot = lambda z: torch.stack([z[..., 0::2] * c - z[..., 1::2] * s_,
                                         z[..., 0::2] * s_ + z[..., 1::2] * c], -1).flatten(-2)
            q, k = rot(q), rot(k)
            kc[l] = torch.cat([kc[l], k[:, None]], 1)
            vc[l] = torch.cat([vc[l], v[:, None]], 1)
            kk = kc[l].repeat_interleave(nh // nkv, 2)
            vv = vc[l].repeat_interleave(nh // nkv, 2)
            att = torch.einsum("bhd,bthd->bht", q, kk) / hs ** 0.5
            a = torch.einsum("bht,bthd->bhd", att.softmax(-1), vv).reshape(n, dim)
            x = x + a @ P["wo"][l].T
            h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["gf"][l]
            x = x + (F.silu(h @ P["w1"][l].T) * (h @ P["w3"][l].T)) @ P["w2"][l].T
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P["gfin"]
        lg = (h @ P["cls"].T) / T
        toks = torch.cat([toks, torch.multinomial(lg.softmax(-1), 1, generator=g)], 1)
    return toks


def mse_scales(w):
    best_s, best_e = None, None
    for c in np.linspace(0.5, 1.0, 26):
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-8) * c / 7
        e = ((torch.clamp(torch.round(w / s), -8, 7) * s - w) ** 2).sum(-1, keepdim=True)
        best_s = s if best_s is None else torch.where(e < best_e, s, best_s)
        best_e = e if best_e is None else torch.minimum(e, best_e)
    return best_s


@torch.no_grad()
def evaluate(P, S, toks, tag):
    t, s = forward(P0, toks), forward(P, toks, S)
    lt, ls = t.log_softmax(-1), s.log_softmax(-1)
    kl = (lt.exp() * (lt - ls)).sum(-1).mean().item()
    top1 = (t.argmax(-1) == s.argmax(-1)).float().mean().item()
    print(f"{tag:28s} KL {kl:.4f}  top1 {top1:.1%}", flush=True)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    t0 = time.time()
    ev = sample(32, T=0.8, seed=123, L_=96)
    pool = torch.cat([sample(256, T=1.0, seed=s) for s in range(6)]
                     + [sample(256, T=0.7, seed=100 + s) for s in range(2)])
    print(f"pool {tuple(pool.shape)} in {time.time() - t0:.0f} s", flush=True)
    with torch.no_grad():
        teach = None  # teacher logits computed per batch
    P = {k: v.clone().requires_grad_(k in MATS) for k, v in P0.items()}
    S = {k: torch.stack([mse_scales(P0[k][l]) for l in range(nl)]) if P0[k].dim() == 3
         else mse_scales(P0[k]) for k in MATS}
    for k in S:
        S[k].requires_grad_(True)
    evaluate(P, S, ev, "int4 MSE-clip (step 0)")
    opt = torch.optim.Adam([P[k] for k in MATS] + list(S.values()), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(7)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pool), (32,), generator=g)
        toks = pool[idx]
        with torch.no_grad():
            lt = forward(P0, toks).log_softmax(-1)
        ls = forward(P, toks, S).log_softmax(-1)
        loss = (lt.exp() * (lt - ls)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for k in S:
                S[k].clamp_(min=1e-6)
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}  {time.time() - t0:.0f} s", flush=True)
        if step % 500 == 0 or step == steps:
            evaluate(P, S, ev, f"int4 QAT step {step}")
    with torch.no_grad():
        out = {}
        for k in MATS:
            q = torch.clamp(torch.round(P[k] / S[k]), -8, 7).to(torch.int8)
            out[f"q_{k}"], out[f"s_{k}"] = q.numpy(), S[k].squeeze(-1).numpy()
        out["emb"] = P0["emb"].numpy()
        np.savez(f"{HERE}/qat_int4.npz", **out)
    print("saved qat_int4.npz", flush=True)
