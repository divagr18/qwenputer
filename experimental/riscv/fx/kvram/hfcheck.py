"""KV-RAM through the stock transformers Qwen3_5Attention module (bf16, CPU), on a window of fx's trace.

The module is built from Qwen3.5-4B's text config (16 q heads, 4 kv heads, head_dim 256, output
gate, partial RoPE) with a small synthetic residual (hidden_states = the attention input after the
layer's input norm). q/k/v/o and the gate are hand-set; q_norm / k_norm weights come from Enc.
- kv head j is bank j (word address bits 0-1); q head 4j is bank j's load port; the gate opens
  (+16 -> sigmoid = 1.0 in bf16) only at loads to bank j and is -128 (sigmoid = 0) elsewhere;
- o_proj adds the open head's 32 data bits, hit flag and 4 bytes into fresh residual dims;
- the residual add is bf16, as in Qwen3_5DecoderLayer.

All stores before the window are injected into a DynamicCache through the module's own
k_proj/k_norm/RoPE/v_proj, then the window runs chunk by chunk (stores, loads and one filler
position per event). Per-bank counters are offset so the window ends at the top of the 21-bit range.
Checks, at every position of the window: the residual's output dims equal the expected word
(winner's bits, hit, bytes at loads; zero elsewhere), bit for bit.

  python hfcheck.py TRACE_NPZ OUT_JSON [--start EVENT] [--events N] [--impl flash,eager]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

from encoding import E, ROT, Enc
from head import Head, text_config
from log import Log
from trace import Trace

NB = 4  # banks = kv heads


class Layout:
    def __init__(self, enc: Enc):
        self.enc = enc
        i = 0

        def take(n):
            nonlocal i
            r = np.arange(i, i + n)
            i += n
            return r
        self.const = take(1)[0]
        self.st = take(NB)
        self.kdig = [[take(b) for b in enc.bases] for _ in range(NB)]
        self.krec = [take(enc.n_rec) for _ in range(NB)]
        self.vbit = [take(32) for _ in range(NB)]
        self.vbyte = [take(4) for _ in range(NB)]
        self.qdig = [take(b) for b in enc.bases]
        self.ld = take(NB)
        self.obit = take(32)
        self.ohit = take(1)[0]
        self.obyte = take(4)
        self.H = (i + 63) // 64 * 64


def build_module(enc: Enc, lay: Layout, impl):
    cfg = text_config()
    cfg.hidden_size = lay.H
    cfg._attn_implementation = impl
    m = Qwen3_5Attention(cfg, layer_idx=0).to(torch.bfloat16).eval()
    H, nq, nkv = lay.H, cfg.num_attention_heads, cfg.num_key_value_heads
    assert (nq, nkv, m.head_dim) == (16, 4, 256)
    Wq = torch.zeros(nq * 2 * E, H)
    Wk = torch.zeros(nkv * E, H)
    Wv = torch.zeros(nkv * E, H)
    Wo = torch.zeros(H, nq * E)
    for j in range(nkv):
        r = j * E
        for gi, b in enumerate(enc.bases):
            for d in range(b):
                Wk[r + enc.addr_off[gi] + d, lay.kdig[j][gi][d]] = 1
        for i in range(enc.n_rec):
            Wk[r + enc.rec_off + i, lay.krec[j][i]] = 2
            Wk[r + enc.rec_off + i, lay.st[j]] = -1
        for w in range(enc.n_w):
            Wk[r + enc.w_off + w, lay.st[j]] = 1
        for d, v in enumerate(enc.k_ballast):
            Wk[r + d, lay.st[j]] = v
        for i in range(32):
            Wv[r + i, lay.vbit[j][i]] = 1
        Wv[r + 32, lay.st[j]] = 1
        for b in range(4):
            Wv[r + 33 + b, lay.vbyte[j][b]] = 1
    for h in range(nq):
        r = h * 2 * E  # q of head h: rows r..r+255, its gate: r+256..r+511
        for gi, b in enumerate(enc.bases):
            for d in range(b):
                Wq[r + enc.addr_off[gi] + d, lay.qdig[gi][d]] = 2
                Wq[r + enc.addr_off[gi] + d, lay.const] = -1
        Wq[r + enc.rec_off:r + enc.rec_off + enc.n_rec, lay.const] = 1
        Wq[r + enc.w_off:r + enc.w_off + enc.n_w, lay.const] = 1
        for d, v in enumerate(enc.q_ballast):
            Wq[r + 32 + d, lay.const] = v
        if h % 4 == 0:
            j = h // 4
            Wq[r + E:r + 2 * E, lay.ld[j]] = 144
        Wq[r + E:r + 2 * E, lay.const] = -128
        if h % 4 == 0:
            for i in range(32):
                Wo[lay.obit[i], h * E + i] = 1
            Wo[lay.ohit, h * E + 32] = 1
            for b in range(4):
                Wo[lay.obyte[b], h * E + 33 + b] = 1
    qw, kw = enc.norm_weights()
    with torch.no_grad():
        m.q_proj.weight.copy_(Wq)
        m.k_proj.weight.copy_(Wk)
        m.v_proj.weight.copy_(Wv)
        m.o_proj.weight.copy_(Wo)
        m.q_norm.weight.copy_(qw)
        m.k_norm.weight.copy_(kw)
    return m, cfg


def residual_rows(lay, enc, kind, addr, t, val):
    """kind: 1 store, 0 load, -1 filler. Returns bf16 [N, H]."""
    N = len(kind)
    h = np.zeros((N, lay.H), np.float32)
    h[:, lay.const] = 1
    a = np.maximum(addr, 0)
    bank = a & (NB - 1)
    dg = enc.digits(a)
    rows = np.arange(N)
    st = kind == 1
    for j in range(NB):
        s = st & (bank == j)
        if not s.any():
            continue
        r = rows[s]
        h[r, lay.st[j]] = 1
        for gi in range(len(enc.bases)):
            h[r, lay.kdig[j][gi][dg[gi][s]]] = 1
        for i in range(enc.n_rec):
            h[r, lay.krec[j][i]] = (t[s] >> i) & 1
        v = val[s].astype(np.int64)
        for i in range(32):
            h[r, lay.vbit[j][i]] = (v >> i) & 1
        for b in range(4):
            h[r, lay.vbyte[j][b]] = (v >> (8 * b)) & 255
    ld = (kind == 0) & (addr >= 0)
    r = rows[ld]
    for gi in range(len(enc.bases)):
        h[r, lay.qdig[gi][dg[gi][ld]]] = 1
    for j in range(NB):
        h[rows[ld & (bank == j)], lay.ld[j]] = 1
    return torch.from_numpy(h).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=153202)  # first event of boot (all instantiate stores in the cache)
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--impl", default="flash,eager")
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(8)
    enc = Enc(n_rec=21)
    lay = Layout(enc)
    tr = Trace(a.trace)
    L = Log(tr, banks=NB, mode="bank")
    SH = 1 << 20
    e0, e1 = a.start, a.start + a.events
    # per-bank counter offset: the window's last store sits at 2^21 - 1
    in_win = (L.st_time < e1 * SH)
    off = np.zeros(NB, np.int64)
    for j in range(NB):
        m = in_win & (L.st_bank == j)
        off[j] = (2 ** 21 - 1) - L.st_t[m].max()
    t_all = L.st_t + off[L.st_bank]
    # events of the window, each followed by a filler position
    sp = (L.st_time >= e0 * SH) & in_win
    lp = (L.ld_time >= e0 * SH) & (L.ld_time < e1 * SH)
    ev_t = np.r_[L.st_time[sp], L.ld_time[lp]]
    o = np.argsort(ev_t, kind="stable")
    kind = np.r_[np.ones(sp.sum(), np.int64), np.zeros(lp.sum(), np.int64)][o]
    addr = np.r_[L.st_addr[sp], L.ld_addr[lp]][o]
    tt = np.r_[t_all[sp], np.zeros(lp.sum(), np.int64)][o]
    val = np.r_[L.st_val[sp], np.zeros(lp.sum(), np.uint32)][o]
    lidx = np.r_[np.full(sp.sum(), -1), np.where(lp)[0]][o]
    n = len(kind)
    kind2 = np.repeat(kind, 2); kind2[1::2] = -1
    addr2 = np.repeat(addr, 2); tt2 = np.repeat(tt, 2); val2 = np.repeat(val, 2); lidx2 = np.repeat(lidx, 2); lidx2[1::2] = -1
    hid = residual_rows(lay, enc, kind2, addr2, tt2, val2)
    pos0 = 3 * e0  # positions continue the session's (about 3 per event)
    # expected output dims
    exp = torch.zeros(2 * n, 37, dtype=torch.bfloat16)
    for p in np.where(lidx2 >= 0)[0]:
        li = lidx2[p]
        if L.ld_hit[li]:
            v = int(L.st_val[L.ld_win[li]])
            exp[p, :32] = torch.tensor([(v >> i) & 1 for i in range(32)], dtype=torch.bfloat16)
            exp[p, 32] = 1
            exp[p, 33:37] = torch.tensor([(v >> (8 * b)) & 255 for b in range(4)], dtype=torch.bfloat16)
    out_dims = list(lay.obit) + [lay.ohit] + list(lay.obyte)
    pre = L.st_time < e0 * SH  # stores before the window go into the cache directly
    res = {"window_events": [e0, e1], "positions": 2 * n, "loads": int((kind == 0).sum()),
           "kv_loads": int(((kind == 0) & (addr >= 0)).sum()), "stores": int((kind == 1).sum()),
           "prefix_cache_entries": int(pre.sum()), "hidden": lay.H, "counter_max": int(tt.max()), "impl": {}}
    for impl in a.impl.split(","):
        hf_impl = "sdpa" if impl in ("flash", "math") else impl
        m, cfg = build_module(enc, lay, hf_impl)
        hd = Head(enc, cfg)
        cache = DynamicCache()
        t0 = time.time()
        with torch.no_grad():
            # prefix: stores before the window, through the module's own projections
            pi = np.where(pre)[0]
            hp = residual_rows(lay, enc, np.ones(len(pi), np.int64), L.st_addr[pi], t_all[pi], L.st_val[pi])
            ppos = torch.from_numpy((L.st_time[pi] >> 20) * 3)[None]
            ks, vs = [], []
            for c0 in range(0, len(pi), 8192):
                x = hp[c0:c0 + 8192][None]
                k = m.k_norm(m.k_proj(x).view(1, x.shape[1], -1, E)).transpose(1, 2)
                v = m.v_proj(x).view(1, x.shape[1], -1, E).transpose(1, 2)
                cos, sin = hd.rope(x, ppos[:, c0:c0 + 8192])
                _, k = apply_rotary_pos_emb(k, k, cos, sin)
                ks.append(k); vs.append(v)
            cache.update(torch.cat(ks, 2), torch.cat(vs, 2), 0)
            del ks, vs, hp
            P = len(pi)
            got = torch.zeros(2 * n, 37, dtype=torch.bfloat16)
            other = 0.0
            for c0 in range(0, 2 * n, a.chunk):
                x = hid[c0:c0 + a.chunk][None]
                C = x.shape[1]
                pos = torch.arange(pos0 + c0, pos0 + c0 + C)[None]
                cos, sin = hd.rope(x, pos)
                tot = P + c0 + C
                causal = torch.ones(C, tot, dtype=torch.bool)
                causal[:, P + c0:] = torch.tril(torch.ones(C, C, dtype=torch.bool))
                if hf_impl == "eager":
                    mask = torch.zeros(1, 1, C, tot, dtype=torch.bfloat16).masked_fill(~causal, torch.finfo(torch.bfloat16).min)
                    ctx = torch.no_grad()
                else:
                    mask = causal[None, None]
                    ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION if impl == "flash" else SDPBackend.MATH])
                with ctx:
                    y, _ = m(x, (cos, sin), mask, past_key_values=cache)
                r = x + y  # the decoder layer's bf16 residual add
                got[c0:c0 + C] = r[0][:, out_dims]
                keep = torch.ones(lay.H, dtype=torch.bool)
                keep[out_dims] = False
                other = max(other, (r[0][:, keep].float() - x[0][:, keep].float()).abs().max().item())
        lp_pos = lidx2 >= 0
        ld_ok = (got == exp).all(1)
        h_dec = got[:, 32].float() > 0.5
        bits = (got[:, :32].float() > 0.5).long()
        v_dec = (bits << torch.arange(32)).sum(1).numpy()
        want = np.zeros(2 * n, np.int64)
        for p in np.where(lp_pos)[0]:
            li = lidx2[p]
            want[p] = int(L.st_val[L.ld_win[li]]) if L.ld_hit[li] else 0
        wrong = ((v_dec != want) | (h_dec.numpy() != (exp[:, 32] > 0.5).numpy())) & lp_pos
        res["impl"][impl] = {"positions_bit_exact": int(ld_ok.sum()), "positions": 2 * n,
                             "loads_wrong": int(wrong.sum()), "loads_bit_exact": int(ld_ok[torch.from_numpy(lp_pos)].sum()),
                             "nonload_positions_exactly_zero": int(ld_ok[torch.from_numpy(~lp_pos)].sum()),
                             "max_change_in_other_residual_dims": other, "seconds": round(time.time() - t0, 1)}
        print(impl, json.dumps(res["impl"][impl]), file=sys.stderr, flush=True)
        del m, cache
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
