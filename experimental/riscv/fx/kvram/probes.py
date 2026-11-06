"""What breaks first: adversarial probe logs at counter widths n_rec = 2..26 (log lengths up to 2^26).

Each probe log holds, for each of Q query addresses:
- a run of same-address stores at the top of the counter range (latest wins by 2g = 128 nats);
- 1-digit-mismatch stores at the very top counters (the strongest wrong candidates);
- an old-winner case: the address's only store at counter 0 while its 1-digit neighbours are
  stored at the top counters;
plus random bulk entries and zero-key filler positions, at positions up to 2^30.
Every kernel reads every query against the whole log; we count wrong loads (decoded), bit-exact
outputs, and the fp32 margin of the winner.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

import kernels as K
from encoding import Enc
from head import EV, Head


def probe(n, kernels, Q=48, run=16, bulk=4096, seed=0):
    enc = Enc(n_rec=n)
    head = Head(enc)
    rng = np.random.default_rng(seed)
    T = 2 ** n - 1
    qa = rng.choice(1 << 21, Q, replace=False)
    kind = np.arange(Q) % 3  # 0: run at the top; 1: run below a 1-digit miss at the top; 2: old winner
    addrs, ts = [], []
    flips = [1 << s for s in (0, 6, 11, 16)]  # one bit in each digit group
    for i, a in enumerate(qa):
        k = min(run, T + 1)
        if kind[i] == 0:  # consecutive same-address stores T, T-1, ...: latest wins by exactly 2g
            for j in range(k):
                addrs.append(a); ts.append(T - j)
        elif kind[i] == 1:  # same-address run just below; the 4 neighbours (one per digit) hold the top counters
            for j in range(4):
                addrs.append(a ^ flips[j]); ts.append(max(T - j, 0))
            for j in range(min(k, max(1, T - 3))):
                addrs.append(a); ts.append(max(T - 4 - j, 0))
        else:  # the address's only store at counter 0, its neighbours at the top counters
            addrs.append(a); ts.append(0)
            for j in range(4):
                addrs.append(a ^ flips[j]); ts.append(max(T - j, 1))
    old = kind == 2
    ab = rng.integers(0, 1 << 21, bulk)
    ab = ab[~np.isin(ab, qa)]
    addrs += ab.tolist(); ts += rng.integers(0, T + 1, len(ab)).tolist()
    addrs, ts = np.array(addrs, np.int64), np.array(ts, np.int64)
    # expected winner per query: latest counter among same-address entries
    win = np.array([np.where(addrs == a)[0][np.argmax(ts[addrs == a])] for a in qa])
    assert all((ts[addrs == a] == ts[w]).sum() == 1 for a, w in zip(qa, win))
    vals = rng.integers(0, 2 ** 32, len(addrs), dtype=np.uint64)
    N = len(addrs)
    nf = N // 4
    Kt = torch.zeros(N + nf, 256, dtype=torch.bfloat16)
    Vt = torch.zeros(N + nf, EV, dtype=torch.bfloat16)
    perm = rng.permutation(N + nf)  # entries in a random cache order (order in the cache is free)
    slot = perm[:N]
    Kt[slot] = head.keys(addrs, ts, rng.integers(0, 2 ** 30, N))
    Vt[slot] = Head.values(vals)
    Qt = head.queries(qa, np.full(Q, 2 ** 30))
    exp = Vt[slot[win]]
    want = vals[win].astype(np.uint32)
    S = K.scores_fp32(Qt, Kt).double() * 16
    sw = S[np.arange(Q), slot[win]].clone()
    S[np.arange(Q), slot[win]] = -math.inf
    marg = ((sw - S.max(1).values) / 16)
    ex_w = enc.score_int(qa, qa, ts[win])
    res = {"n_rec": n, "log_len": 2 ** n, "exact_bound_over_2^24": enc.exact_bound() / 2 ** 24,
           "min_fp32_margin_nats": float(marg.min()), "min_fp32_margin_old_winner_nats": float(marg[old].min()), "min_fp32_margin_top_miss_nats": float(marg[kind == 1].min()),
           "winner_score_fp32_equals_exact": int((sw.numpy() == ex_w.astype(np.float64)).sum()), "queries": Q}
    for kn in kernels:
        if kn == "math":
            o = K.math_replica(Qt, Kt, Vt)
        elif kn == "flash":
            o = K.sdpa(Qt[None, None], Kt[None, None], Vt[None, None], None, "flash")[0, 0]
        elif kn == "fa2":
            o = K.fa2(Qt, Kt, Vt)
        elif kn == "eager":
            o = K.eager(Qt[None, None], Kt[None, None], Vt[None, None])[0, 0]
        hd, vd = Head.decode(o)
        wrong = (~hd) | (vd != want)
        res[kn] = {"wrong": int(wrong.sum()), "bit_exact": int((o == exp).all(1).sum()),
                   "max_abs_err_bits": float((o[:, :33].float() - exp[:, :33].float()).abs().max()),
                   "byte_exact": int((o[:, 33:37] == exp[:, 33:37]).all(1).sum())}
    return res


def main():
    out = sys.argv[1]
    torch.set_num_threads(4)
    rows = []
    for n in range(2, 27):
        ks = ["math", "flash", "fa2"] + (["eager"] if n <= 12 else [])
        r = probe(n, ks)
        rows.append(r)
        print(json.dumps(r), flush=True)
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
