"""KV-RAM precision study on fx's real trace (CPU, real dtypes).

  python study.py TRACE_NPZ OUT_JSON [--configs A,G,E] [--per-bank N] [--kernels math,flash,fa2,eager]

For each configuration:
1. analytic: exact integer scores for every load of the session (winner, previous same-address
   store, designed miss level) -> margins in nats, and whether the counters stay in the exact range;
2. kernels: sampled loads of every bank against the bank's full log (stores + zero-key filler
   positions), run through Qwen3.5's q_norm/k_norm/RoPE (head.py) and each attention kernel;
   reports wrong loads (decoded by threshold), bit-exact outputs, and fp32 margins.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time

import numpy as np
import torch

import kernels as K
from encoding import Enc
from head import EV, Head
from log import Log
from trace import Trace

CONFIGS = {
    # 4 kv heads as banks (address bits 0-1), a store counter per bank: the whole session in one epoch
    "A": dict(banks=4, mode="bank", n_rec=21, epoch_limit=None, epoch_bits=0),
    # one global store counter, no fix: needs 23 counter bits (outside the exact range)
    "G": dict(banks=4, mode="global", n_rec=23, epoch_limit=None, epoch_bits=0),
    # global counter + epochs: compaction at model-call boundaries keeps each epoch under 2^21
    "E": dict(banks=4, mode="global", n_rec=21, epoch_limit=1 << 21, epoch_bits=2),
}
FILL_EVERY = 8  # one zero-key filler position per this many stores (HF has ~8 non-store positions per store)
POS_PER_EVENT = 3


def analytic(L: Log, enc: Enc):
    g, n = enc.g, enc.n_rec
    hit = L.ld_hit
    tw = L.st_t[L.ld_win[hit]]
    Sw = enc.W + enc.G + g * (2 * tw - (2 ** n - 1))  # exact (python float of an integer < 2^53)
    has_prev = L.ld_prev[hit] >= 0
    tp = L.st_t[L.ld_prev[hit][has_prev]]
    m_same = 2 * g * (tw[has_prev] - tp) / 16
    min_gap = enc.alpha  # one flipped digit or epoch bit drops the score by 2 * alpha
    miss_level = enc.W + enc.G - 2 * min_gap + enc.rspan
    m_mis = (Sw - miss_level) / 16
    out = {
        "counter_max": L.max_counter, "counter_fits": L.max_counter < 2 ** n,
        "exact_bound_ok": enc.exact_bound() < 2 ** 24, "exact_bound_over_2^24": enc.exact_bound() / 2 ** 24,
        "hits": int(hit.sum()), "hits_with_older_same_address_store": int(has_prev.sum()),
        "min_margin_vs_older_same_address_nats": float(m_same.min()),
        "min_margin_vs_any_miss_nats": float(m_mis.min()),
        "min_hit_score_over_nonstore_nats": float(Sw.min() / 16),
        "miss_score_ceiling_nats": float(miss_level / 16),
        "kv_misses": int((~hit & ~L.ld_rod).sum()),
    }
    return out


def bank_log(L: Log, b):
    """The bank's KV entries in time order: stores + fillers. Returns arrays and store->entry map."""
    si = np.where(L.st_bank == b)[0]
    nst = len(si)
    nf = nst // FILL_EVERY
    ent_time = np.empty(nst + nf, np.int64)
    is_store = np.zeros(nst + nf, bool)
    # interleave: after every FILL_EVERY-th store, one filler one tick later
    slot = np.arange(nst) + np.arange(nst) // FILL_EVERY  # entry index of each store
    ent_time[slot] = L.st_time[si]
    is_store[slot] = True
    fpos = np.where(~is_store)[0]
    ent_time[fpos] = ent_time[fpos - 1] + 1
    st2ent = np.full(len(L.st_addr), -1, np.int64)
    st2ent[si] = slot
    return si, slot, ent_time, is_store, st2ent


def build_kv(L, head, enc, b, si, slot, ent_time, is_store, chunk=1 << 15, ev=EV):
    """Fill the bank's K/V in place, chunk by chunk (fillers stay exactly 0: k_proj gives 0 at
    non-store positions, k_norm(0) = 0, RoPE(0) = 0, v = 0)."""
    N = len(ent_time)
    Kt = torch.zeros(N, 256, dtype=torch.bfloat16)
    Vt = torch.zeros(N, ev, dtype=torch.bfloat16)
    for c0 in range(0, len(si), chunk):
        s = si[c0:c0 + chunk]
        sl = torch.from_numpy(slot[c0:c0 + chunk])
        et = ent_time[slot[c0:c0 + chunk]]
        pos = (et >> 20) * POS_PER_EVENT + (et & 1)
        Kt[sl] = head.keys(L.st_addr[s], L.st_t[s], pos, L.st_ep[s].astype(np.int64))
        Vt[sl] = head.values(L.st_val[s], width=ev)
    return Kt, Vt


def sample_loads(L, b, per_bank, rng):
    li = np.where((L.ld_bank == b) & ~L.ld_rod)[0]
    hit = L.ld_hit[li]
    picks = [li[-per_bank // 4:]]  # the end of the session (longest log)
    hits = li[hit]
    picks.append(rng.choice(hits, min(per_bank // 4, len(hits)), replace=False))
    # oldest winners (largest age in stores)
    age = L.ld_time[hits] - L.st_time[L.ld_win[hits]]
    picks.append(hits[np.argsort(age)[-per_bank // 8:]])
    # smallest same-address margin (winner right after the previous store)
    hp = hits[L.ld_prev[hits] >= 0]
    gap = L.st_t[L.ld_win[hp]] - L.st_t[L.ld_prev[hp]]
    picks.append(hp[np.argsort(gap, kind="stable")[:per_bank // 8]])
    miss = li[~hit]
    if len(miss):
        picks.append(rng.choice(miss, min(per_bank // 8, len(miss)), replace=False))
    # the compaction loads themselves (epoch configs) and the first loads after each compaction
    if L.compactions:
        cl = li[np.isin(L.ld_time[li] & ((1 << 20) - 1), [0], invert=True)]
        if len(cl):
            picks.append(rng.choice(cl, min(per_bank // 8, len(cl)), replace=False))
    s = np.unique(np.concatenate(picks))
    return s


def extract(L: Log, b, per_bank, rng):
    """Bank b's stores and its sampled loads, re-indexed, so the full log can be freed."""
    from types import SimpleNamespace
    samp = sample_loads(L, b, per_bank, rng)
    si = np.where(L.st_bank == b)[0]
    loc = np.full(len(L.st_addr), -1, np.int64)
    loc[si] = np.arange(len(si))
    win = L.ld_win[samp]
    return SimpleNamespace(
        st_addr=L.st_addr[si].astype(np.int64), st_val=L.st_val[si].copy(), st_time=L.st_time[si].copy(),
        st_ep=L.st_ep[si].copy(), st_t=L.st_t[si].copy(), st_bank=np.full(len(si), b, np.int8),
        ld_addr=L.ld_addr[samp].copy(), ld_val=L.ld_val[samp].copy(), ld_time=L.ld_time[samp].copy(),
        ld_ep=L.ld_ep[samp].copy(), ld_hit=L.ld_hit[samp].copy(), ld_rod=L.ld_rod[samp].copy(),
        ld_win=np.where(win >= 0, loc[np.maximum(win, 0)], -1))


def wait_mem(gb, log=print, max_wait=3600):
    """Wait until the system has `gb` GB of free commit (other jobs share this machine)."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                    ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    t0 = time.time()
    while True:
        m = MS()
        m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        if min(m.apf, m.ap) / 2 ** 30 >= gb or time.time() - t0 > max_wait:
            return
        time.sleep(15)


def run_kernels(L, enc, head, b, kernels, group=16, log=print):
    """L: a bank view from extract() (stores of bank b, the sampled loads)."""
    ev = EV if "flash" in kernels else 64  # CPU flash needs Ev == 256; the others take a narrow V
    wait_mem(2.0 if ev == 64 else 3.0, log)
    si, slot, ent_time, is_store, st2ent = bank_log(L, b)
    Kt, Vt = build_kv(L, head, enc, b, si, slot, ent_time, is_store, ev=ev)
    samp = np.argsort(L.ld_time, kind="stable")
    stats = {k: {"loads": 0, "wrong": 0, "bit_exact": 0, "max_abs_err_bits": 0.0, "max_abs_err_bytes": 0.0,
                 "wrong_examples": []} for k in kernels}
    margins = {"min_fp32_margin_hit_nats": math.inf, "min_exact_margin_hit_nats": math.inf,
               "max_fp32_store_score_on_miss_nats": -math.inf, "fp32_vs_exact_winner_mismatch": 0}
    t0 = time.time()
    for g0 in range(0, len(samp), group):
        ids = samp[g0:g0 + group]
        lt = L.ld_time[ids]
        N = int(np.searchsorted(ent_time, lt.max()))  # entries strictly before the last load
        mask = torch.from_numpy(ent_time[None, :N] < lt[:, None])
        qpos = (lt >> 20) * POS_PER_EVENT + 2
        Q = head.queries(L.ld_addr[ids], qpos, L.ld_ep[ids].astype(np.int64))
        hit = L.ld_hit[ids]
        win_ent = np.where(hit, st2ent[np.maximum(L.ld_win[ids], 0)], -1)
        exp = torch.zeros(len(ids), ev, dtype=torch.bfloat16)
        exp[hit] = Vt[win_ent[hit]]
        # fp32 scores as the kernels form them, for margins
        S = K.scores_fp32(Q, Kt[:N]).double() * 16
        S.masked_fill_(~mask, -math.inf)
        for i in range(len(ids)):
            if hit[i]:
                sw = S[i, win_ent[i]].item()
                S[i, win_ent[i]] = -math.inf
                ru = S[i].max().item()
                margins["min_fp32_margin_hit_nats"] = min(margins["min_fp32_margin_hit_nats"], (sw - ru) / 16)
                ex_w = enc.score_int(L.ld_addr[ids[i]], L.ld_addr[ids[i]], L.st_t[L.ld_win[ids[i]]],
                                     L.ld_ep[ids[i]], L.ld_ep[ids[i]])
                if float(ex_w) != sw:
                    margins["fp32_vs_exact_winner_mismatch"] += 1
                j = int(S[i].argmax())
                if is_store[j]:
                    sj = si[np.searchsorted(slot, j)]
                    ex_r = enc.score_int(L.ld_addr[ids[i]], L.st_addr[sj], L.st_t[sj], L.ld_ep[ids[i]], L.st_ep[sj])
                else:
                    ex_r = 0
                margins["min_exact_margin_hit_nats"] = min(margins["min_exact_margin_hit_nats"], float(ex_w - ex_r) / 16)
            else:
                st_cols = torch.from_numpy(is_store[:N]) & mask[i]
                if st_cols.any():
                    margins["max_fp32_store_score_on_miss_nats"] = max(margins["max_fp32_store_score_on_miss_nats"],
                                                                        S[i][st_cols].max().item() / 16)
        del S
        for kn in kernels:
            if kn == "math":
                o = K.math_replica(Q, Kt[:N], Vt[:N], mask)
            elif kn == "flash":
                o = K.sdpa(Q[None, None], Kt[None, None, :N], Vt[None, None, :N], mask[None, None], "flash")[0, 0]
            elif kn == "fa2":
                o = K.fa2(Q, Kt[:N], Vt[:N], mask)
            elif kn == "eager":
                o = K.eager(Q[None, None], Kt[None, None, :N], Vt[None, None, :N], mask)[0, 0]
            st = stats[kn]
            h_dec, v_dec = Head.decode(o)
            want_v = np.where(hit, L.ld_val[ids], 0).astype(np.uint32)
            wrong = (h_dec != hit) | (v_dec != want_v)
            st["loads"] += len(ids)
            st["wrong"] += int(wrong.sum())
            st["bit_exact"] += int((o == exp).all(1).sum())
            st["max_abs_err_bits"] = max(st["max_abs_err_bits"], (o[:, :33].float() - exp[:, :33].float()).abs().max().item())
            st["max_abs_err_bytes"] = max(st["max_abs_err_bytes"], (o[:, 33:37].float() - exp[:, 33:37].float()).abs().max().item())
            for i in np.where(wrong)[0][:3]:
                if len(st["wrong_examples"]) < 6:
                    st["wrong_examples"].append({"load_time_event": int(lt[i] >> 20), "hit": bool(hit[i]),
                                                 "want": int(want_v[i]), "got": int(v_dec[i]), "got_hit": bool(h_dec[i])})
    log(f"  bank {b}: {len(samp)} loads, log {len(ent_time)} entries, {time.time() - t0:.0f}s")
    del Kt, Vt
    gc.collect()
    return stats, margins, len(samp), int(len(ent_time))


def check_replica(head, enc, n=200_000, nq=16, seed=0):
    """math_replica == torch SDPA MATH, and torch flash == math, bit for bit, on a synthetic log."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << 21, n)
    a[: n // 2] = rng.integers(0, 64, n // 2)  # many same-address stores
    t = np.arange(n)
    Kt = head.keys(a, t, rng.integers(0, 40_000_000, n))
    Vt = Head.values(rng.integers(0, 2 ** 32, n, dtype=np.uint64))
    qa = rng.choice(a, nq)
    Q = head.queries(qa, np.full(nq, 40_000_001))
    mask = torch.from_numpy(rng.random((nq, n)) < 0.999)
    o_t = K.sdpa(Q[None, None], Kt[None, None], Vt[None, None], mask[None, None], "math")[0, 0]
    o_r = K.math_replica(Q, Kt, Vt, mask)
    o_f = K.sdpa(Q[None, None], Kt[None, None], Vt[None, None], mask[None, None], "flash")[0, 0]
    return {"log": n, "math_replica_equals_torch_math": bool(torch.equal(o_t, o_r)),
            "torch_flash_equals_torch_math": bool(torch.equal(o_t, o_f))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--configs", default="A,G,E")
    ap.add_argument("--per-bank", type=int, default=512)
    ap.add_argument("--kernels", default="math,flash,fa2,eager")
    ap.add_argument("--banks", default="0,1,2,3")
    a = ap.parse_args()
    torch.set_num_threads(8)
    log = lambda *x: print(*x, file=sys.stderr, flush=True)
    tr = Trace(a.trace)
    res = {"trace": {"events": tr.n_events, "stores": tr.n_st, "loads": tr.n_ld}, "configs": {}}
    del tr
    kernels = a.kernels.split(",")
    for cn in a.configs.split(","):
        cfg = CONFIGS[cn]
        wait_mem(2.5, log)
        tr = Trace(a.trace)
        L = Log(tr, banks=cfg["banks"], mode=cfg["mode"], epoch_limit=cfg["epoch_limit"])
        del tr
        gc.collect()
        enc = Enc(n_rec=cfg["n_rec"], epoch_bits=cfg["epoch_bits"])
        head = Head(enc)
        r = {"config": cfg, "encoding": enc.summary(), "log": L.check(), "analytic": analytic(L, enc)}
        log(cn, json.dumps(r["analytic"]))
        part_p = a.out + ".partial.json"
        try:
            part = json.load(open(part_p))
        except (OSError, ValueError):
            part = {}
        if cn == "A":
            if "A/replica" not in part:
                part["A/replica"] = check_replica(head, enc)
                json.dump(part, open(part_p, "w"), default=float)
            r["replica_check"] = part["A/replica"]
            log("  replica", r["replica_check"])
        rng = np.random.default_rng(1)
        agg = {k: {"loads": 0, "wrong": 0, "bit_exact": 0, "max_abs_err_bits": 0.0, "max_abs_err_bytes": 0.0,
                   "wrong_examples": []} for k in kernels}
        mg = {"min_fp32_margin_hit_nats": math.inf, "min_exact_margin_hit_nats": math.inf,
              "max_fp32_store_score_on_miss_nats": -math.inf, "fp32_vs_exact_winner_mismatch": 0}
        logs = []
        views = {}
        for b in [int(x) for x in a.banks.split(",")]:
            key = f"{cn}/{b}"
            rng = np.random.default_rng(100 + b)
            if key not in part:
                views[b] = views.get(b) or extract(L, b, a.per_bank, rng)
        del L
        gc.collect()
        for b in [int(x) for x in a.banks.split(",")]:
            key = f"{cn}/{b}"
            if key not in part:
                part[key] = run_kernels(views.pop(b), enc, head, b, kernels, log=log)
                json.dump(part, open(part_p, "w"), default=float)
                gc.collect()
            st, m, ns, ne = part[key]
            logs.append(ne)
            for k in kernels:
                for f in ("loads", "wrong", "bit_exact"):
                    agg[k][f] += st[k][f]
                for f in ("max_abs_err_bits", "max_abs_err_bytes"):
                    agg[k][f] = max(agg[k][f], st[k][f])
                agg[k]["wrong_examples"] += st[k]["wrong_examples"][: max(0, 6 - len(agg[k]["wrong_examples"]))]
            mg["min_fp32_margin_hit_nats"] = min(mg["min_fp32_margin_hit_nats"], m["min_fp32_margin_hit_nats"])
            mg["min_exact_margin_hit_nats"] = min(mg["min_exact_margin_hit_nats"], m["min_exact_margin_hit_nats"])
            mg["max_fp32_store_score_on_miss_nats"] = max(mg["max_fp32_store_score_on_miss_nats"], m["max_fp32_store_score_on_miss_nats"])
            mg["fp32_vs_exact_winner_mismatch"] += m["fp32_vs_exact_winner_mismatch"]
            log("   ", json.dumps({k: {f: st[k][f] for f in ("loads", "wrong", "bit_exact")} for k in kernels}))
        r["kernels"] = agg
        r["margins_sampled"] = mg
        r["bank_log_entries_with_fillers"] = logs
        res["configs"][cn] = r
        gc.collect()
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
