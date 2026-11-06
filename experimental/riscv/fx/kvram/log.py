"""KV-RAM logs built from the fx trace for one configuration: banks, counter mode, epochs.

A log is the sequence of KV entries the RAM layer holds (stores, and for the epoch scheme the
re-stores at each compaction) plus the loads that read it. Times are event indices times 2**20
(int64), so compaction events can sit between two trace events.
"""
from __future__ import annotations

import numpy as np

from trace import Trace


def latest_before(st_key, st_time, ld_key, ld_time):
    """For each load: index of the latest store with the same key and time < load time, and the one
    before it (-1 if none). Keys int32/int64, times int64 (distinct). Written for low memory."""
    ns, nl = len(st_key), len(ld_key)
    t = np.concatenate([st_time, ld_time])
    o_t = np.argsort(t, kind="stable")  # time order of all events
    del t
    key = np.concatenate([st_key, ld_key]).astype(np.int64)[o_t]
    o_k = np.argsort(key, kind="stable")  # by key, time order inside
    k_s = key[o_k]
    del key
    order = o_t[o_k]  # event (st index, or ns + ld index) in (key, time) order
    del o_t, o_k
    n = ns + nl
    ar = np.arange(n, dtype=np.int32)
    gstart = np.where(np.r_[True, k_s[1:] != k_s[:-1]], ar, 0).astype(np.int32)
    del k_s
    np.maximum.accumulate(gstart, out=gstart)
    is_s = order < ns
    pos = np.where(is_s, ar, -1).astype(np.int32)
    del ar
    np.maximum.accumulate(pos, out=pos)
    sp = np.empty(n, np.int32)
    sp[0] = -1
    sp[1:] = pos[:-1]  # latest store strictly before, in sorted order
    del pos
    sp[sp < gstart] = -1
    del gstart
    srt = np.where(sp >= 0, order[np.maximum(sp, 0)], -1)  # its store index
    del sp
    st_prev = np.full(ns, -1, np.int64)
    st_prev[order[is_s]] = srt[is_s]
    win = np.full(nl, -1, np.int64)
    win[order[~is_s] - ns] = srt[~is_s]
    del srt, order, is_s
    prev = np.where(win >= 0, st_prev[np.maximum(win, 0)], -1)
    return win, prev


class Log:
    def __init__(self, tr: Trace, banks=4, mode="bank", epoch_limit=None, n_rec=21):
        """mode: counter per 'bank' or 'global'. epoch_limit: if set, compact (re-store every live,
        non-zero word into a new epoch) at the first phase boundary where the current epoch's
        counter would pass epoch_limit in the next phase."""
        self.banks, self.mode = banks, mode
        SH = 1 << 20
        st_addr, st_val, st_time = tr.st_addr.astype(np.int64), tr.st_val, tr.st_ev.astype(np.int64) * SH
        ld_addr, ld_val, ld_time = tr.ld_addr.astype(np.int64), tr.ld_val, tr.ld_ev.astype(np.int64) * SH
        st_ep = np.zeros(len(st_addr), np.int8)
        ld_ep = np.zeros(len(ld_addr), np.int8)
        self.compactions = []
        extra_st, extra_ld = [], []
        if epoch_limit:
            bounds = np.r_[tr.phase_at, tr.n_events]
            cs = np.searchsorted(tr.st_ev, bounds)  # stores before each boundary
            ep, ep_used = 0, int(cs[0])  # stores of phase 0 (before the first boundary)
            for i in range(len(bounds) - 1):
                nxt = cs[i + 1] - cs[i]
                if ep_used + nxt > epoch_limit and ep_used > 0:
                    # compact at bounds[i]: live words = latest value per address among stores so far
                    n = cs[i]
                    a, v = st_addr[:n], st_val[:n]
                    o = np.lexsort((np.arange(n), a))
                    last = np.r_[a[o][1:] != a[o][:-1], True]
                    la, lv = a[o][last], v[o][last]
                    nz = lv != 0
                    la, lv = la[nz], lv[nz]
                    k = len(la)
                    assert 2 * k + 2 < SH
                    # a re-store = a load in epoch ep (old) then a store in epoch ep+1, in address
                    # order, between the last event of phase i and the first of phase i+1
                    tt = (int(bounds[i]) - 1) * SH + 1 + 2 * np.arange(k, dtype=np.int64)
                    extra_ld.append((la, lv, tt, np.full(k, ep, np.int8)))
                    extra_st.append((la, lv, tt + 1, np.full(k, ep + 1, np.int8)))
                    ep += 1
                    st_ep[cs[i]:] = ep
                    ld_ep[np.searchsorted(tr.ld_ev, bounds[i]):] = ep
                    self.compactions.append({"before_phase": tr.phase_names[i + 1] if i + 1 < len(tr.phase_names) else "?",
                                             "at_event": int(bounds[i]), "live_words": int(k), "epoch": ep})
                    ep_used = k
                ep_used += nxt
        if extra_st:
            cat = lambda base, xs, j: np.concatenate([base] + [x[j] for x in xs])
            st_addr, st_val, st_time, st_ep = (cat(st_addr, extra_st, 0), cat(st_val, extra_st, 1),
                                               cat(st_time, extra_st, 2), cat(st_ep, extra_st, 3))
            ld_addr, ld_val, ld_time, ld_ep = (cat(ld_addr, extra_ld, 0), cat(ld_val, extra_ld, 1),
                                               cat(ld_time, extra_ld, 2), cat(ld_ep, extra_ld, 3))
            o = np.argsort(st_time, kind="stable")
            st_addr, st_val, st_time, st_ep = st_addr[o], st_val[o], st_time[o], st_ep[o]
            o = np.argsort(ld_time, kind="stable")
            ld_addr, ld_val, ld_time, ld_ep = ld_addr[o], ld_val[o], ld_time[o], ld_ep[o]
        self.n_compaction_restores = int(sum(len(x[0]) for x in extra_st))
        self.st_addr, self.st_val, self.st_time, self.st_ep = st_addr, st_val.astype(np.uint32), st_time, st_ep
        self.ld_addr, self.ld_val, self.ld_time, self.ld_ep = ld_addr, ld_val.astype(np.uint32), ld_time, ld_ep
        self.ld_rod = ld_addr < 0
        self.st_bank = (st_addr & (banks - 1)).astype(np.int8)
        self.ld_bank = (np.maximum(ld_addr, 0) & (banks - 1)).astype(np.int8)
        # counters: per (epoch, bank) or per epoch
        grp = self.st_ep.astype(np.int64) * banks + (self.st_bank if mode == "bank" else 0)
        self.st_t = np.zeros(len(st_addr), np.int64)
        for gk in np.unique(grp):
            m = grp == gk
            self.st_t[m] = np.arange(m.sum())
        self.max_counter = int(self.st_t.max())
        self.n_rec_needed = int(self.max_counter).bit_length()
        # winners: same (address, epoch)
        key_s = self.st_addr * 64 + self.st_ep
        key_l = np.where(ld_addr >= 0, ld_addr * 64 + ld_ep, -1 - np.arange(len(ld_addr)))
        self.ld_win, self.ld_prev = latest_before(key_s, self.st_time, key_l, self.ld_time)
        self.ld_hit = self.ld_win >= 0

    def check(self):
        """Last-write-wins on this log reproduces every loaded value (misses read 0 or .rodata)."""
        h = self.ld_hit
        ok_hit = bool((self.st_val[self.ld_win[h]] == self.ld_val[h]).all())
        miss = ~h & ~self.ld_rod
        ok_miss = bool((self.ld_val[miss] == 0).all())
        return {"entries": int(len(self.st_addr)), "loads": int(len(self.ld_addr)), "hits": int(h.sum()),
                "kv_misses": int(miss.sum()), "rodata_loads": int(self.ld_rod.sum()), "values_ok": ok_hit and ok_miss,
                "max_counter": self.max_counter, "counter_bits_needed": self.n_rec_needed,
                "compaction_restores": self.n_compaction_restores, "compactions": self.compactions,
                "entries_per_bank": np.bincount(self.st_bank, minlength=self.banks).tolist()}
