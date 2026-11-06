"""fx's word-level memory trace (gen_trace.py output) as a KV-RAM workload.

Every word store (CPU store or host write) is one KV entry. A load must return the latest store
to its word; a never-stored word reads .rodata (a DROM, outside KV-RAM) or zero.

Address map: Qwenputer would lay fx out compactly, so the wasm2c build's 35 MB span is remapped
to a 21-bit word address: .bss/.rodata page (1-8 MB) at 0, the native stack + linear memory
(from 31 MB) at 2**18 words.
"""
from __future__ import annotations

import numpy as np

RODATA = (0x3DC4F0, 0x3DC4F0 + 316848)
LIN = 0x02000000


def compact(word):
    b = word.astype(np.int64) * 4
    low = b < 0x01000000
    out = np.where(low, (b - 0x400000) >> 2, ((b - 0x01F00000) >> 2) + (1 << 18))
    assert (out >= 0).all() and (out < (1 << 21)).all()
    return out


class Trace:
    """Compact store/load arrays (int32/uint32), built in chunks to keep memory low."""

    def __init__(self, path, chunk=1 << 21):
        z = np.load(path, mmap_mode=None)
        rec = z["rec"]
        self.phase_names = list(z["phase_names"])
        kind = (rec[:, 0] >> 28).astype(np.int8)
        marks = np.where(kind == 3)[0]
        self.phase_at = marks - np.arange(len(marks))  # kept-event index where phase k+1 starts
        self.n_events = int((kind != 3).sum())
        n_st = int(((kind == 1) | (kind == 2)).sum())
        n_ld = int((kind == 0).sum())
        self.st_ev = np.empty(n_st, np.int32)
        self.st_addr = np.empty(n_st, np.int32)
        self.st_val = np.empty(n_st, np.uint32)
        self.ld_ev = np.empty(n_ld, np.int32)
        self.ld_addr = np.empty(n_ld, np.int32)
        self.ld_val = np.empty(n_ld, np.uint32)
        si = li = ev0 = 0
        for c0 in range(0, len(rec), chunk):
            r = rec[c0:c0 + chunk]
            k = kind[c0:c0 + chunk]
            keep = k != 3
            r, k = r[keep], k[keep]
            word = (r[:, 0] & 0x0FFFFFFF).astype(np.int64)
            byte = word * 4
            rod = (byte >= RODATA[0]) & (byte < RODATA[1])
            addr = np.empty(len(word), np.int64)
            addr[~rod] = compact(word[~rod])
            addr[rod] = -1 - word[rod]  # rodata: never stored
            st = k != 0
            assert not (st & rod).any()
            ev = ev0 + np.arange(len(k))
            ns, nl = int(st.sum()), int((~st).sum())
            self.st_ev[si:si + ns], self.st_addr[si:si + ns], self.st_val[si:si + ns] = ev[st], addr[st], r[st, 1]
            self.ld_ev[li:li + nl], self.ld_addr[li:li + nl], self.ld_val[li:li + nl] = ev[~st], addr[~st], r[~st, 1]
            si, li, ev0 = si + ns, li + nl, ev0 + len(k)
        assert si == n_st and li == n_ld and ev0 == self.n_events
        self.n_st, self.n_ld = n_st, n_ld
        del rec, z, kind
