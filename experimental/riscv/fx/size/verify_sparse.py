"""Is the sparse form the published checkpoint, byte for byte?

    PYTHONPATH=<qwenputer-exec riscv tree> python fx/size/verify_sparse.py TOKDIR SCRATCH q4p2 [--all]
    PYTHONPATH=<qwenputer-mem riscv tree>  python fx/size/verify_sparse.py TOKDIR SCRATCH fx

q4p2: emit the chip layers sparse (rv-q4p2's circuit, hidden 3968, m 62), expand each tensor chunk by
chunk and compare it with the bytes on the Hub: nothing is stored, nothing is held but one chunk.
Both modes also write the sparse chip to disk and report its size (raw and compressed).
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sparse as SP
import zstandard as zstd

from chipformer.q35.emit import Layout, allocate
from chipformer.q35.merge import pick_hidden
from chipformer.rv import bridge as BR
from chipformer.rv import core
from chipformer.rv import spec as S


def bf16_bits(v: np.ndarray) -> np.ndarray:
    u = np.asarray(v, np.float32).view(np.uint32)
    assert (u & 0xFFFF).sum() == 0
    return (u >> 16).astype(np.uint16)


def to_arrays(sd):
    """name -> (shape, fill, idx (n, ndim), values) in deterministic order."""
    out = {}
    for k, t in sd.items():
        items = sorted(kv for kv in t.d.items() if kv[1] != t.fill)
        idx = np.array([i for i, _ in items], dtype=np.int64).reshape(len(items), len(t.shape))
        val = np.array([v for _, v in items], dtype=np.float32)
        out[k] = (t.shape, t.fill, idx, val)
    return out


def store(arrs, path):
    """The sparse file: for each tensor, u16 row and col delta streams and a u16 index into one
    palette of bf16 values. Raw = what a loader would map; compressed = the same, zstd --long."""
    pal = np.unique(np.concatenate([v for *_, v in arrs.values() if len(v)]))
    pal_b = bf16_bits(pal)
    blob = [pal_b.tobytes()]
    meta = {}
    for k, (shape, fill, idx, val) in arrs.items():
        if not len(val):
            meta[k] = dict(shape=shape, fill=fill, n=0)
            continue
        vi = np.searchsorted(pal, val).astype(np.uint16)
        r = idx[:, 0].astype(np.uint16)
        c = idx[:, 1 if idx.shape[1] == 2 else 2].astype(np.uint16)
        blob += [r.tobytes(), c.tobytes(), vi.tobytes()]
        meta[k] = dict(shape=shape, fill=fill, n=len(val))
    raw = b"".join(blob)
    m = json.dumps(meta).encode()
    p = zstd.ZstdCompressionParameters.from_level(19, window_log=27, enable_ldm=True)
    z = zstd.ZstdCompressor(compression_params=p).compress(raw)
    zm = zstd.ZstdCompressor(level=19).compress(m)
    with open(path, "wb") as f:
        f.write(zm + z)
    return dict(palette=len(pal), raw_bytes=len(raw) + len(m), zstd_bytes=len(z) + len(zm),
                terms=int(sum(len(v) for *_, v in arrs.values())))


def expand(a, r0, r1):
    """Rows r0..r1 of a sparse tensor, as bf16 bytes."""
    shape, fill, idx, val = a
    s = (r1 - r0, *shape[1:])
    out = np.full(s, fill, np.float32)
    if len(val):
        lo, hi = np.searchsorted(idx[:, 0], [r0, r1])
        i = idx[lo:hi]
        out[(i[:, 0] - r0, *[i[:, d] for d in range(1, i.shape[1])])] = val[lo:hi]
    return bf16_bits(out.ravel()).tobytes()


def main():
    import hub

    tokdir, scratch, which = sys.argv[1:4]
    T = BR.Text(tokdir + "/tokenizer.json")
    if which == "fx":
        from chipformer.rv import fxrun

        image = tuple(sorted(set(range(0xD9270 >> S.WIN_BITS, (0x1A00B0 >> S.WIN_BITS) + 1))
                             | set(range(100, 119))))[:35]
        c = core.build(core.T4B, T, cfg=fxrun.fx_cfg(image))
    else:
        c = core.build(core.T4B, T)
    keep = set(c.keep)
    al = allocate(c, None, keep, core.T4B.intermediate)
    h, m = (3968, 62) if which == "q4p2" else pick_hidden(2560, 1 + al.n_slots)
    print(c.stats(), "slots", al.n_slots, "hidden", h, "m", m, flush=True)
    lay = Layout(hidden=h, off=2560, m=m, slots=al.slots)
    tc = dict(hidden_size=h, linear_num_key_heads=16, linear_num_value_heads=32,
              intermediate_size=9216, linear_conv_kernel_dim=4)
    arrs = to_arrays(SP.emit_sparse(c, lay, tc, al, keep))
    res = {"store": store(arrs, f"{scratch}/chip_{which}.bin")}
    print(res, flush=True)
    if which == "fx":
        json.dump(res, open(f"{scratch}/verify_{which}.json", "w"), indent=1)
        return
    H = pickle.load(open(scratch + "/hdr.pkl", "rb"))["qp"]
    names = sorted(arrs)
    assert all(k in H for k in names), [k for k in names if k not in H][:5]
    t0 = time.time()
    ok = bad = 0
    first_bad = []
    donef = f"{scratch}/verify_{which}.done.json"
    done = json.load(open(donef)) if os.path.exists(donef) else {}
    names = [k for k in names if k not in done]
    print("already verified", len(done), "left", len(names), flush=True)

    def one(k):
        t = H[k]
        a = arrs[k]
        n = t["shape"][0]
        rowb = (t["b"] - t["a"]) // n
        step = max(1, (8 << 20) // rowb)
        same = True
        where = None
        for r in range(0, n, step):
            r1 = min(n, r + step)
            hub_b = hub.read(t, r * rowb, r1 * rowb)
            mine = expand(a, r, r1)
            if hub_b != mine:
                same = False
                if where is None:
                    d = np.nonzero(np.frombuffer(hub_b, "<u2") != np.frombuffer(mine, "<u2"))[0]
                    where = (r, len(d), int(d[0]))
        return k, same, t["b"] - t["a"], where

    total = 0
    with ThreadPoolExecutor(3) as ex:
        for k, same, nb, where in ex.map(one, names):
            total += nb
            done[k] = dict(same=same, bytes=nb)
            with open(donef + ".tmp", "w") as f:
                json.dump(done, f)
            os.replace(donef + ".tmp", donef)
            if same:
                ok += 1
            else:
                bad += 1
                first_bad.append((k, where))
                print("DIFF", k, where, flush=True)
            if (ok + bad) % 14 == 0:
                print(ok, bad, total / 1e9, round(time.time() - t0), flush=True)
    res.update(tensors_identical=sum(v["same"] for v in done.values()),
               tensors_different=sum(not v["same"] for v in done.values()),
               bytes_compared=sum(v["bytes"] for v in done.values()), first_bad=first_bad[:20])
    print(res, flush=True)
    json.dump(res, open(f"{scratch}/verify_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
