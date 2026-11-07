"""Sample real tensors by range reads: nonzeros, distinct values, compressibility.

    python fx/size/sample.py SCRATCH

Measures, per family, on evenly spaced row windows of the published rv-q4p2 checkpoint and of base
Qwen3.5-4B: nonzero fraction, distinct values, all-zero 4 KiB pages (what a sparse file saves), and
lossless ratios: zstd-3 on the raw bytes, zstd-19 on the raw bytes, zstd-19 on split hi/lo byte
streams (ZipNN's grouping), and the order-0 entropy of each byte stream.
"""
import json
import math
import pickle
import sys
from collections import defaultdict

import numpy as np
import zstandard as zstd

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
import hub

sp = sys.argv[1]
H = pickle.load(open(sp + "/hdr.pkl", "rb"))
Z3 = zstd.ZstdCompressor(level=3)
Z19 = zstd.ZstdCompressor(level=19)


def H0(b):
    c = np.bincount(np.frombuffer(b, np.uint8), minlength=256).astype(float)
    c = c[c > 0] / c.sum()
    return float(-(c * np.log2(c)).sum())


def stats(raw):
    a = np.frombuffer(raw, "<u2")
    lo, hi = raw[0::2], raw[1::2]
    pages = [raw[i:i + 4096] for i in range(0, len(raw) - 4095, 4096)]
    zp = sum(1 for p in pages if not any(p)) / max(1, len(pages))
    return dict(n=len(a), nnz=int((a != 0).sum()), distinct=int(len(np.unique(a))), zero_pages=zp,
                z3=len(Z3.compress(raw)) / len(raw), z19=len(Z19.compress(raw)) / len(raw),
                split=(len(Z19.compress(lo)) + len(Z19.compress(hi))) / len(raw),
                h0_split_bits=(H0(lo) + H0(hi)) / 2 / 8)


def sample(t, windows=4, rows=8):
    """Evenly spaced windows of whole rows."""
    n = t["shape"][0]
    rowb = (t["b"] - t["a"]) // n
    rows = min(rows, n)
    out = []
    for w in range(windows):
        r = (n - rows) * w // max(1, windows - 1)
        out.append(hub.read(t, r * rowb, (r + rows) * rowb))
    return b"".join(out)


def run(name, keys):
    agg = defaultdict(list)
    for fam, k in keys:
        t = H[name][k]
        s = stats(sample(t))
        agg[fam].append(s)
    res = {}
    for fam, v in agg.items():
        res[fam] = {f: round(float(np.mean([x[f] for x in v])), 5) for f in v[0]}
        print(name, fam, res[fam], flush=True)
    return res


out = {}
fams = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
chip_keys = [(f"chip {f}", f"model.layers.{L}.{f}.weight") for L in (1, 3, 6, 7, 10) for f in fams]
qwen_keys = [(f"qwen {f}", f"model.layers.{L}.{f}.weight") for L in (11, 20, 41) for f in fams]
emb = H["qp"]["model.embed_tokens.weight"]
if "--base-only" not in sys.argv:
    out["qp_chip"] = run("qp", chip_keys)
    out["qp_qwen"] = run("qp", qwen_keys)
base_keys = [(f"base {f}", f"model.language_model.layers.{L}.{f}.weight") for L in (9,) for f in fams]
base_keys += [("base embed", "model.language_model.embed_tokens.weight")]
out["base"] = run("base", base_keys)

# the embedding by region: Qwen rows and chip rows, Qwen columns and chip columns
R = {}
rowb = 2 * 3968
for tag, r0 in (("qwen rows", 1000), ("qwen rows hi", 200000), ("chip rows V", 248320 + 40000),
                ("chip rows J", 248320 + 120000)):
    raw = hub.read(emb, r0 * rowb, (r0 + 16) * rowb)
    a = np.frombuffer(raw, "<u2").reshape(16, 3968)
    R[tag] = dict(nnz_qwen_cols=float((a[:, :2560] != 0).mean()), nnz_chip_cols=float((a[:, 2560:] != 0).mean()),
                  nnz_per_row=float((a != 0).sum(1).mean()), z19=len(Z19.compress(raw)) / len(raw))
    print(tag, R[tag], flush=True)
out["embed_regions"] = R

# padding check: are the widened Qwen columns zero except the mute column?
P = {}
for f, cols in (("linear_attn.in_proj_z", 3968), ("linear_attn.in_proj_b", 3968), ("mlp.up_proj", 3968)):
    t = H["qp"][f"model.layers.12.{f}.weight"]
    raw = sample(t, 3, 16)
    a = np.frombuffer(raw, "<u2").reshape(-1, cols)
    nz = np.nonzero((a[:, 2560:] != 0).any(0))[0] + 2560
    P[f] = dict(chip_cols_nonzero=[int(x) for x in nz[:8]], n=len(nz))
    print("padding", f, P[f], flush=True)
out["padding"] = P
json.dump(out, open(sp + ("/sample_base.json" if "--base-only" in sys.argv else "/sample.json"), "w"), indent=1)
