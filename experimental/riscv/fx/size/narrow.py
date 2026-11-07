"""Are rv-q4p2's Qwen tensors base Qwen's bytes plus padding? Sampled by range reads.

    python fx/size/narrow.py SCRATCH

For Qwen layers 0, 3 (full attention), 15 and 31, every tensor: read sampled rows from both repos and
compare the first 2560 columns (or rows) with base, then look at what the padding holds. If only the
gain tensors and the mute columns differ, a loader can use base's weights directly.
"""
import json
import pickle
import sys

from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import hub

sp = sys.argv[1]
H = pickle.load(open(sp + "/hdr.pkl", "rb"))
q, b = H["qp"], H["base"]
n_chip = 11
res = {}


def rows_of(t, rs):
    n = t["shape"][0]
    rb = (t["b"] - t["a"]) // n
    with ThreadPoolExecutor(8) as ex:
        raw = list(ex.map(lambda r: hub.read(t, r * rb, (r + 1) * rb), rs))
    return np.stack([np.frombuffer(x, {"BF16": "<u2", "F32": "<f4"}[t["dtype"]]) for x in raw])


for L in (4,):
    pre_b = f"model.language_model.layers.{L}."
    pre_q = f"model.layers.{L + n_chip}."
    for k in [k for k in b if k.startswith(pre_b)]:
        leaf = k[len(pre_b):]
        tb, tq = b[k], q[pre_q + leaf]
        sb, sq = tb["shape"], tq["shape"]
        if len(sb) == 1:
            rs = list(range(sb[0]))[:: max(1, sb[0] // 8)]
            a = np.array(rows_of(tb, rs)).ravel()
            c = np.array(rows_of(tq, rs)).ravel()
            pad = np.array([])
        elif sb == sq:
            rs = [0, sb[0] // 2, sb[0] - 1]
            a, c = rows_of(tb, rs), rows_of(tq, rs)
            pad = np.array([])
        elif sb[1] != sq[1]:  # columns widened
            rs = [0, sb[0] // 2, sb[0] - 1]
            a = rows_of(tb, rs)
            w = rows_of(tq, rs)
            c = w[:, : sb[1]]
            pad = w[:, sb[1]:]
        else:  # rows widened: compare rows inside base's range, then the padded rows
            rs = [0, sb[0] // 2, sb[0] - 1]
            a, c = rows_of(tb, rs), rows_of(tq, rs)
            pad = rows_of(tq, [sb[0], sq[0] - 1])
        same = a.dtype == c.dtype and bool((a == c).all())
        kind = res.setdefault(leaf.replace(".weight", ""), dict(same=0, diff=0, pad_nonzero_cols=set()))
        kind["same" if same else "diff"] += 1
        if pad.size and pad.ndim == 2:
            kind["pad_nonzero_cols"] |= {int(x) for x in np.nonzero((pad != 0).any(0))[0]}
out = {k: dict(same=v["same"], diff=v["diff"], pad_nonzero_cols=sorted(v["pad_nonzero_cols"])[:6],
               n_pad_nonzero=len(v["pad_nonzero_cols"])) for k, v in res.items()}
# the embedding: Qwen's rows, Qwen's columns
e_b, e_q = b["model.language_model.embed_tokens.weight"], q["model.embed_tokens.weight"]
rs = [0, 1000, 100000, 248000]
a, w = rows_of(e_b, rs), rows_of(e_q, rs)
out["embed_tokens"] = dict(same=int((a == w[:, :2560]).all()), diff=int(not (a == w[:, :2560]).all()),
                           chip_cols_nonzero_per_row=[int(x) for x in (w[:, 2560:] != 0).sum(1)])
for k, v in out.items():
    print(k, v, flush=True)
json.dump(out, open(sp + "/narrow_linear.json", "w"), indent=1)
