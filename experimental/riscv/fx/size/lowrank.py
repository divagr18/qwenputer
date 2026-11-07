"""Would low-rank Qwen layers save anything? Singular values of real tensors, by range reads.

    python fx/size/lowrank.py SCRATCH

Reads one MLP tensor (gate_proj, 9216 x 2560) and the z projection of a mid layer. A rank-r factorised
W = U V stores r * (rows + cols) numbers, against rows * cols: this reports the rank each energy level
needs and what that factorisation would save, keeping bf16.
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from concurrent.futures import ThreadPoolExecutor

import hub
import torch

sp = sys.argv[1]
H = pickle.load(open(sp + "/hdr.pkl", "rb"))["base"]
res = {}
for name in ("model.language_model.layers.9.mlp.gate_proj.weight",
             "model.language_model.layers.9.linear_attn.in_proj_z.weight"):
    t = H[name]
    n = t["b"] - t["a"]
    step = 8 << 20
    with ThreadPoolExecutor(4) as ex:
        parts = list(ex.map(lambda a: hub.read(t, a, min(n, a + step)), range(0, n, step)))
    w = torch.frombuffer(bytearray(b"".join(parts)), dtype=torch.bfloat16).reshape(t["shape"]).float()
    s = torch.linalg.svdvals(w.double()).numpy()
    e = np.cumsum(s**2) / (s**2).sum()
    r, c = w.shape
    row = {"shape": [r, c]}
    for lv in (0.5, 0.9, 0.99, 0.999):
        k = int(np.searchsorted(e, lv)) + 1
        row[f"rank_{lv}"] = k
        row[f"size_ratio_{lv}"] = round(k * (r + c) / (r * c), 3)
    row["top_singular_share"] = round(float(s[0] ** 2 / (s**2).sum()), 4)
    res[name.split("layers.")[1]] = row
    print(name, row, flush=True)
json.dump(res, open(sp + "/lowrank.json", "w"), indent=1)
