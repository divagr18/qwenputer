"""Read qstream's checkpoint: how far each scheme's hidden state has moved from base so far."""
import json
import pickle
import sys

import torch

sp = sys.argv[1]
done, st = pickle.load(open(sp + "/qstream.pkl", "rb"))
out = {}
for n, xs in st.items():
    e = b = 0.0
    for x, y in zip(xs, st["base"]):
        e += float((x.float() - y.float()).pow(2).sum())
        b += float(y.float().pow(2).sum())
    out[n] = (e / b) ** 0.5
print("layers done", done)
for n, v in sorted(out.items(), key=lambda kv: kv[1]):
    print(f"{n:14s} rel L2 {v:.4f}")
json.dump({"layers_done": done, "rel_l2": out}, open(sp + "/qpartial.json", "w"), indent=1)
