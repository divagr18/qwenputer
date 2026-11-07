"""How well int8 codes compress: per-row int8 of sampled real rows, zstd-19, by family."""
import json
import pickle
import sys

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import hub

sp = sys.argv[1]
H = pickle.load(open(sp + "/hdr.pkl", "rb"))["base"]
Z = zstd.ZstdCompressor(level=19)
out = {}
for f in ("mlp.gate_proj", "mlp.down_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj"):
    t = H[f"model.language_model.layers.9.{f}.weight"]
    n = t["shape"][0]
    rb = (t["b"] - t["a"]) // n
    raw = hub.read(t, (n // 2) * rb, (n // 2 + 32) * rb)
    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(32, -1).float()
    s = w.abs().amax(1, keepdim=True) / 127
    c = (w / s).round().clamp(-127, 127).to(torch.int8).numpy()
    out[f] = round(len(Z.compress(c.tobytes())) / c.size, 3)
print(out)
json.dump(out, open(sp + "/q8wire.json", "w"), indent=1)
