"""Two small direct tests.

1. Does the stock safetensors loader accept two names over one byte range, or holes? (Sharing
   tensors or a stored zero run depends on it; the docs only mention holes.)
2. What a Qwen layer's dense matmuls cost, one token, by width and by weight format, on this CPU.
   Ratios only: a GPU is memory-bound in a different way, so these are not GPU figures.
"""
import json
import os
import struct
import sys
import tempfile
import time

import torch
from safetensors import safe_open

DT = torch.bfloat16


def st_file(path, entries, payload):
    head = json.dumps({"__metadata__": {"format": "pt"}, **entries}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head + payload)


def try_load(path):
    try:
        with safe_open(path, "pt") as f:
            return "accepted", {k: f.get_tensor(k).shape[0] for k in f.keys()}
    except Exception as e:
        return "rejected", str(e)[:120]


res = {}
d = tempfile.mkdtemp()
e = lambda a, b: {"dtype": "BF16", "shape": [(b - a) // 2], "data_offsets": [a, b]}
pay = bytes(64)
st_file(d + "/a.st", {"x": e(0, 32), "y": e(0, 32), "z": e(32, 64)}, pay)
res["alias_two_names_one_range"] = try_load(d + "/a.st")
st_file(d + "/b.st", {"x": e(0, 32), "z": e(48, 64)}, pay)
res["hole_between_tensors"] = try_load(d + "/b.st")
st_file(d + "/c.st", {"x": e(0, 48), "y": e(16, 64)}, pay)
res["overlap"] = try_load(d + "/c.st")
print(res, flush=True)

R = dict(qkv=8192, z=4096, gate=9216, up=9216)  # input side: rows x hidden
O = dict(out=4096, down=9216)  # output side: hidden x cols
torch.set_num_threads(4)


def layer(h, zero_from=None):
    ws = {}
    for n, r in R.items():
        w = torch.randn(r, h, dtype=torch.float32) * 0.02
        if zero_from:
            w[:, zero_from:] = 0
        ws[n] = ("in", w.to(DT))
    for n, c in O.items():
        w = torch.randn(h, c, dtype=torch.float32) * 0.02
        if zero_from:
            w[zero_from:] = 0
        ws[n] = ("out", w.to(DT))
    return ws


def fwd(ws, x, q, get):
    out = 0.0
    for n, (side, _) in ws.items():
        w = get(n)
        out = out + float((x @ w.T).float().abs().sum() if side == "in" else (q[n] @ w.T).float().abs().sum())
    return out


def bench(ws, h, get, reps=12):
    x = torch.randn(1, h).to(DT)
    q = {n: torch.randn(1, ws[n][1].shape[1]).to(DT) for n in O}
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fwd(ws, x, q, get)
        t.append(time.perf_counter() - t0)
    return sorted(t)[len(t) // 2] * 1e3


base = layer(2560)
pad = layer(3968, 2560)
wide4096 = layer(4096, 2560)
ms = {"bf16 width 2560": bench(base, 2560, lambda n: base[n][1]),
      "bf16 padded to 3968": bench(pad, 3968, lambda n: pad[n][1]),
      "bf16 padded to 4096": bench(wide4096, 4096, lambda n: wide4096[n][1])}
del pad, wide4096
q8 = {}
for n, (_, w) in base.items():
    f = w.float()
    s = f.abs().amax(1, keepdim=True) / 127
    q8[n] = ((f / s).round().to(torch.int8), s.to(DT))


def d8(n):
    i, s = q8[n]
    return i.to(DT) * s  # decode on every forward: what a stock int8 weight-only Linear does


ms["int8 decoded per forward, width 2560"] = bench(base, 2560, d8)
q4 = {}
for n, (_, w) in base.items():
    f = w.float()
    r, c = f.shape
    g = 32
    y = f.view(r, -1, g) if c % g == 0 else None
    if y is None:
        continue
    s = y.abs().amax(2, keepdim=True) / 7
    i = (y / s).round().clamp(-7, 7).to(torch.int8).view(r, -1) + 8
    q4[n] = ((i[:, 0::2] << 4 | i[:, 1::2]).to(torch.uint8), s.to(DT))


def d4(n):
    if n not in q4:
        return base[n][1]
    p, s = q4[n]
    i = torch.stack([(p >> 4).to(torch.int8) - 8, (p & 15).to(torch.int8) - 8], 2).view(p.shape[0], -1)
    return (i.view(p.shape[0], -1, 32).to(DT) * s).view(p.shape[0], -1)


ms["int4 decoded per forward, width 2560"] = bench(base, 2560, d4, 6)
res["ms_per_token_one_layer_cpu_4_threads"] = {k: round(v, 2) for k, v in ms.items()}
b = ms["bf16 width 2560"]
res["ratio_to_bf16_2560"] = {k: round(v / b, 2) for k, v in ms.items()}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(sys.argv[1] + "/bench.json", "w"), indent=1)
