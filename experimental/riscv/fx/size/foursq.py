"""Can the ballast norm be exact at any hidden size, with few ballast dims?

emit.Layout needs m * B^2 / hidden to be a power of 4, so hidden is restricted to m * 4^k and m dims go
to ballast (62 today, 64 for fx). With unequal ballast values the requirement is sum(B_i^2) = hidden *
4^j, and Lagrange says four squares reach any integer. A value must be bf16: an 8-bit integer times a
power of 2. This finds such values for every hidden size in a range and checks the fp32 norm the stock
RMSNorm computes: the chip's scale c must come out exactly a power of 2.
"""
import json
import sys

import torch


def four(n: int):
    """i >= j >= k >= l in 0..255 with squares summing to n, or None."""
    for i in range(min(255, int(n**0.5)), 0, -1):
        for j in range(min(i, int((n - i * i) ** 0.5)), -1, -1):
            r = n - i * i - j * j
            for k in range(min(j, int(r**0.5)), -1, -1):
                s = r - k * k
                l = int(s**0.5)
                if l * l == s and l <= k:
                    return i, j, k, l
    return None


def check(h: int):
    """Ballast values for hidden h (B_i ~ 2^40), and whether the stock fp32 norm is exact."""
    for e in range(0, 8):
        v = four(h * 4**e)
        if v is None:
            continue
        x = torch.zeros(h, dtype=torch.bfloat16)
        b = 2.0 ** (40 - e)
        for i, u in enumerate(v):
            x[i] = u * b
        assert all(float(x[i]) == u * b for i, u in enumerate(v))
        f = x.float()
        rm = torch.rsqrt(f.pow(2).mean(-1) + 1e-6)
        return v, e, float(rm), float(torch.log2(rm))
    return None


res = {"sample": {}}
bad = []
for h in range(2600, 6000):
    r = check(h)
    if r is None or r[3] != round(r[3]):
        bad.append(h)
for h in (3961, 4021, 3072, 2816, 4811):
    res["sample"][h] = check(h)
res["range"] = [2600, 6000]
res["hidden_sizes_not_exact"] = len(bad)
res["first_bad"] = bad[:10]
print(json.dumps(res, indent=1))
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1] + "/foursq.json", "w"), indent=1)
