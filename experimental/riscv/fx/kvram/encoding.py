"""KV-cache RAM: the key/query/value encoding inside one Qwen3.5 full-attention head.

Qwen3.5-4B full attention (config.json + transformers modeling_qwen3_5.py):
- head_dim 256, 16 q heads, 4 kv heads (GQA 4:1), scaling 256**-0.5 = 1/16;
- q_proj gives q and an output gate per head; o = attn * sigmoid(gate);
- q_norm / k_norm: RMSNorm over the head's 256 dims, out = bf16(x / rms(x) * (1 + w)), w shared by
  all heads of the layer;
- RoPE (partial_rotary_factor 0.25) rotates dims 0..63 only (pairs i, i+32); dims 64..255 pass.

The encoding (all in dims 64..255, so RoPE never touches it; dims 0..63 carry the norm ballast
and are zeroed by the norm weights, so q and k are exactly 0 there before and after RoPE):
- address: D one-hot digit groups (bases e.g. 64, 32, 32, 32 = 21 bits). Key: one-hot. Query:
  +alpha at its digit, -alpha at the other digits of the group. A group adds +alpha if the digits
  are equal and -alpha if not.
- recency: n_rec dims. Key: +-1 per bit of the store counter t. Query: g * 2**i on bit i. That adds
  R(t) = g * (2t - (2**n_rec - 1)): consecutive stores differ by 2g.
- W: n_w dims, key 1 at every store, query W_j; sum W puts matches above 0 and mismatches below.
- non-store positions: key = 0 (score exactly 0), value = 0.

Score in q.k units (nats = q.k / 16):
  hit at t        S = W + G + R(t)                 (G = D * alpha)
  miss in h groups  S = W + G - 2 * alpha * h + R(t)
  non-store       S = 0
alpha >= g * (2**n_rec - 1) + M, W chosen so hits >= +M and every miss <= -M (M = 2g).
All products are multiples of g, so every partial sum is an exact fp32 integer multiple of g while
its magnitude stays below 2**24 * g: the bound is G + g * 2**n_rec < 2**24 * g.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

E = 256
ROT = 64  # rotated dims (partial_rotary_factor 0.25)
SCALE = E ** -0.5


def bf16(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).double())


def bf16_ceil(x: float) -> float:
    """Smallest bf16 value >= x (x > 0)."""
    y = bf16(x)
    while y < x:
        y = float(torch.nextafter(torch.tensor(y, dtype=torch.bfloat16), torch.tensor(math.inf, dtype=torch.bfloat16)).double())
    return y


def bf16_split(x: float, parts: int) -> list[float]:
    """x as a sum of `parts` bf16 values (exact), or raise."""
    out, r = [], x
    for _ in range(parts):
        v = bf16(r)
        out.append(v)
        r -= v
    if r != 0:
        raise ValueError(f"{x} is not a sum of {parts} bf16 values")
    return out


def squares(n: int, k: int = 4) -> list[int]:
    """n as a sum of at most k integer squares (greedy with backtracking)."""
    def go(n, k):
        if n == 0:
            return []
        if k == 0:
            return None
        for a in range(int(math.isqrt(n)), 0, -1):
            r = go(n - a * a, k - 1)
            if r is not None:
                return [a] + r
        return None
    r = go(n, k)
    assert r is not None
    return r


@dataclass
class Enc:
    bases: tuple = (64, 32, 32, 32)  # address digit groups, least significant first
    n_rec: int = 21  # recency (store counter) bits
    g: float = 1024.0  # recency LSB weight in q.k units (2g = 128 nats between consecutive stores)
    n_w: int = 3
    epoch_bits: int = 0  # optional epoch number as extra +-1 address-like bits (compaction)
    # derived
    alpha: float = field(init=False)
    W: float = field(init=False)
    W_parts: list = field(init=False)

    def __post_init__(self):
        g, n = self.g, self.n_rec
        self.M = 2 * g
        self.D = len(self.bases)
        self.addr_bits = sum(int(math.log2(b)) for b in self.bases)
        for b in self.bases:
            assert b & (b - 1) == 0
        # layout
        d = ROT
        self.addr_off = []
        for b in self.bases:
            self.addr_off.append(d)
            d += b
        self.ep_off = d
        d += self.epoch_bits
        self.rec_off = d
        d += n
        self.w_off = d
        d += self.n_w
        self.used = d
        assert d <= E, f"layout needs {d} dims"
        # weights
        self.rspan = g * (2 ** n - 1)  # R in [-rspan, rspan]
        self.alpha = bf16_ceil(self.rspan + self.M)
        # epoch bits are +-1 like the address digits' +-alpha queries: a flipped bit drops the score
        # by 2 * ep_alpha, which must cover 2 * (rspan + M), so ep_alpha = alpha
        self.ep_alpha = self.alpha if self.epoch_bits else 0.0
        self.G = self.D * self.alpha + self.epoch_bits * self.ep_alpha
        min_gap = self.alpha  # smallest single-mismatch drop is 2 * alpha
        lo = self.M - self.G + self.rspan
        hi = -self.M - self.G + 2 * min_gap - self.rspan
        assert lo <= hi, (lo, hi)
        mid = (lo + hi) / 2
        # W: a multiple of g near the middle that splits into n_w bf16 parts
        best = None
        for k in range(0, 1 << 16):
            for s in (1, -1):
                cand = (round(mid / g) + s * k) * g
                if lo <= cand <= hi:
                    try:
                        best = (cand, bf16_split(cand, self.n_w))
                        break
                    except ValueError:
                        pass
            if best:
                break
        assert best, "no bf16-splittable W"
        self.W, self.W_parts = best
        # ballast (sum of squares to 256 so rms == 1 exactly)
        kq = self.D + n + self.n_w + self.epoch_bits
        self.k_ballast = squares(E - kq)
        qq = sum(self.bases) + n + self.n_w + self.epoch_bits
        self.q_ballast = squares(E - qq)
        assert len(self.k_ballast) + len(self.q_ballast) <= ROT
        # norm multipliers (1 + w): q targets; k is 1 on used dims
        self.q_mult = np.zeros(E)
        for gi, b in enumerate(self.bases):
            self.q_mult[self.addr_off[gi]:self.addr_off[gi] + b] = self.alpha
        self.q_mult[self.ep_off:self.ep_off + self.epoch_bits] = self.ep_alpha
        for i in range(n):
            self.q_mult[self.rec_off + i] = g * 2 ** i
        self.q_mult[self.w_off:self.w_off + self.n_w] = self.W_parts
        self.k_mult = np.zeros(E)
        self.k_mult[ROT:self.used] = 1.0

    # ---------------------------------------------------------------- vectors before the norms
    def digits(self, a):
        a = np.asarray(a, dtype=np.int64)
        out, s = [], 0
        for b in self.bases:
            out.append((a >> s) & (b - 1))
            s += int(math.log2(b))
        return out

    def key_in(self, addr, t, epoch=0):
        """k_proj output (pre k_norm) for stores (addr, t): float32 [N, 256]; sum of squares 256."""
        addr = np.asarray(addr, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        N = addr.shape[0]
        k = np.zeros((N, E), np.float32)
        for gi, dg in enumerate(self.digits(addr)):
            k[np.arange(N), self.addr_off[gi] + dg] = 1
        for i in range(self.epoch_bits):
            k[:, self.ep_off + i] = 2 * ((np.asarray(epoch) >> i) & 1) - 1
        for i in range(self.n_rec):
            k[:, self.rec_off + i] = 2 * ((t >> i) & 1) - 1
        k[:, self.w_off:self.w_off + self.n_w] = 1
        for j, v in enumerate(self.k_ballast):
            k[:, j] = v
        return k

    def query_in(self, addr, epoch=0):
        addr = np.asarray(addr, dtype=np.int64)
        N = addr.shape[0]
        q = np.zeros((N, E), np.float32)
        for gi, dg in enumerate(self.digits(addr)):
            b = self.bases[gi]
            q[:, self.addr_off[gi]:self.addr_off[gi] + b] = -1
            q[np.arange(N), self.addr_off[gi] + dg] = 1
        for i in range(self.epoch_bits):
            q[:, self.ep_off + i] = 2 * ((np.asarray(epoch) >> i) & 1) - 1
        q[:, self.rec_off:self.rec_off + self.n_rec] = 1
        q[:, self.w_off:self.w_off + self.n_w] = 1
        for j, v in enumerate(self.q_ballast):
            q[:, 32 + j] = v
        return q

    # ---------------------------------------------------------------- the norm weights (zero-centred)
    def norm_weights(self):
        """Qwen3_5RMSNorm stores w with out = x_hat * (1 + w). Pick bf16 w so that (1 + w) is within
        2^-9 relative of the target; the norm's bf16 output cast then snaps to the exact target."""
        qw = torch.tensor(self.q_mult - 1.0, dtype=torch.float64).to(torch.bfloat16)
        kw = torch.tensor(self.k_mult - 1.0, dtype=torch.float64).to(torch.bfloat16)
        return qw, kw

    # ---------------------------------------------------------------- exact scores (python ints)
    def score_int(self, qaddr, kaddr, t, qep=0, kep=0):
        """Exact q.k in q.k units for a store entry (int64 arrays; values fit in int64)."""
        qd, kd = self.digits(qaddr), self.digits(kaddr)
        s = np.full(np.broadcast(np.asarray(qaddr), np.asarray(kaddr)).shape, int(self.W), dtype=np.int64)
        a = int(self.alpha)
        for x, y in zip(qd, kd):
            s += np.where(x == y, a, -a)
        for i in range(self.epoch_bits):
            s += np.where(((np.asarray(qep) >> i) & 1) == ((np.asarray(kep) >> i) & 1), int(self.ep_alpha), -int(self.ep_alpha))
        s += int(self.g) * (2 * np.asarray(t, dtype=np.int64) - (2 ** self.n_rec - 1))
        return s

    def exact_bound(self):
        """Largest partial-sum magnitude for a hit, a same-address older store or a 1-group miss,
        in units of g (exact while < 2**24)."""
        pos = self.G + self.rspan
        neg = abs(self.W) + self.alpha + self.rspan
        return max(pos, neg) / self.g

    def summary(self):
        return {
            "bases": list(self.bases), "addr_bits": self.addr_bits, "n_rec": self.n_rec, "epoch_bits": self.epoch_bits,
            "g_qk": self.g, "step_margin_nats": 2 * self.g * SCALE, "alpha_qk": self.alpha, "W_qk": self.W,
            "W_parts": self.W_parts, "hit_min_nats": (self.W + self.G - self.rspan) * SCALE,
            "miss_max_nats": (self.W + self.G - 2 * self.alpha + self.rspan) * SCALE,
            "dims_used": self.used - ROT, "exact_bound_units_of_g": self.exact_bound(),
            "exact_bound_ok": self.exact_bound() < 2 ** 24,
        }


if __name__ == "__main__":
    import json
    for n in (16, 20, 21, 22, 23, 24):
        e = Enc(n_rec=n)
        print(json.dumps(e.summary()))
