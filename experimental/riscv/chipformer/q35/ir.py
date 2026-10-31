"""Circuit graph for the Qwen3.5 target: SwiGLU units and Gated DeltaNet heads.

Like ``chipformer.dsl.graph``, every value is a scalar dim of the residual stream and linear
forms (``Expr``) are free. The node types are the two sublayers of a Qwen3.5 linear-attention
layer, described closely enough that the emitter only has to scale weights:

``FFN``   a sum of units ``coef * a * relu(b)``, emitted as ``silu(K b) * (a / K)``. silu equals
          relu on every value the circuit produces: 0, >= 18 (sigmoid rounds to 1) or <= -104
          (sigmoid underflows to 0).
``Head``  one Gated DeltaNet key head with its value head(s), channel by channel: q/k/v
          channels are linear forms with conv taps over the last 4 tokens followed by silu;
          keys must be one-hot (an exact overwrite of one state row) or zero, queries
          multi-hot or zero; ``a`` picks the per-token decay (keep / halve / clear); ``z``
          gates each output column. Rows carry a ballast in their ``ref`` columns that
          dominates the per-head output RMSNorm, so reads come back exactly rescaled.

Values live in IR units: a residual value x reaches every sublayer as exactly x * 2^-r because
chip tokens carry a large ballast (see spike.py), and the emitter folds 2^r into the weights.
The constant ONE is that ballast. On text tokens every chip dim, ONE included, is 0, so every
node outputs exactly 0 there.

Tokens are ints: chip tokens, text classes, and (for the object store) a text class with a Qwen
token id above it (``qid << 16 | cls``). Token features see ``t & 0xFFFF`` unless declared
``full``, so only features that read Qwen's own token id (the store's id code) can tell two text
tokens of one class apart.

Exactness is checked numerically: ``evaluate`` runs a token sequence through the graph and
asserts that every intermediate the hardware materializes is a bf16 value, that silu inputs are
in its exact range, and that keys, queries and decays have the allowed shapes.
"""

from dataclasses import dataclass, field

import numpy as np

from chipformer.dsl.graph import ONE, Expr, Num, as_expr

K = 128  # silu gate scale: silu(K n) = K relu(n) for integers n
A_REF = 2**24  # ballast value of a state row (two columns)
A_BIT = 2**7  # value of a set bit in a state row: (A_BIT / A_REF)^2 * 126 is absorbed
REF_COLS = (0, 1)
DECAY = {0.0: 1.0, 110.0: 0.5, 238.0: 0.0}  # a -> exp(g) with dt_bias = -110, A_log = 0
HD = 128  # key/value head dim
TOKEN_MASK = 0xFFFF  # the part of a token that ordinary features see (see above)


@dataclass
class Dim:
    name: str
    kind: str  # "one" | "token" | "ffn" | "head"
    level: int


@dataclass
class FFN:
    name: str
    terms: list[tuple[Num, Expr, Expr]]  # coef * a * relu(b)
    out: int
    level: int
    into: bool = False  # out is an existing dim that this node adds to


@dataclass
class Chan:
    expr: Expr
    taps: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # x_t, x_{t-1}, ...


@dataclass
class VHead:
    """A further value head of a key head (Qwen3.5-4B has two per key head): the same queries
    and keys, its own values, decay, write gate and output gates, so it holds a second table
    with the same rows."""

    v: dict[int, Chan]
    a: Expr
    z: dict[int, Expr]
    out: list[tuple[int, int, float]]
    beta: Expr | None = None  # write gate (0/1); None: always 1
    into: bool = False  # outputs add onto existing dims


@dataclass
class Head:
    name: str
    q: dict[int, Chan]  # key-dim channel -> query
    k: dict[int, Chan]
    v: dict[int, Chan]  # value column -> value (columns REF_COLS are the ballast)
    a: Expr  # decay selector, see DECAY
    z: dict[int, Expr]  # value column -> output gate silu(z)
    out: list[tuple[int, int, float]]  # (value column, residual dim, weight)
    level: int
    n_cols: int = HD
    tiny: float = 0.0  # outputs below this may round (the counter's 2^-j for j >> JMAX)
    beta: Expr | None = None  # write gate of the first value head (0/1); None: always 1
    into: bool = False  # outputs add onto existing dims (several heads summed into one dim)
    vx: list[VHead] = field(default_factory=list)  # further value heads of this key head
    # initial state (a snapshot the host injects): per value head, row -> state row. "bit" rows
    # hold 0 / A_BIT data; "byte" rows (read-only tables) hold integers 0..255 and must be read
    # with a 2-hot query (see kernel_check); "decay" rows hold A_BIT * 2^-n (a halving counter)
    init: list | None = None
    levels: str = "bit"

    def vheads(self) -> list[VHead]:
        return [VHead(self.v, self.a, self.z, self.out, self.beta, self.into), *self.vx]


@dataclass
class Target:
    """Per-layer capacity of the host model."""

    key_heads: int = 16
    value_heads: int = 16
    intermediate: int = 3584

    @property
    def cols(self) -> int:
        return HD  # value columns a head uses (one value head; the output norm is per head)


@dataclass
class Graph:
    target: Target = field(default_factory=Target)

    def __post_init__(self) -> None:
        self.dims: list[Dim] = [Dim("one", "one", 0)]
        self.token_fns: dict[int, object] = {}
        self.full_fns: set[int] = set()  # features that see the whole token (TOKEN_MASK)
        self.nodes: list[FFN | Head] = []
        self.one = Expr({ONE: 1})
        self.units: dict[int, int] = {}  # sublayer -> FFN units used
        self.heads: dict[int, int] = {}  # sublayer -> key heads used

    # --- dims -----------------------------------------------------------------

    def token_feature(self, name: str, fn, full: bool = False) -> Expr:
        d = self._dim(name, "token", 0)
        self.token_fns[d] = fn
        if full:
            self.full_fns.add(d)
        return Expr({d: 1})

    def feature_value(self, d: int, t: int):
        fn = self.token_fns[d]
        return fn(t) if d in self.full_fns else fn(t & TOKEN_MASK)

    def _dim(self, name: str, kind: str, level: int) -> int:
        self.dims.append(Dim(name, kind, level))
        return len(self.dims) - 1

    def level(self, *exprs: Expr) -> int:
        return max((self.dims[d].level for e in exprs for d in e.terms), default=0)

    # --- placement ------------------------------------------------------------

    def _place(self, lvl: int, odd: bool, need: int, used: dict[int, int], cap: int) -> int:
        sub = lvl + 1
        while (sub % 2 == 1) != odd or used.get(sub, 0) + need > cap:
            sub += 1
        used[sub] = used.get(sub, 0) + need
        return sub

    # --- nodes ----------------------------------------------------------------

    def ffn(self, terms, name: str = "ffn", into: Expr | None = None, min_level: int = 0) -> Expr:
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c != 0]
        lvl = max(self.level(*(e for _, a, b in terms for e in (a, b))), min_level)
        sub = self._place(lvl, False, len(terms), self.units, self.target.intermediate)
        if into is not None:
            (d,) = into.terms
            self.dims[d].level = max(self.dims[d].level, sub)
        else:
            d = self._dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, d, sub, into is not None))
        self.nodes[-1].min_level = min_level
        return Expr({d: 1})

    def head(self, name: str, q: dict, k: dict, v: dict, a: "Expr | Num", z: dict,
             outs: list[tuple[int, float]], min_level: int = 0, tiny: float = 0.0,
             beta=None, into: list | None = None, vx: list | None = None) -> list[Expr]:
        """Add a head; ``outs`` lists (value column, weight) pairs, one new dim each (or, with
        ``into``, onto these existing dims). ``vx``: further value heads, dicts with keys v, a,
        z, outs and optionally beta / into, sharing this head's queries and keys; their output
        dims follow the first value head's in the returned list."""
        def chans(ch):
            return {i: c if isinstance(c, Chan) else Chan(as_expr(c)) for i, c in ch.items()}

        q, k, v = chans(q), chans(k), chans(v)
        a = as_expr(a)
        z = {i: as_expr(e) for i, e in z.items()}
        beta = None if beta is None else as_expr(beta)
        vx = [dict(x) for x in (vx or [])]
        assert len(vx) < self.target.value_heads // self.target.key_heads or not vx, \
            "more value heads than the key head has"
        for x in vx:
            x["v"], x["a"] = chans(x["v"]), as_expr(x["a"])
            x["z"] = {i: as_expr(e) for i, e in x["z"].items()}
            x["beta"] = None if x.get("beta") is None else as_expr(x["beta"])
        exprs = [c.expr for ch in (q, k, v) for c in ch.values()] + [a, *z.values()]
        exprs += [] if beta is None else [beta]
        for x in vx:
            exprs += [c.expr for c in x["v"].values()] + [x["a"], *x["z"].values()]
            exprs += [] if x["beta"] is None else [x["beta"]]
        lvl = max(self.level(*exprs), min_level)
        sub = self._place(lvl, True, 1, self.heads, self.target.key_heads)

        def out_dims(tag, outs_, into_):
            if into_ is None:
                return [self._dim(f"{name}{tag}[{col}]", "head", sub) for col, _ in outs_]
            assert len(into_) == len(outs_)
            dims_ = []
            for e in into_:
                (d_,) = e.terms
                self.dims[d_].level = max(self.dims[d_].level, sub)
                dims_.append(d_)
            return dims_

        dims = out_dims("", outs, into)
        extra, ret = [], [Expr({d: 1}) for d in dims]
        for i, x in enumerate(vx):
            if isinstance(x.get("into"), str) and x["into"] == "main":  # onto the first's dims
                x["into"] = [Expr({d: 1}) for d in dims]
            xd = out_dims(f".v{i + 1}", x["outs"], x.get("into"))
            extra.append(VHead(x["v"], x["a"], x["z"],
                               [(col, d, w) for (col, w), d in zip(x["outs"], xd)], x["beta"],
                               x.get("into") is not None))
            ret += [Expr({d: 1}) for d in xd]
        self.nodes.append(Head(name, q, k, v, a, z,
                               [(col, d, w) for (col, w), d in zip(outs, dims)], sub,
                               self.target.cols, tiny, beta, into is not None, extra))
        self.nodes[-1].min_level = min_level
        return ret

    def n_layers(self) -> int:
        top = max((n.level for n in self.nodes), default=0)
        return (top + 1) // 2

    def stats(self) -> dict[str, int]:
        return {
            "layers": self.n_layers(),
            "dims": len(self.dims),
            "heads": sum(isinstance(n, Head) for n in self.nodes),
            "ffn_units": sum(len(n.terms) for n in self.nodes if isinstance(n, FFN)),
            "max_units": max(self.units.values(), default=0),
            "max_heads": max(self.heads.values(), default=0),
        }


# --- exactness checks -------------------------------------------------------------
#
# The evaluator computes every node twice: ideally (exact float64) and as the bf16 hardware does
# (every materialized tensor rounded to bf16, sums accumulated in fp32). The two must agree on
# every node output; fp32 hardware has no rounding and computes the ideal values.


class ExactnessError(AssertionError):
    pass


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to the nearest bf16 value (ties to even), returned as float64."""
    f = np.asarray(x, dtype=np.float32).copy()
    u = f.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return f.astype(np.float64)


def is_bf16(x: np.ndarray) -> np.ndarray:
    return (bf16(x) == x) & np.isfinite(x)


def require(ok, what: str, x) -> None:
    ok, x = np.asarray(ok), np.asarray(x)
    if not np.all(ok):
        t = np.unravel_index(int(np.argmin(ok)), ok.shape)
        raise ExactnessError(f"{what}: bad value {x[t]!r} at {t}")


def silu_ok(x: np.ndarray) -> np.ndarray:
    """silu(x) == relu(x) exactly (fp32 and bf16)."""
    return (x == 0) | (x >= 18) | (x <= -104)


# --- evaluation ---------------------------------------------------------------------


def ev(e: Expr, m: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(m.shape[0])
    idx = np.fromiter(e.terms.keys(), dtype=np.int64, count=len(e.terms))
    coef = np.array([float(c) for c in e.terms.values()])
    return m[:, idx] @ coef


def conv(c: Chan, m: np.ndarray, check: bool, carry: dict | None = None,
         key=None) -> np.ndarray:
    """silu(conv(projection)) as computed in bf16; must equal the exact value. With ``carry``
    the projection's last values of the previous chunk come first (see evaluate)."""
    x = ev(c.expr, m)
    if check:
        require(is_bf16(x), "projection", x)
    n = len(x)
    if carry is not None and any(c.taps[1:]):
        prev = carry.get(key, np.zeros(0))
        xx = np.concatenate([prev, x])
        carry[key] = xx[-3:].copy()
    else:
        xx = x
    p = len(xx) - n
    out = np.zeros(len(xx))
    for lag, w in enumerate(c.taps):
        if w:
            out[lag:] += w * xx[: len(xx) - lag]
    out = out[p:]
    if check:
        require(is_bf16(out) & silu_ok(out), "conv output (silu input)", out)
    return np.maximum(out, 0.0)


def evaluate(g: Graph, tokens: list[int], check: bool = True,
             carry: dict | None = None, kernel: bool = False) -> np.ndarray:
    """Residual stream after every node: (n_tokens, n_dims), exact in float64.

    Text tokens (ids outside the chip vocabulary) can be passed as -1: all chip dims are 0.
    ``carry`` (a dict, empty at the start of a stream) evaluates a long stream chunk by chunk:
    head states and conv windows pass from one call to the next. ``kernel`` also replays every
    head read the way the HF torch kernels compute it (kernel_check).
    """
    n = len(tokens)
    m = np.zeros((n, len(g.dims)))
    toks = np.asarray(tokens, dtype=np.int64)
    chip = toks >= 0
    m[:, ONE] = chip
    uniq, inv = np.unique(toks, return_inverse=True)
    for d in g.token_fns:
        col = np.array([float(g.feature_value(d, int(u))) if u >= 0 else 0.0 for u in uniq])[inv]
        if check:
            require(is_bf16(col), f"token feature {g.dims[d].name}", col)
        m[:, d] = col
    for i, node in enumerate(g.nodes):
        try:
            if isinstance(node, FFN):
                eval_ffn(node, m, chip, check)
            else:
                eval_head(node, m, chip, check, carry, i, kernel)
        except ExactnessError as e:
            raise ExactnessError(f"{node.name}: {e}") from None
    if check:
        # every RMSNorm divides by the ballast alone: other residual values must vanish next to
        # it (x^2 far below m * 2^80)
        big = np.abs(m[:, 1:]).max(0) if m.shape[0] else np.zeros(0)
        if len(big) and big.max() > RESIDUAL_MAX:
            d = int(big.argmax()) + 1
            raise ExactnessError(f"{g.dims[d].name}: residual value {big[d - 1]!r} would move "
                                 f"the norms (> {RESIDUAL_MAX})")
    return m


RESIDUAL_MAX = 2.0**28  # x^2 <= 2^56 is lost in fp32 next to the ballast's m * 2^80


def eval_ffn(node: FFN, m: np.ndarray, chip: np.ndarray, check: bool) -> None:
    ideal = np.zeros(m.shape[0])
    hw = np.zeros(m.shape[0])
    for c, a, b in node.terms:
        bv, av = ev(b, m), ev(a, m)
        ideal += float(c) * av * np.maximum(bv, 0.0)
        if check:
            gate = bf16(K * bv)
            require(silu_ok(gate), "gate outside silu's exact range", gate)
            up = bf16(av / K)
            hw += float(c) * bf16(np.maximum(gate, 0.0) * up)
    if check:
        hw = bf16(hw)
        require(hw == ideal, "bf16 output differs from exact", np.stack([hw, ideal], 1)[:, 0])
    if node.into:
        ideal = m[:, node.out] + ideal
        if check:
            require(is_bf16(ideal), "residual add", ideal)
    m[:, node.out] = ideal
BYTE_LEVELS = np.arange(256.0)
# a decay counter's row (2^-n of a set bit after n halvings), as a snapshot taken at n > 0 holds it
DECAY_LEVELS = np.array([0.0] + [A_BIT * 2.0**-k for k in range(127)])
RMS_EPS = 1e-6  # Qwen3.5 rms_norm_eps (the gated output norm)


def kernel_check(o_rows: list, q: np.ndarray, n_cols: int, ideal: np.ndarray) -> None:
    """A read as the HF torch Gated DeltaNet computes it, against the IR's exact value.

    q (all key dims) is L2-normalized in fp32 and scaled by 1/sqrt(128) (the recurrent kernel
    divides, the chunk kernel multiplies); o = sum of state rows times q in fp32, cast to bf16
    (the kernel's output dtype); the gated RMSNorm then normalizes o in fp32 and casts to bf16.
    The result must equal the IR's exact normalized row on every column. (One-hot queries give a
    scale of 2^-3.5, so only 0/power-of-2 data survives the bf16 cast; 2-hot queries give
    2^-4.)
    """
    import torch

    qv = torch.tensor(q[q > 0], dtype=torch.float32)
    inv = torch.rsqrt((qv * qv).sum() + 1e-6)
    ql = qv * inv
    for name, qs in (("recurrent", ql / (HD ** 0.5)), ("chunk", ql * (HD ** -0.5))):
        o = torch.zeros(n_cols, dtype=torch.float32)
        for (row, qi) in o_rows:
            o = o + torch.tensor(row, dtype=torch.float32) * qs[qi]
        h = o.to(torch.bfloat16).float()
        var = h.pow(2).mean()
        normed = (h * torch.rsqrt(var + RMS_EPS)).to(torch.bfloat16).double().numpy()
        bad = normed != ideal
        if bad.any():
            col = int(np.argmax(bad))
            raise ExactnessError(f"{name} kernel read: column {col} is {normed[col]!r}, "
                                 f"exact {ideal[col]!r}")


def eval_head(node: Head, m: np.ndarray, chip: np.ndarray, check: bool,
              carry: dict | None = None, ni: int = 0, kernel: bool = False) -> None:
    n = m.shape[0]
    q = (np.stack([conv(c, m, check, carry, (ni, "q", s)) for s, c in node.q.items()], 1)
         if node.q else np.zeros((n, 0)))
    k = (np.stack([conv(c, m, check, carry, (ni, "k", s)) for s, c in node.k.items()], 1)
         if node.k else np.zeros((n, 0)))
    qi, ki = list(node.q), list(node.k)
    if check:
        nz = (k > 0).sum(1)
        require(nz <= 1, "key must be one-hot", nz)
        require(~((nz > 0) & ~chip), "key on a text token", nz)
        for t in np.nonzero((q > 0).sum(1) > 1)[0]:
            vals = q[t][q[t] > 0]
            require(vals == vals[0], "query must be multi-hot", vals)
    has_key = k.any(1) if k.shape[1] else np.zeros(n, dtype=bool)
    key_row = [ki[int(np.argmax(k[t]))] if has_key[t] else None for t in range(n)]
    has_q = q.any(1) if q.shape[1] else np.zeros(n, dtype=bool)
    for x_, vh in enumerate(node.vheads()):
        v = np.zeros((n, node.n_cols))
        for col, c in vh.v.items():
            v[:, col] = conv(c, m, check, carry, (ni, "v", x_, col))
        a = ev(vh.a, m)
        z = {col: ev(e, m) for col, e in vh.z.items()}
        beta = np.ones(n) if vh.beta is None else ev(vh.beta, m)
        writes = has_key & (beta > 0)
        if check:
            require(is_bf16(a) & np.isin(a, list(DECAY)), "decay selector a", a)
            # beta = sigmoid(256 g - 128): exactly 1 at g = 1 and exactly 0 for integers g <= 0
            require((beta == 1.0) | ((beta <= 0) & (beta == np.round(beta))), "write gate beta",
                    beta)
            for col, zz in z.items():
                require(is_bf16(zz) & silu_ok(zz), f"gate z[{col}]", zz)
            written = v[writes]
            if len(written):
                data = np.delete(written, REF_COLS, 1)
                require(np.isin(data, [0.0, A_BIT]), "state row data column", data)
                ref = written[:, list(REF_COLS)]
                require((ref == 0) | (ref == A_REF), "state row ballast column", ref)
        if carry is not None and (ni, "S", x_) in carry:
            state = carry[ni, "S", x_]
        else:
            state = {r: np.asarray(row, dtype=np.float64).copy()
                     for r, row in ((node.init or [{}] * (x_ + 1))[x_] if node.init
                                    and x_ < len(node.init) else {}).items()}
            if check and state:
                rows = np.stack(list(state.values()))
                data = np.delete(rows, REF_COLS, 1)
                ok = np.isin(data, BYTE_LEVELS if node.levels == "byte" else
                             DECAY_LEVELS if node.levels == "decay" else [0.0, A_BIT])
                require(ok, f"initial state data ({node.levels})", data)
                ref = rows[:, list(REF_COLS)]
                require((ref == 0) | (ref == A_REF), "initial state ballast column", ref)
        out = np.zeros((n, node.n_cols))
        decay = np.vectorize(DECAY.get)(a)
        # only tokens that decay, write or read change anything
        events = np.nonzero((decay != 1.0) | writes | has_q)[0]
        for t in events:
            if decay[t] == 0.0:
                state.clear()
            elif decay[t] == 0.5:
                state = {s: row * 0.5 for s, row in state.items()}
            if writes[t]:
                state[key_row[t]] = v[t].copy()
            if has_q[t]:
                o = sum((state[s] for s, w in zip(qi, q[t]) if w > 0 and s in state),
                        np.zeros(node.n_cols))
                ref = (o[list(REF_COLS)] ** 2).sum()
                if ref == 0:
                    if check:
                        require(o == 0, "read without ballast", o)
                    continue
                if check:
                    data = (np.delete(o, REF_COLS) ** 2).sum()
                    require(np.array([data <= ref * 2.0**-26]), "data not absorbed by ballast",
                            np.array([data / ref]))
                out[t] = o / np.sqrt(ref / node.n_cols)
                if kernel and check:
                    rows = [(state[s], j) for j, (s, w) in enumerate(
                        (s, w) for s, w in zip(qi, q[t]) if w > 0) if s in state]
                    kernel_check(rows, q[t], node.n_cols, out[t])
        if carry is not None:
            carry[ni, "S", x_] = state
        for col, d, w in vh.out:
            zz = z.get(col, np.zeros(n))
            val = out[:, col] * np.maximum(zz, 0.0)
            if check:
                small = np.abs(out[:, col]) < node.tiny
                require(is_bf16(out[:, col]) | small, f"normed column {col}", out[:, col])
                require(is_bf16(val * w) | small, f"head output column {col}", val * w)
            if vh.into:
                m[:, d] += val * w
                if check:
                    require(is_bf16(m[:, d]), "head output sum", m[:, d])
            else:
                m[:, d] = val * w
