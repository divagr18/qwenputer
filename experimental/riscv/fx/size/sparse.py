"""The chip layers as sparse tensors, from the repo's own emitter.

emit_layers runs unchanged, with its dense torch.zeros swapped for a dict-backed tensor, so a term
here is exactly a nonzero the dense emitter would have written. Nothing about the circuit is
re-derived. Also the overlay analysis: which IR dims could live in Qwen's columns.
"""
import struct
from collections import Counter


class ST:
    """Enough of a tensor for emit.py: t[i][j] += x, t[i, j] = x, t[i, 0, k] = x."""

    def __init__(self, shape, fill=0.0):
        self.shape, self.d, self.fill = tuple(shape), {}, fill

    def __getitem__(self, i):
        if isinstance(i, tuple):
            return self.d.get(tuple(int(x) for x in i), self.fill)
        return _Row(self, int(i))

    def __setitem__(self, i, v):
        self.d[tuple(int(x) for x in i)] = float(v)


class _Row:
    def __init__(self, t, i):
        self.t, self.i = t, i

    def __getitem__(self, j):
        return self.t.d.get((self.i, int(j)), self.t.fill)

    def __setitem__(self, j, v):
        self.t.d[(self.i, int(j))] = float(v)


class Shim:
    """The part of the torch namespace emit_layers touches."""


    @staticmethod
    def zeros(*shape, dtype=None):
        return ST(shape)

    @staticmethod
    def full(shape, v, dtype=None):
        return ST(shape, float(v))

    @staticmethod
    def ones(*shape, dtype=None):
        return ST(shape, 1.0)

    def __getattr__(self, k):
        import torch

        return getattr(torch, k)


def is_bf16(x: float) -> bool:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return (b & 0xFFFF) == 0 and struct.unpack("<f", struct.pack("<f", x))[0] == x


def emit_sparse(g, lay, tc, alloc, keep, prefix="model."):
    """{tensor name: ST}; asserts every value is a bf16 value, as check_bf16 does."""
    from chipformer.q35 import emit as E

    old_t, old_c = E.torch, E.check_bf16

    def chk(name, t):
        for k, v in t.d.items():
            assert is_bf16(v), (name, k, v)

    E.torch, E.check_bf16 = Shim(), chk
    try:
        return E.emit_layers(g, lay, tc, 0, prefix, alloc, keep)
    finally:
        E.torch, E.check_bf16 = old_t, old_c


def text_tokens(T):
    from chipformer.rv import spec as S

    ids = sorted({T.TCO, T.TCC, *T.printable, 0, 1, 1000, 5000, 100000, 248000})
    return [S.TEXT_TOKEN | q for q in ids]


def expr_dims(e):
    from chipformer.dsl.graph import ONE

    return {d for d in e.terms if d != ONE}


def overlay_sets(c, T):
    """Tset: dims that may be nonzero on a text token. C: dims that must sit in chip-only columns.
    Conservative: a dim leaves the overlay candidates when a reader of it cannot be gated."""
    from chipformer.q35.ir import FFN

    tt = text_tokens(T)
    Tset = {d for d in c.token_fns if any(float(c.feature_value(d, t)) != 0 for t in tt)}
    Tinit = set(Tset)

    def is_lag(ch):
        return any(x != 0 for x in ch.taps[1:])

    for n in c.nodes:  # nodes are in topological order
        if isinstance(n, FFN):
            if any(expr_dims(a) & Tset and expr_dims(b) & Tset for _, a, b in n.terms):
                Tset.add(n.out)
        else:
            act = any(expr_dims(ch.expr) & Tset or is_lag(ch) for ch in n.q.values())
            for vh in n.vheads():
                for col, d, _w in vh.out:
                    if act and expr_dims(vh.z[col]) & Tset if col in vh.z else False:
                        Tset.add(d)
    C = set(Tset) | set(c.keep)
    why = Counter()
    changed = True

    def need(dims, tag):
        nonlocal changed
        new = dims - C
        if new:
            C.update(new)
            why[tag] += len(new)
            changed = True

    while changed:
        changed = False
        for n in c.nodes:
            if isinstance(n, FFN):
                for _, a, b in n.terms:
                    if expr_dims(a) & Tset and expr_dims(b) & Tset:
                        need(expr_dims(a) | expr_dims(b), "ffn unit active on text")
            else:
                chs = list(n.q.values()) + list(n.k.values())
                for vh in n.vheads():
                    chs += list(vh.v.values())
                for ch in chs:
                    if is_lag(ch):
                        need(expr_dims(ch.expr), "lag channel")
                    elif expr_dims(ch.expr) & Tset:
                        need(expr_dims(ch.expr), "channel active on text")
                for vh in n.vheads():
                    if expr_dims(vh.a) & Tset:
                        need(expr_dims(vh.a), "decay active on text")
                    if vh.beta is not None and expr_dims(vh.beta) & Tset:
                        need(expr_dims(vh.beta), "beta active on text")
                    for e in vh.z.values():
                        if expr_dims(e) & Tset:
                            need(expr_dims(e), "z active on text")
    return Tinit, Tset, C, why


def peak(c, dims, keep, top):
    """Peak simultaneous live dims within a set (first definition .. last read)."""
    from chipformer.dsl.graph import ONE
    from chipformer.q35.emit import reads

    last = {d: c.dims[d].level for d in range(1, len(c.dims))}
    for n in c.nodes:
        for d in reads(n):
            if d != ONE:
                last[d] = max(last[d], n.level)
    for d in keep:
        last[d] = top
    ev = Counter()
    for d in dims:
        if d == 0:
            continue
        ev[c.dims[d].level] += 1
        ev[last[d] + 1] -= 1
    cur = best = 0
    for lv in sorted(ev):
        cur += ev[lv]
        best = max(best, cur)
    return best
