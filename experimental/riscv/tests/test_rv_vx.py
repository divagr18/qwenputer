"""The vector unit (vx.py) against the reference, teacher-forced (one op family at a time)."""

import random

import pytest

from chipformer.rv import core
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

from test_rv_core import EXIT, run_check


@pytest.fixture(scope="module")
def circ():
    return core.build(vx=True)


def word31(rng):
    return rng.choice([0, 1, -1, (1 << 30) - 1, -(1 << 30), rng.randint(-(1 << 30), (1 << 30) - 1),
                       rng.randint(-50000, 50000)])


def vput(rng, reg, pair, a, b):
    return [f"li t0, {reg * 32 + pair}", f"li t1, {a}", f"li t2, {b}", "vput t0, t1, t2"]


@pytest.mark.parametrize("seed", range(3))
def test_vput(circ, seed):
    rng = random.Random(seed)
    lines = []
    for _ in range(12):
        lines += vput(rng, rng.randrange(16), rng.randrange(32), word31(rng), word31(rng))
        if rng.random() < 0.3:
            lines.append(f"li t3, {rng.randint(-2048, 2047)}")
    toks = run_check(circ, assemble("\n".join(lines) + EXIT))
    assert sum(S.family(t)[0] == "VC" for t in toks) == 12


def vq(src, eoff, n, dst, sh, off, r):
    cfg = src | eoff << 4 | n << 10 | dst << 17 | sh << 19
    return [f"li t0, {cfg}", f"li t1, {off}", f"li t2, {r}", "vq t0, t1, t2"]


def fill(rng, reg, values=None):
    """VPUT a whole register (64 elements)."""
    lines = []
    for p in range(32):
        a, b = (values[2 * p], values[2 * p + 1]) if values else (word31(rng), word31(rng))
        lines += vput(rng, reg, p, a, b)
    return lines


@pytest.mark.parametrize("seed", range(3))
def test_vq(circ, seed):
    rng = random.Random(seed)
    lines = []
    for reg in rng.sample(range(16), 3):
        vals = [rng.choice([word31(rng), rng.randint(-40000, 40000), rng.randint(-300, 300)])
                for _ in range(64)]
        lines += fill(rng, reg, vals)
        for _ in range(4):
            n = rng.choice([2, 8, 16, 44, 64])
            span = 1 << (n - 1).bit_length()
            eoff = rng.randrange(0, 64, span) if span < 64 else 0
            off = rng.randrange(0, 384 - span + 1, max(span, 8))
            lines += vq(reg, eoff, n, 0, rng.randrange(32), off, rng.randrange(1 << 15))
    run_check(circ, assemble("\n".join(lines) + EXIT))


@pytest.mark.parametrize("seed", range(2))
def test_vred_vget(circ, seed):
    rng = random.Random(seed)
    lines = []
    for reg in rng.sample(range(16), 2):
        lines += fill(rng, reg, [rng.randint(-40000, 40000) for _ in range(64)])
        for ch in rng.sample(range(6), 2):
            lines += vq(reg, 0, 64, 0, rng.randrange(4, 12), 64 * ch, rng.randrange(1 << 15))
            for mode in (0, 1):
                n = rng.randint(1, 64)
                lines += [f"li t0, {n | mode << 7 | ch << 8}", "vred t0, zero, zero",
                          "li t1, 1", "vget s1, t1, zero", "xor s2, s2, s1"]
    toks = run_check(circ, assemble("\n".join(lines) + EXIT))
    assert sum(S.family(t)[0] == "VC" for t in toks) > 0


def wrom_cells(rng, n_chunks, smax):
    """Random row-chunks: 64 int4 cells, a scale < smax in 4 nibble cells, 4 unused."""
    out = bytearray()
    for _ in range(n_chunks):
        out += bytes(rng.randrange(16) for _ in range(64))
        sc = rng.randrange(smax)
        out += bytes(sc >> 4 * i & 15 for i in range(4)) + bytes(4)
    return bytes(out)


def vmat(dst, doff, n, k, mode, keep, chunk, base):
    cfg = dst | n << 4 | k << 11 | mode << 13 | keep << 14 | chunk << 15
    return [f"li t0, {cfg}", f"li t1, {base | doff << 13}", f"vmat t0, t1, zero"]


@pytest.mark.parametrize("seed", range(2))
def test_vmat(circ, seed):
    rng = random.Random(seed)
    lines = []
    reg = rng.randrange(16)
    lines += fill(rng, reg, [rng.randint(-30000, 30000) for _ in range(64)])
    for ch in range(3):
        lines += vq(reg, 0, 64, 0, rng.randrange(6, 10), 64 * ch, rng.randrange(1 << 15))
    for _ in range(5):
        k = rng.choice([1, 1, 3])
        mode = 0 if k == 3 else rng.choice([0, 0, 1])
        n = rng.choice([2, 4, 8] if k == 3 else [2, 8, 16])
        base = rng.randrange(0, 1024, 256) + rng.randrange(0, 64, n) if k == 3 else rng.randrange(0, 1024, n)
        dst, doff = rng.randrange(16), rng.randrange(0, 64, n)
        lines += vmat(dst, doff, n, k, mode, rng.random() < 0.3, 0 if k == 3 else rng.randrange(3), base)
        lines += ["li t1, 0", "vget s1, t1, zero", "li t1, 2", "vget s2, t1, zero",
                  "li t1, 3", "vget s3, t1, zero"]
        if mode == 0:
            lines += vq(dst, doff, n, 0, rng.randrange(8, 14), 64 * 5, rng.randrange(1 << 15))
    cells = wrom_cells(rng, 1024, 1 << 12)
    src = "\n".join(lines) + EXIT + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(b) for b in cells[i:i + 72])}" for i in range(0, len(cells), 72))
    run_check(circ, assemble(src))
