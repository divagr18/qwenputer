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


def vput_program(seed):
    rng = random.Random(seed)
    lines = []
    for _ in range(12):
        lines += vput(rng, rng.randrange(16), rng.randrange(32), word31(rng), word31(rng))
        if rng.random() < 0.3:
            lines.append(f"li t3, {rng.randint(-2048, 2047)}")
    return assemble("\n".join(lines) + EXIT)


@pytest.mark.parametrize("seed", range(3))
def test_vput(circ, seed):
    toks = run_check(circ, vput_program(seed))
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


def vq_program(seed):
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
    return assemble("\n".join(lines) + EXIT)


@pytest.mark.parametrize("seed", range(3))
def test_vq(circ, seed):
    run_check(circ, vq_program(seed))


def vred_vget_program(seed):
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
    return assemble("\n".join(lines) + EXIT)


@pytest.mark.parametrize("seed", range(2))
def test_vred_vget(circ, seed):
    toks = run_check(circ, vred_vget_program(seed))
    assert sum(S.family(t)[0] == "VC" for t in toks) > 0


def wrom_cells(rng, n_chunks, smax):
    """Random row-chunks: 64 int4 cells, a scale < smax in 4 nibble cells, 4 unused."""
    out = bytearray()
    for _ in range(n_chunks):
        out += bytes(rng.randrange(16) for _ in range(64))
        sc = rng.randrange(smax)
        out += bytes(sc >> 4 * i & 15 for i in range(4)) + bytes(4)
    return bytes(out)


def vmat(dst, doff, n, k, mode, keep, chunk, base, start=0):
    cfg = dst | n << 4 | k << 11 | mode << 13 | keep << 14 | chunk << 15
    return [f"li t0, {cfg}", f"li t1, {base | doff << 13 | start << 19}", f"vmat t0, t1, zero"]


def vmat_program(seed):
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
        start = 0 if k == 3 else rng.randrange(3)
        span = n if k == 3 else (start + n - 1) // 3 + 1
        base = 64 * rng.randrange(S.VX_CHUNKS // 192) + rng.randrange(65 - span)
        dst, doff = rng.randrange(16), rng.randrange(0, 64, n)
        lines += vmat(dst, doff, n, k, mode, rng.random() < 0.3, 0 if k == 3 else rng.randrange(3), base,
                      start)
        lines += ["li t1, 0", "vget s1, t1, zero", "li t1, 2", "vget s2, t1, zero",
                  "li t1, 3", "vget s3, t1, zero"]
        if mode == 0:
            lines += vq(dst, doff, n, 0, rng.randrange(8, 14), 64 * 5, rng.randrange(1 << 15))
    cells = wrom_cells(rng, S.VX_CHUNKS, 1 << 12)
    src = "\n".join(lines) + EXIT + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(b) for b in cells[i:i + 72])}" for i in range(0, len(cells), 72))
    return assemble(src)


@pytest.mark.parametrize("seed", range(4))
def test_vmat(circ, seed):
    run_check(circ, vmat_program(seed))


def vew(a, b, eoff, n, op, keep, sh, so, shb, p):
    cfg = a | b << 4 | n << 8 | op << 15 | keep << 17 | eoff << 18
    return [f"li t0, {cfg}", f"li t1, {sh | (so & 63) << 5 | shb << 11}", f"li t2, {p}",
            "vew t0, t1, t2"]


def vew_program(seed):
    rng = random.Random(seed)
    lines = []
    regs = [rng.randrange(8), 8 + rng.randrange(8)]
    for reg in regs:
        lines += fill(rng, reg, [rng.choice([rng.randint(-40000, 40000), rng.randint(-(1 << 22), 1 << 22),
                                             word31(rng)]) for _ in range(64)])
    for k in range(4):
        lines += [f"li t0, {512 + k}", f"li t1, {rng.randint(-16384, 16384)}",
                  f"li t2, {rng.randint(-16384, 16384)}", "vput t0, t1, t2"]
    for _ in range(6):
        op = rng.randrange(3)
        n = rng.choice([2, 4, 8, 16])
        eoff = rng.randrange(0, 64, n)
        a = rng.choice(regs)
        if op == 2:
            b = rng.choice([r for r in range(16) if r != a])
        else:
            b = regs[1] if a == regs[0] else regs[0]
        sh = rng.randrange(0, 24)
        so = rng.randrange(-24, 2) if op == 1 else rng.randrange(-32, 8)
        lines += vew(a, b, eoff, n, op, rng.random() < 0.3, sh, so, rng.randrange(0, 24),
                     rng.randrange(1 << 15))
        lines += ["li t1, 0", "vget s1, t1, zero", "li t1, 3", "vget s3, t1, zero"]
        if op == 2:
            lines += vq(b, eoff, n, 0, rng.randrange(8, 14), 64 * 5, rng.randrange(1 << 15))
    return assemble("\n".join(lines) + EXIT)


@pytest.mark.parametrize("seed", range(3))
def test_vew(circ, seed):
    run_check(circ, vew_program(seed))


def vmatr(dst, eoff, n, mode, keep, off, base, s):
    cfg = dst | n << 4 | mode << 10 | keep << 11 | off << 12
    return [f"li t0, {cfg}", f"li t1, {base | eoff << 10}", f"li t2, {s}", "vmatr t0, t1, t2"]


def vexp(src, n, off, ksh, c):  # m: s1 (the scores' max, from VGET 0)
    return [f"li t0, {src | n << 4 | off << 11 | ksh << 20}", f"li t2, {c}", "vexp t0, s1, t2"]


def attention_program(seed):
    """VQ into the K / V caches, scores (VMATR 0), VEXP, VRED, p.V (VMATR 1)."""
    rng = random.Random(seed)
    lines = []
    src = rng.randrange(8)
    lines += fill(rng, src, [rng.randint(-40000, 40000) for _ in range(64)])
    l, n = rng.randrange(5), rng.randint(1, 32)
    for slot in range(n):  # each slot's K and V rows (4 groups of 8)
        for dst in (1, 2):
            lines += vq(src, 32 * (slot % 2), 32, dst, rng.randrange(6, 10), 128 * l + slot,
                        rng.randrange(1 << 15))
    lines += vq(src, 0, 8, 0, rng.randrange(6, 10), 64 * 3, rng.randrange(1 << 15))
    for g in range(4):
        base = (4 * l + g) * 32
        lines += vmatr(6, 0, n, 0, 0, 3, base, 0)
        lines += ["li t1, 0", "vget s1, t1, zero"]
        lines += vexp(6, n, 64 * 4, rng.randrange(20, 30), rng.randrange(1 << 15, 1 << 16))
        lines += [f"li t0, {n | 1 << 7 | 4 << 8}", "vred t0, zero, zero"]
        lines += vmatr(9, 8 * g, n, 1, 0, 4, base, rng.randrange(1 << 17))
        lines += ["li t1, 0", "vget s2, t1, zero"]
    return assemble("\n".join(lines) + EXIT)


@pytest.mark.parametrize("seed", range(2))
def test_attention(circ, seed):
    run_check(circ, attention_program(seed))
