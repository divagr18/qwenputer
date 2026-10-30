"""VDOT / WB (the int8 x int4 dot product on the weight memory) against the reference, teacher-forced."""

import random

import pytest

from chipformer.rv import core, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

from test_rv_core import EXIT, run_check

EXTREME = [0x80, 0x7F, 0xFF, 0x00, 0x01, 0x81, 0xFE]


@pytest.fixture(scope="module")
def circ():
    return core.build()


def word(rng):
    """A register of 4 activation bytes, often with extreme bytes."""
    return sum((rng.choice(EXTREME) if rng.random() < 0.3 else rng.randrange(256)) << 8 * k
               for k in range(4))


def vdot_program(rng: random.Random, n_runs: int, cells: bytes, bases=None) -> str:
    """Runs of VDOT after WB, with other instructions (which skip a word) between some."""
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {sx32(word(rng))}" for r in regs]
    n_words = len(cells) // S.WROM_CELLS
    for i in range(n_runs):
        base = bases[i % len(bases)] if bases else rng.randrange(max(n_words - 40, 1))
        acc = rng.choice(["s2", "s3", "zero"])
        init = rng.choice([0, rng.randint(-2**14, 2**14), rng.randint(-2**31, 2**31 - 1)])
        lines += [f"li s4, {base}", f"li {acc}, {init}" if acc != "zero" else "nop", "wb s4"]
        for _ in range(rng.randint(1, 12)):
            if rng.random() < 0.15:
                lines.append(f"xori {rng.choice(regs)}, {rng.choice(regs)}, {rng.randint(-2048, 2047)}")
            lines.append(f"vdot {acc}, {rng.choice(regs)}, {rng.choice(regs)}")
    return "\n".join(lines) + EXIT + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(b) for b in cells[i:i + 64])}" for i in range(0, len(cells), 64))


def sx32(v):
    return v - (1 << 32) if v >> 31 else v


def test_reference_vdot():
    cells = bytes(random.Random(0).randrange(16) for _ in range(8 * 64))
    m = ref.Machine.from_program(assemble("li a0, 3\nwb a0\nli t0, -1\nvdot a1, t0, t0" + EXIT
                                          + "\n.wrom\n.byte " + ", ".join(map(str, cells))))
    m.run()
    # the vdot is 2 words after the WB: word 3 + 1 (the li) = 4
    assert ref.sx(m.x[11], 32) == sum(-(u - 8) for u in cells[32:40])


@pytest.mark.parametrize("seed", range(4))
def test_vdot_random(circ, seed):
    rng = random.Random(seed)
    cells = bytes(rng.randrange(16) for _ in range(8 * 512))
    toks = run_check(circ, assemble(vdot_program(rng, 6, cells)))
    fams = [S.family(t)[0] for t in toks]
    assert "WB" in fams and fams[-1] == "HALT"


def test_vdot_record_sizes(circ):
    """All weights -8 or 7 and extreme activations: large sums, all three record sizes."""
    rng = random.Random(9)
    cells = bytes(rng.choice([0, 15]) for _ in range(8 * 64))
    src = vdot_program(rng, 8, cells, bases=[0, 17, 40])
    toks = run_check(circ, assemble(src))
    fams = {S.family(t)[0] for t in toks}
    assert {"F0", "F1", "F2"} <= fams


def test_vdot_high_words(circ):
    """Words in the 17th head (WP >= 30720) and past the data (cells read as 0: w = -8)."""
    rng = random.Random(5)
    cells = bytes(rng.randrange(16) for _ in range(8 * 31000))
    src = vdot_program(rng, 4, cells, bases=[30700, 30990, 32700, 2047])
    run_check(circ, assemble(src))


def smaqa_program(rng: random.Random, n: int, cells: bytes) -> str:
    """SMAQA with extreme and random bytes, accumulators of every size, VDOT runs between."""
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {sx32(word(rng))}" for r in regs]
    for _ in range(n):
        acc = rng.choice(["s2", "s3", "zero"])
        if acc != "zero":
            lines.append(f"li {acc}, {rng.choice([0, rng.randint(-2**14, 2**14), rng.randint(-2**31, 2**31 - 1)])}")
        for _ in range(rng.randint(1, 6)):
            lines.append(f"smaqa {acc}, {rng.choice(regs)}, {rng.choice(regs)}")
        if rng.random() < 0.3:
            lines += [f"li s4, {rng.randrange(60)}", "wb s4", f"vdot s5, {rng.choice(regs)}, {rng.choice(regs)}"]
    return "\n".join(lines) + EXIT + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(b) for b in cells[i:i + 64])}" for i in range(0, len(cells), 64))


def test_reference_smaqa():
    m = ref.Machine.from_program(assemble("li t0, 0x7F80FF01\nli t1, 0x80807F7F\nli a1, 5\n"
                                          "smaqa a1, t0, t1" + EXIT))
    m.run()
    assert ref.sx(m.x[11], 32) == 5 + 1 * 127 + (-1) * 127 + (-128) * (-128) + 127 * (-128)


@pytest.mark.parametrize("seed", range(3))
def test_smaqa_random(circ, seed):
    rng = random.Random(100 + seed)
    cells = bytes(rng.randrange(16) for _ in range(8 * 64))
    toks = run_check(circ, assemble(smaqa_program(rng, 10, cells)))
    assert S.family(toks[-1])[0] == "HALT"
