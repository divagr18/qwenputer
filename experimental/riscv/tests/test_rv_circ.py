"""fx circuits in the RV core (fx/docs/fx/circuits.md): the MMOVE and MEQL macro units built
into chipformer/rv/core.py, on fx's own code.

The programs run the instruction words of fx's memcpy / copy_fwd / memmove block and of its
mem.eqlBytes (tests/data/rv/fx_units.json, taken from the row-11 image), relocated into the
test ROM unchanged (all their jumps are PC-relative). The macros sit on memmove's entry
`beq a0, a1` and on eqlBytes' `beq a1, a3`. Three checks:

  1. the traced machine with the macros leaves the same RAM and results as the plain ISA
     emulator stepping fx's original instructions (the macro's architectural effect);
  2. the circuit, teacher-forced on the macro trace, predicts every token (IR exactness);
  3. calls outside a macro's condition run the original code (the decline path).
"""

import json
import random
from pathlib import Path

import pytest

from chipformer.q35 import ir
from chipformer.rv import core, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

DATA = Path(__file__).parent / "data" / "rv"
FX = json.loads((DATA / "fx_units.json").read_text())
BUF = S.DATA_BASE          # random bytes the calls copy and compare
BUF_LEN = 0x1000
RES = BUF + BUF_LEN        # eqlBytes results, one word per call
STACK_LO = S.SP_INIT - 0x100  # memmove's / eqlBytes' frames: dead below sp after they return


def words_block(label, words, marks):
    lines = [f"{label}:"]
    for i, w in enumerate(words):
        for name, off in marks.items():
            if off == 4 * i:
                lines.append(f"{name}:")
        lines.append(f"    .word {w:#x}")
    return lines


def fx_program(calls, seed=0):
    """Calls ("move", d, s, n) / ("eql", a, n, b, bn) on fx's code, then exit."""
    rng = random.Random(seed)
    src = ["_start:", f"    li sp, {S.SP_INIT:#x}"]
    k = 0
    for c in calls:
        if c[0] == "move":
            _, d, s, n = c
            src += [f"    li a0, {BUF + d:#x}", f"    li a1, {BUF + s:#x}", f"    li a2, {n}", "    call fx_memmove"]
        else:
            _, a, n, b, bn = c
            src += [f"    li a1, {BUF + a:#x}", f"    li a2, {n}", f"    li a3, {BUF + b:#x}", f"    li a4, {bn}",
                    "    call fx_eql", f"    li t0, {RES + 4 * k:#x}", "    sw a0, 0(t0)"]
            k += 1
    src += ["    li a7, 93", "    ecall"]
    mb, eq = FX["move_block"], FX["eql"]
    src += words_block("fx_block", mb["words"], {"fx_memmove": mb["memmove"]})
    src += words_block("fx_eql", eq["words"], {})
    data = bytearray(rng.randbytes(BUF_LEN))
    src += [".data", "buf:"] + [f"    .byte {', '.join(str(b) for b in data[i:i + 16])}" for i in range(0, BUF_LEN, 16)]
    src += [f"    .space {4 * max(k, 1)}"]
    prog = assemble("\n".join(src))
    macros = {(prog.labels["fx_memmove"] - S.ROM_BASE) >> 2: "MMOVE",
              (prog.labels["fx_eql"] + FX["eql"]["trigger"] - S.ROM_BASE) >> 2: "MEQL"}
    return prog, macros


def plain_run(prog):
    """fx's original instructions on the plain ISA emulator."""
    cpu = ref.Rv32()
    for i, w in enumerate(prog.text):
        cpu.store(S.ROM_BASE + 4 * i, w, 4)
    for i, b in enumerate(prog.data):
        cpu.mem[prog.data_base + i] = b
    cpu.pc = prog.labels.get("_start", S.ROM_BASE)
    cpu.x[2] = S.SP_INIT
    while not cpu.halted and cpu.steps < 10**6:
        cpu.step()
    assert cpu.exit_code is not None
    return cpu


def macro_run(prog, macros):
    m = ref.Machine.from_program(prog, macros=macros)
    toks = m.run(10**6)
    assert S.family(toks[-1])[0] == "HALT"
    return m, toks


def ram_bytes(m):
    return b"".join(w.to_bytes(4, "little") for w in m.ram)


def check_semantics(prog, macros):
    cpu = plain_run(prog)
    m, toks = macro_run(prog, macros)
    got = ram_bytes(m)
    want = bytes(cpu.mem.get(a, 0) for a in range(4 * S.RAM_WORDS))
    bad = [hex(a) for a in range(STACK_LO) if got[a] != want[a]]
    assert not bad, bad[:8]
    plain_toks = ref.Machine.from_program(prog).run(10**6)
    return m, toks, plain_toks


MOVE_CALLS = [("move", 0x100, 0x400, 32),     # aligned, disjoint: fires
              ("move", 0x400, 0x408, 40),     # aligned, overlapping, d < s: fires
              ("move", 0x800, 0x900, 84),     # the largest: fires
              ("move", 0x200, 0x200, 16),     # d == s: the beq jumps to ret
              ("move", 0x300, 0x601, 6),      # unaligned: the original copy
              ("move", 0x504, 0x500, 8),      # d > s, overlapping: the original (copy_bwd)
              ("move", 0x600, 0x700, 0)]      # n = 0: declines, the original returns
EQL_CALLS = [("eql", 0x10, 12, 0x10, 12),     # a == b: the beq jumps to ret
             ("eql", 0x20, 9, 0x31, 9),       # random bytes: mismatch at once
             ("eql", 0x40, 31, 0x40 + 0x200, 31),  # the largest: equal (made so below)
             ("eql", 0x80, 5, 0x81, 5)]


def eql_program(seed=1):
    prog, macros = fx_program(MOVE_CALLS[:1] + EQL_CALLS, seed)
    # make some compares equal / nearly equal: copy the operand, then flip a late byte
    data = bytearray(prog.data)
    a, b = 0x40, 0x240
    data[b:b + 31] = data[a:a + 31]
    data[0x31:0x31 + 4] = data[0x20:0x20 + 4]  # 4 equal bytes, then a mismatch
    prog.data = bytes(data)
    return prog, macros


@pytest.fixture(scope="module")
def circ():
    return core.build()


def test_move_semantics():
    prog, macros = fx_program(MOVE_CALLS)
    m, toks, plain = check_semantics(prog, macros)
    names = [S.family(t)[0] for t in toks]
    assert names.count("MP") == 3, names.count("MP")
    assert names.count("CWW0") + names.count("CWW1") == (32 + 40 + 84) // 4
    assert len(toks) < len(plain)


def test_eql_semantics():
    prog, macros = eql_program()
    m, toks, plain = check_semantics(prog, macros)
    names = [S.family(t)[0] for t in toks]
    assert names.count("MP") == 4  # the move, and the three eqls with a != b
    res = [m.ram[(RES >> 2) + k] for k in range(4)]
    assert res == [1, 0, 1, 0], res


@pytest.mark.parametrize("seed", range(40))
def test_macro_random_semantics(seed):
    """Random calls (aligned or not, overlapping or not, every length 0..96) against fx's
    original instructions."""
    rng = random.Random(seed)
    calls = []
    for _ in range(6):
        if rng.random() < 0.6:
            n = rng.choice([0, 4, 8, 12, rng.randrange(0, 97), 4 * rng.randrange(1, 22)])
            s = rng.randrange(0, 0x700)
            d = s + rng.randint(-12, 12) if rng.random() < 0.3 else rng.randrange(0, 0x700)
            if rng.random() < 0.6:
                d, s = d & ~3, s & ~3
            calls.append(("move", max(d, 0), s, n))
        else:
            n = rng.randrange(0, 34)
            calls.append(("eql", rng.randrange(0, 0x700), n, rng.randrange(0, 0x700), n))
    prog, macros = fx_program(calls, seed)
    check_semantics(prog, macros)


def run_circuit(c, prog, macros):
    m = ref.Machine.from_program(prog, macros=macros)
    core.snapshot(c, m)
    toks = ref.Machine.from_program(prog, macros=macros).run(10**6)
    mat = ir.evaluate(c, toks, check=True, kernel=len(toks) < 1500)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    return toks


def test_move_circuit(circ):
    prog, macros = fx_program([MOVE_CALLS[0], MOVE_CALLS[1], MOVE_CALLS[3], ("move", 0x302, 0x601, 3)])
    run_circuit(circ, prog, macros)


def test_eql_circuit(circ):
    prog, macros = eql_program()
    run_circuit(circ, prog, macros)


def random_calls(seed, k=5):
    rng = random.Random(1000 + seed)
    calls, eq_pairs = [], []
    for _ in range(k):
        if rng.random() < 0.55:
            n = rng.choice([0, 4, 8, rng.randrange(0, 90), 4 * rng.randrange(1, 22), 4 * rng.randrange(1, 22)])
            s = rng.randrange(0, 0x600) & ~3
            d = s + 4 * rng.randint(-3, 3) if rng.random() < 0.3 else rng.randrange(0, 0x600) & ~3
            if rng.random() < 0.2:
                d += rng.randrange(1, 4)
            calls.append(("move", max(d, 0), s, n))
        else:
            n = rng.choice([1, 2, rng.randrange(1, 32), 31, rng.randrange(32, 40)])
            a, b = rng.randrange(0x700, 0x800), rng.randrange(0x900, 0xA00)
            calls.append(("eql", a, n, b, n))
            eq_pairs.append((a, b, n, rng.choice([None, rng.randrange(n)])))
    prog, macros = fx_program(calls, seed)
    data = bytearray(prog.data)
    for a, b, n, flip in eq_pairs:  # equal operands, or one byte different
        data[b:b + n] = data[a:a + n]
        if flip is not None:
            data[b + flip] ^= 0x20
    prog.data = bytes(data)
    return prog, macros


@pytest.mark.parametrize("seed", range(6))
def test_macro_random_circuit(circ, seed):
    """Random macro calls (fired and declined), teacher-forced through the circuit; their
    semantics checked against fx's original instructions as well."""
    prog, macros = random_calls(seed)
    check_semantics(prog, macros)
    run_circuit(circ, prog, macros)


def test_depth_and_caps(circ):
    st = circ.stats()
    assert st["layers"] <= 12, st
    assert st["max_heads"] <= circ.target.key_heads
    assert st["max_units"] <= circ.target.intermediate
