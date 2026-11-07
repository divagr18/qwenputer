"""The paged map (docs/fx/memory.md): the circuit against PagedMachine, teacher-forced in the IR."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble
from chipformer.rv.programs import ALL, program

DATA = Path(__file__).parent / "data" / "rv"
ELFS = sorted(json.loads((DATA / "qemu.json").read_text()))
SLOW = {"c-sieve", "c-sort", "c-crc32p", "c-gmpmul"}
FREE_RUN = {"c-fib", "c-sudoku", "c-qsort", "c-sieve4", "mpy-s1", "mpy-s5"}
EXIT = "\nli a7, 93\necall"
# 3 banks of ROM across a 64K-word window (W = 19), RAM windows, a frame budget that runs out
BIG = S.MemCfg("big", 22, 19, 40, ((0, 15), (0x100000, 16), (0x3F0000, 15)), 64, (2, 3, 32),
               0x3F8000)


def small_core():
    return core.build(cfg=S.small_cfg())


@pytest.fixture(scope="module")
def small():
    return small_core()


@pytest.fixture(scope="module")
def big():
    return core.build(cfg=BIG)


def run_check(c, mk, kernel=True, ntok=10**7):
    core.snapshot(c, mk())
    m = mk()
    toks = m.run(ntok)
    mat = ir.evaluate(c, toks, check=True, kernel=kernel)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    return toks, m


def small_prog(prog):
    """An assembled legacy-map program on the paged machine."""
    def mk():
        segs = [(S.ROM_BASE, b"".join(w.to_bytes(4, "little") for w in prog.text))]
        if prog.data:
            segs.append((prog.data_base, bytes(prog.data)))
        if getattr(prog, "rodata", b""):
            segs.append((S.DROM_BASE, bytes(prog.rodata)))
        return ref.PagedMachine.from_segments(S.small_cfg(), prog.entry, segs)
    return mk


@pytest.mark.parametrize("name", list(ALL))
def test_programs(small, name):
    run_check(small, small_prog(program(name)))


@pytest.mark.parametrize("name", ELFS)
def test_compiled(small, name):
    if name in FREE_RUN:
        pytest.skip("checked free-running by the executor")
    if name in SLOW and not os.environ.get("RV_SLOW"):
        pytest.skip("long trace (RV_SLOW=1)")
    path = DATA / f"{name}.elf"
    mk = lambda: ref.PagedMachine.from_elf(S.small_cfg(), path)
    n = len(mk().run(10**7))
    toks, _ = run_check(small, mk, kernel=n < 1500)
    assert S.family(toks[-1])[0] == "HALT"


def big_machine(src, cfg=BIG, data=None):
    """Assemble at the big map's ROM base (src: labels resolved by asm.assemble)."""
    p = assemble(src + EXIT, base=cfg.rom_base)
    segs = [(cfg.rom_base, b"".join(w.to_bytes(4, "little") for w in p.text))]
    segs += data or []
    return ref.PagedMachine.from_segments(cfg, p.entry, segs)


def test_ram_every_bank_and_image_fallback(big):
    """Stores in every window and many frames; byte / half ops at frame edges keep the image's
    other bytes; a load of an unwritten word of a mapped page reads the image."""
    lines = ["li s0, 0x10000", "lw s1, 4(s0)"]
    for pg in range(0, 16):
        a = 0x10000 + 512 * pg  # the image window at 2 << 15 is not writable: stores fault
    lines = ["li s1, 0x12345678"]
    for base in (0x100, 0x7000, 0x100000, 0x108000, 0x3F0000, 0x3F7FF0):
        lines += [f"li s0, {base}", "sw s1, 0(s0)", "sb s1, 5(s0)", "sh s1, 10(s0)", "lw a0, 0(s0)",
                  "lbu a1, 5(s0)", "lhu a2, 10(s0)", "lw a3, 4(s0)", "lb a4, 511(s0)",
                  "add a5, a0, a1", "putc_dummy"]
    src = "\n".join(lines).replace("putc_dummy", "addi zero, zero, 0")
    data = [(0x10000, bytes(range(1, 65)))]
    toks, m = run_check(big, lambda: big_machine(src, data=data))
    assert S.family(toks[-1])[0] == "HALT"
    assert sum(S.family(t)[0] == "PA" for t in toks) >= 6


def test_page_crossing_stack(big):
    """Stack-heavy code: a recursive call chain across frames (every frame gets a PA once)."""
    src = """
li sp, 0x3F7FF0
li a0, 300
jal ra, f
j done
f:
addi sp, sp, -32
sw ra, 28(sp)
sw a0, 24(sp)
beq a0, zero, leaf
addi a0, a0, -1
jal ra, f
leaf:
lw t0, 24(sp)
add a0, a0, t0
lw ra, 28(sp)
addi sp, sp, 32
ret
done:
nop"""
    toks, m = run_check(big, lambda: big_machine(src), kernel=False)
    assert m.x[10] == sum(range(301))
    assert sum(S.family(t)[0] == "PA" for t in toks) >= 18


def test_out_of_frames(big):
    """More pages than frames: the store faults, no silent wrap."""
    lines = ["li s0, 0x100000", "li s1, 512", "li s2, 200", "loop:", "sw s1, 0(s0)", "add s0, s0, s1",
             "addi s2, s2, -1", "bne s2, zero, loop"]
    toks, m = run_check(big, lambda: big_machine("\n".join(lines)), kernel=False)
    assert m.fault and S.FAULTS[m.fault[0]] == "store out of range"
    assert m.nf == BIG.frames - 1


def test_faults(big):
    for src, cause in (("li s0, 0x40000\nlw a0, 0(s0)", "load out of range"),
                       ("li s0, 0x10000\nsw a0, 0(s0)", "store out of range"),
                       ("li s0, 0x3\nlw a0, 0(s0)", "misaligned load")):
        toks, m = run_check(big, lambda s=src: big_machine(s))
        assert S.FAULTS[m.fault[0]] == cause


def far_program(cfg=BIG):
    """Code in several banks: far calls and returns across 64K-word windows, a fall-through
    across a window edge (RF), and words at slot 31 / row 127 (the spill tables)."""
    a = cfg.rom_base
    code = {}

    def put(addr, src):
        p = assemble(src, base=addr)
        code[addr] = p.text
    # main at 0: far jal to +300K words (JAL reaches +-256K words: two hops)
    far1 = a + 4 * 0x20000
    far2 = a + 4 * 0x3FFF0   # just below a window edge (W = 19: windows of 64K words)
    edge = a + 4 * (0x10000 - 3)  # falls across the first window
    slot31 = a + 4 * (0x1F + 0x1000)  # the first word is in slot 31
    row127 = a + 4 * 0xFE0  # row 127 of the first value head
    put(a, f"""
li a0, 0
li t0, {far1}
jalr ra, 0(t0)
li t0, {far2}
jalr ra, 0(t0)
li t0, {slot31}
jalr ra, 0(t0)
li t0, {edge}
jalr ra, 0(t0)
li t0, {row127}
jalr ra, 0(t0)
li a7, 93
ecall""")
    put(far1, """
addi a0, a0, 1
ret""")
    put(far2, "\n".join(["addi a0, a0, 2"] * 30) + "\nret")  # falls across the window
    put(slot31, "addi a0, a0, 4\naddi a0, a0, 8\nret")
    put(row127, "addi a0, a0, 256\nret")
    put(edge, "addi a0, a0, 16\naddi a0, a0, 32\naddi a0, a0, 64\naddi a0, a0, 128\nret")
    segs = [(ad, b"".join(w.to_bytes(4, "little") for w in txt)) for ad, txt in code.items()]
    return segs


def test_big_rom_across_banks(big):
    segs = far_program()
    mk = lambda: ref.PagedMachine.from_segments(BIG, BIG.rom_base, segs)
    toks, m = run_check(big, mk, kernel=False)
    assert m.exit_code == 1 + 60 + 12 + 240 + 256
    fams = [S.family(t)[0] for t in toks]
    assert "JF" in fams


def check_bank_forms(c, mat, toks):
    """Exec4's binary view of every bank (core.bank_table) selects exactly the rows the heads'
    own 128 query and key channels fire, at every position."""
    heads = {}
    for i, n in enumerate(c.nodes):
        if isinstance(n, ir.Head) and n.bank:
            heads.setdefault(n.bank, []).append(n)
    checked = 0
    assert set(heads) == set(c.banks)
    for g, ns in heads.items():
        form = c.banks[g]
        gate, head, row = (ir.ev(form[k], mat) for k in ("gate", "head", "row"))
        zrow = form["zrow"]
        n_ = mat.shape[0]
        ks = ir.ev(form["kgate"], mat) if "kgate" in form else np.zeros(n_)
        for n in ns:
            q = np.stack([ir.conv(ch, mat, False) for ch in n.q.values()], 1) > 0
            qi = np.array(list(n.q))
            want_on = (gate > 0) & (head == n.bank_h)
            for t in range(n_):
                got = set(qi[q[t]].tolist())
                exp = set()
                if want_on[t]:
                    exp = {int(row[t])} if int(row[t]) in n.q else set()
                    if zrow is not None:
                        exp.add(zrow)
                assert got == exp, (g, n.name, t, got, exp)
            if n.k:
                k = np.stack([ir.conv(ch, mat, False) for ch in n.k.values()], 1) > 0
                ki = np.array(list(n.k))
                for t in range(1, n_):
                    on = ks[t - 1] > 0 and head[t - 1] == n.bank_h
                    got = set(ki[k[t]].tolist())
                    assert got == ({int(row[t - 1])} if on else set()), (g, n.name, t, got)
            checked += 1
    return checked


def test_bank_forms_small(small):
    toks, _ = run_check(small, lambda: ref.PagedMachine.from_elf(S.small_cfg(), DATA / "c-crc32.elf"),
                        kernel=False, ntok=900)
    mat = ir.evaluate(small, toks, check=False)
    assert check_bank_forms(small, mat, toks) >= 80


def test_bank_forms_big(big):
    segs = far_program()
    mk = lambda: ref.PagedMachine.from_segments(BIG, BIG.rom_base, segs)
    toks, _ = run_check(big, mk, kernel=False)
    assert check_bank_forms(big, ir.evaluate(big, toks, check=False), toks) >= 80
    src = """
li s1, 0x12345678
li s0, 0x100000
sw s1, 0(s0)
sb s1, 700(s0)
lw a0, 0(s0)
lbu a1, 700(s0)
li s0, 0x3F0000
sh s1, 2(s0)
lw a2, 0(s0)"""
    toks, _ = run_check(big, lambda: big_machine(src), kernel=False)
    assert check_bank_forms(big, ir.evaluate(big, toks, check=False), toks) >= 80


def test_dense_fill_roundtrip():
    """Every word of a plan lands in exactly one slot, and reads back through the layout."""
    plan = core.dense_plan(list(range(19)), limit=40, name="t")
    rng = np.random.default_rng(1)
    addr = np.unique(rng.integers(0, 40 << 13, 40000))
    addr = np.concatenate([addr, np.arange(31, 1 << 13, 32), np.arange(127 << 5, 128 << 5)])
    addr = np.unique(addr)
    val = rng.integers(1, 1 << 32, len(addr))
    t = core.dense_fill(plan, addr, val)

    def read(a):
        got = []
        for leaf in plan:
            if not all(a >> b & 1 for b in leaf["gates"]):
                continue
            fld = lambda k: sum((a >> b & 1) << i for i, b in enumerate(leaf[k]))
            h, x, r, s_ = fld("kh"), fld("x"), fld("row"), fld("slot")
            if h >= (leaf["limit"] or 1 << len(leaf["kh"])):
                continue
            row = t[f'{leaf["name"]}.{h}'][x].get(r)
            if row is None or r == 127 or (leaf["kind"] == "dense" and s_ == 31):
                continue
            w = sum(int(row[2 + 4 * s_ + b]) << 8 * b for b in range(4))
            if w:
                got.append(w)
        return got

    for a, v in zip(addr.tolist(), val.tolist()):
        assert read(a) == [v], hex(a)
