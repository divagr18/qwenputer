"""The hot-code units on the paged map (chipformer/rv/hot.py): the reference against fx's original
instructions, and the circuit in the IR against the reference.

fx's memcpy / copy_fwd / memmove block (tests/data/rv/fx_units.json, row 11's words, PC-relative) is
relocated into a test ROM. Every program runs twice on the reference: with the units, and plain.
"""

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from chipformer.q35 import ir
from chipformer.rv import core, hot, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

DATA = Path(__file__).parent / "data" / "rv"
FX = json.loads((DATA / "fx_units.json").read_text())
BASE = S.MemCfg("hot", 22, 19, 40, ((0, 18), (0x3F0000, 15)), 128, (2, 3, 32), 0x3F7FF0)
BUF = 0x10000  # a read-only image window, then writable pages
CFG_IMG = 0x10000


RES = 0x18000  # eql results, one word per call
MASK_TAB = 0x18800  # 12 (ptr, len) pairs, written by the program


def prog_lines(calls, readback=()):
    src = ["_start:", "    j main"]
    src += ["fx_set:", "    addi sp, sp, -16", "    mv t0, a0", "    beq a2, zero, set_e", "set_l:", "    sb a1, 0(t0)",
            "    addi t0, t0, 1", "    addi a2, a2, -1", "    bne a2, zero, set_l", "set_e:", "    addi sp, sp, 16"]
    src += ["fx_set_ret:", "    ret"]
    mb, eq = FX["move_block"], FX["eql"]
    src.append("fx_block:")
    for i, w in enumerate(mb["words"]):
        if 4 * i == mb["memmove"]:
            src.append("fx_memmove:")
        if 4 * i == mb.get("memcpy", 0):
            src.append("fx_memcpy:")
        src.append(f"    .word {w:#x}")
    src.append("fx_eql:")
    src += [f"    .word {w:#x}" for w in eq["words"]]
    # the byte loops, in the form of fx's (the real ones are fuzzed on fx's image, fuzz2.py)
    src += ["fx_tok:", "    lbu a7, 0(s5)", "    addi t0, a7, -9", "    bltu t0, a4, tok_ws", "    beq a7, a5, tok_ws",
            "    addi a2, a2, 1", "    j tok_nx", "tok_ws:", "    addi t0, a2, 3", "    srli t0, t0, 2", "    add a1, a1, t0",
            "    li a2, 0", "tok_nx:", "    addi a6, a6, -1", "    addi s5, s5, 1", "    bne a6, zero, fx_tok", "    ret"]
    src += ["fx_chr:", "    lbu a3, 0(a1)", "    beq a3, a0, chr_x", "    addi a2, a2, -1", "    addi a1, a1, 1",
            "    bne a2, zero, fx_chr", "chr_x:", "    ret"]
    src += ["fx_ascii:", "    li t5, 1", "    add a0, t5, a0", "    bgeu a0, s3, asc_x", "    add t4, a0, s4", "    lb t5, 0(t4)",
            "    bge t5, zero, fx_ascii", "asc_x:", "    ret"]
    src += ["fx_js:", "    addi a3, a3, 1", "    sw a3, 36(a2)", "    beq a4, a3, js_x", "    add a1, a0, a3", "    lbu a1, 0(a1)",
            "    li a5, 32", "    bltu a1, a5, js_x", "    li a5, 128", "    bgeu a1, a5, js_x", "    li a5, 34", "    beq a1, a5, js_x",
            "    li a5, 92", "    bne a1, a5, fx_js", "js_x:", "    ret"]
    src += ["fx_sha:", "    ret", "fx_sha_ret:", "    ret"]
    src += ["fx_jesc:", "    addi s8, s8, 1", "    lw s11, -60(s0)", "    beq s8, s11, je_x", "    add s11, s8, s5",
            "    lbu a1, 0(s11)", "    beq a1, s3, je_x", "    bne a1, s7, je_t", "    j je_x", "je_t:", "    bltu s7, a1, fx_jesc",
            "    andi a2, a1, 126", "    li a3, 32", "    beq a2, a3, fx_jesc", "    addi a2, a1, 221", "    andi a2, a2, 255",
            "    li a3, 56", "    bgeu a3, a2, fx_jesc", "je_x:", "    ret"]
    fold = lambda r: [f"    addi a6, {r}, 191", "    andi a6, a6, 254", "    sltiu a6, a6, 26", "    slli a6, a6, 5", f"    or {r}, a6, {r}"]
    src += ["fx_mask:", "    li s11, 0", "    j mk_head", "mk_next:", "    addi s11, s11, 1", "    li a0, 12", "    beq s11, a0, mk_exit",
            "mk_head:", "    slli a1, s11, 3", "    add a1, a1, s5", "    lw a0, 0(a1)", "    lw a1, 4(a1)", "    beq s6, zero, mk_key",
            "mk_cmp:", "    add a4, a1, s8", "    bgeu a4, s3, mk_next", "    mv a2, s10", "mk_l:", "    lbu a3, 0(a2)", "    lbu a5, 0(a0)"]
    src += fold("a3") + fold("a5")
    src += ["    bne a3, a5, mk_next", "    addi a2, a2, 1", "    addi a1, a1, -1", "    addi a0, a0, 1", "    bne a1, zero, mk_l",
            "    j mk_match", "mk_key:", "    lbu a2, -1(s10)", "    beq a2, s4, mk_next", "    addi a3, a2, 208", "    andi a3, a3, 254",
            "    bltu a3, a7, mk_next", "    andi a2, a2, 223", "    addi a2, a2, 191", "    andi a2, a2, 254", "    li a3, 26",
            "    bltu a2, a3, mk_next", "    j mk_cmp", "mk_exit:", "    ret", "mk_match:", "    ret"]
    src += ["main:", f"    li sp, {BASE.sp:#x}"]
    if any(cc[0] == "maskp" for cc in calls):
        for i in range(12):  # every prefix: 4 bytes at 0x14F00
            src += [f"    li t0, {MASK_TAB + 8 * i:#x}", "    li t1, 0x14F00", "    sw t1, 0(t0)", "    li t1, 4", "    sw t1, 4(t0)"]
    k = 0
    for cc in calls:
        if cc[0] in ("move", "cpy"):
            d, s, n = cc[1:4]
            lbl = {"move": "fx_memmove", "cpy": "fx_memcpy"}[cc[0]]
            src += [f"    li a0, {d:#x}", f"    li a1, {s:#x}", f"    li a2, {n}", f"    call {lbl}"]
        elif cc[0] == "sha":
            inst, st, blk, wsp = cc[1:5]
            src += [f"    li t0, {inst:#x}", f"    li t1, {wsp:#x}", "    sw t1, 8(t0)", f"    li a0, {inst:#x}",
                    f"    li a1, {st:#x}", f"    li a2, {blk:#x}", "    call fx_sha"]
        elif cc[0] == "tok":
            p_, n, run = cc[1:4]
            src += [f"    li s5, {p_:#x}", f"    li a6, {n}", f"    li a2, {run}", "    li a1, 0", "    li a4, 5", "    li a5, 32",
                    "    call fx_tok", f"    li t0, {RES + 4 * k:#x}", "    sw a2, 0(t0)", "    sw a1, 4(t0)", "    sw a6, 8(t0)"]
            k += 3
        elif cc[0] == "chr":
            p_, v, n = cc[1:4]
            src += [f"    li a1, {p_:#x}", f"    li a0, {v}", f"    li a2, {n}", "    call fx_chr", f"    li t0, {RES + 4 * k:#x}",
                    "    sw a1, 0(t0)", "    sw a2, 4(t0)"]
            k += 2
        elif cc[0] == "ascii":
            b_, i, e = cc[1:4]
            src += [f"    li s4, {b_:#x}", f"    li a0, {i}", f"    li s3, {e}", "    call fx_ascii", f"    li t0, {RES + 4 * k:#x}",
                    "    sw a0, 0(t0)"]
            k += 1
        elif cc[0] == "jscan":
            b_, i, e, cur = cc[1:5]
            src += [f"    li a0, {b_:#x}", f"    li a3, {i}", f"    li a4, {e}", f"    li a2, {cur:#x}", "    call fx_js",
                    f"    li t0, {RES + 4 * k:#x}", "    sw a3, 0(t0)"]
            k += 1
        elif cc[0] == "jesc":
            b_, i, e, a3 = cc[1:5]
            src += [f"    li t0, {0x18C00:#x}", f"    li t1, {e}", "    sw t1, 0(t0)", f"    li s0, {0x18C00 + 60:#x}", f"    li s5, {b_:#x}",
                    f"    li s8, {i}", "    li s3, 34", "    li s7, 92", f"    li a3, {a3}", "    call fx_jesc",
                    f"    li t0, {RES + 4 * k:#x}", "    sw s8, 0(t0)", "    sw a3, 4(t0)", "    li s0, 0", "    li s3, 0", "    li s5, 0", "    li s7, 0", "    li s8, 0"]
            k += 2
        elif cc[0] == "maskp":
            p_, i, n, s6 = cc[1:5]
            src += [f"    li s10, {p_:#x}", f"    li s5, {MASK_TAB:#x}", f"    li s8, {i}", f"    li s3, {n}", f"    li s6, {s6}", "    li s4, 95",
                    "    li a7, 10", "    li a4, 111", "    li a5, 222", "    li a6, 333", "    call fx_mask", f"    li t0, {RES + 4 * k:#x}",
                    "    sw a4, 0(t0)", "    sw a5, 4(t0)", "    sw a6, 8(t0)", "    li s10, 0", "    li s5, 0", "    li s8, 0", "    li s3, 0", "    li s6, 0", "    li s4, 0",
                    "    li a7, 0", "    li s11, 0"]
            k += 3
        elif cc[0] == "set":
            d, v, n = cc[1:4]
            src += [f"    li a0, {d:#x}", f"    li a1, {v}", f"    li a2, {n}", "    call fx_set"]
        elif cc[0] == "bcopy":  # fx's realloc loop; its registers are dead after it there
            d, s, n = cc[1:4]
            src += [f"    li a2, {d:#x}", f"    li s11, {s:#x}", f"    li a1, {s + n:#x}", "bloop_%d:" % k,
                    "    lbu a3, 0(s11)", "    addi s11, s11, 1", "    sb a3, 0(a2)", "    addi a2, a2, 1"]
            src += [f"    bne s11, a1, bloop_{k}", "bexit_%d:" % k, "    li a2, 0", "    li s11, 0", "    li a3, 0"]
            k += 1
        else:
            a, n, b, bn = cc[1:5]
            src += [f"    li a1, {a:#x}", f"    li a2, {n}", f"    li a3, {b:#x}", f"    li a4, {bn}",
                    "    call fx_eql", f"    li t0, {RES + 4 * k:#x}", "    sw a0, 0(t0)"]
            k += 1
    if readback:  # the words a run changed, through loads: their values are tokens the circuit predicts
        for a in readback:
            src += [f"    li t0, {a << 2:#x}", "    lw t1, 0(t0)"]
    src += ["    li a7, 93", "    ecall"]
    return src


def build_prog(calls, seed=0, readback=()):
    rng = random.Random(seed)
    p = assemble("\n".join(prog_lines(calls, readback)), base=BASE.rom_base)
    image = rng.randbytes(0x10000)
    return p, [(BASE.rom_base, b"".join(w.to_bytes(4, "little") for w in p.text)), (CFG_IMG, image)]


def machine(calls, units=True, seed=0, readback=()):
    p, segs = build_prog(calls, seed, readback)
    cfg = BASE
    if units:
        mb, eq = FX["move_block"], FX["eql"]
        base = p.labels["fx_block"]
        ret = base + mb["ret"]
        assert p.text[(ret - BASE.rom_base) >> 2] == 0x00008067
        us = [hot.Unit("move", p.labels["fx_memmove"], ret), hot.Unit("cpy", p.labels["fx_memcpy"], ret),
              hot.Unit("set", p.labels["fx_set"], p.labels["fx_set_ret"]),
              hot.Unit("eql", p.labels["fx_eql"] + eq["trigger"], p.labels["fx_eql"] + eq["ret"]),
              hot.Unit("tok", p.labels["fx_tok"], p.labels["fx_tok"]), hot.Unit("chr", p.labels["fx_chr"], p.labels["fx_chr"]),
              hot.Unit("ascii", p.labels["fx_ascii"], p.labels["fx_ascii"]),
              hot.Unit("jscan", p.labels["fx_js"], p.labels["fx_js"]),
              hot.Unit("sha", p.labels["fx_sha"], p.labels["fx_sha_ret"]),
              hot.Unit("jesc", p.labels["fx_jesc"], p.labels["fx_jesc"]),
              hot.Unit("maskp", p.labels["fx_mask"], p.labels["mk_exit"], MASK_TAB)]
        for k in range(len(calls)):
            if f"bloop_{k}" in p.labels:
                us.append(hot.Unit("bcopy", p.labels[f"bloop_{k}"], p.labels[f"bexit_{k}"]))
        cfg = replace(BASE, units=tuple(us[:15]))
    return ref.PagedMachine.from_segments(cfg, p.entry, segs), cfg


def end_state(m, lo=0, hi=0x40000):
    """The memory the program can see, minus the stack; and the callee-saved registers."""
    return ({a: v for a, v in m.wr.items() if v != m.image.get(a, 0) and a < hi >> 2 and a >= lo >> 2},
            [m.x[r] for r in (8, 9, 18, 22, 23, 24, 25, 26, 27)], m.x[2])


def run(calls, seed=0):
    a, cfg = machine(calls, True, seed)
    ta = a.run(10**7)
    b, _ = machine(calls, False, seed)
    tb = b.run(10**7)
    assert end_state(a) == end_state(b)
    return ta, tb, a, cfg


FIXED = [("move", 0x11000, 0x12000, 16), ("move", 0x11004, 0x12000, 84), ("move", 0x11001, 0x12003, 37),
         ("move", 0x11003, 0x12000, 3), ("move", 0x11001, 0x12001, 2), ("move", 0x11002, 0x11003, 200),
         ("move", 0x11203, 0x11200, 300), ("move", 0x111F0, 0x12000, 64), ("move", 0x12000, 0x10100, 700),
         ("cpy", 0x13001, 0x12000, 100), ("cpy", 0x13400, 0x13401, 40), ("cpy", 0x12001, 0x12000, 40),
         ("set", 0x14001, 0x5A, 70), ("set", 0x14100, 0, 8), ("set", 0x141FE, 0xFF, 1), ("set", 0x14300, 1, 700),
         ("eql", 0x10000, 8, 0x10000, 8), ("eql", 0x13001, 40, 0x13001, 40), ("eql", 0x10003, 9, 0x10100, 9),
         ("eql", 0x13401, 40, 0x13400, 40), ("eql", 0x10000, 3, 0x10000, 4), ("eql", 0x12000, 100, 0x12000 + 700, 100),
         ("bcopy", 0x15002, 0x10010, 65), ("bcopy", 0x15400, 0x15401, 30),
         ("set", 0x16001, 0x41, 300), ("set", 0x16080, 0x20, 1), ("set", 0x16120, 0x20, 1), ("set", 0x16100, 0xE0, 1),
         ("tok", 0x16002, 200, 3), ("tok", 0x16082, 180, 0), ("tok", 0x16002, 1, 0),
         ("chr", 0x16003, 0x20, 200), ("chr", 0x16003, 0x7A, 20), ("chr", 0x16081, 0x41, 3),
         ("ascii", 0x16000, 5, 90), ("ascii", 0x16000, 0x90, 0x200), ("ascii", 0x16000, 3, 4),
         ("set", 0x16200, 0x22, 1), ("jscan", 0x16000, 7, 200, 0x17000), ("jscan", 0x16000, 7, 0x8C, 0x17201 - 1), ("jscan", 0x16000, 6, 7, 0x17000),
         ("set", 0x16400, 0x61, 120), ("set", 0x16410, 0x20, 1), ("set", 0x16420, 0x41, 3), ("set", 0x16470, 0x22, 1),
         ("jesc", 0x16400, 2, 200, 7), ("jesc", 0x16400, 0x6F, 200, 7), ("jesc", 0x16400, 2, 0x18, 0), ("jesc", 0x16400, 0x21, 0x60, 9),
         ("set", 0x16421, 0x41, 60), ("jesc", 0x16400, 2, 200, 7),
         ("set", 0x14F00, 0x7A, 4), ("maskp", 0x16402, 3, 500, 0), ("maskp", 0x16411, 3, 500, 0), ("maskp", 0x16402, 3, 500, 1),
         ("maskp", 0x16402, 3, 5, 0)]


def scan_calls(rng):
    """Text with a stop byte, then one of the byte loops over it."""
    p = 0x16000 + rng.randrange(0x1000)
    n = rng.choice([1, 2, 3, 5, 8, 30, 61, 62, 63, 64, 90, 140])
    cut = rng.randrange(n + 2)
    calls = [("set", p, rng.choice([0x41, 0x7A, 0x30]), n + 4)]
    kd = rng.choice(["tok", "chr", "ascii", "jscan", "jesc"])
    if kd == "jesc":
        calls = [("set", p, rng.choice([0x61, 0x41, 0x7A]), n + 4)]
        for _ in range(rng.choice([0, 1, 3])):
            calls.append(("set", p + rng.randrange(n + 4), rng.choice([0x20, 0x21, 0x23, 0x5B, 0x5D, 0xC3]), 1))
        if cut < n + 4:
            calls.append(("set", p + cut, rng.choice([0x22, 0x5C, 0x1F]), 1))
        calls.append(("jesc", p, rng.randrange(4), rng.choice([n + 3, n, 2]), rng.choice([0, 32, 7])))
        return calls
    stop = {"tok": rng.choice([0x20, 9, 13]), "chr": 0x0A, "ascii": rng.choice([0x80, 0xC3]),
            "jscan": rng.choice([0x22, 0x5C, 0x1F, 0x80])}[kd]
    if cut < n + 4:
        calls.append(("set", p + cut, stop, 1))
    if kd == "tok":
        calls.append(("tok", p, n, rng.randrange(9)))
    elif kd == "chr":
        calls.append(("chr", p, 0x0A, n))
    else:
        i = rng.randrange(n + 1)
        calls.append((kd, p, i, n + 1) if kd == "ascii" else (kd, p, i, n + 1, 0x18400 + 4 * rng.randrange(200)))
    return calls


def random_calls(rng, k=5, kinds=("move", "move", "cpy", "set", "eql", "bcopy", "scan")):
    calls = []
    for _ in range(k):
        if rng.choice(kinds) == "scan":
            calls += scan_calls(rng)
            continue
        n = rng.choice([1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 64, 100, 513])
        d = 0x11000 + rng.randrange(0x4000)
        s = rng.choice([0x10000 + rng.randrange(0x8000), d + rng.randint(-40, 40)])
        kd = rng.choice([k_ for k_ in kinds if k_ != "scan"])
        if kd == "set":
            calls.append(("set", d, rng.randrange(256), n))
        elif kd == "eql":
            b = rng.choice([0x10000 + rng.randrange(0x8000), d])
            calls.append(("eql", d, n, b, n))
            if rng.random() < 0.5:  # equal bytes up to a point
                calls.insert(-1, ("move", b, d, max(1, n - rng.randrange(3))) if abs(b - d) > n + 8 else ("set", d, 1, 1))
        elif kd == "bcopy":
            calls.append(("bcopy", d, s, n))
        else:
            calls.append((kd, d, s, n))
    return calls


def test_fixed_reference():
    ta, tb, m, cfg = run(FIXED)
    assert any(S.family(t)[0] == "M" for t in ta)
    assert len(ta) < len(tb)


def test_reference_random():
    rng = random.Random(1)
    for k in range(80):
        run(random_calls(rng, 4), k)


@pytest.fixture(scope="module")
def circ():
    m, cfg = machine(FIXED)
    return core.build(cfg=cfg), cfg


def check_ir(c, calls, seed=0, kernel=True):
    m0 = machine(calls, False, seed)[0]
    m0.run(10**7)
    rb = sorted(a for a, v in m0.wr.items() if a < 0x40000 >> 2 and v != m0.image.get(a, 0))[:400]

    def mk():
        m, cfg = machine(calls, True, seed, rb)
        assert cfg.units == c.cfg.units
        return m

    core.snapshot(c, mk())
    toks = mk().run(10**7)
    mat = ir.evaluate(c, toks, check=True, kernel=kernel)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:6]
    return toks


def test_ir_fixed(circ):
    c, cfg = circ
    check_ir(c, FIXED, kernel=False)


def test_ir_pruned():
    """The ROM-specialized build: logic of uops the ROM does not hold is left out."""
    m, cfg = machine(FIXED)
    cfg = replace(cfg, absent=hot.rom_absent(m.rom, cfg))
    c = core.build(cfg=cfg)
    assert "BMAC" in c.absent and c.stats()["ffn_units"] < 40000

    def mk():
        m2 = machine(FIXED, True, 0)[0]
        m2.cfg = cfg
        return m2

    core.snapshot(c, mk())
    toks = mk().run(10**7)
    pred = core.predictions(c, ir.evaluate(c, toks, check=True, kernel=False))
    assert [p for p, t in zip(pred, toks[1:]) if p != t] == []


@pytest.fixture(scope="module")
def circ4():
    return core.build(cfg=machine([("move", 0x11000, 0x12000, 16)])[1])


@pytest.mark.parametrize("seed", range(12))
def test_ir_random(circ4, seed):
    """Random calls of the kinds without a per-call trigger (bcopy's PC is the program's)."""
    c = circ4
    rng = random.Random(100 + seed)
    check_ir(c, random_calls(rng, 5, ("move", "cpy", "set", "eql", "scan", "scan")), seed, kernel=False)


def test_unit_weights_fit_bf16(circ4):
    """Every weight chipbuild emits for the units' circuit is a finite bf16 value, so it can be a
    chip file (docs/fx/units-file.md). Without the step counter's 2^16 (hot.SHA_SCALE), SHA-256's
    thermometer over 84 steps has gate weights of -2^128 and -2^129 at this layout: past bf16's and
    fp32's range."""
    from chipformer.rv import chipbuild as CB

    c = circ4
    assert "sha" in {u.kind for u in c.cfg.units}
    al, lay = CB.standalone_layout(c)
    tc = dict(hidden_size=lay.hidden, linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads, intermediate_size=c.target.intermediate,
              linear_conv_kernel_dim=4)
    sd = CB.emit_sparse(c, lay, tc, al, set(c.keep))  # asserts every weight is a bf16 value
    big = max(abs(v) for t in sd.values() for v in t.d.values())
    assert big < 2.0**120, big


def test_ir_sha():
    """The compression, circuit against reference against hashlib's state after one block."""
    calls = [("sha", 0x11000, 0x12000, 0x10040, 0x12000), ("sha", 0x11000, 0x13440, 0x10080, 0x13440),
             ("sha", 0x11000, 0x14480, 0x10080, 0x14460)]
    m = machine(calls)[0]
    toks = m.run(10**7)
    assert sum(S.tok("CW") <= t < S.tok("CW") + 65536 for t in toks) > 1200
    h = [hot.K256[0]]
    assert h
    c = core.build(cfg=machine(calls)[1])
    check_ir(c, calls, kernel=False)
