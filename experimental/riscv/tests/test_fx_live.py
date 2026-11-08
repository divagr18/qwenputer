"""The live I/O path (chipformer/rv/fxlive.py, docs/fx/live.md) on the circuit, in the IR: tokens
only, against the reference. The GPU's version of this is fxlivecheck.py."""

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import core, ref
from chipformer.rv import fxlive as FL
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

BIG = S.MemCfg("big", 22, 19, 40, ((0, 15), (0x100000, 16), (0x3F0000, 15)), 64, (2, 3, 32), 0x3F8000)
BIOS = FL.Bios(0x48000)
EA = 0x100000 + 512 - 40  # a capture that crosses a page edge, unaligned at both ends


class IrBackend:
    """As ExecBackend: forced tokens, then predictions until a stop, on the IR."""

    predicts = True

    def __init__(self, c):
        self.c, self.carry = c, {}

    def run(self, toks, max_steps):
        ids, step = [], 0
        cur = list(toks)
        while True:
            t = cur[step] if step < len(cur) else ids[-1]
            mat = ir.evaluate(self.c, [t], check=False, carry=self.carry)
            ids.append(core.predictions(self.c, mat)[0])
            step += 1
            if step < len(cur):
                continue
            p = ids[-1]
            if step >= max_steps or p < 0 or S.family(p)[0] in FL.STOP_FAMS:
                return np.asarray(ids, np.int64)


SRC = f"""
    li t0, 0x100
    li t1, {EA + 3}
    sw t1, 0(t0)
    li t1, 601
    sw t1, 4(t0)
    li s1, 0x7E57
    li s2, 0x600D
    li a0, 0x100
    li a1, 2
    li a7, 1025
    ecall
    mv s3, a0
    li t0, {EA + 3}
    lbu s4, 0(t0)
    lbu s5, 600(t0)
    add a0, s4, s5
    add a0, a0, s3
    li a7, 1026
    li a1, 0
    ecall
"""
EXIT = "\nli a7, 93\necall"


def machine():
    p = assemble(SRC + EXIT, base=BIG.rom_base)
    segs = [(BIG.rom_base, b"".join(w.to_bytes(4, "little") for w in p.text))]
    m = ref.PagedMachine.from_segments(BIG, p.entry, segs)
    BIOS.install(m)
    return m


@pytest.fixture(scope="module")
def session():
    c = core.build(cfg=BIG)
    core.snapshot(c, machine())
    cpu = FL.LiveCPU(IrBackend(c), BIOS, BIG, full_regs=True)
    data = bytes((i * 7 + 1) & 0xFF for i in range(601))
    trace = []
    chip, stop = cpu.go()
    while True:
        kind, pc = cpu.parse_stop(chip, stop)
        regs = cpu.enter()
        if kind == "exit":
            trace.append(("exit", regs))
            break
        a7 = regs[FL.A7]
        a0 = regs[FL.A0]
        if a7 == 1025:
            assert regs[FL.A1] == 2
            p, n = cpu.read_words(a0, 2)
            cpu.write(p, data[:n])
            assert cpu.read(p, n) == data[:n]
            res = 0x5A5A5
        else:
            trace.append(("sum", a0))
            res = a0
        trace.append(("regs", dict(regs)))
        chip, stop = cpu.resume(pc + 4, a7, res)
    return cpu, trace, data


def test_host_call_round_trip(session):
    cpu, trace, data = session
    assert ("sum", data[0] + data[600] + 0x5A5A5) in trace


def test_registers_survive(session):
    _, trace, _ = session
    regs = [t[1] for t in trace if t[0] == "regs"][1]
    assert regs[9] == 0x7E57 and regs[18] == 0x600D  # s1, s2 across the first call
    assert regs[17] == 1026


def test_stream_against_reference(session):
    cpu, _, data = session
    m = machine()
    r = FL.audit(m, *cpu.fed())
    assert r["wrong"] is None, r
    assert r["chip_tokens"] > 100 and r["forced_tokens"] > 100
    p = EA + 3
    got = bytes(m.ram_word(a & ~3) >> 8 * (a & 3) & 0xFF for a in range(p, p + len(data)))
    assert got == data
    # the partial first and last words kept their other bytes; a page was mapped by the probe
    assert m.ram_word(p & ~3) & 0xFFFFFF == 0
    assert m.nf == 4 and cpu.nf == 4  # the args page by the guest, three capture pages by probes


def test_a_wrong_chip_token_is_reported(session):
    cpu, _, _ = session
    toks, lens, forced = cpu.fed()
    toks = toks.copy()
    i = int(np.nonzero(np.repeat(~forced, lens))[0][40])
    toks[i] += 1
    r = FL.audit(machine(), toks, lens, forced)
    assert r["wrong"] is not None


def test_a_launch_boundary_loses_no_token():
    """A run cut into launches gives the same chip tokens (the last prediction of a launch is the
    next one's first row), so the instruction count stays right."""
    out = []
    for chunk in (1 << 20, 7):
        m = machine()
        cpu = FL.LiveCPU(FL.RefBackend(m), BIOS, BIG)
        cpu.chunk = chunk
        chip, stop = cpu.go()
        out.append((chip.tolist(), stop))
        cpu.parse_stop(chip, stop)
        out.append(cpu.instr)
    assert out[0] == out[2] and out[1] == out[3]


def test_capture_is_one_token_a_byte():
    """A 3.3 KB reply (the largest real one, docs/fx/io.md) as SS records, no window."""
    n = 830
    toks = sum((FL.word_tokens(0x41424344, 0) for _ in range(n)), [])
    assert len(toks) == 4 * n and len(toks) / (4 * n) == 1.0


# ---- the hot-code units and the text bridge on the same path (docs/fx/one.md) -----------------

def test_units_with_the_io_rom():
    """A build with units and the I/O ROM (BPRD stays in, every other digit op is pruned): the
    units fire, then the host reads a copy's result and the registers through tokens."""
    from dataclasses import replace

    import test_rv_hot as TH
    from chipformer.rv import hot

    calls = [("move", 0x11001, 0x12003, 37), ("set", 0x14001, 0x5A, 70), ("eql", 0x10003, 9, 0x10100, 9)]
    bios = FL.Bios(0x48000)
    m, cfg = TH.machine(calls)
    bios.install(m)
    cfg = replace(cfg, absent=hot.rom_absent(m.rom, cfg))
    assert "BPRD" not in cfg.absent and "BMAC" in cfg.absent
    m.cfg = cfg
    c = core.build(cfg=cfg)
    core.snapshot(c, m)

    def mk():
        m2, _ = TH.machine(calls)
        bios.install(m2)
        m2.cfg = cfg
        return m2

    ref_end = mk()
    cpu = FL.LiveCPU(IrBackend(c), bios, cfg, full_regs=True)
    chip, stop = cpu.go()
    fams = {S.family(int(t))[0] for t in chip}
    assert "M" in fams and "CW" in fams
    kind, _ = cpu.parse_stop(chip, stop)
    assert kind == "exit"
    regs = cpu.enter()
    a = FL.audit(mk(), *cpu.fed())
    assert a["wrong"] is None, a
    ref_end.run(10**7)
    assert all(regs[r] == ref_end.x[r] for r in regs)
    got = cpu.read_words(0x11000, 12)
    assert got == [ref_end.ram_word(0x11000 + 4 * i) for i in range(12)]
    # and a host write onto a page a unit mapped
    cpu.write(0x14000, b"\x01\x02\x03\x04")
    cpu.cache.clear()
    assert cpu.read(0x14000, 4) == b"\x01\x02\x03\x04"


@pytest.fixture(scope="module")
def T():
    from chipformer.rv import bridge as BR

    try:
        return BR.Text(BR.default_tokenizer())
    except FileNotFoundError:
        pytest.skip("no Qwen3.5 tokenizer in the HF cache")


def text_cfg():
    from dataclasses import replace

    return replace(BIG, text_addr=0x7F0000)


def test_text_buffer_cannot_collide():
    from dataclasses import replace

    with pytest.raises(AssertionError):  # the legacy address is inside this map's first window
        replace(BIG, text_addr=0x4000)


def test_text_bridge_on_the_paged_map(T):
    """A tool call's text reaches the machine at the paged map's buffer; </tool_call> starts the run;
    every token the circuit owns is right."""
    from chipformer.rv import bridge as BR

    cfg = text_cfg()
    src = f"""
    li a7, 93
    ecall
    li t0, {cfg.text_addr:#x}
    lbu a0, 16(t0)
    li a7, 1
    ecall
    lbu a0, 17(t0)
    li a7, 1
    ecall
    lw s1, 16(t0)
    li a7, 93
    ecall
"""

    def mk():
        p = assemble(src, base=cfg.rom_base)
        m = ref.PagedMachine.from_segments(cfg, p.entry, [(cfg.rom_base, b"".join(w.to_bytes(4, "little") for w in p.text))])
        BR.park(m)
        return m

    c = core.build(text=T, cfg=cfg)
    m0, m = mk(), mk()
    ses = BR.Session(T, m)
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    ses.qwen(T.encode("<tool_call>hi</tool_call>"))
    ses.qwen(T.encode("done"))
    assert T.decode_bytes(ses.runs[-1]).startswith(T.bytes[T.encode("hi")[0]][:2])
    assert m.x[9] != 0
    core.snapshot(c, m0)
    mat = ir.evaluate(c, ses.toks)
    pred = core.predictions(c, mat)
    own = np.rint(ir.ev(c.owner, mat)).astype(int)
    assert [i for i in range(len(ses.toks)) if own[i] != int(ses.own[i])] == []
    assert [i for i in range(len(ses.toks) - 1) if ses.own[i] and pred[i] != ses.toks[i + 1]] == []


def test_a_parked_machine_ignores_text(T):
    """The live path with the bridge built in: at an ecall the machine parks, Qwen's text (with a
    tool call in it) goes through the circuit, and fx's registers and memory are as they were."""
    cfg = text_cfg()
    c = core.build(text=T, cfg=cfg)

    def mk():
        p = assemble(SRC + EXIT, base=cfg.rom_base)
        segs = [(cfg.rom_base, b"".join(w.to_bytes(4, "little") for w in p.text))]
        m = ref.PagedMachine.from_segments(cfg, p.entry, segs)
        BIOS.install(m)
        return m

    core.snapshot(c, mk())

    class Ir(IrBackend):
        def text(self, qids, dbg=False):
            ir.evaluate(self.c, [S.TEXT_TOKEN | q for q in qids], check=False, carry=self.carry)
            return None, None

    cpu = FL.LiveCPU(Ir(c), BIOS, cfg, full_regs=True)
    data = bytes((i * 3 + 1) & 0xFF for i in range(601))
    chip, stop = cpu.go()
    seen = []
    while True:
        kind, pc = cpu.parse_stop(chip, stop)
        regs = cpu.enter()
        if kind == "exit":
            break
        a7, a0 = regs[FL.A7], regs[FL.A0]
        res = a0
        if a7 == 1025:
            p, n = cpu.read_words(a0, 2)
            cpu.park()
            cpu.text(T.encode("Sure.<tool_call>\nabc\n</tool_call>\n<|im_end|>"))
            cpu.unpark()
            cpu.write(p, data[:n])
            res = 0x5A5A5
        else:
            seen.append(a0)
        chip, stop = cpu.resume(pc + 4, a7, res)
    assert seen == [data[0] + data[600] + 0x5A5A5]
    r = FL.audit(mk(), *cpu.fed())
    assert r["wrong"] is None, r
