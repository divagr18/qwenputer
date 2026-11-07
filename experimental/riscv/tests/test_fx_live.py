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
