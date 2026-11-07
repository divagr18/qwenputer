"""fx's I/O through the token stream (docs/fx/live.md): no state pokes after the boot snapshot.

An ecall the circuit does not know prints its fault record, FLT(7) HEX x 8, and the host takes
the machine from there with tokens it feeds as model input, as text is fed to Qwen. Three facts of
the circuit make that enough, none of them new:

  - a forced J / V JF token moves the PC. Its record writes rd of the instruction the PC stands
    on, so the host only jumps from an instruction with no rd: the ecall, BPRD or a store.
  - a forced value record (F0 | V F1 | V V F2) on an instruction with an rd writes that register:
    a commit takes its value from the token's payload.
  - a forced V V V SSx on a SH writes the whole word from the V payloads, into the slot the SH's
    EA names. Word slots, frames and copy-on-write are the circuit's own.

The I/O ROM (`Bios`) is a table of instructions after fx's code in the ROM's fetch space. It is
not in the ELF, and no fx word moves:

  emit       addi rX, rX, 0            the chip prints a register as its value record
  read sled  lw a7, imm(a0)            the chip prints a memory word as its value record
  write sled sh x0, imm(a0)            the host's V V V SSx writes a memory word
  set        addi rX, rX, 0            the host's forced value record sets a register
  probe      lw a7, 0(a0); sw a7, 0(a0)   a page's frame: PA(f) if unmapped, ST0 / ST1 (see below)
  stop       BPRD                      its BYTE prediction ends a launch; never fed, never commits

Which ends a launch: the live decode table gives HALT and BYTE no embedding row (Exec4's lut), so
the kernel stops there with the token as its last prediction and does not run it. Everything else
is a normal launch: forced rows, then the circuit's own predictions.

Who predicts what: read and emit tokens are the chip's own. Forced tokens are the host's input.
SSx needs the slot's frame parity. Only the circuit's page table knows the frame, so the first
host write to a page probes it: the chip reads a word and stores it back, which maps the page as
a guest store would and prints the parity as ST0 / ST1. The stored value is the word it read.

The capture window (docs/fx/io.md blocker 1) goes with this: a reply is a stream of SS records
into paged RAM, 1 token a byte, as long as fx's own buffer.

`audit` walks a run's token log on the reference machine: every chip token must equal the
reference's, and every forced token is applied as above. It is an offline check: nothing in the
production path reads the reference.
"""

from __future__ import annotations

import time

import numpy as np

from chipformer.rv import core, elf, fxrun, ref
from chipformer.rv import spec as S

A0, A1, A7 = 10, 11, 17
EMIT_ORDER = [A7, A0, A1] + [r for r in range(1, 32) if r not in (A0, A1, A7)]
SLED = 1024  # 4 KB of words around a0: imm -2048 .. 2044
FX_IO_BASE = 0x1BE000  # fx's I/O ROM, as a ROM word index: after .text, inside the 1.835M-word ROM
CHUNK = 1 << 20  # chip tokens a launch at most (the kernel's token buffer)
STOP_FAMS = ("HALT", "BYTE")


def i_addi(rd: int, rs: int, imm: int = 0) -> int:
    return (imm & 0xFFF) << 20 | rs << 15 | rd << 7 | 0x13


def i_lw(rd: int, rs: int, imm: int) -> int:
    return (imm & 0xFFF) << 20 | rs << 15 | 2 << 12 | rd << 7 | 0x03


def _store(f3: int, rs2: int, rs1: int, imm: int) -> int:
    i = imm & 0xFFF
    return (i >> 5) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (i & 31) << 7 | 0x23


def i_sh(rs2: int, rs1: int, imm: int) -> int:
    return _store(1, rs2, rs1, imm)


def i_sw(rs2: int, rs1: int, imm: int) -> int:
    return _store(2, rs2, rs1, imm)


I_STOP = 0x2B  # BPRD d0[0]


class Bios:
    """The I/O ROM's layout, at ROM word index ``base`` (inside one 64K-word window)."""

    E_SHORT, E_FULL, SET_A0, SET_RET, PROBE = 0, 4, 40, 42, 45
    RSLED, WSLED = 512, 2048
    WORDS = 2048 + SLED + 1

    def __init__(self, base: int) -> None:
        assert base >> 16 == (base + self.WORDS - 1) >> 16, "the I/O ROM crosses a PC window"
        self.base = base

    def words(self) -> list[int]:
        w = [0] * self.WORDS
        for i, r in enumerate(EMIT_ORDER[:3]):
            w[self.E_SHORT + i] = i_addi(r, r)
        w[self.E_SHORT + 3] = I_STOP
        for i, r in enumerate(EMIT_ORDER):
            w[self.E_FULL + i] = i_addi(r, r)
        w[self.E_FULL + 31] = I_STOP
        w[self.SET_A0], w[self.SET_A0 + 1] = i_addi(A0, A0), I_STOP
        w[self.SET_RET], w[self.SET_RET + 1], w[self.SET_RET + 2] = i_addi(A7, A7), i_addi(A0, A0), I_STOP
        w[self.PROBE], w[self.PROBE + 1], w[self.PROBE + 2] = i_lw(A7, A0, 0), i_sw(A7, A0, 0), I_STOP
        for k in range(SLED):
            w[self.RSLED + k] = i_lw(A7, A0, 4 * k - 2048)
            w[self.WSLED + k] = i_sh(0, A0, 4 * k - 2048)
        w[self.RSLED + SLED] = w[self.WSLED + SLED] = I_STOP
        return w

    def install(self, m) -> None:
        """Into a machine's ROM list (fetch space; loads never see it)."""
        assert len(m.rom) <= self.base + self.WORDS or not any(m.rom[self.base:self.base + self.WORDS])
        m.rom += [0] * max(0, self.base + self.WORDS - len(m.rom))
        m.rom[self.base:self.base + self.WORDS] = self.words()


def far(w: int) -> list[int]:
    return [S.tok("V", w & 0x7FFF), S.tok("JF", w >> 15)]


def near(w: int) -> list[int]:
    return [S.tok("J", w & 0xFFFF)]


def word_tokens(v: int, x: int) -> list[int]:
    """The SS record that writes the word v into a slot of value-head parity x."""
    return [S.tok("V", v & 0x7FFF), S.tok("V", v >> 15 & 0x7FFF), S.tok("V", v >> 30), S.tok(f"SS{x}")]


def values(toks) -> list[int]:
    """The value records in a chip stretch, in order (R / RF between them are skipped)."""
    out, rec = [], []
    for t in toks:
        fam = S.family(int(t))[0]
        if fam == "V":
            rec.append(int(t))
        elif fam in ("F0", "F1", "F2"):
            rec.append(int(t))
            out.append(S.decode_value(rec))
            rec = []
        elif fam in ("R", "RF", "PA"):
            continue
        else:
            raise ValueError(f"not a value record: {S.name(int(t))}")
    assert not rec, "a value record cut short"
    return out


def install_stops(ex) -> None:
    """HALT and BYTE end a launch: the kernel stops when the predicted token has no embedding
    row. They stay this launch's last prediction, unfed."""
    lut = ex.lut.view(-1, 4)
    for fam in STOP_FAMS:
        fid = core.FAM_ID[fam]
        lut[fid << 16:(fid + 1) << 16, 1] = -1


class ExecBackend:
    """Executor v4: forced rows, then the circuit's own tokens."""

    predicts = True

    def __init__(self, ex) -> None:
        self.ex = ex
        self.dev = ex.dev
        self.off, self.dims, self.vals = ex.tables[2], ex.tables[3], ex.tables[4]

    def rows(self, toks: list[int]):
        import torch

        out = np.zeros((len(toks), self.ex.D), np.float32)
        for i, t in enumerate(toks):
            a, b = self.off[t], self.off[t + 1]
            out[i, self.dims[a:b]] = self.vals[a:b]
        return torch.from_numpy(out).to(self.dev)

    def run(self, toks: list[int], max_steps: int) -> np.ndarray:
        return self.ex.run(self.rows(toks), mode=1, max_steps=max_steps, keep=False)


class RefBackend:
    """The reference machine in the chip's place, for tests of the host (no GPU). Never the
    production path: it is the thing the chip is checked against."""

    predicts = False

    def __init__(self, m) -> None:
        self.m, self.a, self.q, self.held = m, Audit(m), [], None

    def run(self, toks: list[int], max_steps: int) -> np.ndarray:
        n, m = len(toks), self.m
        if self.held is not None and toks == [self.held]:
            self.held = None
        else:
            self.a.forced_run(np.asarray(toks, np.int64))
        ids = [0] * (n - 1)
        while True:
            if not self.q:
                if m.n != S.N_MAX and self.a._ins().op == "BPRD":
                    ids.append(S.tok("BYTE", 48))
                    return np.asarray(ids, np.int64)
                self.q = m.record()
            t = self.q.pop(0)
            ids.append(t)
            if t == S.tok("HALT"):
                m.halted, m.fault = False, None
                return np.asarray(ids, np.int64)
            if len(ids) >= max_steps:
                self.held = t
                return np.asarray(ids, np.int64)


class ChipFault(RuntimeError):
    """The chip stopped somewhere a run may not: a fault other than the ecall, no token, a token
    of the wrong family."""


class LiveCPU:
    """The machine as the host sees it. Every transfer is tokens; nothing touches the state."""

    def __init__(self, backend, bios: Bios, cfg, full_regs: bool = False) -> None:
        self.b, self.bios, self.cfg, self.full = backend, bios, cfg, full_regs
        self.x: dict[int, int] = {}  # page -> the frame's value-head parity, as the chip printed it
        self.nf = 0  # frames the chip allocated (PA tokens)
        self.cache: dict[int, int] = {}  # word address -> value, while the chip is parked
        self.log: list[tuple[np.ndarray, bool]] = []  # (tokens fed, forced)
        self.seconds = {"fx": 0.0, "io": 0.0, "audit": 0.0}
        self.tokens = {"fx": 0, "io": 0, "audit": 0}
        self.launches = {"fx": 0, "io": 0, "audit": 0}
        self.instr = 0
        self.n_stops = 0
        self.ok_forced = 0
        self.chunk = CHUNK
        n = S.VOCAB2
        fin = np.zeros(n, bool)
        for fam, (b_, sz) in list(S.FAMILIES.items()) + list(S.NEW_FAMILIES.items()):
            if fam in S.FINAL and fam not in ("R", "RF"):
                fin[b_:b_ + sz] = True
        for nm in ("NT", "ST0", "ST1", "SS0", "SS1"):
            fin[S.CONTROLS[nm]] = True
        self.final_mask = fin
        self.halt, self.fam_byte = S.tok("HALT"), S.FAMILIES["BYTE"]

    # ---- launches ----------------------------------------------------------------------------
    def _note(self, toks, forced, kind) -> None:
        a = np.asarray(toks, np.int32)
        self.log.append((a, forced))
        self.tokens[kind] += len(a)

    def _is_stop(self, t: int) -> bool:
        return t == self.halt or self.fam_byte[0] <= t < self.fam_byte[0] + self.fam_byte[1]

    def run(self, forced: list[int], kind: str = "io", budget: int | None = None, fkind: str = "io") -> tuple:
        """Forced tokens, then the chip until a stop. (the chip's tokens, the stop token)."""
        chip, rows, is_forced = [], forced, True
        budget = budget or self.chunk
        while True:
            n = len(rows)
            t0 = time.perf_counter()
            ids = self.b.run(rows, n + budget)
            dt = time.perf_counter() - t0
            self.seconds[kind] += dt
            self.launches[kind] += 1
            k = len(ids)
            self._note(rows, is_forced, fkind if is_forced else kind)
            if not is_forced:  # the last launch's final prediction, fed as this one's first row
                chip.append(np.asarray(rows, np.int64))
            got = np.asarray(ids[n - 1:k - 1], np.int64)
            if len(got):
                self._note(got, False, kind)
                chip.append(got)
            last = int(ids[k - 1])
            if last < 0:
                raise ChipFault(f"no token at the end of a launch: {last} after {len(self.log)} stretches")
            if self._is_stop(last):
                return (np.concatenate(chip) if chip else np.zeros(0, np.int64)), last
            rows, is_forced = [last], False

    def forced_only(self, toks: list[int], check: dict | None = None, kind: str = "io") -> None:
        """Forced tokens only. check: {index: token}: the circuit's prediction before that token
        must be this one (a stretch the host writes still has positions the circuit knows)."""
        t0 = time.perf_counter()
        ids = self.b.run(toks, len(toks))
        self.seconds[kind] += time.perf_counter() - t0
        self.launches[kind] += 1
        self._note(toks, True, kind)
        for i, want in (check if self.b.predicts and check else {}).items():
            if int(ids[i - 1]) != want:
                raise ChipFault(f"forced stretch, position {i}: the circuit predicts "
                                f"{ids[i - 1]}, the host fed {S.name(want)}")
        self.ok_forced += len(check or ())

    # ---- the machine -------------------------------------------------------------------------
    def go(self) -> tuple:
        return self.run([S.tok("GO")], "fx", fkind="io")

    def parse_stop(self, chip: np.ndarray, stop: int) -> tuple:
        """Where a run of fx code stopped: ("ecall", pc) or ("exit", None)."""
        self.instr += int(self.final_mask[chip].sum())
        if stop != self.halt:
            raise ChipFault(f"fx stopped on {S.name(stop)}")
        if len(chip) < 9 or S.family(int(chip[-9]))[0] != "FLT":
            self.instr += 1  # (HALT, the exit ecall)
            return "exit", None
        cause = S.family(int(chip[-9]))[1]
        pc = 0
        for t in chip[-8:]:
            fam, p = S.family(int(t))
            assert fam == "HEX"
            pc = pc << 4 | p
        if S.FAULTS[cause] != "ecall":
            raise ChipFault(f"{S.FAULTS[cause]} at {pc:#x}")
        self.instr += 1  # the ecall
        self.n_stops += 1
        return "ecall", pc

    def enter(self, full: bool | None = None) -> dict:
        """From the ecall (or exit) the chip stopped on into the I/O ROM; the registers as tokens."""
        full = self.full if full is None else full
        w = self.bios.base + (self.bios.E_FULL if full else self.bios.E_SHORT)
        chip, stop = self.run(far(w), "audit" if full else "io", budget=256)
        self._byte(stop)
        vals = values(chip)
        order = EMIT_ORDER if full else EMIT_ORDER[:3]
        assert len(vals) == len(order), (len(vals), len(order))
        return dict(zip(order, vals))

    def _byte(self, stop: int) -> None:
        if stop == self.halt:
            raise ChipFault("a fault in the I/O ROM")

    def _a0(self, base: int) -> list[int]:
        return near(self.bios.base + self.bios.SET_A0) + S.value_record(base)

    def read_words(self, addr: int, n: int, kind: str = "io") -> list[int]:
        """n words from a word-aligned address, printed by the chip."""
        out: list[int] = []
        while n > 0:
            k = min(n, SLED)
            base = addr + 4 * k - 2048
            forced = self._a0(base) + near(self.bios.base + self.bios.RSLED + SLED - k)
            chip, stop = self.run(forced, kind, budget=3 * k + k // 32 + 4)
            self._byte(stop)
            v = values(chip)
            assert len(v) == k, (len(v), k)
            out += v
            addr += 4 * k
            n -= k
        return out

    def probe(self, addr: int) -> None:
        """Maps addr's page if the chip has not, and learns its frame's parity."""
        forced = self._a0(addr) + near(self.bios.base + self.bios.PROBE)
        chip, stop = self.run(forced, "io", budget=8)
        self._byte(stop)
        t = int(chip[-1])
        nm = S.family(t)[0]
        if nm not in ("ST0", "ST1"):
            raise ChipFault(f"probe of {addr:#x}: {S.name(t)}")
        for u in chip.tolist():
            if S.family(u)[0] == "PA":
                self.nf = S.family(u)[1]
        self.x[addr >> S.PAGE_BITS] = int(nm[-1])

    def write_words(self, addr: int, words: list[int]) -> None:
        """Whole words at a word-aligned address, as SS records: the capture path."""
        for i in range(len(words)):
            pg = (addr + 4 * i) >> S.PAGE_BITS
            if pg not in self.x:
                self.probe(addr + 4 * i)
        i = 0
        while i < len(words):
            k = min(len(words) - i, SLED)
            base = addr + 4 * (i + k) - 2048
            entry = self.bios.base + self.bios.WSLED + SLED - k
            toks = self._a0(base) + near(entry)
            check = {}
            w, c = entry, 0
            for j in range(k):
                if c == S.N_MAX:
                    w += c
                    c = 0
                    check[len(toks)] = S.tok("R", w & 0xFFFF)
                    toks += [S.tok("R", w & 0xFFFF)]
                x = self.x[(addr + 4 * (i + j)) >> S.PAGE_BITS]
                toks += word_tokens(words[i + j], x)
                check[len(toks) - 1] = S.tok(f"SS{x}")
                c += 1
            self.forced_only(toks, check)
            i += k

    def read(self, addr: int, n: int) -> bytes:
        """Guest bytes; word reads are kept while the chip is parked."""
        if n <= 0:
            return b""
        w0, w1 = addr & ~3, (addr + n + 3) & ~3
        ws = list(range(w0, w1, 4))
        miss = [a for a in ws if a not in self.cache]
        if miss:
            for a, v in zip(range(miss[0], miss[-1] + 4, 4), self.read_words(miss[0], (miss[-1] + 4 - miss[0]) // 4)):
                self.cache[a] = v
        b = b"".join(self.cache[a].to_bytes(4, "little") for a in ws)
        return b[addr - w0:addr - w0 + n]

    def write(self, addr: int, data: bytes) -> None:
        if not data:
            return
        w0, w1 = addr & ~3, (addr + len(data) + 3) & ~3
        cur = bytearray()
        if addr != w0 or addr + len(data) != w1:  # partial words: the rest comes from the chip
            cur += self.read(w0, 4)
            if w1 - w0 > 4:
                cur += bytes(w1 - w0 - 8) + self.read(w1 - 4, 4)
        else:
            cur += bytes(w1 - w0)
        cur[addr - w0:addr - w0 + len(data)] = data
        words = [int.from_bytes(cur[i:i + 4], "little") for i in range(0, len(cur), 4)]
        self.write_words(w0, words)
        for i, v in enumerate(words):
            self.cache[w0 + 4 * i] = v

    def resume(self, pc: int, a7: int, a0: int) -> tuple:
        """Back into fx at pc with a0 = the result (and a7 as it was): one launch."""
        self.cache.clear()
        forced = (near(self.bios.base + self.bios.SET_RET) + S.value_record(a7) + S.value_record(a0)
                  + far((pc - self.cfg.rom_base) >> 2))
        return self.run(forced, "fx")

    def fed(self) -> tuple:
        """The whole stream: (tokens, stretch lengths, forced flags)."""
        return (np.concatenate([a for a, _ in self.log]).astype(np.int32),
                np.array([len(a) for a, _ in self.log], np.int64),
                np.array([f for _, f in self.log], bool))


# ---- the boot snapshot and the offline audit ----------------------------------------------------

def machine_at(elf_path, ad, val, x, pc, frames, bios: Bios):
    """(cfg, a PagedMachine standing on the ecall at pc): rv32emu's memory at boot's end as the
    image, as fxrun.load, with the I/O ROM in the fetch space."""
    entry, segs = elf.segments(elf_path)
    rom_seg = [(va, b) for va, b in segs if va == 1 << fxrun.ROM_BIT]
    assert len(rom_seg) == 1
    code_end = (1 << fxrun.ROM_BIT) + len(rom_seg[0][1])
    data = [(va, b) for va, b in segs if va != 1 << fxrun.ROM_BIT]
    cfg = fxrun.fx_cfg(fxrun.windows(ad, data, code_end), frames)
    m = ref.PagedMachine.from_segments(cfg, entry, rom_seg, x=[int(v) for v in x], code_end=code_end)
    m.code_end = code_end
    for a, v in zip(np.asarray(ad).tolist(), np.asarray(val).tolist()):
        if not cfg.rom_base <= a << 2 < code_end:
            m.image[a] = v
    m.anchor, m.n = (pc - cfg.rom_base) >> 2, 0
    bios.install(m)
    return cfg, m


class Audit:
    """A run's stream on the reference machine. Chip tokens must equal the reference's; forced
    ones are applied as the module doc says."""

    def __init__(self, m) -> None:
        self.m, self.bad, self.chip, self.forced = m, None, 0, 0
        self.pending: list[int] = []  # the reference's next tokens not matched yet

    def _ins(self):
        m = self.m
        return ref.decode(m.rom[(m.pc - m.rom_base) >> 2])

    def chip_run(self, toks) -> None:
        m = self.m
        i, n = 0, len(toks)
        toks = toks.tolist()
        while i < n:
            if not self.pending:
                if m.n != S.N_MAX and self._ins().op == "BPRD":
                    self.bad = {"at": self.chip, "got": S.name(toks[i]), "want": "the stop"}
                    return
                m.halted = False
                self.pending = m.record()
            k = min(len(self.pending), n - i)
            if self.pending[:k] != toks[i:i + k]:
                j = next(j for j in range(k) if self.pending[j] != toks[i + j])
                self.bad = {"at": self.chip + j, "got": S.name(toks[i + j]), "want": S.name(self.pending[j]),
                            "pc": hex(m.pc)}
                return
            self.pending = self.pending[k:]
            i += k
            self.chip += k

    def stop_pending(self) -> None:
        """A forced stretch begins: what the chip did not finish is its stop. A fault record's
        HALT was predicted and never fed: the record did not commit."""
        m = self.m
        if self.pending:
            assert self.pending == [S.tok("HALT")], [S.name(t) for t in self.pending]
            self.pending = []
            m.halted, m.fault = False, None

    def forced_run(self, toks) -> None:
        m = self.m
        self.stop_pending()
        toks = toks.tolist()
        self.forced += len(toks)
        i = 0
        while i < len(toks):
            fam, p = S.family(toks[i])
            if fam == "GO":
                i += 1
            elif fam == "J":
                m.anchor, m.n = (m.anchor >> 16) << 16 | p, 0
                i += 1
            elif fam == "R":
                m.anchor, m.n = (m.anchor >> 16) << 16 | p, 0
                i += 1
            elif fam == "F0":
                self._set(S.decode_value([toks[i]]))
                i += 1
            elif fam == "V":
                j = i
                while S.family(toks[j])[0] == "V":
                    j += 1
                end = S.family(toks[j])[0]
                if end == "JF":
                    assert j == i + 1
                    m.anchor, m.n = p | S.family(toks[j])[1] << 15, 0
                elif end in ("F1", "F2"):
                    self._set(S.decode_value(toks[i:j + 1]))
                elif end in ("SS0", "SS1"):
                    assert j == i + 3
                    ins = self._ins()
                    assert ins.op == "SH", ins
                    ea = (m.x[ins.rs1] + ins.imm) & ref.M32
                    assert ea % 4 == 0 and m.cfg.rw_window(ea) is not None
                    pg = ea >> S.PAGE_BITS
                    assert m.pt[pg] & 1 == int(end[-1]), "SS parity"
                    v = S.family(toks[i])[1] | S.family(toks[i + 1])[1] << 15 | S.family(toks[i + 2])[1] << 30
                    m.wr[ea >> 2] = v
                    m.n += 1
                    m.steps += 1
                else:
                    raise AssertionError(end)
                i = j + 1
            else:
                raise AssertionError(fam)

    def _set(self, v: int) -> None:
        m, ins = self.m, self._ins()
        assert ins.op == "ADDI" and ins.rd and ins.rd == ins.rs1, ins
        m.x[ins.rd] = v
        m.n += 1
        m.steps += 1


def audit(m, tokens, lengths, forced) -> dict:
    """The run's stream against the machine m (its boot state). Stops at the first difference."""
    a = Audit(m)
    p = 0
    for n, f in zip(lengths.tolist(), forced.tolist()):
        part = tokens[p:p + n]
        p += n
        if f:
            a.forced_run(part)
        else:
            a.chip_run(part)
        if a.bad:
            break
    return {"chip_tokens": a.chip, "forced_tokens": a.forced, "wrong": a.bad, "instructions": m.steps,
            "pages": len(m.pt), "frames": m.nf}
