"""Tokens, records and the address map of the RV32 machine.

A trace is one record per instruction. Machine state changes only at a record's last token (its
*commit*); the PC is implied: PC = ROM_BASE + 4 * (anchor + n), where the anchor is the payload of
the latest J or R token and n counts the commits since then (an R refresh comes before the
instruction when n reaches 63).

Token families (chip vocabulary ids from 0):

  V     15-bit chunk of a value, not final
  F0    final; the value is this payload sign-extended from bit 14 (a 1-token value record)
  F1    final; after one V: value bits 15..29, sign-extended from bit 29
  F2    final; after two V: value bits 30..31
  J     final; jump: anchor = payload (a 16-bit ROM word index), n = 0
  R     final; anchor refresh: anchor = payload, n = 0, nothing executes
  BYTE  final; putchar (ecall a7 = 1): a0 & 511 (0..255 a byte; 256.. a special token, P3)
  FLT   not final; a fault: payload = cause (FAULTS), then 8 HEX digits of the PC, then HALT
  HEX   not final; a hex digit of a fault record
  QS    not final; DIV/DIVU/REM/REMU: k0, the index of the first quotient digit computed (0..8)
  QD    not final; a quotient digit: payload k << 4 | q (digit k of 8, most significant first)
  BD    not final; a digit-file row step (custom ops): payload digit | carry << 4
  NT, ST0, ST1, SS0, SS1, HALT   final controls (ST/SS: the RAM value head, address bit 6)
  GO    not final: the start of a run (after a state snapshot)
  SC    not final: BSCAN found one more ASCII digit
  CW    not final; a macro's 16-bit data token (half of a RAM word it streams)
  EB    not final; the compare macro's byte of operand a
  MP, CWW0, CWW1, EC0, EC1   not final controls of the macro records (fx circuits, below)

Records:

  rd-writing instruction (ALU, MUL, load, LUI, AUIPC, CSR read)  F0 | V F1 | V V F2 (by size)
  DIV / DIVU / REM / REMU with rd != x0                          QS QD.. then the value record
  BMAC / BMSB / BLD (n digits)                                    BD x n, then F0(carry) | NT
  BGET                                                            F0(digit) | NT
  BSCAN (n digit chars)                                           SC x n, then F0(n) | NT
  BPRD                                                            BYTE('0' + digit)
  the same with rd = x0, not-taken branch, FENCE                NT
  taken branch, JAL, JALR                                        J  (the link is in-circuit)
  SW                                                             ST0 / ST1
  SB, SH                                                         V V V SS0 / SS1 (merged word)
  ECALL a7 = 1 (putchar a0)                                      BYTE
  ECALL a7 = 93 (exit)                                           HALT
  a fault                                                        FLT(cause) HEX x 8 HALT
  before the instruction, when n == 63                           R

Macro records (fx/circuits.md: exact units specialized to fx's ROM). A macro sits on a `beq`
whose taken target is its function's `ret`; the ROM word is the macro uop MMOVE / MEQL with the
beq's registers, RD = a2, and the forward word offset in S0..S9 (ref.rom_word). When its
condition holds the record is the macro's; otherwise the beq runs (J / NT).

  MMOVE (memmove's entry, beq a0, a1): a0, a1, a2 4-aligned, 4 <= a2 <= 84, a0 != a1,
        a0 < a1 or a0 - a1 >= a2, both buffers below 0x7F80:
        MP, then for each word i: CW(src_i & 0xFFFF) CW(src_i >> 16) CWW0/1 (the destination
        word i is written, from the two payloads; 0/1 = address bit 6), then J(target)
  MEQL  (mem.eqlBytes' beq a1, a3): a1 != a3, 1 <= a2 <= 31, both below 0x7F80:
        MP, then for each byte k: EB(a[k]) EC1 (b[k] == a[k]) or EC0 (and the compare stops),
        then J(target), which also commits a0 = 1 if every byte matched, else 0

Memory (P2):

  RAM   0x0000_0000 + 32 KB   words; data / bss / heap from 0x100, the stack below 0x8000
                              (data pointers below 16 KB fit a 1-token value)
  DROM  0x0002_0000 + 128 KB  read-only data (.rodata), bytes; loads only
  ROM   0x0004_0000 + 256 KB  code only, pre-decoded (ref.rom_word), filled from a snapshot

The digit file (custom-0 / custom-1 ops, see ref.py): 8 registers of 128 decimal digits (least
significant first), outside the address space, saved in the snapshot like RAM.

The paged map (MemCfg, docs/fx/memory.md): a bigger ROM, a dense read-only image and demand-paged
RAM, for programs the size of fx. New tokens (ids after the legacy VOCAB, so the rv-q4p2
checkpoint's ids do not move):

  PA    not final; page allocation before a store to an unmapped page: payload = the new frame
  JF    final; far jump after one V: V = target bits 0..14, payload = target bits 15.. (the PC's
        high head is written; a plain J keeps it)
  RF    final; far refresh when the PC crosses a 64K-word window: payload = low 6 bits of the word
        | high bits << 6 (the word's bits 6..15 are 0 there)

  M     not final; the hot-code units' control family (chipformer/rv/hot.py, docs/fx/circuits2.md):
        payload tag (bits 12..13) | fields. One family, so the output code keeps its 5 family bits

  a store to an unmapped page    PA(f), then the store's record
  a jump leaving the window      V(lo) JF(hi) instead of J
  n == 63, crossing a window     RF instead of R

A DIV-class record: QS(k0), then QD(k << 4 | q_k) for k = k0..7, then the result's value record.
q_k are the hex digits of |A| / |B| (most significant first); k0 skips digits that are known to
be 0 from the operands' sizes (see ref.div_start), so small quotients take few tokens.
"""

from dataclasses import dataclass

FAMILY_SIZES = [("V", 1 << 15), ("F0", 1 << 15), ("F1", 1 << 15), ("F2", 4), ("J", 1 << 16),
                ("R", 1 << 16), ("BYTE", 512), ("FLT", 8), ("HEX", 16), ("QS", 16), ("QD", 128),
                ("BD", 256), ("CW", 1 << 16), ("EB", 256)]
FAMILIES = {}
_base = 0
for _n, _s in FAMILY_SIZES:
    FAMILIES[_n] = (_base, _s)
    _base += _s
CTRL = _base
CONTROL_NAMES = ["NT", "ST0", "ST1", "SS0", "SS1", "HALT", "GO", "SC", "MP", "CWW0", "CWW1", "EC0", "EC1"]
CONTROLS = {n: CTRL + i for i, n in enumerate(CONTROL_NAMES)}
VOCAB = CTRL + len(CONTROL_NAMES)
FINAL = {"F0", "F1", "F2", "J", "R", "BYTE", "NT", "ST0", "ST1", "SS0", "SS1", "HALT"}
PAYLOAD_BITS = {n: (s - 1).bit_length() for n, s in FAMILY_SIZES}

FAULTS = ["illegal", "ebreak", "misaligned load", "misaligned store", "load out of range",
          "store out of range", "bad jump", "ecall"]

ROM_BASE, ROM_WORDS = 0x40000, 1 << 16  # 256 KB of code (two sublayers)
DROM_BASE, DROM_WORDS = 0x20000, 1 << 15  # 128 KB of read-only data (one sublayer)
RAM_BASE, RAM_WORDS = 0x0, 1 << 13  # 32 KB (two sublayers)
DATA_BASE = 0x100  # .data starts here (address 0 stays a null pointer)
SP_INIT = RAM_BASE + 4 * RAM_WORDS
N_MAX = 63  # the R refresh comes before the instruction when n == N_MAX
MISA = 0x40001100  # RV32, I and M
# (P3) the text buffer, loads only: row i (1..CAP_MAX) holds the i-th token of the latest tool
# call's text, its UTF-8 bytes NUL-padded to TEXT_ROW, at TEXT_ADDR + TEXT_ROW * i
TEXT_ADDR, TEXT_ROW, TEXT_ROWS = 0x8000, 16, 128
# the capture counter's thermometer weights are 8 * 2^m: m < 88 keeps them below bf16's max
# after the emitter's 2^36 fold
CAP_MAX = 80
TEXT_TOKEN = 1 << 24  # (P3) a Qwen text token in the circuit's token stream: TEXT_TOKEN | qwen_id


# --- the paged map's tokens: after the legacy ids, which stay put ------------------------------
NEW_FAMILY_SIZES = [("PA", 1 << 12), ("JF", 1 << 6), ("RF", 1 << 11), ("M", 1 << 14)]
NEW_FAMILIES = {}
_base = VOCAB
for _n, _s in NEW_FAMILY_SIZES:
    NEW_FAMILIES[_n] = (_base, _s)
    _base += _s
VOCAB2 = _base
FINAL |= {"JF", "RF"}
PAGE_BITS = 9  # a RAM frame is one value head: 128 word slots, 512 bytes
WIN_BITS = 15  # an image window is one key head: 32 KB


@dataclass(frozen=True)
class MemCfg:
    """The paged machine's address map (see docs/fx/memory.md).

    rom_bit  ROM_BASE = 1 << rom_bit
    wbits    bits of the ROM word index (the PC)
    rom_kh   main ROM key heads, 2^13 words each (the ROM holds rom_kh << 13 words)
    rw       writable windows, (base, log2 size): each is direct-mapped by a page table
    frames   RAM frames (ids 1..frames-1, so this many value heads, half as many key heads)
    image    window numbers (address >> WIN_BITS) of the read-only image
    sp       the initial stack pointer
    units    the hot-code units built in (hot.Unit): fixed ROM PCs, see chipformer/rv/hot.py
    absent   uops no ROM word holds: core.build leaves their logic out (hot.rom_absent)"""

    name: str
    rom_bit: int
    wbits: int
    rom_kh: int
    rw: tuple
    frames: int
    image: tuple
    sp: int = 0
    units: tuple = ()
    absent: tuple = ()

    @property
    def rom_base(self) -> int:
        return 1 << self.rom_bit

    @property
    def rom_words(self) -> int:
        return self.rom_kh << 13

    def in_rom(self, t: int) -> bool:
        return t % 4 == 0 and self.rom_base <= t < self.rom_base + 4 * self.rom_words

    def rw_window(self, ea: int) -> int | None:
        for i, (b, lg) in enumerate(self.rw):
            if b <= ea < b + (1 << lg):
                return i
        return None

    def in_load(self, ea: int) -> bool:
        return self.rw_window(ea) is not None or ea >> WIN_BITS in self.image

    def load_regions(self) -> list:
        """Disjoint aligned regions (base, log2 size) that loads may touch."""
        out = list(self.rw)
        for w in self.image:
            a = w << WIN_BITS
            if self.rw_window(a) is None:
                out.append((a, WIN_BITS))
        return out


def small_cfg(frames: int = 128, rom_kh: int = 8) -> MemCfg:
    """The legacy addresses on the paged machinery: RAM at 0 (32 KB, by pages), DROM and .data in
    the image, ROM at 0x40000. The existing ELFs run on it unchanged."""
    return MemCfg("small", 18, 16, rom_kh, ((0, 15),), frames,
                  (0, 4, 5, 6, 7), RAM_BASE + 4 * RAM_WORDS)


def is_text(t: int) -> bool:
    return t >= TEXT_TOKEN


def family(t: int) -> tuple[str, int]:
    """(family or control name, payload) of a token."""
    if t >= VOCAB:
        for name, (base, size) in NEW_FAMILIES.items():
            if base <= t < base + size:
                return name, t - base
        raise ValueError(f"not an rv token: {t}")
    if t >= CTRL:
        return CONTROL_NAMES[t - CTRL], 0
    for name, (base, size) in FAMILIES.items():
        if base <= t < base + size:
            return name, t - base
    raise ValueError(f"not an rv token: {t}")


def tok(name: str, payload: int = 0) -> int:
    if name in CONTROLS:
        return CONTROLS[name]
    base, size = FAMILIES[name] if name in FAMILIES else NEW_FAMILIES[name]
    assert 0 <= payload < size, (name, payload)
    return base + payload


def is_final(t: int) -> bool:
    return family(t)[0] in FINAL


def value_record(v: int) -> list[int]:
    """Tokens of a 32-bit value: as few chunks as its sign-extended size needs."""
    v &= 0xFFFFFFFF
    s = v - (1 << 32) if v >> 31 else v
    c0, c1, c2 = v & 0x7FFF, v >> 15 & 0x7FFF, v >> 30
    if -(1 << 14) <= s < 1 << 14:
        return [tok("F0", c0)]
    if -(1 << 29) <= s < 1 << 29:
        return [tok("V", c0), tok("F1", c1)]
    return [tok("V", c0), tok("V", c1), tok("F2", c2)]


def decode_value(tokens: list[int]) -> int:
    """Inverse of value_record (the commit's view: the final token and the V before it)."""
    names = [family(t) for t in tokens]
    (fam, p) = names[-1]
    if fam == "F0":
        v = p | (0xFFFF8000 if p >> 14 else 0)
    elif fam == "F1":
        v = names[-2][1] | p << 15 | (0xC0000000 if p >> 14 else 0)
    elif fam == "F2":
        v = names[-3][1] | names[-2][1] << 15 | p << 30
    else:
        raise ValueError(fam)
    return v & 0xFFFFFFFF


def fault_record(cause: int, pc: int) -> list[int]:
    return ([tok("FLT", cause)] + [tok("HEX", pc >> 4 * (7 - i) & 15) for i in range(8)]
            + [tok("HALT")])


def sign32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


def name(t: int) -> str:
    fam, p = family(t)
    return fam if fam in CONTROLS else f"{fam}({p})"
