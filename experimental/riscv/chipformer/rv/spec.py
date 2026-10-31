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
  VC    not final; 15 bits of a vector op's configuration, at the start of its record (vx.py)
  VE    not final; 16 bits of a vector op's streamed results (vx.py)
  VH    not final; the same, the token that completes an element or a pair (vx.py)
  GO    not final: the start of a run (after a state snapshot)
  SC    not final: BSCAN found one more ASCII digit

Records:

  rd-writing instruction (ALU, MUL, load, LUI, AUIPC, CSR read)  F0 | V F1 | V V F2 (by size)
  DIV / DIVU / REM / REMU with rd != x0                          QS QD.. then the value record
  BMAC / BMSB / BLD (n digits)                                    BD x n, then F0(carry) | NT
  BGET                                                            F0(digit) | NT
  BSCAN (n digit chars)                                           SC x n, then F0(n) | NT
  BPRD                                                            BYTE('0' + digit)
  vector op (vx.py)                                              VC.. then VE / VH.., then NT
  the same with rd = x0, not-taken branch, FENCE                NT
  taken branch, JAL, JALR                                        J  (the link is in-circuit)
  SW                                                             ST0 / ST1
  SB, SH                                                         V V V SS0 / SS1 (merged word)
  ECALL a7 = 1 (putchar a0)                                      BYTE
  ECALL a7 = 93 (exit)                                           HALT
  a fault                                                        FLT(cause) HEX x 8 HALT
  before the instruction, when n == 63                           R

Memory (P2):

  RAM   0x0000_0000 + 32 KB   words; data / bss / heap from 0x100, the stack below 0x8000
                              (data pointers below 16 KB fit a 1-token value)
  DROM  0x0002_0000 + 128 KB  read-only data (.rodata), bytes; loads only
  ROM   0x0004_0000 + 256 KB  code only, pre-decoded (ref.rom_word), filled from a snapshot

  WROM  the vector unit's weights: row-chunks of 72 cells (vx.py), VMAT only (not in the address
        space; the ELF's .wrom section at WROM_BASE, one byte per cell in its low nibble)

The digit file (custom-0 / custom-1 ops, see ref.py): 8 registers of 128 decimal digits (least
significant first), outside the address space, saved in the snapshot like RAM.

A DIV-class record: QS(k0), then QD(k << 4 | q_k) for k = k0..7, then the result's value record.
q_k are the hex digits of |A| / |B| (most significant first); k0 skips digits that are known to
be 0 from the operands' sizes (see ref.div_start), so small quotients take few tokens.
"""

FAMILY_SIZES = [("V", 1 << 15), ("F0", 1 << 15), ("F1", 1 << 15), ("F2", 4), ("J", 1 << 16),
                ("R", 1 << 16), ("BYTE", 512), ("FLT", 8), ("HEX", 16), ("QS", 16), ("QD", 128),
                ("BD", 256), ("VC", 1 << 15), ("VE", 1 << 16), ("VH", 1 << 16)]
FAMILIES = {}
_base = 0
for _n, _s in FAMILY_SIZES:
    FAMILIES[_n] = (_base, _s)
    _base += _s
CTRL = _base
CONTROL_NAMES = ["NT", "ST0", "ST1", "SS0", "SS1", "HALT", "GO", "SC"]
CONTROLS = {n: CTRL + i for i, n in enumerate(CONTROL_NAMES)}
VOCAB = CTRL + len(CONTROL_NAMES)
FINAL = {"F0", "F1", "F2", "J", "R", "BYTE", "NT", "ST0", "ST1", "SS0", "SS1", "HALT"}
PAYLOAD_BITS = {n: (s - 1).bit_length() for n, s in FAMILY_SIZES}

FAULTS = ["illegal", "ebreak", "misaligned load", "misaligned store", "load out of range",
          "store out of range", "bad jump", "ecall"]

ROM_BASE, ROM_WORDS = 0x40000, 1 << 16  # 256 KB of code (two sublayers)
DROM_BASE, DROM_WORDS = 0x20000, 1 << 15  # 128 KB of read-only data (one sublayer)
RAM_BASE, RAM_WORDS = 0x0, 1 << 13  # 32 KB (two sublayers)
WROM_BASE, WROM_WORDS, WROM_CELLS = 0x100000, 15 * 2048 + 15 * 128, 8  # (the old VDOT bound)
WP_BITS = 15
VX_CHUNKS = 3 * 128 * 12  # the VX weight memory: row-chunks of 72 cells (vx.py), 3 a row pair, 12 key heads
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


def is_text(t: int) -> bool:
    return t >= TEXT_TOKEN


def family(t: int) -> tuple[str, int]:
    """(family or control name, payload) of a token."""
    if t >= CTRL:
        if t >= VOCAB:
            raise ValueError(f"not an rv token: {t}")
        return CONTROL_NAMES[t - CTRL], 0
    for name, (base, size) in FAMILIES.items():
        if base <= t < base + size:
            return name, t - base
    raise ValueError(f"not an rv token: {t}")


def tok(name: str, payload: int = 0) -> int:
    if name in CONTROLS:
        return CONTROLS[name]
    base, size = FAMILIES[name]
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
