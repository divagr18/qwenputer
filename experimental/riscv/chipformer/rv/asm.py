"""A small two-pass RV32IM assembler for tests (gcc in WSL is the real toolchain).

    label:                      define a label
    .text / .data / .rodata     switch section (text at ROM_BASE, data at DATA_BASE, rodata at
    .wrom                       DROM_BASE, the weight memory's cells (one per byte) at WROM_BASE)
    .word 1, 2, label           32-bit words      .byte / .half    likewise
    .space N                    N zero bytes      .align N         to 2^N bytes
    ; or #                      comment

Instructions: RV32IM and Zba (sh1add / sh2add / sh3add) with ABI register names, plus the
pseudo-instructions
li, la, mv, not, neg, seqz, snez, nop, j, jr, ret, call, tail, beqz, bnez, blez, bgez, bltz,
bgtz, bgt, ble, bgtu, bleu.
"""
# ruff: noqa: B023  (the per-pass helpers close over the pass's own text/data/sec)

import re

from chipformer.rv.spec import DATA_BASE, DROM_BASE, ROM_BASE, WROM_BASE

ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"] + [f"a{i}" for i in range(8)] \
    + [f"s{i}" for i in range(2, 12)] + ["t3", "t4", "t5", "t6"]
REG = {n: i for i, n in enumerate(ABI)} | {f"x{i}": i for i in range(32)} | {"fp": 8}

R_OPS = {  # funct7, funct3 (opcode 0x33)
    "add": (0, 0), "sub": (0x20, 0), "sll": (0, 1), "slt": (0, 2), "sltu": (0, 3), "xor": (0, 4),
    "srl": (0, 5), "sra": (0x20, 5), "or": (0, 6), "and": (0, 7),
    "mul": (1, 0), "mulh": (1, 1), "mulhsu": (1, 2), "mulhu": (1, 3), "div": (1, 4),
    "divu": (1, 5), "rem": (1, 6), "remu": (1, 7),
    "sh1add": (0x10, 2), "sh2add": (0x10, 4), "sh3add": (0x10, 6)}
I_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
SH_OPS = {"slli": (0, 1), "srli": (0, 5), "srai": (0x20, 5)}
LOADS = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORES = {"sb": 0, "sh": 1, "sw": 2}
BRANCHES = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
CSRS = {"cycle": 0xC00, "time": 0xC01, "instret": 0xC02, "cycleh": 0xC80, "instreth": 0xC82,
        "mhartid": 0xF14, "misa": 0x301, "mscratch": 0x340}


def enc_r(f7, rs2, rs1, f3, rd, op):
    return f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op


def enc_i(imm, rs1, f3, rd, op):
    assert -2048 <= imm < 2048, imm
    return (imm & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op


def enc_s(imm, rs2, rs1, f3, op=0x23):
    assert -2048 <= imm < 2048, imm
    imm &= 0xFFF
    return (imm >> 5) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 31) << 7 | op


def enc_b(off, rs2, rs1, f3):
    assert -4096 <= off < 4096 and off % 2 == 0, off
    o = off & 0x1FFF
    return ((o >> 12 & 1) << 31 | (o >> 5 & 63) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12
            | (o >> 1 & 15) << 8 | (o >> 11 & 1) << 7 | 0x63)


def enc_u(imm20, rd, op):
    return (imm20 & 0xFFFFF) << 12 | rd << 7 | op


def enc_j(off, rd):
    assert -(1 << 20) <= off < 1 << 20 and off % 2 == 0, off
    o = off & 0x1FFFFF
    return ((o >> 20 & 1) << 31 | (o >> 1 & 0x3FF) << 21 | (o >> 11 & 1) << 20
            | (o >> 12 & 0xFF) << 12 | rd << 7 | 0x6F)


def hi_lo(v):
    """lui/addi split of a 32-bit constant."""
    v &= 0xFFFFFFFF
    lo = v & 0xFFF
    lo = lo - 4096 if lo >= 2048 else lo
    return ((v - lo) >> 12) & 0xFFFFF, lo


class Program:
    def __init__(self, text: list[int], data: bytes, labels: dict[str, int],
                 data_base: int = DATA_BASE, rodata: bytes = b"", wrom: bytes = b""):
        self.text, self.data, self.labels, self.data_base = text, data, labels, data_base
        self.rodata = rodata  # read-only data from DROM_BASE
        self.wrom = wrom  # the weight memory's cells (VDOT), one per byte

    @property
    def entry(self) -> int:
        return self.labels.get("_start", ROM_BASE)


VX_ASM = {"vmat": 0, "vmatr": 1, "vq": 2, "vew": 3, "vexp": 4, "vget": 5, "vred": 6, "vput": 7}


def assemble(src: str) -> Program:
    lines = []
    for raw in src.splitlines():
        line = re.split(r"[;#]", raw)[0].strip()
        while line:
            m = re.match(r"([A-Za-z_.$][\w.$]*):\s*(.*)", line)
            if not m:
                break
            lines.append(("label", m[1]))
            line = m[2].strip()
        if line:
            op, _, rest = line.partition(" ")
            args = [a.strip() for a in rest.split(",")] if rest.strip() else []
            lines.append((op.lower(), args))
    labels: dict[str, int] = {}
    for pass_ in range(2):
        text, data, rodata, wrom = [], bytearray(), bytearray(), bytearray()
        sec = "text"

        def here():
            if sec == "text":
                return ROM_BASE + 4 * len(text)
            if sec == "wrom":
                return WROM_BASE + len(wrom)
            return DATA_BASE + len(data) if sec == "data" else DROM_BASE + len(rodata)

        def buf():
            return {"data": data, "rodata": rodata, "wrom": wrom}[sec]

        def val(s):
            s = s.strip()
            m = re.fullmatch(r"%(hi|lo)\((.+)\)", s)
            if m:
                h, lo = hi_lo(val(m[2]))
                return h if m[1] == "hi" else lo
            if s in labels:
                return labels[s]
            if re.fullmatch(r"[A-Za-z_.$][\w.$]*", s):
                if pass_:
                    raise ValueError(f"unknown label {s}")
                return here()  # (a forward label in pass 0: any in-range placeholder)
            return int(s, 0)

        def reg(s):
            return REG[s.strip()]

        def mem(s):  # "off(reg)"
            m = re.fullmatch(r"(.*)\((\w+)\)", s.strip())
            return (val(m[1]) if m[1].strip() else 0), reg(m[2])

        def emit(*words):
            assert sec == "text", "instructions belong in .text"
            text.extend(w & 0xFFFFFFFF for w in words)

        for op, args in lines:
            if op == "label":
                labels[args] = here()
                continue
            if op in (".text", ".data", ".rodata", ".wrom"):
                sec = op[1:]
                continue
            if op in (".word", ".half", ".byte"):
                n = {".word": 4, ".half": 2, ".byte": 1}[op]
                for a in args:
                    v = val(a) & ((1 << 8 * n) - 1)
                    if sec != "text":
                        buf().extend(v.to_bytes(n, "little"))
                    else:
                        assert n == 4
                        text.append(v)
                continue
            if op == ".space":
                buf().extend(bytes(val(args[0])))
                continue
            if op == ".align":
                a = 1 << val(args[0])
                if sec != "text":
                    buf().extend(bytes(-len(buf()) % a))
                continue
            if op == ".globl":
                continue
            pc = here()
            a = args
            if op in R_OPS:
                f7, f3 = R_OPS[op]
                emit(enc_r(f7, reg(a[2]), reg(a[1]), f3, reg(a[0]), 0x33))
            elif op in ("bmac", "bmsb", "bld", "bget"):  # rd, rs1, rs2 (operands in rs2)
                f3 = {"bmac": 0, "bmsb": 1, "bld": 2, "bget": 3}[op]
                emit(enc_r(0, reg(a[2]), reg(a[1]), f3, reg(a[0]), 0x0B))
            elif op == "bscan":  # rd, rs1
                emit(enc_r(0, 0, reg(a[1]), 4, reg(a[0]), 0x0B))
            elif op in VX_ASM:  # the vector unit (vx.py): rd, rs1, rs2
                emit(enc_r(0, reg(a[2]), reg(a[1]), VX_ASM[op], reg(a[0]), 0x5B))
            elif op == "bprd":  # dD, k
                emit(enc_i(val(a[1]), 0, 0, val(a[0]), 0x2B))
            elif op in I_OPS:
                emit(enc_i(val(a[2]), reg(a[1]), I_OPS[op], reg(a[0]), 0x13))
            elif op in SH_OPS:
                f7, f3 = SH_OPS[op]
                emit(enc_r(f7, val(a[2]) & 31, reg(a[1]), f3, reg(a[0]), 0x13))
            elif op in LOADS:
                off, rs1 = mem(a[1])
                emit(enc_i(off, rs1, LOADS[op], reg(a[0]), 0x03))
            elif op in STORES:
                off, rs1 = mem(a[1])
                emit(enc_s(off, reg(a[0]), rs1, STORES[op]))
            elif op in BRANCHES:
                emit(enc_b(val(a[2]) - pc, reg(a[1]), reg(a[0]), BRANCHES[op]))
            elif op in ("bgt", "ble", "bgtu", "bleu"):
                real = {"bgt": "blt", "ble": "bge", "bgtu": "bltu", "bleu": "bgeu"}[op]
                emit(enc_b(val(a[2]) - pc, reg(a[0]), reg(a[1]), BRANCHES[real]))
            elif op in ("beqz", "bnez", "blez", "bgez", "bltz", "bgtz"):
                r, t = reg(a[0]), val(a[1]) - pc
                emit({"beqz": enc_b(t, 0, r, 0), "bnez": enc_b(t, 0, r, 1),
                      "blez": enc_b(t, r, 0, 5), "bgez": enc_b(t, 0, r, 5),
                      "bltz": enc_b(t, 0, r, 4), "bgtz": enc_b(t, r, 0, 4)}[op])
            elif op == "lui":
                emit(enc_u(val(a[1]), reg(a[0]), 0x37))
            elif op == "auipc":
                emit(enc_u(val(a[1]), reg(a[0]), 0x17))
            elif op == "jal":
                if len(a) == 1:
                    emit(enc_j(val(a[0]) - pc, 1))
                else:
                    emit(enc_j(val(a[1]) - pc, reg(a[0])))
            elif op == "jalr":
                if len(a) == 1:
                    emit(enc_i(0, reg(a[0]), 0, 1, 0x67))
                elif "(" in a[1]:
                    off, rs1 = mem(a[1])
                    emit(enc_i(off, rs1, 0, reg(a[0]), 0x67))
                else:
                    emit(enc_i(val(a[2]) if len(a) > 2 else 0, reg(a[1]), 0, reg(a[0]), 0x67))
            elif op == "j":
                emit(enc_j(val(a[0]) - pc, 0))
            elif op == "jr":
                emit(enc_i(0, reg(a[0]), 0, 0, 0x67))
            elif op == "ret":
                emit(enc_i(0, 1, 0, 0, 0x67))
            elif op in ("call", "tail"):  # auipc + jalr (always two words)
                off = val(a[0]) - pc
                hi, lo = hi_lo(off)
                link = 1 if op == "call" else 0
                tmp = 1 if op == "call" else 6
                emit(enc_u(hi, tmp, 0x17), enc_i(lo, tmp, 0, link, 0x67))
            elif op == "li":
                v = val(a[1])
                if -2048 <= v < 2048:
                    emit(enc_i(v, 0, 0, reg(a[0]), 0x13))
                else:
                    hi, lo = hi_lo(v)
                    emit(enc_u(hi, reg(a[0]), 0x37), enc_i(lo, reg(a[0]), 0, reg(a[0]), 0x13))
            elif op == "la":  # lui + addi (absolute: the code is not relocated)
                hi, lo = hi_lo(val(a[1]))
                emit(enc_u(hi, reg(a[0]), 0x37), enc_i(lo, reg(a[0]), 0, reg(a[0]), 0x13))
            elif op == "mv":
                emit(enc_i(0, reg(a[1]), 0, reg(a[0]), 0x13))
            elif op == "not":
                emit(enc_i(-1, reg(a[1]), 4, reg(a[0]), 0x13))
            elif op == "neg":
                emit(enc_r(0x20, reg(a[1]), 0, 0, reg(a[0]), 0x33))
            elif op == "seqz":
                emit(enc_i(1, reg(a[1]), 3, reg(a[0]), 0x13))
            elif op == "snez":
                emit(enc_r(0, reg(a[1]), 0, 3, reg(a[0]), 0x33))
            elif op == "nop":
                emit(enc_i(0, 0, 0, 0, 0x13))
            elif op == "ecall":
                emit(0x00000073)
            elif op == "ebreak":
                emit(0x00100073)
            elif op == "fence":
                emit(0x0FF0000F)
            elif op == "csrr":  # csrrs rd, csr, x0
                emit(CSRS.get(a[1], 0) << 20 | 2 << 12 | reg(a[0]) << 7 | 0x73
                     if a[1] in CSRS else val(a[1]) << 20 | 2 << 12 | reg(a[0]) << 7 | 0x73)
            elif op == "csrw":  # csrrw x0, csr, rs
                csr = CSRS[a[0]] if a[0] in CSRS else val(a[0])
                emit(csr << 20 | reg(a[1]) << 15 | 1 << 12 | 0x73)
            else:
                raise ValueError(f"unknown instruction {op} {args}")
    return Program(text, bytes(data), labels, rodata=bytes(rodata), wrom=bytes(wrom))
