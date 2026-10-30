"""VX: a streaming vector unit for the RV core (reference semantics and trace records; docs/vx.md).

Custom-2 (opcode 0x5B, R-type, funct7 = 0). The unit sits in the circuit's first layers with its
own memories, so it never sees the registers or RAM: an op's record starts with VC tokens that
carry its configuration, then streams its results, then commits with NT. A VC payload is a slot
(3 bits) and 12 bits of data; the unit keeps the latest of each slot. Slots 0-3 hold the fields
that address its memories, the same for every op (SLOTS); 4-7 the op's own parameters. Stream
tokens are VE (16 bits) and VH (16 bits, the token that completes an element or a pair). A word
result is VE(bits 0-15) VH(bits 16-30 | best << 15): vector words are 31-bit, and best marks the
tracker's new max. VGET answers with a value record; VPUT streams its pair of words after one VC.

Memories (all the unit's own):
  VM    16 vector registers of 64 int32 (bank A v0..v7, bank B v8..v15; an op reads at most one
        register per bank), stored as element pairs: results are written a pair at a time
  VA    384 int8 activations: 6 chunks of 64, read a chunk at a time
  KC    the K cache: 1024 rows of 8 int8          VCC  the V cache, likewise
  WROM  row-chunks of 72 cells (`.wrom`): 64 int4 weights (cell u means u - 8), then the row's
        scale S < 2^16 in 4 cells (nibbles, least significant first), then 4 unused
  RoPE  4 (cos, sin) int16 pairs                  trackers: vmx, vsum, the argmax (vbest, varg)

Sized for the circuit: multiplies of a 16-bit window of a value, clamp16(v >> sh), by a 15- or
16-bit factor; dot products of at most 64 int8 x int4 or 32 int8 x int8 lanes a token; SiLU and
exp as fixed-point function units. `check` raises on operands outside those widths.

shift(x, s) is x << s for s >= 0, x >> -s below; clamp16 clamps to [-32768, 32767]. The fields
below are the instruction's (register bits, in order); SLOTS gives their VC packing.

  f3 0  VMAT   rd: dst 4, n 7, k 2, mode 1, keep 1, chunk 2; rs1: base 13, doff 6
               rows r < n: acc = sum over c < k of VA[64 (chunk + c):][:64] . row-chunk
               base + r k + c; out = acc * S (the last chunk's) mod 2^32. Mode 0: VM[dst][doff + r]
               = out (n, doff even), max |out| tracked; mode 1: argmax of out, the best row's
               (last) row-chunk index. A row's tokens: the running sum after each chunk (VE VE,
               32 bits), then out (a word): 2 k + 2.
  f3 1  VMATR  rd: dst 4, n 6, mode 1, keep 1, off 8; rs1: base 10, eoff 6; rs2 (mode 1): s 17
               (unsigned)
               mode 0 (scores): out[t] = VA chunk off [:8] . KC[base + t], t < n, signed max tracked
               mode 1 (p.V): out[j] = (VA chunk off [:n] . VCC[base + t][j] over t) * s >> 12, j < 8,
               max |out| tracked. out goes to VM[dst][eoff + i], 2 tokens each; an odd n's last
               pair gets a 0 (not tracked).
  f3 2  VQ     rd: src 4, eoff 6, n 7, dst 2, sh 5; rs1: off 10; rs2: r 15
               q_i = clamp(round(clamp16(VM[src][eoff + i] >> sh) * r / 2^15), +-127), i < n
               (n even); dst 0: VA[off + i]; 1 / 2: KC / VCC row off + 32 (i / 8), byte i mod 8.
               A token per element.
  f3 3  VEW    rd: a 4, b 4, n 7, op 2, keep 1, eoff 6; rs1: sh 5, so 6 (signed), shb 5;
               rs2: p 15
               i < n (n even) of VM[a][eoff + i], in place, max |result| tracked:
               op 0 ADD  a += shift(clamp16(b >> sh) * p, so)                       2 tokens each
               op 1 GLU  y = shift(clamp16(a >> sh) * p, so), a = SILU(y) * clamp16(b >> shb) >> 8
                                                                                    4 tokens each
               (b = VM[b][eoff + i])
               op 2 ROPE pairs (a, a'): u = clamp16(a >> sh), w = clamp16(a' >> sh), by RoPE pair
                         (i mod 8) / 2: (shift(u c - w s, sh - 14), shift(u s + w c, sh - 14))
                                                                                    2 tokens each
  f3 4  VEXP   rd: src 4, n 7, off 9, ksh 5; rs1: m 31 (the scores' max: VGET 0 after them);
               rs2: c 16
               i < n: e = min(m - VM[src][i], 65535), d = e * c >> ksh; VA[off + i] = EXP(d)
               (an odd n's last pair gets a 0). A token per element.
  f3 5  VGET   rs1 = sel: rd = vmx / vsum / varg / bit length of vmx: a value record
  f3 6  VRED   rd: n 7, mode 1, off 8: vsum = sum q^2 (mode 0) or q over q in VA chunk off [:n]:
               a word's two tokens.
  f3 7  VPUT   rd: target 10; rs1, rs2: the data, streamed (VE VH VE VH)
               target < 512: VM pair target (register target / 32) = (rs1, rs2);
               512 + k: RoPE pair k = (cos rs1, sin rs2)
SILU(y) (y in Q16): the Q12 table round(x / (1 + e^-x) 4096) at x = (y >> 11) / 32 for |y| < 8.0,
min(y >> 4, 2^23) from 8.0, 0 below -8.0. EXP(d) = round(127 e^(-d / 32)) for d < 256, else 0.
Streams are at most 64 tokens (software splits longer vectors into ops). Every op counts elements
(a VH completes one), and an element offset is a multiple of the op's length rounded up to a power
of two, so the unit addresses its registers by concatenating bits.
"""

import math

from chipformer.rv import spec as S

OPS = ["VMAT", "VMATR", "VQ", "VEW", "VEXP", "VGET", "VRED", "VPUT"]
MAX_STREAM = 64
M32 = 0xFFFFFFFF
CHUNK_CELLS = 72
SILU_T = [round((x := (i - 256) / 32) / (1 + math.exp(-x)) * 4096) for i in range(512)]
EXP_T = [round(127 * math.exp(-i / 32)) for i in range(256)]
FIELDS = {  # op -> [(name, register, lo bit, width)]: the instruction's operands
    "VMAT": [("dst", "rd", 0, 4), ("n", "rd", 4, 7), ("k", "rd", 11, 2), ("mode", "rd", 13, 1),
             ("keep", "rd", 14, 1), ("chunk", "rd", 15, 2), ("base", "rs1", 0, 13),
             ("doff", "rs1", 13, 6)],
    "VMATR": [("dst", "rd", 0, 4), ("n", "rd", 4, 6), ("mode", "rd", 10, 1), ("keep", "rd", 11, 1),
              ("off", "rd", 12, 8), ("base", "rs1", 0, 10), ("eoff", "rs1", 10, 6),
              ("s", "rs2", 0, 17)],
    "VQ": [("src", "rd", 0, 4), ("eoff", "rd", 4, 6), ("n", "rd", 10, 7), ("dst", "rd", 17, 2),
           ("sh", "rd", 19, 5), ("off", "rs1", 0, 10), ("r", "rs2", 0, 15)],
    "VEW": [("a", "rd", 0, 4), ("b", "rd", 4, 4), ("n", "rd", 8, 7), ("op", "rd", 15, 2),
            ("keep", "rd", 17, 1), ("eoff", "rd", 18, 6), ("sh", "rs1", 0, 5), ("so", "rs1", 5, 6),
            ("shb", "rs1", 11, 5), ("p", "rs2", 0, 15)],
    "VEXP": [("src", "rd", 0, 4), ("n", "rd", 4, 7), ("off", "rd", 11, 9), ("ksh", "rd", 20, 5),
             ("m", "rs1", 0, 31), ("c", "rs2", 0, 16)],
    "VRED": [("n", "rd", 0, 7), ("mode", "rd", 7, 1), ("off", "rd", 8, 8)],
    "VPUT": [("target", "rd", 0, 10), ("lo", "rs1", 0, 32), ("hi", "rs2", 0, 32)],
}


def reg3(f, name, bank):
    """A register field as a bank's 3-bit index (0 when it is in the other bank)."""
    return f[name] & 7 if f[name] >> 3 == bank else 0


# each op's VC tokens, in order: (slot, [(value, width)] packed from bit 0). Every op sends slot 0
# = (dst register 4, element offset / 2 5, write target 3: 0 none, 1 vsum, 2 VA, 4 VM (max |v|
# tracked), 5 VM (max v tracked), 6 KC, 7 VCC; the VM codes share bits 1-2, the caches bits 1-2);
# 1 = (bank A register 3, bank B register 3, more 6), 2 = (VA byte offset 9, more 3), 3 = (cache
# row 10, more 2) are the other memory addresses; 4-7 are the op's own
W_SUM, W_VA, W_VM, W_VMS, W_KC, W_VC = 1, 2, 4, 5, 6, 7
SLOTS = {
    "VMAT": lambda f: [(0, [(f["dst"], 4), (f["doff"] >> 1, 5), ((f["mode"] == 0) * W_VM, 3)]),
                       (2, [(64 * f["chunk"], 9)]),
                       (4, [(f["base"], 12)]),
                       (5, [(f["base"] >> 12, 1), (f["n"], 7), (f["mode"], 1), (f["keep"], 1),
                            (f["k"] == 3, 1)])],
    "VMATR": lambda f: [(0, [(f["dst"], 4), (f["eoff"] >> 1, 5), (W_VM + (f["mode"] == 0), 3)]),
                        (2, [(64 * f["off"], 9)]),
                        (3, [(f["base"], 10), (f["mode"], 1), (f["keep"], 1)]),
                        (4, [(f["n"], 6), (f["s"], 6)]),
                        (5, [(f["s"] >> 6, 11)])],
    "VQ": lambda f: [(0, [(0, 4), (f["eoff"] >> 1, 5), ((W_VA, W_KC, W_VC)[f["dst"]], 3)]),
                     (1, [(reg3(f, "src", 0), 3), (reg3(f, "src", 1), 3)]),
                     (2, [(f["off"], 9), (f["n"] >> 5, 2), (f["src"] >> 3, 1)]),
                     (3, [(f["off"], 10)]),
                     (4, [(f["r"], 12)]),
                     (5, [(f["r"] >> 12, 3), (f["sh"], 5), (f["n"] >> 1, 4)])],
    "VEW": lambda f: [(0, [(f["a"], 4), (f["eoff"] >> 1, 5), (W_VM, 3)]),
                      (1, [(reg3(f, "a", 0) + reg3(f, "b", 0), 3),
                           (reg3(f, "a", 1) + reg3(f, "b", 1), 3), (f["shb"], 5)]),
                      (4, [(f["n"] >> 1, 6), (f["sh"], 5), (f["keep"], 1)]),
                      (5, [(f["so"], 6), (f["p"], 6)]),
                      (6, [(f["p"] >> 6, 9), (f["op"], 2)])],
    "VEXP": lambda f: [(0, [(0, 4), (0, 5), (W_VA, 3)]),
                       (1, [(reg3(f, "src", 0), 3), (reg3(f, "src", 1), 3), (f["src"] >> 3, 1)]),
                       (2, [(f["off"], 9), (f["n"], 3)]),
                       (4, [(f["n"] >> 3, 4), (f["ksh"], 5), (f["c"], 3)]),
                       (5, [(f["c"] >> 3, 12)]),
                       (6, [(f["c"] >> 15, 1), (f["m"], 11)]),
                       (7, [(f["m"] >> 11, 12)]),
                       (3, [(0, 2), (f["m"] >> 23, 8)])],
    "VRED": lambda f: [(0, [(0, 4), (0, 5), (W_SUM, 3)]),
                       (2, [(64 * f["off"], 9), (f["mode"], 1), (f["n"], 2)]),
                       (4, [(f["n"] >> 2, 5)])],
    "VPUT": lambda f: [(0, [(f["target"] >> 5 & 15, 4), (f["target"] & 31, 5),
                            ((f["target"] < 512) * W_VM, 3)])],
}


def s32(v):
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def sx(v, n):
    v &= (1 << n) - 1
    return v - (1 << n) if v >> (n - 1) else v


def clamp16(v):
    return max(-32768, min(32767, v))


def shift(x, s):
    return x << s if s >= 0 else x >> -s


def check(ok, what):
    if not ok:
        raise ValueError(f"VX operand out of range: {what}")


def silu(y):
    if y >= 8 << 16:
        return min(y >> 4, 1 << 23)
    if y < -(8 << 16):
        return 0
    return SILU_T[(y >> 11) + 256]


def ve(v):
    return S.tok("VE", v & 0xFFFF)


def vh(v, best=False):
    return S.tok("VH", v & 0xFFFF | best << 15)


def word(v, best=False):
    """A 31-bit word's two tokens."""
    return [ve(v), vh(v >> 16 & 0x7FFF, best)]


def config(op, regs):
    """The op's fields (by name) and its VC tokens."""
    f = {name: regs[r] >> lo & ((1 << w) - 1) for name, r, lo, w in FIELDS[op]}
    toks = []
    for slot, parts in SLOTS[op](f):
        bits, nb = 0, 0
        for val, w in parts:
            bits |= (int(val) & ((1 << w) - 1)) << nb
            nb += w
        assert nb <= 12, (op, slot)
        toks.append(S.tok("VC", slot << 12 | bits))
    return f, toks


class VX:
    def __init__(self):
        self.vm = [[0] * 64 for _ in range(16)]
        self.va = [0] * 384
        self.kc = [[0] * 8 for _ in range(1024)]
        self.vc = [[0] * 8 for _ in range(1024)]
        self.rope = [(0, 0)] * 4
        self.vmx = self.vsum = self.varg = 0
        self.vbest = None  # the argmax value so far


def wrom_chunk(m, c):
    """(64 weights, the scale) of WROM row-chunk c (cells past the data read as 0: w = -8)."""
    o = CHUNK_CELLS * c
    cell = [m.wrom[o + i] & 15 if o + i < len(m.wrom) else 0 for i in range(68)]
    return [u - 8 for u in cell[:64]], sum(cell[64 + i] << 4 * i for i in range(4))


def execute(m, ins) -> list[int]:
    """The op's record (commits it)."""
    v, x, op = m.vx, m.x, ins.op
    m.n += 1
    regs = {"rd": x[ins.rd], "rs1": x[ins.rs1], "rs2": x[ins.rs2]}
    if op == "VGET":
        r = [v.vmx, v.vsum, v.varg, max(v.vmx, 0).bit_length()][regs["rs1"] & 3]
        if ins.rd:
            x[ins.rd] = r & M32
            return S.value_record(r)
        return [S.tok("NT")]
    f, toks = config(op, regs)
    n_vc = len(toks)

    def tracked(val, signed=False, first=False):
        """A word result's two tokens, marked when it is the tracker's new max (always at the
        first element of an op that does not keep the tracker)."""
        t = val if signed else abs(val)
        best = first or t > v.vmx
        if best:
            v.vmx = t
        return word(val, best)

    def put(reg, i, val):
        check(0 <= i < 64 and -(1 << 30) <= val < 1 << 30, f"VM element {i} = {val}")
        v.vm[reg][i] = val

    def aligned(off, n):
        check(off % (1 << max(n - 1, 0).bit_length()) == 0, f"offset {off} for {n} elements")

    if op == "VMAT":
        dst, n, k, mode, keep, chunk, base, doff = (
            f[x] for x in ("dst", "n", "k", "mode", "keep", "chunk", "base", "doff"))
        check(k in (1, 3) and (mode or n % 2 == doff % 2 == 0) and 0 < n <= 64,
              f"VMAT n {n} k {k} doff {doff}")
        if mode == 0:
            aligned(doff, n)
        for r in range(n):
            acc, sc, part = 0, 0, []
            for c in range(k):
                w, sc = wrom_chunk(m, base + r * k + c)
                acc += sum(v.va[64 * (chunk + c) + i] * w[i] for i in range(64))
                part += [ve(acc), ve(acc >> 16)]  # the running sum, then the row's scaling
            o = s32(acc * sc)
            if mode == 0:
                o = sx(o, 31)
                put(dst, doff + r, o)
                toks += part + tracked(o, first=r == 0 and not keep)
            else:
                best = r == 0 and not keep or o > v.vbest
                if best:
                    v.vbest, v.varg = o, base + r * k + k - 1
                check(-(1 << 30) <= o < 1 << 30, f"VMAT argmax out {o}")
                toks += part + word(o, best)
    elif op == "VMATR":
        dst, n, mode, keep, off, base, eoff, s = (
            f[x] for x in ("dst", "n", "mode", "keep", "off", "base", "eoff", "s"))
        check(0 < n <= 32, f"VMATR n {n}")
        check(off < 6, f"VA chunk {off}")
        va = v.va[64 * off:64 * off + 64]
        if mode == 0:
            outs = [sum(va[i] * v.kc[base + t][i] for i in range(8)) for t in range(n)]
        else:
            outs = [(sum(va[t] * v.vc[base + t][j] for t in range(n)) * s) >> 12 for j in range(8)]
        aligned(eoff, len(outs) + len(outs) % 2)
        for i, o in enumerate(outs):
            check(-(1 << 30) <= o < 1 << 30, f"VMATR out {o}")
            put(dst, eoff + i, o)
            toks += tracked(o, signed=mode == 0, first=i == 0 and not keep)
        if len(outs) % 2:
            put(dst, eoff + len(outs), 0)
            toks += word(0)
    elif op == "VQ":
        src, eoff, n, dst, sh, off, r = (f[x] for x in ("src", "eoff", "n", "dst", "sh", "off", "r"))
        check(n % 2 == 0 and 0 < n <= 64 and (dst or off + n <= 384), f"VQ n {n}")
        aligned(eoff, n)
        if dst == 0:
            aligned(off, n)
        qs = []
        for i in range(n):
            p = clamp16(v.vm[src][eoff + i] >> sh) * r
            q = (abs(p) + (1 << 14)) >> 15
            q = max(-127, min(127, q if p >= 0 else -q))
            qs.append(q)
            if dst == 0:
                v.va[off + i] = q
            else:
                (v.kc if dst == 1 else v.vc)[off + 32 * (i // 8)][i % 8] = q
        toks += [vh(q & 255) for q in qs]
    elif op == "VEW":
        a, b, n, sub, keep, eoff, sh, so, shb, p = (
            f[x] for x in ("a", "b", "n", "op", "keep", "eoff", "sh", "so", "shb", "p"))
        so = sx(so, 6)
        check(n % 2 == 0 and 0 < n <= 64 and sub < 3 and (sub == 2 or a >> 3 != b >> 3),
              f"VEW a {a} b {b} n {n} op {sub}")
        aligned(eoff, n)
        old = v.vm[a][eoff:eoff + n]
        out = []
        for i in range(n):
            ai = old[i]
            if sub == 0:
                o = ai + shift(clamp16(v.vm[b][eoff + i] >> sh) * p, so)
            elif sub == 1:
                o = (silu(shift(clamp16(ai >> sh) * p, so)) * clamp16(v.vm[b][eoff + i] >> shb)) >> 8
            else:
                c, sn = v.rope[(i % 8) // 2]
                u, w = clamp16(old[i & ~1] >> sh), clamp16(old[i | 1] >> sh)
                o = shift(u * c - w * sn, sh - 14) if i % 2 == 0 else shift(u * sn + w * c, sh - 14)
            out.append(sx(o, 31))
        for i, o in enumerate(out):
            put(a, eoff + i, o)
            toks += [ve(0), ve(0)] * (sub == 1) + tracked(o, first=i == 0 and not keep)
    elif op == "VEXP":
        src, n, off, ksh, c, mx = (f[x] for x in ("src", "n", "off", "ksh", "c", "m"))
        check(0 < n <= 64 and off + n + n % 2 <= 384, f"VEXP n {n}")
        aligned(off, n + n % 2)
        ps = []
        for i in range(n):
            e = min(sx(mx, 31) - v.vm[src][i], 65535)
            check(e >= 0, "VEXP above the max")
            d = (e * c) >> ksh
            ps.append(EXP_T[d] if d < 256 else 0)
            v.va[off + i] = ps[-1]
        if n % 2:
            ps.append(0)
            v.va[off + n] = 0
        toks += [vh(p) for p in ps]
    elif op == "VRED":
        n, mode, off = f["n"], f["mode"], f["off"]
        check(off < 6 and n <= 64, f"VRED chunk {off} n {n}")
        vals = v.va[64 * off:64 * off + n]
        v.vsum = sum(q * q for q in vals) if mode == 0 else sum(vals)
        toks += word(v.vsum)
    elif op == "VPUT":
        target, lo, hi = f["target"], s32(f["lo"]), s32(f["hi"])
        check(-(1 << 30) <= lo < 1 << 30 and -(1 << 30) <= hi < 1 << 30, "VPUT words")
        if target < 512:
            put(target >> 5, 2 * (target & 31), lo)
            put(target >> 5, 2 * (target & 31) + 1, hi)
        else:
            check(target < 516, f"VPUT target {target}")
            v.rope[target - 512] = (sx(lo, 16), sx(hi, 16))
        toks += word(lo) + word(hi)
    if len(toks) - n_vc > MAX_STREAM:
        raise ValueError(f"{op}: a {len(toks) - n_vc}-token stream (max {MAX_STREAM})")
    return toks + [S.tok("NT")]
