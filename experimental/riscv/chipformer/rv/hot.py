"""Hot-code units on the paged map (docs/fx/circuits2.md): exact macro-ops at fixed ROM PCs.

A unit is not a ROM word. Its trigger is a constant comparison on the PC (hot.Unit.pc), so the ROM
stays byte-identical to fx's image, and a decline is one token (M.MD) after which the original
instruction runs as compiled. ``record`` below is the reference (called by ref.PagedMachine);
``Hot`` is the circuit (called from core.build when cfg.units is not empty).

The trace of a unit. Tokens come from the family M (spec.py), the data family CW, PA and J:

  M tags (bits 12..13)   0  op: 0 MD (decline), 1 MR (a read position)
                         1  MP: x | f << 1 | dir << 2 | unit << 3: the unit fires
                         2  MC: cursor (a word index): the chunk boundary
                         3  MH: x | pre << 1: the chunk's header

  copy:  MP [PA] MH [MR] { CW(lo) CW(hi) }* { MC [PA] MH { CW(lo) CW(hi) }* }* J(ret)
         (descending: MP MC [PA] MH [MR] ...)
  scan:  MP [CW(start)] MF { SC.. CW(idx) MR }* SC.. CW(k) MK { value record }+ [PA] [MH CW CW] J(pc)
         a compiled byte loop's plain iterations: it exits at its own head, before the iteration
         that leaves the loop (which then runs as compiled), so the unit declines at k = 0
  fill:  the same, no source (memset)
  eql:   MP { SC SC }* [MC { SC SC }*].. J(ret), or at a mismatch F0(0) J(ret), or MD when a byte
         is outside every load region (eql has written nothing: the original then faults as
         compiled)

Why a chunk: the record counter j is the position in a chunk (2^-j, 66 at most), and a write's value
head comes from a token. So a chunk is at most 31 words, in one destination page, and its header's
x is that page's frame bit. The cursor MC carries is the only way down to the address adder: no
state climbs back through the layers without a token (docs/fx/circuits2.md).

Positions (each computes the next token; the role follows from the token it stands on):

  first   the trigger, no unit active: the core's adder gives d - s; fire? MP : MD
  R       EA = the source word before the first (a funnel needs two)
  a       EA = source word: V = funnel(prev, mem), merged with old by the byte mask; CW(V.lo)
  b       EA = dst word; CW(V.hi, from a stash); the hi token stands on the next position, which
          writes the word: key from this position (EA = dst), value from the two CW tokens
  H       EA = dst word: the frame; PA if unmapped; MH(x). Its read is the chunk's `old`

So a word is 2 tokens, a tail partial word costs a chunk boundary (its old word), a funnel costs
one MR, a call MP MH J and the ret. Declines: anything outside the first RAM window, 16 KB or more, a page budget under 34
free frames (the original then faults exactly as compiled), an overlapping memcpy.
"""

from dataclasses import dataclass

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan
from chipformer.q35.prims import from_bits, thead
from chipformer.rv import spec as S

CHUNK_J = 62  # positions in a chunk: 31 words
NMAX = 0x3FFC  # a call copies fewer bytes than this
FRAME_RESERVE = 34
TAG_OP, TAG_MP, TAG_MC, TAG_MH = 0, 1, 2, 3
OP_MD, OP_MR, OP_MF, OP_MK = 0, 1, 2, 3
A0, A1, A2, A3, A4, A6, S3, S4, S5 = 10, 11, 12, 13, 14, 16, 19, 20, 21
# a byte loop's unit: the registers read at the trigger and in the loop (rs1 = base, rs2, rd = end),
# start = the first index comes from rs2 + 1, last = the loop's count is in rd, so it stops one byte
# earlier; the commits (register, +1 base + k, -1 base - k, 0 k)
SCAN = {"tok": dict(first=(S5, 12, A6), loop=(S5, 12, A6), start=False, commits=((12, 1), (S5, 1), (A6, -1))),
        "chr": dict(first=(A1, A0, A2), loop=(A1, A0, A2), start=False, commits=((A1, 1), (A2, -1))),
        "ascii": dict(first=(A0, S3, S3), loop=(S4, A0, S3), start=True, commits=((A0, 0),)),
        "jscan": dict(first=(A2, A0, A4), loop=(A0, A3, A4), start=True, commits=((A3, 0),))}


@dataclass(frozen=True)
class Unit:
    """kind: move (memmove: d = a0, s = a1, n = a2), cpy (memcpy: also declines s < d < s + n),
    set (memset: d = a0, c = a1, n = a2), bcopy (a byte loop: d = a2, s = s11, end = a1),
    eql (a = a1, b = a3, n = a2, a0 = 1 on entry). pc: the trigger; ret: where the unit exits (a
    `ret` for function units), in the same 64K-word window."""

    kind: str
    pc: int
    ret: int


def m_tok(tag: int, v: int = 0) -> int:
    return S.tok("M", tag << 12 | v)


def tok_decline() -> int:
    return m_tok(TAG_OP, OP_MD)


def mp_tok(x, f, dr, unit) -> int:
    return m_tok(TAG_MP, x | f << 1 | dr << 2 | unit << 3)


def mh_tok(x, pre=0) -> int:
    return m_tok(TAG_MH, x | pre << 1)


def uid(m, u: Unit) -> int:
    return m.cfg.units.index(u)


def window_ok(m, a: int) -> bool:
    """a is in the first writable window, 16 KB below its end."""
    b, lg = m.cfg.rw[0]
    assert b == 0
    return a >> lg == 0 and (a >> 14) != (1 << (lg - 14)) - 1


def frames_ok(m) -> bool:
    return m.nf >> 5 < (m.cfg.frames - FRAME_RESERVE) >> 5


def move_fires(m, u: Unit, d: int, s: int, n: int) -> tuple:
    """(fires, dir). dir = 1: the descending copy (memmove, s < d < s + n)."""
    fwd = d < s or d - s >= n
    ok = (n != 0 and n < NMAX and d != s and window_ok(m, d) and window_ok(m, s) and frames_ok(m)
          and (fwd or u.kind == "move"))
    if ok:  # the original's frames are below sp: a buffer must not reach them (circuits2.md)
        assert m.cfg.rw_window(m.x[2] - 16) != 0 or m.x[2] > d + n + 64 and m.x[2] > s + n + 64 or (
            m.x[2] + 64 < min(d, s)), "the stack in a unit's window"
    return ok, int(not fwd)


def funnel(prev: int, mem: int, r: int, dr: int) -> int:
    """The destination word from two source words: byte b of the result is byte b + r of the pair
    (lower word first; dr = 1: mem is the lower one)."""
    if r == 0:
        return mem
    lo, hi = (mem, prev) if dr else (prev, mem)
    return ((lo >> 8 * r) | (hi << (32 - 8 * r))) & 0xFFFFFFFF


def lane_mask(lo: int, hi: int) -> int:
    return sum(0xFF << 8 * b for b in range(4) if lo <= b <= hi)


def move_record(m, u: Unit, d: int, s: int, n: int, dr: int, fill: int | None = None) -> list:
    dl, sl = (d & 3, d & 3) if fill is not None else (d & 3, s & 3)
    dw, sw = d & ~3, s & ~3
    tt = dl + n + 3
    k_words, e = tt >> 2, tt & 3
    last = k_words - 1
    r = (sl - dl) & 3
    f = int(sl < dl) if dr else int(sl > dl)
    toks = [mp_tok(0, f, dr, uid(m, u))]
    if dr:
        toks.append(m_tok(TAG_MC, last))
    c = last if dr else 0
    toks += m.alloc(dw + 4 * c)
    pre = int(r != 0 and e + r >= 4) if dr else f
    toks.append(mh_tok(m.pt[(dw + 4 * c) >> S.PAGE_BITS] & 1, pre))
    old = m.ram_word(dw + 4 * c)
    prev = 0
    if pre:
        prev = m.ram_word((sw + 4 * (c + 1 - f if dr else 0)) & 0xFFFFFFFF)
        toks.append(m_tok(TAG_OP, OP_MR))
    j = 0
    while True:
        i = j >> 1
        ci = c - i if dr else c + i
        if (ci < 0) if dr else (ci == k_words):
            break
        dst = dw + 4 * ci
        pg = (dst >> 2) & 127
        edge = (pg == 127 or (ci == 0 and dl != 0)) if dr else (pg == 0 or (ci == last and e != 3))
        if j > 0 and (j >= CHUNK_J or edge):
            toks.append(m_tok(TAG_MC, ci))
            toks += m.alloc(dst)
            toks.append(mh_tok(m.pt[dst >> S.PAGE_BITS] & 1))
            old = m.ram_word(dst)
            c, j = ci, 0
            continue
        if fill is None:
            w = m.ram_word((sw + 4 * (ci - f if dr else ci + f)) & 0xFFFFFFFF)
            v = funnel(prev, w, r, dr)
        else:
            w = 0
            v = fill * 0x01010101
        mask = lane_mask(dl if ci == 0 else 0, e if ci == last else 3)
        v = v & mask | old & ~mask & 0xFFFFFFFF
        toks += [S.tok("CW", v & 0xFFFF), S.tok("CW", v >> 16)]
        m.wr[dst >> 2] = v
        prev = w
        j += 2
    m.steps += 1
    return toks + m.jump_tokens(u.ret)


def eql_record(m, u: Unit, a: int, b: int, n: int) -> list:
    """a0 is 1 at the trigger. A byte outside every load region: MD, nothing committed."""
    toks = [mp_tok(0, 0, 0, uid(m, u))]
    res = 1
    j = c = 0

    def byte(ea):
        return m.ram_word(ea & ~3) >> 8 * (ea & 3) & 0xFF

    while True:
        ci = c + (j >> 1)
        if ci == n:
            break
        if j >= CHUNK_J:
            toks.append(m_tok(TAG_MC, ci))
            c, j = ci, 0
            continue
        ea, eb = (a + ci) & 0xFFFFFFFF, (b + ci) & 0xFFFFFFFF
        if not m.cfg.in_load(ea):
            m.skip_unit = True
            return toks + [tok_decline()]
        toks.append(S.tok("SC"))
        if not m.cfg.in_load(eb):
            m.skip_unit = True
            return toks + [tok_decline()]
        if byte(ea) != byte(eb):
            res = 0
            m.n += 1
            toks.append(S.tok("F0", 0))
            m.x[10] = 0
            break
        toks.append(S.tok("SC"))
        j += 2
    m.steps += 1
    return toks + m.jump_tokens(u.ret)


def stops(kind: str, b: int, sc: int) -> bool:
    """The loop leaves (or stops being plain) at byte b."""
    if kind == "tok":
        return 9 <= b <= 13 or b == 32
    if kind == "chr":
        return b == sc & 0xFF and sc < 256
    if kind == "ascii":
        return b >= 128
    return not (0x20 <= b < 0x80) or b in (0x22, 0x5C)


def scan_fires(m, u: Unit) -> bool:
    x, t = m.x, SCAN[u.kind]
    r1, r2, r3 = (x[i] for i in t["first"])
    if r3 >> 16:
        return False
    k = u.kind
    if k == "tok":  # (the run's carry)
        return r2 >> 31 == 0 and r3 != 0
    if k == "chr":
        return r3 != 0
    if k == "ascii":  # the loop's bound is an unsigned >=: the unit's is an equality
        return r1 < r2
    if k == "jscan":
        cur = x[A2] + 0x24
        return cur & 3 == 0 and m.cfg.rw_window(cur) == 0 and window_ok(m, cur) and frames_ok(m) and (
            (r1 - r2) & 0xFFFFFFFF >= r3 and (r2 - r1) & 0xFFFFFFFF > 0x28)
    return True


def scan_record(m, u: Unit) -> list | None:
    """The plain iterations of a byte loop (SCAN). None / MD: nothing committed."""
    x, t, kind = m.x, SCAN[u.kind], u.kind
    if not scan_fires(m, u):
        return None
    base, sc, end = (x[i] for i in t["loop"])
    toks = [mp_tok(0, 0, 0, uid(m, u))]
    idx = 0
    if t["start"]:
        idx = sc + 1
        if idx >> 16:
            m.skip_unit = True
            return toks + [tok_decline()]
        toks.append(S.tok("CW", idx))
    stop_at = end - 1 if kind in ("tok", "chr") else end
    toks.append(m_tok(TAG_OP, OP_MF))
    j = 0
    first = True
    while True:
        ea = (base + idx) & 0xFFFFFFFF
        bad = not m.cfg.in_load(ea)
        b = 0 if bad else m.ram_word(ea & ~3) >> 8 * (ea & 3) & 0xFF
        stop = idx == stop_at or idx == 0xFFFF or stops(kind, b, sc)
        if bad or (stop and first):
            m.skip_unit = True
            return toks + [tok_decline()]
        if stop:
            break
        idx += 1
        first = False
        if j >= CHUNK_J:
            toks += [S.tok("CW", idx), m_tok(TAG_OP, OP_MR)]
            j = 0
            continue
        toks.append(S.tok("SC"))
        j += 1
    k = idx if not t["start"] else idx - 1
    toks += [S.tok("CW", k), m_tok(TAG_OP, OP_MK)]
    for r, sg in t["commits"]:
        v = (x[r] + sg * k if sg else k) & 0xFFFFFFFF
        toks += S.value_record(v)
        x[r] = v
        m.n += 1
    if kind == "jscan":
        cur = x[A2] + 0x24
        toks += m.alloc(cur)
        toks += [mh_tok(m.pt[cur >> S.PAGE_BITS] & 1), S.tok("CW", k), S.tok("CW", 0)]
        m.wr[cur >> 2] = k
    m.steps += 1
    return toks + m.jump_tokens(u.ret)


def record(m, u: Unit) -> list | None:
    """The unit's whole record, committed; None when it declines (nothing changed)."""
    x = m.x
    if u.kind in ("move", "cpy"):
        d, s, n = x[10], x[11], x[12]
        fires, dr = move_fires(m, u, d, s, n)
        return move_record(m, u, d, s, n, dr) if fires else None
    if u.kind == "bcopy":
        d, s, n = x[12], x[27], (x[11] - x[27]) & 0xFFFFFFFF
        fires, dr = move_fires(m, u, d, s, n)
        return move_record(m, u, d, s, n, 0) if fires and not dr else None
    if u.kind == "set":
        d, n = x[10], x[12]
        if n == 0 or n >= NMAX or not window_ok(m, d) or not frames_ok(m):
            return None
        return move_record(m, u, d, d, n, 0, x[11] & 0xFF)
    if u.kind == "eql":
        a, b, n = x[11], x[13], x[12]
        if a == b or n == 0 or n >= NMAX:
            return None
        return eql_record(m, u, a, b, n)
    return scan_record(m, u)


# row 24's image (sha256 282e53d4...): name -> (kind, trigger, its instruction word, ret)
FX_R24 = {"memmove": ("move", 0x208f918), "memcpy": ("cpy", 0x208f73c), "memset": ("set", 0x208fa50),
          "eql": ("eql", 0x26733e4), "bcopy": ("bcopy", 0x21e3a98, 0x21e3aac),
          "tok": ("tok", 0x22bf6c0, 0x22bf6c0), "jscan": ("jscan", 0x264c040, 0x264c040),
          "ascii": ("ascii", 0x2649914, 0x2649914), "chr1": ("chr", 0x20cf5e8, 0x20cf5e8),
          "chr2": ("chr", 0x20cf610, 0x20cf610)}


def fx_units(elf_path, names) -> tuple:
    """The units of row 24's image, by name ("all": every built one). The exit of memmove is its
    entry beq's target (a ret); checked against the image's instruction words."""
    from chipformer.rv import elf, ref

    _, segs = elf.segments(elf_path)
    base, text = next((va, b) for va, b in segs if va == 1 << 25)

    def word(pc):
        return int.from_bytes(text[pc - base:pc - base + 4], "little")

    ins = ref.decode(word(FX_R24["memmove"][1]))
    assert ins.op == "BEQ" and (ins.rs1, ins.rs2) == (10, 11), "not row 24's image"
    ret = FX_R24["memmove"][1] + ins.imm
    assert word(ret) == 0x00008067
    names = list(FX_R24) if list(names) == ["all"] else list(names)
    out = []
    for n in names:
        e = FX_R24[n]
        r = e[2] if len(e) > 2 else ret
        if e[0] == "eql":
            ins = ref.decode(word(e[1]))
            assert ins.op == "BEQ" and (ins.rs1, ins.rs2) == (11, 13), "not row 24's image"
            r = e[1] + ins.imm
            assert word(r) == 0x00008067
        out.append(Unit(e[0], e[1], r))
    return tuple(out)


PORTS = {"move": (10, 11, 12), "cpy": (10, 11, 12), "set": (10, 11, 12), "bcopy": (12, 27, 11),
         "eql": (11, 13, 12)}  # the registers a unit reads: rs1, rs2, rd ports


class Hot:
    """The circuit (see the module doc). A unit's address goes through the core's own adder: the
    ports read the base (rs1) and the ROM's immediate is off, so EA = A + 4 * offset; the trigger
    position uses it as d - s. Everything else runs every step, whether a unit is active or not,
    so a unit's cost is its units: keep them few."""

    def __init__(self, c, cfg) -> None:
        self.c, self.cfg = c, cfg
        self.units = cfg.units
        self.kinds = {u.kind for u in cfg.units}
        self.tables = []
        self.frames = cfg.frames
        fam, pay = c.fam, c.pay
        one = c.one
        f = c.token_feature

        def tag_is(tg):
            def fn(t):
                nm, p = S.family(t) if t < S.VOCAB2 else (None, 0)
                return int(nm == "M" and p >> 12 == tg)
            return fn

        self.tMP = f("m.mp", tag_is(TAG_MP), full=True)
        self.tMC = f("m.mc", tag_is(TAG_MC), full=True)
        self.tMH = f("m.mh", tag_is(TAG_MH), full=True)
        self.tOP = f("m.op", tag_is(TAG_OP), full=True)
        self.tMD = c.and_([self.tOP, 1 - pay["M"][0], 1 - pay["M"][1]], "m.md")
        self.tMR = c.and_([self.tOP, pay["M"][0], 1 - pay["M"][1]], "m.mr")
        self.tMF = c.and_([self.tOP, 1 - pay["M"][0], pay["M"][1]], "m.mf")
        self.tMK = c.and_([self.tOP, pay["M"][0], pay["M"][1]], "m.mk")
        self.tpre = c.and_([self.tMH, pay["M"][1]], "m.pre")
        mp = pay["M"]
        # the unit's flags, for the whole record
        (hl,) = thead(c, "h.l", {0: one}, {0: one},
                      [{"data": {2 + i: e for i, e in enumerate([self.tMP] + mp[1:7])}, "ref": one,
                        "beta": self.tMP + fam["J"] + self.tMD}])
        self.tables.append("h.l")
        c.tables["h.l"] = c.nodes[-1]
        self.inm, self.hf, self.hdir, self.hu = hl[0], hl[1], hl[2], hl[3:7]
        ((self.hx,),) = thead(c, "h.x", {0: one}, {0: one},
                              [{"data": {2: mp[0]}, "ref": one, "beta": self.tMH}])
        self.tables.append("h.x")
        c.tables["h.x"] = c.nodes[-1]
        self.scans = [u for u, un in enumerate(cfg.units) if un.kind in SCAN]
        huc = [c.and_([self.inm] + _lits(self.hu, u), f"h.uc{u}") for u in self.scans]
        self.iscan = sum(huc, Expr())
        self.tscw = c.and_([self.iscan, fam["CW"]], "h.tscw") if self.scans else Expr()
        cw_ = pay["CW"]
        (self.hc,) = thead(c, "h.c", {0: one}, {0: one},
                           [{"data": {2 + i: (mp[i] if i < 12 else Expr()) + (cw_[i] if self.scans else Expr())
                                      for i in range(16)}, "ref": one,
                             "beta": self.tMC + fam["J"] + fam["R"] + fam["JF"] + fam["RF"] + self.tscw}])
        self.tables.append("h.c")
        c.tables["h.c"] = c.nodes[-1]
        if self.scans:  # a unit's commits so far, since MK (cleared by MP)
            ((self.incm,),) = thead(c, "h.kf", {0: one}, {0: one},
                                    [{"data": {2: self.tMK}, "ref": one, "beta": self.tMK + self.tMP + fam["J"]}])
            self.tables.append("h.kf")
            c.tables["h.kf"] = c.nodes[-1]
            xk = c.counter("h.k", self.tMK, c.final)
            self.kt = c.thermo(xk, self.tMK, 3, "h.k")
            self.tables.append("h.k")

    def setup(self, w, jt, j0, at_max, isJA) -> None:
        from chipformer.rv import core as C

        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        self.jt, self.j0 = jt, j0
        jmax = len(jt)

        def jeq(v):
            return (one if v == 0 else jt[v - 1]) - (jt[v] if v < jmax else 0)

        jint = sum(jt, Expr())
        self.jodd = sum((jeq(v) for v in range(1, jmax + 1, 2)), Expr())
        self.first, self.hun, self.act = [], [], []
        for u, un in enumerate(self.units):
            wi = (un.pc - cfg.rom_base) >> 2
            assert wi >> 16 == (un.ret - cfg.rom_base) >> 18, "a unit's exit leaves its 64K-word window"
            trig = c.and_(C.lits_of(w, wi), f"h.t{u}")
            fi = c.and_([trig, j0, 1 - at_max, 1 - self.inm, 1 - self.tMD], f"h.first{u}")
            hu = c.and_([self.inm] + C.lits_of(self.hu, u), f"h.u{u}")
            self.first.append(fi)
            self.hun.append(hu)
            self.act.append(fi + hu)
        self.mact = sum(self.act, Expr())
        self.first_all = sum(self.first, Expr())
        # the roles
        cw = c.fam["CW"]
        COPY = ("move", "cpy", "set", "bcopy")
        imv = sum((h for h, un in zip(self.hun, self.units) if un.kind in COPY), Expr())
        ieq = sum((h for h, un in zip(self.hun, self.units) if un.kind == "eql"), Expr())
        self.imv, self.ieq = imv, ieq
        self.kact = {k: sum((a for a, un in zip(self.act, self.units) if un.kind == k), Expr())
                     for k in ("set", "bcopy", "eql")}
        tmp, tmh, tmc, tmr = self.tMP, self.tMH, self.tMC, self.tMR
        tpa = c.fam["PA"]
        sc = c.fam["SC"]
        self.regk = {}
        if "eql" in self.kinds:
            self.re2 = c.and_([ieq, sc, self.jodd], "h.re2")
            self.re1 = c.and_([ieq, sc, 1 - self.jodd], "h.re1a") + c.and_([ieq, tmp + tmc], "h.re1b")
            self.rx = c.and_([ieq, c.fam["F0"]], "h.rx")
            self.regk = {10: ieq}
        else:
            self.re1 = self.re2 = self.rx = zero
        self.rC = c.and_([imv, tmp, self.hdir], "h.rC")
        self.rR = c.and_([imv, self.tpre], "h.rR")
        self.rH = c.and_([imv, tmc + tpa], "h.rH") + c.and_([imv, tmp, 1 - self.hdir], "h.rH1")
        self.rb = c.and_([imv, cw, self.jodd], "h.rb")
        self.rhi = c.and_([imv, cw, 1 - self.jodd], "h.rhi")
        self.ra = self.rhi + c.and_([imv, tmr], "h.ra1") + c.and_([imv, tmh, 1 - self.tpre], "h.ra2")
        self.ead = self.rb + self.rH + self.re1
        self.eas = self.ra + self.rR + self.re2
        self.ea = self.ead + self.eas
        self.store = self.rb + self.rH
        self.cv = zero  # a scan unit's commit (of the value A +- k)
        self.sub_k = zero
        self.rbyte = zero
        js = [u for u in self.scans if self.units[u].kind == "jscan"]
        if self.scans:
            iscan, incm = self.iscan, self.incm
            kt = self.kt
            mq = [1 - kt[0], kt[0] - kt[1], kt[1] - kt[2], kt[2]]
            self.rbyte = c.and_(
                [iscan, 1 - incm, self.tMF + self.tMR + sc], "h.rbyte")
            self.rcm = []  # per scan unit, per commit: a position of that record
            self.rend = []
            self.ports_cm = []
            for u in self.scans:
                cm = SCAN[self.units[u].kind]["commits"]
                rc = [c.and_([self.hun[u], incm, 1 - c.fam["PA"], 1 - tmh, 1 - cw, mq[i]], f"h.rc{u}.{i}")
                      for i in range(len(cm))]
                self.rcm.append(rc)
                self.rend.append(c.and_([self.hun[u], incm, mq[len(cm)]], f"h.rend{u}"))
                for i, (r, sg) in enumerate(cm):
                    self.regk[r] = self.regk.get(r, zero) + rc[i]
                    self.cv = self.cv + rc[i]
                    if sg < 0:
                        self.sub_k = self.sub_k + rc[i]
            self.ea = self.ea + self.rbyte + self.cv
        self.jsH = self.jsa = self.jsb = self.jsw = zero
        if js:  # the cursor's store: H, then its word
            je = self.rend[self.scans.index(js[0])]
            self.jsH = c.and_([je, 1 - tmh, 1 - cw], "h.jsH")
            self.jsa = c.and_([je, tmh], "h.jsa")
            self.jsb = c.and_([je, cw, self.jodd], "h.jsb")
            self.jsw = c.and_([je, cw, 1 - self.jodd], "h.jsw")
            self.ea = self.ea + self.jsH + self.jsb
            self.store = self.store + self.jsH + self.jsb
        self.wr = [c.and_([self.rhi + self.jsw, 1 - self.hx], "h.wr0"),
                   c.and_([self.rhi + self.jsw, self.hx], "h.wr1")]
        # the ports: rs1 is the base of EA, rs2 the other buffer, rd the length
        self.ports = {"rs1": {}, "rs2": {}, "rd": {}}
        self.swap = zero

        def add(pn, r, e):
            self.ports[pn][r] = self.ports[pn].get(r, Expr()) + e

        for u, un in enumerate(self.units):
            if un.kind in SCAN:
                t = SCAN[un.kind]
                for pn, a_, b_ in zip(("rs1", "rs2", "rd"), t["first"], t["loop"]):
                    add(pn, a_, self.first[u])
                    add(pn, b_, c.and_([self.hun[u], 1 - self.incm], f"h.pl{u}{pn}"))
                x_ = self.scans.index(u)
                for i, (r, sg) in enumerate(t["commits"]):
                    if sg:
                        add("rs1", r, self.rcm[x_][i])
                if un.kind == "jscan":
                    add("rs1", A2, self.jsH + self.jsb)
                continue
            d_, s_, n_ = PORTS[un.kind]
            sw = zero if un.kind == "set" else c.and_([self.hun[u], self.eas], f"h.swap{u}")
            self.swap = self.swap + sw
            add("rs1", d_, self.act[u] - sw)
            add("rs1", s_, sw)
            add("rs2", s_, self.act[u] - sw)
            add("rs2", d_, sw)
            add("rd", n_, self.act[u])
        # the word index ci = c +- (j >> 1), and the source word's offset
        ib = [c.node(t, f"h.i{b}") for b, t in
              enumerate(c.fn_bits(jint, 0, jmax, lambda v: v >> 1, 5, "h.i"))]
        if self.scans:  # a byte loop's index is c + j (a commit's: c)
            jb = [c.node(t, f"h.jb{b}") for b, t in
                  enumerate(c.fn_bits(jint, 0, jmax, lambda v: v, 6, "h.jb"))]
            ib = [c.node(c.t_and([1 - self.iscan, b_]) if b_ else [], f"h.ib{i}") if b_ else zero
                  for i, b_ in enumerate(ib + [zero])]
            ib = [i_ + c.and_([self.rbyte, b_], f"h.ib2{k}") for k, (i_, b_) in enumerate(zip(ib, jb))]
        cb = self.hc
        ci_f = C.add_nibbles(c, cb, (ib + [zero] * 16)[:16], 4, [1 - self.hdir], "h.cif")
        if "move" in self.kinds:
            nib_ = [1 - b for b in ib] + [one] * 11
            sums = [[(from_bits(cb[4 * k:4 * k + 4]) + from_bits(nib_[4 * k:4 * k + 4]) + (1 if k == 0 else 0),
                      [self.hdir])] for k in range(4)]
            g, p_, d = C.stage1_digits(c, sums, "h.cib", 31)
            ci_b = C.adder_stage2(c, g, p_, d, 4, "h.cib")[0]
            self.ci = [ci_f[b] + ci_b[b] for b in range(14)] + ci_f[14:]  # (bit 13: negative, dir = 1)
        else:
            self.ci = ci_f
        ci = self.ci
        cp1, cm1 = [], []
        for b in range(16):
            low = ci[:b]
            cp1.append(c.node(c.t_and([ci[b]]) + c.t_and(low + [1 - ci[b]]) + c.t_and(low + [ci[b]], -1)
                              if b else c.t_const(1) + c.t_and([ci[0]], -1), f"h.cp{b}"))
            if "move" in self.kinds:
                nlow = [1 - e for e in low]
                cm1.append(c.node(c.t_and([ci[b]]) + c.t_and(nlow + [1 - ci[b]]) + c.t_and(nlow + [ci[b]], -1)
                                  if b else c.t_const(1) + c.t_and([ci[0]], -1), f"h.cm{b}"))
        self.cp1 = cp1
        sel_p = c.node(c.t_and([self.ra, 1 - self.hdir, self.hf]) + c.t_and([self.rR, self.hdir, 1 - self.hf]),
                       "h.selp")
        sel_m = c.and_([self.ra, self.hdir, self.hf], "h.selm") if cm1 else zero
        wordrole = self.ead + self.eas
        sel_z = wordrole - self.re1 - self.re2 - sel_p - sel_m
        self.off = [zero, zero] + [
            c.node(c.t_and([sel_z, ci[b]]) + c.t_and([sel_p, cp1[b]])
                   + (c.t_and([sel_m, cm1[b]]) if cm1 else []), f"h.off{b}") for b in range(14)] + [zero] * 16
        byt = self.re1 + self.re2 + self.rbyte + self.cv
        if self.scans or "eql" in self.kinds:  # a byte index
            self.off = [o + c.and_([byt, ci[b]], f"h.offb{b}") if b < 16 else o for b, o in enumerate(self.off)]
        if js:  # the cursor's offset
            self.off[2] = self.off[2] + self.jsH + self.jsb
            self.off[5] = self.off[5] + self.jsH + self.jsb

    def port(self, pname, r):
        return self.ports[pname].get(r, Expr())

    def regkey(self, r):
        return self.regk.get(r, Expr())

    def wbeta(self, x):
        return self.wr[x]

    def _add2(self, a, b, name):
        """(a + b) & 3 of two-bit values."""
        c = self.c
        z = from_bits(a) + from_bits(b)
        return [c.node(t, f"{name}{i}") for i, t in enumerate(c.fn_bits(z, 0, 6, lambda v: v & 3, 2, name))]

    def top(self, A, B, CR, r, eq, sltu, mem, fb, mapped, anext, af, EA, in_load, case) -> None:
        """Above the RAM: stashes, the funnel, the mask, the conditions, the token cases."""
        from chipformer.rv import core as C

        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        ci = self.ci
        hdir = self.hdir
        kinds = self.kinds
        nib = lambda bits, k: from_bits(bits[4 * k:4 * k + 4])
        lits_of = C.lits_of

        def eqbits(a, b, name):
            n = len(a)
            parts = [c.node(c.t_eq(from_bits(a[k:k + 4]) - from_bits(b[k:k + 4]), 0, -15, 15), f"{name}.{k}")
                     for k in range(0, n, 4)]
            return c.and_(parts, name)

        def or_(lits, name):
            return one - c.and_([1 - e for e in lits], name)

        # the two buffers' low bits at any position: the ports swap on source reads
        swap = self.swap
        dv = [c.node(c.t_and([swap, B[i]]) + c.t_and([1 - swap, A[i]]), f"h.dv{i}") for i in range(2)]
        sv = [c.node(c.t_and([swap, A[i]]) + c.t_and([1 - swap, B[i]]), f"h.sv{i}") for i in range(2)]
        isset, isbc = self.kact["set"], self.kact["bcopy"]
        if "bcopy" in kinds:  # n = end - s
            s32 = [c.node(c.t_and([swap, A[i]]) + c.t_and([1 - swap, B[i]]), f"h.s32{i}") for i in range(32)]
            ns = [[(nib(CR, k) + 15 - nib(s32, k) + (1 if k == 0 else 0), [])] for k in range(8)]
            g, p_, d = C.stage1_digits(c, ns, "h.ns", 31)
            sub = C.adder_stage2(c, g, p_, d, 8, "h.ns")[0]
            CR = [c.node(c.t_and([1 - isbc, CR[i]]) + c.t_and([isbc, sub[i]]), f"h.nv{i}") for i in range(32)]
        n_ok = c.and_([1 - b for b in CR[14:32]] + [1 - c.and_(CR[2:14], "h.ntop")], "h.nok")
        n_nz = or_(CR[:14], "h.nnz")
        d_ne = 1 - eq
        copy = kinds & {"move", "cpy", "set", "bcopy"}
        if copy:
            # n's arithmetic: U = (d & 3) + n + 3: K = U >> 2 words, e = U & 3 the last byte's lane
            sums = [[(nib(CR, 0) + from_bits(dv) + 3, [])]] + [[(nib(CR, k), [])] for k in range(1, 4)]
            g, p_, d = C.stage1_digits(c, sums, "h.u", 30)
            U = C.adder_stage2(c, g, p_, d, 4, "h.u")[0]
            Kb, e2 = U[2:16], U[0:2]
            eq_k = eqbits(ci, Kb, "h.eqk")
            lastw = eqbits(self.cp1, Kb, "h.last")
            first0 = c.and_([1 - b for b in ci], "h.first0")
            rr = [c.node([t for sl in range(4) for dl in range(4) if (sl - dl) & 3 == rv
                          for t in c.t_and(lits_of(sv, sl) + lits_of(dv, dl) + [1 - isset])], f"h.r{rv}")
                  for rv in range(4)]
            e_ne3 = 1 - c.and_(list(e2), "h.e3")
            d_ne0 = or_(list(dv), "h.dz")
            # the byte mask of word ci
            dint, eint = from_bits(dv), from_bits(e2)
            mask = []
            for b in range(4):
                ge = c.node(c.t_ge(b - dint, 0, -3, 3), f"h.ge{b}")
                le = c.node(c.t_ge(eint - b, 0, -3, 3), f"h.le{b}")
                m1 = or_([1 - first0, ge], f"h.m1{b}")
                m2 = or_([1 - lastw, le], f"h.m2{b}")
                mask.append(c.and_([m1, m2], f"h.mask{b}"))
        # the conditions at the trigger: r = d - s (the core's adder)
        lg = cfg.rw[0][1]
        wok = lambda v, nm: c.and_([1 - b for b in v[lg:32]] + [1 - c.and_(v[14:lg], nm + "t")], nm)
        lim = (self.frames - FRAME_RESERVE) >> 5
        gt = [c.node(c.t_lt(nib(CR, k) - nib(r, k), 0, -15, 15), f"h.gt{k}") for k in range(4)]
        eqn = {k: c.node(c.t_eq(nib(CR, k) - nib(r, k), 0, -15, 15), f"h.eq{k}") for k in (1, 2, 3)}
        r_ge = c.node(c.t_and([gt[3]]) + c.t_and([eqn[3], gt[2]]) + c.t_and([eqn[3], eqn[2], gt[1]])
                      + c.t_and([eqn[3], eqn[2], eqn[1], gt[0]]), "h.rgt")
        # (gt: r > n on the low 14 bits' nibbles; the equal case, r == n, is r >= n too)
        r_eq = eqbits(r[:14], CR[:14], "h.req")
        r_hi = or_(r[14:32], "h.rhi0")
        fwd = c.node(c.t_and([sltu]) + c.t_and([1 - sltu, r_hi]) + c.t_and([1 - sltu, 1 - r_hi, r_ge])
                     + c.t_and([1 - sltu, 1 - r_hi, 1 - r_ge, r_eq]), "h.fwd")
        self.fge = fwd
        lag = (0, 1, 0, 0)
        imv = self.imv
        s_gt = c.node(c.t_lt(from_bits(dv) - from_bits(sv), 0, -3, 3), "h.sgt")
        s_lt = c.node(c.t_lt(from_bits(sv) - from_bits(dv), 0, -3, 3), "h.slt")
        f_now = c.node(c.t_and([fwd, s_gt]) + c.t_and([1 - fwd, s_lt]), "h.fnow")
        fok = c.node(c.t_lt(from_bits(af[5:]), lim, 0, (self.frames - 1) >> 5), "h.fok")
        wa = wok(A, "h.wa")
        common = c.and_([wa, wok(B, "h.wb"), n_ok, n_nz, d_ne, fok], "h.common") if copy else zero
        self.fok = fok
        # stashes: prev (the source word read last), old (the destination word of the header)
        vhs = [{"data": {2 + i: Chan(mem[i], lag) for i in range(32)}, "ref": one,
                "beta": self.rb + c.and_([imv, self.tMR], "h.bm") + self.re2}]
        if copy:
            vhs.append({"data": {2 + i: Chan(mem[i], lag) for i in range(32)}, "ref": one,
                        "beta": c.and_([imv, self.tMH], "h.bo")})
        outs = thead(c, "h.s", {0: one}, {0: one}, vhs)
        prev = outs[0]
        self.tables.append("h.s")
        c.tables["h.s"] = c.nodes[-1]

        def pay_const(tag, vbits):
            return list(vbits) + [zero] * (12 - len(vbits)) + [one if tag >> b & 1 else zero for b in range(2)]

        def ubits(u):
            return [one if u >> b & 1 else zero for b in range(4)]

        decline = pay_const(TAG_OP, [zero])  # OP_MD
        if copy:
            old = outs[1]
            # the funnel, merged with old
            vt = [[] for _ in range(32)]
            for b in range(4):
                for t in range(8):
                    i = 8 * b + t
                    vt[i] += c.t_and([rr[0], mask[b], mem[i]])
                    vt[i] += c.t_and([isset, mask[b], B[t]])
                    for rv in (1, 2, 3):
                        if b + rv < 4:
                            src = 8 * (b + rv) + t
                            vt[i] += c.t_and([rr[rv], mask[b], 1 - hdir, prev[src]])
                            vt[i] += c.t_and([rr[rv], mask[b], hdir, mem[src]])
                        else:
                            src = 8 * (b + rv - 4) + t
                            vt[i] += c.t_and([rr[rv], mask[b], 1 - hdir, mem[src]])
                            vt[i] += c.t_and([rr[rv], mask[b], hdir, prev[src]])
                    vt[i] += c.t_and([1 - mask[b], old[i]])
            v = [c.node(t, f"h.v{i}") for i, t in enumerate(vt)]
            # is this word the last of its page, in the copy's direction (at b: EA is the word)
            pe = c.node(c.t_and([1 - hdir] + EA[2:9]) + c.t_and([hdir] + [1 - x for x in EA[2:9]]), "h.pe")
            hi, (pgedge,) = thead(c, "h.hi", {0: one}, {0: one},
                                  [{"data": {2 + i: Chan(e, lag) for i, e in enumerate(v[16:])},
                                    "ref": one, "beta": self.rb},
                                   {"data": {2: Chan(pe, lag)}, "ref": one, "beta": self.rhi}])
            self.tables.append("h.hi")
            c.tables["h.hi"] = c.nodes[-1]
            full = self.jt[CHUNK_J - 1]
            edge = c.node(c.t_and([1 - hdir, lastw, e_ne3]) + c.t_and([hdir, first0, d_ne0]), "h.edge")
            bnd = c.node(c.t_and([self.jt[0]]) + c.t_and([self.jt[0], 1 - full, 1 - pgedge, 1 - edge], -1), "h.bnd")
            done = c.node(c.t_and([1 - hdir, eq_k]) + c.t_and([hdir, ci[13]]), "h.done")
        if "eql" in kinds:
            # the bytes at a + ci (read one position ago, in prev) and b + ci
            ca, cb = self._add2(dv, ci[0:2], "h.laa"), self._add2(sv, ci[0:2], "h.lab")
            la = [c.and_(lits_of(ca, k), f"h.la{k}") for k in range(4)]
            lb = [c.and_(lits_of(cb, k), f"h.lb{k}") for k in range(4)]
            ab = [c.node([x for k in range(4) for x in c.t_and([la[k], prev[8 * k + t]])], f"h.ab{t}")
                  for t in range(8)]
            bb = [c.node([x for k in range(4) for x in c.t_and([lb[k], mem[8 * k + t]])], f"h.bb{t}")
                  for t in range(8)]
            same = c.node(c.t_eq(from_bits(ab) - from_bits(bb), 0, -255, 255), "h.same")
            eq_n = eqbits(ci, CR[:14], "h.eqn")
            fire_eql = c.and_([d_ne, n_ok, n_nz], "h.feql")
            full = self.jt[CHUNK_J - 1]
        for u, un in enumerate(self.units):
            fi = self.first[u]
            t = (un.ret - cfg.rom_base) >> 2 & 0xFFFF
            jbits = [one if t >> b & 1 else zero for b in range(16)]
            if un.kind in SCAN:
                continue
            if un.kind == "eql":
                case([fi, 1 - fire_eql], "M", decline)
                case([fi, fire_eql], "M", pay_const(TAG_MP, [zero, zero, zero] + ubits(u)))
                case([self.hun[u], self.re1, eq_n], "J", jbits)
                case([self.hun[u], self.rx], "J", jbits)
                continue
            if un.kind == "set":
                fire = c.and_([wa, n_ok, n_nz, fok], "h.fset")
                flags = [zero, zero]
            elif un.kind == "move":
                fire, flags = common, [f_now, 1 - fwd]
            else:
                fire = c.and_([common, fwd], f"h.fire{u}")
                flags = [f_now, zero]
            case([fi, 1 - fire], "M", decline)
            case([fi, fire], "M", pay_const(TAG_MP, [zero] + flags + ubits(u)))
            case([self.hun[u], self.ra, done], "J", jbits)
        if copy:
            case([self.ra, 1 - done, bnd], "M", pay_const(TAG_MC, ci[:12]))
            case([self.ra, 1 - done, 1 - bnd], "CW", v[:16])
            case([self.rb], "CW", hi)
            if "move" in kinds:  # K - 1, the descending copy's first MC
                k1 = []
                for b in range(12):
                    nl = [1 - x for x in Kb[:b]]
                    k1.append(c.node(c.t_and([Kb[b]]) + c.t_and(nl + [1 - Kb[b]]) + c.t_and(nl + [Kb[b]], -1)
                                     if b else c.t_const(1) + c.t_and([Kb[0]], -1), f"h.k1{b}"))
                case([self.rC], "M", pay_const(TAG_MC, k1))
            case([self.rR], "M", pay_const(TAG_OP, [one]))  # OP_MR
            r_int = rr[1] + 2 * rr[2] + 3 * rr[3]
            pre = c.node(c.t_ge(eint + r_int, 4, 0, 6, lits=[hdir, lastw, 1 - rr[0]])
                         + c.t_and([1 - hdir, first0, self.hf]), "h.pre")
            case([self.rH, mapped], "M", pay_const(TAG_MH, [fb[0], pre]))
            case([self.rH, 1 - mapped], "PA", anext)
        if self.scans:
            self.top_scan(A, B, CR, r, sltu, mem, fb, mapped, anext, EA, in_load, case, pay_const, ubits, or_, eqbits,
                          decline, wok, fok)
        if "eql" in kinds:
            bad = 1 - in_load
            case([self.re1, 1 - eq_n, 1 - full, in_load], "SC")
            case([self.re1, 1 - eq_n, full], "M", pay_const(TAG_MC, ci[:12]))
            case([self.re1, 1 - eq_n, 1 - full, bad], "M", decline)
            case([self.re2, bad], "M", decline)
            case([self.re2, in_load, same], "SC")
            case([self.re2, in_load, 1 - same], "F0", [zero] * 15)

    def top_scan(self, A, B, CR, r, sltu, mem, fb, mapped, anext, EA, in_load, case, pay_const, ubits, or_, eqbits,
                 decline, wok, fok) -> None:
        """A byte loop: the byte's class at the top, the index through tokens (module doc)."""
        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        ci, cp1 = self.ci, self.cp1
        lane = EA[0:2]
        lk = [c.and_(_lits(lane, k), f"h.lk{k}") for k in range(4)]
        by = [c.node([x for k in range(4) for x in c.t_and([lk[k], mem[8 * k + t]])], f"h.by{t}") for t in range(8)]
        bint = from_bits(by)
        n_hi0 = c.and_([1 - b for b in CR[16:32]], "h.n16")
        rs = self.rbyte
        kinds = {self.units[u].kind for u in self.scans}
        eq_e = eqbits(ci[:16], CR[:16], "h.eqe") if kinds & {"ascii", "jscan"} else zero
        eq_l = eqbits(cp1[:16], CR[:16], "h.eql2") if kinds & {"tok", "chr"} else zero
        cls = {}
        if "tok" in kinds:
            cls["tok"] = c.node(c.t_in(bint, 9, 14, 0, 255) + c.t_eq(bint, 32, 0, 255), "h.ctok")
        if "chr" in kinds:
            bh0 = c.and_([1 - x for x in B[8:32]], "h.b8")
            cls["chr"] = c.node(c.t_eq(bint - from_bits(B[:8]), 0, -255, 255, lits=[bh0]), "h.cchr")
        if "ascii" in kinds:
            cls["ascii"] = by[7]
        if "jscan" in kinds:
            cls["jscan"] = c.node(c.t_lt(bint, 0x20, 0, 255) + c.t_ge(bint, 0x80, 0, 255) + c.t_eq(bint, 0x22, 0, 255)
                                  + c.t_eq(bint, 0x5C, 0, 255), "h.cjs")
        stop_u, full = [], self.jt[CHUNK_J - 1]
        idx_top = c.and_(ci[:16], "h.itop")
        n_nz = or_(CR[:16], "h.nnz2")
        for x_, u in enumerate(self.scans):
            k = self.units[u].kind
            st = or_([cls[k], eq_l if k in ("tok", "chr") else eq_e, idx_top], f"h.st{u}")
            stop_u.append(c.and_([self.hun[u], rs, st], f"h.stop{u}"))
        stop = sum(stop_u, Expr())
        keep = rs - stop
        bad = 1 - in_load
        k_end = []
        for b in range(16):  # the loops with a start index keep idx - 1 in their register
            low = [1 - e for e in ci[:b]]
            k_end.append(c.node(c.t_and([ci[b]]) + c.t_and(low + [1 - ci[b]]) + c.t_and(low + [ci[b]], -1)
                                if b else c.t_const(1) + c.t_and([ci[0]], -1), f"h.ke{b}"))
        ks = sum((self.hun[u] for u in self.scans if SCAN[self.units[u].kind]["start"]), Expr())
        kv = [c.node(c.t_and([1 - ks, ci[b]]) + c.t_and([ks, k_end[b]]), f"h.kv{b}") for b in range(16)]
        lag = (0, 1, 0, 0)
        # what a CW stands for at the next position: the stop, or a start
        (st0, st1), = thead(c, "h.st", {0: one}, {0: one},
                            [{"data": {2: Chan(stop, lag), 3: Chan(self.tMP, lag)}, "ref": one, "beta": self.tscw}])
        self.tables.append("h.st")
        c.tables["h.st"] = c.nodes[-1]
        case([rs, bad], "M", decline)
        case([rs, in_load, stop, self.tMF], "M", decline)
        case([rs, in_load, stop, 1 - self.tMF], "CW", kv)
        case([rs, in_load, keep, 1 - full], "SC")
        case([rs, in_load, keep, full], "CW", cp1[:16])
        rcw = self.tscw
        case([rcw, 1 - self.incm, st0], "M", pay_const(TAG_OP, [one, one]))  # OP_MK
        case([rcw, 1 - self.incm, 1 - st0, st1], "M", pay_const(TAG_OP, [zero, one]))  # OP_MF
        case([rcw, 1 - self.incm, 1 - st0, 1 - st1], "M", pay_const(TAG_OP, [one, zero]))  # OP_MR
        lim_b = c.and_([1 - b for b in B[16:32]], "h.b16")
        bp1 = []
        for b in range(16):
            low = B[:b]
            bp1.append(c.node(c.t_and([B[b]]) + c.t_and(low + [1 - B[b]]) + c.t_and(low + [B[b]], -1)
                              if b else c.t_const(1) + c.t_and([B[0]], -1), f"h.bp{b}"))
        b_carry = c.and_(B[:16], "h.bc")
        for x_, u in enumerate(self.scans):
            un = self.units[u]
            t = (un.ret - cfg.rom_base) >> 2 & 0xFFFF
            jbits = [one if t >> b & 1 else zero for b in range(16)]
            fi = self.first[u]
            if un.kind == "tok":
                fire = c.and_([n_hi0, n_nz, 1 - B[31]], f"h.f{u}")
            elif un.kind == "jscan":
                # first: A = a2, B = a0, C = a4: the cursor is outside [a0, a0 + a4) by a margin
                top = c.and_(r[6:32], f"h.rtop{u}")
                fire = c.and_([n_hi0, wok(A, "h.wa2"), self.fge, 1 - top, 1 - A[0], 1 - A[1], fok], f"h.f{u}")
            elif un.kind == "chr":
                fire = c.and_([n_hi0, n_nz], f"h.f{u}")
            else:
                fire = c.and_([n_hi0, sltu], f"h.f{u}")
            case([fi, 1 - fire], "M", decline)
            case([fi, fire], "M", pay_const(TAG_MP, [zero, zero, zero] + ubits(u)))
            ist = c.and_([self.hun[u], self.tMP], f"h.ist{u}")
            if SCAN[un.kind]["start"]:
                case([ist, 1 - lim_b], "M", decline)
                case([ist, lim_b, 1 - b_carry], "CW", bp1)
                case([ist, lim_b, b_carry], "M", decline)
            else:
                case([ist], "M", pay_const(TAG_OP, [zero, one]))  # OP_MF
            if un.kind == "jscan":
                case([self.jsH, 1 - mapped], "PA", anext)
                case([self.jsH, mapped], "M", pay_const(TAG_MH, [fb[0], zero]))
                case([self.jsa], "CW", self.hc)
                case([self.jsb], "CW", [zero] * 16)
                case([self.jsw], "J", jbits)
            else:
                case([self.rend[x_]], "J", jbits)


def _lits(bits, v):
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def rom_absent(rom, cfg) -> tuple:
    """The uops no ROM word holds (the logic they gate is pruned: ROM-specialized, as the units
    are). ILLEGAL stays: an unwritten ROM word reads as it."""
    from chipformer.rv import ref

    used = {ref.U["ILLEGAL"]}
    for w in set(rom):
        if w:
            used.add(ref.rom_word(w, cfg.rom_base, None, cfg) & 0x7F)
    return tuple(sorted(u for u in ref.UOPS if ref.U[u] not in used))
