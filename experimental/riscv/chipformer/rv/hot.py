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
  scan:  MP [CW(start)] MF { SC.. EC1 CW(idx) MR }* SC.. EC0 CW(k) MK { value record }+ [PA MH CW CW] J(pc)
         a compiled byte loop's plain iterations: it exits at its own head, before the iteration
         that leaves the loop (which then runs as compiled), so the unit declines at k = 0
  sha:   MP CW CW MN | MN | MN | { CW CW MN }x8 | { [PA] MH CW(W lo) CW(W hi) CW(a lo) CW(a hi) CW(e lo)
         CW(e hi) MN }x64 | { [PA] MH CW CW MN }x8 J(ret): the wasm stack pointer, two overlap checks,
         the state, the 64 rounds (their state and schedule live in rings written from these tokens,
         the rounds' logic is in the MLPs below the RAM), the state back
  jesc:  MP CW(len) CW(start) MF { SC | EB(class) .. EC1 CW(idx) MR }* SC.. EC0 CW(k) MK {s8}{a3} J(pc):
         a scan whose bound comes from memory and that keeps the last byte's path in a3 (EB)
  maskp: MP J(exit), or MP MD: every prefix of the probe is skipped, nothing is compared
  fill:  the same, no source (memset)
  eql:   MP { EB(a[k]) EC1 }* [MC { EB EC1 }*].. J(ret), or at a mismatch F0(0) J(ret), or MD when a
         byte is outside every load region (eql has written nothing: the original then faults as
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

CHUNK_J = 62  # positions in a chunk: 31 words (eql: bytes)
CHUNK_S = 31  # a byte loop's chunk, in bytes
NMAX = 0x3FFC  # a call copies fewer bytes than this
FRAME_RESERVE = 34
TAG_OP, TAG_MP, TAG_MC, TAG_MH = 0, 1, 2, 3
OP_MD, OP_MR, OP_MF, OP_MK, OP_MN = 0, 1, 2, 3, 4
K256 = [int(x, 16) for x in """428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 d807aa98 12835b01
243185be 550c7dc3 72be5d74 80deb1fe 9bdc06a7 c19bf174 e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc
76f988da 983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967 27b70a85 2e1b2138 4d2c6dfc 53380d13
650a7354 766a0abb 81c2c92e 92722c85 a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070 19a4c116
1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3 748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb
bef9a3f7 c67178f2""".split()]
SHA_LOAD, SHA_ROUND, SHA_FINAL, SHA_END = 3, 11, 75, 83  # the steps
SHA_SCALE = 16  # the step counter's output is 2^16 * 2^-step, so its thermometer's weights fit bf16
A0, A1, A2, A3, A4, A6, S3, S4, S5 = 10, 11, 12, 13, 14, 16, 19, 20, 21
A7, S0, S6, S7, S8, S10 = 17, 8, 22, 23, 24, 26
JESC_Q, JESC_B = 0x22, 0x5C  # '"' and '\\' in s3 and s7: fx's only order at this head (M)
# a byte loop's unit: the registers read at the trigger and in the loop (rs1 = base, rs2, rd = end),
# start = the first index comes from rs2 + 1, last = the loop's count is in rd, so it stops one byte
# earlier; the commits (register, +1 base + k, -1 base - k, 0 k)
SCAN = {"tok": dict(first=(S5, 12, A6), loop=(S5, 12, A6), start=False, commits=((12, 1), (S5, 1), (A6, -1))),
        "chr": dict(first=(A1, A0, A2), loop=(A1, A0, A2), start=False, commits=((A1, 1), (A2, -1))),
        "ascii": dict(first=(A0, S3, S3), loop=(S4, A0, S3), start=True, commits=((A0, 0),)),
        "jscan": dict(first=(A2, A0, A4), loop=(A0, A3, A4), start=True, commits=((A3, 0),)),
        # (first: the two compares' registers; a3's commit is the last byte's path, sg = 2; s11, a1, a2
        # are dead at the head)
        "jesc": dict(first=(S3, S7, None), loop=(S5, S8, None), start=True, commits=((S8, 0), (A3, 2)))}


@dataclass(frozen=True)
class Unit:
    """kind: move (memmove: d = a0, s = a1, n = a2), cpy (memcpy: also declines s < d < s + n),
    set (memset: d = a0, c = a1, n = a2), bcopy (a byte loop: d = a2, s = s11, end = a1),
    eql (a = a1, b = a3, n = a2, a0 = 1 on entry). pc: the trigger; ret: where the unit exits (a
    `ret` for function units), in the same 64K-word window. arg: a constant of the unit (maskp: the
    prefix table's address)."""

    kind: str
    pc: int
    ret: int
    arg: int = 0


def m_tok(tag: int, v: int = 0) -> int:
    return S.tok("M", tag << 12 | v)


def tok_decline() -> int:
    return m_tok(TAG_OP, OP_MD)


def group(kind: str) -> int:
    """MP's group bits: a role needs the unit's class one stage after the token."""
    return {"bcopy": 5, "eql": 2, "sha": 8, "jesc": 4}.get(kind, 1 if kind in COPY_KINDS else 0)


def mp_tok(x, f, dr, unit, kind) -> int:
    return m_tok(TAG_MP, x | f << 1 | dr << 2 | unit << 3 | group(kind) << 7)


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
    toks = [mp_tok(0, f, dr, uid(m, u), u.kind)]
    if u.kind == "bcopy":  # (n is not in a register: it goes down in a token)
        toks.append(S.tok("CW", n))
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
    toks = [mp_tok(0, 0, 0, uid(m, u), u.kind)]
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
        toks.append(S.tok("EB", byte(ea)))
        if not m.cfg.in_load(eb):
            m.skip_unit = True
            return toks + [tok_decline()]
        if byte(ea) != byte(eb):
            m.n += 1
            toks.append(S.tok("F0", 0))
            m.x[10] = 0
            break
        toks.append(S.tok("EC1"))
        j += 2
    m.steps += 1
    return toks + m.jump_tokens(u.ret)


def rr32(v: int, n: int) -> int:
    return (v >> n | v << (32 - n)) & 0xFFFFFFFF


def sha_record(m, u: Unit) -> list | None:
    """One block's compression: the unit's stream (module doc). A check fails: MD, nothing written."""
    x = m.x
    inst, st, blk = x[10], x[11], x[12]
    r = (st - blk) & 0xFFFFFFFF
    if st & 3 or blk & 3 or not window_ok(m, st) or not window_ok(m, blk) or not frames_ok(m) or (
            r >> 6 == 0 or r >> 5 == (1 << 27) - 1):
        return None
    toks = [mp_tok(0, 0, 0, uid(m, u), u.kind)]
    if inst & 3 or not m.cfg.in_load(inst + 8):
        m.skip_unit = True
        return toks + [tok_decline()]
    wsp = m.ram_word(inst + 8)
    if wsp >> 22 or wsp & 3 or wsp >> 8 == 0:
        m.skip_unit = True
        return toks + [tok_decline()]
    w0 = (wsp - 0x100) & 0xFFFFFFFF
    toks += [S.tok("CW", w0 & 0xFFFF), S.tok("CW", w0 >> 16), m_tok(TAG_OP, OP_MN)]
    for a in (st, blk):
        r = (a - w0) & 0xFFFFFFFF
        if r >> 8 == 0 or r >> 6 == (1 << 26) - 1:
            m.skip_unit = True
            return toks + [tok_decline()]
        toks.append(m_tok(TAG_OP, OP_MN))
    h = [m.ram_word(st + 4 * k) for k in range(8)]
    for v in h:
        toks += [S.tok("CW", v & 0xFFFF), S.tok("CW", v >> 16), m_tok(TAG_OP, OP_MN)]
    W = []
    a, b, c, d, e, f, g, hh = h
    for i in range(64):
        if i < 16:
            w = int.from_bytes(m.ram_word(blk + 4 * i).to_bytes(4, "little"), "big")
        else:
            s0 = rr32(W[i - 15], 7) ^ rr32(W[i - 15], 18) ^ W[i - 15] >> 3
            s1 = rr32(W[i - 2], 17) ^ rr32(W[i - 2], 19) ^ W[i - 2] >> 10
            w = (W[i - 16] + s0 + W[i - 7] + s1) & 0xFFFFFFFF
        W.append(w)
        dst = w0 + 4 * i
        toks += m.alloc(dst)
        toks += [mh_tok(m.pt[dst >> S.PAGE_BITS] & 1), S.tok("CW", w & 0xFFFF), S.tok("CW", w >> 16)]
        m.wr[dst >> 2] = w
        t1 = (hh + (rr32(e, 6) ^ rr32(e, 11) ^ rr32(e, 25)) + ((e & f) ^ (~e & g & 0xFFFFFFFF)) + K256[i] + w) & 0xFFFFFFFF
        t2 = ((rr32(a, 2) ^ rr32(a, 13) ^ rr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & 0xFFFFFFFF
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF
        toks += [S.tok("CW", a & 0xFFFF), S.tok("CW", a >> 16), S.tok("CW", e & 0xFFFF), S.tok("CW", e >> 16),
                 m_tok(TAG_OP, OP_MN)]
    for k, v in enumerate((a, b, c, d, e, f, g, hh)):
        n = (h[k] + v) & 0xFFFFFFFF
        dst = st + 4 * k
        toks += m.alloc(dst)
        toks += [mh_tok(m.pt[dst >> S.PAGE_BITS] & 1), S.tok("CW", n & 0xFFFF), S.tok("CW", n >> 16),
                 m_tok(TAG_OP, OP_MN)]
        m.wr[dst >> 2] = n
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
    if kind == "jesc":  # (bytes >= 0x80 are plain here)
        return b < 0x20 or b in (0x22, 0x5C)
    return not (0x20 <= b < 0x80) or b in (0x22, 0x5C)


def jesc_class(b: int) -> int:
    """The path a plain byte takes, s7 = '\\': 0 no write to a3, 1 a3 = 32, 2 a3 = 56."""
    return 1 if b in (0x20, 0x21) else 2 if 0x23 <= b < 0x5C else 0


def scan_fires(m, u: Unit) -> bool:
    x, t = m.x, SCAN[u.kind]
    k = u.kind
    if k == "jesc":
        return x[S3] == JESC_Q and x[S7] == JESC_B
    r1, r2, r3 = (x[i] for i in t["first"])
    if r3 >> 16:
        return False
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
    base, sc = x[t["loop"][0]], x[t["loop"][1]]
    end = x[t["loop"][2]] if t["loop"][2] is not None else 0
    toks = [mp_tok(0, 0, 0, uid(m, u), u.kind)]
    idx = 0
    if kind == "jesc":  # the bound, from its stack slot
        ea = (x[S0] - 60) & 0xFFFFFFFF
        if ea & 3 or not m.cfg.in_load(ea) or m.ram_word(ea) >> 16:
            m.skip_unit = True
            return toks + [tok_decline()]
        end = m.ram_word(ea)
        toks.append(S.tok("CW", end))
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
    path = 0  # jesc: the last plain byte's path that wrote a3
    while True:
        ea = (base + idx) & 0xFFFFFFFF
        bad = not m.cfg.in_load(ea)
        b = 0 if bad else m.ram_word(ea & ~3) >> 8 * (ea & 3) & 0xFF
        stop = idx == stop_at or idx == 0xFFFF or stops(kind, b, sc)
        cl = jesc_class(b) if kind == "jesc" else 0
        stop = stop or (cl != 0 and j >= CHUNK_S)  # (a chunk's end carries no class)
        if bad or (stop and first):
            m.skip_unit = True
            return toks + [tok_decline()]
        if stop:
            break
        idx += 1
        first = False
        if j >= CHUNK_S:
            toks += [S.tok("EC1"), S.tok("CW", idx), m_tok(TAG_OP, OP_MR)]
            j = 0
            continue
        if cl:
            path = cl
            toks.append(S.tok("EB", cl - 1))
        else:
            toks.append(S.tok("SC"))
        j += 1
    k = idx if not t["start"] else idx - 1
    toks += [S.tok("EC0"), S.tok("CW", k), m_tok(TAG_OP, OP_MK)]
    for r, sg in t["commits"]:
        if sg == 2:
            v = {0: x[r], 1: 32, 2: 56}[path]
        else:
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


def is_keychar(b: int) -> bool:
    return b == 0x5F or 0x30 <= b <= 0x39 or 0x41 <= (b & 0xDF) <= 0x5A


def maskp_record(m, u: Unit) -> list | None:
    """findSecretSpan's loop over the 12 secret_prefixes at one position, when every prefix is
    skipped: s6 == 0 and the byte before the position is an assignment-key character. Each of the 12
    then does the same thing (two table loads, into dead registers), so nothing is compared and no
    live register changes: the loop's exit, with the table only checked to be the table."""
    x = m.x
    if x[S5] != u.arg or x[S4] != 0x5F or x[A7] != 10:
        return None
    toks = [mp_tok(0, 0, 0, uid(m, u), u.kind)]
    ea = (x[S10] - 1) & 0xFFFFFFFF
    if x[S6] or not m.cfg.in_load(ea) or not is_keychar(m.ram_word(ea & ~3) >> 8 * (ea & 3) & 0xFF):
        m.skip_unit = True
        return toks + [tok_decline()]
    m.steps += 1
    return toks + m.jump_tokens(u.ret)


def record(m, u: Unit) -> list | None:
    """The unit's whole record, committed; None when it declines (nothing changed)."""
    x = m.x
    if u.kind == "maskp":
        return maskp_record(m, u)
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
        if a == b or n == 0 or n >= NMAX or not frames_ok(m):  # (the original pushes a frame)
            return None
        return eql_record(m, u, a, b, n)
    if u.kind == "sha":
        return sha_record(m, u)
    return scan_record(m, u)


# row 24's image (sha256 282e53d4...): name -> (kind, trigger, its instruction word, ret)
FX_R24 = {"memmove": ("move", 0x208f918), "memcpy": ("cpy", 0x208f73c), "memset": ("set", 0x208fa50),
          "eql": ("eql", 0x26733e4), "bcopy": ("bcopy", 0x21e3a98, 0x21e3aac),
          "tok": ("tok", 0x22bf6c0, 0x22bf6c0), "jscan": ("jscan", 0x264c040, 0x264c040),
          "ascii": ("ascii", 0x2649914, 0x2649914), "chr1": ("chr", 0x20cf5e8, 0x20cf5e8),
          "chr2": ("chr", 0x20cf610, 0x20cf610), "sha": ("sha", 0x25b5d70, None),
          "jesc": ("jesc", 0x264A154, 0x264A154), "maskp": ("maskp", 0x24D8958, 0x24D8A24, 0x1262A0)}


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
    names = [n for n in FX_R24 if n != "sha" or "all+" in names] if "all" in names or "all+" in names else list(names)
    out = []
    for n in names:
        e = FX_R24[n]
        r = e[2] if len(e) > 2 else ret
        if e[0] == "sha":  # any ret of its window: the first after the trigger
            r = next(a for a in range(e[1], e[1] + 1 << 19, 4) if word(a) == 0x00008067)
        if e[0] == "eql":
            ins = ref.decode(word(e[1]))
            assert ins.op == "BEQ" and (ins.rs1, ins.rs2) == (11, 13), "not row 24's image"
            r = e[1] + ins.imm
            assert word(r) == 0x00008067
        out.append(Unit(e[0], e[1], r, e[3] if len(e) > 3 else 0))
    return tuple(out)


COPY_KINDS = {"move", "cpy", "set", "bcopy"}
PORTS = {"move": (10, 11, 12), "cpy": (10, 11, 12), "set": (10, 11, 12), "bcopy": (12, 27, 11),
         "eql": (11, 13, 12)}  # the registers a unit reads: rs1, rs2, rd ports


class Hot:
    """The circuit (see the module doc). A unit's address goes through the core's own adder: the
    ports read the base (rs1) and the ROM's immediate is off, so EA = A + off; the trigger position
    uses it as d - s. All of this runs every step whether a unit is active or not, so a unit costs
    its MLP units, and an MLP or head that must wait for another one costs a phase of Executor v4:
    nothing here may stand in the chain from the registers to the exit."""

    def __init__(self, c, cfg) -> None:
        self.c, self.cfg = c, cfg
        self.units = cfg.units
        self.kinds = {u.kind for u in cfg.units}
        jesc = "jesc" in self.kinds
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

        def head(name, vhs):
            out = thead(c, name, {0: one}, {0: one}, vhs)
            self.tables.append(name)
            c.tables[name] = c.nodes[-1]
            return out

        self.head = head
        self.tMP = f("m.mp", tag_is(TAG_MP), full=True)
        self.tMC = f("m.mc", tag_is(TAG_MC), full=True)
        self.tMH = f("m.mh", tag_is(TAG_MH), full=True)
        self.tOP = f("m.op", tag_is(TAG_OP), full=True)
        mp = pay["M"]
        b0, b1 = mp[0], mp[1]
        self.tMD = c.and_([self.tOP, 1 - b0, 1 - b1, 1 - mp[2]], "m.md")
        self.tMN = f("m.mn", lambda t: int(t < S.VOCAB2 and S.family(t)[0] == "M" and (S.family(t)[1] >> 12, S.family(t)[1] & 7) == (TAG_OP, OP_MN)),
                     full=True)
        self.tMR = c.and_([self.tOP, b0, 1 - b1], "m.mr")
        self.tMF = c.and_([self.tOP, 1 - b0, b1], "m.mf")
        self.tMK = c.and_([self.tOP, b0, b1], "m.mk")
        self.tpre = c.and_([self.tMH, b1], "m.pre")
        # the unit's flags, for the whole record
        (hl,) = head("h.l", [{"data": {2 + i: e for i, e in enumerate([self.tMP] + mp[1:11])}, "ref": one,
                              "beta": self.tMP + fam["J"] + self.tMD}])
        self.inm, self.hf, self.hdir, self.hu = hl[0], hl[1], hl[2], hl[3:7]
        self.gcopy, self.geq, self.gbc, self.gsh = hl[7], hl[8], hl[9], hl[10]
        if "sha" in self.kinds:  # the step, as a thermometer: everything else of it is linear in it
            xs = c.counter("h.ri", self.tMP, self.tMN, SHA_SCALE)  # 2^16 * 2^-step: see core.thermo
            self.tables.append("h.ri")
            self.st = c.thermo(xs, self.tMP, SHA_END + 1, "h.ri", SHA_SCALE)
        ((self.hx,),) = head("h.x", [{"data": {2: mp[0]}, "ref": one, "beta": self.tMH}])
        self.scans = [u for u, un in enumerate(cfg.units) if un.kind in SCAN]
        self.iscan = c.and_([self.inm, 1 - self.gcopy, 1 - self.geq, 1 - self.gsh], "h.iscan")
        self.tscw = c.and_([self.inm, 1 - self.gcopy, 1 - self.geq, 1 - self.gsh, fam["CW"]], "h.tscw") if self.scans else Expr()
        cw_ = pay["CW"]
        (self.hc,) = head("h.c", [{"data": {2 + i: (mp[i] if i < 12 else Expr()) + (cw_[i] if self.scans else Expr())
                                            for i in range(16)}, "ref": one,
                                   "beta": self.tMC + fam["J"] + fam["R"] + fam["JF"] + fam["RF"] + self.tscw}])
        self.pMP = self.pEC0 = Expr()
        if self.scans:
            lag = (0, 1, 0, 0)  # what the token before was (a head written at every position)
            cols = {2: Chan(self.tMP, lag), 3: Chan(fam["EC0"], lag)} | ({4: Chan(self.tMP, (0, 0, 1, 0))} if jesc else {})
            (pp,) = head("h.p", [{"data": cols, "ref": one}])
            self.pMP, self.pEC0 = pp[0], pp[1]
            self.pMP2 = pp[2] if jesc else Expr()
            # a unit's commits so far, since MK
            ((self.incm,),) = head("h.kf", [{"data": {2: self.tMK}, "ref": one, "beta": self.tMK + self.tMP + fam["J"]}])
            xk = c.counter("h.k", self.tMK, c.final)
            self.kt = c.thermo(xk, self.tMK, 3, "h.k")
            self.tables.append("h.k")
        self.hn = [Expr()] * 16
        self.pH = Expr()
        if jesc:  # the last class token (EB): has one come, and which
            ((self.jseen, self.jcls),) = head("h.cl", [{"data": {2: fam["EB"], 3: pay["EB"][0]}, "ref": one,
                                                        "beta": self.tMP + fam["EB"]}])
        if self.kinds & {"bcopy", "jesc"}:  # a length read once: the byte loop's, the escape scan's
            lag = (0, 1, 0, 0)
            ((self.pMPb,),) = head("h.pb", [{"data": {2: Chan(self.tMP, lag)}, "ref": one}])
            self.pH = 1 - self.pMPb
            (self.hn,) = head("h.n", [{"data": {2 + i: cw_[i] for i in range(16)}, "ref": one, "gate": self.gbc,
                                       "beta": c.and_([self.inm, fam["CW"], self.pMPb], "h.bn")}])
        if "eql" in self.kinds:
            (self.he,) = head("h.e", [{"data": {2 + i: pay["EB"][i] for i in range(8)}, "ref": one, "beta": fam["EB"]}])

    def setup(self, w, jt, j0, at_max, isJA) -> None:
        from chipformer.rv import core as C

        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        self.jt, self.j0 = jt, j0
        jmax = len(jt)

        def jeq(v):
            return (one if v == 0 else jt[v - 1]) - (jt[v] if v < jmax else 0)

        ij = sum(jt, Expr())  # j
        iw = sum(jt[1::2], Expr())  # j >> 1
        self.jodd = sum((jeq(v) for v in range(1, jmax + 1, 2)), Expr())
        self.first, self.hun, self.act = [], [], []
        for u, un in enumerate(self.units):
            wi = (un.pc - cfg.rom_base) >> 2
            assert wi >> 16 == (un.ret - cfg.rom_base) >> 18, "a unit's exit leaves its 64K-word window"
            fi = c.and_(C.lits_of(w, wi) + [j0, 1 - at_max, 1 - self.inm, 1 - self.tMD], f"h.first{u}")
            hu = c.and_([self.inm] + C.lits_of(self.hu, u), f"h.u{u}")
            self.first.append(fi)
            self.hun.append(hu)
            self.act.append(fi + hu)
        self.mact = sum(self.act, Expr())
        self.first_all = sum(self.first, Expr())
        self.sub_r = self.first_all
        # the roles
        cw = c.fam["CW"]
        COPY = ("move", "cpy", "set", "bcopy")
        imv, ieq = self.gcopy, self.geq
        self.imv, self.ieq = imv, ieq
        self.kact = {k: sum((a for a, un in zip(self.act, self.units) if un.kind == k), Expr())
                     for k in ("set", "bcopy", "eql")}
        tmp, tmh, tmc, tmr = self.tMP, self.tMH, self.tMC, self.tMR
        tpa = c.fam["PA"]
        sc = c.fam["SC"]
        ec0, ec1 = c.fam["EC0"], c.fam["EC1"]
        self.regk = {}
        self.re1 = self.re2 = self.rx = zero
        if "eql" in self.kinds:
            self.re2 = c.and_([ieq, c.fam["EB"]], "h.re2")
            self.re1 = c.and_([ieq, ec1 + tmp + tmc], "h.re1")
            self.rx = c.and_([ieq, c.fam["F0"]], "h.rx")
            self.regk = {10: self.re2}
        self.rN = zero
        if "bcopy" in self.kinds:
            self.rN = c.and_([self.gbc, self.gcopy, tmp], "h.rN")  # MP stands here: CW(n) comes next
            self.sub_r = self.sub_r + self.rN
            jn = c.and_([self.gbc, self.gcopy, cw, 1 - self.pH], "h.jn")  # (the token after MP, once: pH is set by it)
        self.rC = c.and_([imv, tmp, self.hdir], "h.rC")
        self.rR = c.and_([imv, self.tpre], "h.rR")
        self.rH = (c.and_([imv, tmc + tpa + tmp - self.rC - self.rN], "h.rH")
                   + (jn if "bcopy" in self.kinds else zero))
        cwb = (cw - jn) if "bcopy" in self.kinds else cw
        self.rb = c.and_([imv, cwb, self.jodd], "h.rb")
        self.rhi = c.and_([imv, cwb, 1 - self.jodd], "h.rhi")
        self.ra = self.rhi + c.and_([imv, tmr + tmh - self.tpre], "h.ra1")
        self.ead = self.rb + self.rH + self.re1
        self.eas = self.ra + self.rR + self.re2
        self.ea = self.ead + self.eas
        self.store = self.rb + self.rH
        self.cv = zero  # a scan unit's commit (of the value A +- k)
        self.cvk = zero  # those that add k
        self.sub_k = zero
        self.rbyte = self.rse = zero
        js = [u for u in self.scans if self.units[u].kind == "jscan"]
        if self.scans:
            iscan, incm = self.iscan, self.incm
            kt = self.kt
            mq = [1 - kt[0], kt[0] - kt[1], kt[1] - kt[2], kt[2]]
            self.rbyte = c.and_([iscan, 1 - incm, self.tMF + self.tMR + sc + (c.fam["EB"] if "jesc" in self.kinds else zero)],
                                "h.rbyte")
            self.rse = c.and_([iscan, 1 - incm, ec0 + ec1], "h.rse")  # the stop's or the chunk's index
            self.rcm, self.rend = [], []
            for u in self.scans:
                cm = SCAN[self.units[u].kind]["commits"]
                rc = [c.and_([self.hun[u], incm, 1 - tpa, 1 - tmh, 1 - cw, mq[i]], f"h.rc{u}.{i}")
                      for i in range(len(cm))]
                self.rcm.append(rc)
                self.rend.append(c.and_([self.hun[u], incm, mq[len(cm)]], f"h.rend{u}"))
                for i, (r, sg) in enumerate(cm):
                    self.regk[r] = self.regk.get(r, zero) + rc[i]
                    self.cv = self.cv + rc[i]
                    if sg != 2:
                        self.cvk = self.cvk + rc[i]
                    if sg < 0:
                        self.sub_k = self.sub_k + rc[i]
            self.ea = self.ea + self.rbyte + self.rse + self.cv
        self.jr1 = self.mr = zero
        je = [u for u, un in enumerate(self.units) if un.kind == "jesc"]
        if je:  # the bound: MP's position reads s0 - 60
            self.jr1 = c.and_([self.hun[je[0]], tmp], "h.jr1")
            self.ea = self.ea + self.jr1
            self.sub_k = self.sub_k + self.jr1
        mk = [u for u, un in enumerate(self.units) if un.kind == "maskp"]
        if mk:  # MP's position reads s10 - 1
            self.mr = c.and_([self.hun[mk[0]], tmp], "h.mr")
            self.ea = self.ea + self.mr
            self.sub_k = self.sub_k + self.mr
        self.jsH = self.jsa = self.jsb = self.jsw = zero
        if js:  # the cursor's store: H, then its word
            je = self.rend[self.scans.index(js[0])]
            self.jsH = c.and_([je, 1 - tmh, 1 - cw], "h.jsH")
            self.jsa = c.and_([je, tmh], "h.jsa")
            self.jsb = c.and_([je, cw, self.jodd], "h.jsb")
            self.jsw = c.and_([je, cw, 1 - self.jodd], "h.jsw")
            self.ea = self.ea + self.jsH + self.jsb
            self.store = self.store + self.jsH + self.jsb
        # the ports: rs1 is the base of EA, rs2 the other buffer, rd the length
        self.ports = {"rs1": {}, "rs2": {}, "rd": {}}
        self.swap = zero

        def add(pn, r, e):
            self.ports[pn][r] = self.ports[pn].get(r, Expr()) + e

        if "sha" in self.kinds:
            self.setup_sha(jt, add)
        for u, un in enumerate(self.units):
            if un.kind == "sha":
                add("rs1", 11, self.first[u])  # st - blk
                add("rs2", 12, self.first[u])
                continue
            if un.kind == "maskp":
                add("rs1", S5, self.first[u])
                add("rs2", S4, self.first[u])
                add("rd", A7, self.first[u])
                add("rs1", S10, self.mr)
                add("rs2", S6, self.mr)
                continue
            if un.kind in SCAN:
                t = SCAN[un.kind]
                for pn, a_, b_ in zip(("rs1", "rs2", "rd"), t["first"], t["loop"]):
                    if a_ is not None:
                        add(pn, a_, self.first[u])
                    if b_ is not None:
                        off_ = self.jr1 if un.kind == "jesc" and pn == "rs1" else zero
                        add(pn, b_, c.and_([self.hun[u], 1 - self.incm], f"h.pl{u}{pn}") - off_)
                x_ = self.scans.index(u)
                for i, (r, sg) in enumerate(t["commits"]):
                    if sg == 2:  # the old value, unless a class token came
                        add("rs1", r, c.and_([self.rcm[x_][i], 1 - self.jseen], "h.ja3"))
                    elif sg:
                        add("rs1", r, self.rcm[x_][i])
                if un.kind == "jesc":
                    add("rs1", S0, self.jr1)
                if un.kind == "jscan":
                    add("rs1", A2, self.jsH + self.jsb)
                continue
            d_, s_, n_ = PORTS[un.kind]
            sw = zero if un.kind == "set" else c.and_([self.hun[u], self.eas], f"h.swap{u}")
            self.swap = self.swap + sw
            if un.kind == "bcopy":  # trigger and MP's position: end - s by the core's adder, d in rd
                rn = c.and_([self.hun[u], tmp], f"h.rn{u}")
                add("rs1", n_, self.first[u] + rn)
                add("rs2", s_, self.first[u] + rn)
                add("rd", d_, self.first[u] + rn)
                add("rs1", d_, self.hun[u] - sw - rn)
                add("rs1", s_, sw)
                add("rs2", s_, self.hun[u] - sw - rn)
                add("rs2", d_, sw)
                continue
            add("rs1", d_, self.act[u] - sw)
            add("rs1", s_, sw)
            add("rs2", s_, self.act[u] - sw)
            add("rs2", d_, sw)
            add("rd", n_, self.act[u])
        ww = self.jsw + (self.sw if "sha" in self.kinds else zero)
        self.wr = [c.and_([self.rhi + ww, 1 - self.hx], "h.wr0"), c.and_([self.rhi + ww, self.hx], "h.wr1")]
        # the offset, in one MLP stage: c + j (or j >> 1), carry or borrow into the high bits. The
        # copy's source word is one off (f); the true word index ci is for the top, not the adder
        hc = self.hc
        cl = from_bits(hc[:5])
        dirn, hf = self.hdir, self.hf
        wf = self.ra + self.rR + self.rb + self.rH  # (eas's and ead's word roles)
        sel_p = c.node(c.t_and([self.ra, 1 - dirn, hf]) + c.t_and([self.rR, dirn, 1 - hf]), "h.selp")
        sel_m = c.and_([self.ra, dirn, hf], "h.selm") if "move" in self.kinds else zero
        modes = []
        if COPY_KINDS & self.kinds:
            modes.append((cl + iw + sel_p, 0, 63, [wf, 1 - dirn], 2, False))
            if "move" in self.kinds:
                modes.append((cl - iw - sel_m + sel_p, -32, 32, [wf, dirn], 2, True))
        if "eql" in self.kinds:
            modes.append((cl + iw, 0, 63, [self.re1 + self.re2], 0, False))
        if self.scans:
            modes.append((cl + ij, 0, 31 + jmax, [self.rbyte + self.rse], 0, False))
        oterm = [[] for _ in range(18)]
        for z, lo, hi, lits, sh, back in modes:
            for b in range(5):
                f_ = lambda v, b=b: ((v & 31) >> b) & 1
                for v in range(lo + 1, hi + 1):
                    if f_(v) != f_(v - 1):
                        oterm[b + sh] += c.t_ge(z, v, lo, hi, f_(v) - f_(v - 1), lits=lits)
                if f_(lo):
                    oterm[b + sh] += c.t_and(lits)
            for b in range(5, 16):
                lows = [(1 - x if back else x) for x in hc[5:b]]
                carry = lambda co, ex: (c.t_lt(z, 0, lo, hi, co, lits=lits + lows + ex) if back
                                        else c.t_ge(z, 32, lo, hi, co, lits=lits + lows + ex))
                oterm[b + sh] += c.t_and(lits + [hc[b]]) + carry(1, [1 - hc[b]]) + carry(-1, [hc[b]])
        for b in range(16):
            if self.cvk.terms:
                oterm[b] += c.t_and([self.cvk, hc[b]])
        self.off = [c.node(t, f"h.off{b}") if t else zero for b, t in enumerate(oterm)] + [zero] * 14
        if js:  # the cursor's offset
            self.off[2] = self.off[2] + self.jsH + self.jsb
            self.off[5] = self.off[5] + self.jsH + self.jsb
        if "jesc" in self.kinds:  # 60; a3's commit: 32 or 56
            for b in range(2, 6):
                self.off[b] = self.off[b] + self.jr1
            rc3 = next(self.rcm[x_][1] for x_, u in enumerate(self.scans) if self.units[u].kind == "jesc")
            sn = c.and_([rc3, self.jseen], "h.j32")
            cl = c.and_([rc3, self.jseen, self.jcls], "h.j56")
            self.off[5] = self.off[5] + sn
            self.off[3] = self.off[3] + cl
            self.off[4] = self.off[4] + cl
        if "maskp" in self.kinds:
            self.off[0] = self.off[0] + self.mr
        if "sha" in self.kinds:
            self.off = [o + e for o, e in zip(self.off, self.soff)]
        # the word index, for the top (its depth does not matter there)
        ib = [c.node(t, f"h.i{b}") for b, t in enumerate(c.fn_bits(iw, 0, 40, lambda v: v, 6, "h.i"))]
        cb = hc
        ci_f = C.add_nibbles(c, cb, (ib + [zero] * 16)[:16], 4, [1 - dirn], "h.cif")
        if "move" in self.kinds:
            nib_ = [1 - b for b in ib] + [one] * 10
            sums = [[(from_bits(cb[4 * k:4 * k + 4]) + from_bits(nib_[4 * k:4 * k + 4]) + (1 if k == 0 else 0),
                      [dirn])] for k in range(4)]
            g, p_, d = C.stage1_digits(c, sums, "h.cib", 31)
            ci_b = C.adder_stage2(c, g, p_, d, 4, "h.cib")[0]
            self.ci = [ci_f[b] + ci_b[b] for b in range(14)] + ci_f[14:]  # (bit 13: negative, dir = 1)
        else:
            self.ci = ci_f
        ci = self.ci
        self.cp1 = [self.incr(ci, b, f"h.cp{b}") for b in range(14)]
        # the parity of the source word (which of the two stash heads it goes to)
        self.par = c.node(c.t_and([ci[0], 1 - hf]) + c.t_and([1 - ci[0], hf]), "h.par")

    xa = [Expr()] * 32  # a unit's own base (added to rs1 in the core's adder)

    def setup_sha(self, jt, add) -> None:
        """SHA-256: the rounds below the RAM. Rings of 8 (a.., e..) and of 16 (W) in L2's attention,
        written from CW tokens, read at rows linear in the step's thermometer."""
        from chipformer.rv import core as C

        c = self.c
        one, zero = c.one, Expr()
        fam, pay = c.fam, c.pay
        st, gsh, jmax = self.st, self.gsh, len(jt)

        def seq(k):
            return (one if k == 0 else st[k - 1]) - st[k]

        def rng(a, b):  # step in [a, b)
            return (one if a == 0 else st[a - 1]) - (st[b - 1] if b else 0)

        def jeq(v):
            return (one if v == 0 else jt[v - 1]) - (jt[v] if v < jmax else 0)

        cw = fam["CW"]
        tn, tp, th, tpa = self.tMN, self.tMP, self.tMH, fam["PA"]
        s_load, s_round, s_final = rng(SHA_LOAD, SHA_ROUND), rng(SHA_ROUND, SHA_FINAL), rng(SHA_FINAL, SHA_END)
        s_msg, s_sch = rng(SHA_ROUND, SHA_ROUND + 16), rng(SHA_ROUND + 16, SHA_FINAL)
        wstep = s_round + s_final
        cwj = lambda v, nm: c.and_([gsh, cw, jeq(v)], nm)
        self.sH = c.and_([gsh, wstep, tn + tpa], "s.H")
        self.sc1 = c.and_([gsh, seq(1), tn], "s.c1")  # the overlap checks: r = st / blk - W
        self.sc2 = c.and_([gsh, seq(2), tn], "s.c2")
        self.sa = c.and_([gsh, s_load + seq(0), tn + tp], "s.a0") + c.and_([gsh, th], "s.a1")
        j1, j2, j3, j4, j5, j6 = (cwj(v, f"s.j{v}") for v in range(1, 7))
        self.sb, self.j2, self.j3, self.j4, self.j5, self.j6 = j1, j2, j3, j4, j5, j6
        self.s_round, self.s_final, self.s_msg, self.s_sch, self.s_load, self.wstep = (
            s_round, s_final, s_msg, s_sch, s_load, wstep)
        self.seq = seq
        self.sw = c.and_([j2, wstep], "s.w")  # the word's store
        self.sdst = self.sH + c.and_([j1, wstep], "s.dst")
        self.store = self.store + self.sdst
        sea = self.sa + self.sdst + self.sc1 + self.sc2
        self.ea = self.ea + sea
        self.sub_k = self.sub_k + self.sc1 + self.sc2
        # the wasm stack's frame: W = wsp - 0x100, word i at W + 4 i
        (hw,) = self.head("s.w", [{"data": {2 + i: Chan(pay["CW"][i % 16], (0, 1, 0, 0) if i < 16 else (1, 0, 0, 0))
                                            for i in range(22)}, "ref": one,
                                   "beta": gsh + seq(0) + cw + jeq(2) - 3}])
        i_ = [sum((seq(SHA_ROUND + i) for i in range(64) if i >> b & 1), zero) for b in range(6)]
        rnd_dst = c.and_([self.sdst, s_round], "s.rd")
        self.xa = [c.and_([rnd_dst, hw[b]], f"s.xa{b}") for b in range(22)] + [zero] * 10  # W, plus 4 i in off
        k_ = [sum((seq(SHA_LOAD + k) + seq(SHA_FINAL + k) for k in range(8) if k >> b & 1), zero) for b in range(3)]
        msg = [sum((seq(SHA_ROUND + i) for i in range(16) if i >> b & 1), zero) for b in range(4)]
        rd_a = c.and_([self.sa, s_msg], "s.ms")
        self.soff = ([zero, zero] + [c.and_([sea - self.sc1 - self.sc2 - rnd_dst - rd_a, k_[b]], f"s.ok{b}") for b in range(3)]
                     + [zero] * 27)
        self.soff[3] = self.soff[3] + c.and_([self.sa, seq(0)], "s.o8")  # 8(a0)
        for b in range(4):
            self.soff[2 + b] = self.soff[2 + b] + c.and_([rd_a, msg[b]], f"s.om{b}")
        chk = self.sc1 + self.sc2
        for b in range(22):
            self.soff[b] = self.soff[b] + c.and_([chk, hw[b]], f"s.ow{b}")
        for b in range(6):
            self.soff[2 + b] = self.soff[2 + b] + c.and_([rnd_dst, i_[b]], f"s.oi{b}")
        add("rs1", 10, c.and_([self.sa, seq(0)], "s.p0"))
        add("rs1", 11, self.sc1 + c.and_([self.sa + self.sdst, s_load + s_final], "s.p1"))
        add("rs1", 12, self.sc2 + rd_a)
        # the rings
        rows = lambda f_, n: {r: sum((seq(SHA_ROUND + i) for i in range(64) if f_(i) & (n - 1) == r), zero) for r in range(n)}
        lag = {i: (0, 1, 0, 0) if i < 16 else (1, 0, 0, 0) for i in range(32)}

        def word(i_col):
            return Chan(pay["CW"][i_col % 16], lag[i_col])

        ae_key = {r: Chan(e, (0, 1, 0, 0)) for r, e in rows(lambda i: i, 8).items()}
        for k in range(4):
            ae_key[7 - k] = Chan(ae_key[7 - k].expr + seq(SHA_LOAD + k) + seq(SHA_LOAD + 4 + k), (0, 1, 0, 0))
        # (linear write gates: these heads must read early)
        wgate = lambda v, rg: gsh + cw + jeq(v) + rg - 3
        ba = c.and_([gsh, cw, jeq(4), s_round], "s.ba") + c.and_([gsh, cw, jeq(2), rng(SHA_LOAD, SHA_LOAD + 4)], "s.ba2")
        be = c.and_([gsh, cw, jeq(6), s_round], "s.be") + c.and_([gsh, cw, jeq(2), rng(SHA_LOAD + 4, SHA_ROUND)], "s.be2")
        self.ra_ = []
        self.re_ = []
        for p_ in range(4):
            q = {r: e for r, e in rows(lambda i: i - 1 - p_, 8).items()}
            q[(63 - p_) & 7] = q[(63 - p_) & 7] + s_final
            ra_, re_ = thead(c, f"s.ae{p_}", {r: Chan(e.expr, e.taps) for r, e in ae_key.items()}, q,
                             [{"data": {2 + i: word(i) for i in range(32)}, "ref": one, "beta": ba},
                              {"data": {2 + i: word(i) for i in range(32)}, "ref": one, "beta": be}])
            self.ra_.append(ra_)
            self.re_.append(re_)
            self.tables.append(f"s.ae{p_}")
            c.tables[f"s.ae{p_}"] = c.nodes[-1]
        w_key = {r: Chan(e, (0, 1, 0, 0)) for r, e in rows(lambda i: i, 16).items()}
        for k in range(8):
            w_key[16 + k] = Chan(seq(SHA_LOAD + k), (0, 1, 0, 0))
        bw_ = wgate(2, s_round)
        bh = wgate(2, s_load)
        self.wp = []
        for p_, off_ in enumerate((0, 1, 9, 14)):
            q = rows(lambda i: i + off_, 16)
            vhs = [{"data": {2 + i: word(i) for i in range(32)}, "ref": one, "beta": bw_}]
            if p_ == 0:
                for k in range(8):
                    q[16 + k] = seq(SHA_FINAL + k)
                vhs.append({"data": {2 + i: word(i) for i in range(32)}, "ref": one, "beta": bh})
            out = thead(c, f"s.w{p_}", w_key, q, vhs)
            self.wp.append(out[0])
            if p_ == 0:
                self.h0 = out[1]
            self.tables.append(f"s.w{p_}")
            c.tables[f"s.w{p_}"] = c.nodes[-1]
        self.res = self.sha_res()  # (in the MLPs under the ROM's heads: free, and the residual is empty there)

    def csa(self, x, y, z, name):
        """Three 32-bit forms to two: the carry-save step."""
        c = self.c
        s_, cy = [], [Expr()]
        for i in range(32):
            t = x[i] + y[i] + z[i]
            s_.append(c.node(c.t_ge(t, 1, 0, 3) + c.t_ge(t, 2, 0, 3, -1) + c.t_ge(t, 3, 0, 3), f"{name}.s{i}"))
            if i < 31:
                cy.append(c.node(c.t_ge(t, 2, 0, 3), f"{name}.c{i}"))
        return s_, cy

    def sha_res(self) -> list:
        """The one 32-bit sum a position needs: W, a', e' or h + value."""
        from chipformer.rv import core as C

        c = self.c
        zero = Expr()
        a, b, cc, d = self.ra_
        e, f, g, h = self.re_
        w16, w15, w7, w2 = self.wp

        def rot(x, n):
            return [x[(i + n) % 32] for i in range(32)]

        def shr(x, n):
            return [x[i + n] if i + n < 32 else zero for i in range(32)]

        def xor3(p, q, r, nm):
            return [c.node(c.t_ge(p[i] + q[i] + r[i], 1, 0, 3) + c.t_ge(p[i] + q[i] + r[i], 2, 0, 3, -1)
                           + c.t_ge(p[i] + q[i] + r[i], 3, 0, 3), f"{nm}{i}") for i in range(32)]

        S0 = xor3(rot(a, 2), rot(a, 13), rot(a, 22), "s.S0")
        S1 = xor3(rot(e, 6), rot(e, 11), rot(e, 25), "s.S1")
        s0 = xor3(rot(w15, 7), rot(w15, 18), shr(w15, 3), "s.s0")
        s1 = xor3(rot(w2, 17), rot(w2, 19), shr(w2, 10), "s.s1")
        ch = [c.node(c.t_and([e[i], f[i]]) + c.t_and([1 - e[i], g[i]]), f"s.ch{i}") for i in range(32)]
        mj = [c.node(c.t_ge(a[i] + b[i] + cc[i], 2, 0, 3), f"s.mj{i}") for i in range(32)]
        kk = [sum((self.seq(SHA_ROUND + i) for i in range(64) if K256[i] >> b_ & 1), zero) for b_ in range(32)]
        x1, y1 = self.csa(h, S1, ch, "s.t1")
        x2, y2 = self.csa(x1, y1, kk, "s.t2")
        x3, y3 = self.csa(x2, y2, w16, "s.t3")  # W[i] is in w16's row after its tokens
        xe, ye = self.csa(x3, y3, d, "s.te")
        x4, y4 = self.csa(x3, y3, S0, "s.ta")
        xa, ya = self.csa(x4, y4, mj, "s.tb")
        x5, y5 = self.csa(w16, s0, w7, "s.w1")
        xw, yw = self.csa(x5, y5, s1, "s.w2")
        # the final's value: a..h at step k
        fin = [zero] * 32
        for k, r in enumerate((a, b, cc, d, e, f, g, h)):
            fin = [o + c.and_([self.seq(SHA_FINAL + k), r[i]], f"s.f{k}.{i}") for i, o in enumerate(fin)]
        sa = self.j2 + self.j3
        se = self.j4 + self.j5
        sw_ = c.and_([self.sa + self.sb, self.s_sch], "s.ssw")
        sf = c.and_([self.sa + self.sb, self.s_final], "s.ssf")
        sa, se = c.and_([sa, self.s_round], "s.ssa"), c.and_([se, self.s_round], "s.sse")
        X = [c.node(c.t_and([sa, xa[i]]) + c.t_and([se, xe[i]]) + c.t_and([sw_, xw[i]]) + c.t_and([sf, fin[i]]), f"s.X{i}")
             for i in range(32)]
        Y = [c.node(c.t_and([sa, ya[i]]) + c.t_and([se, ye[i]]) + c.t_and([sw_, yw[i]]) + c.t_and([sf, self.h0[i]]), f"s.Y{i}")
             for i in range(32)]
        return C.add_nibbles(c, X, Y, 8, [self.gsh], "s.add")

    def incr(self, x, b, name):
        c = self.c
        low = x[:b]
        return c.node(c.t_and([x[b]]) + c.t_and(low + [1 - x[b]]) + c.t_and(low + [x[b]], -1)
                      if b else c.t_const(1) + c.t_and([x[0]], -1), name)

    def _decr(self, x, b, name):
        c = self.c
        low = [1 - e for e in x[:b]]
        return c.node(c.t_and([x[b]]) + c.t_and(low + [1 - x[b]]) + c.t_and(low + [x[b]], -1)
                      if b else c.t_const(1) + c.t_and([x[0]], -1), name)

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

        def pay_const(tag, vbits):
            return list(vbits) + [zero] * (12 - len(vbits)) + [one if tag >> b & 1 else zero for b in range(2)]

        def ubits(u):
            g = group(self.units[u].kind)
            return [one if u >> b & 1 else zero for b in range(4)] + [one if g >> b & 1 else zero for b in range(4)]

        decline = pay_const(TAG_OP, [zero])  # OP_MD
        # the two buffers' low bits at any position: the ports swap on source reads
        swap = self.swap
        dv = [c.node(c.t_and([swap, B[i]]) + c.t_and([1 - swap, A[i]]), f"h.dv{i}") for i in range(2)]
        sv = [c.node(c.t_and([swap, A[i]]) + c.t_and([1 - swap, B[i]]), f"h.sv{i}") for i in range(2)]
        isset, isbc = self.kact["set"], self.kact["bcopy"]
        C0 = CR
        if kinds & {"bcopy", "jesc"}:  # n came down in a token (rd holds nothing there)
            CR = [e + (self.hn[i] if i < 16 else zero) for i, e in enumerate(CR)]
        n_ok = c.and_([1 - b for b in CR[14:32]] + [1 - c.and_(CR[2:14], "h.ntop")], "h.nok")
        n_nz = or_(CR[:14], "h.nnz")
        d_ne = 1 - eq
        copy = kinds & COPY_KINDS
        if copy:
            # n's arithmetic: U = (d & 3) + n + 3: K = U >> 2 words, e = U & 3 the last byte's lane
            sums = [[(nib(CR, 0) + from_bits(dv) + 3, [])]] + [[(nib(CR, k), [])] for k in range(1, 4)]
            g, p_, d = C.stage1_digits(c, sums, "h.u", 30)
            U = C.adder_stage2(c, g, p_, d, 4, "h.u")[0]
            Kb, e2 = U[2:16], U[0:2]
            eq_k = eqbits(ci[:14], Kb, "h.eqk")
            lastw = eqbits(self.cp1, Kb, "h.last")
            first0 = c.and_([1 - b for b in ci[:14]], "h.first0")
            rr = [c.node([t for sl in range(4) for dl in range(4) if (sl - dl) & 3 == rv
                          for t in c.t_and(lits_of(sv, sl) + lits_of(dv, dl) + [1 - isset])], f"h.r{rv}")
                  for rv in range(4)]
            # the byte mask of word ci (one MLP stage from first0 and lastw)
            dint, eint = from_bits(dv), from_bits(e2)
            mask = []
            for b in range(4):
                ge = c.node(c.t_ge(b - dint, 0, -3, 3), f"h.ge{b}")
                le = c.node(c.t_ge(eint - b, 0, -3, 3), f"h.le{b}")
                mask.append(c.node(c.t_const(1) + c.t_and([first0, 1 - ge], -1) + c.t_and([lastw, 1 - le], -1)
                                   + c.t_and([first0, 1 - ge, lastw, 1 - le]), f"h.mask{b}"))
            edge = c.node(c.t_and([1 - hdir, lastw] + [1 - c.and_(list(e2), "h.e3")])
                          + c.t_and([hdir, first0, or_(list(dv), "h.dz")]), "h.edge")
            done = c.node(c.t_and([1 - hdir, eq_k]) + c.t_and([hdir, ci[13]]), "h.done")
        # the conditions at the trigger: r = d - s (the core's adder)
        lg = cfg.rw[0][1]
        wok = lambda v, nm: c.and_([1 - b for b in v[lg:32]] + [1 - c.and_(v[14:lg], nm + "t")], nm)
        lim = (self.frames - FRAME_RESERVE) >> 5
        self.nb = r[:16]  # (at the position after MP a bcopy's ports are the trigger's again: below)
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
        if "bcopy" in kinds:
            # first: A = end, B = s, C0 = d, r = n: forward iff d < s or d >= end
            def below(x, y, name):  # x < y, unsigned
                gt_ = [c.node(c.t_lt(nib(x, k) - nib(y, k), 0, -15, 15), f"{name}g{k}") for k in range(8)]
                eq_ = [c.node(c.t_eq(nib(x, k) - nib(y, k), 0, -15, 15), f"{name}e{k}") for k in range(8)]
                t = []
                for k in range(8):
                    t += c.t_and(eq_[k + 1:] + [gt_[k]])
                return c.node(t, name)

            r_ok = c.and_([1 - b for b in r[14:32]] + [1 - c.and_(r[2:14], "h.rtop"), or_(r[:14], "h.rnz")], "h.rok")
            fire_bc = c.and_([wok(C0, "h.wd"), wok(B, "h.ws"), r_ok, fok, or_([below(C0, B, "h.ls"), 1 - below(C0, A, "h.le2")], "h.bf")],
                             "h.firebc")
            bcf = c.node(c.t_lt(from_bits(C0[:2]) - from_bits(B[:2]), 0, -3, 3), "h.bcf")  # (s & 3) > (d & 3)
        if copy:
            # the source words: two stash heads by parity, so W(k) and W(k + 1) survive a chunk
            # boundary; x1 is the word read last, x0 the one before; old: the header's word; pg: is
            # this word the last of its page, in the copy's direction (at b, EA is the word)
            par = self.par
            x0, x1 = c.zeros(32, "h.x0"), c.zeros(32, "h.x1")
            cols = {2 + i: Chan(mem[i], lag) for i in range(32)} | {34 + i: Chan(mem[i], lag) for i in range(32)}
            vhs = []
            for p in (0, 1):
                gate_new, gate_old = (par if p else 1 - par), (1 - par if p else par)
                v_ = {col: C.chan(e, C.A_BIT) for col, e in cols.items()}
                for rc in C.REF_COLS:
                    v_[rc] = C.chan(one, C.A_REF)
                outs_ = [(2 + i, C.OUT_W) for i in range(32)] + [(34 + i, C.OUT_W) for i in range(32)]
                z_ = {2 + i: C.K * gate_new for i in range(32)} | {34 + i: C.K * gate_old for i in range(32)}
                vhs.append({"v": v_, "z": z_, "outs": outs_, "into": x1 + x0,
                            "beta": c.and_([self.rb, gate_new], f"h.bp{p}") + c.and_([imv, self.tMR, gate_old], f"h.bm{p}")})
            c.table("h.s", {0: one}, {0: one}, vhs)
            self.tables.append("h.s")
            pe = c.node(c.t_and([1 - hdir] + EA[2:9]) + c.t_and([hdir] + [1 - x for x in EA[2:9]]), "h.pe")
            old, (pgedge,) = self.head("h.o", [{"data": {2 + i: Chan(mem[i], lag) for i in range(32)}, "ref": one,
                                                "beta": c.and_([imv, self.tMH], "h.bo")},
                                               {"data": {2: Chan(pe, lag)}, "ref": one, "beta": self.rhi}])
            # the funnel, merged with old: the low half at a (mem, x0 + x1 of the words before),
            # the high half at b (the word a read is in x1 by then)
            vt = [[] for _ in range(32)]
            for b in range(4):
                for t in range(8):
                    i = 8 * b + t
                    nw = mem if i < 16 else x1  # the newest source word at this position
                    pv = x0                     # the one before it
                    vt[i] += c.t_and([rr[0], mask[b], nw[i]])
                    vt[i] += c.t_and([isset, mask[b], B[t]])
                    for rv in (1, 2, 3):
                        if b + rv < 4:
                            src = 8 * (b + rv) + t
                            vt[i] += c.t_and([rr[rv], mask[b], 1 - hdir, pv[src]])
                            vt[i] += c.t_and([rr[rv], mask[b], hdir, nw[src]])
                        else:
                            src = 8 * (b + rv - 4) + t
                            vt[i] += c.t_and([rr[rv], mask[b], 1 - hdir, nw[src]])
                            vt[i] += c.t_and([rr[rv], mask[b], hdir, pv[src]])
                    vt[i] += c.t_and([1 - mask[b], old[i]])
            v = [c.node(t, f"h.v{i}") for i, t in enumerate(vt)]
            full = self.jt[CHUNK_J - 1]
            bnd = c.node(c.t_and([self.jt[0]]) + c.t_and([self.jt[0], 1 - full, 1 - pgedge, 1 - edge], -1), "h.bnd")
        if "eql" in kinds:
            # a byte of a came down as EB; compare it with the byte of b at this position
            lk = [c.and_(_lits(EA[0:2], k), f"h.ek{k}") for k in range(4)]
            bb = [c.node([x for k in range(4) for x in c.t_and([lk[k], mem[8 * k + t]])], f"h.bb{t}")
                  for t in range(8)]
            same = c.node(c.t_eq(from_bits(bb) - from_bits(self.he), 0, -255, 255), "h.same")
            eq_n = eqbits(ci[:14], CR[:14], "h.eqn")
            fire_eql = c.and_([d_ne, n_ok, n_nz, fok], "h.feql")
            full_e = self.jt[CHUNK_J - 1]
        for u, un in enumerate(self.units):
            fi = self.first[u]
            t = (un.ret - cfg.rom_base) >> 2 & 0xFFFF
            jbits = [one if t >> b & 1 else zero for b in range(16)]
            if un.kind in SCAN or un.kind in ("sha", "maskp"):
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
            elif un.kind == "bcopy":
                fire, flags = fire_bc, [bcf, zero]
            else:
                fire = c.and_([common, fwd], f"h.fire{u}")
                flags = [f_now, zero]
            case([fi, 1 - fire], "M", decline)
            case([fi, fire], "M", pay_const(TAG_MP, [zero] + flags + ubits(u)))
            case([self.hun[u], self.ra, done], "J", jbits)
        if "bcopy" in kinds:
            case([self.rN], "CW", self.nb)
        if copy:
            case([self.ra, 1 - done, bnd], "M", pay_const(TAG_MC, ci[:12]))
            case([self.ra, 1 - done, 1 - bnd], "CW", v[:16])
            case([self.rb], "CW", v[16:])
            if "move" in kinds:  # K - 1, the descending copy's first MC
                case([self.rC], "M", pay_const(TAG_MC, [self._decr(Kb, b, f"h.k1{b}") for b in range(12)]))
            case([self.rR], "M", pay_const(TAG_OP, [one]))  # OP_MR
            r_int = rr[1] + 2 * rr[2] + 3 * rr[3]
            pre = c.node(c.t_ge(eint + r_int, 4, 0, 6, lits=[hdir, lastw, 1 - rr[0]])
                         + c.t_and([1 - hdir, first0, self.hf]), "h.pre")
            case([self.rH, mapped], "M", pay_const(TAG_MH, [fb[0], pre]))
            case([self.rH, 1 - mapped], "PA", anext)
        if "sha" in kinds:
            self.top_sha(A, B, r, eq, sltu, mem, fb, mapped, anext, fok, wok, or_, pay_const, ubits, decline, case, in_load)
        if self.scans:
            self.top_scan(A, B, CR, r, sltu, mem, fb, mapped, anext, EA, in_load, case, pay_const, ubits, or_, eqbits,
                          decline, wok, fok)
        if "maskp" in kinds:
            self.top_maskp(A, B, CR, mem, EA, in_load, case, pay_const, ubits, decline)
        if "eql" in kinds:
            bad = 1 - in_load
            case([self.re1, 1 - eq_n, 1 - full_e, in_load], "EB", bb)
            case([self.re1, 1 - eq_n, full_e], "M", pay_const(TAG_MC, ci[:12]))
            case([self.re1, 1 - eq_n, 1 - full_e, bad], "M", decline)
            case([self.re2, bad], "M", decline)
            case([self.re2, in_load, same], "EC1")
            case([self.re2, in_load, 1 - same], "F0", [zero] * 15)

    def top_sha(self, A, B, r, eq, sltu, mem, fb, mapped, anext, fok, wok, or_, pay_const, ubits, decline, case, in_load) -> None:
        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        res = self.res
        lag = (0, 1, 0, 0)
        (x1,) = self.head("s.s", [{"data": {2 + i: Chan(mem[i], lag) for i in range(32)}, "ref": one, "beta": self.sb}])
        tn_pay = pay_const(TAG_OP, [zero, zero, one])  # OP_MN
        u = next(i for i, un in enumerate(self.units) if un.kind == "sha")
        fi = self.first[u]
        rtop = c.and_(r[6:32], "s.rt")
        rtop5 = c.and_(r[5:32], "s.rt5")
        r0 = c.and_([1 - b for b in r[6:32]], "s.r0")
        fire = c.and_([1 - A[0], 1 - A[1], 1 - B[0], 1 - B[1], wok(A, "s.wa"), wok(B, "s.wb"), fok, 1 - rtop5, 1 - r0], "s.fire")
        case([fi, 1 - fire], "M", decline)
        case([fi, fire], "M", pay_const(TAG_MP, [zero, zero, zero] + ubits(u)))
        # byte order: the message's words are big-endian
        bs = [mem[8 * (3 - i // 8) + i % 8] for i in range(16)]
        bsh = [x1[8 * (3 - i // 8) + i % 8] for i in range(16, 32)]
        seq = self.seq
        s0 = c.and_([self.sa, seq(0)], "s.c00")
        wz = c.and_([1 - e for e in mem[8:22]], "s.wz")
        bad0 = or_([1 - in_load, A[0], A[1], mem[0], mem[1], wz] + mem[22:32], "s.bad0")
        m8 = [self._decr(mem[8:32], b, f"s.dm{b}") for b in range(8)]
        x8 = [self._decr(x1[8:32], b, f"s.dx{b}") for b in range(24)]
        case([s0, bad0], "M", decline)
        case([s0, 1 - bad0], "CW", mem[:8] + m8)
        case([c.and_([self.sa, self.s_load], "s.c0")], "CW", mem[:16])
        case([self.sb, seq(0)], "CW", x8[8:24])
        case([c.and_([self.sa, self.s_msg], "s.c1b")], "CW", bs)
        case([c.and_([self.sa, self.s_sch + self.s_final], "s.c2b")], "CW", res[:16])
        case([c.and_([self.sb, self.s_load], "s.c3")], "CW", x1[16:])
        case([c.and_([self.sb, self.s_msg], "s.c4")], "CW", bsh)
        case([c.and_([self.sb, self.s_sch + self.s_final], "s.c5")], "CW", res[16:])
        for rr_, lo in ((self.j2, True), (self.j3, False), (self.j4, True), (self.j5, False)):
            case([rr_, self.s_round], "CW", res[:16] if lo else res[16:])
        case([self.j6], "M", tn_pay)
        case([self.j2, seq(0) + self.s_load + self.s_final], "M", tn_pay)
        # the overlap checks: r = st / blk - W is not in [0, 256) nor just below it; W is word-aligned
        hi8 = or_(r[8:32], "s.rh")
        ok = c.and_([hi8, 1 - rtop], "s.ok")
        case([self.sc1 + self.sc2, 1 - ok], "M", decline)
        case([self.sc1 + self.sc2, ok], "M", tn_pay)
        case([self.sH, 1 - mapped], "PA", anext)
        case([self.sH, mapped], "M", pay_const(TAG_MH, [fb[0], zero]))
        t = (self.units[u].ret - cfg.rom_base) >> 2 & 0xFFFF
        case([self.gsh, self.tMN, seq(SHA_END)], "J", [one if t >> b & 1 else zero for b in range(16)])

    def top_maskp(self, A, B, CR, mem, EA, in_load, case, pay_const, ubits, decline) -> None:
        """maskp_record: three constants at the trigger, s6 and the byte before at MP's position."""
        from chipformer.rv import core as C

        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        u = next(i for i, un in enumerate(self.units) if un.kind == "maskp")
        un, fi = self.units[u], self.first[u]
        fire = c.and_(C.lits_of(A, un.arg) + C.lits_of(B, 0x5F) + C.lits_of(CR, 10), "h.fmp")
        case([fi, 1 - fire], "M", decline)
        case([fi, fire], "M", pay_const(TAG_MP, [zero, zero, zero] + ubits(u)))
        lk = [c.and_(_lits(EA[0:2], k), f"h.mk{k}") for k in range(4)]
        z = [from_bits(mem[8 * k:8 * k + 8]) for k in range(4)]
        key = c.node([x for k in range(4) for lg in ([lk[k]],)
                      for x in c.t_eq(z[k], 0x5F, 0, 255, lits=lg) + c.t_in(z[k], 0x30, 0x3A, 0, 255, lits=lg)
                      + c.t_in(z[k], 0x41, 0x5B, 0, 255, lits=lg) + c.t_in(z[k], 0x61, 0x7B, 0, 255, lits=lg)], "h.mkey")
        ok = c.and_([in_load, key] + [1 - b for b in B], "h.mok")
        t = (un.ret - cfg.rom_base) >> 2 & 0xFFFF
        case([self.mr, ok], "J", [one if t >> b & 1 else zero for b in range(16)])
        case([self.mr, 1 - ok], "M", decline)

    def top_scan(self, A, B, CR, r, sltu, mem, fb, mapped, anext, EA, in_load, case, pay_const, ubits, or_, eqbits,
                 decline, wok, fok) -> None:
        """A byte loop: the byte's class at the top, the index through tokens (module doc)."""
        c, cfg = self.c, self.cfg
        one, zero = c.one, Expr()
        idx = self.off[:16]  # (a byte role: c + j)
        lane = EA[0:2]
        lk = [c.and_(_lits(lane, k), f"h.lk{k}") for k in range(4)]
        bytes_ = [from_bits(mem[8 * k:8 * k + 8]) for k in range(4)]

        def on_lane(tf, name):
            return c.node([x for k in range(4) for x in tf(bytes_[k], [lk[k]])], name)

        n_hi0 = c.and_([1 - b for b in CR[16:32]], "h.n16")
        rs = self.rbyte
        kinds = {self.units[u].kind for u in self.scans}
        eq_e = eqbits(idx, CR[:16], "h.eqe") if kinds & {"ascii", "jscan", "jesc"} else zero
        ip1 = [self.incr(idx, b, f"h.ip{b}") for b in range(16)]
        eq_l = eqbits(ip1, CR[:16], "h.eql2") if kinds & {"tok", "chr"} else zero
        cls = {}
        if "tok" in kinds:
            cls["tok"] = on_lane(lambda z, l: c.t_in(z, 9, 14, 0, 255, lits=l) + c.t_eq(z, 32, 0, 255, lits=l), "h.ctok")
        if "chr" in kinds:
            bh0 = c.and_([1 - x for x in B[8:32]], "h.b8")
            bl = from_bits(B[:8])
            cls["chr"] = on_lane(lambda z, l: c.t_eq(z - bl, 0, -255, 255, lits=l + [bh0]), "h.cchr")
        if "ascii" in kinds:
            cls["ascii"] = on_lane(lambda z, l: c.t_ge(z, 128, 0, 255, lits=l), "h.casc")
        if "jscan" in kinds:
            cls["jscan"] = on_lane(lambda z, l: c.t_lt(z, 0x20, 0, 255, lits=l) + c.t_ge(z, 0x80, 0, 255, lits=l)
                                   + c.t_eq(z, 0x22, 0, 255, lits=l) + c.t_eq(z, 0x5C, 0, 255, lits=l), "h.cjs")
        full = self.jt[CHUNK_S - 1]
        jB = jC = zero
        ju = next((u for u in self.scans if self.units[u].kind == "jesc"), None)
        if ju is not None:  # (a class byte stops a chunk's last position: EC1 has no class)
            cls["jesc"] = on_lane(lambda z, l: c.t_lt(z, 0x20, 0, 255, lits=l) + c.t_eq(z, 0x22, 0, 255, lits=l)
                                  + c.t_eq(z, 0x5C, 0, 255, lits=l) + c.t_in(z, 0x20, 0x22, 0, 255, lits=l + [full])
                                  + c.t_in(z, 0x23, 0x5C, 0, 255, lits=l + [full]), "h.cje")
            jB = on_lane(lambda z, l: c.t_in(z, 0x20, 0x22, 0, 255, lits=l + [self.hun[ju]]), "h.jB")
            jC = on_lane(lambda z, l: c.t_in(z, 0x23, 0x5C, 0, 255, lits=l + [self.hun[ju]]), "h.jC")
        idx_top = c.and_(idx, "h.itop")
        n_nz = or_(CR[:16], "h.nnz2")
        stop = zero
        for x_, u in enumerate(self.scans):
            k = self.units[u].kind
            stop = stop + c.node(c.t_and([self.hun[u], rs]) + c.t_and(
                [self.hun[u], rs, 1 - cls[k], 1 - (eq_l if k in ("tok", "chr") else eq_e), 1 - idx_top], -1), f"h.stop{u}")
        keep = rs - stop
        bad = 1 - in_load
        ks = sum((self.hun[u] for u in self.scans if SCAN[self.units[u].kind]["start"]), Expr())
        # at the position after EC0, index = stop + 1: the loop's count of plain bytes k
        m1 = [self._decr(idx, b, f"h.im{b}") for b in range(16)]
        m2 = [self._decr(m1, b, f"h.jm{b}") for b in range(16)]
        kv = [c.node(c.t_and([1 - ks, m1[b]]) + c.t_and([ks, m2[b]]), f"h.kv{b}") for b in range(16)]
        tf = self.tMF
        case([rs, bad], "M", decline)
        case([rs, in_load, stop, tf], "M", decline)
        case([rs, in_load, stop, 1 - tf], "EC0")
        if ju is None:
            case([rs, in_load, keep, 1 - full], "SC")
        else:
            case([rs, in_load, keep, 1 - full, 1 - jB, 1 - jC], "SC")
            case([rs, in_load, keep, 1 - full, jB], "EB", [zero] * 8)
            case([rs, in_load, keep, 1 - full, jC], "EB", [one] + [zero] * 7)
        case([rs, in_load, keep, full], "EC1")
        case([self.rse, c.fam["EC0"]], "CW", kv)
        case([self.rse, c.fam["EC1"]], "CW", idx)
        rcw = c.and_([self.tscw, 1 - self.incm], "h.rcw")
        case([rcw, self.pEC0], "M", pay_const(TAG_OP, [one, one]))  # OP_MK
        lim_b = c.and_([1 - b for b in B[16:32]], "h.b16")
        bp1 = [self.incr(B, b, f"h.bp{b}") for b in range(16)]
        b_carry = c.and_(B[:16], "h.bc")
        if ju is None:
            case([rcw, self.pMP], "M", pay_const(TAG_OP, [zero, one]))  # OP_MF
            case([rcw, 1 - self.pEC0, 1 - self.pMP], "M", pay_const(TAG_OP, [one, zero]))  # OP_MR
        else:  # MP CW(len) CW(start) MF: the two positions after MP are the escape scan's
            hj = self.hun[ju]
            case([rcw, self.pMP, 1 - hj], "M", pay_const(TAG_OP, [zero, one]))
            case([rcw, self.pMP, hj, 1 - lim_b], "M", decline)
            case([rcw, self.pMP, hj, lim_b, 1 - b_carry], "CW", bp1)
            case([rcw, self.pMP, hj, lim_b, b_carry], "M", decline)
            case([rcw, self.pMP2, hj], "M", pay_const(TAG_OP, [zero, one]))
            j2 = c.and_([hj, self.pMP2], "h.j2")
            case([rcw, 1 - self.pEC0, 1 - self.pMP, 1 - j2], "M", pay_const(TAG_OP, [one, zero]))
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
            elif un.kind == "jesc":
                from chipformer.rv import core as C

                fire = c.and_(C.lits_of(A, JESC_Q) + C.lits_of(B, JESC_B), f"h.f{u}")
            else:
                fire = c.and_([n_hi0, sltu], f"h.f{u}")
            case([fi, 1 - fire], "M", decline)
            case([fi, fire], "M", pay_const(TAG_MP, [zero, zero, zero] + ubits(u)))
            ist = c.and_([self.hun[u], self.tMP], f"h.ist{u}")
            if un.kind == "jesc":
                nb = or_([1 - in_load, EA[0], EA[1]] + mem[16:32], "h.jbad")
                case([ist, nb], "M", decline)
                case([ist, 1 - nb], "CW", mem[:16])
            elif SCAN[un.kind]["start"]:
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
