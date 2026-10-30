"""The RV32IM + Zba core as a q35 circuit (spec.py has the trace format, ref.py the ROM word).

Every position computes the instruction that the next token belongs to: the state it reads is
the state after the position's own commit, so at a record's last token the circuit already works
on the next instruction, and inside a record it recomputes the current one (nothing changes
until the record's last token). Writes take their key from t-1 (a conv tap), their write gate
beta from the token at t, and their value from token payloads (lags 0..3) or from dims the
circuit computed at t-1 below the writing layer.

  L1   attn  anchor (PC head), count (2^-n), j (tokens of the record, 2^-j), v (tokens since the
             last QS / QD)
       mlp   thermometers
  L2   mlp   PC word w = anchor + n; link word w + 1; nd (quotient digits so far)
  L3-4 attn  ROM, two sublayers of 16 key heads x 2 value heads: the 4-byte pre-decoded word w
       mlp   uop / register one-hots, raw field bits (the immediates are linear in them)
  L5   attn  register file (3 read ports: RS1, RS2, RD slot; writes commit from t-1); the quotient
             head (the record's quotient digits, right-aligned by nd)
       mlp   adder stage 1 (4 operand modes), logic ops, shifter stage 1, multiplier stage 1
             (A x B, or Q x B for the remainder), store data, DIV signs / sizes
  L6   mlp   adder stage 2, shifter stage 2, multiplier stage 2; DIV: |A| >> 4s, Q x |B| and
             m x |B| products, the signed quotient
  L7-8 attn  RAM, two sublayers (32 KB of words)
       mlp   multiplier stages 3-4, loads, SB/SH merge, compares, faults, the Zba and branch
             target adders; DIV: product splits, the 15 compare columns
  L9   attn  DROM (128 KB of read-only bytes)
       mlp   multiplier stage 5, DROM bytes to bits; DIV: compare lookahead
  L10  mlp   DIV: the 15 compare signs (the next quotient digit); rd value select and size
  L11  mlp   the output code (the exit)

The vector unit (VX, docs/vx.md and vx.py) replaces VDOT / WB / SMAQA, which this circuit does
not implement.

DIV / REM (radix 16, one digit per token): at the digit position for digit k (s = 7 - k) with the
digits so far Q (right-aligned), q_k = #{m in 1..15 : |A| >> 4s >= (16 Q + m) |B|}; each test is
one sign of D - m |B| (D = (|A| >> 4s) - 16 Q |B|) computed mod 2^40 from un-normalized columns
(products of nibbles) in three MLP stages. The remainder is A + Q B or A - Q B (by the result's
sign) through the multiplier.
"""

from contextlib import contextmanager

import numpy as np

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS, Chan, K, Target
from chipformer.q35.prims import Circ, chan, from_bits, sel, thead
from chipformer.rv import ref as RF
from chipformer.rv import vxcirc
from chipformer.rv import spec as S

T4B = Target(key_heads=16, value_heads=32, intermediate=9216)  # Qwen3.5-4B's layers
# output family ids: the chip families and controls, then Q (P3: a printable Qwen token, the echo
# of a putchar; its payload indexes the bridge's printable tokens)
OUT_NAMES = list(S.FAMILIES) + S.CONTROL_NAMES + ["Q"]
FAM_ID = {n: i for i, n in enumerate(OUT_NAMES)}
NFAM = (len(OUT_NAMES) - 1).bit_length()  # family id bits in the output code
NPAY = 16  # payload bits in the output code
OUT_W = 2.0**7  # head output weight of a one-hot read (as thead)
JMAX = 66  # longest record: a digit-file row op (63 digits, then its value)
CODE_AMP = 128  # a token's code in its (tied) embedding row: CODE_AMP * (2 * bit - 1)
KAPPA = 2.0**-20  # the predicted code after the exit: KAPPA * (2 * bit - 1)
TAU_TEXT = 2.0**-40  # (P3) tau on text tokens; -1 on chip tokens (see q35.merge)
PAY_W = {"V": 15, "F0": 15, "F1": 15, "F2": 2, "J": 16, "R": 16, "BYTE": 9, "FLT": 3, "HEX": 4,
         "QS": 4, "QD": 7, "BD": 8, "VC": 15, "VE": 16, "VH": 16}


def code_bits(fam: str, p: int) -> list[int]:
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def token_code(t: int) -> list[int]:
    """The output-code bits of a chip token: family id (NFAM) then payload (16)."""
    return code_bits(*S.family(t))


def famof(t: int) -> tuple:
    """(family, payload) of a chip token; (None, 0) for a text token (P3)."""
    return S.family(t) if t < S.VOCAB else (None, 0)


def lits_of(bits, v):
    """0/1 literals that are all 1 exactly when the bits spell v."""
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(lits) -> Expr:
    """sum(lits) - (n - 1): 1 when all literals are 1, <= 0 otherwise (a key / query / z form)."""
    return sum(lits, Expr()) - (len(lits) - 1)


def col_offsets(bounds: list[tuple[int, int]], top: int = 255) -> list[int]:
    """Offsets off_i = 16 g_i - g_(i-1) (so sum off_i 16^i is a multiple of 16^n) that put every
    column (positive part <= pos_i, negative part <= neg_i) into [0, top]."""
    reach, hist = {0: None}, []
    for pos, neg in bounds:
        nxt = {}
        for gp in reach:
            for g in range(64):
                if neg <= 16 * g - gp <= top - pos and g not in nxt:
                    nxt[g] = gp
        assert nxt, "no column offsets"
        hist.append(nxt)
        reach = nxt
    g, offs = min(reach), []
    for i in range(len(bounds) - 1, -1, -1):
        gp = hist[i][g]
        offs.append(16 * g - gp)
        g = gp
    return offs[::-1]


class RvCore(Circ):
    def __init__(self, target: Target | None = None, text=None) -> None:
        """``text`` (P3, bridge.Text): the Qwen vocabulary the bridge reads and prints; None
        builds the standalone machine."""
        super().__init__(target or T4B)
        f = self.token_feature
        self.text = text

        def fam_is(nm):
            return lambda t: int(famof(t)[0] == nm)

        def pay_bit(nm, i):
            return lambda t: (famof(t)[1] >> i & 1) if famof(t)[0] == nm else 0

        names = list(S.FAMILIES) + S.CONTROL_NAMES
        paid = ("V", "F0", "F1", "F2", "J", "R", "QD", "BD", "VC", "VE", "VH") + (("BYTE",) if text else ())
        self.fam = {n: f(f"is{n}", fam_is(n), full=True) for n in names}
        self.pay = {n: [f(f"{n}.b{i}", pay_bit(n, i), full=True) for i in range(w)]
                    for n, w in PAY_W.items() if n in paid}

        def next_digit(k):
            """The token says the next token is quotient digit k: QS(k) or QD(k - 1, q)."""
            def fn(t):
                nm, p = famof(t)
                return int(nm == "QS" and p == k or nm == "QD" and p >> 4 == k - 1)
            return fn

        self.kk = [f(f"digit=={k}", next_digit(k), full=True) for k in range(8)]
        self.final = sum((self.fam[n] for n in S.FINAL), Expr())
        self.tables = {}  # name -> head node (for snapshots)

        def code_of(t):
            """A token's own output code, or None (a text token the circuit never prints)."""
            if t < S.VOCAB:
                return token_code(t)
            p = text.printable_index(t - S.TEXT_TOKEN) if text else None
            return None if p is None else code_bits("Q", p)

        # the token's own code (read by the tied output head; cleared before the exit writes the
        # predicted one)
        self.code = [f(f"code{i}", lambda t, i=i: 0 if (cb := code_of(t)) is None
                       else CODE_AMP * (2 * cb[i] - 1), full=True)
                     for i in range(NFAM + NPAY)]
        if text:
            tt = S.TEXT_TOKEN
            self.is_text = f("is_text", lambda t: int(t >= tt), full=True)
            self.mu = f("mu", lambda t: int(t < S.VOCAB), full=True)  # Qwen mutes on chip tokens
            self.tau = f("tau", lambda t: TAU_TEXT if t >= tt else -1, full=True)
            self.isTCO = f("isTCO", lambda t: int(t == tt + text.TCO), full=True)
            self.isTCC = f("isTCC", lambda t: int(t == tt + text.TCC), full=True)

            def tbyte(k, i):
                return lambda t: (text.row_bytes(t - tt)[k] >> i & 1) if t >= tt else 0

            # a text token's UTF-8 bytes (the first TEXT_ROW), bit by bit
            self.tb = [[f(f"tb{k}.b{i}", tbyte(k, i), full=True) for i in range(8)]
                       for k in range(S.TEXT_ROW)]

    floor = 0  # no node goes below this level (see at_least)

    def ffn(self, terms, name="ffn", into=None, min_level=0):
        return super().ffn(terms, name, into, max(min_level, self.floor))

    @contextmanager
    def at_least(self, sub: int):
        """Place the nodes made inside at sublayer >= sub (to move work out of a full MLP)."""
        old, self.floor = self.floor, max(self.floor, sub - 1)
        try:
            yield
        finally:
            self.floor = old

    def fn_bits(self, z, lo: int, hi: int, f, n: int, name: str, lits=()) -> list:
        """Bits of f(z) for an integer z in [lo, hi] (AND lits), by thresholds at its flips."""
        terms = [[] for _ in range(n)]
        for b in range(n):
            if f(lo) >> b & 1:
                terms[b] += self.t_and(list(lits)) if lits else self.t_const(1)
            for theta in range(lo + 1, hi + 1):
                dd = (f(theta) >> b & 1) - (f(theta - 1) >> b & 1)
                if dd:
                    terms[b] += self.t_ge(z, theta, lo, hi, dd, lits=list(lits))
        return terms

    def zeros(self, n: int, name: str) -> list:
        """Fresh dims that start at 0 (heads add onto them)."""
        return [self.token_feature(f"{name}[{i}]", lambda t: 0, full=True) for i in range(n)]

    def table(self, name, queries, keys, vheads, levels="bit", tiny=0.0, min_level=0):
        """A head with explicit per-column values / gates: vheads are dicts with v {col: Chan},
        z {col: Expr}, outs [(col, weight)], into [Expr] (one per out), beta, a."""
        first, rest = vheads[0], vheads[1:]
        outs = self.head(name, {s: sel(self, e) for s, e in queries.items()},
                         {s: sel(self, e) for s, e in keys.items()}, first["v"], first.get("a", 0),
                         first["z"], first["outs"], tiny=tiny, beta=first.get("beta"),
                         into=first.get("into"), min_level=min_level,
                         vx=[{"v": x["v"], "a": x.get("a", 0), "z": x["z"], "outs": x["outs"],
                              "beta": x.get("beta"), "into": x.get("into")} for x in rest])
        node = self.nodes[-1]
        node.levels = levels
        self.tables[name] = node
        return outs

    def int_bits(self, z: Expr, hi: int, name: str, n: int) -> list:
        """Bits of an integer 0..hi (thresholds at every flip of each bit)."""
        out = []
        for j in range(n):
            terms = []
            for theta in range(1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += self.t_ge(z, theta, 0, hi, d)
            out.append(self.node(terms, f"{name}.b{j}"))
        return out

    def counter(self, name, reset, step):
        """A 2^-k counter head: k = the `step` tokens since the latest `reset` token (0 there)."""
        one = self.one
        ((x,),) = thead(self, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        [{"data": {2: reset}, "ref": one - reset,
                          "a": 110 * step + 238 * reset}], tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "decay"  # a snapshot mid-count holds 2^-k
        return x

    def thermo(self, x, reset, n, name):
        """[k > m] for m < n, from x = 2^-k (0 at reset tokens)."""
        one = self.one
        return [self.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                           (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
                for m in range(n)]

    def split(self, z: Expr, lo: int, hi: int, name: str) -> Expr:
        """floor(z / 16) of an integer z in [lo, hi] (0 <= lo)."""
        terms = self.t_const(lo // 16) if lo // 16 else []
        for t in range(lo // 16 + 1, hi // 16 + 1):
            terms += self.t_ge(z, 16 * t, lo, hi)
        return self.node(terms, name)


def adder_stage2(c, g, p_, d, n_nib, name):
    """Result bits of a carry-lookahead add from per-nibble generate / propagate / digit bits;
    also the carry out of the top nibble."""
    one = c.one

    def carry_sources(k):
        """Mutually exclusive literal sets, one of them true iff a carry enters nibble k."""
        return [[g[j]] + p_[j + 1:k] for j in range(k)]

    r = []
    for k in range(n_nib):
        for b in range(4):
            terms = c.t_and([d[k][b]])
            for src in carry_sources(k):
                low = d[k][:b]
                terms += c.t_and(src + low + [one - d[k][b]])
                terms += c.t_and(src + low + [d[k][b]], -1)
            r.append(c.node(terms, f"{name}.r{4 * k + b}"))
    cout = c.node([t for src in carry_sources(n_nib) for t in c.t_and(src)], f"{name}.cout")
    return r, cout


def stage1_digits(c, sums, name, hi):
    """Generate, propagate and digit bits of nibble sums; sums[k] = [(int Expr, literals)]."""
    g, p_, d = [], [], []
    for k, forms in enumerate(sums):
        gt, pt, dt = [], [], [[] for _ in range(4)]
        for s, lits in forms:
            gt += c.t_ge(s, 16, 0, hi, lits=lits)
            pt += c.t_eq(s, 15, 0, hi, lits=lits)
            for b in range(4):
                for theta in range(1, hi + 1):
                    dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                    if dd:
                        dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=lits)
        g.append(c.node(gt, f"{name}.g{k}"))
        p_.append(c.node(pt, f"{name}.p{k}"))
        d.append([c.node(dt[b], f"{name}.d{k}.{b}") for b in range(4)])
    return g, p_, d


def add_nibbles(c, a_bits, b_bits, n_nib, lits, name):
    """A plain carry-lookahead add of two bit vectors (n_nib nibbles), gated by literals."""
    sums = [[(from_bits(a_bits[4 * k:4 * k + 4]) + from_bits(b_bits[4 * k:4 * k + 4]), lits)]
            for k in range(n_nib)]
    g, p_, d = stage1_digits(c, sums, name, 30)
    r, _ = adder_stage2(c, g, p_, d, n_nib, name)
    return r


def stage1_cols(c, cols, lits, name):
    """stage1_digits with a range per column: cols[k] = (int Expr, hi)."""
    g, p_, d = [], [], []
    for k, (s, hi) in enumerate(cols):
        g.append(c.node(c.t_ge(s, 16, 0, hi, lits=lits), f"{name}.g{k}"))
        p_.append(c.node(c.t_eq(s, 15, 0, hi, lits=lits), f"{name}.p{k}"))
        dt = [[] for _ in range(4)]
        for b in range(4):
            for theta in range(1, hi + 1):
                dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                if dd:
                    dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=lits)
        d.append([c.node(dt[b], f"{name}.d{k}.{b}") for b in range(4)])
    return g, p_, d


def select_bits_w(c, cases, n: int, name: str) -> list:
    """Bits of sum(wt * cond * value) (prims.select_bits with a weight per case)."""
    out = []
    for i in range(n):
        terms = []
        for lits, val, wt in cases:
            if isinstance(val, int):
                if val >> i & 1:
                    terms += c.t_and(lits, wt)
            elif i < len(val):
                b = val[i]
                if isinstance(b, int):
                    if b:
                        terms += c.t_and(lits, wt)
                elif b.terms:
                    terms += c.t_and([*lits, b], wt)
        out.append(c.node(terms, f"{name}.b{i}"))
    return out


def build(target: Target | None = None, text=None, vx: bool = False) -> RvCore:
    """The circuit; with ``text`` (P3) also the bridge to Qwen's text: a tool call's text goes
    into a buffer the program reads (TEXT_ADDR), ``</tool_call>`` starts a run (GO), putchar
    output is echoed as Qwen tokens, and the circuit owns the positions of a run."""
    c = RvCore(target, text)
    c.vx_on = vx
    no_text = c.is_text if text else 0  # text tokens do not count as tokens of a record
    one, fam, pay, kk = c.one, c.fam, c.pay, c.kk
    isV, isJR = fam["V"], fam["J"] + fam["R"]
    isSS = fam["SS0"] + fam["SS1"]
    isQ = fam["QS"] + fam["QD"]
    dp = sum(kk, Expr())  # the next token is a quotient digit

    # ---------------------------------------------------------------- L1: anchor, counters
    jr_bits = [pay["J"][i] + pay["R"][i] for i in range(16)]
    (anchor,) = thead(c, "pc", {0: one}, {0: one},
                      [{"data": {2 + i: jr_bits[i] for i in range(16)}, "ref": one, "beta": isJR}])
    c.tables["pc"] = c.nodes[-1]
    xn = c.counter("count", isJR, c.final - isJR)
    rec_start = c.final + fam["GO"]
    xj = c.counter("j", rec_start, one - rec_start - no_text)
    v_reset = rec_start + isQ
    xv = c.counter("vcnt", v_reset, one - v_reset - no_text)
    cnt = c.thermo(xn, isJR, S.N_MAX, "n")
    jt = c.thermo(xj, rec_start, JMAX, "j")
    vt = c.thermo(xv, v_reset, 3, "v")
    n_int = sum(cnt, Expr())
    at_max = cnt[S.N_MAX - 1]  # n == 63: the refresh comes first

    def jeq(v):
        return (one if v == 0 else jt[v - 1]) - (jt[v] if v < JMAX else 0)

    j0, j1, j2, j3 = (jeq(v) for v in range(4))
    v0, v1, v2 = one - vt[0], vt[0] - vt[1], vt[1] - vt[2]

    if text:  # ------------------------------------------------ P3 bridge state
        isGO, isHALT = fam["GO"], fam["HALT"]
        # running: from GO to HALT; armed: inside a tool call (its text is captured)
        ((running,),) = thead(c, "run", {0: one}, {0: one},
                              [{"data": {2: isGO}, "ref": one, "beta": isGO + isHALT}])
        c.tables["run"] = c.nodes[-1]
        ((armed,),) = thead(c, "arm", {0: one}, {0: one},
                            [{"data": {2: c.isTCO}, "ref": one, "beta": c.isTCO + c.isTCC}])
        c.tables["arm"] = c.nodes[-1]
        capw = c.and_([c.is_text, armed, one - c.isTCO], "cap.w")  # a captured text token
        xcap = c.counter("cap", c.isTCO, capw)  # its index (1, 2, ..) since <tool_call>
        capt = c.thermo(xcap, c.isTCO, S.CAP_MAX, "cap")
        caprow = {i: capt[i - 1] - (capt[i] if i < S.CAP_MAX else 0)
                  for i in range(1, S.CAP_MAX + 1)}

    # ---------------------------------------------------------------- L2: PC word, link, nd
    def add_small(abits, extra, nb, name):
        """Bits 0..nb-1 of (anchor + n + extra) for n <= 63: low 6 bits by thresholds, the rest
        by the carry into bit 6."""
        L = from_bits(abits[:6]) + n_int + extra
        lo, hi = 0, 63 + 63 + extra
        bits = []
        for j in range(6):
            terms = []
            for theta in range(lo + 1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += c.t_ge(L, theta, lo, hi, d)
            bits.append(c.node(terms, f"{name}.b{j}"))
        for j in range(6, nb):
            below = abits[6:j]
            if j < 16:
                terms = c.t_and([abits[j]])
                terms += c.t_ge(L, 64, lo, hi, 1, lits=below + [one - abits[j]])
                terms += c.t_ge(L, 64, lo, hi, -1, lits=below + [abits[j]])
            else:  # the carry out of bit 15
                terms = c.t_ge(L, 64, lo, hi, 1, lits=below)
            bits.append(c.node(terms, f"{name}.b{j}"))
        return bits

    w = add_small(anchor, 0, 16, "w")
    w1 = add_small(anchor, 1, 17, "w+1")  # the link: PC + 4 = ROM_BASE + 4 * (w + 1)
    assert S.ROM_BASE == 1 << 18 and S.ROM_WORDS == 1 << 16
    pc_bits = [Expr(), Expr()] + w + [one] + [Expr()] * 13  # PC = 0x40000 + 4 w
    link = [Expr()] * 2 + w1[:16] + [one - w1[16], w1[16]] + [Expr()] * 12
    # digits so far: nd = j - 1 - v inside a DIV record (QS at j = 0, then the digits)
    nd_int = sum(jt, Expr()) - 1 - sum(vt, Expr())
    nd_is = {v: c.node(c.t_eq(nd_int, v, -4, JMAX), f"nd=={v}") for v in range(1, 9)}

    # ---------------------------------------------------------------- L3-4: ROM fetch
    # word w: key head w[15:11] (two sublayers of 16), value head w[10], row pair (p, p + 64)
    # with p = w[9:4], half w[3] (row p: columns 2..33, row p + 64: 34..65), word w[2:0], 4 byte
    # columns (b0..b3 of the ROM word)
    fdims = c.zeros(4, "rom.b")
    # with the vector unit (vx): a 128 KB ROM pinned to L4, so L3 holds the unit's weights
    for h in range((S.ROM_WORDS >> 11) // (2 if vx else 1)):
        hl = lits_of(w[11:16], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(w[4:10], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for half in range(2):
                for k in range(8):
                    ze = K * andpre(lits_of([w[3]], half) + lits_of(w[0:3], k)
                                    + lits_of([w[10]], x))
                    for b in range(4):
                        col = 2 + 32 * half + 4 * k + b
                        z[col] = ze
                        outs.append((col, OUT_W))
                        into.append(fdims[b])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"rom{h}", q, {}, vhs, levels="byte", min_level=6 if vx else 0)
    B0, B1, B2, B3 = (256 * d for d in fdims)  # the word's bytes

    # ---------------------------------------------------------------- L4: decode
    def uop_terms(u, lits=()):
        v = RF.U[u]
        return c.t_eq(B0, v, 0, 255, lits=list(lits)) + c.t_eq(B0, v + 128, 0, 255, lits=list(lits))

    uops = {u: c.node(uop_terms(u), f"uop={u}") for u in RF.UOPS}

    def any_of(*names):
        return sum((uops[n] for n in names), Expr())

    s0 = c.node(c.t_ge(B0, 128, 0, 255), "rom.S0")
    bits1, bits2, bits3 = (c.int_bits(Bx, 255, f"rom.b{i}", 8)
                           for i, Bx in ((1, B1), (2, B2), (3, B3)))
    spare = [s0] + bits1[0:3] + bits2[0:3] + bits3[0:3]  # S0..S9
    RDb, RS1b, RS2b = bits1[3:8], bits2[3:8], bits3[3:8]
    rd = {r: c.node(c.t_in(B1, 8 * r, 8 * r + 8, 0, 255), f"rd={r}") for r in range(1, 32)}
    rs1 = {r: c.node(c.t_in(B2, 8 * r, 8 * r + 8, 0, 255), f"rs1={r}") for r in range(1, 32)}
    rs2 = {r: c.node(c.t_in(B3, 8 * r, 8 * r + 8, 0, 255), f"rs2={r}") for r in range(1, 32)}
    rdw_any = sum(rd.values(), Expr())
    imm = RS2b + spare[0:7] + [spare[6]] * 20  # I-type (and stores): sign-extended imm12
    boff = RDb + spare[0:6] + [spare[5]] * 5  # branch: word offset, sign-extended to 16 bits
    bbad, jbad = spare[6], spare[6]
    jtgt = RS1b + RS2b + spare[0:6]  # JAL: target word index
    upper = RS1b + RS2b + spare[0:10]  # LUI / AUIPC: value bits 12..31
    jl = [Expr() if not link[i].terms else
          c.node(uop_terms("JAL", [link[i]]) + uop_terms("JALR", [link[i]]), f"link.b{i}")
          for i in range(32)]
    divc = any_of(*RF.DIV_OPS)
    signed = any_of("DIV", "REM")
    branches = any_of(*RF.BRANCH_OPS)
    stores = any_of("SB", "SH", "SW")
    bprd, bget = uops["BPRD"], uops["BGET"]
    bnrow = any_of(*RF.BN_ROW)
    k7 = RS2b + spare[0:2]  # BPRD's digit index

    # ---------------------------------------------------------------- L5: register file
    P = pay["V"]
    isF1, isF2 = fam["F1"], fam["F2"]
    vcols = {}  # bit -> list of value Chans (the commit's sources; exactly one is live)
    for i in range(32):
        cs = []
        if i < 15:
            cs.append(Chan(pay["F0"][i]))
            cs.append(Chan(-1 * P[i] + isF1 + isV, (1, -1, 0, 0)))  # V at t-1 when t is F1
            cs.append(Chan(-1 * P[i] + isF2 + isV, (1, 0, -1, 0)))  # V at t-2 when t is F2
        elif i < 30:
            cs.append(Chan(pay["F1"][i - 15] + pay["F0"][14]))
            cs.append(Chan(-1 * P[i - 15] + isF2 + isV, (1, -1, 0, 0)))
        else:
            cs.append(Chan(pay["F2"][i - 30] + pay["F1"][14] + pay["F0"][14]))
        if jl[i].terms:
            cs.append(Chan(jl[i], (0, 1, 0, 0)))  # the link, computed at t-1
        vcols[i] = cs
    reg_beta = fam["F0"] + fam["F1"] + fam["F2"] + fam["J"]
    nw = stores + branches  # the RD slot holds a store's data register / a branch offset
    reg_keys = {r: Chan(rd[r] - nw, (0, 1, 0, 0)) for r in range(1, 32)}
    ports = []
    for pname, rq in (("rs1", rs1), ("rs2", rs2), ("rd", rd)):
        dims = c.zeros(32, f"x.{pname}")
        v, z, outs, into, col = {}, {}, [], [], 2
        c.reg_cols = {}
        for i in range(32):
            for ch in vcols[i]:
                v[col] = chan(ch, A_BIT)
                z[col] = K * one
                outs.append((col, OUT_W))
                into.append(dims[i])
                c.reg_cols.setdefault(i, []).append(col)
                col += 1
        for rc in REF_COLS:
            v[rc] = chan(one, A_REF)
        c.table(f"regs.{pname}", rq, reg_keys,
                [{"v": v, "z": z, "outs": outs, "into": into, "beta": reg_beta}])
        ports.append(dims)
    A, B, C = ports

    # the quotient heads: value head d (4 key heads x 2) holds the record's d-th digit in one row,
    # written at its QD token (beta: QD and j == d + 2), in one copy per alignment nd' > d
    # (nibble nd' - 1 - d); one-hot reads gated by the current nd give the digits so far
    # right-aligned (the whole |Q| after the last). (One-hot reads only: a multi-hot read of
    # often-rewritten rows picks up fp32 residues in fla's chunk kernel.)
    qal = c.zeros(32, "quo")
    for h in range(4):
        vhs = []
        for d in (2 * h, 2 * h + 1):
            v, z, outs, into, col = {}, {}, [], [], 2
            for ndp in range(d + 1, 9):
                for b in range(4):
                    v[col] = chan(Chan(pay["QD"][b]), A_BIT)
                    z[col] = K * nd_is[ndp]
                    outs.append((col, OUT_W))
                    into.append(qal[4 * (ndp - 1 - d) + b])
                    col += 1
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                        "beta": fam["QD"] + jeq(d + 2) - 1})
        c.table(f"quo{h}", {0: one}, {0: fam["QD"]}, vhs)

    # ---------------------------------------------------------------- L5-6: the digit file
    # 8 registers of 128 digits; read port 1 at row j (a row op's digit i is its record index),
    # read port 2 at row j + off (BMAC / BMSB / BLD / BGET) or k (BPRD); every write (a BD
    # token: the digit just computed) goes to both ports' copies at port 2's row of t-1
    def nib(bits, k, inv=False):
        x = from_bits(bits[4 * k:4 * k + 4])
        return 15 - x if inv else x

    jint = sum(jt, Expr())
    jbin = c.int_bits(jint, JMAX, "jbin", 7)
    n_rows = from_bits(B[7:13])
    off = from_bits(B[13:20])
    bnq = bnrow + bget
    q2 = [c.node(c.t_eq(jint + off, r, 0, JMAX + 127, lits=[bnq])
                 + c.t_eq(from_bits(k7), r, 0, 127, lits=[bprd]), f"bn.q{r}") for r in range(128)]
    lt_n = c.node(c.t_lt(jint - n_rows, 0, -63, JMAX), "bn.lt")
    eq_n = c.node(c.t_eq(jint - n_rows, 0, -63, JMAX), "bn.eq")
    mcl = [c.node(t, f"bn.m{b}") for b, t in
           enumerate(c.fn_bits(from_bits(B[20:24]), 0, 15, lambda v: min(v, 9), 4, "bn.m"))]
    sel_d = [c.node(c.t_and(lits_of(B[0:3], R) + [1 - bprd])
                    + c.t_and(lits_of(spare[7:10], R) + [bprd]), f"bn.dd{R}") for R in range(8)]
    sel_s = [c.and_(lits_of(B[3:6], R), f"bn.ds{R}") for R in range(8)]
    ovw = B[6]
    # BLD's address: rs1 - 1 - j (the digit chars end at rs1 - 1); BSCAN's: rs1 + j
    bscan = uops["BSCAN"]
    notj, jj = [1 - b for b in jbin] + [one] * 9, jbin + [Expr()] * 9
    ea_sums = [[(nib(A, k) + nib(notj, k), [uops["BLD"]]), (nib(A, k) + nib(jj, k), [bscan])]
               for k in range(4)]
    eg, ep, ed = stage1_digits(c, ea_sums, "bn.ea", 30)
    eab, _ = adder_stage2(c, eg, ep, ed, 4, "bn.ea")
    dig = {1: {}, 2: {}}
    for port in (1, 2):
        for h in range(4):
            vhs = []
            for x in range(2):
                R = 2 * h + x
                dims = c.zeros(4, f"bn{port}.r{R}")
                dig[port][R] = dims
                v = {2 + b: chan(Chan(pay["BD"][b]), A_BIT) for b in range(4)}
                for rc in REF_COLS:
                    v[rc] = chan(one, A_REF)
                vhs.append({"v": v, "z": {2 + b: K * one for b in range(4)},
                            "outs": [(2 + b, OUT_W) for b in range(4)], "into": dims,
                            "beta": fam["BD"] + sel_d[R] - 1})
            q = {r: jeq(r) for r in range(64)} if port == 1 else dict(enumerate(q2))
            c.table(f"bn{port}.{h}", q, {r: Chan(q2[r], (0, 1, 0, 0)) for r in range(128)}, vhs)
    sdig = [c.node([t for R in range(8) for t in c.t_and([sel_s[R], dig[1][R][b]])], f"bn.s{b}")
            for b in range(4)]
    ddig = [c.node([t for R in range(8) for t in c.t_and([sel_d[R], dig[2][R][b]])], f"bn.d{b}")
            for b in range(4)]
    bprod = c.node([t for v in range(1, 16) for t in c.t_and(lits_of(sdig, v), v,
                                                               a=from_bits(mcl))], "bn.prod")

    # ---------------------------------------------------------------- L5-6: adder
    md = {"ADD": any_of("ADD"),
          "ADDI": any_of("ADDI", "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW", "JALR"),
          "SUB": any_of("SUB", "SLT", "SLTU", *RF.BRANCH_OPS),
          "SUBI": any_of("SLTI", "SLTIU")}
    bsrc = {"ADD": (B, False, 0), "ADDI": (imm, False, 0), "SUB": (B, True, 1),
            "SUBI": (imm, True, 1)}
    sums = [[(nib(A, k) + nib(bsrc[m][0], k, bsrc[m][1]) + (bsrc[m][2] if k == 0 else 0), [lit])
             for m, lit in md.items()] for k in range(8)]
    g, p_, d = stage1_digits(c, sums, "add", 31)
    r, c32 = adder_stage2(c, g, p_, d, 8, "add")

    # ---------------------------------------------------------------- L6-7: logic, shifter
    floor_l6 = c.at_least(12)
    floor_l6.__enter__()
    logic = []
    for i in range(32):
        terms = []
        for op, bb in (("AND", B), ("ANDI", imm)):
            terms += c.t_and([A[i], bb[i], uops[op]])
        for op, bb in (("OR", B), ("ORI", imm)):
            terms += c.t_and([uops[op]]) + c.t_and([1 - A[i], 1 - bb[i], uops[op]], -1)
        for op, bb in (("XOR", B), ("XORI", imm)):
            terms += c.t_and([A[i], 1 - bb[i], uops[op]]) + c.t_and([1 - A[i], bb[i], uops[op]])
        logic.append(c.node(terms, f"logic.b{i}"))

    sh_ops = {"L": ("SLL", "SLLI"), "R": ("SRL", "SRLI"), "A": ("SRA", "SRAI")}

    def src_bit(dirn, i):
        if 0 <= i < 32:
            return A[i]
        return A[31] if dirn == "A" and i >= 32 else None

    y = {}
    for dirn, (rop, iop) in sh_ops.items():
        y[dirn] = []
        for i in range(32):
            terms = []
            for op, sb_ in ((rop, B), (iop, imm)):
                for kk_ in range(8):
                    j = i - 4 * kk_ if dirn == "L" else i + 4 * kk_
                    bit = src_bit(dirn, j)
                    if bit is None:
                        continue
                    terms += c.t_and([uops[op], bit] + lits_of(sb_[2:5], kk_))
            if dirn == "L":  # Zba: A << 1..3 (stage 1 passes A)
                terms += c.t_and([any_of(*RF.ZBA), A[i]])
            y[dirn].append(c.node(terms, f"sh{dirn}1.b{i}"))
    shift = []
    for i in range(32):
        terms = []
        for dirn, (rop, iop) in sh_ops.items():
            yy = y[dirn]
            for op, sb_ in ((rop, B), (iop, imm)):
                for kk_ in range(4):
                    j = i - kk_ if dirn == "L" else i + kk_
                    bit = yy[j] if 0 <= j < 32 else (yy[31] if dirn == "A" and j >= 32 else None)
                    if bit is None:
                        continue
                    terms += c.t_and([uops[op], bit] + lits_of(sb_[0:2], kk_))
        for op, k in RF.ZBA.items():
            if i >= k:
                terms += c.t_and([uops[op], y["L"][i - k]])
        shift.append(c.node(terms, f"shift.b{i}"))
    floor_l6.__exit__(None, None, None)

    # ---------------------------------------------------------------- L5-9: multiplier
    # 64-bit product of nibbles: p_ij = a_i b_j <= 225 (stage 1; a = rs1, or the quotient for
    # REM), split into nibbles (stage 2); column k sums lo_ij (i + j = k), hi_ij (i + j = k - 1)
    # and, for MULH / MULHSU, the two's-complement corrections -a31 * b and -b31 * a of the high
    # word (as ~x + 1); a column stays <= 255, so it splits into a digit and a carry (stage 3);
    # then a carry-lookahead add of digits and carries (stages 4-5). REM: R = A - sigma Q B mod
    # 2^32 (sigma = -1 when the quotient is negative) in the low 8 columns.
    mh_a = any_of("MULH", "MULHSU")  # rs1 signed
    mh_b = uops["MULH"]  # rs2 signed
    mul_any = any_of(*RF.MUL_OPS)
    remc = any_of("REM", "REMU")
    prod = {}
    for i in range(8):
        for j in range(8):
            terms = []
            for v in range(1, 16):
                terms += c.t_and(lits_of(A[4 * i:4 * i + 4], v) + [1 - divc], v, a=nib(B, j))
                terms += c.t_and(lits_of(qal[4 * i:4 * i + 4], v) + [divc], v, a=nib(B, j))
            prod[i, j] = c.node(terms, f"mul.p{i}{j}")
    corr_a = [c.node(c.t_and([A[31], mh_a], a=nib(B, j, inv=True)), f"mul.ca{j}") for j in range(8)]
    corr_b = [c.node(c.t_and([B[31], mh_b], a=nib(A, j, inv=True)), f"mul.cb{j}") for j in range(8)]
    plus1 = c.node(c.t_and([A[31], mh_a]) + c.t_and([B[31], mh_b]), "mul.plus1")
    lo, hi = {}, {}
    for (i, j), pij in prod.items():
        hi[i, j] = c.node([t for m in range(1, 15) for t in c.t_ge(pij, 16 * m, 0, 225)],
                          f"mul.hi{i}{j}")
        lo[i, j] = pij - 16 * hi[i, j]
    # DIV signs (L5): sa / sb (signed operand negative), sigma (the quotient is negative)
    sa = c.and_([signed, A[31]], "div.sa")
    sbn = c.and_([signed, B[31]], "div.sb")
    bz = c.and_([1 - b for b in B], "div.bz")
    sigma = c.node(c.t_and([signed, A[31], 1 - B[31]])
                   + c.t_and([signed, A[31]] + [1 - b for b in B], -1)
                   + c.t_and([signed, 1 - A[31], B[31]]), "div.sigma")
    cols = []
    rem_p, rem_m = [], []  # REM columns, sigma = 1 (A + QB) and sigma = 0 (A - QB, offset)
    for k in range(16):
        s = sum((lo[i, k - i] for i in range(8) if 0 <= k - i < 8), Expr())
        s += sum((hi[i, k - 1 - i] for i in range(8) if 0 <= k - 1 - i < 8), Expr())
        if k < 8:
            nlo = sum(1 for i in range(8) if 0 <= k - i < 8)
            nhi = sum(1 for i in range(8) if 0 <= k - 1 - i < 8)
            rem_p.append((nib(A, k) + s, 15 + 15 * nlo + 14 * nhi))
            rem_m.append((nib(A, k) - s, (15, 15 * nlo + 14 * nhi)))
        if k >= 8:
            s += corr_a[k - 8] + corr_b[k - 8]
        if k == 8:
            s += plus1
        cols.append(s)
    rm_off = col_offsets([b for _, b in rem_m])
    # each mode's columns split into digit + carry at every position (so every stage-4 sum is in
    # range), and stage 4 picks the instruction's mode
    modes = [("mul", cols, [0] * 16, [255] * 16, [mul_any]),
             ("remp", [e for e, _ in rem_p], [0] * 8, [hp for _, hp in rem_p], [remc, sigma]),
             ("remm", [e + o for (e, _), o in zip(rem_m, rm_off)], [o - b[1] for (_, b), o in
                                                                    zip(rem_m, rm_off)],
              [o + 15 for o in rm_off], [remc, 1 - sigma])]
    tsum = [[] for _ in range(16)]
    for mname, mcols, los, his, lits in modes:
        car = [c.split(e, lo_, hi_, f"mul.{mname}.c{k}")
               for k, (e, lo_, hi_) in enumerate(zip(mcols, los, his))]
        for k, e in enumerate(mcols):
            tsum[k].append((e - 16 * car[k] + (car[k - 1] if k else 0), lits))
    mg, mp, mdg = stage1_digits(c, tsum, "mul", 30)
    mres, _ = adder_stage2(c, mg, mp, mdg, 16, "mul")
    mul_lo, mul_hi = mres[:32], mres[32:]

    # store data for SW (0 on V tokens, so an SB/SH record's last V does not leak it)
    sd = [c.and_([C[i], one - isV], f"sd.b{i}") for i in range(32)]

    # ---------------------------------------------------------------- L5-10: DIV digits
    # |B| = B' + sb with B' = B xor sb (as 8 nibbles, the +1 folded into nibble 0: 0..16)
    bp = [c.node(c.t_and([B[i]]) + c.t_and([B[i], signed, B[31]], -1)
                 + c.t_and([1 - B[i], signed, B[31]]), f"div.bp{i}") for i in range(32)]
    bm = [nib(bp, 0) + sbn] + [nib(bp, j) for j in range(1, 8)]
    nzA = [c.node(c.t_const(1) + c.t_and([1 - b for b in A[4 * i:]], -1), f"div.nza{i}")
           for i in range(8)]
    # |A| >> 4s for s = 7 - k: (A xor sa) >> 4s, plus 1 when sa and A's low 4s bits are 0
    ahi = []
    for i in range(32):
        terms = []
        for k in range(8):
            j = i + 4 * (7 - k)
            if j < 32:
                terms += c.t_and([kk[k], A[j], 1 - sa]) + c.t_and([kk[k], 1 - A[j], sa])
        ahi.append(c.node(terms, f"div.ahi{i}"))
    ca = c.node([t for k in range(8)
                 for t in c.t_and([kk[k], sa] + [1 - A[j] for j in range(4 * (7 - k))])],
                "div.ca")
    nzB = [c.node(c.t_const(1) + c.t_and([1 - b for b in bp[4 * i:]], -1), f"div.nzb{i}")
           for i in range(8)]
    # k0 = 7 - dA + dB clamped to 0..8 (0 when B = 0): the QS payload
    kv = 7 - sum(nzA, Expr()) + sum(nzB, Expr())
    k0b = []
    for bit in range(4):
        terms = []
        for v in range(1, 8):
            if v >> bit & 1:
                terms += c.t_eq(kv, v, -1, 15, lits=[1 - bz])
        if bit == 3:
            terms += c.t_ge(kv, 8, -1, 15, lits=[1 - bz])
        k0b.append(c.node(terms, f"div.k0.b{bit}"))
    # products Q (digits so far, at most 7 nibbles) x |B| nibbles, and m |B| for m = 1..15
    prod2, hi2, lo2 = {}, {}, {}
    for a in range(7):
        for b in range(8):
            top = 15 * (16 if b == 0 else 15)
            terms = []
            for v in range(1, 16):
                terms += c.t_and(lits_of(qal[4 * a:4 * a + 4], v), v, a=bm[b])
            prod2[a, b] = c.node(terms, f"div.p{a}{b}")
            hi2[a, b] = c.split(prod2[a, b], 0, top, f"div.hi{a}{b}")
            lo2[a, b] = prod2[a, b] - 16 * hi2[a, b]
    him, lom = {}, {}
    for m in range(1, 16):
        for b in range(8):
            top = 16 if b == 0 else 15
            terms = []
            for x in range(1, top + 1):
                dd = m * x // 16 - m * (x - 1) // 16
                if dd:
                    terms += c.t_ge(bm[b], x, 0, top, dd)
            him[m, b] = c.node(terms, f"div.mh{m}.{b}")
            lom[m, b] = m * bm[b] - 16 * him[m, b]
    # the compare columns of D - m |B| (mod 16^10): |A|>>4s - 16 Q |B| - m |B|
    bounds, base = [], []
    for i in range(10):
        e = (nib(ahi, i) if i < 8 else Expr()) + (ca if i == 0 else 0)
        pos = (15 if i < 8 else 0) + (1 if i == 0 else 0)
        neg = 0
        for a in range(7):
            b = i - 1 - a
            if 0 <= b < 8:
                e -= lo2[a, b]
                neg += 15
            b = i - 2 - a
            if 0 <= b < 8:
                e -= hi2[a, b]
                neg += 15 if b == 0 else 14
        base.append(e)
        neg += (15 if i < 8 else 0) + ((15 if i == 1 else 14) if 1 <= i <= 8 else 0)
        bounds.append((pos, neg))
    coff = col_offsets(bounds)
    cmp_ge = []  # C_m = [D >= m |B|]
    for m in range(1, 16):
        dig, car = [], []
        for i in range(10):
            e = base[i] + coff[i]
            if i < 8:
                e -= lom[m, i]
            if 1 <= i <= 8:
                e -= him[m, i - 1]
            lo_, hi_ = coff[i] - bounds[i][1], coff[i] + bounds[i][0]
            cc = c.split(e, lo_, hi_, f"div.c{m}.{i}")
            car.append((cc, hi_ // 16))
            dig.append(e - 16 * cc)
        g_, p2_ = [], []
        for i in range(9):
            s = dig[i] + (car[i - 1][0] if i else 0)
            shi = 15 + (car[i - 1][1] if i else 0)
            g_.append(c.node(c.t_ge(s, 16, 0, shi), f"div.g{m}.{i}"))
            p2_.append(c.node(c.t_eq(s, 15, 0, shi), f"div.pp{m}.{i}"))
        s9 = dig[9] + car[8][0]
        shi = 15 + car[8][1]
        u = []
        for inc in (0, 1):
            terms = []
            for theta in range(1, shi + 1):
                dd = int((theta + inc) % 16 >= 8) - int((theta - 1 + inc) % 16 >= 8)
                if dd:
                    terms += c.t_ge(s9, theta, 0, shi, dd)
            u.append(c.node(terms, f"div.u{inc}.{m}"))
        # bit 39 of the sum: u1 if a carry enters nibble 9, else u0
        terms = c.t_and([u[0]])
        for j in range(9):
            src = [g_[j]] + p2_[j + 1:9]
            terms += c.t_and(src + [u[1]]) + c.t_and(src + [u[0]], -1)
        cmp_ge.append(one - c.node(terms, f"div.neg{m}"))
    qbits = [sum(((m >> bit & 1) - ((m - 1) >> bit & 1)) * cmp_ge[m - 1] for m in range(1, 16))
             for bit in range(4)]
    kbits = [sum((kk[k] for k in range(8) if k >> bit & 1), Expr()) for bit in range(3)]
    # the signed quotient: -Q = ~Q + 1 when sigma
    qs = []
    for i in range(32):
        zero_below = [1 - qal[j] for j in range(i)]
        qs.append(c.node(c.t_and([1 - sigma, qal[i]]) + c.t_and([sigma, qal[i]] + zero_below)
                         + c.t_and([sigma, 1 - qal[i]])
                         + c.t_and([sigma, 1 - qal[i]] + zero_below, -1), f"div.qs{i}"))

    # ---------------------------------------------------------------- L7-8: Zba, branch target
    zres = add_nibbles(c, shift, B, 8, [any_of(*RF.ZBA)], "zba")
    with c.at_least(14):
        btgt = add_nibbles(c, w, boff, 4, [branches], "btgt")

    # ---------------------------------------------------------------- L7-8: RAM
    # word index wi = EA[14:2] (RAM at 0): key head = (wi[12], wi[3:0]) = (EA[14], EA[5:2]),
    # value head wi[4] = EA[6], row wi[11:5] = EA[13:7]
    EA = [r[i] + eab[i] if i < 15 else r[i] for i in range(32)]  # (BLD: r = 0; else eab = 0)
    is_store = stores
    mem = c.zeros(32, "mem")
    n_ram_heads = S.RAM_WORDS >> 8
    for h in range(n_ram_heads):
        hl = lits_of(EA[2:6], h & 15) + lits_of(EA[14:14 + (n_ram_heads >> 4).bit_length() - 1],
                                                 h >> 4)
        keys, q = {}, {}
        for row in range(128):
            rl = lits_of(EA[7:14], row)
            keys[row] = Chan(andpre(rl + hl + [is_store]), (0, 1, 0, 0))
            q[row] = andpre(rl + hl)
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for i in range(32):  # block A: SW data from t-1
                v[2 + i] = chan(Chan(sd[i], (0, 1, 0, 0)), A_BIT)
            for i in range(32):  # block B: the SB/SH word from the record's V tokens
                lag = 3 - i // 15 if i < 30 else 1
                taps = [0, 0, 0, 0]
                taps[0], taps[lag] = 1, -1
                v[34 + i] = chan(Chan(-1 * P[i % 15 if i < 30 else i - 30] + isSS + isV,
                                      tuple(taps)), A_BIT)
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            for col in range(2, 66):
                z[col] = K * (EA[6] if x else 1 - EA[6])
                outs.append((col, OUT_W))
                into.append(mem[(col - 2) % 32])
            vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                        "beta": fam[f"ST{x}"] + fam[f"SS{x}"]})
        c.table(f"ram{h}", q, keys, vhs)

    # ---------------------------------------------------------------- L9: DROM
    # word index wi = EA[16:2] (DROM at 0x20000): key head wi[14:11] = EA[16:13], value head
    # wi[10] = EA[12], row pair (p, p + 64) with p = EA[11:6], half EA[5], word EA[4:2]
    ddims = c.zeros(4, "drom.b")
    for h in range(S.DROM_WORDS >> 11):
        hl = lits_of(EA[13:17], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(EA[6:12], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for half in range(2):
                for k in range(8):
                    ze = K * andpre(lits_of([EA[5]], half) + lits_of(EA[2:5], k)
                                    + lits_of([EA[12]], x))
                    for b in range(4):
                        col = 2 + 32 * half + 4 * k + b
                        z[col] = ze
                        outs.append((col, OUT_W))
                        into.append(ddims[b])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"drom{h}", q, {}, vhs, levels="byte")
    dbits = [b for k in range(4) for b in c.int_bits(256 * ddims[k], 255, f"drom.b{k}", 8)]

    # ---------------------------------------------------------------- P3: the text buffer
    # row i = the i-th captured token's bytes (2 value heads of 8 bytes), read as memory at
    # TEXT_ADDR + 16 i: row EA[10:4], value head EA[3], word EA[2]
    if text:
        tmem = c.zeros(32, "tmem")
        tsel = [EA[15]] + [1 - EA[b] for b in range(11, 15)]
        q = {i: andpre(lits_of(EA[4:11], i) + tsel) for i in caprow}
        keys = {i: Chan(e) for i, e in caprow.items()}
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for k in range(8):
                for i in range(8):
                    col = 2 + 8 * k + i
                    v[col] = chan(c.tb[8 * x + k][i], A_BIT)
                    z[col] = K * andpre(lits_of([EA[3]], x) + lits_of([EA[2]], k // 4))
                    outs.append((col, OUT_W))
                    into.append(tmem[8 * (k % 4) + i])
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into, "beta": capw,
                        "a": 238 * c.isTCO})  # cleared at <tool_call>
        c.table("tbuf", q, keys, vhs)

    # ---------------------------------------------------------------- loads, merge, compares
    lane = EA[0:2]
    in_ram = c.and_([1 - b for b in EA[15:32]], "in_ram")
    in_drom = c.and_([1 - b for b in EA[18:32]] + [EA[17]], "in_drom")
    in_text = (c.and_(tsel + [1 - b for b in EA[16:32]], "in_text") if text else None)
    ld = []
    for i in range(32):
        terms = c.t_and([uops["LW"], mem[i], in_ram] + lits_of(lane, 0))
        for k in range(4):
            if i < 8:
                terms += c.t_and([uops["LBU"], mem[8 * k + i], in_ram] + lits_of(lane, k))
                terms += c.t_and([uops["LB"], mem[8 * k + i], in_ram] + lits_of(lane, k))
            else:
                terms += c.t_and([uops["LB"], mem[8 * k + 7], in_ram] + lits_of(lane, k))
        for hh in range(2):
            if i < 16:
                terms += c.t_and([uops["LHU"], mem[16 * hh + i], in_ram] + lits_of(lane, 2 * hh))
                terms += c.t_and([uops["LH"], mem[16 * hh + i], in_ram] + lits_of(lane, 2 * hh))
            else:
                terms += c.t_and([uops["LH"], mem[16 * hh + 15], in_ram] + lits_of(lane, 2 * hh))
        ld.append(c.node(terms, f"load.b{i}"))
    bnib = [c.node([t for k in range(4) for t in c.t_and([uops["BLD"], mem[8 * k + b], in_ram]
                                                          + lits_of(lane, k))], f"bn.nib{b}")
            for b in range(4)]
    isdig = c.node([t for k in range(4) for t in
                    c.t_and([bscan] + lits_of(lane, k) + lits_of(mem[8 * k + 3:8 * k + 8], 6))
                    + c.t_and([bscan] + lits_of(lane, k) + lits_of(mem[8 * k + 1:8 * k + 8], 28))],
                   "bn.isdig")
    cin = from_bits(pay["BD"][4:8])
    dv, pv = from_bits(ddig), bprod
    forms = [(dv + pv + cin, 165, [uops["BMAC"], 1 - ovw], lambda v: (v % 10, v // 10)),
             (pv + cin, 150, [uops["BMAC"], ovw], lambda v: (v % 10, v // 10)),
             (150 + dv - pv - cin, 165, [uops["BMSB"]],
              lambda v: (v % 10, max(15 - v // 10, 0))),
             (from_bits(bnib), 15, [uops["BLD"]], lambda v: (min(v, 9), 0))]
    with c.at_least(18):
        dt, ct = [[] for _ in range(4)], [[] for _ in range(4)]
        for z, hi_, lits, f in forms:
            for b, t in enumerate(c.fn_bits(z, 0, hi_, lambda v: f(v)[0], 4, "bn", lits)):
                dt[b] += t
            for b, t in enumerate(c.fn_bits(z, 0, hi_, lambda v: f(v)[1], 4, "bn", lits)):
                ct[b] += t
        bd_dig = [c.node(dt[b], f"bn.dig{b}") for b in range(4)]
        bd_car = [c.node(ct[b], f"bn.car{b}") for b in range(4)]
    merged = []
    for i in range(32):
        terms = []
        for k in range(4):
            inl = 8 * k <= i < 8 * k + 8
            src = C[i - 8 * k] if inl else mem[i]
            terms += c.t_and([uops["SB"], src] + lits_of(lane, k))
        for hh in range(2):
            inl = 16 * hh <= i < 16 * hh + 16
            src = C[i - 16 * hh] if inl else mem[i]
            terms += c.t_and([uops["SH"], src] + lits_of(lane, 2 * hh))
        merged.append(c.node(terms, f"merge.b{i}"))
    b31 = {"SUB": B[31], "SUBI": imm[31]}
    slt = c.node([t for m in b31 for t in c.t_and([md[m], A[31], 1 - b31[m]])
                  + c.t_and([md[m], A[31], b31[m], r[31]])
                  + c.t_and([md[m], 1 - A[31], 1 - b31[m], r[31]])], "slt")
    sltu = c.node(c.t_and([md["SUB"] + md["SUBI"], 1 - c32]), "sltu")
    eq = c.and_([1 - b for b in r], "eq")

    # ---------------------------------------------------------------- faults, ecalls
    loads = any_of(*RF.LOAD_OPS)
    halfs, words = any_of("LH", "LHU"), uops["LW"]
    s_halfs, s_words = uops["SH"], uops["SW"]
    a7_is = {v: c.and_(lits_of(A, v), f"a7=={v}") for v in (1, 93)}
    ecall = uops["ECALL"]
    mis_ld = c.node(c.t_and([halfs, EA[0]]) + c.t_and([words, EA[0]])
                    + c.t_and([words, 1 - EA[0], EA[1]]), "fault.misld")
    mis_st = c.node(c.t_and([s_halfs, EA[0]]) + c.t_and([s_words, EA[0]])
                    + c.t_and([s_words, 1 - EA[0], EA[1]]), "fault.misst")
    oor_ld = c.node(c.t_and([loads, 1 - mis_ld]) + c.t_and([loads, 1 - mis_ld, in_ram], -1)
                    + c.t_and([loads, 1 - mis_ld, in_drom], -1)
                    + (c.t_and([loads, 1 - mis_ld, in_text], -1) if text else []), "fault.oorld")
    oor_st = c.node(c.t_and([is_store, 1 - mis_st])
                    + c.t_and([is_store, 1 - mis_st, in_ram], -1), "fault.oorst")
    jalr = uops["JALR"]
    # a bad JALR target (EA & ~1): misaligned, or outside the ROM (EA[31:19] = 0, EA[18] = 1)
    rom_ok = [1 - b for b in EA[19:32]] + [EA[18]]
    fc6_terms = (c.t_and([jalr, EA[1]]) + c.t_and([jalr, 1 - EA[1]])
                 + c.t_and([jalr, 1 - EA[1]] + rom_ok, -1) + c.t_and([uops["JAL"], jbad]))
    fc = {0: c.and_([uops["ILLEGAL"]], "fault.0"), 1: c.and_([uops["EBREAK"]], "fault.1"),
          2: mis_ld, 3: mis_st, 4: oor_ld, 5: oor_st,
          7: c.node(c.t_and([ecall]) + c.t_and([ecall, a7_is[1]], -1)
                    + c.t_and([ecall, a7_is[93]], -1), "fault.7")}

    # ---------------------------------------------------------------- rd value, size, branch
    addish = any_of("ADD", "ADDI", "SUB")
    sltish, sltuish = any_of("SLT", "SLTI"), any_of("SLTU", "SLTIU")
    logicops = any_of("AND", "ANDI", "OR", "ORI", "XOR", "XORI")
    shiftops = any_of("SLL", "SLLI", "SRL", "SRLI", "SRA", "SRAI")
    mulhs = any_of("MULH", "MULHSU", "MULHU")
    zero = Expr()
    misa = [one if S.MISA >> i & 1 else zero for i in range(32)]
    srcs = [([addish], r), ([uops["LUI"]], [zero] * 12 + upper),
            ([uops["AUIPC"]], [zero] * 2 + w[:10] + upper), ([uops["MISA"]], misa),
            ([logicops], logic), ([shiftops], shift), ([loads, in_ram], ld),
            ([uops["MUL"]], mul_lo), ([mulhs], mul_hi), ([any_of(*RF.ZBA)], zres),
            ([uops["LW"], in_drom], dbits)]
    for k in range(4):  # DROM bytes and halves
        lk = lits_of(lane, k)
        byte = dbits[8 * k:8 * k + 8]
        srcs.append(([uops["LBU"], in_drom] + lk, byte + [zero] * 24))
        srcs.append(([uops["LB"], in_drom] + lk, byte + [byte[7]] * 24))
        if k % 2 == 0:
            half = dbits[8 * k:8 * k + 16]
            srcs.append(([uops["LHU"], in_drom] + lk, half + [zero] * 16))
            srcs.append(([uops["LH"], in_drom] + lk, half + [half[15]] * 16))
    if text:  # text-buffer loads: the same word / byte / half selection
        srcs.append(([uops["LW"], in_text], tmem))
        for k in range(4):
            lk = lits_of(lane, k)
            byte = tmem[8 * k:8 * k + 8]
            srcs.append(([uops["LBU"], in_text] + lk, byte + [zero] * 24))
            srcs.append(([uops["LB"], in_text] + lk, byte + [byte[7]] * 24))
            if k % 2 == 0:
                half = tmem[8 * k:8 * k + 16]
                srcs.append(([uops["LHU"], in_text] + lk, half + [zero] * 16))
                srcs.append(([uops["LH"], in_text] + lk, half + [half[15]] * 16))
    n_plain = len(srcs)
    srcs += [([any_of("DIV", "DIVU")], qs), ([remc], mul_lo)]
    val, k0t, k1t = [], [], []
    for i in range(32):
        terms = []
        for gate, bits in srcs:
            if bits[i].terms:
                terms += c.t_and([bits[i]] + gate)
        if i == 0:
            terms += c.t_and([slt, sltish]) + c.t_and([sltu, sltuish])
        val.append(c.node(terms, f"val.b{i}"))
    for gate, bits in srcs:
        k0t += c.t_and(gate + bits[14:32]) + c.t_and(gate + [1 - b for b in bits[14:32]])
        k1t += c.t_and(gate + bits[29:32]) + c.t_and(gate + [1 - b for b in bits[29:32]])
    k0t += c.t_and([sltish + sltuish])
    k1t += c.t_and([sltish + sltuish])
    k0 = c.node(k0t, "fits15")
    k1 = c.node(k1t, "fits30")
    taken = c.node(c.t_and([uops["BEQ"], eq]) + c.t_and([uops["BNE"], 1 - eq])
                   + c.t_and([uops["BLT"], slt]) + c.t_and([uops["BGE"], 1 - slt])
                   + c.t_and([uops["BLTU"], sltu]) + c.t_and([uops["BGEU"], 1 - sltu]),
                   "taken")
    # fault 6: a bad JALR target, a bad JAL target, or a taken branch to a bad target
    fc[6] = c.node(fc6_terms + c.t_and([taken, bbad]), "fault.6")
    any_fault = sum(fc.values(), Expr())
    cause = [sum((fc[k] for k in fc if k >> b & 1), Expr()) for b in range(3)]

    # ---------------------------------------------------------------- the next token
    valc = (addish + uops["LUI"] + uops["AUIPC"] + uops["MISA"] + logicops + shiftops + loads
            + uops["MUL"] + mulhs + any_of(*RF.ZBA) + sltish + sltuish)
    ok = one - any_fault
    nr = [j0, one - at_max, ok]  # the first token of a record (no refresh, no fault)
    cases = []  # (literals, family, payload bits or int)
    # P3: at a BYTE / HALT the next token is the echo (a Qwen token), at </tool_call> it is GO;
    # every other case is off there
    nb = [one - fam["BYTE"] - fam["HALT"] - c.isTCC] if text else []

    def case(lits, fam_name, payload=0, wt=1):
        cases.append((lits + nb, fam_name, payload, wt))

    case([j0, at_max], "R", w)
    case(nr + [valc, rdw_any, k0], "F0", val[0:15])
    case(nr + [valc, rdw_any, 1 - k0], "V", val[0:15])
    case([j1, ok, valc, rdw_any, k1], "F1", val[15:30])
    case([j1, ok, valc, rdw_any, 1 - k1], "V", val[15:30])
    case([j2, ok, valc, rdw_any], "F2", val[30:32])
    case(nr + [valc, 1 - rdw_any], "NT")
    # DIV-class: QS(k0), the digits QD(k << 4 | q), then the value record (v counts its tokens)
    case(nr + [divc, rdw_any], "QS", k0b)
    case(nr + [divc, 1 - rdw_any], "NT")
    case([divc, dp], "QD", qbits + kbits)
    dv = [divc, 1 - dp, 1 - j0]
    case(dv + [v0, k0], "F0", val[0:15])
    case(dv + [v0, 1 - k0], "V", val[0:15])
    case(dv + [v1, k1], "F1", val[15:30])
    case(dv + [v1, 1 - k1], "V", val[15:30])
    case(dv + [v2], "F2", val[30:32])
    case(nr + [branches, taken], "J", btgt)
    case(nr + [branches, 1 - taken], "NT")
    case(nr + [uops["JAL"]], "J", jtgt)
    case(nr + [uops["JALR"]], "J", EA[2:18])  # word = (EA - 0x40000) >> 2
    case(nr + [uops["SW"], 1 - EA[6]], "ST0")
    case(nr + [uops["SW"], EA[6]], "ST1")
    sbsh = any_of("SB", "SH")
    case(nr + [sbsh], "V", merged[0:15])
    case([j1, ok, sbsh], "V", merged[15:30])
    case([j2, ok, sbsh], "V", merged[30:32])
    case([j3, ok, sbsh, 1 - EA[6]], "SS0")
    case([j3, ok, sbsh, EA[6]], "SS1")
    case(nr + [ecall, a7_is[1]], "BYTE", B[0:9])  # a byte, or (256..) a P3 special token
    case(nr + [ecall, a7_is[93]], "HALT")
    case(nr + [uops["FENCE"]], "NT")
    # digit-file ops: BD per digit (j < n), then rd = the last carry (or NT)
    for start in ([j0, one - at_max], [one - j0]):
        case(start + [bnrow, lt_n], "BD", bd_dig + bd_car)
        case(start + [bnrow, eq_n, rdw_any], "F0", pay["BD"][4:8])
        case(start + [bnrow, eq_n, 1 - rdw_any], "NT")
    lt63 = one - jt[62]
    for start in ([j0, one - at_max], [one - j0]):
        case(start + [bscan, isdig, lt63], "SC")
        for stop in ([1 - isdig], [isdig, 1 - lt63]):
            case(start + [bscan, rdw_any] + stop, "F0", jbin)
            case(start + [bscan, 1 - rdw_any] + stop, "NT")
    case(nr + [bget, rdw_any], "F0", ddig)
    case(nr + [bget, 1 - rdw_any], "NT")
    case(nr + [bprd], "BYTE", ddig + [one, one])  # '0' + digit
    # a fault: FLT(cause), the PC's 8 hex digits (most significant first), HALT
    case([j0, one - at_max, any_fault], "FLT", cause)
    for k in range(8):
        case([jeq(1 + k), any_fault], "HEX", pc_bits[4 * (7 - k):4 * (8 - k)])
    case([jeq(9), any_fault], "HALT")
    # the vector unit (vxcirc.py): its state, memories and exit cases
    c.vx = None if not vx else vxcirc.build(c, one, fam, pay, uops, A, B, C, jt, jeq, nr, rec_start, case,
                         rdw_any, ok)
    owner = None
    if text:
        cases.append(([fam["BYTE"]], "Q", c.pay["BYTE"], 1))
        nl = text.printable_index(text.NL)  # a run ends in a newline
        cases.append(([fam["HALT"]], "Q", [one if nl >> i & 1 else Expr() for i in range(9)], 1))
        cases.append(([c.isTCC], "GO", 0, 1))
        # who predicts the next token: the circuit from </tool_call> through a run's last echo
        owner = c.node(c.t_and([running]) + c.t_and([fam["HALT"]])
                       + c.t_and([c.isTCC, one - running]), "owner")
    c.owner = owner
    own = [owner] if text else []
    fam_cases = [(l_, FAM_ID[fn], wt) for l_, fn, _, wt in cases if fn is not None]  # (None: payload only)
    pay_cases = [(l_, pl, wt) for l_, _, pl, wt in cases if not isinstance(pl, int)]
    if text:
        # merged: no separate select dims (they would outlive the exit and reach Qwen's norms);
        # predictions read the exit's code dims
        c.out_fam = c.out_pay = None
        exit_min = max(n.level for n in c.nodes)
    else:
        fam_bits = select_bits_w(c, fam_cases, NFAM, "exit.fam")
        pay_bits = select_bits_w(c, pay_cases, NPAY, "exit.pay")
        c.out_fam, c.out_pay = fam_bits, pay_bits
        exit_min = max(c.level(b) for b in fam_bits) - 1
    # the exit: the input code is cleared, and the same cases write KAPPA * (2 * bit - 1) into
    # the code dims in the output stage itself (no extra layer); a stock output head tied to the
    # embedding then scores every token by how many code bits agree
    for e in c.code:
        c.ffn([(-1, one, e), (1, one, -1 * e)], "code.clear", into=e)
    for i, e in enumerate(c.code):
        cs, bit = (fam_cases, i) if i < NFAM else (pay_cases, i - NFAM)
        terms = c.t_and(own, -KAPPA) if text else c.t_const(-KAPPA)
        for lits, val_, wt in cs:
            if isinstance(val_, int):
                if val_ >> bit & 1:
                    terms += c.t_and(lits + own, 2 * KAPPA * wt)
            elif bit < len(val_) and val_[bit].terms:
                terms += c.t_and([*lits, val_[bit]] + own, 2 * KAPPA * wt)
        c.ffn(terms, f"exit.code{i}", into=e, min_level=exit_min)
    c.keep = [next(iter(e.terms)) for e in c.code]
    if text:
        # tau: 0 wherever the circuit predicts (chip tokens: -1 + 1; owned text: TAU - TAU), the
        # text value where Qwen does, which keeps chip tokens out of Qwen's text (q35.merge)
        c.ffn([(1, c.mu, one)] + c.t_and([owner, c.is_text], -TAU_TEXT), "exit.tau",
              into=c.tau, min_level=exit_min)
        c.keep += [next(iter(c.tau.terms)), next(iter(c.mu.terms))]
    c.probe = {"w": w, "anchor": anchor, "cnt": cnt, "jt": jt, "vt": vt, "A": A, "B": B, "C": C,
               "imm": imm, "r": r, "val": val, "mem": mem, "ld": ld, "k0": k0, "k1": k1,
               "taken": taken, "logic": logic, "shift": shift, "merged": merged,
               "mul": mres, "fault": fc, "qal": qal, "qs": qs, "cmp": cmp_ge, "ahi": ahi,
               "k0b": k0b, "dbits": dbits, "btgt": btgt, "zres": zres, "bn_s": sdig,
               "bn_d": ddig, "bn_prod": bprod, "bn_dig": bd_dig, "bn_car": bd_car, "q2": q2}
    c.n_plain = n_plain
    return c


# --- snapshots (the state the host injects) ---------------------------------------------------


def reg_row(c: RvCore, value: int) -> np.ndarray:
    row = np.zeros(HD)
    row[list(REF_COLS)] = A_REF
    for i in range(32):
        if value >> i & 1:
            row[c.reg_cols[i][0]] = A_BIT
    return row


def counter_rows(k: int) -> dict:
    r0 = np.zeros(HD)
    r0[2] = A_BIT * 2.0**-k
    r1 = np.zeros(HD)
    r1[list(REF_COLS)] = A_REF
    return {0: r0, 1: r1}


def byte_rows(words: np.ndarray, n_heads: int) -> list:
    """Per head, the two value heads' rows of a byte table of 32-bit words (word index: head,
    value head, row pair, half, word in row); every row carries the ballast."""
    per = len(words) // n_heads
    out = []
    for h in range(n_heads):
        vh = []
        for x in range(2):
            blk = words[h * per + x * (per // 2):h * per + (x + 1) * (per // 2)]
            rows = np.zeros((128, HD))
            rows[:, list(REF_COLS)] = A_REF
            wl = np.arange(len(blk))
            pp, half, k = wl >> 4, wl >> 3 & 1, wl & 7
            rr = pp + 64 * half
            for b in range(4):
                rows[rr, 2 + 32 * half + 4 * k + b] = blk >> (8 * b) & 0xFF
            vh.append({i: rows[i] for i in range(128)})
        out.append(vh)
    return out


_ROM_CACHE: dict = {}


def snapshot(c: RvCore, m: "RF.Machine") -> None:
    """Load a machine's state into the circuit's tables (the heads' initial states), as at the
    start of a record (j = 0)."""
    pc = np.zeros(HD)
    pc[list(REF_COLS)] = A_REF
    for i in range(16):
        pc[2 + i] = A_BIT * (m.anchor >> i & 1)
    c.tables["pc"].init = [{0: pc}]
    c.tables["count"].init = [counter_rows(m.n)]
    c.tables["j"].init = [counter_rows(0)]
    c.tables["vcnt"].init = [counter_rows(0)]
    regs = {r: reg_row(c, m.x[r]) for r in range(1, 32)}
    for p in ("rs1", "rs2", "rd"):
        c.tables[f"regs.{p}"].init = [regs]
    quo = np.zeros(HD)
    quo[list(REF_COLS)] = A_REF
    for h in range(4):
        c.tables[f"quo{h}"].init = [{0: quo.copy()} for _ in range(2)]
    key = (hash(tuple(m.rom)), hash(m.drom), hash(m.wrom))  # (one firmware's rows share them)
    if _ROM_CACHE.get("key") != key:
        rom = np.array([RF.rom_word(wd, S.ROM_BASE + 4 * i) if wd else 0
                        for i, wd in enumerate(m.rom)], dtype=np.int64)
        drom = np.frombuffer(m.drom.ljust(4 * S.DROM_WORDS, bytes(1)), dtype="<u4").astype(np.int64)
        _ROM_CACHE.update(key=key, rom=byte_rows(rom, S.ROM_WORDS >> 11),
                          drom=byte_rows(drom, S.DROM_WORDS >> 11))
    for h, vh in enumerate(_ROM_CACHE["rom"][:sum(n.startswith("rom") for n in c.tables)]):
        c.tables[f"rom{h}"].init = vh
    for h, vh in enumerate(_ROM_CACHE["drom"]):
        c.tables[f"drom{h}"].init = vh
    for port in (1, 2):
        for h in range(4):
            vh = []
            for x in range(2):
                rows = {}
                for r, dg in enumerate(m.bn[2 * h + x]):
                    row = np.zeros(HD)
                    row[list(REF_COLS)] = A_REF
                    for b in range(4):
                        row[2 + b] = A_BIT * (dg >> b & 1)
                    rows[r] = row
                vh.append(rows)
            c.tables[f"bn{port}.{h}"].init = vh
    for h in range(S.RAM_WORDS >> 8):
        r0, r1 = {}, {}
        for wi in range(S.RAM_WORDS):
            if (wi & 15 | (wi >> 12) << 4) != h or not m.ram[wi]:
                continue
            row = np.zeros(HD)
            row[list(REF_COLS)] = A_REF
            for i in range(32):
                row[2 + i] = A_BIT * (m.ram[wi] >> i & 1)
            (r1 if wi >> 4 & 1 else r0)[wi >> 5 & 127] = row
        c.tables[f"ram{h}"].init = [r0, r1]
    if getattr(c, "vx", None):
        vxcirc.snapshot(c, m)


def decode_code(fid: int, p: int, text=None) -> int:
    """The token of an output code (family id, payload), -1 if it names none; a Q code is the
    text token of ``text``'s printable token p."""
    if not 0 <= fid < len(OUT_NAMES):
        return -1
    nm = OUT_NAMES[fid]
    if nm == "Q":
        q = text.printable[p] if text and p < len(text.printable) else None
        return -1 if q is None else S.TEXT_TOKEN + q
    if nm in S.CONTROLS:
        return S.tok(nm)
    return S.tok(nm, p) if p < S.FAMILIES[nm][1] else -1


def predictions(c: RvCore, mat: np.ndarray) -> list[int]:
    """The predicted next token at every position (from the output code dims)."""
    def bits(es):
        return [np.rint(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()))
                if e.terms else np.zeros(mat.shape[0]) for e in es]

    if c.out_fam is None:  # merged: the exit's code dims (KAPPA * (2 bit - 1) where it owns)
        cb = [(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()) > 0).astype(int)
              for e in c.code]
        fb, pb = cb[:NFAM], cb[NFAM:]
    else:
        fb, pb = bits(c.out_fam), bits(c.out_pay)
    out = []
    for t in range(mat.shape[0]):
        fid = int(sum(int(fb[i][t]) << i for i in range(NFAM)))
        p = int(sum(int(pb[i][t]) << i for i in range(NPAY)))
        out.append(decode_code(fid, p, c.text))
    return out
