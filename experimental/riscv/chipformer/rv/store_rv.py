"""The object store (store.py) as part of the RV circuit: ``core.build(text=T, store=Config())``.

``capture`` (inside a tool call, as Qwen writes it): the op, the parameters and their values
(token features of Qwen's rows: their 18 id bits and a few token detectors, previous tokens
through conv taps), the name buffer, the content heads' writes, and ``S``: 1 from a store
call's ``</tool_call>`` to the last NL of its answer (where Qwen takes over).

``respond`` (the answer): the store's records and its printed ``<tool_response>``. Its cases
join the core's output cases (gated by S instead of the core's owner); prints of arbitrary
tokens ("any-prints") go out through the id code instead of the class code.

Tables (Gated DeltaNet heads with one-hot keys: an exact overwrite of one row, no decay):

  dir      slot -> (name 6 x 18 bits, length 3, alive, used, head chunk) | tail chunk
           (8 key heads x 2 value heads at 1024 slots)
  order    order index -> slot (value head = index bit 0)
  len      chunk -> text length (value head = chunk bit 0)
  next     chunk -> next chunk, has next
  content  rows of 8 tokens, 13 rows per chunk packed back to back: a key head's 2 value heads
           hold 4 tokens each, written token by token with conv taps (lags 0..3)
  regs     NEXTG, NORD, GCUR, LCUR (one row each, a column block each, read all at once)

Addresses are sets of 2-bit one-hot groups: a key or query channel is the sum of its group
literals minus (n - 1), exactly 1 on its own row and <= 0 elsewhere. The SP payload's groups
are token features, so a probe reads its slot one sublayer in, and the writes after it (SN, SA,
SD, SO) take that slot from the previous tokens through conv taps.
"""

from fractions import Fraction

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Chan, K
from chipformer.q35.prims import from_bits, sel, thead
from chipformer.rv import spec as S
from chipformer.rv import store as ST

CODE_AMP = 128  # the id code in a row's output dims: CODE_AMP * (2 * bit - 1)
L1, L2, L3 = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
PAY_BITS = 10
XS = 26  # counters come out as x = 2^(XS - n) (see counter)


def match_form(lits):
    """1 when all literals are 1, <= 0 otherwise (a key / query channel or AND-form)."""
    return sum(lits, Expr()) - (len(lits) - 1)


def groups(c, bits, name):
    """2-bit one-hot groups of an integer's bits: groups[k][v] = [bits 2k..2k+1 == v]."""
    out = []
    for k in range(0, len(bits), 2):
        bb = bits[k:k + 2]
        oh = []
        for v in range(1 << len(bb)):
            lits = [b if v >> i & 1 else c.one - b for i, b in enumerate(bb)]
            oh.append(c.and_(lits, f"{name}.g{k // 2}=={v}"))
        out.append(oh)
    return out


def addr_lits(gr, value, skip=()):
    """The group literals of one address (groups of 2 bits, the last may have 1); bits in
    ``skip`` are left out (a group with one skipped bit gives a one-bit literal)."""
    nbits = 2 * len(gr) - (len(gr[-1]) == 2)
    assert value >> nbits == 0, (value, nbits)
    out = []
    for k, g in enumerate(gr):
        bits = [b for b in (2 * k, 2 * k + 1) if b < nbits]
        keep = [b for b in bits if b not in skip]
        if len(keep) == len(bits):
            out.append(g[value >> 2 * k & (len(g) - 1)])
        elif keep:
            out.append(bit_lit(gr, keep[0], value >> keep[0] & 1))
    return out


def bit_lit(gr, j, v):
    """[bit j == v] from 2-bit groups (a sum of one-hot entries)."""
    g = gr[j // 2]
    return sum((e for w, e in enumerate(g) if (w >> (j % 2) & 1) == v), Expr())


def lits_of(bits, v):
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def neg(terms):
    return [(-coef, a, b) for coef, a, b in terms]


def counter(c, name, reset, count, tiny=2.0**-118):
    """x = 2^(XS - n), n = tokens with count = 1 since the last reset (0 at the reset itself).

    The row written at a reset decays by half at each counted token; a ballast row is
    rewritten at every other token, so a read is normed by a constant. 2^-(n + 14) is a normal
    float up to n = 112, and x^2 <= 2^52 vanishes next to every RMSNorm's ballast."""
    one = c.one
    q = {0: one - reset, 1: one - reset}
    k = {0: reset, 1: one - reset}
    v = {2: Chan(reset * A_BIT)}
    for col in REF_COLS:
        v[col] = Chan((one - reset) * A_REF)
    (x,) = c.head(name, {s: sel(c, e) for s, e in q.items()}, {s: sel(c, e) for s, e in k.items()},
                  v, 110 * count + 238 * reset, {2: K * one}, [(2, 2.0**(7 + XS))], tiny=tiny)
    c.big.add(next(iter(x.terms)))  # up to 2^XS: see emit.allocate
    return x


def ge_terms(c, x, reset, m, a=None):
    """Unit terms of a * [n >= m] (m >= 1) from a counter's x = 2^(XS - n); a defaults to
    1 - reset (x is 0 at reset tokens)."""
    one = c.one
    k = 8 * Fraction(2) ** (m - 1 - XS)
    a = one - reset if a is None else a
    return [(1, a, 7 * one - k * x), (-1, a, 6 * one - k * x)]


def count_bits(c, x, reset, nmax, nbits, name):
    """Bits of max(min(n, nmax) - 1, 0)."""
    out = []
    for b in range(nbits):
        terms = []
        for v in range(2, nmax + 1):
            d = ((v - 1) >> b & 1) - ((v - 2) >> b & 1)
            if d:
                terms += [(d * co, a_, b_) for co, a_, b_ in ge_terms(c, x, reset, v)]
        out.append(c.node(terms, f"{name}.b{b}"))
    return out


def count_bits_n(c, x, reset, nmax, nbits, name):
    """Bits of min(n, nmax)."""
    out = []
    for b in range(nbits):
        terms = []
        for v in range(1, nmax + 1):
            d = (v >> b & 1) - ((v - 1) >> b & 1)
            if d:
                terms += [(d * co, a_, b_) for co, a_, b_ in ge_terms(c, x, reset, v)]
        out.append(c.node(terms, f"{name}.b{b}"))
    return out


def inc_bits(c, bits, name, gate=None):
    """Bits of (v + 1) mod 2^n, AND gate (a 0/1 literal) when given."""
    g = [] if gate is None else [gate]
    out = [c.node(c.t_and(g + [c.one - bits[0]]), f"{name}.b0")]
    for i in range(1, len(bits)):
        out.append(c.node(c.t_and(g + [bits[i]]) + c.t_and(g + bits[:i])
                          + c.t_and(g + [bits[i], *bits[:i]], -2), f"{name}.b{i}"))
    return out


def eq_bits(c, a_bits, b_bits, name):
    """[a == b] for two bit vectors (two MLP sublayers)."""
    one = c.one
    mism = []
    for a_, b_ in zip(a_bits, b_bits):
        mism += [(1, one, a_ - b_), (1, one, b_ - a_)]
    n = c.node(mism, f"{name}.mism")
    return c.node(c.t_eq(n, 0, 0, len(a_bits)), name)


class Store:
    """Handles of the store's circuit."""


def build(c, cfg: "ST.Config", in_call):
    """The store's capture and answer. Returns a Store with the output cases (``cases``: core
    format, gated by S), the any-print cases, S, the call flag and the id code dims."""
    one = c.one
    f = c.token_feature
    T = c.text
    qid = ST.qid_of
    st = Store()
    st.cfg = cfg
    st.persistent = {}
    nlset = {q for q in range(T.n) if b"\n" in T.bytes[q]}
    st.nlset = nlset

    # ------------------------------------------------------------------ token features
    def fq(name, pred):
        return f(name, lambda t: int((q := qid(t)) is not None and pred(q)), full=True)

    def fid(name, i):
        return fq(name, lambda q, i=i: q == i)

    st.xb = [f(f"st.id.b{i}", lambda t, i=i: ST.id_bits(qid(t))[i], full=True)
             for i in range(ST.ID_BITS)]
    st.yc = [f(f"st.id.code{i}", lambda t, i=i: 0 if (q := qid(t)) is None or q >= ST.SPECIAL
               else CODE_AMP * (2 * (q >> i & 1) - 1), full=True) for i in range(ST.ID_BITS)]
    st.row_features = list(st.xb) + list(st.yc)  # features that read a text row's own id
    f_nl = fq("st.nl", lambda q: q in nlset)
    f_eq, f_fn = fid("st.eq", ST.Q_EQ), fid("st.function", ST.Q_FUNCTION)
    f_par, f_lts = fid("st.parameter", ST.Q_PARAMETER), fid("st.lt_slash", ST.Q_LTSLASH)
    f_gt = fid("st.gt", ST.Q_GT)
    f_trc, f_tho, f_thc = (fid(f"st.{n}", q) for n, q in (("trc", ST.Q_TRC), ("tho", ST.Q_THO),
                                                           ("thc", ST.Q_THC)))
    op_bare = {op: fid(f"st.op.{op}", i) for op, i in ST.OP_BARE.items()}
    op_join = {op: fid(f"st.op={op}", i) for op, i in ST.OP_JOINED.items()}
    p_bare = {p: fid(f"st.p.{p}", i) for p, i in ST.PARAM_BARE.items()}
    p_join = {p: fid(f"st.p={p}", i) for p, i in ST.PARAM_JOINED.items()}
    st.row_features += [f_nl, f_eq, f_fn, f_par, f_lts, f_gt, f_trc, f_tho, f_thc,
                        *op_bare.values(), *op_join.values(), *p_bare.values(), *p_join.values()]

    def sfam(t):
        return S.family(t)[0] if t < S.VOCAB else None

    isF = {n: f(f"st.is{n}", lambda t, n=n: int(sfam(t) == n), full=True)
           for n in S.STORE_FAMILIES}
    is_st = sum(isF.values(), Expr())  # a store record

    def spay(t):
        return S.family(t)[1] if sfam(t) in S.STORE_FAMILIES else 0

    def is_store_tok(t):
        return sfam(t) in S.STORE_FAMILIES

    pay = [f(f"st.pay.b{b}", lambda t, b=b: spay(t) >> b & 1, full=True) for b in range(PAY_BITS)]
    pg = [[f(f"st.pay.g{k}=={v}", lambda t, k=k, v=v: int(is_store_tok(t)
                                                          and spay(t) >> 2 * k & 3 == v), full=True)
           for v in range(4)] for k in range(PAY_BITS // 2)]
    isSP, isSN, isSA, isSO, isSG, isSL, isSD = (isF[n] for n in ("SP", "SN", "SA", "SO", "SG",
                                                                  "SL", "SD"))
    tco, tcc, is_text = c.isTCO, c.isTCC, c.is_text

    # ------------------------------------------------------------------ previous tokens
    lag = {"eq1": (f_eq, L1), "eq2": (f_eq, L2), "fn2": (f_fn, L2), "fn3": (f_fn, L3),
           "par1": (f_par, L1), "par2": (f_par, L2), "lts1": (f_lts, L1), "nl1": (f_nl, L1),
           "trc1": (f_trc, L1), "tho1": (f_tho, L1), "tho2": (f_tho, L2), "thc1": (f_thc, L1),
           "thc2": (f_thc, L2)}
    for op in ST.OPS:
        lag[f"opb1.{op}"] = (op_bare[op], L1)
        if op in op_join:
            lag[f"opj1.{op}"] = (op_join[op], L1)
    names = list(lag)
    outs = c.slot_head("st.prev", {0: one}, {0: one},
                       {2 + i: Chan(lag[n][0], lag[n][1]) for i, n in enumerate(names)}, one)
    pv = dict(zip(names, outs))

    # ------------------------------------------------------------------ inside a tool call
    op_t = {}
    for op in ST.OPS:
        terms = c.t_and([f_gt, pv[f"opb1.{op}"], pv["eq2"], pv["fn3"], in_call])
        if op in op_join:
            terms += c.t_and([f_gt, pv[f"opj1.{op}"], pv["fn2"], in_call])
        op_t[op] = c.node(terms, f"st.op_tok.{op}")
    op_any = sum(op_t.values(), Expr())
    pn, pt = (c.node(c.t_and([p_join[p], pv["par1"], in_call])
                     + c.t_and([p_bare[p], pv["eq1"], pv["par2"], in_call]), f"st.param.{p}")
              for p in ("name", "text"))
    closer = c.and_([f_par, pv["lts1"], in_call], "st.closer")
    fin = c.and_([f_nl, pv["nl1"], pv["thc2"]], "st.final_nl")
    ops = c.slot_head("st.op", {0: tco + op_any}, {0: one},
                      {2 + i: op_t[op] for i, op in enumerate(ST.OPS)}, tco + op_any)
    op = dict(zip(ST.OPS, ops))
    st.op = op
    op_set = sum(ops, Expr())
    pk_name, pk_text = c.slot_head("st.pk", {0: tco + pn + pt + closer}, {0: one},
                                   {2: pn, 3: pt}, tco + pn + pt + closer)
    # a newline token since the parameter's name (written at the token after it)
    ((hdr,),) = thead(c, "st.hdr", {0: Chan(f_nl, L1)}, {0: one},
                      [{"data": {2: one}, "ref": one, "a": 238 * (pn + pt)}])
    val_name = c.and_([pk_name, hdr, in_call], "st.val.name")
    val_text = c.and_([pk_text, hdr, in_call], "st.val.text")

    # the name: up to NMAX tokens in a buffer (one column block each, all read at once), and its
    # length (with the closer's "</")
    nreset = tco + pn
    x_ni = counter(c, "st.ni", nreset, val_name)
    ni = [c.node(ge_terms(c, x_ni, nreset, n + 1) + neg(ge_terms(c, x_ni, nreset, n + 2)),
                 f"st.ni=={n}") for n in range(ST.NMAX)]
    st.nlen = count_bits(c, x_ni, nreset, ST.NMAX + 2, ST.LEN_BITS, "st.nlen")
    data = {2 + ST.ID_BITS * n + b: match_form([st.xb[b], ni[n]])
            for n in range(ST.NMAX) for b in range(ST.ID_BITS)}
    keys = {n: match_form([val_name, ni[n]]) for n in range(ST.NMAX)}
    keys[127] = nreset
    # name rows carry no ballast; the reset writes row 127 with it (and clears the rest)
    nbuf = thead(c, "st.nbuf", keys, {s: one for s in keys},
                 [{"data": data, "ref": Chan(nreset), "a": 238 * nreset}])[0]
    c.nodes[-1].qhot = len(keys)  # (a multi-hot read: the executors' top-k)
    st.nbuf = [nbuf[ST.ID_BITS * n:ST.ID_BITS * (n + 1)] for n in range(ST.NMAX)]

    # the text: each value token's index, its length
    treset = tco + pt
    x_ci = counter(c, "st.ci", treset, val_text)
    st.tlen = count_bits(c, x_ci, treset, ST.CMAX + 1, 7, "st.tlen")  # max(n - 1, 0) after it

    def at_index(pred, name):
        """val_text * [the token's index i < CMAX has pred(i)] (n = i + 1)."""
        terms = []
        for i in range(ST.CMAX):
            if pred(i):
                terms += ge_terms(c, x_ci, treset, i + 1, a=val_text)
                terms += neg(ge_terms(c, x_ci, treset, i + 2, a=val_text))
        return c.node(terms, name)

    W = ST.TPR // 2  # tokens per value head row: 7
    slot = [at_index(lambda i, k=k: i % W == k, f"st.slot=={k}") for k in range(W)]
    half1 = at_index(lambda i: i // W % 2 == 1, "st.half")
    crow = [at_index(lambda i, r=r: i // ST.TPR == r, f"st.crow=={r}")
            for r in range(ST.ROWS_PER_CHUNK)]
    content = sum(slot, Expr())  # a text token with index < CMAX
    # the current value head row so far (tokens i - i % 7 .. i), one block each: row k holds
    # the token of slot k; row 0 carries the ballast, and its write clears the others
    gdata = {2 + ST.ID_BITS * k + b: match_form([st.xb[b], slot[k]])
             for k in range(W) for b in range(ST.ID_BITS)}
    gbuf = thead(c, "st.gbuf", {k: slot[k] for k in range(W)}, {k: one for k in range(W)},
                 [{"data": gdata, "ref": slot[0], "a": 238 * slot[0]}])[0]
    c.nodes[-1].qhot = W

    # ------------------------------------------------------------------ registers
    # NEXTG (next free chunk), NORD (order entries), GCUR (a get's chunk), LCUR (a list's
    # index). Their rows carry no ballast; row 127 has it, rewritten at every text token.
    regs = {"NEXTG": (isSN + isSA, inc_bits(c, pay, "st.nextg+1", isSN + isSA)),
            "NORD": (isSO, inc_bits(c, pay, "st.nord+1", isSO)),
            "GCUR": (isSG, [match_form([b, isSG]) for b in pay]),
            "LCUR": (isSL, [match_form([b, isSL]) for b in pay])}
    rkeys, rdata, blocks, col = {127: Chan(is_text)}, {}, {}, 2
    for slot, (rname, (wk, vals)) in enumerate(regs.items()):
        rkeys[slot] = wk
        blocks[rname] = list(range(col, col + PAY_BITS))
        for v in vals:
            rdata[col] = v
            col += 1
    out = thead(c, "st.regs", rkeys, {s: one for s in rkeys},
                [{"data": rdata, "ref": Chan(is_text)}])[0]
    c.nodes[-1].qhot = len(rkeys)
    by_col = dict(zip(sorted(rdata), out))
    reg = {rname: [by_col[k] for k in cols] for rname, cols in blocks.items()}
    st.reg = reg
    st.persistent["st.regs"] = c.nodes[-1]

    # ------------------------------------------------------------------ S and the flags
    store_tcc = c.and_([tcc, op_set], "st.call_end")
    (Sx,) = c.slot_head("st.S", {0: tcc + tco + fin}, {0: one}, {2: store_tcc}, tcc + tco + fin)
    st.S, st.call = Sx, op_set
    # rec: an SG / SL record since the call's end (a get's chain, a list's walk); tl: the
    # answer's </tool_response> was printed
    ((rec,), (tl,)) = thead(c, "st.flags", {0: tcc + isSG + isSL + f_trc}, {0: one},
                            [{"data": {2: isSG + isSL}, "ref": one, "beta": tcc + isSG + isSL},
                             {"data": {2: f_trc}, "ref": one, "beta": tcc + f_trc}])
    # H: text tokens since the call's end or the latest store record
    hreset = tcc + is_st
    x_h = counter(c, "st.H", hreset, is_text - tcc)
    hbits = count_bits_n(c, x_h, hreset, ST.CMAX + 2, 7, "st.H")
    hv = from_bits(hbits)
    hs = [c.node(ge_terms(c, x_h, hreset, k) + neg(ge_terms(c, x_h, hreset, k + 1)) if k else
                 [(1, one, one)] + neg(ge_terms(c, x_h, hreset, 1)), f"st.H=={k}")
          for k in range(8)]
    chain = c.and_([op["get"], rec], "st.chain")
    listing = c.and_([op["list"], rec], "st.listing")

    # ------------------------------------------------------------------ order, dir
    nbits_n = cfg.n_names.bit_length() - 1  # order index bits
    lg = groups(c, reg["LCUR"][:max(nbits_n, 1)], "st.lcur")
    o_rows = None
    n_oh = max(cfg.n_names // 256, 1)
    for h in range(n_oh):
        keys, qs = {}, {}
        for r in range(128):
            n_ = 256 * h + 2 * r
            if n_ >= cfg.n_names:
                continue
            keys[r] = match_form([isSO, *addr_lits(pg, n_, skip=(0,))])
            qs[r] = match_form([listing, *addr_lits(lg, n_, skip=(0,))])
        vh = [{"data": {2 + b: Chan(pay[b], L2) for b in range(cfg.slot_bits)}, "ref": one,
               "beta": isSO + bit_lit(pg, 0, x) - 1, "gate": bit_lit(lg, 0, x),
               "into": o_rows if x == 0 else (o_rows or "main")} for x in range(2)]
        o = thead(c, f"st.order{h}", keys, qs, vh)
        o_rows = o_rows or o[0]
        st.persistent[f"st.order{h}"] = c.nodes[-1]
    o_slot = o_rows  # the listed entry's slot

    # the dir's query: the probe's slot (SP payload), or the listed entry's
    sg = [[c.node(c.t_and([isSP, pg[k][v]])
                  + c.t_and([listing, *lits_of(o_slot[2 * k:2 * k + 2], v)]), f"st.q.g{k}=={v}")
           for v in range(4)] for k in range(cfg.slot_bits // 2)]
    vcols = {}
    col = 2
    for n in range(ST.NMAX):
        for b in range(ST.ID_BITS):
            vcols[col] = st.nbuf[n][b]
            col += 1
    for b in range(ST.LEN_BITS):
        vcols[col] = st.nlen[b]
        col += 1
    alive_col, used_col = col, col + 1
    vcols[alive_col], vcols[used_col] = isSN, one
    col += 2
    head_cols = list(range(col, col + PAY_BITS))
    for b in range(PAY_BITS):
        vcols[col] = pay[b]
        col += 1
    d0 = d1 = None
    for h in range(max(cfg.n_slots // 128, 1)):
        keys, qs = {}, {}
        for r in range(min(128, cfg.n_slots)):
            s = 128 * h + r
            keys[r] = Chan(match_form([isSP, *addr_lits(pg, s)]), L1)
            qs[r] = match_form(addr_lits(sg, s))
        o = thead(c, f"st.dir{h}", keys, qs,
                  [{"data": vcols, "ref": one, "beta": isSN + isSD, "into": d0},
                   {"data": {2 + b: pay[b] for b in range(PAY_BITS)}, "ref": one,
                    "beta": isSN + isSA, "into": d1}])
        d0, d1 = d0 or o[0], d1 or o[1]
        st.persistent[f"st.dir{h}"] = c.nodes[-1]
    cols0 = sorted(vcols)
    dv = dict(zip(cols0, d0))
    d_name = [[dv[2 + ST.ID_BITS * n + b] for b in range(ST.ID_BITS)] for n in range(ST.NMAX)]
    d_len = [dv[2 + ST.ID_BITS * ST.NMAX + b] for b in range(ST.LEN_BITS)]
    d_alive, d_used = dv[alive_col], dv[used_col]
    d_head = [dv[cc] for cc in head_cols]
    d_tail = d1

    # ------------------------------------------------------------------ the probe
    mism = []
    for n in range(ST.NMAX):
        for b in range(ST.ID_BITS):
            a_, b_ = d_name[n][b], st.nbuf[n][b]
            mism += [(1, one, a_ - b_), (1, one, b_ - a_)]
    for b in range(ST.LEN_BITS):
        mism += [(1, one, d_len[b] - st.nlen[b]), (1, one, st.nlen[b] - d_len[b])]
    nb_ = ST.NMAX * ST.ID_BITS + ST.LEN_BITS
    n_mism = c.node(mism, "st.mismatch")
    match = c.node(c.t_eq(n_mism, 0, 0, nb_, lits=[d_used]), "st.match")
    found = c.node(c.t_eq(n_mism, 0, 0, nb_, lits=[d_used, d_alive]), "st.found")
    stop = one - d_used + match
    putapp = op["put"] + op["append"]
    ng_full = reg["NEXTG"][cfg.n_chunks.bit_length() - 1]
    no_full = reg["NORD"][cfg.n_names.bit_length() - 1]
    full = c.node(c.t_and([putapp, ng_full]) + c.t_and([putapp, one - ng_full, one - d_used,
                                                         no_full]), "st.full")
    # the hash of the name: each bit the parity of a fixed subset of its bits
    kbits = [st.nbuf[k // ST.ID_BITS][k % ST.ID_BITS] for k in range(ST.NMAX * ST.ID_BITS)]
    kbits += st.nlen
    hsh = []
    for j, row in enumerate(cfg.hash_matrix):
        sel_ = [kbits[k] for k in range(len(row)) if row[k]]
        s_ = sum(sel_, Expr())
        terms = []
        for th in range(1, len(sel_) + 1):
            terms += c.t_ge(s_, th, 0, len(sel_), 1 if th % 2 else -1)
        hsh.append(c.node(terms, f"st.hash.b{j}"))
    # the probe's result, for the tokens after it
    res_cols = {2: found, 3: full, 4: one - d_used}
    for b in range(PAY_BITS):
        res_cols[5 + b] = d_head[b]
    res = thead(c, "st.res", {0: isSP}, {0: one}, [{"data": res_cols, "ref": one}])[0]
    r_found, r_full, r_fresh, r_head = res[0], res[1], res[2], res[3:]

    # ------------------------------------------------------------------ len, next, content
    nbits_c = cfg.n_chunks.bit_length() - 1
    gcur_g = groups(c, reg["GCUR"], "st.gcur")
    l_rows = None
    for h in range(max(cfg.n_chunks // 256, 1)):
        keys, qs = {}, {}
        for r in range(128):
            g = 256 * h + 2 * r
            if g >= cfg.n_chunks:
                continue
            keys[r] = match_form([isSN + isSA, *addr_lits(pg, g, skip=(0,))])
            qs[r] = match_form([chain, *addr_lits(gcur_g, g, skip=(0,))])
        vh = [{"data": {2 + b: st.tlen[b] for b in range(7)}, "ref": one,
               "beta": isSN + isSA + bit_lit(pg, 0, x) - 1, "gate": bit_lit(gcur_g, 0, x),
               "into": l_rows if x == 0 else (l_rows or "main")} for x in range(2)]
        o = thead(c, f"st.len{h}", keys, qs, vh)
        l_rows = l_rows or o[0]
        st.persistent[f"st.len{h}"] = c.nodes[-1]
    c_len = l_rows
    tail_g = groups(c, d_tail, "st.tail")
    x_rows = None
    for h in range(-(-cfg.n_chunks // 128)):
        keys, qs = {}, {}
        for r in range(min(128, cfg.n_chunks)):
            g = 128 * h + r
            keys[r] = Chan(match_form(addr_lits(tail_g, g)), L1)
            qs[r] = match_form([chain, *addr_lits(gcur_g, g)])
        data = {2 + b: pay[b] for b in range(PAY_BITS)}
        data[2 + PAY_BITS] = one
        o = thead(c, f"st.next{h}", keys, qs,
                  [{"data": data, "ref": one, "beta": isSA, "into": x_rows}])[0]
        x_rows = x_rows or o
        st.persistent[f"st.next{h}"] = c.nodes[-1]
    c_next, c_has = x_rows[:PAY_BITS], x_rows[PAY_BITS]

    # content: row (g, i // 14) of each chunk's 8 rows, back to back; value head (i // 7) % 2
    # holds tokens 7 at a time. Written at every text token with the row so far (the staging
    # head), read at chunk GCUR, index H (a get's print): the output gates pick the token.
    gw = groups(c, reg["NEXTG"], "st.nextg")
    get_print = c.and_([Sx, chain], "st.get_print")

    def h_at(pred, name):
        """[H has pred(H)] for H < CMAX."""
        terms = [(1, one, one)] + neg(ge_terms(c, x_h, hreset, 1)) if pred(0) else []
        for v in range(1, ST.CMAX):
            if pred(v):
                terms += ge_terms(c, x_h, hreset, v) + neg(ge_terms(c, x_h, hreset, v + 1))
        return c.node(terms, name)

    hrow = [h_at(lambda v, r=r: v // ST.TPR == r, f"st.hrow=={r}")
            for r in range(ST.ROWS_PER_CHUNK)]
    hgate = [[h_at(lambda v, x=x, k=k: v // W % 2 == x and v % W == k, f"st.hgate{x}.{k}")
              for k in range(W)] for x in range(2)]
    c_tok = c.zeros(ST.ID_BITS, "st.ctok")
    n_rows = cfg.n_chunks * ST.ROWS_PER_CHUNK
    for h in range(-(-n_rows // 128)):
        keys, qs = {}, {}
        for r in range(128):
            R = 128 * h + r
            if R >= n_rows:
                break
            g, ir = divmod(R, ST.ROWS_PER_CHUNK)
            keys[r] = match_form([crow[ir], *addr_lits(gw, g)])
            qs[r] = match_form([get_print, hrow[ir], *addr_lits(gcur_g, g)])
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for k in range(W):
                for b in range(ST.ID_BITS):
                    col = 2 + ST.ID_BITS * k + b
                    v[col] = Chan(gbuf[col - 2] * A_BIT)
                    z[col] = K * hgate[x][k]
                    outs.append((col, 2.0**7))
                    into.append(c_tok[b])
            for rc in REF_COLS:
                v[rc] = Chan(one * A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                        "beta": half1 if x else content - half1})
        c.table(f"st.content{h}", qs, keys, vhs)
        st.persistent[f"st.content{h}"] = c.nodes[-1]

    # ------------------------------------------------------------------ comparisons
    lenv = from_bits(c_len)
    in_chunk = c.node(c.t_lt(hv - lenv, 0, -127, 127), "st.H<len")
    d_lenv = from_bits(d_len)
    h_lt_n = c.node(c.t_lt(hv, ST.NMAX, 0, 127), "st.H<NMAX")
    in_name = c.node(c.t_lt(hv - d_lenv, 0, -127, 127, lits=[d_alive, h_lt_n]), "st.name_tok")
    nord = reg["NORD"]
    nord0 = c.and_([one - b for b in nord], "st.nord==0")
    lcur1 = inc_bits(c, reg["LCUR"], "st.lcur+1")
    last = eq_bits(c, lcur1, nord, "st.list_last")
    sp1 = inc_bits(c, pay[:cfg.slot_bits], "st.sp+1")

    # ------------------------------------------------------------------ the answer
    TQ = c.text

    def q_print(qid_):
        return ("Q", TQ.printable_index(qid_))

    NL, TRO, TRC = q_print(ST.Q_NL), q_print(ST.Q_TRO), q_print(ST.Q_TRC)
    THO, THC = q_print(ST.Q_THO), q_print(ST.Q_THC)
    cases, anyp_cases = [], []

    def case(lits, out):
        fam_name, payload = out if isinstance(out, tuple) else (out, 0)
        if isinstance(payload, int):
            payload = [one if payload >> i & 1 else Expr() for i in range(16)]
        cases.append(([Sx, *lits], fam_name, payload))

    def anyc(lits, bits):
        anyp_cases.append(([Sx, *lits], bits))

    p_put, p_app, p_get, p_lst, p_del = (op[o_] for o_ in ST.OPS)
    nonlist = one - p_lst
    # the call's end
    case([tcc, nonlist], ("SP", hsh))
    case([tcc, p_lst], NL)
    # a probe
    sp = [isSP, nonlist]
    case([*sp, one - stop], ("SP", sp1))
    case([*sp, stop, putapp, full], NL)
    case([*sp, stop, p_put, one - full], ("SN", reg["NEXTG"]))
    case([*sp, stop, p_app, found, one - full], ("SA", reg["NEXTG"]))
    case([*sp, stop, p_app, one - found, one - full], ("SN", reg["NEXTG"]))
    case([*sp, stop, p_del, found], "SD")
    case([*sp, stop, p_del, one - found], NL)
    case([*sp, stop, p_get], NL)
    # the writes after it
    case([isSN, r_fresh], ("SO", nord))
    case([isSN, one - r_fresh], NL)
    case([isSA], NL)
    case([isSD], NL)
    case([isSO], NL)
    # the header, then the body
    hdr_l = [is_text, one - rec, one - tl]
    case([*hdr_l, hs[1]], TRO)
    case([*hdr_l, hs[2]], NL)
    case([*hdr_l, hs[3], p_get, r_found], ("SG", r_head))
    case([*hdr_l, hs[3], p_lst, one - nord0], ("SL", 0))
    # the messages (each condition a list of exclusive literal sets: no extra sublayer)
    msgs = [([[putapp, one - r_full], [p_del, r_found]], "ok"), ([[putapp, r_full]], "full"),
            ([[p_del, one - r_found], [p_get, one - r_found]], "not_found"),
            ([[p_lst, nord0]], "empty")]
    for conds, mname in msgs:
        toks = ST.MSG[mname]
        for m_ in conds:
            for i, q in enumerate(toks):
                anyc([*hdr_l, hs[3 + i], *m_], ST.id_bits(q))
            case([*hdr_l, hs[3 + len(toks)], *m_], NL)
            case([*hdr_l, hs[4 + len(toks)], *m_], TRC)
    # a get's chunks: content[GCUR][H] while H < length, then the next chunk or the end
    gp = [chain, one - tl]
    anyc([*gp, in_chunk], c_tok)
    case([*gp, one - in_chunk, c_has], ("SG", c_next))
    case([*gp, one - in_chunk, one - c_has], TRC)
    # a list: at SL(i) and each printed name token
    lp = [listing, one - tl]
    for n in range(ST.NMAX):
        anyc([*lp, in_name, hs[n]], d_name[n])
    case([*lp, one - in_name, one - last], ("SL", lcur1))
    case([*lp, one - in_name, last], TRC)
    # the tail: </tool_response> \n <think> \n \n </think> \n \n (Qwen follows the last \n)
    case([tl, f_trc], NL)
    case([tl, f_nl, pv["trc1"]], THO)
    case([tl, f_tho], NL)
    case([tl, f_nl, pv["tho1"]], NL)
    case([tl, f_nl, pv["nl1"], pv["tho2"]], THC)
    case([tl, f_thc], NL)
    case([tl, f_nl, pv["thc1"]], NL)

    # any-prints: (literals, the 18 id bits as ints or forms); the exit computes the id code
    # from them directly (core.build)
    st.cases, st.any_cases = cases, anyp_cases
    st.debug = dict(match=match, found=found, stop=stop, full=full, H=hbits, rec=rec, tl=tl,
                    hash=hsh, in_chunk=in_chunk, in_name=in_name, d_used=d_used, d_alive=d_alive,
                    d_tail=d_tail, d_head=d_head, c_len=c_len, c_has=c_has, c_next=c_next,
                    o_slot=o_slot, last=last, val_name=val_name, val_text=val_text,
                    content=content, op=op, nlen=st.nlen, tlen=st.tlen, S=Sx)
    for name, node in st.persistent.items():
        c.tables[name] = node
    return st


def persistent_heads(c) -> list[str]:
    """Names of the heads that hold the store's files and registers (a snapshot keeps exactly
    these)."""
    return list(c.store.persistent) if getattr(c, "store", None) else []
