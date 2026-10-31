"""The vector unit (vx.py, docs/vx.md) in the RV circuit: its state, memories, datapaths and exit
cases. core.build calls `build` once the register ports and the uops exist.

Layout (docs/vx.md):
  L1   attn  config: 8 slots of 12 bits, a single-row value head each, written at a VC token
             whose payload names the slot; the element counters vel (VH tokens since the record's
             last VC) and vsub (VE tokens since the last VH or VC)
  L2   attn  VM: two banks (A = v0..v7, B = v8..v15), rows (register, pair) in 2 key heads, the
             even element in value head 0 and the odd in 1; a read is the pair holding the next
             token's element, a write is one element at its VH (VE bits from lag 1, VH bits 0-14
             from lag 0)
Element addresses are concatenations: an op's element offset is aligned to its length, so the
element = offset | vel, bit by bit.
"""

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Chan, K
from chipformer.q35.prims import chan, thead
from chipformer.rv import spec as S

ZERO = Expr()
OUT_W = 2.0**7  # a one-hot read's output weight (as core.OUT_W)


def bits_of(th, n_bits, top):
    """Binary bits of an integer k in 0..top from its thermometer th[m] = [k > m]."""
    out = []
    for i in range(n_bits):
        e = Expr()
        for v in range(1, top + 1):
            d = (v >> i & 1) - ((v - 1) >> i & 1)
            if d:
                e = e + d * th[v - 1]
        out.append(e)
    return out


def lits(bits, v):
    """0/1 literals, all 1 exactly when the bits spell v."""
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(ls):
    return sum(ls, Expr()) - (len(ls) - 1)


class VX:
    pass


def count_fn(c, x, f, top, name, a=None, lits_=()):
    """An L1 MLP node: a * f(k) for the count k = 0..top of the 2^-k counter x (thresholds on x,
    as RvCore.thermo), f 0/1-valued; a: a linear 0/1 form (default ONE), lits_: 0/1 literals
    ANDed in (held in the gates, so any mix of them is exact). At the counter's reset tokens
    (c.vx_resets) x is not 2^0, so the thresholds are held off there: k = 0."""
    one = c.one
    a = one if a is None else a
    rs = 16 * c.vx_resets[id(x)] + 16 * (len(lits_) - sum(lits_, Expr()))
    terms = (c.t_and(list(lits_)) if lits_ else [(1, a, one)]) if f(0) else []
    for v in range(1, top + 1):
        d = f(v) - f(v - 1)
        if d:
            m = v - 1  # [k > m] = [k >= v]
            terms += [(d, a, 7 * one - 8 * 2**m * x - rs), (-d, a, 6 * one - 8 * 2**m * x - rs)]
    return c.node(terms, name)


def count_bit(c, x, b, top, name, base=None, shift=0):
    """base OR bit b of (k - shift) (0 where k < shift), as one L1 MLP node; base: a 0/1 dim or
    None."""
    f = lambda k: (k - shift) >> b & 1 if k >= shift else 0  # noqa: E731
    if base is None:
        return count_fn(c, x, f, top, name)
    node = count_fn(c, x, f, top, name, a=c.one - base)
    return node + base


def build(c, one, fam, pay, uops, A, B, C, jt, jeq, nr, rec_start, case, rdw_any, ok):
    """Builds the unit and adds its exit cases; returns its state (for snapshots and probes)."""
    u = VX()
    # ---------------------------------------------------------------- L1: config, counters
    vcp = pay["VC"]
    slot_bits = vcp[12:15]
    cfg = []
    for kh in range(4):
        outs = thead(c, f"vx.cfg{kh}", {0: one}, {0: one},
                     [{"data": {2 + b: vcp[b] for b in range(12)}, "ref": one,
                       "beta": andpre([fam["VC"]] + lits(slot_bits, 2 * kh + x))}
                      for x in range(2)])
        c.tables[f"vx.cfg{kh}"] = c.nodes[-1]
        cfg += outs
    u.cfg = cfg  # cfg[slot][bit]
    isVC, isVE, isVH = fam["VC"], fam["VE"], fam["VH"]
    rel = isVC + rec_start
    xel = c.counter("vx.el", rel, isVH)
    rsub = isVC + isVH + rec_start
    xsub = c.counter("vx.sub", rsub, isVE)
    velt = c.thermo(xel, rel, 64, "vx.el")
    vsubt = c.thermo(xsub, rsub, 8, "vx.sub")
    c.vx_resets = {id(xel): rel, id(xsub): rsub}
    vel = bits_of(velt, 6, 64)  # the next token's element
    u.velt, u.vsubt, u.vel = velt, vsubt, vel

    def el_eq(k):
        return (one if k == 0 else velt[k - 1]) - velt[k]

    def sub_eq(k):
        return (one if k == 0 else vsubt[k - 1]) - vsubt[k]

    u.el_eq, u.sub_eq = el_eq, sub_eq

    # config fields (slot 0 = dst 4, element offset / 2 5, write target 3; slot 1 = bank A
    # register 3, bank B register 3)
    s0, s1 = cfg[0], cfg[1]
    dst, hoff, wt = s0[0:4], s0[4:9], s0[9:12]
    reg_a, reg_b = s1[0:3], s1[3:6]
    # the element's pair index (offset / 2 | vel >> 1) and parity, as exact 0/1 nodes (L1 MLP); at
    # a VH, the element it completes (vel - 1)
    pair = [count_bit(c, xel, i + 1, 64, f"vx.pair{i}", hoff[i]) for i in range(5)]
    parity = count_bit(c, xel, 0, 64, "vx.par")
    pairw = [count_bit(c, xel, i + 1, 64, f"vx.pairw{i}", hoff[i], shift=1) for i in range(5)]
    parw = count_bit(c, xel, 0, 64, "vx.parw", shift=1)
    u.pair, u.parity = pair, parity

    def w_is(v):
        return lits(wt, v)

    # ---------------------------------------------------------------- L2: VM
    vm_read = {}
    for bank, rq in (("A", reg_a), ("B", reg_b)):
        dims = [c.zeros(31, f"vx.vm{bank}.e"), c.zeros(31, f"vx.vm{bank}.o")]
        for kh in range(2):
            q, keys = {}, {}
            for r in range(8):
                for p in range(16):
                    row = 16 * r + p
                    q[row] = andpre(lits(rq, r) + lits(pair[:4], p) + lits(pair[4:5], kh))
                    keys[row] = Chan(andpre(lits(dst[:3], r) + lits(pair[:4], p)
                                            + lits(pair[4:5], kh)), (0, 1, 0, 0))
            vhs = []
            for x in range(2):
                data = {}
                for i in range(16):
                    data[2 + i] = Chan(pay["VE"][i], (0, 1, 0, 0))
                for i in range(15):
                    data[18 + i] = Chan(pay["VH"][i], (1, 0, 0, 0))
                # the element this VH completes: offset | vel - 1 (its pair's bit 4, parity)
                gate = ([isVH, wt[2], 1 - wt[1]] + lits(dst[3:4], int(bank == "B"))
                        + lits(pairw[4:5], kh) + lits([parw], x))
                vhs.append({"data": data, "ref": one, "beta": andpre(gate), "into": dims[x]})
            thead(c, f"vx.vm{bank}{kh}", keys, q, vhs)
            c.tables[f"vx.vm{bank}{kh}"] = c.nodes[-1]
        vm_read[bank] = dims
    u.vm_read = vm_read


    # ---------------------------------------------------------------- L2: VA
    # 6 chunks of 32 byte pairs; group g (bytes 8g..8g+7 of a chunk) is value head g % 2 of key
    # head g // 2, rows (chunk, pair j in the group), pair j in columns 2 + 16 j. A chunk read is
    # 4-hot in every key head. A write (VQ / VEXP, write target VA) is a pair at the odd byte's
    # VH: the even byte from lag 1, the odd from lag 0, each in its pair's columns (gated dims,
    # so a row writes only its own). VA byte offsets of a write are multiples of 8.
    s2 = cfg[2]
    voff, vchunk = s2[0:9], s2[6:9]
    k3 = s2[9]
    cstep = [count_fn(c, xsub, (lambda k, b=b: (k >> (b + 1)) & 1 if k < 6 else 0), 8,
                      f"vx.cstep{b}", a=k3) for b in range(2)]
    ech = [vchunk[0] + cstep[0], vchunk[1] + cstep[1], vchunk[2]]  # (k = 3 reads from chunk 0)
    pay8 = pay["VH"][0:8]
    gated = [[count_fn(c, xel, (lambda k, j=j: int(k >= 1 and (k - 1) >> 1 & 3 == j)), 64,
                       f"vx.vag{j}.{b}", a=pay8[b]) for b in range(8)] for j in range(4)]
    # the byte just completed (vel - 1): pair j in its group, the group, and odd
    bj = [count_bit(c, xel, b, 64, f"vx.vaj{b}", shift=1) for b in (1, 2)]
    bg = [count_bit(c, xel, b, 64, f"vx.vagr{b}", base=voff[b], shift=1) for b in (3, 4, 5)]
    bodd = count_bit(c, xel, 0, 64, "vx.vaodd", shift=1)
    va_read = []
    for kh in range(4):
        q, keys = {}, {}
        for ch in range(6):
            for j in range(4):
                q[4 * ch + j] = andpre(lits(ech, ch))
                keys[4 * ch + j] = Chan(andpre(lits(vchunk, ch) + lits(bj, j)), (1, 0, 0, 0))
        vhs = []
        for x in range(2):
            data = {}
            for j in range(4):
                for b in range(8):
                    data[2 + 16 * j + b] = Chan(gated[j][b], (0, 1, 0, 0))
                    data[10 + 16 * j + b] = Chan(gated[j][b], (1, 0, 0, 0))
            gate = [isVH] + w_is(2) + [bodd] + lits(bg, 2 * kh + x)
            vhs.append({"data": data, "ref": one, "beta": andpre(gate)})
        outs = thead(c, f"vx.va{kh}", keys, q, vhs)
        c.tables[f"vx.va{kh}"] = c.nodes[-1]
        c.nodes[-1].qhot = 4  # (a chunk read: 4 rows in every key head; the executors read top-k)
        for x in range(2):
            va_read += [outs[x][16 * j:16 * j + 16] for j in range(4)]  # 4 pairs of 16 bits
    # a 4-hot read returns each row at 1/4
    u.va = [[4 * d for d in bits] for pr in va_read for bits in (pr[0:8], pr[8:16])]  # 64 bytes

    s4, s5 = cfg[4], cfg[5]

    # ---------------------------------------------------------------- L2: trackers
    # stored from the payloads of a result's two tokens (VE bits 0-15 at lag 1, VH bits 0-14 at
    # lag 0): vsum at VRED's VH (write target 1); vmx at a flagged VH writing VM (targets 4 / 5,
    # bit 0 = signed, kept with it); vbest at a flagged VH writing nothing (VMAT's argmax), with
    # varg = the row's last row-chunk (base | vel - 1)
    word_v = {2 + i: Chan(pay["VE"][i], (0, 1, 0, 0)) for i in range(16)}
    word_v.update({18 + i: Chan(pay["VH"][i], (1, 0, 0, 0)) for i in range(15)})
    flag = pay["VH"][15]
    s4c, s5c = cfg[4], cfg[5]
    wbase = s4c[0:12] + s5c[0:1]
    varg = ([count_bit(c, xel, b, 64, f"vx.varg{b}", shift=1) for b in range(6)]
            + list(wbase))  # VMAT's argmax row: base << 6 | r
    trk = {}
    for name, extra, gate in (
            ("vsum", {}, [isVH, wt[0], 1 - wt[1], 1 - wt[2]]),
            ("vmx", {33: wt[0]}, [isVH, flag, wt[2], 1 - wt[1]]),
            ("vbest", {33 + b: varg[b] for b in range(19)}, [isVH, flag] + lits(wt, 0))):
        data = dict(word_v)
        data.update(extra)
        (outs,) = thead(c, f"vx.{name}", {0: one}, {0: one},
                        [{"data": data, "ref": one, "beta": andpre(gate)}])
        c.tables[f"vx.{name}"] = c.nodes[-1]
        trk[name] = outs
    u.trk = trk

    # ---------------------------------------------------------------- VRED
    # vsum = sum over i < n of VA chunk byte i, squared (mode 0) or as is (mode 1), by bit counts
    # over the lanes: with q = sum_b c_b q_b (c_b = 2^b, c_7 = -2^7), q^2 = sum_b c_b^2 q_b +
    # sum_{b < b'} 2 c_b c_b' q_b q_b'
    n_red = s2[11] + 2 * from_bits(s4c[0:6])
    mode1 = s2[10]
    lane = [c.node(c.t_ge(n_red, i + 1, 0, 127), f"vx.red.g{i}") for i in range(64)]
    cw = [1 << b for b in range(7)] + [-128]
    cols = [[] for _ in range(8)]
    for b in range(8):
        for b2 in range(b, 8):
            for md in ((0, 1) if b == b2 else (0,)):
                gate = [mode1] if md else [1 - mode1]
                cnt = c.node([t for i, q in enumerate(u.va)
                              for t in c.t_and([q[b], q[b2], lane[i]] + gate if b2 != b else
                                               [q[b], lane[i]] + gate)], f"vx.red.n{b}{b2}{md}")
                w = cw[b] if md else cw[b] * cw[b2] * (1 if b == b2 else 2)
                place_count(c, cols, cnt, 64, w, f"vx.red.s{b}{b2}{md}")
    red = sum_columns(c, cols, 8, "vx.red")
    u.red = red

    # ---------------------------------------------------------------- VGET
    # rd = vmx (|s|, or s when it was tracked signed) / vsum / varg / bit length of vmx, by rs1
    sv_bits = trk["vmx"][0:31]
    signed_t = trk["vmx"][31]
    neg = [1 - b for b in sv_bits] + [1 - sv_bits[30]]  # -s = ~s + 1 (32 bits, s sign-extended)
    from chipformer.rv import core as CORE
    negs = CORE.add_nibbles(c, neg, [one] + [ZERO] * 31, 8, [], "vx.get.neg")
    sgn = sv_bits[30]
    absv = [c.node(c.t_and([1 - sgn, sv_bits[min(i, 30)]]) + c.t_and([sgn, 1 - signed_t, negs[i]])
                   + c.t_and([sgn, signed_t, sv_bits[min(i, 30)]]), f"vx.get.abs{i}")
            for i in range(32)]
    th = [c.node(c.t_const(1) + c.t_and([1 - b for b in absv[m:31]], -1), f"vx.get.ge{m}")
          for m in range(31)]
    blen = c.int_bits(sum(th, Expr()), 31, "vx.get.bl", 5)
    srcs = [absv, list(trk["vsum"][0:31]) + [trk["vsum"][30]], list(trk["vbest"][31:50]),
            blen]
    gv = []
    for i in range(32):
        terms = []
        for sel, bl in enumerate(srcs):
            if i < len(bl):
                terms += c.t_and(lits(A[0:2], sel) + [bl[i]])
        gv.append(c.node(terms, f"vx.get.v{i}"))
    k0 = c.node(c.t_and(gv[14:32]) + c.t_and([1 - b for b in gv[14:32]]), "vx.get.fits15")
    k1 = c.node(c.t_and(gv[29:32]) + c.t_and([1 - b for b in gv[29:32]]), "vx.get.fits30")
    u.get = gv


    # ---------------------------------------------------------------- VMAT
    # A row's tokens: the running sum after each chunk (VE lo, VE hi), then out = acc * S (VE lo,
    # VH hi | the tracker flag). The WROM (L3) holds 3 row-chunks a row pair p; the op's row r
    # reads pair base + d (d = r // 3 or r, by an adder in L2's MLP: the pairs stay in one block
    # of 64), slot r % 3 (k = 1) or the chunk step (k = 3). The running sum and the scaling read
    # the earlier sums back through the lag head.
    (lagd,) = thead(c, "vx.lag", {0: one}, {0: one},
                    [{"data": {**{2 + i: Chan(pay["VE"][i], (0, 1, 0, 0)) for i in range(16)},
                               **{18 + i: Chan(pay["VE"][i], (0, 0, 1, 0)) for i in range(16)}},
                      "ref": one}])
    lag0, lag1, lag2 = list(pay["VE"][0:16]), lagd[0:16], lagd[16:32]
    s5v = cfg[5]
    n_vm = from_bits(s5v[1:8])
    mode_am, keep = s5v[8], s5v[9]
    # k = 1: chunk 3 base + start + r; the start slot by literals (slot 2 bits 10, 11: other ops
    # may set both, which then matches none)
    st = [[1 - k3, 1 - s2[10], 1 - s2[11]], [1 - k3, s2[10], 1 - s2[11]], [1 - k3, 1 - s2[10], s2[11]]]
    dsel = [sum((count_fn(c, xel, (lambda k, b=b, x=x: ((x + k) // 3) >> b & 1), 64,
                          f"vx.wd{x}.{b}", lits_=st[x]) for x in range(3)), Expr())
            + count_fn(c, xel, (lambda k, b=b: k >> b & 1), 64, f"vx.wd3{b}", a=k3)
            for b in range(6)]
    wp = c.int_bits(from_bits(wbase[0:6]) + from_bits(dsel), 63, "vx.wp", 6)
    slot = [sum((count_fn(c, xel, (lambda k, x=x, y=y: int((y + k) % 3 == x)), 64,
                          f"vx.wsl{y}.{x}", lits_=st[y]) for y in range(3)), Expr())
            + count_fn(c, xsub, (lambda k, x=x: int(min(k >> 1, 2) == x)), 8, f"vx.wsl3{x}", a=k3)
            for x in range(3)]
    wdims = c.zeros(34, "vx.wrom")
    for kh in range(S.VX_CHUNKS // 384):
        q = {}
        for p in range(64):
            q[p] = q[p + 64] = andpre(lits(wp, p) + lits(wbase[7:11], kh))
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for sl in range(3):  # slots 0, 1 in row p, 2 in row p + 64
                ze = K * andpre(lits([wbase[6]], x) + [slot[sl]])
                for i in range(34):
                    col = 2 + 34 * sl + i
                    z[col] = ze
                    outs.append((col, OUT_W))
                    into.append(wdims[i])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"vx.wrom{kh}", q, {}, vhs, levels="byte")
    wbyte = [256 * d for d in wdims]
    whi = [c.split(wbyte[i], 0, 255, f"vx.wu{i}") for i in range(34)]
    wlo = [wbyte[i] - 16 * whi[i] for i in range(34)]
    wts = [x_ for i in range(32) for x_ in (wlo[i] - 8, whi[i] - 8)]
    sn = [wlo[32], whi[32], wlo[33], whi[33]]  # the row scale's nibbles

    # the earlier 32-bit sum: lags (1, 0) at an even vsub, (2, 1) at an odd one
    odd = count_fn(c, xsub, lambda k: k & 1, 8, "vx.subodd")
    dprev = count_fn(c, xsub, lambda k: int(2 <= k < 6), 8, "vx.dprev", a=k3)
    l01, l12 = lag1 + lag0, lag2 + lag1
    pdot = [c.node(c.t_and([1 - odd, dprev, l01[b]]) + c.t_and([odd, dprev, l12[b]]),
                   f"vx.mat.pd{b}") for b in range(32)]
    pacc = [c.node(c.t_and([1 - odd, l01[b]]) + c.t_and([odd, l12[b]]), f"vx.mat.pa{b}")
            for b in range(32)]
    # the dot product, by bit planes (bitplane_cols), plus the earlier running sum
    cols = bitplane_cols(c, u.va, wts, "vx.mat.bp")
    for k_ in range(8):
        cols[k_].append((from_bits(pdot[4 * k_:4 * k_ + 4]), 0, 15))
    acc = sum_columns(c, cols, 8, "vx.mat.dot")
    # out = acc * S: acc as 20 bits signed (5 nibbles), S as 4 nibbles
    pcols = [[] for _ in range(8)]
    for i in range(5):
        for j in range(4):
            sg = i == 4
            pt = c.node([t for b in range(4) for t in c.t_and(
                [pacc[4 * i + b]], -8 if sg and b == 3 else 1 << b, a=sn[j])], f"vx.mat.m{i}{j}")
            pcols[i + j].append((pt, -120 if sg else 0, 105 if sg else 225))
    o = sum_columns(c, pcols, 8, "vx.mat.o")
    # the tracker flag: |o| > s (mode 0: o > s or ~o >= s, s the tracker's |value|, or its value
    # when tracked signed) or o > s (argmax, s = vbest); signed compares nibble by nibble
    scmp = [c.node(c.t_and([1 - mode_am, absv[b]]) + c.t_and([mode_am, trk["vbest"][min(b, 30)]]),
                   f"vx.mat.s{b}") for b in range(32)]
    cmp = compare(c, list(o[0:31]) + [o[30]], scmp, "vx.mat")
    better = [[cmp["o"]], [1 - cmp["o"], cmp["n"], 1 - mode_am]]
    done_m = c.node(c.t_eq(sum(velt, Expr()) - n_vm, 0, -127, 64), "vx.mat.done")
    u.mat = (acc, o)
    u.wrom = (wp, slot, wdims, cmp, scmp)


    # ---------------------------------------------------------------- RoPE pairs, SiLU table
    # RoPE (cos, sin) pairs: VPUT 512 + k (write target 3) at its second VH, cos from lag 3 and
    # sin from lag 1; read by the element's pair (i mod 8) / 2
    (rope,) = thead(c, "vx.rope",
                    {r: Chan(andpre(lits(hoff[0:2], r)), (0, 1, 0, 0)) for r in range(4)},
                    {r: andpre(lits(u.vel[1:3], r)) for r in range(4)},
                    [{"data": {**{2 + i: Chan(pay["VE"][i], (0, 0, 0, 1)) for i in range(16)},
                               **{18 + i: Chan(pay["VE"][i], (0, 1, 0, 0)) for i in range(16)}},
                      "ref": one, "beta": andpre([isVH, wt[0], wt[1], 1 - wt[2], el_eq(2)])}])
    c.tables["vx.rope"] = c.nodes[-1]

    def nib16(bits):
        """Nibble values of a 16-bit signed bit list (the top one signed)."""
        return [from_bits(bits[0:4]), from_bits(bits[4:8]), from_bits(bits[8:12]),
                from_bits(bits[12:15]) - 8 * bits[15]]

    cn, sn_ = nib16(rope[0:16]), nib16(rope[16:32])
    # SiLU(y), y from the lags (the GLU's y tokens): a table of SILU_T + 2048 (2 bytes an entry,
    # 4 entries a row pair) by idx = (y >> 11) + 256, one head per lag phase (vsub even / odd)
    ys = {0: l01, 1: l12}
    sdims = c.zeros(2, "vx.silu")
    for ph in range(2):
        yb = ys[ph]
        idx = yb[11:19] + [1 - yb[19]]
        gate = odd if ph else 1 - odd
        q = {}
        for p_ in range(64):
            q[p_] = q[p_ + 64] = andpre(lits(idx[2:8], p_) + [gate])
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for sl in range(4):
                ze = K * andpre(lits([idx[8]], x) + lits(idx[0:2], sl))
                for byte in range(2):
                    z[2 + 2 * sl + byte] = ze
                    outs.append((2 + 2 * sl + byte, OUT_W))
                    into.append(sdims[byte])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"vx.silu{ph}", q, {}, vhs, levels="byte")
    tb = [256 * d for d in sdims]
    tbh = [c.split(tb[i], 0, 255, f"vx.silu.u{i}") for i in range(2)]
    tn = [tb[0] - 16 * tbh[0], tbh[0], tb[1] - 16 * tbh[1] - 8, tbh[1]]  # SILU_T: -2048 in nibble 2
    # y's range: >= 2^19 (big: y >> 4, saturating at 2^23), < -2^19 (small: 0)
    yr = {}
    for nm, sg, lo_, hi_ in (("big", 0, 19, 31), ("sat", 0, 27, 31)):
        yr[nm] = c.node([t for ph in range(2) for t in
                         c.t_and([odd if ph else 1 - odd, 1 - ys[ph][31]])
                         + c.t_and([odd if ph else 1 - odd, 1 - ys[ph][31]]
                                   + [1 - b for b in ys[ph][lo_:hi_]], -1)], f"vx.silu.{nm}")
    yr["small"] = c.node([t for ph in range(2) for t in
                          c.t_and([odd if ph else 1 - odd, ys[ph][31]])
                          + c.t_and([odd if ph else 1 - odd] + ys[ph][19:32], -1)], "vx.silu.small")
    yn = [from_bits(pacc[4 + 4 * j:8 + 4 * j]) for j in range(6)]  # y >> 4, 24 bits

    # ---------------------------------------------------------------- the MAC: VQ, VEW
    # t = shift(W1 F1 + W2 F2 (+ VQ's rounding), sft), W = clamp16(VM element >> shamt):
    #   VQ        W1 = src, F1 = r, sft = -15 (+2^14, -1 when negative; then the clamp to +-127)
    #   ADD       W1 = b, F1 = p, sft = so                 GLU y   W1 = a, F1 = p, sft = so
    #   GLU t     W1 = b >> shb, F1 = SILU(y), sft = -8    ROPE    W1, W2 = the pair of a, F =
    #             (cos, -sin) / (sin, cos), sft = sh - 14
    s1, s6 = cfg[1], cfg[6]
    srcb = cfg[3][10]  # VQ's source bank
    pb = s4[0:12] + s5[0:3]
    shv, abank, opb, keepw = s5[3:8], s5[8], s5[9:11], s5[11]
    shb, sob = s1[6:11], s6[6:12]
    lt2 = 1 - vsubt[1]
    vewl = [wt[2], 1 - wt[1], 1 - wt[0]]
    cs = {"vq": [wt[1]], "add": vewl + lits(opb, 0), "gy": vewl + lits(opb, 1) + [lt2],
          "gt": vewl + lits(opb, 1) + [1 - lt2], "rope": vewl + lits(opb, 2)}
    bank = {"A": 0, "B": 1}
    # W1's source: (case, literal that is 1 for bank B, parity choice)
    src1 = [("vq", srcb, None), ("add", 1 - abank, None), ("gy", abank, None),
            ("gt", 1 - abank, None), ("rope", abank, 0)]

    def pick(i, cases):
        terms = []
        for cn_, bb, par in cases:
            for bk in ("A", "B"):
                for x in (range(2) if par is None else [par]):
                    pl = lits([parity], x) if par is None else []
                    terms += c.t_and(cs[cn_] + lits([bb], bank[bk]) + pl + [vm_read[bk][x][i]])
        return terms

    w1src = [c.node(pick(i, src1), f"vx.mac.s1.{i}") for i in range(31)]
    w2src = [c.node(pick(i, [("rope", abank, 1)]), f"vx.mac.s2.{i}") for i in range(31)]
    sham = [c.node(c.t_and([shv[b_]]) + c.t_and(cs["gt"] + [shv[b_]], -1) + c.t_and(cs["gt"] + [shb[b_]]),
                   f"vx.mac.sh{b_}") for b_ in range(5)]

    def shifted(src, amt, name):
        return [c.node([t for s_ in range(32) for t in c.t_and(lits(amt, s_) + [src[min(i + s_, 30)]])],
                       f"{name}{i}") for i in range(31)]

    z1 = shifted(w1src, sham, "vx.mac.z1.")
    z2 = shifted(w2src, shv, "vx.mac.z2.")
    # the factors, nibble by nibble (ranges in fr)
    pn = [from_bits(pb[4 * j:4 * j + 4]) for j in range(4)]
    f1, f2 = [], []
    for j in range(6):
        terms = []
        if j < 4:
            for cn_ in ("vq", "add", "gy"):
                terms += c.t_and(cs[cn_], a=pn[j])
            terms += c.t_and(cs["gt"] + [1 - yr["big"], 1 - yr["small"]], a=tn[j])
            terms += c.t_and(cs["rope"] + [1 - parity], a=cn[j])
            terms += c.t_and(cs["rope"] + [parity], a=sn_[j])
        terms += c.t_and(cs["gt"] + [yr["big"], 1 - yr["sat"]], a=yn[j])
        if j == 5:
            terms += c.t_and(cs["gt"] + [yr["sat"]], 8)
        f1.append(c.node(terms, f"vx.mac.f1.{j}"))
    for j in range(4):
        f2.append(c.node(c.t_and(cs["rope"] + [1 - parity], a=-1 * sn_[j])
                         + c.t_and(cs["rope"] + [parity], a=cn[j]), f"vx.mac.f2.{j}"))
    fr1, fr2 = (-8, 15), (-15, 15)
    # products: W bits (clamp16 folded in) times factor nibbles, grouped by weight 2^(4k + o)
    cl1, cl2 = clamp_terms(z1), clamp_terms(z2)
    cols = [[] for _ in range(10)]
    const = 0
    for k_ in range(10):
        for o_ in range(4):
            terms, lo_, hi_ = [], 0, 0
            for wl, fl, fr in ((cl1, f1, fr1), (cl2, f2, fr2)):
                for j, fj in enumerate(fl):
                    b_ = 4 * (k_ - j) + o_
                    if not 0 <= b_ < 16:
                        continue
                    sg = -1 if b_ == 15 else 1
                    for coef, ls in wl[b_]:
                        terms += c.t_and(ls, sg * coef, a=fj)
                    lo_ += min(0, sg * fr[0], sg * fr[1])
                    hi_ += max(0, sg * fr[0], sg * fr[1])
            if terms:
                g_ = c.node(terms, f"vx.mac.g{k_}.{o_}")
                const += place_signed(c, cols, g_, lo_, hi_, 4 * k_ + o_, f"vx.mac.c{k_}.{o_}")
    neg = c.node(c.t_and(cs["vq"] + [z1[30]]), "vx.mac.vqneg")
    cols[3].append((4 * wt[1], 0, 4))  # VQ: + 2^14
    cols[0].append((-1 * neg, -1, 0))  # VQ: -1 when negative
    const %= 1 << 40
    for k_ in range(10):
        v_ = const >> 4 * k_ & 15
        if v_:
            cols[k_].append((v_ * one, v_, v_))
    rr = sum_columns(c, cols, 10, "vx.mac.r")
    # the shift by sft (6 bits signed): by 8 q (q = sft >> 3), then by sft & 7
    sm14 = c.fn_bits(from_bits(shv) + 50, 50, 81, lambda v_: v_ % 64, 6, "vx.mac.sm14")
    sm14 = [c.node(t, f"vx.mac.sm14.{b_}") for b_, t in enumerate(sm14)]
    sft = []
    for b_ in range(6):
        terms = c.t_and(cs["add"] + [sob[b_]]) + c.t_and(cs["gy"] + [sob[b_]])
        terms += c.t_and(cs["rope"] + [sm14[b_]])
        if 49 >> b_ & 1:
            terms += c.t_and(cs["vq"])
        if 56 >> b_ & 1:
            terms += c.t_and(cs["gt"])
        sft.append(c.node(terms, f"vx.mac.sft{b_}"))
    ta = {}
    for m_ in range(-7, 32):
        terms = []
        for q_ in range(-4, 4):
            n_ = m_ - 8 * q_
            if n_ >= 0:
                terms += c.t_and(lits(sft[3:6], q_ % 8) + [rr[min(n_, 39)]])
        ta[m_] = c.node(terms, f"vx.mac.ta{m_}")
    tsh = [c.node([t for r_ in range(8) for t in c.t_and(lits(sft[0:3], r_) + [ta[i - r_]])],
                  f"vx.mac.t{i}") for i in range(32)]
    q8 = sat127_bits(c, tsh[0:16], "vx.vq.q")
    u.vq = q8
    # VEW's word: o = t (from the lags) + a (ADD), tracked |o|
    ad = [c.node(pick(i, [("add", abank, None)]), f"vx.ew.a{i}") for i in range(31)]
    ad32 = ad + [ad[30]]
    ocols = [[(from_bits(pacc[4 * k_:4 * k_ + 4]), 0, 15), (from_bits(ad32[4 * k_:4 * k_ + 4]), 0, 15)]
             for k_ in range(8)]
    vo = sum_columns(c, ocols, 8, "vx.ew.o")
    ecmp = compare(c, list(vo[0:31]) + [vo[30]], absv, "vx.ew")
    ebetter = [[ecmp["o"]], [1 - ecmp["o"], ecmp["n"]]]
    done_w = c.node(c.t_eq(sum(velt, Expr()) - 2 * from_bits(s6[0:6]), 0, -128, 64), "vx.ew.done")
    u.mac = (rr, tsh, vo)


    # ---------------------------------------------------------------- K / V caches
    # VQ with write target 6 (KC) / 7 (VCC) writes 4 bytes at once, at the VH of a byte i with
    # i % 4 == 3 (the bytes at lags 3..0). KC: key head l = off >> 7, row 32 g + slot (g = i >> 3),
    # half (i >> 2) & 1 in value head 0 / 1. VCC: one value head per slot (16 key heads), rows
    # (l, g, half), so p.V reads byte j of every slot at once.
    s3 = cfg[3]
    coff = s3[0:10]
    w4 = count_fn(c, xel, lambda k: int(k >= 1 and (k - 1) % 4 == 3), 64, "vx.c.w4")
    halfw = count_bit(c, xel, 2, 64, "vx.c.half", shift=1)
    gk = [count_bit(c, xel, b_, 64, f"vx.c.g{b_}") for b_ in (2, 3, 4)]  # i's bits at t - 1
    cdata = {2 + 8 * q_ + b_: Chan(pay["VH"][b_], tuple(int(lg == 3 - q_) for lg in range(4)))
             for q_ in range(4) for b_ in range(8)}
    s2b = cfg[2]
    mbase = s3[0:10]  # VMATR's base (slot 3)
    kdims = c.zeros(64, "vx.kc")
    for kh in range(5):
        keys = {32 * g_ + sl: Chan(andpre(lits(coff[7:10], kh) + lits(gk[1:3], g_) + lits(coff[0:5], sl)),
                                   (0, 1, 0, 0)) for g_ in range(4) for sl in range(32)}
        q = {32 * g_ + t_: andpre(lits(mbase[7:10], kh) + lits(mbase[5:7], g_) + lits(u.vel[0:5], t_))
             for g_ in range(4) for t_ in range(32)}
        vhs = []
        for x in range(2):
            beta = andpre([isVH, wt[1], wt[2], 1 - wt[0], w4] + lits([halfw], x))
            vhs.append({"data": dict(cdata), "ref": one, "beta": beta,
                        "into": kdims[32 * x:32 * x + 32]})
        thead(c, f"vx.kc{kh}", keys, q, vhs)
        c.tables[f"vx.kc{kh}"] = c.nodes[-1]
    pdims = [c.zeros(8, f"vx.vc.p{t_}") for t_ in range(32)]
    for kh in range(16):
        keys = {2 * lg + h: Chan(andpre(lits(coff[7:10], lg >> 2) + lits(gk[1:3], lg & 3) + lits(gk[0:1], h)),
                                 (0, 1, 0, 0)) for lg in range(20) for h in range(2)}
        q = {2 * lg + h: andpre(lits(mbase[5:10], lg) + lits(u.vel[2:3], h)) for lg in range(20)
             for h in range(2)}
        vhs = []
        for x in range(2):
            t_ = 2 * kh + x
            beta = andpre([isVH, wt[0], wt[1], wt[2], w4] + lits(coff[0:5], t_))
            z, outs, into = {}, [], []
            for q_ in range(4):
                ze = K * andpre(lits(u.vel[0:2], q_))
                for b_ in range(8):
                    z[2 + 8 * q_ + b_] = ze
                    outs.append((2 + 8 * q_ + b_, OUT_W))
                    into.append(pdims[t_][b_])
            v = {col: chan(e_, A_BIT) for col, e_ in cdata.items()}
            for col in REF_COLS:
                v[col] = chan(one, A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into, "beta": beta})
        c.table(f"vx.vc{kh}", q, keys, vhs)
    pv = pdims
    kv = [kdims[8 * i:8 * i + 8] for i in range(8)]

    # ---------------------------------------------------------------- VMATR
    # scores: VA chunk [:8] . K row (bit-pair counts); p.V: VA chunk [:n] . V column j (bit-pair
    # counts over the n lanes), then out = raw * s >> 12 from the lags
    s4m, s5m = cfg[4], cfg[5]
    mode_r = s3[10]
    n_r = from_bits(s4m[0:6])
    sfac = s4m[6:12] + s5m[0:11]  # s: 17 bits
    cw8 = [1 << b_ for b_ in range(7)] + [-128]

    def dot8(qs, ws, lane_lits, name):
        cols_ = [[] for _ in range(8)]
        for b_ in range(8):
            for b2 in range(8):
                cnt = c.node([t for i, (qa, wa) in enumerate(zip(qs, ws))
                              for t in c.t_and([qa[b_], wa[b2]] + lane_lits(i))], f"{name}.n{b_}{b2}")
                place_count(c, cols_, cnt, len(qs), cw8[b_] * cw8[b2], f"{name}.s{b_}{b2}")
        return cols_

    sc = sum_columns(c, dot8(u.va[0:8], kv, lambda i: [], "vx.mr.sc"), 8, "vx.mr.sc")
    rlane = [c.node(c.t_ge(n_r, t_ + 1, 0, 63), f"vx.mr.g{t_}") for t_ in range(32)]
    raw = sum_columns(c, dot8(u.va[0:32], pv, lambda i: [rlane[i]], "vx.mr.pv"), 8, "vx.mr.pv")
    mcols = [[] for _ in range(11)]
    sn5 = [from_bits(sfac[4 * j:4 * j + 4]) for j in range(5)]
    for i in range(5):
        for j in range(5):
            sg = i == 4
            pt = c.node([t for b_ in range(4) for t in c.t_and(
                [pacc[4 * i + b_]], -8 if sg and b_ == 3 else 1 << b_, a=sn5[j])], f"vx.mr.m{i}{j}")
            mcols[i + j].append((pt, -120 if sg else 0, 105 if sg else 225))
    ro = sum_columns(c, mcols, 11, "vx.mr.o")[12:44]
    rcmp0 = compare(c, list(sc[0:31]) + [sc[30]], absv, "vx.mr.c0")
    rcmp1 = compare(c, list(ro[0:31]) + [ro[30]], absv, "vx.mr.c1")
    done_r0 = c.node(c.t_eq(sum(velt, Expr()) - n_r - (n_r - 2 * from_bits(s4m[1:6])), 0, -128, 64),
                     "vx.mr.done0")
    pad_r = c.node(c.t_eq(sum(velt, Expr()) - n_r, 0, -64, 64, lits=[s4m[0]]), "vx.mr.pad")
    done_r1 = c.node(c.t_eq(sum(velt, Expr()), 8, 0, 64), "vx.mr.done1")

    # ---------------------------------------------------------------- VEXP
    # a pair's tokens: e, e' (VE: min(m - x, 65535)), then p, p' (VH: EXP(e c >> ksh), e from
    # lag 1)
    s6x, s7x = cfg[6], cfg[7]
    mb = s6x[2:12] + s7x[0:12] + s3[0:9]
    n_x = s2b[10] + 2 * s2b[11] + 4 * from_bits(s4m[0:5])
    kshb = s4m[5:10]
    cb = s4m[10:12] + s5m[0:12] + s6x[0:2]
    xb = cfg[1][6]
    xel_ = [c.node([t for bk in ("A", "B") for x in range(2)
                    for t in c.t_and(lits([xb], bank[bk]) + [sub_eq(x)] + [vm_read[bk][x][i]])],
                   f"vx.ex.x{i}") for i in range(31)]
    x32 = xel_ + [xel_[30]]
    m32 = list(mb) + [mb[30]]
    ecols = [[(from_bits(m32[4 * k_:4 * k_ + 4]), 0, 15), (15 - from_bits(x32[4 * k_:4 * k_ + 4]), 0, 15)]
             for k_ in range(8)]
    ecols[0].append((one, 1, 1))
    eb = sum_columns(c, ecols, 8, "vx.ex.e")
    esat = [c.node(c.t_const(1) + c.t_and([1 - eb[b_]] + [1 - x_ for x_ in eb[16:32]], -1),
                   f"vx.ex.es{b_}") for b_ in range(16)]
    cn4 = [from_bits(cb[4 * j:4 * j + 4]) for j in range(4)]
    pcols, const = [[] for _ in range(8)], 0
    for k_ in range(8):
        for o_ in range(4):
            terms, hi_ = [], 0
            for j in range(4):
                b_ = 4 * (k_ - j) + o_
                if 0 <= b_ < 16:
                    terms += c.t_and([lag1[b_]], a=cn4[j])
                    hi_ += 15
            if terms:
                g_ = c.node(terms, f"vx.ex.g{k_}.{o_}")
                const += place_signed(c, pcols, g_, 0, hi_, 4 * k_ + o_, f"vx.ex.c{k_}.{o_}")
    pp = sum_columns(c, pcols, 8, "vx.ex.p")
    hior = {p_: c.node(c.t_const(1) + c.t_and([1 - b_ for b_ in pp[p_:32]], -1), f"vx.ex.h{p_}")
            for p_ in range(8, 32)}
    dd = [c.node([t for s_ in range(32) if i + s_ < 32 for t in c.t_and(lits(kshb, s_) + [pp[i + s_]])],
                 f"vx.ex.d{i}") for i in range(8)]
    big = c.node([t for s_ in range(24) for t in c.t_and(lits(kshb, s_) + [hior[s_ + 8]])], "vx.ex.big")
    from chipformer.rv.vx import EXP_T
    pexp = [c.node([t for v_ in range(256) if EXP_T[v_] >> b_ & 1
                    for t in c.t_and(lits(dd, v_) + [1 - big])], f"vx.ex.q{b_}") for b_ in range(7)]
    pad_e = c.node(c.t_eq(sum(velt, Expr()) + 1 - n_x, 0, -64, 64), "vx.ex.pade")
    pad_p = c.node(c.t_eq(sum(velt, Expr()) - n_x, 0, -128, 64), "vx.ex.padp")
    done_x = c.node(c.t_eq(sum(velt, Expr()) - n_x - s2b[10], 0, -128, 64), "vx.ex.done")
    u.attn = (sc, raw, ro, eb, pp, pexp)

    # ---------------------------------------------------------------- the exit
    def vc_case(op, k, slot, parts):
        """VC token k of op: slot, then the 12 data bits packed from (bit list, width) parts."""
        bits = []
        for bl, w in parts:
            bl = list(bl) + [ZERO] * (w - len(bl))
            bits += bl[:w]
        assert len(bits) <= 12, (op, slot)
        bits += [ZERO] * (12 - len(bits))
        bits += [one if slot >> i & 1 else ZERO for i in range(3)]
        start = nr if k == 0 else [jeq(k)]
        case(start + [uops[op]], "VC", bits)

    def stream(op, nvc, lits_, fam_name, payload=0):
        case([jt[nvc - 1], uops[op]] + lits_, fam_name, payload)

    # VPUT: one VC (slot 0: target >> 5, target & 31, write target VM when target < 512), then
    # rs1 and rs2 as words
    vc_case("VPUT", 0, 0, [(C[5:9], 4), (C[0:5], 5), ([C[9], C[9], one - C[9]], 3)])
    for e, src in enumerate((A, B)):
        stream("VPUT", 1, [el_eq(e), sub_eq(0)], "VE", src[0:16])
        stream("VPUT", 1, [el_eq(e), sub_eq(1)], "VH", src[16:31] + [ZERO])
    stream("VPUT", 1, [el_eq(2)], "NT")

    # VQ: VC slots 0, 1, 2, 3, 4, 5, then a VH per element (the byte), NT after n
    dst_any = c.node(c.t_and([C[17]]) + c.t_and([C[18]]) + c.t_and([C[17], C[18]], -1), "vx.vq.dany")
    vc_case("VQ", 0, 0, [([ZERO] * 4, 4), (C[5:10], 5), ([C[18], one, dst_any], 3)])
    vc_case("VQ", 1, 1, [(and_bits(c, C[0:3], 1 - C[3], "vx.vq.ra"), 3),
                         (and_bits(c, C[0:3], C[3], "vx.vq.rb"), 3)])
    vc_case("VQ", 2, 2, [(A[0:9], 9), ([ZERO], 1), (C[15:17], 2)])
    vc_case("VQ", 3, 3, [(A[0:10], 10), ([C[3]], 1)])
    vc_case("VQ", 4, 4, [(B[0:12], 12)])
    vc_case("VQ", 5, 5, [(B[12:15], 3), (C[19:24], 5), (C[11:15], 4)])
    n_vq = 2 * (from_bits(s5[8:12]) + 16 * from_bits(s2[10:12]))
    done = c.node(c.t_eq(sum(velt, Expr()) - n_vq, 0, -126, 64), "vx.vq.done")
    stream("VQ", 6, [1 - done], "VH", q8 + [ZERO] * 8)
    stream("VQ", 6, [done], "NT")

    # VRED: VC slots 0, 2, 4, then the sum as a word, NT
    vc_case("VRED", 0, 0, [([ZERO] * 9, 9), ([one, ZERO, ZERO], 3)])
    vc_case("VRED", 1, 2, [([ZERO] * 6 + C[8:11], 9), ([ZERO], 1), ([C[7]], 1), (C[0:1], 1)])
    vc_case("VRED", 2, 4, [(C[1:7], 6)])
    stream("VRED", 3, [el_eq(0), sub_eq(0)], "VE", red[0:16])
    stream("VRED", 3, [el_eq(0), sub_eq(1)], "VH", red[16:31] + [ZERO])
    stream("VRED", 3, [el_eq(1)], "NT")
    # VGET: rd's value record (as an ALU result), NT for rd = x0
    vg = uops["VGET"]
    case(nr + [vg, rdw_any, k0], "F0", gv[0:15])
    case(nr + [vg, rdw_any, 1 - k0], "V", gv[0:15])
    case([jeq(1), ok, vg, rdw_any, k1], "F1", gv[15:30])
    case([jeq(1), ok, vg, rdw_any, 1 - k1], "V", gv[15:30])
    case([jeq(2), ok, vg, rdw_any], "F2", gv[30:32])
    case(nr + [vg, 1 - rdw_any], "NT")

    # VMAT: VC slots 0, 2, 4, 5; per row the running sums (VE lo, VE hi) then out (VE, VH | flag)
    vc_case("VMAT", 0, 0, [(C[0:4], 4), (A[14:19], 5), ([ZERO, ZERO, 1 - C[13]], 3)])
    vc_case("VMAT", 1, 2, [([ZERO] * 6 + C[15:17] + [ZERO], 9), ([c.and_([C[11], C[12]], "vx.mat.k3")], 1),
                           ([c.and_([A[19], 1 - A[20]], "vx.mat.st1")], 1),
                           ([c.and_([A[20], 1 - A[19]], "vx.mat.st2")], 1)])
    vc_case("VMAT", 2, 4, [(A[0:12], 12)])
    vc_case("VMAT", 3, 5, [(A[12:13], 1), (C[4:11], 7), ([C[13]], 1), ([C[14]], 1)])
    go = [1 - done_m]
    for lit in ([sub_eq(0)], [k3, sub_eq(2)], [k3, sub_eq(4)]):
        stream("VMAT", 4, go + lit, "VE", acc[0:16])
    for lit in ([sub_eq(1)], [k3, sub_eq(3)], [k3, sub_eq(5)]):
        stream("VMAT", 4, go + lit, "VE", acc[16:32])
    for lit in ([1 - k3, sub_eq(2)], [k3, sub_eq(6)]):
        stream("VMAT", 4, go + lit, "VE", o[0:16])
    for lit in ([1 - k3, sub_eq(3)], [k3, sub_eq(7)]):
        stream("VMAT", 4, go + lit, "VH", o[16:31] + [ZERO])
        flag = [ZERO] * 15 + [one]
        stream("VMAT", 4, go + lit + [el_eq(0), 1 - keep], None, flag)  # an op's first row
        for cond in better:
            stream("VMAT", 4, go + lit + [velt[0]] + cond, None, flag)
            stream("VMAT", 4, go + lit + [el_eq(0), keep] + cond, None, flag)
    stream("VMAT", 4, [done_m], "NT")
    # VEW: VC slots 0, 1, 4, 5, 6; per element (GLU: y lo, y hi), t lo, t hi, then o (VE, VH | flag)
    op2 = C[16]  # ROPE writes register b
    vc_case("VEW", 0, 0, [([c.node(c.t_and([op2, C[4 + i]]) + c.t_and([1 - op2, C[i]]), f"vx.ew.dst{i}")
                            for i in range(4)], 4), (C[19:24], 5), ([ZERO, ZERO, one], 3)])
    ra = [c.node(c.t_and([1 - C[3], C[i]]) + c.t_and([C[3], 1 - op2, 1 - C[7], C[4 + i]]), f"vx.ew.ra{i}")
          for i in range(3)]
    rb = [c.node(c.t_and([C[3], C[i]]) + c.t_and([1 - C[3], 1 - op2, C[7], C[4 + i]]), f"vx.ew.rb{i}")
          for i in range(3)]
    vc_case("VEW", 1, 1, [(ra, 3), (rb, 3), (A[11:16], 5)])
    vc_case("VEW", 2, 4, [(B[0:12], 12)])
    vc_case("VEW", 3, 5, [(B[12:15], 3), (A[0:5], 5), ([C[3]], 1), (C[15:17], 2), ([C[17]], 1)])
    vc_case("VEW", 4, 6, [(C[9:15], 6), (A[5:11], 6)])
    gw = [1 - done_w]
    glu = opb[0]
    for lit in ([1 - glu, sub_eq(0)], [glu, sub_eq(0)], [glu, sub_eq(2)]):
        stream("VEW", 5, gw + lit, "VE", tsh[0:16])
    for lit in ([1 - glu, sub_eq(1)], [glu, sub_eq(1)], [glu, sub_eq(3)]):
        stream("VEW", 5, gw + lit, "VE", tsh[16:32])
    for lit in ([1 - glu, sub_eq(2)], [glu, sub_eq(4)]):
        stream("VEW", 5, gw + lit, "VE", vo[0:16])
    for lit in ([1 - glu, sub_eq(3)], [glu, sub_eq(5)]):
        stream("VEW", 5, gw + lit, "VH", vo[16:31] + [ZERO])
        flag = [ZERO] * 15 + [one]
        stream("VEW", 5, gw + lit + [el_eq(0), 1 - keepw], None, flag)
        for cond in ebetter:
            stream("VEW", 5, gw + lit + [velt[0]] + cond, None, flag)
            stream("VEW", 5, gw + lit + [el_eq(0), keepw] + cond, None, flag)
    stream("VEW", 5, [done_w], "NT")
    # VMATR: VC slots 0, 2, 3, 4, 5. Scores: a word an output (signed max); p.V: raw (VE VE), then
    # out = raw s >> 12 (a word)
    vc_case("VMATR", 0, 0, [(C[0:4], 4), (A[11:16], 5), ([1 - C[10], ZERO, one], 3)])
    vc_case("VMATR", 1, 2, [([ZERO] * 6 + C[12:15], 9)])
    vc_case("VMATR", 2, 3, [(A[0:10], 10), ([C[10]], 1), ([C[11]], 1)])
    vc_case("VMATR", 3, 4, [(C[4:10], 6), (B[0:6], 6)])
    vc_case("VMATR", 4, 5, [(B[6:17], 11)])
    flag = [ZERO] * 15 + [one]
    m0 = [1 - mode_r]
    g0 = [1 - done_r0, 1 - pad_r]
    stream("VMATR", 5, m0 + g0 + [sub_eq(0)], "VE", sc[0:16])
    stream("VMATR", 5, m0 + g0 + [sub_eq(1)], "VH", sc[16:31] + [ZERO])
    for cond in ([rcmp0["o"]],):
        stream("VMATR", 5, m0 + g0 + [sub_eq(1), velt[0]] + cond, None, flag)
        stream("VMATR", 5, m0 + g0 + [sub_eq(1), el_eq(0), s3[11]] + cond, None, flag)
    stream("VMATR", 5, m0 + g0 + [sub_eq(1), el_eq(0), 1 - s3[11]], None, flag)
    stream("VMATR", 5, m0 + [pad_r, sub_eq(0)], "VE", 0)
    stream("VMATR", 5, m0 + [pad_r, sub_eq(1)], "VH", 0)
    stream("VMATR", 5, m0 + [done_r0], "NT")
    m1 = [mode_r, 1 - done_r1]
    stream("VMATR", 5, m1 + [sub_eq(0)], "VE", raw[0:16])
    stream("VMATR", 5, m1 + [sub_eq(1)], "VE", raw[16:32])
    stream("VMATR", 5, m1 + [sub_eq(2)], "VE", ro[0:16])
    stream("VMATR", 5, m1 + [sub_eq(3)], "VH", ro[16:31] + [ZERO])
    for cond in ([rcmp1["o"]], [1 - rcmp1["o"], rcmp1["n"]]):
        stream("VMATR", 5, m1 + [sub_eq(3), velt[0]] + cond, None, flag)
        stream("VMATR", 5, m1 + [sub_eq(3), el_eq(0), s3[11]] + cond, None, flag)
    stream("VMATR", 5, m1 + [sub_eq(3), el_eq(0), 1 - s3[11]], None, flag)
    stream("VMATR", 5, [mode_r, done_r1], "NT")

    # VEXP: VC slots 0, 1, 2, 4, 5, 6, 7, 3; per pair e, e' (VE), p, p' (VH); NT
    vc_case("VEXP", 0, 0, [([ZERO] * 9, 9), ([ZERO, one, ZERO], 3)])
    vc_case("VEXP", 1, 1, [(and_bits(c, C[0:3], 1 - C[3], "vx.ex.ra"), 3),
                           (and_bits(c, C[0:3], C[3], "vx.ex.rb"), 3), ([C[3]], 1)])
    vc_case("VEXP", 2, 2, [(C[11:20], 9), ([ZERO], 1), (C[4:6], 2)])
    vc_case("VEXP", 3, 4, [(C[6:11], 5), (C[20:25], 5), (B[0:2], 2)])
    vc_case("VEXP", 4, 5, [(B[2:14], 12)])
    vc_case("VEXP", 5, 6, [(B[14:16], 2), (A[0:10], 10)])
    vc_case("VEXP", 6, 7, [(A[10:22], 12)])
    vc_case("VEXP", 7, 3, [(A[22:31], 9)])
    gx = [1 - done_x]
    stream("VEXP", 8, gx + [sub_eq(0), 1 - parity], "VE", esat)
    stream("VEXP", 8, gx + [sub_eq(1), 1 - pad_e], "VE", esat)
    stream("VEXP", 8, gx + [sub_eq(1), pad_e], "VE", 0)
    stream("VEXP", 8, gx + [sub_eq(2)], "VH", pexp + [ZERO] * 9)
    stream("VEXP", 8, gx + [sub_eq(0), parity, 1 - pad_p], "VH", pexp + [ZERO] * 9)
    stream("VEXP", 8, gx + [sub_eq(0), parity, pad_p], "VH", 0)
    stream("VEXP", 8, [done_x], "NT")


    return u


# --- datapath pieces ---------------------------------------------------------------------------


def from_bits(bits):
    return sum((b * (1 << i) for i, b in enumerate(bits)), Expr())


def compare(c, o32, s32, name):
    """o > s and ~o >= s for 32-bit signed bit lists, nibble by nibble (the sign bit flipped:
    unsigned order): {"o": node, "n": node}."""
    def nibs(bits):
        return [from_bits(bits[4 * k:4 * k + 4]) if k < 7
                else from_bits(bits[28:31]) + 8 * (1 - bits[31]) for k in range(8)]

    sN = nibs(s32)
    out = {}
    for nm, bits in (("o", o32), ("n", [1 - b for b in o32])):
        oN = nibs(bits)
        gt = [c.node(c.t_ge(oN[k] - sN[k], 1, -15, 15), f"{name}.{nm}gt{k}") for k in range(8)]
        eq = [c.node(c.t_eq(oN[k] - sN[k], 0, -15, 15), f"{name}.{nm}eq{k}") for k in range(8)]
        terms = [t for k in range(8) for t in c.t_and([gt[k]] + eq[k + 1:])]
        if nm == "n":
            terms += c.t_and(eq)  # ~o >= s
        out[nm] = c.node(terms, f"{name}.{nm}cmp")
    return out


def clamp_terms(z):
    """clamp16 of a 31-bit signed bit list z, each bit as (coef, literals) AND terms: in range,
    z_i; above, 0x7FFF; below, 0x8000."""
    zeros = [1 - b for b in z[15:31]]
    ones = list(z[15:31])
    out = [[(1, [z[i]] + zeros), (1, [z[i]] + ones), (1, [1 - z[30]]), (-1, zeros)]
           for i in range(15)]
    return out + [[(1, [z[30]])]]


def place_signed(c, cols, n_, lo, hi, e, name):
    """Adds n_ (an integer in [lo, hi]) * 2^e to nibble columns as the digit and carry of
    n_ - lo; returns the constant lo * 2^e for the caller to add."""
    car = c.split(n_ - lo, 0, hi - lo, name)
    for x, xt, xe in ((n_ - lo - 16 * car, 15, e), (car, (hi - lo) // 16, e + 4)):
        if xe // 4 < len(cols):
            cols[xe // 4].append(((1 << xe % 4) * x, 0, (1 << xe % 4) * xt))
    return lo << e if lo >= 0 else -((-lo) << e)


def place_count(c, cols, n_, top, w, name):
    """Adds w * n_ (n_ in 0..top, w = +-2^e) to nibble columns: n_'s digit and carry by 16."""
    car = c.split(n_, 0, top, name)
    e = abs(w).bit_length() - 1
    sg = 1 if w > 0 else -1
    for x, xt, xe in ((n_ - 16 * car, 15, e), (car, top // 16, e + 4)):
        if xe // 4 < len(cols):
            f = sg * (1 << xe % 4)
            cols[xe // 4].append((f * x, min(0, f * xt), max(0, f * xt)))


def bitplane_cols(c, qs, ws, name, gsz=16):
    """sum_i q_i w_i as nibble-column terms for sum_columns (8 columns): q_i 8-bit signed bit
    lists, w_i linear in [-8, 7]. Per bit plane b and group of gsz lanes, a node N = sum q_ib w_i
    (|N| <= 8 gsz), split into a digit and a carry, at weight 2^b (-2^7 for the sign plane)."""
    cols = [[] for _ in range(8)]
    lo, hi = -8 * gsz, 7 * gsz
    const = 0
    for b in range(8):
        sg = -1 if b == 7 else 1
        for g in range(0, len(qs), gsz):
            n_ = c.node([t for i in range(g, min(g + gsz, len(qs))) for t in c.t_and([qs[i][b]], a=ws[i])],
                        f"{name}.n{b}.{g}")
            car = c.split(n_ - lo, 0, hi - lo, f"{name}.c{b}.{g}")
            dig = n_ - lo - 16 * car  # n = 16 car + dig + lo
            for e, top, w in ((dig, 15, b), (car, (hi - lo) // 16, b + 4)):
                f = sg * (1 << w % 4)
                cols[w // 4].append((f * e, min(0, f * top), max(0, f * top)))
            const += sg * lo << b
    const %= 1 << 32
    for k in range(8):
        v = const >> 4 * k & 15
        if v:
            cols[k].append((v * c.one, v, v))
    return cols


def mul_su(c, t, rn, name, const=None, lits_=()):
    """Bits 0..31 of t * r + const (mod 2^32): t a 16-bit signed bit list, r given as 4 nibble
    values rn (linear, the top one < 8), const {nibble: int or Expr} added into the columns.
    Stages: nibble products, their splits, the column splits, then a carry-lookahead add."""
    from chipformer.rv import core as CORE

    def sv(i, v):
        return v - 16 if i == 3 and v >= 8 else v

    lo, hi, hib = {}, {}, {}
    for i in range(4):
        tn = t[4 * i:4 * i + 4]
        for j in range(4):
            top = 15 if j < 3 else 7
            p = c.node([tt for v in range(1, 16) for tt in c.t_and(CORE.lits_of(tn, v), sv(i, v),
                                                                   a=rn[j])], f"{name}.p{i}{j}")
            if i < 3:
                h = c.split(p, 0, 15 * top, f"{name}.h{i}{j}")
                lo[i, j], hi[i, j], hib[i, j] = p - 16 * h, h, (0, 15 * top // 16)
            else:  # signed: p + 128 = l + 16 h', h = h' - 8
                h = c.split(p + 128, 128 - 8 * top, 128 + 7 * top, f"{name}.h{i}{j}")
                lo[i, j], hi[i, j] = p + 128 - 16 * h, h - 8
                hib[i, j] = (-8 + (128 - 8 * top) // 16, -8 + (128 + 7 * top) // 16)
    cols, bounds = [], []
    for k in range(8):
        e, pos, neg = Expr(), 0, 0
        for (i, j), l_ in lo.items():
            if i + j == k:
                e, pos = e + l_, pos + 15
        for (i, j), h_ in hi.items():
            if i + j == k - 1:
                e = e + h_
                pos, neg = pos + max(hib[i, j][1], 0), neg + max(-hib[i, j][0], 0)
        for kk, v in (const or {}).items():
            if kk == k:
                if isinstance(v, int):
                    e = e + v
                    pos, neg = pos + max(v, 0), neg + max(-v, 0)
                else:  # a 0/1 Expr to subtract
                    e = e - v
                    neg += 1
        cols.append(e)
        bounds.append((pos, neg))
    offs = CORE.col_offsets(bounds)
    car, dig, cmax = [], [], []
    for k in range(8):
        lo_, hi_ = offs[k] - bounds[k][1], offs[k] + bounds[k][0]
        cc = c.split(cols[k] + offs[k], lo_, hi_, f"{name}.c{k}")
        car.append(cc)
        dig.append(cols[k] + offs[k] - 16 * cc)
        cmax.append(hi_ // 16)
    g, p_, d = CORE.stage1_cols(c, [(dig[k] + (car[k - 1] if k else 0), 15 + (cmax[k - 1] if k else 0))
                                    for k in range(8)], list(lits_), name)
    res, _ = CORE.adder_stage2(c, g, p_, d, 8, name)
    return res


def clamp16_bits(c, z, name):
    """clamp16 of a 31-bit signed bit list z (bit 30 the sign): 16 bits."""
    zeros = [1 - b for b in z[15:31]]
    ones = list(z[15:31])
    out = []
    for i in range(15):  # in range: z_i; overflow: 0x7FFF (positive) or 0x8000 (negative)
        terms = (c.t_and([z[i]] + zeros) + c.t_and([z[i]] + ones)
                 + c.t_and([1 - z[30]]) + c.t_and(zeros, -1))
        out.append(c.node(terms, f"{name}.b{i}"))
    out.append(z[30])
    return out


def sat127_bits(c, q0, name):
    """clamp(q0, -127, 127) of a 16-bit signed bit list: 8 bits."""
    zeros = [1 - b for b in q0[7:16]]
    ones = list(q0[7:16])
    m128 = ones + [1 - b for b in q0[0:7]]  # q0 = -128
    out = []
    for i in range(7):
        terms = c.t_and([q0[i]] + zeros) + c.t_and([q0[i]] + ones)
        terms += c.t_and([1 - q0[15]]) + c.t_and(zeros, -1)  # 127 when above
        if i == 0:  # -127 (0x81) when below, or at -128
            terms += c.t_and([q0[15]]) + c.t_and(ones, -1) + c.t_and(m128)
        out.append(c.node(terms, f"{name}.b{i}"))
    out.append(q0[15])
    return out


def and_bits(c, bits, lit, name):
    """Each bit AND a 0/1 literal, as nodes."""
    return [c.and_([b, lit], f"{name}.b{i}") for i, b in enumerate(bits)]


_WROM: dict = {}


def snapshot(c, m):
    """The unit's memories as initial head states (every VA row carries the ballast, so a chunk
    read always sums four)."""
    import numpy as np

    from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS

    def row():
        r = np.zeros(HD)
        r[list(REF_COLS)] = A_REF
        return r

    v = m.vx
    key = hash(m.wrom)
    nwh = S.VX_CHUNKS // 384
    if _WROM.get("key") != key:
        tabs = [[{rw: row() for rw in range(128)} for _ in range(2)] for _ in range(nwh)]
        for a in range(-(-len(m.wrom) // 72)):
            cell = [m.wrom[72 * a + i] & 15 if 72 * a + i < len(m.wrom) else 0 for i in range(68)]
            rp, sl = divmod(a, 3)
            r = tabs[rp >> 7][rp >> 6 & 1][(rp & 63) + 64 * (sl == 2)]
            for i in range(34):
                r[2 + 34 * sl + i] = cell[2 * i] + 16 * cell[2 * i + 1]
        _WROM.update(key=key, tabs=tabs)
    for kh in range(nwh):
        c.tables[f"vx.wrom{kh}"].init = _WROM["tabs"][kh]
    if "silu" not in _WROM:
        from chipformer.rv.vx import SILU_T
        tab = []
        for x in range(2):
            rows = {rw: row() for rw in range(128)}
            for p in range(64):
                for sl in range(4):
                    tb = SILU_T[sl + 4 * p + 256 * x] + 2048
                    rows[p][2 + 2 * sl], rows[p][3 + 2 * sl] = tb & 255, tb >> 8
            tab.append(rows)
        _WROM["silu"] = tab
    for ph in range(2):
        c.tables[f"vx.silu{ph}"].init = _WROM["silu"]
    rr = {}
    for k, (cs_, sn_) in enumerate(v.rope):
        r = row()
        for b in range(16):
            r[2 + b] = A_BIT * (cs_ >> b & 1)
            r[18 + b] = A_BIT * (sn_ >> b & 1)
        rr[k] = r
    c.tables["vx.rope"].init = [rr]
    for kh in range(5):  # K cache: rows 32 g + slot, half x in value head x
        vh = []
        for x in range(2):
            rows = {}
            for rw in range(128):
                by = v.kc[128 * kh + rw][4 * x:4 * x + 4]
                if any(by):
                    r = row()
                    for q in range(4):
                        for b in range(8):
                            r[2 + 8 * q + b] = A_BIT * ((by[q] & 255) >> b & 1)
                    rows[rw] = r
            vh.append(rows)
        c.tables[f"vx.kc{kh}"].init = vh
    for kh in range(16):  # V cache: value head = slot, rows (l g, half)
        vh = []
        for x in range(2):
            t = 2 * kh + x
            rows = {}
            for lg in range(20):
                for h in range(2):
                    by = v.vc[32 * lg + t][4 * h:4 * h + 4]
                    if any(by):
                        r = row()
                        for q in range(4):
                            for b in range(8):
                                r[2 + 8 * q + b] = A_BIT * ((by[q] & 255) >> b & 1)
                        rows[2 * lg + h] = r
            vh.append(rows)
        c.tables[f"vx.vc{kh}"].init = vh
    for kh in range(4):
        vh = []
        for x in range(2):
            rows = {}
            g = 2 * kh + x
            for ch in range(6):
                for j in range(4):
                    r = row()
                    for half in range(2):
                        byte = v.va[64 * ch + 8 * g + 2 * j + half] & 255
                        for b in range(8):
                            r[2 + 16 * j + 8 * half + b] = A_BIT * (byte >> b & 1)
                    rows[4 * ch + j] = r
            vh.append(rows)
        c.tables[f"vx.va{kh}"].init = vh
    for bank in ("A", "B"):
        for kh in range(2):
            vh = []
            for x in range(2):
                rows = {}
                for rr in range(8):
                    reg = rr + (8 if bank == "B" else 0)
                    for p in range(16):
                        val = v.vm[reg][2 * (16 * kh + p) + x] & 0x7FFFFFFF
                        if val:
                            r = row()
                            for b in range(31):
                                r[2 + b] = A_BIT * (val >> b & 1)
                            rows[16 * rr + p] = r
                vh.append(rows)
            c.tables[f"vx.vm{bank}{kh}"].init = vh


def sum_columns(c, cols, n_nib, name, lits_=()):
    """Bits 0..4 n_nib - 1 of sum_k 16^k cols[k] (mod 16^n_nib): cols[k] is a list of (Expr, lo,
    hi) integer terms. Columns whose range exceeds a byte are split into a digit (stays) and a
    carry (moves up) one stage at a time (terms grouped so each group's sum fits), then a
    carry-lookahead add. Negative parts are offset per column (col_offsets), the offsets summing
    to 0 mod 16^n_nib."""
    from chipformer.rv import core as CORE

    cols = [list(cl) for cl in cols] + [[] for _ in range(n_nib - len(cols))]
    cols = cols[:n_nib]
    rnd = 0
    while True:
        bounds = [(sum(max(h, 0) for _, _, h in cl), sum(max(-l_, 0) for _, l_, _ in cl))
                  for cl in cols]
        if all(p + n_ + 15 <= 255 for p, n_ in bounds):
            break
        new = [[] for _ in range(n_nib)]
        for k, cl in enumerate(cols):
            if bounds[k][0] + bounds[k][1] + 15 <= 255:
                new[k] += cl
                continue
            groups, cur, cp, cn = [], [], 0, 0  # groups whose sum spans at most 240
            for t in cl:
                tp, tn = max(t[2], 0), max(-t[1], 0)
                if cur and cp + tp + cn + tn > 240:
                    groups.append((cur, cp, cn))
                    cur, cp, cn = [], 0, 0
                cur.append(t)
                cp, cn = cp + tp, cn + tn
            groups.append((cur, cp, cn))
            for gi, (g, gp, gn) in enumerate(groups):
                off = -(-gn // 16) * 16  # the group sum + off >= 0, off a multiple of 16
                e = sum((t[0] for t in g), Expr()) + off
                hi_ = gp + off
                car = c.split(e, 0, hi_, f"{name}.r{rnd}.c{k}.{gi}")
                new[k].append((e - 16 * car, 0, 15))
                if k + 1 < n_nib:
                    new[k + 1].append((car - off // 16, -(off // 16), hi_ // 16 - off // 16))
        cols = new
        rnd += 1
    bounds = [(sum(max(h, 0) for _, _, h in cl), sum(max(-l_, 0) for _, l_, _ in cl)) for cl in cols]
    offs = CORE.col_offsets(bounds)
    car, dig, cmax = [], [], []
    for k, cl in enumerate(cols):
        e = sum((t[0] for t in cl), Expr()) + offs[k]
        lo_, hi_ = offs[k] - bounds[k][1], offs[k] + bounds[k][0]
        cc = c.split(e, lo_, hi_, f"{name}.f{k}")
        car.append(cc)
        dig.append(e - 16 * cc)
        cmax.append(hi_ // 16)
    g, p_, d = CORE.stage1_cols(c, [(dig[k] + (car[k - 1] if k else 0), 15 + (cmax[k - 1] if k else 0))
                                    for k in range(n_nib)], list(lits_), name)
    res, _ = CORE.adder_stage2(c, g, p_, d, n_nib, name)
    return res
