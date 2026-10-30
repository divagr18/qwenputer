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
from chipformer.q35.ir import Chan
from chipformer.q35.prims import thead

ZERO = Expr()


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


def count_fn(c, x, f, top, name, a=None):
    """An L1 MLP node: a * f(k) for the count k = 0..top of the 2^-k counter x (thresholds on x,
    as RvCore.thermo), f 0/1-valued; a: a linear 0/1 form (default ONE)."""
    one = c.one
    a = one if a is None else a
    terms = [(1, a, one)] if f(0) else []
    for v in range(1, top + 1):
        d = f(v) - f(v - 1)
        if d:
            m = v - 1  # [k > m] = [k >= v]
            terms += [(d, a, 7 * one - 8 * 2**m * x), (-d, a, 6 * one - 8 * 2**m * x)]
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
                q[4 * ch + j] = andpre(lits(vchunk, ch))
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
        for x in range(2):
            va_read += [outs[x][16 * j:16 * j + 16] for j in range(4)]  # 4 pairs of 16 bits
    # a 4-hot read returns each row at 1/4
    u.va = [[4 * d for d in bits] for pr in va_read for bits in (pr[0:8], pr[8:16])]  # 64 bytes

    # ---------------------------------------------------------------- VQ
    # q = clamp(round(clamp16(VM[src][e] >> sh) * r / 2^15), +-127): the element (bank and
    # parity), an arithmetic shift by 4 (sh >> 2) then by sh & 3, clamp16, the multiply with the
    # rounding folded in (+2^14, -1 when negative), then the clamp to +-127
    s4, s5 = cfg[4], cfg[5]
    sh = s5[3:8]
    srcb = s2[11]
    rbits = s4[0:12] + s5[0:3]
    rn = [from_bits(rbits[4 * j:4 * j + 4]) for j in range(4)]
    srcs = {(bk, x): vm_read[bk][x] for bk in ("A", "B") for x in range(2)}
    y = []
    for i in range(31):
        terms = []
        for (bk, x), v in srcs.items():
            bl = lits([srcb], int(bk == "B")) + lits([parity], x)
            for k in range(8):
                j = i + 4 * k
                terms += c.t_and(bl + lits(sh[2:5], k) + [v[min(j, 30)]])
        y.append(c.node(terms, f"vx.vq.y{i}"))
    z = [c.node([t for k in range(4) for t in c.t_and(lits(sh[0:2], k) + [y[min(i + k, 30)]])],
                f"vx.vq.z{i}") for i in range(31)]
    t16 = clamp16_bits(c, z, "vx.vq.t")
    p = mul_su(c, t16, rn, "vx.vq.m", const={3: 4, 0: t16[15]})
    q8 = sat127_bits(c, p[15:31], "vx.vq.q")
    u.vq = q8


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
    varg = [count_bit(c, xel, b, 64, f"vx.varg{b}", base=wbase[b], shift=1) for b in range(13)]
    trk = {}
    for name, extra, gate in (
            ("vsum", {}, [isVH, wt[0], 1 - wt[1], 1 - wt[2]]),
            ("vmx", {33: wt[0]}, [isVH, flag, wt[2], 1 - wt[1]]),
            ("vbest", {33 + b: varg[b] for b in range(13)}, [isVH, flag] + lits(wt, 0))):
        data = dict(word_v)
        data.update(extra)
        (outs,) = thead(c, f"vx.{name}", {0: one}, {0: one},
                        [{"data": data, "ref": one, "beta": andpre(gate)}])
        c.tables[f"vx.{name}"] = c.nodes[-1]
        trk[name] = outs
    u.trk = trk

    # ---------------------------------------------------------------- VRED
    # vsum = sum over i < n of VA chunk byte i, squared (mode 0) or as is (mode 1): per lane the
    # nibble products lo^2, 2 lo hs, hs^2 (q = lo + 16 hs, hs signed), summed by sum_columns
    n_red = from_bits(s2[10:12]) + 4 * from_bits(s4c[0:5])
    mode1 = s2[9]
    lane = [c.node(c.t_ge(n_red, i + 1, 0, 127), f"vx.red.g{i}") for i in range(64)]
    cols = [[], [], []]
    for i, q in enumerate(u.va):
        lo_b, hs_b = q[0:4], q[4:8]
        lo_v = from_bits(lo_b)
        hs_v = from_bits(hs_b[0:3]) - 8 * hs_b[3]
        g0, g1 = [lane[i], 1 - mode1], [lane[i], mode1]

        def sv(v):
            return v - 16 if v >= 8 else v

        c0 = (sum((c.t_and(lits(lo_b, v) + g0, v, a=lo_v) for v in range(1, 16)), [])
              + sum((c.t_and([lo_b[b]] + g1, 1 << b) for b in range(4)), []))
        c1 = (sum((c.t_and(lits(lo_b, v) + g0, 2 * v, a=hs_v) for v in range(1, 16)), [])
              + sum((c.t_and([hs_b[b]] + g1, -8 if b == 3 else 1 << b) for b in range(4)), []))
        c2 = sum((c.t_and(lits(hs_b, v) + g0, sv(v), a=hs_v) for v in range(1, 16)), [])
        cols[0].append((c.node(c0, f"vx.red.c0.{i}"), 0, 225))
        cols[1].append((c.node(c1, f"vx.red.c1.{i}"), -240, 210))
        cols[2].append((c.node(c2, f"vx.red.c2.{i}"), 0, 64))
    red = sum_columns(c, cols, 8, "vx.red")
    u.red = red

    # ---------------------------------------------------------------- VGET
    # rd = vmx (|s|, or s when it was tracked signed) / vsum / varg / bit length of vmx, by rs1
    sv_bits = trk["vmx"][0:31]
    signed_t = trk["vmx"][31]
    neg = [1 - b for b in sv_bits] + [one]  # -s = ~s + 1 (32 bits, s sign-extended)
    from chipformer.rv import core as CORE
    negs = CORE.add_nibbles(c, neg, [one] + [ZERO] * 31, 8, [], "vx.get.neg")
    sgn = sv_bits[30]
    absv = [c.node(c.t_and([1 - sgn, sv_bits[min(i, 30)]]) + c.t_and([sgn, 1 - signed_t, negs[i]])
                   + c.t_and([sgn, signed_t, sv_bits[min(i, 30)]]), f"vx.get.abs{i}")
            for i in range(32)]
    th = [c.node(c.t_const(1) + c.t_and([1 - b for b in absv[m:31]], -1), f"vx.get.ge{m}")
          for m in range(31)]
    blen = c.int_bits(sum(th, Expr()), 31, "vx.get.bl", 5)
    srcs = [absv, list(trk["vsum"][0:31]) + [trk["vsum"][30]], list(trk["vbest"][31:44]),
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
    vc_case("VPUT", 0, 0, [(C[5:9], 4), (C[0:5], 5), ([ZERO, ZERO, one - C[9]], 3)])
    for e, src in enumerate((A, B)):
        stream("VPUT", 1, [el_eq(e), sub_eq(0)], "VE", src[0:16])
        stream("VPUT", 1, [el_eq(e), sub_eq(1)], "VH", src[16:31] + [ZERO])
    stream("VPUT", 1, [el_eq(2)], "NT")

    # VQ: VC slots 0, 1, 2, 3, 4, 5, then a VH per element (the byte), NT after n
    dst_any = c.node(c.t_and([C[17]]) + c.t_and([C[18]]) + c.t_and([C[17], C[18]], -1), "vx.vq.dany")
    vc_case("VQ", 0, 0, [([ZERO] * 4, 4), (C[5:10], 5), ([C[18], one, dst_any], 3)])
    vc_case("VQ", 1, 1, [(and_bits(c, C[0:3], 1 - C[3], "vx.vq.ra"), 3),
                         (and_bits(c, C[0:3], C[3], "vx.vq.rb"), 3)])
    vc_case("VQ", 2, 2, [(A[0:9], 9), (C[15:17], 2), ([C[3]], 1)])
    vc_case("VQ", 3, 3, [(A[0:10], 10)])
    vc_case("VQ", 4, 4, [(B[0:12], 12)])
    vc_case("VQ", 5, 5, [(B[12:15], 3), (C[19:24], 5), (C[11:15], 4)])
    n_vq = 2 * (from_bits(s5[8:12]) + 16 * from_bits(s2[9:11]))
    done = c.node(c.t_eq(sum(velt, Expr()) - n_vq, 0, -126, 64), "vx.vq.done")
    stream("VQ", 6, [1 - done], "VH", q8 + [ZERO] * 8)
    stream("VQ", 6, [done], "NT")

    # VRED: VC slots 0, 2, 4, then the sum as a word, NT
    vc_case("VRED", 0, 0, [([ZERO] * 9, 9), ([one, ZERO, ZERO], 3)])
    vc_case("VRED", 1, 2, [([ZERO] * 6 + C[8:11], 9), ([C[7]], 1), (C[0:2], 2)])
    vc_case("VRED", 2, 4, [(C[2:7], 5)])
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
    return u


# --- datapath pieces ---------------------------------------------------------------------------


def from_bits(bits):
    return sum((b * (1 << i) for i, b in enumerate(bits)), Expr())


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
