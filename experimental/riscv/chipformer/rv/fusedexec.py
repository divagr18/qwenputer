"""Executor v3: Executor2's kernels without the apply launches, and the token decode on the GPU.

Each launch is the only sync. The residual at sublayer i is r_i = X[i % 2] + Y[(i - 1) % 3],
read on the fly; the sublayer adds its outputs into Y[i % 3] (atomics, as before), and its
programs also do the bookkeeping that the apply kernel did, each for its own slice of the dims:

  X[(i + 1) % 2] = r_i              (the next sublayer's base: r_(i+1) = that + Y[i % 3])
  Y[(i + 1) % 3] = 0                (last read by sublayer i - 1; written next by sublayer i + 1)
  the previous head sublayer's lag history moves on, its input r_(i-1) = X[i % 2] becoming lag 1
                                    (only sublayer i - 1 reads that history, and not now)

After the last sublayer one program decodes the output code into the next token (decode_read),
then every program writes that token's embedding as X[0] and zeroes the Y buffers (decode_embed):
24 launches a step instead of Executor2's 2 per sublayer plus FreeRun's torch ops. Same
arithmetic as Executor2 (fp32, every value exact), so the same tokens.

    fr = FreeRun3(c); core.snapshot(c, m); fr.load(0, S.tok("GO")); toks = fr.run(n)
"""

import numpy as np
import torch
import triton
import triton.language as tl

from chipformer.rv.tritonexec import Executor2

OWN = 1024  # dims per bookkeeping chunk


@triton.jit
def _book(xc, yc, xn, yz, hs, D, pid, nprog, SHIFT: tl.constexpr):
    """This program's slice of the bookkeeping (see the module doc)."""
    ch = tl.cdiv(D, nprog)
    lo = pid * ch
    for off in range(0, ch, 1024):
        i = lo + off + tl.arange(0, 1024)
        m = (i < D) & (i < lo + ch)
        base = tl.load(xc + i, mask=m, other=0.0)
        tl.store(xn + i, base + tl.load(yc + i, mask=m, other=0.0), mask=m)
        tl.store(yz + i, tl.zeros([1024], dtype=tl.float32), mask=m)
        if SHIFT:
            h0 = tl.load(hs + i, mask=m, other=0.0)
            h1 = tl.load(hs + D + i, mask=m, other=0.0)
            tl.store(hs + 2 * D + i, h1, mask=m)
            tl.store(hs + D + i, h0, mask=m)
            tl.store(hs + i, base, mask=m)


@triton.jit
def _res(xc, yc, i, m):
    return tl.load(xc + i, mask=m, other=0.0) + tl.load(yc + i, mask=m, other=0.0)


@triton.jit
def _ffn3(xc, yc, yn, xn, yz, hs, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd,
          BLOCK: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    nprog = tl.num_programs(0)
    xb, yb = xc + bat * D, yc + bat * D
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    ka = tl.arange(0, KA)
    ma = m[:, None] & (ka[None, :] < na[:, None])
    ia = tl.load(ixa + a0[:, None] + ka[None, :], mask=ma, other=0)
    av = tl.sum(tl.load(wa + a0[:, None] + ka[None, :], mask=ma, other=0.0) * _res(xb, yb, ia, ma), axis=1)
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    kb = tl.arange(0, KB)
    mb = m[:, None] & (kb[None, :] < nb[:, None])
    ib = tl.load(ixb + b0[:, None] + kb[None, :], mask=mb, other=0)
    bv = tl.sum(tl.load(wb + b0[:, None] + kb[None, :], mask=mb, other=0.0) * _res(xb, yb, ib, mb), axis=1)
    val = tl.load(coef + u, mask=m, other=0.0) * av * tl.maximum(bv.to(tl.bfloat16).to(tl.float32), 0.0)
    od = tl.load(outd + u, mask=m, other=0)
    tl.atomic_add(yn + bat * D + od, val, mask=m)
    _book(xb, yb, xn + bat * D, yz + bat * D, hs + bat * 3 * D, D, pid, nprog, SHIFT)


@triton.jit
def _ffn4(xc, yc, yn, xn, yz, hs, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd, bka, bkb,
          BLOCK: tl.constexpr, KC: tl.constexpr, SHIFT: tl.constexpr):
    """_ffn3 over units sorted by row length: each block loops over KC-wide chunks of its rows,
    only as far as its own longest row (bka / bkb), instead of the sublayer's longest padded."""
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    nprog = tl.num_programs(0)
    xb, yb = xc + bat * D, yc + bat * D
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    kc = tl.arange(0, KC)
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    av = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, tl.load(bka + pid), KC):
        mk = m[:, None] & ((k0 + kc)[None, :] < na[:, None])
        ia = tl.load(ixa + a0[:, None] + k0 + kc[None, :], mask=mk, other=0)
        av += tl.sum(tl.load(wa + a0[:, None] + k0 + kc[None, :], mask=mk, other=0.0) * _res(xb, yb, ia, mk), axis=1)
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    bv = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, tl.load(bkb + pid), KC):
        mk = m[:, None] & ((k0 + kc)[None, :] < nb[:, None])
        ib = tl.load(ixb + b0[:, None] + k0 + kc[None, :], mask=mk, other=0)
        bv += tl.sum(tl.load(wb + b0[:, None] + k0 + kc[None, :], mask=mk, other=0.0) * _res(xb, yb, ib, mk), axis=1)
    val = tl.load(coef + u, mask=m, other=0.0) * av * tl.maximum(bv.to(tl.bfloat16).to(tl.float32), 0.0)
    od = tl.load(outd + u, mask=m, other=0)
    tl.atomic_add(yn + bat * D + od, val, mask=m)
    _book(xb, yb, xn + bat * D, yz + bat * D, hs + bat * 3 * D, D, pid, nprog, SHIFT)


@triton.jit
def _lag3(xb, yb, hb, i, m, D):
    """The lagged residual: index lag * D + d (lag 0: r_i, lags 1..3: this sublayer's history)."""
    lo = i < D
    return tl.where(lo, _res(xb, yb, i, m & lo), tl.load(hb + i - D, mask=m & (i >= D), other=0.0))


@triton.jit
def _rows3(xb, yb, hb, D, rp, ix, w, base, ch, KP: tl.constexpr):
    r0 = tl.load(rp + base + ch)
    n = tl.load(rp + base + ch + 1) - r0
    kk = tl.arange(0, KP)
    mk = kk[None, :] < n[:, None]
    i = tl.load(ix + r0[:, None] + kk[None, :], mask=mk, other=0)
    acc = tl.sum(tl.load(w + r0[:, None] + kk[None, :], mask=mk, other=0.0) * _lag3(xb, yb, hb, i, mk, D), axis=1)
    return tl.maximum(acc, 0.0)


@triton.jit
def _scalar3(xb, yb, hb, D, rp, ix, w, vh):
    lane = tl.arange(0, 32)
    s0 = tl.load(rp + vh)
    n = tl.load(rp + vh + 1) - s0
    mk = lane < n
    i = tl.load(ix + s0 + lane, mask=mk, other=0)
    return tl.sum(tl.load(w + s0 + lane, mask=mk, other=0.0) * _lag3(xb, yb, hb, i, mk, D), axis=0)


@triton.jit
def _head3(xc, yc, yn, xn, yz, hs, h_ptr, S_ptr, D, V,
           qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
           arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
           KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
           QHOT: tl.constexpr, SHIFT: tl.constexpr):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    nprog = tl.num_programs(0)
    ch = tl.arange(0, 128)
    xb, yb = xc + bat * D, yc + bat * D
    hb = h_ptr + bat * 3 * D
    h = tl.load(khead + vh)
    k = _rows3(xb, yb, hb, D, krp, kix, kw, h * 128, ch, KK)
    v = _rows3(xb, yb, hb, D, vrp, vix, vw, vh * 128, ch, KV)
    z = _rows3(xb, yb, hb, D, zrp, zix, zw, vh * 128, ch, KZ)
    q = _rows3(xb, yb, hb, D, qrp, qix, qw, h * 128, ch, KQ)
    bv = _scalar3(xb, yb, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    nr = tl.load(nrows + vh)
    if nr > 0:
        av = _scalar3(xb, yb, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    o = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            o += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(yn + bat * D + dd, out * z * wv, mask=wv != 0.0)
    _book(xb, yb, xn + bat * D, yz + bat * D, hs + bat * 3 * D, D, vh, nprog, SHIFT)


@triton.jit
def _rows4(xb, yb, hb, D, rp, ix, w, ids, ent, ch, NRC: tl.constexpr, KP: tl.constexpr):
    """_rows3 over only the entity's non-empty rows (ids: their row numbers, -1 padding), placed
    into the 128-wide vector by a compare-and-sum."""
    c = tl.arange(0, NRC)
    base = ent * NRC
    r0 = tl.load(rp + base + c)
    n = tl.load(rp + base + c + 1) - r0
    kk = tl.arange(0, KP)
    mk = kk[None, :] < n[:, None]
    i = tl.load(ix + r0[:, None] + kk[None, :], mask=mk, other=0)
    acc = tl.maximum(tl.sum(tl.load(w + r0[:, None] + kk[None, :], mask=mk, other=0.0)
                            * _lag3(xb, yb, hb, i, mk, D), axis=1), 0.0)
    rid = tl.load(ids + base + c)
    return tl.sum(tl.where(ch[:, None] == rid[None, :], acc[None, :], 0.0), axis=1)


@triton.jit
def _head4(xc, yc, yn, xn, yz, hs, h_ptr, S_ptr, D, V,
           qrp, qix, qw, qid, krp, kix, kw, kid, vrp, vix, vw, vid, zrp, zix, zw, zid,
           arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
           KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
           NQ: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr, NZ: tl.constexpr,
           QHOT: tl.constexpr, SHIFT: tl.constexpr):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    nprog = tl.num_programs(0)
    ch = tl.arange(0, 128)
    xb, yb = xc + bat * D, yc + bat * D
    hb = h_ptr + bat * 3 * D
    h = tl.load(khead + vh)
    k = _rows4(xb, yb, hb, D, krp, kix, kw, kid, h, ch, NK, KK)
    v = _rows4(xb, yb, hb, D, vrp, vix, vw, vid, vh, ch, NV, KV)
    z = _rows4(xb, yb, hb, D, zrp, zix, zw, zid, vh, ch, NZ, KZ)
    q = _rows4(xb, yb, hb, D, qrp, qix, qw, qid, h, ch, NQ, KQ)
    bv = _scalar3(xb, yb, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    nr = tl.load(nrows + vh)
    if nr > 0:
        av = _scalar3(xb, yb, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    o = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            o += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(yn + bat * D + dd, out * z * wv, mask=wv != 0.0)
    _book(xb, yb, xn + bat * D, yz + bat * D, hs + bat * 3 * D, D, vh, nprog, SHIFT)


@triton.jit
def _decode_read(xf, yf, D, fam_idx, pay_idx, ids, tok_out, fid_out, p_out, pay_out,
                 NF: tl.constexpr, NP: tl.constexpr, NFAM: tl.constexpr):
    """One program per row: the output code (family and payload bits) into the next token."""
    bat = tl.program_id(0)
    j = tl.arange(0, 32)
    w2 = tl.exp2(j.to(tl.float32))
    mf = j < NF
    fb = tl.floor(_res(xf + bat * D, yf + bat * D, tl.load(fam_idx + j, mask=mf, other=0), mf) + 0.5)
    fid = tl.sum(tl.where(mf, fb * w2, 0.0), axis=0).to(tl.int32)
    fid = tl.minimum(tl.maximum(fid, 0), NFAM - 1)
    mp = j < NP
    pb = tl.floor(_res(xf + bat * D, yf + bat * D, tl.load(pay_idx + j, mask=mp, other=0), mp) + 0.5)
    p = tl.sum(tl.where(mp, pb * w2, 0.0), axis=0).to(tl.int64)
    tl.store(tok_out + bat, tl.load(ids + fid) + p)
    tl.store(fid_out + bat, fid)
    tl.store(p_out + bat, p)
    tl.store(pay_out + bat * 32 + j, pb, mask=mp)


@triton.jit
def _decode_embed(x0, y0, y1, y2, D, fid_in, p_in, pay_in, small, row0, table, base, bits, NT,
                  NP: tl.constexpr):
    """The next token's embedding into X[0] (small families from the table, big ones linear in
    the payload bits), and the Y buffers zeroed."""
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    i = pid * 1024 + tl.arange(0, 1024)
    m = i < D
    fid = tl.load(fid_in + bat)
    if tl.load(small + fid) != 0:
        row = tl.minimum(tl.maximum(tl.load(row0 + fid) + tl.load(p_in + bat), 0), NT - 1)
        x = tl.load(table + row * D + i, mask=m, other=0.0)
    else:
        x = tl.load(base + fid * D + i, mask=m, other=0.0)
        for b in range(NP):
            x += tl.load(pay_in + bat * 32 + b) * tl.load(bits + (fid * NP + b) * D + i, mask=m, other=0.0)
    o = bat * D + i
    tl.store(x0 + o, x, mask=m)
    z = tl.zeros([1024], dtype=tl.float32)
    tl.store(y0 + o, z, mask=m)
    tl.store(y1 + o, z, mask=m)
    tl.store(y2 + o, z, mask=m)


class FreeRun3:
    """fastexec.FreeRun's interface on Executor3: generation on the GPU, one graph per step."""

    def __init__(self, c, batch: int = 1, device="cuda"):
        from chipformer.rv import fastexec

        self.c, self.B, self.dev = c, batch, device
        # the tables of the v2 FreeRun (token embeddings, decode maps), and Executor2's arrays
        fr2 = fastexec.FreeRun(c, batch, device, compile_=False, v2=True)
        self.fr2, self.ex = fr2, fr2.ex
        ex, D = self.ex, self.ex.D
        self.D = D
        self.X = [torch.zeros(batch, D, device=device) for _ in range(2)]
        self.Y = [torch.zeros(batch, D, device=device) for _ in range(3)]
        self.L = len(ex.steps)
        assert ex.steps[-1]["kind"] == "ffn", "the last sublayer is the exit MLP"
        self.dummy = torch.zeros(batch, 3, D, device=device)
        self.fam_idx = fr2.fam_idx.to(torch.int32)
        self.pay_idx = fr2.pay_idx.to(torch.int32)
        self.NF, self.NP = len(self.fam_idx), len(self.pay_idx)
        assert self.NF <= 32 and self.NP <= 32
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.fid = torch.zeros(batch, dtype=torch.int32, device=device)
        self.p = torch.zeros(batch, dtype=torch.long, device=device)
        self.pay = torch.zeros(batch, 32, dtype=torch.float32, device=device)
        self.small = fr2.small.to(torch.int32)
        self.NP_bits = fr2.bits.shape[1]
        self.bits = fr2.bits.contiguous()
        self.sorted, self.compact = {}, {}
        for i, st in enumerate(ex.steps):
            if st["kind"] == "ffn":
                self.sorted[i] = self._sort_units(st)
            else:
                self.compact[i] = {nm: self._compact(st[nm]) for nm in ("q", "k", "v", "z")}
        self.use4, self.use_h4 = {}, {}
        self.graph = None

    def _compact(self, csr3):
        """Rows grouped by entity (128 each: a key head's q / k, a value head's v / z), only the
        non-empty ones, padded to the sublayer's most: (rp, ix, w, ids, NRC)."""
        rp, ix, w = (t.cpu().numpy() for t in csr3)
        n = np.diff(rp)
        ents = len(n) // 128
        keep = [[r for r in range(128) if n[e * 128 + r]] for e in range(ents)]
        nrc = max(2, 1 << max(max((len(k) for k in keep), default=1) - 1, 1).bit_length())
        rows, ids = [], []
        for e, kr in enumerate(keep):
            for j in range(nrc):
                if j < len(kr):
                    r = e * 128 + kr[j]
                    rows.append((ix[rp[r]:rp[r + 1]], w[rp[r]:rp[r + 1]]))
                    ids.append(kr[j])
                else:
                    rows.append((np.zeros(0, np.int32), np.zeros(0, np.float32)))
                    ids.append(-1)
        nrp = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int32)
        nix = np.concatenate([r[0] for r in rows] + [np.zeros(1, np.int32)]).astype(np.int32)
        nw = np.concatenate([r[1] for r in rows] + [np.zeros(1, np.float32)]).astype(np.float32)
        t = lambda a: torch.tensor(a, device=self.dev)
        return [t(nrp), t(nix), t(nw), t(np.asarray(ids, np.int32))], nrc

    def _sort_units(self, st):
        """The sublayer's units by b-row length (longest first), with each block's longest rows."""
        dev, BL = self.dev, self.ex.BLOCK
        rpa, ixa, wa = (t.cpu().numpy() for t in st["a"])
        rpb, ixb, wb = (t.cpu().numpy() for t in st["b"])
        na, nb = np.diff(rpa), np.diff(rpb)
        order = np.lexsort((-na, -nb))

        def csr(rp, ix, w):
            rows = [(ix[rp[u]:rp[u + 1]], w[rp[u]:rp[u + 1]]) for u in order]
            nrp = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int32)
            nix = np.concatenate([r[0] for r in rows] + [np.zeros(1, np.int32)]).astype(np.int32)
            nw = np.concatenate([r[1] for r in rows] + [np.zeros(1, np.float32)]).astype(np.float32)
            return [torch.tensor(a, device=dev) for a in (nrp, nix, nw)]

        U = st["U"]
        nblk = -(-U // BL)
        bka = [int(na[order[b * BL:(b + 1) * BL]].max()) for b in range(nblk)]
        bkb = [int(nb[order[b * BL:(b + 1) * BL]].max()) for b in range(nblk)]
        t32 = lambda a: torch.tensor(np.asarray(a, np.int32), device=dev)
        return {"a": csr(rpa, ixa, wa), "b": csr(rpb, ixb, wb),
                "coef": st["coef"][torch.tensor(order, device=dev)].contiguous(),
                "out": st["out"][torch.tensor(order, device=dev)].contiguous(),
                "bka": t32(bka), "bkb": t32(bkb)}

    def load(self, row: int, first_token: int):
        self.ex.load_init(row)
        self.ex.reset([row])
        self.X[0][row] = self.ex.embed([first_token])[0]
        for y in self.Y:
            y[row].zero_()

    def _step(self):
        ex, D, B = self.ex, self.D, self.B
        prev_hist = None  # the history the previous sublayer (a head) left to move on
        for i, st in enumerate(ex.steps):
            xc, yc, yn = self.X[i % 2], self.Y[(i - 1) % 3], self.Y[i % 3]
            xn, yz = self.X[(i + 1) % 2], self.Y[(i + 1) % 3]
            shift = prev_hist is not None
            hs = prev_hist if shift else self.dummy
            if st["kind"] == "ffn":
                self._ffn(i, st, xc, yc, yn, xn, yz, hs, shift, self.use4.get(i, False))
                prev_hist = None
            else:
                self._head(i, st, xc, yc, yn, xn, yz, hs, shift, self.use_h4.get(i, False))
                prev_hist = st["hist"]
        L = self.L
        xf, yf = self.X[L % 2], self.Y[(L - 1) % 3]
        fr2 = self.fr2
        _decode_read[(B,)](xf, yf, D, self.fam_idx, self.pay_idx, fr2.ids, self.tok, self.fid,
                           self.p, self.pay, NF=self.NF, NP=self.NP, NFAM=len(fr2.ids))
        _decode_embed[(triton.cdiv(D, 1024), B)](
            self.X[0], self.Y[0], self.Y[1], self.Y[2], D, self.fid, self.p, self.pay, self.small,
            fr2.row0, fr2.table, fr2.base, self.bits, len(fr2.table), NP=self.NP_bits)

    def _ffn(self, i, st, xc, yc, yn, xn, yz, hs, shift, sorted_):
        ex, D, B = self.ex, self.D, self.B
        grid = (triton.cdiv(st["U"], ex.BLOCK), B)
        if sorted_:
            so = self.sorted[i]
            _ffn4[grid](xc, yc, yn, xn, yz, hs, D, st["U"], *so["a"], *so["b"], so["coef"],
                        so["out"], so["bka"], so["bkb"], BLOCK=ex.BLOCK, KC=16, SHIFT=shift)
        else:
            _ffn3[grid](xc, yc, yn, xn, yz, hs, D, st["U"], *st["a"], *st["b"], st["coef"],
                        st["out"], BLOCK=ex.BLOCK, KA=st["KA"], KB=st["KB"], SHIFT=shift)

    def _head(self, i, st, xc, yc, yn, xn, yz, hs, shift, compact):
        D, B = self.D, self.B
        if compact:
            cq = self.compact[i]
            _head4[(st["V"], B)](
                xc, yc, yn, xn, yz, hs, st["hist"], st["S"], D, st["V"], *cq["q"][0], *cq["k"][0],
                *cq["v"][0], *cq["z"][0], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["khead"], st["nrows"], KQ=st["KQ"], KK=st["KK"], KV=st["KV"], KZ=st["KZ"],
                NQ=cq["q"][1], NK=cq["k"][1], NV=cq["v"][1], NZ=cq["z"][1], QHOT=st["QHOT"],
                SHIFT=shift)
        else:
            _head3[(st["V"], B)](
                xc, yc, yn, xn, yz, hs, st["hist"], st["S"], D, st["V"], *st["q"], *st["k"],
                *st["v"], *st["z"], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["khead"], st["nrows"], KQ=st["KQ"], KK=st["KK"], KV=st["KV"], KZ=st["KZ"],
                QHOT=st["QHOT"], SHIFT=shift)

    def tune(self, reps: int = 30):
        """Per MLP sublayer, the faster of _ffn3 (padded rows) and _ffn4 (sorted, chunked) on
        this GPU (timed on scratch buffers; both give the same values)."""
        scratch = [torch.zeros_like(self.X[0]) for _ in range(5)] + [torch.zeros_like(self.dummy)]
        self.use4 = {}
        for i, st in enumerate(self.ex.steps):
            if st["kind"] != "ffn":
                continue
            best = []
            for v in (False, True):
                for _ in range(3):
                    self._ffn(i, st, *scratch, False, v)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(reps):
                    self._ffn(i, st, *scratch, False, v)
                e1.record()
                torch.cuda.synchronize()
                best.append(e0.elapsed_time(e1))
            self.use4[i] = best[1] < best[0]
        for i, st in enumerate(self.ex.steps):
            if st["kind"] != "head":
                continue
            S0, H0 = st["S"].clone(), st["hist"].clone()
            best = []
            for v in (False, True):
                for _ in range(3):
                    self._head(i, st, *scratch, False, v)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(reps):
                    self._head(i, st, *scratch, False, v)
                e1.record()
                torch.cuda.synchronize()
                best.append(e0.elapsed_time(e1))
            st["S"].copy_(S0)
            st["hist"].copy_(H0)
            self.use_h4[i] = best[1] < best[0]

    def capture(self):
        if not hasattr(self, "use4"):
            self.tune()
        heads = [st for st in self.ex.steps if st["kind"] == "head"]
        saved = [t.clone() for t in self.X + self.Y] + [st["S"].clone() for st in heads] \
            + [st["hist"].clone() for st in heads]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        bufs = self.X + self.Y + [st["S"] for st in heads] + [st["hist"] for st in heads]
        for b, s0 in zip(bufs, saved):
            b.copy_(s0)
        self.graph = g

    def run(self, n_steps: int) -> np.ndarray:
        if self.graph is None:
            self.capture()
        out = torch.zeros(self.B, n_steps, dtype=torch.long, device=self.dev)
        for t in range(n_steps):
            self.graph.replay()
            out[:, t].copy_(self.tok)
        torch.cuda.synchronize()
        return out.cpu().numpy()
