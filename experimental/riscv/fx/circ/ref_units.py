"""Python reference of every unit's semantics (circuits.md): written from the unit specs,
independently of the C units in rv/circ_units.h and rv/circ_loops.h. Each function takes the
input registers x (32 ints), a byte reader rd(addr) and the trigger's params, and returns None
(the unit declines) or (exit index, {reg: value} for every register it changes, {addr: byte}
for every byte it writes). Dead registers are then left alone by the caller (fuzz.py)."""
from __future__ import annotations

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1


def r32(rd, a):
    return rd(a) | rd(a + 1) << 8 | rd(a + 2) << 16 | rd(a + 3) << 24


def put32(w, a, v):
    for i in range(4):
        w[a + i] = v >> (8 * i) & 0xFF


def sext8(b):
    return (b - 256 if b & 0x80 else b) & M32


# ---- function units: exit 0 is the return to ra -------------------------------------------------

def memmove(x, rd, p, is_memcpy=False):
    d, s, n = x[10], x[11], x[12]
    w = {}
    if n and d != s:
        if is_memcpy and s < d < s + n:
            return None
        sp = x[2]
        if (d < sp and d + n > sp - 64) or (s < sp and s + n + 4 > sp - 64):
            return None
        src = [rd(s + i) for i in range(n)]
        for i in range(n):
            w[d + i] = src[i]
    return 0, {}, w


def memcpy(x, rd, p):
    return memmove(x, rd, p, True)


def memset(x, rd, p):
    d, c, n = x[10], x[11] & 0xFF, x[12]
    if n and d < x[2] and d + n > x[2] - 64:
        return None
    return 0, {}, {d + i: c for i in range(n)}


def eql(x, rd, p):
    a, n, b, bn = x[11], x[12], x[13], x[14]
    if n != bn:
        res = 0
    elif n == 0 or a == b:
        res = 1
    else:
        res = int(all(rd(a + i) == rd(b + i) for i in range(n)))
    return 0, {10: res}, {}


K256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]


def _rr(v, n):
    return (v >> n | v << (32 - n)) & M32


def sha(x, rd, p):
    inst, st, blk = x[10], x[11], x[12]
    wsp = r32(rd, inst + 8)
    if st & 3 or wsp < 0x100 or wsp > 0x100000 or (st < blk + 64 and blk < st + 32):
        return None
    if (st < wsp and st + 32 > wsp - 0x100) or (blk < wsp and blk + 64 > wsp - 0x100):
        return None
    W = [rd(blk + 4 * i) << 24 | rd(blk + 4 * i + 1) << 16 | rd(blk + 4 * i + 2) << 8 | rd(blk + 4 * i + 3)
         for i in range(16)]
    for i in range(16, 64):
        s0 = _rr(W[i - 15], 7) ^ _rr(W[i - 15], 18) ^ W[i - 15] >> 3
        s1 = _rr(W[i - 2], 17) ^ _rr(W[i - 2], 19) ^ W[i - 2] >> 10
        W.append((W[i - 16] + s0 + W[i - 7] + s1) & M32)
    h = [r32(rd, st + 4 * i) for i in range(8)]
    a, b, c, d, e, f, g, hh = h
    for i in range(64):
        t1 = (hh + (_rr(e, 6) ^ _rr(e, 11) ^ _rr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + W[i]) & M32
        t2 = ((_rr(a, 2) ^ _rr(a, 13) ^ _rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M32
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
    out = [(v + u) & M32 for v, u in zip(h, (a, b, c, d, e, f, g, hh))]
    w = {}
    for i in range(64):
        put32(w, wsp - 0x100 + 4 * i, W[i])
    for i in range(8):
        put32(w, st + 4 * i, out[i])
    return 0, {}, w


WYS = [0xa0761d6478bd642f, 0xe7037ed1a0b428db, 0x8ebc6af09c88c6e3, 0x589965cc75374cc3]


def wyhash(x, rd, p):
    inst, ptr, n = x[10], x[11], x[12]
    wsp = r32(rd, inst + 8)
    if wsp < 0x60 or wsp > 0x100000 or (ptr < wsp and ptr + n > wsp - 0x60):
        return None
    data = bytes(rd(ptr + i) for i in range(n))

    def rdn(o, k):
        return int.from_bytes(data[o:o + k], "little")

    last = {}

    def mix(a, b, slot=None):
        z = a * b
        lo, hi = z & M64, z >> 64
        if slot:
            last[slot] = (lo, hi)
        return lo ^ hi

    s0 = mix(WYS[0], WYS[1])
    st = [s0, s0, s0]
    if n <= 16:
        if n >= 4:
            end, q = n - 4, (n >> 3) << 2
            a, b = rdn(0, 4) << 32 | rdn(q, 4), rdn(end, 4) << 32 | rdn(end - q, 4)
        elif n:
            a, b = data[0] << 16 | data[n >> 1] << 8 | data[n - 1], 0
        else:
            a = b = 0
    else:
        i = 0
        if n >= 48:
            while i + 48 < n:
                for k in range(3):
                    st[k] = mix(rdn(i + 16 * k, 8) ^ WYS[k + 1], rdn(i + 16 * k + 8, 8) ^ st[k], f"r{k}")
                i += 48
            st[0] ^= st[1] ^ st[2]
        m, j = n - i, 0
        while j + 16 < m:
            st[0] = mix(rdn(i + j, 8) ^ WYS[1], rdn(i + j + 8, 8) ^ st[0], "f1")
            j += 16
        a, b = rdn(n - 16, 8), rdn(n - 8, 8)
    z = (a ^ WYS[1]) * (b ^ st[0])
    a, b = z & M64, z >> 64
    h = mix(a ^ WYS[0] ^ n, b ^ WYS[1], "fin")
    w = {}

    def put64(ad, v):
        put32(w, ad, v & M32)
        put32(w, ad + 4, v >> 32)

    put64(wsp - 0x60, last["fin"][0]); put64(wsp - 0x58, last["fin"][1])
    put64(wsp - 0x50, a); put64(wsp - 0x48, b)
    if "f1" in last:
        put64(wsp - 0x40, last["f1"][0]); put64(wsp - 0x38, last["f1"][1])
    for k in range(3):
        if f"r{k}" in last:
            put64(wsp - 0x10 - 0x10 * k, last[f"r{k}"][0]); put64(wsp - 0x08 - 0x10 * k, last[f"r{k}"][1])
    return 0, {10: h & M32, 11: h >> 32}, w


# ---- loop units -----------------------------------------------------------------------------------

def tokest(x, rd, p):
    ptr, n = x[21], x[16]
    if x[14] != 5 or x[15] != 0x20 or n == 0:
        return None
    run, tot = x[13] << 32 | x[12], x[10] << 32 | x[11]
    a7, t0 = x[17], x[5]
    for i in range(n):
        b = rd(ptr + i)
        if 9 <= b <= 13 or b == 0x20:
            q = (run >> 2) + (run & 3 != 0)
            s = (tot + q) & M64
            ov = s < tot
            a7, t0 = s & M32, (-ov) & M32
            tot = M64 if ov else s
            run = 0
        else:
            run = (run + 1) & M64
            a7 = M32 if run == 0 else 0
            if run == 0:
                run = M64
        if run >= 65536 or tot >= 65536:
            return None
    return 0, {12: run & M32, 13: run >> 32, 11: tot & M32, 10: tot >> 32, 17: a7, 5: t0,
               21: (ptr + n) & M32, 16: 0}, {}


def _plain(p, b):
    return p[1 + (b >> 5)] >> (b & 31) & 1


def jscan(x, rd, p):
    i0, end, buf, cur = x[13], x[14], x[10], (x[12] + 0x24) & M32
    if cur & 3:
        return None
    k, stop = 0, 0
    while True:
        idx = (i0 + k + 1) & M32
        if idx == end:
            break
        if not _plain(p, rd((buf + idx) & M32)):
            stop = 1
            break
        k += 1
        if k == 65535:
            break
    if k == 0:
        return None
    lo, hi = (buf + i0 + 1) & M32, (buf + i0 + k + stop) & M32
    if cur <= hi and cur + 4 > lo:
        return None
    w = {}
    put32(w, cur, (i0 + k) & M32)
    return 0, {13: (i0 + k) & M32}, w


def jesc(x, rd, p):
    fr, q3, q7 = (x[8] - 0x3C) & M32, x[19], x[23]
    if q3 > 255 or q7 > 255 or fr & 3:
        return None
    ln, i0, base = r32(rd, fr), x[24], x[21]
    k, a1, a2, a3 = 0, x[11], x[12], x[13]
    while True:
        idx = (i0 + k + 1) & M32
        if idx == ln:
            break
        b = rd((base + idx) & M32)
        n2, n3, plain = a2, a3, False
        if b != q3 and b != q7:
            if q7 < b:
                plain = True
            else:
                n2, n3 = b & 0x7E, 0x20
                if n2 == 0x20:
                    plain = True
                else:
                    n2, n3 = (b + 0xDD) & 0xFF, 0x38
                    plain = n2 <= 0x38
        if not plain:
            break
        a1, a2, a3 = b, n2, n3
        k += 1
        if k == 65535:
            break
    if k == 0:
        return None
    s8 = (i0 + k) & M32
    return 0, {24: s8, 27: (s8 + base) & M32, 11: a1, 12: a2, 13: a3}, {}


def ascii_run(x, rd, p):
    i, bound, base, t4 = x[10], x[19], x[20], x[29]
    k = 0
    while True:
        t5 = 1
        i = (i + 1) & M32
        if i >= bound:
            xi = 0
            break
        t4 = (i + base) & M32
        t5 = sext8(rd(t4))
        k += 1
        if t5 >> 31:
            xi = 1
            break
        if k > 65535:
            return None
    return xi, {10: i, 29: t4, 30: t5}, {}


def maskp(x, rd, p):
    tab = x[21]
    if tab != p[2] or x[20] != 0x5F or x[17] != 10:
        return None
    s6, s8, s3, s10 = x[22], x[24], x[19], x[26]
    a0, a1, a2, a3, a4, a5, a6 = (x[r] for r in range(10, 17))
    for k in range(12):
        if not 0 < r32(rd, tab + 8 * k + 4) <= 64:
            return None
    matched = False
    for k in range(12):
        a0, a1 = r32(rd, tab + 8 * k), r32(rd, tab + 8 * k + 4)
        if s6 == 0:
            a2 = rd((s10 - 1) & M32)
            if a2 == 0x5F:
                continue
            a3 = (a2 + 0xD0) & 0xFE
            if a3 < 10:
                continue
            a2 = ((a2 & 0xDF) + 0xBF) & 0xFE
            a3 = 0x1A
            if a2 < 0x1A:
                continue
        a4 = (a1 + s8) & M32
        if a4 >= s3:
            continue
        a2 = s10
        ne = False
        while True:
            a3, a5 = rd(a2), rd(a0)
            a6 = int(((a3 + 0xBF) & 0xFE) < 0x1A) << 5
            a3 |= a6
            a6 = int(((a5 + 0xBF) & 0xFE) < 0x1A) << 5
            a5 |= a6
            if a3 != a5:
                ne = True
                break
            a2, a1, a0 = (a2 + 1) & M32, (a1 - 1) & M32, (a0 + 1) & M32
            if a1 == 0:
                break
        if not ne:
            matched = True
            break
    if not matched:
        k, a0 = 12, 12
    return int(matched), {27: k, 10: a0, 11: a1, 12: a2, 13: a3, 14: a4, 15: a5, 16: a6}, {}


def memchr(x, rd, p):
    ptr, n, c = x[11], x[12], x[10]
    if n == 0 or n > 65535:
        return None
    a3 = x[13]
    for i in range(n):
        a3 = rd(ptr + i)
        if a3 == c:
            return 0, {13: a3, 11: (ptr + i) & M32, 12: n - i}, {}
    return 1, {13: a3, 11: (ptr + n) & M32, 12: 0}, {}


def bcopy(x, rd, p):
    s, d = x[27], x[12]
    n = (x[11] - s) & M32
    if n == 0 or s < d < s + n + 4 or (d < x[2] and d + n > x[2] - 64):
        return None
    src = [rd(s + i) for i in range(n)]
    return 0, {13: src[-1], 27: (s + n) & M32, 12: (d + n) & M32}, {d + i: src[i] for i in range(n)}


def move_ir(x, rd, p):
    """The core-IR MOVE circuit's coverage: 4-aligned d, s, n; 4 <= n <= 84; forward-safe."""
    d, s, n = x[10], x[11], x[12]
    if (d | s | n) & 3 or not 4 <= n <= 84 or d == s or not (d < s or d - s >= n):
        return None
    sp = x[2]
    if (d < sp and d + n > sp - 64) or (s < sp and s + n > sp - 64):
        return None
    src = [rd(s + i) for i in range(n)]
    return 0, {}, {d + i: src[i] for i in range(n)}


def eql_ir(x, rd, p):
    """The core-IR EQL circuit's coverage: a1 != a3, 1 <= n <= 31 (at eqlBytes' beq a1, a3)."""
    a, n, b = x[11], x[12], x[13]
    if a == b or not 1 <= n <= 31:
        return None
    return 0, {10: int(all(rd(a + i) == rd(b + i) for i in range(n)))}, {}


REF = {14: move_ir, 15: eql_ir, 1: memmove, 2: memcpy, 3: memset, 4: eql, 5: tokest, 6: jscan, 7: jesc, 8: ascii_run, 9: maskp,
       10: memchr, 11: bcopy, 12: sha, 13: wyhash}
