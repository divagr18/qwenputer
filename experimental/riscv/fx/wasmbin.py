"""Minimal WebAssembly binary reader and per-function instruction-count instrumenter.

instrument(wasm) adds three mutable i64 globals per defined function, exported as
`n<idx>` (instructions), `c<idx>` (calls) and `b<idx>` (bytes moved by memory.copy and
memory.fill), and bumps n at the start of every
straight-line segment by the segment's static op count. Ops are weighted exactly
like wasmtime's fuel (0 for nop, drop, block, loop, unreachable, return, else, end;
1 for everything else). wasmtime also charges 1 per call and 1 per byte of
memory.copy/fill, so n + c + b over all functions equals the fuel the uninstrumented
module burns on the same run.
"""
from __future__ import annotations

import struct


def uleb(d, i):
    r = s = 0
    while True:
        b = d[i]
        i += 1
        r |= (b & 0x7F) << s
        s += 7
        if b < 0x80:
            return r, i


def sleb(d, i):
    r = s = 0
    while True:
        b = d[i]
        i += 1
        r |= (b & 0x7F) << s
        s += 7
        if b < 0x80:
            if b & 0x40:
                r -= 1 << s
            return r, i


def enc_uleb(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def enc_sleb(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if (v == 0 and not b & 0x40) or (v == -1 and b & 0x40):
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def name_str(d, i):
    n, i = uleb(d, i)
    return d[i:i + n].decode("utf-8", "replace"), i + n


class Module:
    def __init__(self, data: bytes):
        assert data[:4] == b"\0asm"
        self.data = data
        self.sections = []  # (id, start_of_payload, end, custom_name)
        i = 8
        while i < len(data):
            sid = data[i]
            sz, j = uleb(data, i + 1)
            cname = None
            if sid == 0:
                cname, _ = name_str(data, j)
            self.sections.append((sid, i, j, j + sz, cname))
            i = j + sz
        self.n_imported_funcs = 0
        self.n_imported_globals = 0
        self.imports = []
        self.func_names = {}
        self.code = []  # (body_start, body_end)
        for sid, _, s, e, cname in self.sections:
            if sid == 2:
                self._imports(s)
            elif sid == 10:
                n, i = uleb(data, s)
                for _ in range(n):
                    sz, i = uleb(data, i)
                    self.code.append((i, i + sz))
                    i += sz
            elif sid == 0 and cname == "name":
                self._names(s, e)

    def _imports(self, i):
        d = self.data
        n, i = uleb(d, i)
        for _ in range(n):
            m, i = name_str(d, i)
            f, i = name_str(d, i)
            k = d[i]
            i += 1
            if k == 0:
                t, i = uleb(d, i)
                self.imports.append((m, f, self.n_imported_funcs))
                self.n_imported_funcs += 1
            elif k == 1:
                i += 1
                fl = d[i]
                i += 1
                _, i = uleb(d, i)
                if fl & 1:
                    _, i = uleb(d, i)
            elif k == 2:
                fl = d[i]
                i += 1
                _, i = uleb(d, i)
                if fl & 1:
                    _, i = uleb(d, i)
            elif k == 3:
                i += 2
                self.n_imported_globals += 1

    def _names(self, s, e):
        d = self.data
        _, i = name_str(d, s)
        while i < e:
            sub = d[i]
            sz, i = uleb(d, i + 1)
            if sub == 1:
                j = i
                n, j = uleb(d, j)
                for _ in range(n):
                    idx, j = uleb(d, j)
                    nm, j = name_str(d, j)
                    self.func_names[idx] = nm
            i += sz

    def section(self, sid, cname=None):
        for x in self.sections:
            if x[0] == sid and (cname is None or x[4] == cname):
                return x
        return None


# Ops whose fuel cost is 0 in wasmtime.
FREE = {0x00, 0x01, 0x02, 0x03, 0x05, 0x0B, 0x0F, 0x1A}
# Ops that end a straight-line segment (control may leave or enter after them).
BOUNDARY = {0x00, 0x02, 0x03, 0x04, 0x05, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13}


def skip_blocktype(d, i):
    b = d[i]
    if b == 0x40 or b in (0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x70, 0x6F):
        return i + 1
    _, i = sleb(d, i)
    return i


def skip_memarg(d, i):
    a, i = uleb(d, i)
    if a & 0x40:
        _, i = uleb(d, i)
    _, i = uleb(d, i)
    return i


def ops(d, i, end):
    """Yield (op_start, op_end, opcode, sub) for the instruction stream d[i:end]."""
    while i < end:
        st = i
        op = d[i]
        i += 1
        sub = None
        if op in (0x02, 0x03, 0x04):
            i = skip_blocktype(d, i)
        elif op in (0x0C, 0x0D, 0x10, 0x12, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0xD2):
            _, i = uleb(d, i)
        elif op == 0x0E:
            n, i = uleb(d, i)
            for _ in range(n + 1):
                _, i = uleb(d, i)
        elif op in (0x11, 0x13):
            _, i = uleb(d, i)
            _, i = uleb(d, i)
        elif op == 0x1C:
            n, i = uleb(d, i)
            i += n
        elif 0x28 <= op <= 0x3E:
            i = skip_memarg(d, i)
        elif op in (0x3F, 0x40):
            _, i = uleb(d, i)
        elif op == 0x41:
            _, i = sleb(d, i)
        elif op == 0x42:
            _, i = sleb(d, i)
        elif op == 0x43:
            i += 4
        elif op == 0x44:
            i += 8
        elif op == 0xD0:
            i += 1
        elif op == 0xFC:
            sub, i = uleb(d, i)
            if sub <= 7:
                pass
            elif sub == 8:
                _, i = uleb(d, i)
                _, i = uleb(d, i)
            elif sub == 9 or sub == 13:
                _, i = uleb(d, i)
            elif sub in (10, 12, 14):
                _, i = uleb(d, i)
                _, i = uleb(d, i)
            elif sub in (11, 15, 16, 17):
                _, i = uleb(d, i)
            else:
                raise ValueError(f"unknown 0xFC sub-op {sub} at {st}")
        elif op in (0x01, 0x00, 0x05, 0x0B, 0x0F, 0x1A, 0x1B, 0xD1) or 0x45 <= op <= 0xC4:
            pass
        else:
            raise ValueError(f"unsupported opcode 0x{op:02x} at {st}")
        yield st, i, op, sub


def body_code_start(d, s):
    n, i = uleb(d, s)
    for _ in range(n):
        _, i = uleb(d, i)
        i += 1
    return i


def static_census(m: Module):
    """Static op histogram over all defined functions: {mnemonic_key: count}."""
    hist = {}
    d = m.data
    for s, e in m.code:
        i = body_code_start(d, s)
        for _, _, op, sub in ops(d, i, e):
            k = op if sub is None else (op << 8) | sub
            hist[k] = hist.get(k, 0) + 1
    return hist


def _counter(g, n):
    return b"\x23" + enc_uleb(g) + b"\x42" + enc_sleb(n) + b"\x7c" + b"\x24" + enc_uleb(g)


def _param_counts(m: Module):
    d = m.data
    ts = m.section(1)
    n, i = uleb(d, ts[2])
    params = []
    for _ in range(n):
        assert d[i] == 0x60
        i += 1
        np_, i = uleb(d, i)
        i += np_
        nr, i = uleb(d, i)
        i += nr
        params.append(np_)
    fs = m.section(3)
    n, i = uleb(d, fs[2])
    out = []
    for _ in range(n):
        t, i = uleb(d, i)
        out.append(params[t])
    return out


K = 3  # globals per function: n (ops), c (calls), b (bytes moved by memory.copy/fill)


def instrument(m: Module) -> bytes:
    d = m.data
    nf = len(m.code)
    gsec = m.section(6)
    if not gsec:
        raise ValueError("module without a global section is not handled")
    n_def_globals = uleb(d, gsec[2])[0]
    gbase = m.n_imported_globals + n_def_globals
    nparams = _param_counts(m)
    new_bodies = []
    for fi, (s, e) in enumerate(m.code):
        gi = gbase + K * fi
        gc, gb = gi + 1, gi + 2
        # locals: append one i32 temp
        ng, i = uleb(d, s)
        total = 0
        for _ in range(ng):
            c, i = uleb(d, i)
            i += 1
            total += c
        cs = i
        tmp = nparams[fi] + total
        out = bytearray(enc_uleb(ng + 1) + d[uleb(d, s)[1]:cs] + b"\x01\x7f")
        out += _counter(gc, 1)
        oplist = list(ops(d, cs, e))
        cur_start = 0
        cur_cost = 0
        pieces = []
        for k, (a_, b_, op, sub) in enumerate(oplist):
            cur_cost += 0 if op in FREE else 1
            if op in BOUNDARY or k == len(oplist) - 1:
                pieces.append((cur_start, k + 1, cur_cost))
                cur_start = k + 1
                cur_cost = 0
        for (k0, k1, c) in pieces:
            if c:
                out += _counter(gi, c)
            for k in range(k0, k1):
                a_, b_, op, sub = oplist[k]
                if op == 0xFC and sub in (10, 11):
                    out += b"\x22" + enc_uleb(tmp)  # local.tee tmp (the length)
                    out += d[a_:b_]
                    out += b"\x23" + enc_uleb(gb) + b"\x20" + enc_uleb(tmp) + b"\xad\x7c\x24" + enc_uleb(gb)
                else:
                    out += d[a_:b_]
        new_bodies.append(enc_uleb(len(out)) + bytes(out))

    res = bytearray(d[:8])

    def emit(sid, payload):
        res.append(sid)
        res.extend(enc_uleb(len(payload)))
        res.extend(payload)

    for sid, st, s, e, cname in m.sections:
        if sid == 6:
            n, i = uleb(d, s)
            emit(6, enc_uleb(n + K * nf) + d[i:e] + b"\x7e\x01\x42\x00\x0b" * (K * nf))
        elif sid == 7:
            n, i = uleb(d, s)
            extra = bytearray()
            for fi in range(nf):
                for off, kind in enumerate("ncb"):
                    nm = f"{kind}{m.n_imported_funcs + fi}".encode()
                    extra += enc_uleb(len(nm)) + nm + b"\x03" + enc_uleb(gbase + K * fi + off)
            emit(7, enc_uleb(n + K * nf) + d[i:e] + bytes(extra))
        elif sid == 10:
            emit(10, enc_uleb(nf) + b"".join(new_bodies))
        elif sid == 0:
            continue  # drop custom sections (names, debug info)
        else:
            res.extend(d[st:e])
    return bytes(res)


def code_bytes(m: Module) -> bytes:
    s = m.section(10)
    return m.data[s[2]:s[3]]


if __name__ == "__main__":
    import sys
    m = Module(open(sys.argv[1], "rb").read())
    print("imported funcs", m.n_imported_funcs, "defined", len(m.code), "names", len(m.func_names))
    h = static_census(m)
    print("distinct ops", len(h), "total", sum(h.values()))
