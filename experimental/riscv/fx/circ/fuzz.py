"""Differential fuzzing of every unit (circuits.md, verification): random input states,
edge cases included (zero lengths, overlap both ways, every alignment, capacity / end
boundaries, inputs a unit must decline). Each case is checked three ways:

  1. rv_unit_check: the C unit against stepping the original instructions from the same state
     (PC, every register live at the exit, every byte either wrote);
  2. the Python reference (ref_units.py) against the C unit's final state;
  3. a decline must leave the state untouched (checked in C) and must match the reference's.

  python -I fuzz.py DLL ELF OUT_JSON [--cases 100000] [--units NAME,..] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import machine as MC  # noqa: E402
import ref_units as R  # noqa: E402
import units as U  # noqa: E402

SCR = 0x00400000      # scratch buffers
WIN = 0x3000          # bytes compared there
FRAME = 0x00800000    # loop units' frame / struct pointers
INST = 0x00300000     # a wasm2c instance (its __stack_pointer at +8)
STRS = 0x00500000     # the secret prefixes' bytes (maskSecrets)
SP = 0x07FF0000
WS = b" \t\n\r\x0b\x0c"


def rlen(rng, hi=400):
    k = rng.random()
    if k < 0.08:
        return 0
    if k < 0.35:
        return rng.randint(1, 8)
    if k < 0.45:
        return rng.choice([3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65])
    if k < 0.8:
        return rng.randint(9, 80)
    return rng.randint(81, hi)


def rbytes(rng, n, alphabet=None):
    if alphabet is None:
        return rng.randbytes(n)
    return bytes(rng.choices(alphabet, k=n))


def sprinkle(rng, data, special, p):
    """Replace a fraction p of the bytes with bytes from `special`."""
    data = bytearray(data)
    k = int(len(data) * p)
    for i in rng.sample(range(len(data)), k):
        data[i] = rng.choice(special)
    return bytes(data)


class Fuzzer:
    def __init__(self, m: MC.Machine, seed):
        self.m = m
        self.rng = random.Random(seed)
        self.table = m.by_label.get("text_utils.maskSecrets secret_prefixes probe")
        if self.table:
            tab = self.table[0]["params"][2]
            a = STRS
            for i, s in enumerate(U.SECRET_PREFIXES):
                m.wr(a, s.encode())
                m.w32(tab + 8 * i, a)
                m.w32(tab + 8 * i + 4, len(s))
                a += 32

    def base_regs(self):
        rng = self.rng
        x = [rng.getrandbits(32) for _ in range(32)]
        x[0] = 0
        x[1] = MC.SENTINEL
        x[2] = SP
        return x

    # ---- generators: (registers, memory writes {addr: bytes}, extra windows to compare) ----
    def g_move(self, memcpy=False):
        rng = self.rng
        x = self.base_regs()
        n = rlen(rng, 600)
        s = SCR + 64 + rng.randrange(0, 0x1000)
        k = rng.random()
        if k < 0.05:
            d = s
        elif k < 0.4:
            d = s + rng.randint(-24, 24)
        else:
            d = SCR + rng.randrange(0, 0x1000)
        x[10], x[11], x[12] = d, s, n
        return x, {SCR: rbytes(rng, WIN)}, []

    def g_set(self):
        rng = self.rng
        x = self.base_regs()
        x[10], x[12] = SCR + rng.randrange(0, 0x1000), rlen(rng, 600)
        return x, {SCR: rbytes(rng, WIN)}, []

    def g_eql(self):
        rng = self.rng
        x = self.base_regs()
        n = rlen(rng, 300)
        a, b = SCR + rng.randrange(0, 0x1000), SCR + 0x1400 + rng.randrange(0, 0x1000)
        mem = bytearray(rbytes(rng, WIN))
        if rng.random() < 0.7:
            mem[b - SCR:b - SCR + n] = mem[a - SCR:a - SCR + n]
            if n and rng.random() < 0.5:
                mem[b - SCR + rng.randrange(n)] ^= 1 << rng.randrange(8)
        if rng.random() < 0.05:
            b = a
        bn = n if rng.random() < 0.85 else rlen(rng, 300)
        x[11], x[12], x[13], x[14] = a, n, b, bn
        return x, {SCR: bytes(mem)}, []

    def wsp(self):
        return 0x1000 + 16 * self.rng.randrange(0, 0xF000 // 16) * 15 // 16

    def g_sha(self):
        rng = self.rng
        x = self.base_regs()
        wsp = rng.randrange(0x200, 0xF0000) & ~15
        st = SCR + 4 * rng.randrange(0, 0x200)
        blk = SCR + 0x1000 + rng.randrange(0, 0x1000)
        x[10], x[11], x[12] = INST, st, blk
        mem = {SCR: rbytes(rng, WIN), INST: struct.pack("<III", 0, 0, wsp), wsp - 0x100: rbytes(rng, 0x100)}
        return x, mem, [(wsp - 0x100, 0x100)]

    def g_wyhash(self):
        rng = self.rng
        x = self.base_regs()
        wsp = rng.randrange(0x200, 0xF0000) & ~15
        n = rlen(rng, 400)
        x[10], x[11], x[12] = INST, SCR + rng.randrange(0, 0x1000), n
        mem = {SCR: rbytes(rng, WIN), INST: struct.pack("<III", 0, 0, wsp), wsp - 0x60: rbytes(rng, 0x60)}
        return x, mem, [(wsp - 0x60, 0x60)]

    def g_tokest(self):
        rng = self.rng
        x = self.base_regs()
        n = max(1, rlen(rng, 1200)) if rng.random() < 0.98 else 0
        p = SCR + rng.randrange(0, 0x1000)
        ws = rng.random() * 0.5
        data = sprinkle(rng, rng.randbytes(WIN), WS, ws)
        k = rng.random()
        x[12], x[13] = (rng.randrange(0, 40), 0) if k < 0.9 else (rng.getrandbits(32), rng.getrandbits(2))
        x[11], x[10] = (rng.randrange(0, 3000), 0) if rng.random() < 0.9 else (rng.getrandbits(32), rng.getrandbits(1))
        x[14] = 5 if rng.random() < 0.97 else rng.randrange(8)
        x[15] = 0x20 if rng.random() < 0.97 else rng.randrange(64)
        x[21], x[16] = p, n
        return x, {SCR: data}, []

    def plainish(self, n, special, p_special):
        rng = self.rng
        return sprinkle(rng, rbytes(rng, n, range(0x20, 0x7F)), special, p_special)

    def g_jscan(self):
        rng = self.rng
        x = self.base_regs()
        i0 = rng.randrange(0, 600)
        data = self.plainish(WIN, b'"\\\x00\x01\x1f\x80\xc3\xf5\xff', rng.choice([0.01, 0.05, 0.2, 0.6]))
        end = i0 + 1 + rlen(rng, 1500) if rng.random() < 0.9 else i0 + 1
        x[13], x[14], x[10] = i0, end, SCR
        if rng.random() < 0.9:
            x[12] = FRAME + 4 * rng.randrange(0, 64)
        else:
            x[12] = SCR + 64 + i0 + rng.randrange(-40, 40) & ~3
        return x, {SCR: data, FRAME: rbytes(rng, 0x200)}, [(FRAME, 0x200)]

    def g_jesc(self):
        rng = self.rng
        x = self.base_regs()
        i0 = rng.randrange(0, 600)
        data = self.plainish(WIN, b'"\\\x00\x01\x1f\x20\x21\x22\x5c\x80\xff', rng.choice([0.01, 0.05, 0.2, 0.6]))
        ln = i0 + 1 + rlen(rng, 1500)
        k = rng.random()
        q3, q7 = (0x5C, 0x22) if k < 0.45 else (0x22, 0x5C) if k < 0.9 else (rng.randrange(256), rng.randrange(256)) \
            if k < 0.97 else (rng.getrandbits(32), 0x22)
        s0 = FRAME + 0x100
        x[8], x[19], x[23], x[24], x[21] = s0, q3, q7, i0, SCR
        mem = {SCR: data, FRAME: rbytes(rng, 0x200)}
        mem[s0 - 0x3C] = struct.pack("<I", ln)
        return x, mem, [(FRAME, 0x200)]

    def g_ascii(self):
        rng = self.rng
        x = self.base_regs()
        i0 = rng.randrange(0, 600)
        p_hi = rng.choice([0.0, 0.01, 0.1, 0.5])
        data = sprinkle(rng, rbytes(rng, WIN, range(0x80)), range(0x80, 256), p_hi)
        x[10], x[19], x[20] = i0, i0 + rlen(rng, 1500) if rng.random() < 0.95 else rng.randrange(0, i0 + 1), SCR
        return x, {SCR: data}, []

    def g_maskp(self):
        rng = self.rng
        x = self.base_regs()
        alpha = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_=-:. \n\""
        text = bytearray(rbytes(rng, 1024, alpha))
        pos = rng.randrange(1, 900)
        if rng.random() < 0.5:
            pre = rng.choice(U.SECRET_PREFIXES)
            cut = len(pre) if rng.random() < 0.7 else rng.randrange(1, len(pre))
            s = "".join(c.lower() if rng.random() < 0.3 else c for c in pre[:cut]).encode()
            text[pos:pos + len(s)] = s
        tlen = pos + rng.randrange(0, 60) if rng.random() < 0.9 else 1024
        x[26], x[24], x[19] = SCR + pos, pos, tlen
        x[22] = rng.choice([0, 0, 1, rng.getrandbits(32)])
        x[21] = self.table[0]["params"][2] if rng.random() < 0.97 else rng.getrandbits(32)
        x[20] = 0x5F if rng.random() < 0.98 else rng.randrange(256)
        x[17] = 10 if rng.random() < 0.98 else rng.randrange(16)
        return x, {SCR: bytes(text)}, []

    def g_memchr(self):
        rng = self.rng
        x = self.base_regs()
        n = max(1, rlen(rng, 800))
        p = SCR + rng.randrange(0, 0x1000)
        data = bytearray(rbytes(rng, WIN, b"abcdefgh\r"))
        c = rng.choice(b"\n\r") if rng.random() < 0.9 else rng.getrandbits(32)
        if rng.random() < 0.5:
            data[p - SCR + rng.randrange(n)] = c & 0xFF
        x[11], x[12], x[10] = p, n if rng.random() < 0.99 else 0, c
        return x, {SCR: bytes(data)}, []

    def g_bcopy(self):
        rng = self.rng
        x = self.base_regs()
        n = max(1, rlen(rng, 600))
        s = SCR + 64 + rng.randrange(0, 0x1000)
        d = s + rng.randint(-24, 24) if rng.random() < 0.4 else SCR + rng.randrange(0, 0x1000)
        x[27], x[12], x[11] = s, d, s + n if rng.random() < 0.99 else s
        return x, {SCR: rbytes(rng, WIN)}, []

    GEN = {1: "g_move", 2: "g_move", 3: "g_set", 4: "g_eql", 5: "g_tokest", 6: "g_jscan", 7: "g_jesc",
           8: "g_ascii", 9: "g_maskp", 10: "g_memchr", 11: "g_bcopy", 12: "g_sha", 13: "g_wyhash",
           14: "g_move_ir", 15: "g_eql_ir"}

    def g_move_ir(self):
        """g_move, with half of the cases 4-aligned (the circuit's coverage)."""
        x, mem, extra = self.g_move()
        if self.rng.random() < 0.5:
            x[10] &= ~3
            x[11] &= ~3
            x[12] = 4 * self.rng.randint(0, 24)
        return x, mem, extra

    def g_eql_ir(self):
        x, mem, extra = self.g_eql()
        x[10] = 1  # set by eqlBytes before its beq a1, a3
        if self.rng.random() < 0.6:
            x[12] = x[14] = self.rng.randint(0, 34)
        return x, mem, extra

    def case(self, spec, idx):
        m = self.m
        x, mem, extra = getattr(self, self.GEN[spec["kind"]])()
        for a, b in mem.items():
            m.wr(a, b)
        wins = [(SCR, WIN)] + extra
        before = {a: bytearray(m.rd(a, n)) for a, n in wins}

        def rd(a):
            for w, n in wins:
                if w <= a < w + n:
                    return before[w][a - w]
            return m.rd(a, 1)[0]

        ref = R.REF[spec["kind"]](x, rd, spec["params"])
        m.set_regs(x)
        m.e.rv_clear_poison()
        st, steps, otok, utok, fail = m.check(spec["pc"])
        res = {"status": st, "steps": steps, "otok": otok, "utok": utok}
        if st < 0:
            res["error"] = f"C unit != original: {fail}"
            return res
        if st == 0:
            if ref is not None:
                res["error"] = "C declined, reference fired"
            return res
        if ref is None:
            res["error"] = "C fired, reference declined"
            return res
        xi, regs, writes = ref
        dead = spec["dead"][xi]
        want = list(x)
        for r, v in regs.items():
            if not dead >> r & 1:
                want[r] = v & 0xFFFFFFFF
        got = m.get_regs()
        if got[1:] != want[1:]:
            bad = [(U.REG[i], hex(got[i]), hex(want[i])) for i in range(1, 32) if got[i] != want[i]]
            res["error"] = f"registers vs reference: {bad[:4]}"
            return res
        want_pc = MC.SENTINEL if not spec["params"] or spec.get("returns") else spec["params"][xi]
        if m.e.rv_get_pc() != want_pc:
            res["error"] = f"exit PC {m.e.rv_get_pc():#x} vs reference {want_pc:#x}"
            return res
        for a, n in wins:
            exp = before[a]
            for ad, b in writes.items():
                if a <= ad < a + n:
                    exp[ad - a] = b
            now = m.rd(a, n)
            if now != bytes(exp):
                k = next(i for i in range(n) if now[i] != exp[i])
                res["error"] = f"memory vs reference at {a + k:#x}: {now[k]} != {exp[k]}"
                return res
        outside = [ad for ad in writes if not any(w <= ad < w + n for w, n in wins)]
        if outside:
            res["error"] = f"reference writes outside the windows: {hex(outside[0])}"
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("elf")
    ap.add_argument("out")
    ap.add_argument("--cases", type=int, default=100000)
    ap.add_argument("--units", default="all")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    m = MC.Machine(a.dll, a.elf, a.units)
    m.e.rv_units_mode(2)
    fz = Fuzzer(m, a.seed)
    seen_kind = set()
    report = []
    for spec in m.specs:
        if spec["kind"] in seen_kind and "2nd copy" in spec["label"]:
            continue
        seen_kind.add(spec["kind"])
        t0 = time.perf_counter()
        n = a.cases
        stats = {"unit": spec["label"], "cases": 0, "fired": 0, "declined": 0, "errors": 0, "first_errors": [],
                 "orig_instructions": 0, "orig_tokens": 0, "unit_tokens": 0}
        for i in range(n):
            r = fz.case(spec, i)
            stats["cases"] += 1
            if r["status"] == 1:
                stats["fired"] += 1
                stats["orig_instructions"] += r["steps"]
                stats["orig_tokens"] += r["otok"]
                stats["unit_tokens"] += r["utok"]
            elif r["status"] == 0:
                stats["declined"] += 1
            if "error" in r:
                stats["errors"] += 1
                if len(stats["first_errors"]) < 5:
                    stats["first_errors"].append(r["error"])
        stats["seconds"] = round(time.perf_counter() - t0, 1)
        report.append(stats)
        print(f"{spec['label'][:52]:52s} cases {stats['cases']:7d} fired {stats['fired']:7d} declined {stats['declined']:6d} "
              f"errors {stats['errors']:4d} {stats['seconds']:6.1f}s {stats['first_errors'][:1]}", flush=True)
    json.dump({"elf": os.path.abspath(a.elf), "seed": a.seed, "cases_per_unit": a.cases, "units": report},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
