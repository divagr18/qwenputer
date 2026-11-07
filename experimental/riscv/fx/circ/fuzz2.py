"""Differential fuzz of the built units (chipformer/rv/hot.py) on fx's own image: the unit's reference
against stepping the original instructions on the same paged machine.

    python fx/circ/fuzz2.py LOG.npz ELF OUT.json --unit memmove --cases 100000 --seed 1

A case is a random machine state at the unit's trigger: registers, written words, which pages have
a frame, how many frames are left. Both runs start from it; the unit's run has the unit on. The run
stops at the sentinel return address (function units) or at the unit's exit PCs (loop units).
Compared afterwards: fault, PC, every register live there, every word either run wrote, except the
C stack (the original's own frames, dead below sp). Generators cover page crossings, unmapped pages,
image-backed words, overlap, zero length, every alignment and the frame budget's edge.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from chipformer.rv import fxrun, hot, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

CALLEE = (1, 2, 3, 4, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
SP = 0x07FF0000
SENT = 0x2000000  # the sentinel return address: _start's first word (the run stops on it)


def make(log, elf, name):
    cfg, m, _, _ = fxrun.load(log, elf, False, 1024, [name])
    plain = ref.PagedMachine(rom=m.rom, ram=[], x=list(m.x), anchor=m.anchor, cfg=replace(cfg, units=()),
                             image=m.image)
    plain.rom_base = cfg.rom_base
    plain.macros = {}
    return m, plain, cfg


def reset(m, x, pc, wr, pt, nf):
    m.x = list(x)
    m.anchor, m.n = (pc - m.rom_base) >> 2, 0
    m.wr, m.pt, m.nf = dict(wr), dict(pt), nf
    m.halted, m.fault, m.skip_unit = False, None, False
    m.steps = 0


def run_until(m, stops, limit=400000):
    n = 0
    toks = 0
    while not m.halted and n < limit:
        if m.pc in stops and n:
            return toks
        if m.n == 0 and m.pc in stops:
            return toks
        toks += len(m.record())
        n += 1
    return toks


def mem_diff(a, b, skip_lo, skip_hi):
    keys = set(a.wr) | set(b.wr)
    out = []
    for k in keys:
        ad = k << 2
        if skip_lo <= ad < skip_hi:
            continue
        if a.ram_word(ad) != b.ram_word(ad):
            out.append((hex(ad), hex(a.ram_word(ad)), hex(b.ram_word(ad))))
    return out


def random_pages(rng, pages, cfg, mode, m):
    """wr, pt, nf for the pages a case touches. mode 0: random; 1: all mapped; 2: none mapped."""
    wr, pt = {}, {}
    fr = 1
    for pg in sorted(pages):
        r = rng.random()
        if mode == 2 or (mode == 0 and r < 0.3):
            continue
        pt[pg] = fr
        fr += 1
        if cfg.rw_window(pg << S.PAGE_BITS) is None:
            continue
        kind = rng.randrange(3)  # no word written (image-backed) / a few / all
        for a in range(pg << 7, (pg + 1) << 7):
            if kind == 2 or (kind == 1 and rng.random() < 0.3):
                wr[a] = rng.getrandbits(32)
    return wr, pt, fr - 1


def addr_near_page(rng, lo, hi):
    pg = rng.randrange(lo >> 9, hi >> 9)
    r = rng.random()
    if r < 0.5:
        return (pg << 9) + rng.randrange(512)
    return (pg << 9) + rng.choice([0, 1, 2, 3, 4, 5, 508, 509, 510, 511, 256, 257])


def lengths(rng):
    r = rng.random()
    if r < 0.03:
        return 0
    if r < 0.45:
        return rng.randint(1, 16)
    if r < 0.75:
        return rng.randint(17, 128)
    if r < 0.93:
        return rng.randint(129, 1100)
    if r < 0.99:
        return rng.randint(1100, 3000)
    return rng.choice([0x3FF8, 0x3FFB, 0x3FFC, 0x3FFF, 0x4000, 0x4001])


def gen_move(rng, cfg, m):
    img = [w << S.WIN_BITS for w in cfg.image if w << S.WIN_BITS < 1 << 22]
    base = rng.choice(img + [0x100000, 0x200000, 0x300000])
    d = addr_near_page(rng, base, base + 0x8000)
    n = lengths(rng)
    r = rng.random()
    if r < 0.3:
        s = (d + rng.randint(-48, 48)) & 0x3FFFFF
    elif r < 0.4:
        s = (d + rng.randint(-n - 8, n + 8)) & 0x3FFFFF
    elif r < 0.45:
        s = d
    else:
        s = addr_near_page(rng, base, base + 0x8000)
    r = rng.random()
    if r < 0.02:
        d = 0x3FFFF0 - rng.randrange(0x5000)  # the window's end
    elif r < 0.04:
        s = 0x3FFFF0 - rng.randrange(0x5000)
    elif r < 0.05:
        d = 0x026C0000 + rng.randrange(0x1000)  # another writable window: a decline
    elif r < 0.06:
        s = 0x07FD0000 + rng.randrange(0x1000)
    pages = set(range(d >> 9, (d + n + 8 >> 9) + 1)) | set(range(s >> 9, (s + n + 8 >> 9) + 1))
    if len(pages) > 60:
        pages = set(sorted(pages)[:60])
    wr, pt, nf = random_pages(rng, pages, cfg, rng.choice([0, 0, 1, 2]), m)
    r = rng.random()
    if r < 0.08:
        nf = rng.randint(cfg.frames - 40, cfg.frames - 1)  # the budget's edge
    x = [rng.getrandbits(32) for _ in range(32)]
    x[0] = 0
    x[10], x[11], x[12] = d, s, n
    x[2] = SP - rng.randrange(0x100) * 16
    x[1] = SENT
    return x, wr, pt, nf


def gen_set(rng, cfg, m):
    x, wr, pt, nf = gen_move(rng, cfg, m)
    x[11] = rng.choice([0, 0x22, 0xFF, rng.getrandbits(32)])
    return x, wr, pt, nf


def gen_bcopy(rng, cfg, m):
    x, wr, pt, nf = gen_move(rng, cfg, m)
    d, s, n = x[10], x[11], max(x[12], 1)
    x[12], x[27], x[11] = d, s, (s + n) & 0xFFFFFFFF
    return x, wr, pt, nf


def gen_eql(rng, cfg, m):
    """a1, a3, a2: a0 = 1. Bytes equal up to a point, in any mix of written and image words."""
    x, wr, pt, nf = gen_move(rng, cfg, m)
    a, b, n = x[10], x[11], max(x[12], 1)
    n = min(n, 4000)
    r = rng.random()
    if r < 0.15:  # .rodata (the image), and outside every load region (a fault)
        b = 0x2680000 + rng.randrange(0x20000)
    elif r < 0.2:
        b = rng.choice([0x05000000, 0x03000000 - rng.randrange(8) - 2, 0x01FFFFF0]) + rng.randrange(8)
    x[11], x[13], x[12], x[14] = a, b, n, n
    x[10] = 1
    if rng.random() < 0.7 and cfg.in_load(a) and cfg.in_load(b + n) and cfg.in_load(a + n):
        cut = rng.choice([n, n, rng.randrange(n + 1)])
        for k in range(cut):  # b's bytes become a's (read through both layers)
            ea, eb = a + k, b + k
            if cfg.rw_window(eb) is None or not (eb >> 2 == eb >> 2):
                break
            byte = (wr.get(ea >> 2, m.image.get(ea >> 2, 0)) >> 8 * (ea & 3)) & 0xFF
            w = wr.get(eb >> 2, m.image.get(eb >> 2, 0))
            wr[eb >> 2] = w & ~(0xFF << 8 * (eb & 3)) & 0xFFFFFFFF | byte << 8 * (eb & 3)
            pt.setdefault(eb >> S.PAGE_BITS, 1)
    return x, wr, pt, nf


def text_bytes(rng, kind, n):
    plain = {"tok": range(0x21, 0x7F), "chr": range(0x20, 0xFF), "ascii": range(0, 0x80), "jscan": [c for c in range(0x20, 0x80) if c not in (0x22, 0x5C)]}[kind]
    stop = {"tok": [9, 10, 11, 12, 13, 32], "chr": [0x0A], "ascii": [0x80, 0xC3, 0xFF], "jscan": [0x22, 0x5C, 0, 0x1F, 0x80, 0xFF]}[kind]
    p_stop = rng.choice([0.0, 0.01, 0.05, 0.3])
    return bytes(rng.choice(stop) if rng.random() < p_stop else rng.choice(plain) for _ in range(n))


def gen_scan(kind):
    def gen(rng, cfg, m):
        img = [w << S.WIN_BITS for w in cfg.image if w << S.WIN_BITS < 1 << 22]
        base = rng.choice(img + [0x100000, 0x200000])
        p = addr_near_page(rng, base, base + 0x8000)
        n = rng.choice([1, 2, 3, 4, 8, 20, 61, 62, 63, 64, 65, 66, 100, 130, 200, 700])
        txt = text_bytes(rng, kind, n + 8)
        pages = set(range(p >> 9, (p + n + 16 >> 9) + 1))
        wr, pt, nf = random_pages(rng, pages, cfg, rng.choice([0, 1]), m)
        for k, b in enumerate(txt):  # the text, in written words (a page has a frame then)
            ea = p + k
            if rng.random() < 0.97:
                w = wr.get(ea >> 2, m.image.get(ea >> 2, 0))
                wr[ea >> 2] = w & ~(0xFF << 8 * (ea & 3)) & 0xFFFFFFFF | b << 8 * (ea & 3)
                pt.setdefault(ea >> S.PAGE_BITS, 1)
        x = [rng.getrandbits(32) for _ in range(32)]
        x[0] = 0
        x[2] = SP
        x[1] = SENT
        if rng.random() < 0.04:
            n = rng.choice([0x10000, 0x10001, 0xFFFFFFFF, 0])
        if kind == "tok":
            x[21], x[16], x[12] = p, n, rng.choice([0, 3, 100, 0x7FFFFFF0, 0xFFFFFFFE, rng.getrandbits(16)])
            x[13], x[10], x[11], x[14], x[15] = rng.choice([0, 0, 1]), 0, rng.getrandbits(16), 5, 0x20
        elif kind == "chr":
            x[11], x[12], x[10] = p, n, rng.choice([0x0A, 0x0A, 0x0D, 0x10A])
        elif kind == "ascii":
            i = rng.randrange(0, 20)
            x[20], x[10], x[19] = p - rng.randrange(1), i, rng.choice([i + 1 + n, i + 1, i, i + 2, rng.getrandbits(17)])
        else:
            i = rng.randrange(0, 20)
            r = rng.random()
            cur = (p + rng.choice([-0x24, -0x20, 8, 40, n + 8]) if r < 0.1 else p + 0x2000 + rng.randrange(0x2000)) & ~3
            if r > 0.97:
                cur = 0x07FD0000
            x[10], x[13], x[14], x[12] = p, i, rng.choice([i + 1 + n, i + 2, i + 1, 0x10002]), (cur - 0x24) & 0xFFFFFFFF
            if rng.random() < 0.03:
                x[12] += 2
        return x, wr, pt, nf
    return gen


GEN = {"move": gen_move, "cpy": gen_move, "set": gen_set, "bcopy": gen_bcopy, "eql": gen_eql, "tok": gen_scan("tok"),
       "chr": gen_scan("chr"), "ascii": gen_scan("ascii"), "jscan": gen_scan("jscan")}
INDUCTION = {"tok": 16, "chr": 12, "ascii": 10, "jscan": 13}
DEAD = {"tok": {5, 17}, "chr": {13}, "ascii": {29, 30}, "jscan": {11, 15, 16}}
ALLR = set(range(1, 32))
LIVE = {"bcopy": sorted(ALLR - {1, 5, 6, 7, 8, 10 + 1, 12, 13, 14, 15, 16, 17, 19, 23, 27, 28, 29, 30, 31})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("elf")
    ap.add_argument("out")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--cases", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    m, plain, cfg = make(a.log, a.elf, a.unit)
    u = cfg.units[0]
    gen = GEN[u.kind]
    rng = random.Random(a.seed)
    orig = hot.record
    fired = [0]

    def rec(mm, uu):
        r = orig(mm, uu)
        fired[0] += r is not None
        return r

    hot.record = rec
    st = {"cases": 0, "fired": 0, "declined": 0, "errors": 0, "faulted": 0, "tok_unit": 0, "tok_orig": 0}
    t0 = time.time()
    errs = []
    for k in range(a.cases):
        x, wr, pt, nf = gen(rng, cfg, m)
        reset(m, x, u.pc, wr, pt, nf)
        reset(plain, x, u.pc, wr, pt, nf)
        fired[0] = 0
        if u.kind in INDUCTION:  # one firing against the iterations it stands for
            ta = len(m.record())
            st["cases"] += 1
            if not fired[0] or m.skip_unit:
                st["declined"] += 1
                bad = [("declined, but changed", r) for r in range(32) if m.x[r] != x[r]]
                if m.wr != wr:
                    bad.append("declined, but wrote")
            else:
                st["fired"] += 1
                ind, tv = INDUCTION[u.kind], m.x[INDUCTION[u.kind]]
                tb = 0
                for _ in range(1000000):
                    if plain.halted:
                        break
                    tb += len(plain.record())
                    if plain.pc == u.pc and plain.x[ind] == tv:
                        break
                st["tok_unit"] += ta
                st["tok_orig"] += tb
                bad = []
                if plain.pc != m.pc or plain.halted:
                    bad.append(("pc", hex(plain.pc), hex(m.pc), plain.fault))
                for r in range(1, 32):
                    if r not in DEAD[u.kind] and m.x[r] != plain.x[r]:
                        bad.append(("x", r, hex(m.x[r]), hex(plain.x[r])))
                bad += mem_diff(m, plain, 0x07FC0000 if u.kind != "jscan" else 0, 0x08000000 if u.kind != "jscan" else 0)[:4]
            if bad:
                st["errors"] += 1
                if len(errs) < 5:
                    errs.append({"case": k, "x": [hex(x[r]) for r in (10, 11, 12, 13, 14, 16, 19, 20, 21)], "bad": bad[:6]})
                    print(json.dumps(errs[-1]), flush=True)
            if k % 10000 == 9999:
                print(k + 1, st, round(time.time() - t0), flush=True)
            continue
        stops = {SENT} if u.kind != "bcopy" else {u.ret}
        live = LIVE.get(u.kind, (*CALLEE, 10))
        ta = run_until(m, stops)
        tb = run_until(plain, stops)
        st["cases"] += 1
        st["fired"] += fired[0]
        st["declined"] += 1 - fired[0]
        if fired[0]:
            st["tok_unit"] += ta
            st["tok_orig"] += tb
        bad = []
        if m.fault != plain.fault or m.halted != plain.halted:
            bad.append(("fault", m.fault, plain.fault))
        if m.fault:
            st["faulted"] += 1
        elif m.pc != plain.pc:
            bad.append(("pc", hex(m.pc), hex(plain.pc)))
        if not m.fault:
            for r in live:
                if m.x[r] != plain.x[r]:
                    bad.append(("x", r, hex(m.x[r]), hex(plain.x[r])))
            bad += mem_diff(m, plain, 0x07FC0000, 0x08000000)[:4]
        if bad:
            st["errors"] += 1
            if len(errs) < 5:
                errs.append({"case": k, "x": [hex(x[r]) for r in (10, 11, 12, 13, 27)], "bad": bad[:6]})
                print(json.dumps(errs[-1]), flush=True)
        if k % 10000 == 9999:
            print(k + 1, st, round(time.time() - t0), flush=True)
    st["seconds"] = round(time.time() - t0)
    st["unit"] = a.unit
    st["errs"] = errs
    json.dump(st, open(a.out, "w"))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
