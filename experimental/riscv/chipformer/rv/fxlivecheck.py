"""The live I/O path on Executor v4 at fx's size (docs/fx/live.md): the GPU's checks.

    python -m chipformer.rv.fxlivecheck LOG.npz ELF [--out OUT.json]

From fx's boot snapshot: the registers, 8 KB out of the image, a capture of the largest real reply
(3,314 bytes, unaligned, across pages) and of 9 KB (past the sled), read back through the chip;
every chip token against the reference (fxlive.audit); and the whole stream teacher-forced, the
residual against ir.evaluate on every dim.
"""

import argparse
import json
import time

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.rv import core, exec4
from chipformer.rv import fxlive as FL
from chipformer.rv import spec as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("elf")
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--tf", type=int, default=20000)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.log)
    bios = FL.Bios(FL.FX_IO_BASE)
    b = {"ad": z["boot_ad"], "val": z["boot_val"], "x": z["boot_x"], "pc": int(z["boot_pc"])}
    mk = lambda: FL.machine_at(a.elf, b["ad"], b["val"], b["x"], b["pc"], a.frames, bios)
    cfg, m = mk()
    c = core.build(cfg=cfg)
    print("built", c.stats(), round(time.time() - t0, 1), flush=True)
    core.snapshot(c, m)
    ex = exec4.Exec4(c, text=None, v0=0, tables=exec4.cached_tables(c, None, 0))
    ex.load_init()
    FL.install_stops(ex)
    S0 = ex.S.clone()
    print("exec4", ex.NPH, ex.n_slots, ex.smem, round(time.time() - t0, 1), flush=True)
    be = FL.ExecBackend(ex)
    # the rows of the table path are the embedding's
    sample = [0, 5, S.tok("V", 77), S.tok("JF", 3), S.tok("HALT"), S.tok("SS1"), S.tok("PA", 9), S.tok("R", 400)]
    assert bool((be.rows(sample) == ex.embed(sample)).all())
    ex.reset()
    cpu = FL.LiveCPU(be, bios, cfg, full_regs=True)
    res = {"frames": a.frames}
    chip, stop = cpu.go()
    kind, pc = cpu.parse_stop(chip, stop)
    regs = cpu.enter()
    res["registers_equal"] = all(regs[r] == int(b["x"][r]) for r in regs) and kind == "ecall" and pc == b["pc"]
    # out of the image: 8 KB at the start of the boot-written heap (ROM of the snapshot's words)
    addr = 0xD9270 & ~3
    got = cpu.read_words(addr, 2048)
    res["read_words"] = 2048
    res["read_diff"] = sum(1 for i, v in enumerate(got) if v != m.ram_word(addr + 4 * i))
    rng = np.random.default_rng(1)
    caps = []
    for n, at in ((3314, 0x280000 + 509), (9000, 0x290000 + 2)):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        before = {k: cpu.tokens["io"] for k in [0]}[0]
        t1 = time.perf_counter()
        cpu.write(at, data)
        dt = time.perf_counter() - t1
        cpu.cache.clear()
        back = cpu.read(at, n)
        caps.append({"bytes": n, "addr": hex(at), "tokens": cpu.tokens["io"] - before, "write_s": round(dt, 3),
                     "read_back_equal": back == data})
    res["captures"] = caps
    res["forced_positions_checked"] = cpu.ok_forced
    toks, lens, forced = cpu.fed()
    r = FL.audit(mk()[1], toks, lens, forced)
    res["audit"] = r
    print(json.dumps(res), flush=True)
    # the whole stream teacher-forced: every dim, every position
    n = min(a.tf, len(toks))
    toks = toks[:n]
    ex.S.copy_(S0)
    ex.reset()
    _, _, _, xs = ex.run(ex.embed(toks[:n].tolist()).contiguous(), mode=0, max_steps=n, dbg=True)
    _, m2 = mk()
    core.snapshot(c, m2)
    carry, bad, first = {}, 0, None
    for s in range(0, n, 2048):
        want = ir.evaluate(c, toks[s:s + 2048].tolist(), check=False, carry=carry)
        got = xs[s:s + 2048]
        diff = np.nonzero((got != want).any(1))[0]
        if len(diff) and first is None:
            d = int(np.nonzero(got[diff[0]] != want[diff[0]])[0][0])
            first = (s + int(diff[0]), c.dims[d].name, float(got[diff[0], d]), float(want[diff[0], d]))
        bad += len(diff)
        print("tf", s + len(got), bad, round(time.time() - t0), flush=True)
    res["tf"] = {"positions": n, "dims": ex.D, "differing": bad, "first": first}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
