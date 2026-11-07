"""Executor v4 on the paged map (docs/fx/memory.md): the GPU checks. Standalone circuits (no text).

    python -m chipformer.rv.memcheck tf    [--progs a b] [--n 4000]  # the residual against the IR
    python -m chipformer.rv.memcheck free  [--progs a b]             # free-running vs the reference
    python -m chipformer.rv.memcheck time  [--progs c-sieve] [--frames 8 16 ..]  # step time
    python -m chipformer.rv.memcheck big                             # the big-map programs

``small`` is spec.small_cfg: the existing ELFs at their own addresses, on the paged machinery.
``time`` measures banks on and off, the same program, over the frame count (RAM layers).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.rv import core, exec4, ref
from chipformer.rv import spec as S
from chipformer.rv.measure import DATA


def elf_names() -> list:
    return sorted(json.loads((DATA / "qemu.json").read_text()))


def mk_elf(cfg, name):
    return lambda: ref.PagedMachine.from_elf(cfg, DATA / f"{name}.elf")


def make(c, **kw):
    return exec4.Exec4(c, text=None, v0=0, tables=exec4.cached_tables(c, None, 0), **kw)


def tf(c, ex, mk, n_max) -> dict:
    """Teacher-forced: every dim of the residual after every step equals the IR's."""
    toks = mk().run(n_max + 2)[:n_max]
    core.snapshot(c, mk())
    ex.load_init()
    ex.reset()
    _, _, _, xs = ex.run(ex.embed(toks).contiguous(), mode=0, max_steps=len(toks), dbg=True)
    carry, bad, first = {}, 0, None
    core.snapshot(c, mk())
    for a in range(0, len(toks), 1024):
        want = ir.evaluate(c, toks[a:a + 1024], check=False, carry=carry)
        got = xs[a:a + 1024]
        diff = np.nonzero((got != want).any(1))[0]
        if len(diff) and first is None:
            d = int(np.nonzero(got[diff[0]] != want[diff[0]])[0][0])
            first = (a + int(diff[0]), c.dims[d].name, float(got[diff[0], d]), float(want[diff[0], d]))
        bad += len(diff)
    return {"positions": len(toks), "differing": bad, "first": first}


def free(c, ex, mk, max_steps=None) -> dict:
    """From GO on the GPU alone: every token equals the reference machine's."""
    m = mk()
    toks = m.run(10**8)
    core.snapshot(c, mk())
    ex.load_init()
    ex.reset()
    n = len(toks) - 1 if max_steps is None else min(len(toks) - 1, max_steps)
    go = ex.embed(toks[:1]).contiguous()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = ex.run(go, mode=1, max_steps=n, keep=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    want = np.asarray(toks[1:n + 1], np.int64)
    same = len(ids) == n and bool((ids == want).all())
    at = None if same else int(np.argmax(ids[:min(n, len(ids))] != want[:min(n, len(ids))]))
    return {"tokens": n, "instructions": m.steps, "same": same, "at": at, "us_step": 1e6 * dt / n,
            "tok_s": n / dt}


def big_programs(cfg):
    sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
    import test_rv_mem as T

    src = "\n".join(["li s1, 0x12345678"] + [
        f"li s0, {b}\nsw s1, 0(s0)\nsb s1, 5(s0)\nsh s1, 10(s0)\nlw a0, 0(s0)\nlbu a1, 5(s0)\nlhu a2, 10(s0)"
        for b in (0x100, 0x7000, 0x100000, 0x108000, 0x3F0000, 0x3F7FF0)])
    return {"far": lambda: ref.PagedMachine.from_segments(T.BIG, T.BIG.rom_base, T.far_program()),
            "ram": lambda: T.big_machine(src)}, T.BIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tf", "free", "time", "big"))
    ap.add_argument("--progs", nargs="*")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--frames", nargs="*", type=int, default=[128])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    if a.mode == "big":
        progs, cfg = big_programs(None)
        c = core.build(cfg=cfg)
        print(c.stats(), flush=True)
        ex = None
        for nm, mk in progs.items():
            core.snapshot(c, mk())
            ex = ex or make(c)
            r = {"prog": nm, "tf": tf(c, ex, mk, a.n), "free": free(c, ex, mk)}
            print(json.dumps(r), flush=True)
            res.append(r)
    elif a.mode == "time":
        for frames in a.frames:
            cfg = S.small_cfg(frames)
            c = core.build(cfg=cfg)
            for banks in (True, False):
                nm = (a.progs or ["c-sieve"])[0]
                mk = mk_elf(cfg, nm)
                core.snapshot(c, mk())
                try:
                    ex = make(c, banks=banks)
                except (AssertionError, RuntimeError) as e:
                    print(json.dumps({"frames": frames, "banks": banks, "error": str(e)[:120]}), flush=True)
                    continue
                best = None
                for _ in range(3):
                    r = free(c, ex, mk, 20000)
                    best = r if best is None or r["us_step"] < best["us_step"] else best
                row = {"frames": frames, "banks": banks, "layers": c.stats()["layers"], "phases": ex.NPH,
                       "slots": int(ex.n_slots), "smem": int(ex.smem), **best}
                print(json.dumps(row), flush=True)
                res.append(row)
                del ex
                torch.cuda.empty_cache()
    else:
        cfg = S.small_cfg()
        c = core.build(cfg=cfg)
        print(c.stats(), flush=True)
        ex = None
        names = a.progs or elf_names()
        tot = bad = toks = 0
        for nm in names:
            mk = mk_elf(cfg, nm)
            core.snapshot(c, mk())
            ex = ex or make(c)
            if a.mode == "tf":
                r = tf(c, ex, mk, a.n)
                tot += r["positions"]
                bad += r["differing"]
            else:
                r = free(c, ex, mk)
                toks += r["tokens"]
                bad += not r["same"]
            print(json.dumps({"prog": nm, **r}), flush=True)
            res.append({"prog": nm, **r})
        print(json.dumps({"mode": a.mode, "programs": len(names), "positions": tot, "tokens": toks,
                          "bad": bad}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
