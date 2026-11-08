"""docs/fx/units-file.md: the units' test programs (tests/test_rv_hot.py), run on Executor v4 from a chip
file, against the Python circuit in the IR and against the reference machine.

    python fx/fxio/units_tf.py OUT.json [--random 12] [--work DIR]

One circuit per program on the test map, ROM-specialized as fx's is (hot.rom_absent, as
test_ir_pruned): the unpruned test circuit is 5,410 dims wide, past an L4's shared memory under
Executor v4. Programs: FIXED (every unit kind, its bcopy loops included), SHA-256 calls, and random
calls as test_ir_random. Each circuit is written by chipbuild.standalone, which fails on a weight that
is not a finite bf16 value; then lifted from the file (chipfile.load), linted against core.build, and
planned by Executor v4. The program is loaded onto the file's circuit with core.snapshot. From GO:
  tf    teacher-forced on the reference's stream: every IR dim of the residual after the chip's
        layers against ir.evaluate of the Python circuit (chipcheck.dim_map);
  free  the GPU alone from GO: every token against the reference machine.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(os.path.dirname(HERE))  # experimental/riscv
sys.path[:0] = [RV, os.path.join(RV, "tests")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_rv_hot as T  # noqa: E402
from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.ir import Head  # noqa: E402
from chipformer.rv import chipbuild, chipcheck, core, exec4, hot  # noqa: E402
from chipformer.rv import chipfile as F  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402


def programs(n_random: int) -> list:
    """(name, calls, seed, readback, circuit key)."""
    out = []
    m0 = T.machine(T.FIXED, False, 0)[0]
    m0.run(10**7)
    rb = sorted(a for a, v in m0.wr.items() if a < 0x40000 >> 2 and v != m0.image.get(a, 0))[:400]
    out.append(("fixed", T.FIXED, 0, rb, "fixed"))
    sha = [("sha", 0x11000, 0x12000, 0x10040, 0x12000), ("sha", 0x11000, 0x13440, 0x10080, 0x13440),
           ("sha", 0x11000, 0x14480, 0x10080, 0x14460)]
    out.append(("sha", sha, 0, (), "shared"))
    for seed in range(n_random):
        rng = random.Random(100 + seed)
        calls = T.random_calls(rng, 5, ("move", "cpy", "set", "eql", "scan", "scan"))
        out.append((f"random{seed}", calls, seed, (), "shared"))
    return out


def tiny_of(c) -> dict:
    """IR dim -> its head's ``tiny`` (outputs below it may round: a 2^-k counter far down)."""
    out = {}
    for n in c.nodes:
        if isinstance(n, Head) and n.tiny:
            for vh in n.vheads():
                for _, d, _ in vh.out:
                    out[d] = n.tiny
    return out


def pruned_cfg(calls, seed, rb):
    m, cfg = T.machine(calls, True, seed, rb)
    return replace(cfg, absent=hot.rom_absent(m.rom, cfg))


def run(a) -> dict:
    res = {"circuits": {}, "programs": []}
    progs = [p for p in programs(a.random) if a.only is None or p[0] in a.only]
    for name, calls, seed, rb, _ in progs:
        key = name
        t0 = time.time()
        cfg = pruned_cfg(calls, seed, rb)
        cp = core.build(cfg=cfg)
        path = os.path.join(a.work, f"units-test-{key}.safetensors")
        info = chipbuild.standalone(cp, path, None, {"units_test": key})  # asserts every weight is bf16
        t1 = time.time()
        cf = F.load(path)
        t2 = time.time()
        lint = chipcheck.lint(cf, cp)
        ir_d, lift_d, missing, extra = chipcheck.dim_map(cf, cp)
        assert not extra, extra[:5]
        missing = np.asarray(missing, np.int64)
        ex = exec4.Exec4(cf, text=None, v0=cf.v0, tables=cf.exec_tables())
        cres = {"stats": cp.stats(), "file_bytes": info["file_bytes"], "terms": info["terms"],
                "build_and_write_s": round(t1 - t0, 1), "lift_s": round(t2 - t1, 1), "lint": lint,
                "phases": ex.NPH, "slots": int(ex.n_slots),
                "max_abs_weight": float(np.abs(cf.file.arr("weights.values")).max()),
                "all_weights_finite": bool(np.isfinite(cf.file.arr("weights.values")).all())}
        print(key, json.dumps(cres), flush=True)
        os.unlink(path)
        res["circuits"][key] = cres
        if True:
            def mk():
                m, _ = T.machine(calls, True, seed, rb)
                m.cfg = cfg
                return m

            stream = mk().run(10**7)
            # tf: the residual after the chip's layers, every IR dim
            core.snapshot(cf, mk())
            ex.load_init()
            ex.reset()
            _, _, _, xs = ex.run(ex.embed(stream).contiguous(), mode=0, max_steps=len(stream), dbg=True)
            core.snapshot(cp, mk())
            carry, bad, first = {}, 0, None
            tiny = tiny_of(cp)
            by_dim, below_tiny = {}, True
            for s in range(0, len(stream), 1024):
                want = ir.evaluate(cp, stream[s:s + 1024], check=False, carry=carry)
                got = xs[s:s + 1024]
                ne = got[:, lift_d] != want[:, ir_d]
                d = ne.any(1)
                if len(missing):
                    d |= (want[:, missing] != 0).any(1)
                if d.any() and first is None:
                    t = int(np.argmax(d))
                    cols = np.flatnonzero(ne[t])[:4]
                    first = {"pos": s + t, "dims": [(cp.dims[int(ir_d[i])].name, float(got[t, lift_d[i]]),
                                                     float(want[t, ir_d[i]])) for i in cols]}
                for t, i in zip(*np.nonzero(ne)):  # which dims, and are they below their head's `tiny`
                    dd = int(ir_d[i])
                    e = by_dim.setdefault(cp.dims[dd].name, {"positions": 0, "max_abs_ir": 0.0, "max_abs_exec": 0.0,
                                                            "tiny": tiny.get(dd, 0.0)})
                    e["positions"] += 1
                    e["max_abs_ir"] = max(e["max_abs_ir"], abs(float(want[t, dd])))
                    e["max_abs_exec"] = max(e["max_abs_exec"], abs(float(got[t, lift_d[i]])))
                    below_tiny &= max(abs(float(want[t, dd])), abs(float(got[t, lift_d[i]]))) < tiny.get(dd, 0.0)
                bad += int(d.sum())
            if a.python:  # the executor from the Python circuit on the same stream: bit for bit?
                exp = exec4.Exec4(cp, text=None, v0=0, tables=exec4.build_tables(cp, None, 0))
                core.snapshot(cp, mk())
                exp.load_init()
                exp.reset()
                _, _, _, xp = exp.run(exp.embed(stream).contiguous(), mode=0, max_steps=len(stream), dbg=True)
                # compare under the dim map: the lifted executor's dims against the Python one's
                py_same = bool((xp[:, ir_d] == xs[:, lift_d]).all())
                del exp, xp
            del xs
            # free: the GPU alone from GO
            core.snapshot(cf, mk())
            ex.load_init()
            ex.reset()
            go = ex.embed(stream[:1]).contiguous()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = ex.run(go, mode=1, max_steps=len(stream) - 1, keep=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            got, want = out.tolist(), stream[1:]
            at = None if got == want else next((i for i, (x, y) in enumerate(zip(got, want)) if x != y),
                                               min(len(got), len(want)))
            n_m = sum(S.family(t)[0] == "M" for t in stream if t < S.VOCAB2)
            r = {"prog": name, "absent": len(cfg.absent), "positions": len(stream), "dims": len(ir_d) + len(missing),
                 "tf_positions_differing": bad, "tf_first": first, "tf_differing_dims": by_dim,
                 "tf_all_differences_below_tiny": bool(below_tiny) if bad else None,
                 "exec_file_equals_exec_python": py_same if a.python else None, "free_tokens": len(want),
                 "free_same": got == want, "free_first_diff": at, "unit_tokens_M": int(n_m),
                 "tok_s": round(len(want) / dt)}
            print(json.dumps(r), flush=True)
            res["programs"].append(r)
    res["total"] = {"programs": len(res["programs"]),
                    "positions": sum(r["positions"] for r in res["programs"]),
                    "tf_positions_differing": sum(r["tf_positions_differing"] for r in res["programs"]),
                    "free_tokens": sum(r["free_tokens"] for r in res["programs"]),
                    "free_programs_different": sum(not r["free_same"] for r in res["programs"])}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--random", type=int, default=6)
    ap.add_argument("--work", default=".")
    ap.add_argument("--only", nargs="*", default=None, help="these programs only")
    ap.add_argument("--python", action="store_true", help="also run Executor v4 from the Python circuit, compare")
    a = ap.parse_args()
    r = run(a)
    print(json.dumps(r["total"]), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
