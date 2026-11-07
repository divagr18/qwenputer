"""Checks and timing of executor v4 (exec4.py) on the merged checkpoint's circuit.

    python -m chipformer.rv.exec4check MODEL tf   [--progs c-crc32 ...] [--n 4000]
    python -m chipformer.rv.exec4check MODEL free [--progs ...]   # free-running vs the reference
    python -m chipformer.rv.exec4check MODEL time [--progs ...]   # v4 and Executor2, standalone

``tf``: teacher-forced, the residual after every step (every dim) against ir.evaluate.
``free``: each program runs from GO on the GPU alone (mode OWN: chip tokens and the circuit's
echoes, until the circuit owns no next token); the tokens must equal bridge.Session.run's, and
ir.evaluate (teacher-forced on the same stream) must predict every owned token and own exactly
those positions. Programs are ELFs in tests/data/rv, run on the bridge circuit (rv-q4p2's).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.measure import DATA


def circuit(model: str):
    model = Path(model)
    meta = json.loads((model / "chipformer.json").read_text())
    cfg = json.loads((model / "config.json").read_text())
    T = BR.Text(model / "tokenizer.json")
    c = core.build(Target(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
                          cfg["intermediate_size"]), T)
    return c, T, meta["chip_token_offset"]


_TABLES: dict = {}


def tables(c, T, v0, cache: Path | None = None):
    """exec4.build_tables, cached in memory and on disk (they take about a minute)."""
    from chipformer.rv import exec4

    if "t" in _TABLES:
        return _TABLES["t"]
    if cache is not None and cache.exists():
        z = np.load(cache)
        t = tuple(z[k] for k in ("lut_id", "lut_row", "off", "dims", "vals"))
    else:
        t = exec4.build_tables(c, T, v0)
        if cache is not None:
            np.savez(cache, lut_id=t[0], lut_row=t[1], off=t[2], dims=t[3], vals=t[4])
    _TABLES["t"] = t
    return t


def check_tables(c, T, v0, t, sample: int | None = None) -> dict:
    """The sparse embedding rows equal Executor2.embed's dense rows (all, or a sample), and the
    decode table equals core.decode_code (every code)."""
    from chipformer.rv.tritonexec import Executor2

    lut_id, lut_row, off, dims, vals = t
    ex = Executor2.__new__(Executor2)
    ex.c, ex.D, ex.dev, ex.feat_ids = c, len(c.dims), "cpu", sorted(c.token_fns)
    echo = sorted(set(T.printable) | {T.NL})
    toks = list(range(S.VOCAB)) + [BR.text_token(q) for q in echo]
    idx = range(len(toks)) if sample is None else np.random.default_rng(0).choice(
        len(toks), sample, replace=False)
    bad = 0
    idx = sorted(idx)
    for a in range(0, len(idx), 4096):
        part = idx[a:a + 4096]
        dense = ex.embed([toks[i] for i in part]).numpy()
        for r, i in enumerate(part):
            row = np.zeros(ex.D, np.float32)
            row[dims[off[i]:off[i + 1]]] = vals[off[i]:off[i + 1]]
            if row.tobytes() != dense[r].tobytes():
                bad += 1
    nbad_lut = 0
    for fid in range(1 << core.NFAM):
        for p in range(1 << 16):
            want = core.decode_code(fid, p, T)
            got = lut_id[fid << 16 | p]
            if want < 0:
                nbad_lut += got != -2
            elif S.is_text(want):
                nbad_lut += got != want - S.TEXT_TOKEN
            else:
                nbad_lut += got != v0 + want or lut_row[fid << 16 | p] != want
    return {"emb_rows_checked": len(idx), "emb_bad": bad, "lut_codes": (1 << core.NFAM) << 16,
            "lut_bad": int(nbad_lut)}


def reference(T, prog_name: str):
    """bridge.Session's stream of a program run from GO (records, echoes, the HALT newline)."""
    prog = elf.program(DATA / f"{prog_name}.elf")
    m = ref.Machine.from_program(prog)
    ses = BR.Session(T, m)
    ses.run(10**8)
    return prog, ses.toks, ses.own


def to_ids(stream, v0):
    return [t + v0 if not S.is_text(t) else t - S.TEXT_TOKEN for t in stream]


def from_id(i, v0):
    return i - v0 if i >= v0 else BR.text_token(int(i))


def make_exec(c, T, v0, cache=None, **kw):
    from chipformer.rv import exec4

    return exec4.Exec4(c, text=T, v0=v0, tables=tables(c, T, v0, cache), **kw)


def tf_check(c, T, v0, progs, n_max, cache) -> list:
    """Teacher-forced: every dim of the residual after every step equals the IR's."""
    out = []
    ex = None
    for name in progs:
        prog, stream, own = reference(T, name)
        stream = stream[:n_max]
        core.snapshot(c, ref.Machine.from_program(prog))
        if ex is None:
            ex = make_exec(c, T, v0, cache)
        ex.load_init()
        ex.reset()
        emb = ex.embed(stream)
        toks, kf, kl, xs = ex.run(emb, mode=0, max_steps=len(stream), dbg=True)
        carry, bad, first = {}, 0, None
        core.snapshot(c, ref.Machine.from_program(prog))
        for a in range(0, len(stream), 2048):
            want = ir.evaluate(c, stream[a:a + 2048], check=False, carry=carry)
            got = xs[a:a + 2048]
            diff = np.nonzero((got != want).any(1))[0]
            if len(diff) and first is None:
                t = int(diff[0])
                dd = np.nonzero(got[t] != want[t])[0][:5]
                first = {"pos": a + t, "dims": [(c.dims[d].name, float(got[t, d]),
                                                  float(want[t, d])) for d in dd]}
            bad += len(diff)
        r = {"prog": name, "positions": len(stream), "positions_differing": bad, "first": first}
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def free_check(c, T, v0, progs, cache, ir_check=True) -> list:
    """Free-running on the GPU (mode OWN) against bridge.Session; then the IR on that stream."""
    out = []
    ex = None
    for name in progs:
        prog, stream, own = reference(T, name)
        core.snapshot(c, ref.Machine.from_program(prog))
        if ex is None:
            ex = make_exec(c, T, v0, cache)
        ex.load_init()
        ex.reset()
        emb = ex.embed([stream[0]])
        torch.cuda.synchronize()
        t0 = time.time()
        toks, kf, kl = ex.run(emb, mode=2, max_steps=len(stream) + 10)
        dt = time.time() - t0
        got = [from_id(i, v0) if i >= 0 else i for i in toks]
        # the circuit predicts stream[t + 1] at every owned t; after the HALT newline it owns
        # nothing (code 0 -> -1)
        want = stream[1:] + [-1]
        n = min(len(got), len(want))
        first = next((t for t in range(n) if got[t] != want[t]), None)
        same = first is None and len(got) == len(want)
        st = ref.trace_stats([t for t in stream if not S.is_text(t)])
        r = {"prog": name, "tokens": len(stream), "instructions": st["instructions"],
             "gpu_same_as_reference": same, "first_diff": first, "seconds": round(dt, 3),
             "tok_per_s": round(len(stream) / dt), "instr_per_s": round(st["instructions"] / dt)}
        if ir_check:
            core.snapshot(c, ref.Machine.from_program(prog))
            carry, preds, owns = {}, [], []
            for a in range(0, len(stream), 2048):
                mat = ir.evaluate(c, stream[a:a + 2048], check=False, carry=carry)
                preds.append(core.predictions(c, mat))
                owns.append(np.rint(ir.ev(c.owner, mat)).astype(int))
            pred, ow = np.concatenate(preds), np.concatenate(owns)
            r["ir_wrong_predictions"] = int(sum(1 for x in range(len(stream) - 1)
                                                if own[x] and pred[x] != stream[x + 1]))
            r["ir_owner_mismatch"] = int(sum(ow[x] != int(own[x]) for x in range(len(stream))))
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def time_check(c, T, v0, progs, cache, reps: int = 3, v2: bool = True, **kw) -> list:
    """Standalone speed. v4: each program free-running from GO on the GPU (one launch, mode
    OWN), best of reps, checked against the reference. v2: Executor2 under its CUDA graph,
    teacher-forced on the same stream (fastexec.timed2: the step alone, no decode)."""
    from chipformer.rv import fastexec

    out = []
    ex = None
    for name in progs:
        prog, stream, own = reference(T, name)
        st = ref.trace_stats([t for t in stream if not S.is_text(t)])
        core.snapshot(c, ref.Machine.from_program(prog))
        if ex is None:
            ex = make_exec(c, T, v0, cache, **kw)
        best = None
        for _ in range(reps):
            ex.load_init()
            ex.reset()
            emb = ex.embed([stream[0]])
            torch.cuda.synchronize()
            t0 = time.time()
            toks, _, _ = ex.run(emb, mode=2, max_steps=len(stream) + 10)
            dt = time.time() - t0
            best = dt if best is None else min(best, dt)
        got = [from_id(i, v0) if i >= 0 else i for i in toks]
        r = {"prog": name, "tokens": len(stream), "instructions": st["instructions"],
             "tok_per_instr": round(len(stream) / st["instructions"], 3),
             "v4_same_as_reference": got == stream[1:] + [-1],
             "v4_us_per_step": round(best / len(stream) * 1e6, 2),
             "v4_tok_per_s": round(len(stream) / best),
             "v4_instr_per_s": round(st["instructions"] / best)}
        if v2:
            n = min(3000, len(stream) - 25)
            core.snapshot(c, ref.Machine.from_program(prog))
            s2 = fastexec.timed2(c, stream, steps=n)
            r.update({"v2_us_per_step": round(s2 * 1e6, 2), "v2_tok_per_s": round(1 / s2),
                      "v2_instr_per_s": round(1 / s2 / r["tok_per_instr"])})
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def ir_check(model: str, name: str) -> dict:
    """The IR (teacher-forced, chunked) on a program's reference stream: it must predict every
    owned next token and own exactly the reference's positions."""
    c, T, v0 = circuit(model)
    prog, stream, own = reference(T, name)
    core.snapshot(c, ref.Machine.from_program(prog))
    carry, preds, owns = {}, [], []
    t0 = time.time()
    for a in range(0, len(stream), 2048):
        mat = ir.evaluate(c, stream[a:a + 2048], check=False, carry=carry)
        preds.append(core.predictions(c, mat))
        owns.append(np.rint(ir.ev(c.owner, mat)).astype(int))
    pred, ow = np.concatenate(preds), np.concatenate(owns)
    return {"prog": name, "tokens": len(stream),
            "ir_wrong_predictions": int(sum(1 for x in range(len(stream) - 1)
                                            if own[x] and pred[x] != stream[x + 1])),
            "ir_owner_mismatch": int(sum(ow[x] != int(own[x]) for x in range(len(stream)))),
            "owned": int(sum(own)), "ir_seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("what", choices=("tf", "free", "time", "tables", "plan", "ir"))
    ap.add_argument("--progs", nargs="*", default=["c-crc32"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--cache", default=str(Path.home() / "exec4_tables.npz"))
    ap.add_argument("--no-ir", action="store_true")
    ap.add_argument("--no-v2", action="store_true")
    ap.add_argument("--extra-layers", type=int, default=0, help="time: scaling (exec4)")
    ap.add_argument("--no-flatten", action="store_true", help="time: one phase per level")
    args = ap.parse_args()
    c, T, v0 = circuit(args.model)
    cache = Path(args.cache)
    if args.what == "tables":
        t = tables(c, T, v0, cache)
        print(json.dumps(check_tables(c, T, v0, t)), flush=True)
    elif args.what == "plan":
        core.snapshot(c, ref.Machine.from_program(elf.program(DATA / f"{args.progs[0]}.elf")))
        ex = make_exec(c, T, v0, cache)
        print(json.dumps({"phases": ex.NPH, "JM": ex.JM, "NB": ex.NB, "smem": ex.smem, "HMAX": ex.HMAX,
                          "PARTMAX": ex.PARTMAX, "slots": ex.n_slots,
                          "phase_stats": ex.phase_stats}), flush=True)
    elif args.what == "tf":
        tf_check(c, T, v0, args.progs, args.n, cache)
    elif args.what == "free":
        free_check(c, T, v0, args.progs, cache, not args.no_ir)
    elif args.what == "time":
        time_check(c, T, v0, args.progs, cache, v2=not args.no_v2,
                   extra_layers=args.extra_layers, flatten=not args.no_flatten)
    elif args.what == "ir":  # one process per program (the IR is single-threaded numpy)
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(len(args.progs), 12)) as pool:
            for r in pool.map(ir_check, [args.model] * len(args.progs), args.progs):
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
