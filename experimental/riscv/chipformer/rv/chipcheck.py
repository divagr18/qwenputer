"""Audits of the executor running from a chip file, against the Python circuit and the reference.
Production never runs this: it is where the circuit builder may appear.

    python -m chipformer.rv.chipcheck lint CHIP --dense DIR        # the lift against core.build
    python -m chipformer.rv.chipcheck tf   CHIP --dense DIR [--progs ..] [--n 4000]
    python -m chipformer.rv.chipcheck free CHIP --dense DIR [--progs ..]
    python -m chipformer.rv.chipcheck time CHIP --dense DIR [--progs c-sieve]
    python -m chipformer.rv.chipcheck tf   CHIP --frames 1024      # the paged circuit (small_cfg)
    python -m chipformer.rv.chipcheck fx   CHIP --log LOG --elf ELF   # fx's whole session
    python -m chipformer.rv.chipcheck fxtf CHIP --log LOG --elf ELF [--n 4000]
        (fx, fxtf: --frames 4096 --units all+ for the 13-unit map, as fxrun; the file must be built so)
    python -m chipformer.rv.chipcheck vram CHIP --model BASE       # load time, VRAM, tok/s in RvChat
    python -m chipformer.rv.chipcheck qwen CHIP --base BASE [--model MERGED]  # text against base

--dense names the merged checkpoint: its circuit is the text-bridged one, run as exec4check does.
Run --dense on a tree whose core.build is that checkpoint's circuit: rv-q4p2's is fx-exec, not
fx-int, whose spec moved the control tokens (docs/fx/chipfile.md).

tf: every IR dim of the residual after every step against ir.evaluate of the Python circuit. A
lifted dim is the IR dim that owns its column at that version (dim_map).
free: from GO on the GPU alone, every token against the reference machine.
"""

import argparse
import json
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from chipformer.dsl.graph import ONE
from chipformer.q35 import ir
from chipformer.q35.emit import allocate
from chipformer.q35.ir import FFN, Head
from chipformer.rv import chipfile as F
from chipformer.rv import core, exec4, ref
from chipformer.rv import spec as S
from chipformer.rv.measure import DATA


def elf_names() -> list:
    return sorted(json.loads((DATA / "qemu.json").read_text()))


def owners(c, alloc) -> list:
    """(IR dim, column version) in the order allocate hands slots out: each next owner of a slot
    took it by exactly one handover."""
    order = sorted(c.token_fns)
    defined = defaultdict(list)
    for n in c.nodes:
        if isinstance(n, FFN):
            if not n.into:
                defined[n.level].append(n.out)
        else:
            for vh in n.vheads():
                if not vh.into:
                    defined[n.level].extend(d for _, d, _ in vh.out)
    for lv in sorted(defined):
        order += defined[lv]
    seen, out = defaultdict(int), []
    for d in order:
        s = alloc.slots[d]
        out.append((d, s, seen[s]))
        seen[s] += 1
    return out


def dim_map(cf, cp) -> tuple:
    """(IR dims, their lifted dims, IR dims with no lifted dim, lifted dims with no IR dim)."""
    al = allocate(cp, None, set(cp.keep), cp.target.intermediate)
    first = cf.ball_end + 1  # the chip's slots start after the ballast and the reserved column
    ir_d, lift_d, missing = [], [], []
    for d, s, v in owners(cp, al):
        col = first + s
        k = int(cf.idmap[col, v]) if v < cf.idmap.shape[1] else -1
        if k >= 0:
            ir_d.append(d)
            lift_d.append(k)
        else:
            missing.append(d)
    extra = sorted(set(range(1, len(cf.dims))) - set(lift_d))
    return np.asarray(ir_d), np.asarray(lift_d), missing, extra


def lint(cf, cp) -> dict:
    """Every IR node's terms appear in the lifted circuit, under dim_map, and nothing else does."""
    ir_d, lift_d, missing, extra = dim_map(cf, cp)
    m = dict(zip(ir_d.tolist(), lift_d.tolist()))
    m[ONE] = ONE

    def tr(e):
        return {m[d]: float(v) for d, v in e.terms.items()}

    bad = []
    heads = [n for n in cp.nodes if isinstance(n, Head)]
    lifted = [n for n in cf.nodes if isinstance(n, Head)]
    per = defaultdict(list)
    for n in lifted:
        per[n.level].append(n)
    seen = defaultdict(int)
    for n in heads:
        x = per[n.level][seen[n.level]]
        seen[n.level] += 1
        pairs = [("q", n.q, x.q), ("k", n.k, x.k)]
        for vi, (a, b) in enumerate(zip(n.vheads(), x.vheads())):
            pairs.append((f"v{vi}", a.v, b.v))
            if {c: tr(e) for c, e in a.z.items() if e.terms} != {c: e.terms for c, e in b.z.items()}:
                bad.append((n.name, "z", vi))
            if tr(a.a) != b.a.terms:
                bad.append((n.name, "a", vi))
            if sorted((c, m[d], w) for c, d, w in a.out) != sorted(b.out):
                bad.append((n.name, "out", vi))
            ba = None if a.beta is None or a.beta.terms == {ONE: 1} else tr(a.beta)
            if ba != (None if b.beta is None else b.beta.terms):
                bad.append((n.name, "beta", vi))
        if len(n.vheads()) != len(x.vheads()):
            bad.append((n.name, "vheads"))
        for nm, a, b in pairs:
            ea = {i: (tr(ch.expr), tuple(float(t) for t in ch.taps)) for i, ch in a.items()
                  if ch.expr.terms}
            eb = {i: (ch.expr.terms, tuple(ch.taps)) for i, ch in b.items()}
            if ea != eb:
                bad.append((n.name, nm))
    def units(c, mp):
        out = defaultdict(list)
        for n in c.nodes:
            if isinstance(n, FFN):
                for cf_, a, b in n.terms:
                    if not float(cf_):
                        continue
                    out[n.level, mp(n.out)].append(
                        (float(cf_), tuple(sorted((mp(d), float(v)) for d, v in a.terms.items())),
                         tuple(sorted((mp(d), float(v)) for d, v in b.terms.items()))))
        return out

    ua, ub = units(cp, lambda d: m[d]), units(cf, lambda d: d)
    ubad = sum(sorted(ua[k]) != sorted(ub.get(k, [])) for k in ua) + len(set(ub) - set(ua))
    bank_bad = []
    for g, form in getattr(cp, "banks", {}).items():
        lifted_f = cf.banks.get(g)
        if lifted_f is None:
            bank_bad.append((g, "missing"))
            continue
        for k in ("gate", "row", "head", "kgate"):
            if k in form and tr(form[k]) != lifted_f[k].terms:
                bank_bad.append((g, k))
        if form["zrow"] != lifted_f["zrow"] or form["nh"] != lifted_f["nh"]:
            bank_bad.append((g, "zrow / nh"))
    return {"ir_dims": len(cp.dims), "lifted_dims": len(cf.dims), "ir_dims_with_no_lifted_dim":
            len(missing), "lifted_dims_with_no_ir_dim": len(extra), "heads": len(heads),
            "heads_lifted": len(lifted), "head_diffs": len(bad), "first": bad[:6],
            "unit_groups": len(ua), "unit_group_diffs": int(ubad),
            "banks": len(cf.banks), "bank_form_diffs": len(bank_bad), "bank_first": bank_bad[:4]}


def table_check(cf, cp, T, v0) -> dict:
    """The file's decode table and rows against exec4.build_tables on the Python circuit."""
    a = exec4.build_tables(cp, T, v0)
    b = cf.exec_tables()
    ir_d, lift_d, missing, _ = dim_map(cf, cp)
    to_lift = np.full(len(cp.dims), -1, np.int64)
    to_lift[ir_d] = lift_d
    to_lift[ONE] = ONE
    bad = 0
    n = len(a[2]) - 1
    assert n == len(b[2]) - 1
    d0, d1 = a[3], b[3]
    keep = to_lift[d0] >= 0
    for i in range(n):
        o0, o1 = a[2][i], a[2][i + 1]
        ka = keep[o0:o1]
        x = {int(to_lift[d]): float(v) for d, v in zip(d0[o0:o1][ka], a[4][o0:o1][ka])}
        o0, o1 = b[2][i], b[2][i + 1]
        y = dict(zip(d1[o0:o1].tolist(), b[4][o0:o1].tolist()))
        bad += x != y
    return {"lut_codes": len(a[0]), "lut_id_diff": int((a[0] != b[0]).sum()),
            "lut_row_diff": int((a[1] != b[1]).sum()), "emb_rows": n, "emb_row_diff": int(bad)}


class Rig:
    """The file-loaded executor and, beside it, the Python circuit the audit compares with."""

    def __init__(self, a) -> None:
        self.a = a
        t0 = time.time()
        self.cf = F.load(a.chip)
        self.t_lift = time.time() - t0
        self.dense = bool(a.dense)
        if a.dense:
            from chipformer.rv import exec4check as EC

            self.cp, self.T, self.v0 = EC.circuit(a.dense)
        elif a.log:
            from chipformer.rv import fxrun

            self.cfg, _, _, _ = fx_load(a)
            self.cp, self.T, self.v0 = core.build(cfg=self.cfg), None, 0
        else:
            self.cfg = S.small_cfg(a.frames or 128)
            self.cp, self.T, self.v0 = core.build(cfg=self.cfg), None, 0
        ir_d, lift_d, missing, extra = dim_map(self.cf, self.cp)
        self.ir_d, self.lift_d, self.missing = ir_d, lift_d, np.asarray(missing, np.int64)
        assert not extra, extra[:5]
        self.ex = None

    def exec(self):
        if self.ex is None:
            t0 = time.time()
            self.ex = exec4.Exec4(self.cf, text=None, v0=self.cf.v0, tables=self.cf.exec_tables())
            print(json.dumps({"exec4_plan_s": round(time.time() - t0, 1), "phases": self.ex.NPH,
                              "slots": int(self.ex.n_slots), "smem": int(self.ex.smem),
                              "NB": self.ex.NB}), flush=True)
        return self.ex

    def stream(self, name):
        """(a fresh machine, the program's stream from GO)."""
        if self.dense:
            from chipformer.rv.exec4check import reference

            prog, s, _ = reference(self.T, name)
            return (lambda: ref.Machine.from_program(prog)), s
        mk = lambda: ref.PagedMachine.from_elf(self.cfg, DATA / f"{name}.elf")  # noqa: E731
        return mk, mk().run(10**8)


_RIG = None  # (forked into the workers: a circuit's token functions do not pickle)


def tf_one(args) -> dict:
    """ir.evaluate against the dumped residual, in a worker."""
    name, path, n = args
    rig = _RIG
    mk, stream = rig.stream(name)
    stream = stream[:n]
    core.snapshot(rig.cp, mk())
    xs = np.load(path, mmap_mode="r")
    carry, bad, first = {}, 0, None
    for a in range(0, len(stream), 2048):
        want = ir.evaluate(rig.cp, stream[a:a + 2048], check=False, carry=carry)
        got = xs[a:a + 2048]
        diff = (got[:, rig.lift_d] != want[:, rig.ir_d]).any(1)
        if len(rig.missing):
            diff |= (want[:, rig.missing] != 0).any(1)
        if diff.any() and first is None:
            t = int(np.argmax(diff))
            cols = np.flatnonzero(got[t, rig.lift_d] != want[t, rig.ir_d])[:4]
            first = {"pos": a + t, "dims": [(rig.cp.dims[int(rig.ir_d[i])].name,
                                             float(got[t, rig.lift_d[i]]),
                                             float(want[t, rig.ir_d[i]])) for i in cols]}
        bad += int(diff.sum())
    Path(path).unlink()
    return {"prog": name, "positions": len(stream), "dims": len(rig.ir_d) + len(rig.missing),
            "positions_differing": bad, "first": first}


def tf(rig, names, n, workdir, workers=6) -> dict:
    ex = rig.exec()
    jobs = []
    for name in names:
        mk, stream = rig.stream(name)
        stream = stream[:n]
        core.snapshot(rig.cf, mk())
        ex.load_init()
        ex.reset()
        emb = ex.embed(stream).contiguous()
        _, _, _, xs = ex.run(emb, mode=0, max_steps=len(stream), dbg=True)
        p = Path(workdir) / f"tf-{name}.npy"
        np.save(p, xs)
        jobs.append((name, str(p), n))
    tot = bad = 0
    res = []
    global _RIG
    _RIG = rig
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for r in pool.map(tf_one, jobs):
            print(json.dumps(r), flush=True)
            tot += r["positions"]
            bad += r["positions_differing"]
            res.append(r)
    return {"mode": "tf", "programs": len(names), "positions": tot, "dims": res[0]["dims"],
            "positions_differing": bad, "per_program": res}


def free(rig, names) -> dict:
    ex = rig.exec()
    res = []
    toks = bad = 0
    for name in names:
        mk, stream = rig.stream(name)
        core.snapshot(rig.cf, mk())
        ex.load_init()
        ex.reset()
        go = ex.embed(stream[:1]).contiguous()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if rig.dense:
            out = ex.run(go, mode=2, max_steps=len(stream) + 10, keep=False)
            got = [int(i) - rig.cf.v0 if i >= rig.cf.v0 else (S.TEXT_TOKEN | int(i)) if i >= 0
                   else int(i) for i in out]
            want = stream[1:] + [-1]
        else:
            n = len(stream) - 1
            out = ex.run(go, mode=1, max_steps=n, keep=False)
            got, want = out.tolist(), stream[1:]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        same = got == want
        at = None if same else next((i for i, (x, y) in enumerate(zip(got, want)) if x != y),
                                    min(len(got), len(want)))
        r = {"prog": name, "tokens": len(want), "same": same, "at": at,
             "us_step": round(dt / len(want) * 1e6, 2), "tok_s": round(len(want) / dt)}
        print(json.dumps(r), flush=True)
        res.append(r)
        toks += len(want)
        bad += not same
    return {"mode": "free", "programs": len(names), "tokens": toks, "different": bad,
            "per_program": res}


def timing(rig, names, reps=3) -> dict:
    """Best of reps: the file-loaded executor and the Python-built one, in one process, on one GPU."""
    res = []
    for kind in ("file", "python"):
        if kind == "file":
            ex, c = rig.exec(), rig.cf
        else:
            ex = None
            c = rig.cp
        for name in names:
            mk, stream = rig.stream(name)
            core.snapshot(c, mk())
            if ex is None:
                ex = exec4.Exec4(c, text=None, v0=rig.v0, tables=exec4.cached_tables(c, rig.T, rig.v0))
            best = None
            for _ in range(reps):
                ex.load_init()
                ex.reset()
                go = ex.embed(stream[:1]).contiguous()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = ex.run(go, mode=2 if rig.dense else 1, max_steps=len(stream) - 1, keep=False)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            r = {"kind": kind, "prog": name, "steps": len(out), "us_step": round(best / len(out) * 1e6, 2),
                 "tok_s": round(len(out) / best), "phases": ex.NPH, "slots": int(ex.n_slots),
                 "smem": int(ex.smem)}
            print(json.dumps(r), flush=True)
            res.append(r)
        if kind == "python":
            del ex
    return {"mode": "time", "per_program": res}


def fx_load(a):
    """fxrun.load with this command line's frames and units."""
    from chipformer.rv import fxrun

    return fxrun.load(a.log, a.elf, frames=a.frames or fxrun.FRAMES,
                      units=[u for u in a.units.split(",") if u])


def fx_session(rig, a) -> dict:
    """fxrun's run mode, on the file: the boot state comes from the file, each ecall's host service
    from the log, and every token is compared with the reference machine's."""
    from chipformer.rv import fxrun

    cfg, m, log, z = fx_load(a)
    cf = rig.cf
    assert cf.has_state(), "this chip file has no firmware state"
    t0 = time.time()
    ex = rig.exec()
    t_plan = time.time() - t0
    t0 = time.time()
    cf.load_state(ex)
    t_state = time.time() - t0
    # the file's state is the snapshot of the reference's own boot machine, bit for bit
    m0 = fx_load(a)[1]
    fxrun.serve(m0, log[0])
    m0.n += 1
    core.snapshot(cf, m0)
    s_file = ex.S.clone()
    ex.load_init()
    state_diff = int((ex.S != s_file).sum())
    ex.S.copy_(s_file)
    print(json.dumps({"plan_s": round(t_plan, 1), "state_s": round(t_state, 1),
                      "state_diff_vs_snapshot": state_diff,
                      "S_bytes": int(ex.S.numel() * 4)}), flush=True)
    go = ex.embed([S.tok("GO")]).contiguous()
    known: set = set()
    steps, gpu, seg, wrong = 0, 0.0, 0, None
    instr0 = m.steps
    for toks, touched in fxrun.segments(m, log, a.max_tokens or None, True):
        if toks is None:
            ex.poke(fxrun.state_rows(cf, m, touched, known))
            continue
        ex.reset()
        n = len(toks) - 1
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ids = ex.run(go, mode=1, max_steps=n, keep=False)
        torch.cuda.synchronize()
        gpu += time.perf_counter() - t1
        steps += n
        seg += 1
        want = np.asarray(toks[1:], np.int64)
        if len(ids) != n or (ids != want).any():
            k = int(np.argmax(ids[:n] != want[:len(ids)])) if len(ids) else 0
            wrong = {"segment": seg, "at": k, "of": n, "got": int(ids[k]) if k < len(ids) else None,
                     "want": int(want[k])}
            print("MISMATCH", wrong, flush=True)
            break
        if seg % 40 == 0:
            print(f"seg {seg} tokens {steps} gpu {gpu:.1f}s {steps / max(gpu, 1e-9):.0f} tok/s",
                  flush=True)
    res = {"mode": "fx", "tokens": int(steps), "instructions": int(m.steps - instr0), "segments": seg,
           "wrong": wrong, "gpu_s": round(gpu, 2), "tok_s": round(steps / max(gpu, 1e-9)),
           "us_step": round(1e6 * gpu / max(steps, 1), 2), "pages": len(m.pt), "state_diff": state_diff}
    if not wrong and not a.max_tokens:
        res["end"] = fxrun.end_check(m, z)
    return res


def fx_tf(rig, a) -> dict:
    """The first segment of the session, teacher-forced: every IR dim against ir.evaluate."""
    from chipformer.rv import fxrun

    cfg, m, log, z = fx_load(a)
    m0 = fx_load(a)[1]
    fxrun.serve(m0, log[0])
    m0.n += 1
    stream = []
    for toks, _ in fxrun.segments(m, log, a.n, True):
        if toks is not None:
            stream = toks[:a.n]
            break
    ex = rig.exec()
    rig.cf.load_state(ex)
    ex.reset()
    _, _, _, xs = ex.run(ex.embed(stream).contiguous(), mode=0, max_steps=len(stream), dbg=True)
    core.snapshot(rig.cp, m0)
    carry, bad = {}, 0
    for s in range(0, len(stream), 512):
        want = ir.evaluate(rig.cp, stream[s:s + 512], check=False, carry=carry)
        got = xs[s:s + 512]
        d = (got[:, rig.lift_d] != want[:, rig.ir_d]).any(1)
        if len(rig.missing):
            d |= (want[:, rig.missing] != 0).any(1)
        bad += int(d.sum())
    return {"mode": "fxtf", "positions": len(stream), "dims": len(rig.ir_d) + len(rig.missing),
            "positions_differing": bad}


def gpu_used() -> int:
    import subprocess

    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


CALC = ["Use calc: what is 123456789 * 987654321?", "Use calc: what is 2**64 - 1234567?",
        "Use calc: what is 355/113 + 22/7?", "Use calc: what is 99999999 * 99999999 * 7?"]


def vram(a) -> dict:
    """RvChat as production loads it: load time, VRAM with Qwen's weights beside the executor, and
    chip tokens/s on calc prompts. --python: the old path, core.build on the merged checkpoint."""
    from chipformer.rv import bridge as BR
    from chipformer.rv import elf
    from chipformer.rv.chat import RvChat

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if a.python:
        chat = RvChat(a.model, executor=True, python_circuit=True)
    else:
        chat = RvChat(a.model, executor=True, chip=a.chip, base=a.base or None)
    torch.cuda.synchronize()
    r = {"model": a.model, "path": "python" if a.python else "file", "slim": chat.slim,
         "load_s": round(time.time() - t0, 1),
         "params_GB": round(sum(p.numel() * p.element_size() for p in chat.model.parameters()) / 1e9, 3),
         "allocated_after_load_GB": round(torch.cuda.memory_allocated() / 1e9, 3),
         "reserved_after_load_GB": round(torch.cuda.memory_reserved() / 1e9, 3),
         "nvidia_smi_after_load_MiB": gpu_used()}
    print(json.dumps(r), flush=True)
    rows = []
    tok = sec = text = 0
    fw = DATA / "calc-bridge.elf"
    for rep in range(a.reps):
        for q in CALC:
            ids = chat.prompt_ids([{"role": "user", "content": q}])
            m = None
            if a.python or not chat.c.has_state():
                m = ref.Machine.from_program(elf.program(fw))
                BR.park(m)
            o = chat.generate(ids, m, 300, 60000)
            tt = chat.text_of(o["tokens"])
            rows.append({"q": q, "chip": o["chip_tokens"], "text": o["text_tokens"],
                         "chip_s": o["chip_seconds"], "text_s": o["text_seconds"],
                         "tail": tt[-90:]})
            if rep:  # the first pass compiles and warms up
                tok += o["chip_tokens"]
                sec += o["chip_seconds"]
                text += o["text_tokens"]
    r.update(rows=rows[-len(CALC):], chip_tokens=tok, chip_s=round(sec, 2),
             chip_tok_s=round(tok / max(sec, 1e-9)),
             peak_allocated_GB=round(torch.cuda.max_memory_allocated() / 1e9, 3),
             peak_reserved_GB=round(torch.cuda.max_memory_reserved() / 1e9, 3),
             nvidia_smi_end_MiB=gpu_used())
    return r


def qwen(a) -> dict:
    """Qwen beside the chip against base: the cut-down tensors against base's, bit for bit, then
    the same text as textcheck.py: base's greedy continuations, teacher-forced next-token
    distributions, KL(base || this) against KL(base sdpa || base eager). "this" is load_base's
    model in one chunk, and RvChat's _qwen_step a token at a time."""
    import gc

    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    from chipformer.q35.fast import exact_matmuls
    from chipformer.q35.report import PROMPTS, encode
    from chipformer.rv.chat import RvChat
    from chipformer.rv.textcheck import N_NEW, kl, logprobs

    exact_matmuls()  # as RvChat sets it: base and this run with the same cuBLAS settings
    dev = "cuda"
    res = {}
    base_m = F.load_base(a.base, "cpu")
    if a.model:
        c = F.load(a.chip)
        merged = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        slim = F.slim_merged(merged, c, a.base, "cpu")
        del merged
        sa, sb = base_m.state_dict(), slim.state_dict()
        bad = [k for k in sa if k != "lm_head.weight" and not torch.equal(sa[k], sb[k])]
        res.update(slim_tensors=len(sa), slim_tensors_different=len(bad), first=bad[:4],
                   slim_extra=sorted(set(sb) ^ set(sa))[:4])
        del slim, sa, sb
    del base_m
    gc.collect()
    print(json.dumps(res), flush=True)
    tb = AutoTokenizer.from_pretrained(a.base)
    prompts = [encode(tb, k, t) for k, t in PROMPTS]
    nq = len(tb)
    base = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
    seqs = []
    with torch.no_grad():
        for ids in prompts:
            g = base.generate(torch.tensor([ids], device=dev), max_new_tokens=N_NEW, do_sample=False)
            seqs.append(g[0].tolist())
    lb = logprobs(base, seqs, nq, dev)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    eager = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16,
                                                        attn_implementation="eager").to(dev)
    le = logprobs(eager, seqs, nq, dev)
    del eager
    gc.collect()
    torch.cuda.empty_cache()
    chat = RvChat(a.base, executor=True, chip=a.chip)
    lm = logprobs(chat.model, seqs, nq, dev)
    same = []
    with torch.no_grad():
        for ids, g in zip(prompts, seqs):
            h = chat.model.generate(torch.tensor([ids], device=dev), max_new_tokens=N_NEW,
                                    do_sample=False)
            same.append(h[0].tolist() == g)
    # RvChat's own text path: the prompt in one chunk, then _qwen_step token by token
    step = []
    with torch.no_grad():
        for ids, g in zip(prompts, seqs):
            o = chat.model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
            cache = o.past_key_values
            lp = [torch.log_softmax(o.logits[0, -1, :nq].float(), -1).half().cpu()]
            for i in range(len(ids), len(g) - 1):
                lg = chat._qwen_step(chat.lm.embed_tokens.weight[g[i]], cache, i)
                lp.append(torch.log_softmax(lg[:nq].float(), -1).half().cpu())
            step.append(torch.stack(lp))
    tail = [x[len(ids) - 1:len(g) - 1] for x, ids, g in zip(lb, prompts, seqs)]
    tail_e = [x[len(ids) - 1:len(g) - 1] for x, ids, g in zip(le, prompts, seqs)]
    k_m, k_e, k_s, k_se = kl(lb, lm), kl(lb, le), kl(tail, step), kl(tail, tail_e)
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    res.update(kl_this_mean=mean(k_m), kl_noise_floor_mean=mean(k_e), greedy_identical=sum(same),
               greedy_total=len(same), kl_qwen_step_mean=mean(k_s),
               kl_noise_floor_on_those_positions=mean(k_se),
               top1_agree=mean([float((p.argmax(-1) == q.argmax(-1)).float().mean())
                                for p, q in zip(lb, lm)]),
               top1_agree_step=mean([float((p.argmax(-1) == q.argmax(-1)).float().mean())
                                     for p, q in zip(tail, step)]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("lint", "tables", "tf", "free", "time", "fx", "fxtf", "vram",
                                     "qwen"))
    ap.add_argument("chip")
    ap.add_argument("--dense", default="")
    ap.add_argument("--frames", type=int, default=0, help="small_cfg's (128), or fx's (fxrun.FRAMES)")
    ap.add_argument("--units", default="", help="fx: unit names (hot.FX_R24), comma-separated, or all+")
    ap.add_argument("--log", default="")
    ap.add_argument("--elf", default="")
    ap.add_argument("--progs", nargs="*")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--work", default=".")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="", help="vram, qwen: base's directory, or a merged one")
    ap.add_argument("--base", default="", help="base Qwen's directory")
    ap.add_argument("--python", action="store_true", help="vram: core.build, the old path")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.what in ("vram", "qwen"):
        r = vram(a) if a.what == "vram" else qwen(a)
        print(json.dumps(r), flush=True)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)
        return
    rig = Rig(a)
    print(json.dumps({"lift_s": round(rig.t_lift, 2), **rig.cf.stats()}), flush=True)
    names = a.progs or elf_names()
    if a.what == "lint":
        r = lint(rig.cf, rig.cp)
    elif a.what == "tables":
        r = table_check(rig.cf, rig.cp, rig.T, rig.v0)
    elif a.what == "tf":
        r = tf(rig, names, a.n, a.work)
    elif a.what == "free":
        r = free(rig, names)
    elif a.what == "time":
        r = timing(rig, a.progs or ["c-sieve"])
    elif a.what == "fx":
        r = fx_session(rig, a)
    else:
        r = fx_tf(rig, a)
    print(json.dumps({k: v for k, v in r.items() if k != "per_program"}), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
