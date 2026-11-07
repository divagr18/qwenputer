"""One fx session on rv32emu with fx's exact binary, optionally with the exact units of
units.py installed (circuits.md). Records RV32 instructions and trace tokens per turn, the
host-call sequence (every import call with its arguments and result, every byte the host read
from or wrote to guest memory) as a hash chain, the final repository, and per-unit counters.

  python -I run.py DLL ELF IMPORTS OUT_JSON [--units all|none|NAME,NAME..] [--mode on|verify]
         [--opt KEY=VALUE ..] [--pcdump FILE.npz]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path.insert(0, FX)
sys.path.insert(0, os.path.join(FX, "rv"))
sys.path.insert(0, HERE)
import host as H  # noqa: E402
import rvhost as RH  # noqa: E402
import tinyrepo  # noqa: E402
import units as U  # noqa: E402


class Recorder:
    """Wraps an RvHost: the hash chain of host calls and guest-memory traffic."""

    def __init__(self, h):
        self.h = h
        self.sha = hashlib.sha256()
        self.calls = 0
        rd, wr, hc = h.rd, h.wr, h.hostcall

        def rd2(p, n):
            b = rd(p, n)
            self.sha.update(b"R%d:%d:" % (p, n) + b)
            return b

        def wr2(p, b):
            self.sha.update(b"W%d:" % p + bytes(b))
            return wr(p, b)

        def hc2(ident, argp, nwords):
            words = ctypes.string_at(h.base + argp, 4 * nwords) if nwords else b""
            r = hc(ident, argp, nwords)
            self.sha.update(b"C%d:" % ident + words + b"=%d;" % r)
            self.calls += 1
            return r

        h.rd, h.wr, h.hostcall = rd2, wr2, hc2


def setup_dll(dll):
    e = dll
    e.rv_tokens.restype = ctypes.c_uint64
    e.rv_unit_add.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    e.rv_unit_stats.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    e.rv_unit_fail.restype = ctypes.POINTER(ctypes.c_uint32)
    e.rv_violations.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    e.rv_tokpc.restype = ctypes.POINTER(ctypes.c_uint32)
    e.rv_unit_check.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    e.rv_frozen_violations.restype = ctypes.c_uint64
    return e


def install(e, specs):
    ids = []
    for s in specs:
        p = (ctypes.c_uint32 * 16)(*s["params"])
        d = (ctypes.c_uint32 * 4)(*s["dead"])
        i = e.rv_unit_add(s["pc"], s["kind"], p, len(s["params"]), d, len(s["dead"]))
        assert i >= 0, s
        ids.append(i)
    return ids


def unit_stats(e, specs, ids):
    out = []
    for s, i in zip(specs, ids):
        v = (ctypes.c_uint64 * 8)()
        e.rv_unit_stats(i, v)
        out.append({"unit": s["name"], "trigger": s["label"], "pc": hex(s["pc"]), "fires": v[0], "declines": v[1],
                    "unit_tokens": v[2], "replaced_instructions": v[3], "replaced_tokens": v[4], "bytes": v[5],
                    "check_failures": v[6]})
    return out


def run(dll_path, elf, imports, units="none", mode="on", opts=(), turns=2, pcdump=None):
    H.set_opts(**RH.parse_opts(list(opts)))
    dll = setup_dll(ctypes.CDLL(os.path.abspath(dll_path)))
    tmp = H.repo_dir(tempfile.mkdtemp(prefix="fxrepo-"))
    tinyrepo.make_repo(tmp)
    h = RH.RvHost(dll, elf, json.load(open(imports)), tmp, turns, log=lambda *x: None, prof=bool(pcdump), memtrace=False)
    rec = Recorder(h)
    specs, skipped = U.specs_for(elf, units)
    ids = install(dll, specs)
    dll.rv_units_mode({"on": 1, "verify": 2}[mode] if specs else 0)
    tok_marks = []
    mark0 = h.mark

    def mark(kind, detail=None):  # token count at every event (turn boundaries)
        tok_marks.append((h.fuel(), dll.rv_tokens(), kind, detail))
        return mark0(kind, detail)

    h.mark = mark
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    viol = (ctypes.c_uint64 * 6)()
    dll.rv_violations(viol)
    fail = dll.rv_unit_fail()
    trap = [ev for ev in h.events if ev[1] == "trap"]
    ev = [(f, k, d) for f, k, d in h.events if k != "deliver"]
    # per-turn instructions and tokens: from session/prompt request to its response
    tm = [(f, t, k, d) for f, t, k, d in tok_marks]
    starts = [(f, t) for f, t, k, d in tm if k == "request" and d == "session/prompt"]
    ends = [(f, t) for f, t, k, d in tm if k == "response" and d == "session/prompt"]
    turns_out = [{"instructions": e_[0] - s_[0], "tokens": e_[1] - s_[1]} for s_, e_ in zip(starts, ends)]
    res = {
        "elf": os.path.abspath(elf), "units": units, "mode": mode, "opts": dict(H.OPTS),
        "installed": [{"name": s["name"], "trigger": s["label"], "pc": hex(s["pc"])} for s in specs],
        "not_installed": skipped,
        "rv_instructions_total": h.fuel(), "tokens_total": dll.rv_tokens(), "turns": turns_out,
        "boot": next(({"instructions": f, "tokens": t} for f, t, k, d in tm if k == "boot_end"), None),
        "host_calls": rec.calls, "host_io_sha256": rec.sha.hexdigest(),
        "stdout_events_sha256": hashlib.sha256(json.dumps([(k, d) for _, k, d in ev], default=str).encode()).hexdigest(),
        "final_repo": {r: open(os.path.join(tmp, r), encoding="utf-8").read() for r in h.repo.files()},
        "trap": trap, "violations": {"reg_reads": viol[0], "mem_reads": viol[1], "first_pc": hex(viol[2]), "what": viol[3],
                       "frozen_stores": dll.rv_frozen_violations()},
        "check_fail": [fail[i] for i in range(8)], "unit_stats": unit_stats(dll, specs, ids), "wall_s": wall,
        "native_calls": h.native_calls,
    }
    res["final_repo_sha256"] = hashlib.sha256(json.dumps(res["final_repo"], sort_keys=True).encode()).hexdigest()
    if pcdump:
        import numpy as np
        n = (h.text_hi - h.text_lo) // 4
        pcc = np.ctypeslib.as_array(h.e.rv_pccounts(), shape=(n,)).copy()
        tpc = np.ctypeslib.as_array(dll.rv_tokpc(), shape=(n,)).copy()
        np.savez_compressed(pcdump, pcc=pcc, tokpc=tpc, text_lo=h.text_lo, total=h.fuel(), tokens=dll.rv_tokens())
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("elf")
    ap.add_argument("imports")
    ap.add_argument("out")
    ap.add_argument("--units", default="none")
    ap.add_argument("--mode", default="on", choices=("on", "verify"))
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--pcdump")
    a = ap.parse_args()
    r = run(a.dll, a.elf, a.imports, a.units, a.mode, a.opt, a.turns, a.pcdump)
    json.dump(r, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: r[k] for k in ("rv_instructions_total", "tokens_total", "turns", "host_io_sha256",
                                        "final_repo_sha256", "trap", "violations", "check_fail")}, default=str))
    for s in r["unit_stats"]:
        print(" ", s)


if __name__ == "__main__":
    main()
