"""The units' waterfall (circuits.md): fx sessions with the units added one at a time, for the
speed-iter rows 11 and 24, plus the two IR-built circuits alone, plus every unit in verify
mode. Each step is its own process (rv32emu keeps its state in statics).

  python -I waterfall.py DLL BUILDS_JSON OUT_JSON [--rows 11,24] [--jobs 3]

BUILDS_JSON: {"11": {"elf": ..., "imports": ...}, "24": {...}}.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GW = ["provider=gateway", "env=AI_GATEWAY_API_KEY=placeholder", "stream=event_chunk", "clock=frozen"]
ROWS = {"11": GW, "24": GW + ["native=1", "fs=repo"]}
# the order of the waterfall: by the tokens each unit saves in the row-11 session
ORDER = ["move", "tokest", "sha", "jscan", "maskp", "wyhash", "jesc", "eql", "ascii", "bcopy", "memchr", "set"]
LABEL = {"move": "memmove / memcpy (streaming copy)", "tokest": "token estimate (whitespace-run scan)",
         "sha": "SHA-256 block", "jscan": "JSON scanner string scan", "maskp": "maskSecrets prefix probe (compare with constants)",
         "wyhash": "wyhash", "jesc": "JSON escape scan (encodeJsonString)", "eql": "mem.eqlBytes (compare)",
         "ascii": "JSON write ASCII-run scan", "bcopy": "realloc byte copy", "memchr": "writeOneLine newline / CR search",
         "set": "memset"}


def one(dll, build, row, units, mode):
    out = tempfile.mktemp(suffix=".json")
    cmd = [sys.executable, "-I", os.path.join(HERE, "run.py"), dll, build["elf"], build["imports"], out,
           "--units", units, "--mode", mode]
    for o in ROWS[row]:
        cmd += ["--opt", o]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    r = json.load(open(out))
    os.remove(out)
    r.pop("final_repo", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("builds")
    ap.add_argument("out")
    ap.add_argument("--rows", default="11,24")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    builds = json.load(open(a.builds))
    jobs = []
    for row in a.rows.split(","):
        jobs.append((row, "baseline", "none", "on"))
        for k in range(1, len(ORDER) + 1):
            jobs.append((row, "+" + ORDER[k - 1], ",".join(ORDER[:k]), "on"))
        jobs.append((row, "all units, verify mode", "all", "verify"))
        jobs.append((row, "IR-built circuits only (move_ir + eql_ir)", "move_ir,eql_ir", "verify"))
        for u in ORDER:  # each unit alone (its own saving, independent of the order)
            jobs.append((row, "alone: " + u, u, "on"))
    res = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(one, a.dll, builds[row], row, units, mode): (row, name, units, mode)
                for row, name, units, mode in jobs}
        for f in cf.as_completed(futs):
            row, name, units, mode = futs[f]
            r = f.result()
            res[f"{row}|{name}"] = {"row": row, "step": name, "units": units, "mode": mode, **r}
            t = r["turns"]
            print(f"row {row} {name:45s} turn1 {t[0]['instructions']:>9,} instr {t[0]['tokens']:>9,} tok  "
                  f"session {r['rv_instructions_total']:>9,} instr {r['tokens_total']:>10,} tok  io {r['host_io_sha256'][:8]}",
                  flush=True)
    ordered = [res[f"{row}|{name}"] for row, name, _, _ in jobs]
    json.dump({"order": ORDER, "labels": LABEL, "runs": ordered}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
