"""Runs the speed-iteration configurations (speed-iter.md) on the RV32 interpreter and
writes speed-iter.json plus the waterfall table.

  python -I speed_iter.py RV32EMU_DLL BUILDS_JSON OUT_JSON [--only KEY ...] [--prof KEY]

BUILDS_JSON maps a build label to the directory with fx.elf and imports.json (or to
{"elf": ..., "imports": ...} for P0's images). Each configuration is a build plus host
options (rvhost.py --opt). Every run is checked to leave the repository in P0's final state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = (2300, 15000)  # instructions per second: today's machine, a future executor

GW = ["provider=gateway", "env=AI_GATEWAY_API_KEY=placeholder"]

# key, label, build, host options, waterfall?, verdict
CONFIGS = [
    ("p0", "P0 baseline: ReleaseSmall wasm, wasm2c, clang -Os, unaligned scalar access, configured "
     "Chat Completions provider, one SSE event per token", "p0", [], True,
     "exact fx code (P0)"),
    # ---- host and configuration ----
    ("gateway", "+ fx's built-in AI Gateway route instead of FX_CONFIGURED_PROVIDERS", "p0", GW, True,
     "exact fx code, same semantics; config picks fx's default provider route (gateway serializer "
     "and Vercel-protocol SSE reducer instead of the Chat Completions ones)"),
    ("event", "+ the whole model reply as one SSE event", "p0", GW + ["stream=event"], True,
     "exact fx code, same semantics; the model server's chunking"),
    ("chunk", "+ that event in one network read", "p0", GW + ["stream=event_chunk"], True,
     "exact fx code, same semantics; transport chunking"),
    ("clock", "+ frozen virtual clock", "p0", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; OS clock (deterministic)"),
    # ---- toolchain ----
    ("fast", "+ ReleaseFast wasm (Zig build mode)", "fast_os_ua", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; build flag (both modes drop safety checks)"),
    ("fast_split", "same, built as 96 wasm2c files + ThinLTO (the pipeline -O2 needs; clang runs out of memory on one 47 MB file)",
     "fast_os_ua_lto", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; build pipeline"),
    ("o2", "+ clang -O2", "fast_o2_ua", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; C compiler flag"),
    ("base0", "+ wasm linear memory at address 0", "fast_o2_ua_b0", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; wasm2c runtime layout"),
    ("aligned", "aligned-only CPU: memargs' alignment hints honoured, under-aligned accesses checked",
     "fast_o2_check_b0", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; wasm2c patch (alignment hints); no misaligned access left"),
    ("fastcopy", "+ word-at-a-time memcpy/memmove in the runtime for any alignment", "fast_o2_check_b0_fc",
     GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; platform libc routine (wasm memory.copy)"),
    ("zbb", "+ Zbb + Zbs (ISA extension)", "fast_o2_check_b0_zbb_fc", GW + ["stream=event_chunk", "clock=frozen"], True,
     "exact fx code, same semantics; CPU extension (costs circuit)"),
    # ---- isolated, against P0 ----
    ("i_event", "P0 + one SSE event per reply only", "p0", ["stream=event"], False, "exact fx code, same semantics"),
    ("i_chunk_only", "P0 + token events in one read", "p0", ["stream=chunk"], False, "exact fx code, same semantics"),
    ("i_clock", "P0 + frozen clock only", "p0", ["clock=frozen"], False, "exact fx code, same semantics"),
    ("i_fast", "P0 + ReleaseFast only", "fast_os_ua", [], False, "exact fx code, same semantics"),
    ("i_os_split", "P0 build split into 96 files (-Os)", "rs_os_ua", [], False, "exact fx code, same semantics"),
    ("i_o2", "P0 + -O2 only", "rs_o2_ua", [], False, "exact fx code, same semantics"),
    ("i_o3", "final stack (row 12) at clang -O3 instead of -O2", "fast_o3_check_b0_fc",
     GW + ["stream=event_chunk", "clock=frozen"], False, "exact fx code, same semantics"),
    ("i_aligned_bytes_fast", "final stack but under-aligned accesses as byte ops (no address check)", "fast_o2_bytes_b0",
     GW + ["stream=event_chunk", "clock=frozen"], False, "exact fx code, same semantics"),
    ("i_aligned_bytes", "P0 on an aligned-only CPU, hints honoured, under-aligned = byte ops", "rs_os_bytes", [], False,
     "exact fx code, same semantics"),
    ("i_aligned_p0", "P0 on an aligned-only CPU, no hints (P0's 60.4M build)", "p0_aligned", [], False,
     "exact fx code, same semantics"),
    ("i_tools1", "P0 tool set cut to read_file (turn 2 only is comparable)", "p0", ["tools=read_file"], False,
     "changes fx's behaviour (fewer tools); turn 1's tool calls then fail"),
    # ---- native tools ----
    ("native", "best honest stack + fx's own read/edit/write tools in the guest",
     "tools_fast_o2_check_b0_fc", GW + ["stream=event_chunk", "clock=frozen", "native=1", "fs=repo"], False,
     "changes which fx code runs, toward more of it: fx's native read/edit/write run on the CPU "
     "(platform libc/std patched for WASI permissions); grep/glob stay host tools"),
    ("native_p0", "P0 + fx's own read/edit/write tools in the guest", "tools_rs_os_ua",
     ["native=1", "fs=repo"], False, "as above, on P0's stack"),
]


def fmt_time(instr, speed):
    s = instr / speed
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.0f} min"
    return f"{s:.0f} s"


def run(dll, build, opts, out, prof=False):
    if "elf" in build:
        elf, imp = build["elf"], build["imports"]
    else:
        elf, imp = os.path.join(build["dir"], "fx.elf"), os.path.join(build["dir"], "imports.json")
    cmd = [sys.executable, "-I", os.path.join(HERE, "rv", "rvhost.py"), dll, elf, imp, out, "--no-memtrace"]
    if prof:
        cmd = cmd[:-1] + ["--prof"]
    for o in opts:
        cmd += ["--opt", o]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("builds")
    ap.add_argument("out")
    ap.add_argument("--only", action="append")
    ap.add_argument("--runs", default=None, help="directory for the raw run JSON")
    a = ap.parse_args()
    builds = json.load(open(a.builds))
    builds = {k: (v if isinstance(v, dict) else {"dir": v}) for k, v in builds.items()}
    runs = a.runs or os.path.join(os.path.dirname(os.path.abspath(a.out)), "runs")
    os.makedirs(runs, exist_ok=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {"configs": {}}
    res = prev
    ref_repo = None
    for key, label, b, opts, wf, verdict in CONFIGS:
        if a.only and key not in a.only and key != "p0":
            continue
        if b not in builds:
            print("skip (no build)", key)
            continue
        r = run(a.dll, builds[b], opts, os.path.join(runs, f"{key}.json"))
        if key == "p0":
            ref_repo = r["final_repo"]
        s = r["summary"]
        res["configs"][key] = {
            "label": label, "build": b, "host_opts": opts, "waterfall": wf, "verdict": verdict,
            "turn1": s["turns"][0], "turn2": s["turns"][1], "total": r["rv_instructions_total"],
            "boot_to_first_read": s["boot"], "initialize": s.get("initialize"), "steps": s["steps"],
            "tool_in_guest": s["tool_in_guest"], "misaligned_accesses": r["misaligned_accesses"],
            "same_repo_as_p0": r["final_repo"] == ref_repo if ref_repo is not None else None,
            "native_calls": r.get("native_calls"), "unsupported_imports": r.get("unsupported_imports"),
            "dynamic_ops": r["dynamic_ops"],
        }
        print(f"{key:18s} turn1 {s['turns'][0]:>11,} turn2 {s['turns'][1]:>10,} mis {r['misaligned_accesses']:>7} "
              f"repo_ok {res['configs'][key]['same_repo_as_p0']}")
    base = res["configs"]["p0"]
    for k, c in res["configs"].items():
        c["x_turn1"] = base["turn1"] / c["turn1"]
        c["x_turn2"] = base["turn2"] / c["turn2"]
        for sp in SPEEDS:
            c[f"wall_turn1_at_{sp}"] = c["turn1"] / sp
            c[f"wall_turn2_at_{sp}"] = c["turn2"] / sp
    json.dump(res, open(a.out, "w"), indent=1)
    print()
    print("| # | change | turn 1 (M instr) | turn 2 (M instr) | x turn 1 | x turn 2 | turn 1 at 2.3k/s | at 15k/s | turn 2 at 2.3k/s | at 15k/s | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    i = 0
    for key, label, b, opts, wf, verdict in CONFIGS:
        c = res["configs"].get(key)
        if not c:
            continue
        i += 1
        print(f"| {i} | {label} | {c['turn1'] / 1e6:.2f} | {c['turn2'] / 1e6:.2f} | {c['x_turn1']:.2f} | {c['x_turn2']:.2f} | "
              f"{fmt_time(c['turn1'], 2300)} | {fmt_time(c['turn1'], 15000)} | {fmt_time(c['turn2'], 2300)} | "
              f"{fmt_time(c['turn2'], 15000)} | {verdict} |")


if __name__ == "__main__":
    main()
