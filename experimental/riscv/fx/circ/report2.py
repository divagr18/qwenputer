"""The tables of docs/fx/circuits2.md and circuits2.json, from the GPU runs, the circuit stats and fuzz2.

    python fx/circ/report2.py RUNS_DIR OUT.json

RUNS_DIR holds wf_<k>.json (fxrun run, k = 0 no units, 1.. the cumulative sets), stats.json and
fuzz_<unit>.json. Prints the Markdown tables. Turn 1 ends at segment 320, the last ecall before
rv32emu's turn-1 boundary (the session here starts at the boot snapshot).
"""

import json
import sys
from pathlib import Path

STEPS = ["baseline, no units", "+ memmove, memcpy", "+ token estimate", "+ JSON string scan", "+ eqlBytes",
         "+ ASCII run", "+ realloc copy", "+ writeOneLine (2 loops)", "+ memset", "+ SHA-256"]
# row 24's emulator estimate of each step's saving (tokens, whole run from reset), docs/fx/circuits.md
EST = [0, 1932868, 568519, 338408, 203517, 127907, 119103, 81322, 50024, 1175007]
TURN1 = 320


def main():
    d = Path(sys.argv[1])
    runs = [json.loads((d / f"wf_{k}.json").read_text()) for k in range(len(STEPS))]
    stats = json.loads((d / "stats.json").read_text())
    t0, tt0 = runs[0]["tokens"], runs[0]["cum"][TURN1][0]
    rows = []
    for k, (nm, r, st) in enumerate(zip(STEPS, runs, stats)):
        tt = r["cum"][TURN1][0]
        rows.append({
            "step": nm, "session_tokens": r["tokens"], "saved_pct": round(100 * (1 - r["tokens"] / t0), 1),
            "turn1_tokens": tt, "turn1_saved_pct": round(100 * (1 - tt / tt0), 1), "instructions": r["instructions"],
            "tok_s": r["tok_s"], "us_step": r["us_step"], "gpu_s": r["gpu_s"], "turn1_s": round(tt / r["tok_s"], 1),
            "phases": r["phases"], "shared_memory": r["smem"], "gpu_slots": r["slots"], "wrong": r["wrong"],
            "end": r.get("end"), "step_saved_tokens": (runs[k - 1]["tokens"] - r["tokens"]) if k else 0,
            "emulator_estimate_saved": EST[k], **st})
    out = {"status": "measured on the GPU (NVIDIA L4), Executor v4, unless marked", "turn1_segment": TURN1, "rows": rows,
           "fuzz": [json.loads(p.read_text()) for p in sorted(d.glob("fuzz_*.json"))]}
    for f in out["fuzz"]:
        f.pop("errs", None)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print("| # | step | session tokens | saved | turn 1 tokens | saved | instructions | tok/s | us / step | session | turn 1 | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in enumerate(rows):
        print(f'| {k} | {r["step"]} | {r["session_tokens"]:,} | {r["saved_pct"]}% | {r["turn1_tokens"]:,} | {r["turn1_saved_pct"]}% '
              f'| {r["instructions"]:,} | {r["tok_s"]:,} | {r["us_step"]} | {r["gpu_s"]:.0f} s | {r["turn1_s"]:.0f} s ({r["turn1_s"] / 60:.1f} min) | {0 if r["wrong"] is None else r["wrong"]} |')
    print()
    print("| step | saved, measured (M) | emulator estimate (E, from reset) | layers | IR dims | heads | FFN units | exec phases | residual slots | hidden | checkpoint (E) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f'| {r["step"]} | {r["step_saved_tokens"]:,} | {r["emulator_estimate_saved"]:,} | {r["layers"]} | {r["dims"]:,} | {r["heads"]} '
              f'| {r["ffn_units"]:,} | {r["phases"]} | {r["slots"]} | {r["hidden"]} | {r["ckpt_gb"]:.1f} GB |')
    print()
    print("| unit | cases | fired | declined | faulted (both ways) | errors | tokens: unit / original (fired cases) |")
    print("|---|---|---|---|---|---|---|")
    for f in out["fuzz"]:
        print(f'| {f["unit"]} | {f["cases"]:,} | {f["fired"]:,} | {f["declined"]:,} | {f["faulted"]:,} | {f["errors"]} | {f["tok_unit"]:,} / {f["tok_orig"]:,} |')


if __name__ == "__main__":
    main()
