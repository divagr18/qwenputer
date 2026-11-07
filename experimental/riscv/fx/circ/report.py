"""Builds docs/fx/circuits.json and the tables of circuits.md from the waterfall, fuzz and
IR-cost runs.

  python -I report.py WATERFALL_JSON FUZZ_JSON[,FUZZ_JSON..] IRCOST_BASE IRCOST_NEW OUT_JSON > tables.md
"""
from __future__ import annotations

import json
import sys

TOK_RATES = (2300, 15000)


def mins(x, rate):
    s = x / rate
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def main():
    wf = json.load(open(sys.argv[1]))
    fuzz = [json.load(open(p)) for p in sys.argv[2].split(",")]
    base, new = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
    runs = {(r["row"], r["step"]): r for r in wf["runs"]}
    out = {"waterfall": {}, "alone": {}, "verify": {}, "ir_built": {}, "fuzz": [], "ir_cost": {}}
    for row in ("11", "24"):
        b = runs.get((row, "baseline"))
        if not b:
            continue
        print(f"\n### Row {row}\n")
        print("| # | step | turn 1 instr | turn 1 tokens | turn 2 tokens | session instr | session tokens | "
              "session tokens saved | turn 1 at 2.3k tok/s | at 15k tok/s | host I/O = baseline |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        steps = ["baseline"] + ["+" + u for u in wf["order"]]
        rows = []
        for i, st in enumerate(steps):
            r = runs[(row, st)]
            t1, t2 = r["turns"][0], r["turns"][1]
            saved = 1 - r["tokens_total"] / b["tokens_total"]
            same = r["host_io_sha256"] == b["host_io_sha256"] and r["final_repo_sha256"] == b["final_repo_sha256"]
            name = st if st == "baseline" else "+ " + wf["labels"][st[1:]]
            print(f"| {i} | {name} | {t1['instructions'] / 1e6:.2f}M | {t1['tokens'] / 1e6:.2f}M | {t2['tokens'] / 1e6:.2f}M | "
                  f"{r['rv_instructions_total'] / 1e6:.2f}M | {r['tokens_total'] / 1e6:.2f}M | {100 * saved:.1f}% | "
                  f"{mins(t1['tokens'], 2300)} | {mins(t1['tokens'], 15000)} | {'yes' if same else 'NO'} |")
            rows.append({"step": st, "turn1_instructions": t1["instructions"], "turn1_tokens": t1["tokens"],
                         "turn2_instructions": t2["instructions"], "turn2_tokens": t2["tokens"],
                         "session_instructions": r["rv_instructions_total"], "session_tokens": r["tokens_total"],
                         "session_tokens_saved": saved, "host_io_same": same, "measured": True})
        out["waterfall"][row] = rows
        v = runs.get((row, "all units, verify mode"))
        if v:
            out["verify"][row] = {k: v[k] for k in ("turns", "rv_instructions_total", "tokens_total", "host_io_sha256",
                                                    "final_repo_sha256", "violations", "check_fail", "unit_stats", "trap")}
            out["verify"][row]["host_io_same_as_baseline"] = v["host_io_sha256"] == b["host_io_sha256"]
            print(f"\nVerify run (row {row}, every unit, every firing checked against the original code): "
                  f"host I/O {'identical' if v['host_io_sha256'] == b['host_io_sha256'] else 'DIFFERENT'}, final repo "
                  f"{'identical' if v['final_repo_sha256'] == b['final_repo_sha256'] else 'DIFFERENT'}, "
                  f"check failures {v['check_fail'][1] != 0 and 'YES' or 0}, poisoned register reads "
                  f"{v['violations']['reg_reads']}, poisoned byte reads {v['violations']['mem_reads']}, "
                  f"stores into baked constants {v['violations']['frozen_stores']}.\n")
            print("| unit (trigger) | firings | declines | instructions replaced | their tokens | unit tokens | tokens saved |")
            print("|---|---|---|---|---|---|---|")
            for s in v["unit_stats"]:
                print(f"| {s['trigger']} | {s['fires']:,} | {s['declines']:,} | {s['replaced_instructions']:,} | "
                      f"{s['replaced_tokens']:,} | {s['unit_tokens']:,} | {s['replaced_tokens'] - s['unit_tokens']:,} |")
        ir = runs.get((row, "IR-built circuits only (move_ir + eql_ir)"))
        if ir:
            out["ir_built"][row] = {k: ir[k] for k in ("turns", "rv_instructions_total", "tokens_total", "host_io_sha256",
                                                       "violations", "check_fail", "unit_stats")}
            print(f"\nIR-built circuits only (row {row}): session tokens {ir['tokens_total'] / 1e6:.2f}M "
                  f"({100 * (1 - ir['tokens_total'] / b['tokens_total']):.1f}% saved), turn 1 "
                  f"{ir['turns'][0]['tokens'] / 1e6:.2f}M tokens; host I/O "
                  f"{'identical' if ir['host_io_sha256'] == b['host_io_sha256'] else 'DIFFERENT'}.")
            for s in ir["unit_stats"]:
                print(f"  - {s['trigger']}: {s['fires']:,} firings, {s['declines']:,} declines, "
                      f"{s['replaced_tokens']:,} -> {s['unit_tokens']:,} tokens")
        alone = []
        for u in wf["order"]:
            r = runs.get((row, "alone: " + u))
            if r:
                alone.append({"unit": u, "session_tokens": r["tokens_total"],
                              "saved": b["tokens_total"] - r["tokens_total"],
                              "saved_share": 1 - r["tokens_total"] / b["tokens_total"]})
        out["alone"][row] = alone
        if alone:
            print(f"\nEach unit alone (row {row}): " + ", ".join(
                f"{a['unit']} {100 * a['saved_share']:.1f}%" for a in alone) + ".")
    print("\n### Fuzzing\n")
    print("| unit | cases | fired | declined | errors |")
    print("|---|---|---|---|---|")
    for f in fuzz:
        for u in f["units"]:
            out["fuzz"].append({"elf": f["elf"], **u})
            print(f"| {u['unit']} | {u['cases']:,} | {u['fired']:,} | {u['declined']:,} | {u['errors']} |")
    out["ir_cost"] = {"base": base["stats"], "with_macros": new["stats"]}
    json.dump(out, open(sys.argv[5], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
