"""docs/fx/io.json from the harness's run files.

  python -I report.py RUNS_DIR IMPORTS_JSON RENDER_JSON OUT_JSON

Every number from a run is a measurement of fx's binary and of the mapping. The projections are
marked as such and carry their inputs. With the scripted backend, `solved` says the packet path
carried a correct plan; it is not a solve rate of Qwen, and the run file says so in `backend`.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol as P  # noqa: E402

TASKS = ["t1_bug", "t2_rename", "t3_feature", "t4_config", "t5_parse"]
CAP_MAX = 80  # chipformer/rv/spec.py: the capture window, in tokens
BYTE_CHIP_TOKENS = 2  # one BYTE record, one echo: what printing a byte costs the chip today

# inputs of the projections, all from earlier docs
CHIP_TOK_S = {"exec_v4_rvchat_calc_M": 23104, "with_fx_memory_layers_E": 15000}
DECODE_S = {"rvchat_hf_E": 0.065, "fast_decode_E": 0.030}
PREFILL_TOK_S = 5000  # speedups §4: 4-7k
P0C_NOSHELL_OUTPUT_TOKENS_MEAN = 870  # p0c-report.md, M


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def summarize(run, units):
    calls = run["model_calls"]
    later = calls[1:]
    row = {
        "backend": run["backend"], "elf_sha256": run["elf_sha256"], "solved_by_harness_check": run["solved"], "steps": run["steps"],
        "turn_instr": run["turn"]["instructions"], "turn_tokens": run["turn"]["tokens"],
        "boot_instr": run["boot"]["instructions"], "boot_tokens": run["boot"]["tokens"],
        "misaligned": run["misaligned"], "traps": len(run["trap"]), "unsupported_imports": run["unsupported_imports"],
        "kinds": [c["kind"] for c in calls], "rolled_back_tokens": sum(c["rolled_back"] for c in calls),
        "generated_tokens": sum(c["generated_tokens"] for c in calls),
        "generated_max_call": max(c["generated_tokens"] for c in calls),
        "reply_bytes_max_call": max(c["reply_bytes"] for c in calls),
        "full_tokens_all_calls": sum(c["full_tokens"] for c in calls), "full_bytes_all_calls": sum(c["full_bytes"] for c in calls),
        "delta_tokens_all_calls": sum(c["delta_tokens"] for c in calls), "delta_bytes_all_calls": sum(c["delta_bytes"] for c in calls),
        "delta_tokens_per_later_call": sum(c["delta_tokens"] for c in later) / max(1, len(later)),
        "full_tokens_per_later_call": sum(c["full_tokens"] for c in later) / max(1, len(later)),
        "delta_bytes_per_later_call": sum(c["delta_bytes"] for c in later) / max(1, len(later)),
        "full_bytes_per_later_call": sum(c["full_bytes"] for c in later) / max(1, len(later)),
        "request_bytes_all_calls": sum(c["request_bytes"] for c in calls), "tools": [t for c in calls for t in c["tool_calls"]],
    }
    if units is not None:
        same = units["host_io_sha256"] == run["host_io_sha256"] and units["final_repo_sha256"] == run["final_repo_sha256"]
        row.update(turn_instr_units=units["turn"]["instructions"], turn_tokens_units=units["turn"]["tokens"],
                   replay_request_mismatches=units["replay_mismatches"], replay_used=units["replay_used"],
                   units_host_io_identical=same, units_installed=len(units["unit_stats"]),
                   units_check_failures=sum(u["check_failures"] for u in units["unit_stats"]))
    return row


def project(r):
    """Seconds for one task, by term. Estimates: tokens are measured, the rates are not."""
    out = {}
    for cn, cr in CHIP_TOK_S.items():
        for dn, ds in DECODE_S.items():
            chip = r["turn_tokens_units"] / cr
            chip_no_units = r["turn_tokens"] / cr
            gen = r["generated_tokens"] * ds
            # the packet path: the delta as BPE, one chunked prefill per call
            ingest_delta = r["delta_tokens_all_calls"] / PREFILL_TOK_S
            # no delta, no batching: every call prints the whole render, a byte at a time
            bytes_full = r["full_bytes_all_calls"]
            print_full = bytes_full * BYTE_CHIP_TOKENS / cr
            ingest_full = bytes_full * ds
            out[f"{cn}|{dn}"] = {
                "chip_s_no_units": chip_no_units, "chip_s_units": chip, "generate_s": gen, "ingest_delta_batched_s": ingest_delta,
                "total_s_units_delta": chip + gen + ingest_delta, "total_s_no_units_delta": chip_no_units + gen + ingest_delta,
                "print_full_render_s": print_full, "ingest_full_render_bytewise_s": ingest_full,
                "total_s_no_units_full_bytewise": chip_no_units + gen + print_full + ingest_full,
            }
    return out


def main():
    d, imports, render, out = sys.argv[1:]
    res = {"status": "Live Qwen did not run: every run here uses the scripted backend (see io.md).",
           "projection_inputs": {"chip_tok_s": CHIP_TOK_S, "decode_s_per_token": DECODE_S, "prefill_tok_s": PREFILL_TOK_S,
                                 "byte_chip_tokens": BYTE_CHIP_TOKENS, "capture_window_tokens": CAP_MAX,
                                 "p0c_noshell_output_tokens_mean_M": P0C_NOSHELL_OUTPUT_TOKENS_MEAN}}
    calls: dict = {}
    for row_name in ("r24", "r11"):
        rows = {}
        for t in TASKS:
            run, units = load(d, f"{row_name}_{t}.json"), load(d, f"{row_name}_{t}_units.json")
            if run is None:
                continue
            r = summarize(run, units)
            if units is not None:
                r["projection_s"] = project(r)
            rows[t] = r
            for k, v in run["import_calls"].items():
                calls.setdefault(k, {"r24": 0, "r11": 0})[row_name] += v
        res[row_name] = rows
    table = P.classify(json.load(open(imports)))
    for r in table:
        c = calls.get(r["name"], {"r24": 0, "r11": 0})
        r["calls_r24_5_tasks"], r["calls_r11_5_tasks"] = c["r24"], c["r11"]
        del r["calls"]
    res["imports"] = table
    res["import_groups"] = {g: sum(1 for r in table if r["group"] == g) for g in "abc"}
    res["imports_called"] = sum(1 for r in table if r["calls_r24_5_tasks"] or r["calls_r11_5_tasks"])
    res["render_rv"] = {k: v for k, v in json.load(open(render)).items() if k != "rows"}
    for name in ("chat_route", "two_turns", "two_turns_row11"):
        v = load(d, f"{name}.json")
        if v is not None:
            c = v["model_calls"]
            res[name] = {"provider": v["provider"], "solved_by_harness_check": v["solved"], "steps": v["steps"], "trap": v["trap"],
                         "turns": v["turns"], "calls": [{k: x[k] for k in ("call", "kind", "kept", "rolled_back", "full_tokens", "delta_tokens", "generated_tokens", "request_bytes", "reply_bytes")} for x in c]}
    r24 = res["r24"]
    if r24:
        n = len(r24)
        tot = lambda k: sum(r[k] for r in r24.values())
        res["r24_totals"] = {"tasks": n, "steps": tot("steps"), "later_calls": tot("steps") - n, "turn_instr": tot("turn_instr"),
                             "turn_tokens": tot("turn_tokens"), "turn_instr_units": tot("turn_instr_units"),
                             "turn_tokens_units": tot("turn_tokens_units"), "generated_tokens": tot("generated_tokens"),
                             "full_tokens": tot("full_tokens_all_calls"), "delta_tokens": tot("delta_tokens_all_calls"),
                             "full_bytes": tot("full_bytes_all_calls"), "delta_bytes": tot("delta_bytes_all_calls")}
        t = res["r24_totals"]
        t["delta_saves_tokens_pct"] = 100 * (1 - t["delta_tokens"] / t["full_tokens"])
        t["delta_saves_bytes_pct"] = 100 * (1 - t["delta_bytes"] / t["full_bytes"])
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("r24_totals", "import_groups", "imports_called", "render_rv")}, indent=1))


if __name__ == "__main__":
    main()
