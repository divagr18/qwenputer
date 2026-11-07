"""The live runs (real Qwen) into one JSON: every run listed, failures kept, nothing averaged away.

  python -I report_live.py RUNS_DIR OUT_JSON

RUNS_DIR holds e2e.py's files, r24_t1_bug_1.json and their _units.json replays.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import delta as D  # noqa: E402
import protocol as P  # noqa: E402

TASKS = ["t1_bug", "t2_rename", "t3_feature", "t4_config", "t5_parse"]
CHIP_TOK_S = {"23k_exec_v4_calc_M": 23104, "15k_with_fx_memory_layers_E": 15000}
BYTE_CHIP_TOKENS = 2
CAP_MAX = 80


def tool_failures(run) -> dict:
    """fx's tool results in the last request, by outcome (what Qwen was told)."""
    pk = P.decode_stream(bytes.fromhex(run["packets_hex"])) if run["packets_hex"] else []
    reqs = [p for p in pk if p.kind == P.REQUEST]
    out = Counter()
    if reqs:
        for m in json.loads(P.parse_request(reqs[-1])[2]).get("prompt", []):
            if m["role"] == "tool":
                for p in m["content"]:
                    kind = p.get("output", {}).get("type", "")
                    out[("error " if "error" in kind else "ok ") + p.get("toolName", "?")] += 1
    return dict(out)


def divergence_cause(c) -> str:
    """Why a call could not extend the context, from the first differing text."""
    d = c.get("diverged") or {}
    had, new = d.get("had", ""), d.get("render", "")
    if had == new:
        return "same text, different tokens"
    if had.startswith("<think>") and not new.startswith("<think>"):
        return "template drops think blocks before the newest question"
    if "<parameter" in had + new or "</parameter" in had + new or "<function" in had + new or "<tool_call" in had + new:
        return "tool call layout"
    if "</think>" in had + new:
        return "think framing"
    return "other"


def one(run, units):
    calls = run["model_calls"]
    later = calls[1:]
    t = run.get("turn")
    r = {"row": "24" if run["native"] else "11", "task": run["task"], "solved": run["solved"], "check": run["check"],
         "finished": t is not None, "steps": run["steps"], "tools": [x for c in calls for x in c["tool_calls"]],
         "tool_results": tool_failures(run), "trap": run["trap"], "unsupported_imports": run["unsupported_imports"],
         "misaligned": run["misaligned"], "elf_sha256": run["elf_sha256"][:8],
         "turn_instr": t and t["instructions"], "turn_tokens": t and t["tokens"],
         "generated": sum(c["generated_tokens"] for c in calls), "generated_max_call": max((c["generated_tokens"] for c in calls), default=0),
         "reply_bytes_max_call": max((c["reply_bytes"] for c in calls), default=0),
         "stopped_all": all(c["stopped"] for c in calls), "warnings": [w for c in calls for w in c["warnings"]],
         "kinds": dict(Counter(c["kind"] for c in calls)), "rolled_back": sum(c["rolled_back"] for c in calls),
         "later_calls": len(later), "delta_tokens_later": sum(c["delta_tokens"] for c in later),
         "full_tokens_later": sum(c["full_tokens"] for c in later), "delta_bytes_later": sum(c["delta_bytes"] for c in later),
         "full_bytes_later": sum(c["full_bytes"] for c in later), "full_tokens_all": sum(c["full_tokens"] for c in calls),
         "full_bytes_all": sum(c["full_bytes"] for c in calls), "delta_tokens_all": sum(c["delta_tokens"] for c in calls),
         "noncanonical_calls": sum(1 for c in calls if not c["canonical"]), "calls": len(calls),
         "causes": dict(Counter(divergence_cause(c) for c in later if c["kind"] != "exact")),
         "first_divergence": next((c["diverged"] for c in later if c["kind"] != "exact"), None),
         "seconds": calls[-1]["seconds"] if calls else {}, "snapshot_bytes": calls[-1]["snapshot_bytes"] if calls else 0,
         "emulator_wall_s": run["emulator_wall_s"]}
    if units:
        r.update(turn_instr_units=units["turn"] and units["turn"]["instructions"], turn_tokens_units=units["turn"] and units["turn"]["tokens"],
                 units_same_io=units["host_io_sha256"] == run["host_io_sha256"] and units["final_repo_sha256"] == run["final_repo_sha256"],
                 replay_mismatches=units["replay_mismatches"], unit_check_failures=sum(u["check_failures"] for u in units["unit_stats"]))
    return r


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def wall(r, decode_s, prefill_tok_s):
    out = {}
    for name, cr in CHIP_TOK_S.items():
        chip = (r.get("turn_tokens_units") or 0) / cr
        out[name] = {"chip_units_s": chip, "chip_no_units_s": (r["turn_tokens"] or 0) / cr, "generate_s": r["generated"] * decode_s,
                     "delta_prefill_s": r["delta_tokens_all"] / prefill_tok_s,
                     "total_units_delta_s": chip + r["generated"] * decode_s + r["delta_tokens_all"] / prefill_tok_s,
                     "no_units_no_delta_bytewise_s": (r["turn_tokens"] or 0) / cr + r["generated"] * decode_s
                     + r["full_bytes_all"] * (BYTE_CHIP_TOKENS / cr + decode_s)}
    return out


def main():
    d, out = sys.argv[1:]
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "r*_?.json"))):
        run = json.load(open(f))
        u = f[:-5] + "_units.json"
        r = one(run, json.load(open(u)) if os.path.exists(u) else None)
        r["rep"] = int(re.search(r"_(\d)\.json$", f).group(1))
        rows.append(r)
    pc = {}
    p = os.path.join(d, "prefill.json")
    if os.path.exists(p):
        pc = json.load(open(p))
    gs = sum(r["seconds"].get("generate", 0) for r in rows)
    gt = sum(r["generated"] for r in rows)
    decode_s = gs / gt if gt else None
    fs = sum(r["seconds"].get("feed", 0) for r in rows)
    ft = sum(r["delta_tokens_all"] for r in rows)
    prefill = ft / fs if fs else None
    res = {"status": "M: real Qwen3.5-4B, HF transformers, one L4, fx's exact RV32 binaries.", "measured_decode_s_per_token": decode_s,
           "measured_feed_tok_s": prefill, "prefill_check": pc, "capture_window_tokens": CAP_MAX, "runs": rows}
    for row in ("24", "11"):
        rs = [r for r in rows if r["row"] == row]
        if not rs:
            continue
        for r in rs:
            r["wall_E"] = wall(r, decode_s, prefill)
        later = sum(r["later_calls"] for r in rs)
        kinds = Counter()
        causes = Counter()
        for r in rs:
            kinds.update(r["kinds"])
            causes.update(r["causes"])
        s = {"runs": len(rs), "solved": sum(r["solved"] for r in rs), "failures": [
            {k: r[k] for k in ("task", "rep", "check", "finished", "steps", "generated", "tools", "tool_results", "warnings", "stopped_all")}
            for r in rs if not r["solved"]],
            "later_calls": later, "kinds": dict(kinds), "causes": dict(causes),
            "exact_rate_later": kinds["exact"] / later if later else None,
            "rollback_rate_later": (kinds["rollback"] + kinds["rebuild"]) / later if later else None,
            "delta_tokens_per_later_call": sum(r["delta_tokens_later"] for r in rs) / later if later else None,
            "full_tokens_per_later_call": sum(r["full_tokens_later"] for r in rs) / later if later else None,
            "delta_bytes_per_later_call": sum(r["delta_bytes_later"] for r in rs) / later if later else None,
            "full_bytes_per_later_call": sum(r["full_bytes_later"] for r in rs) / later if later else None,
            "saved_later_pct": 100 * (1 - sum(r["delta_tokens_later"] for r in rs) / max(1, sum(r["full_tokens_later"] for r in rs))),
            "saved_all_pct": 100 * (1 - sum(r["delta_tokens_all"] for r in rs) / max(1, sum(r["full_tokens_all"] for r in rs))),
            "rolled_back_tokens": sum(r["rolled_back"] for r in rs),
            "calls": sum(r["calls"] for r in rs), "noncanonical_calls": sum(r["noncanonical_calls"] for r in rs),
            "generated_max_call": max(r["generated_max_call"] for r in rs), "reply_bytes_max_call": max(r["reply_bytes_max_call"] for r in rs),
            "steps_mean": mean([r["steps"] for r in rs]), "generated_mean": mean([r["generated"] for r in rs]),
            "misaligned": sum(r["misaligned"] for r in rs), "traps": sum(len(r["trap"]) for r in rs),
            "unsupported_imports": [r["unsupported_imports"] for r in rs if r["unsupported_imports"]],
            "units_all_same_io": all(r.get("units_same_io", False) for r in rs if r["finished"]),
            "replay_mismatches": sum(r.get("replay_mismatches") or 0 for r in rs),
            "unit_check_failures": sum(r.get("unit_check_failures") or 0 for r in rs)}
        s["per_task"] = {t: {k: mean([r[k] for r in rs if r["task"] == t]) for k in
                             ("steps", "generated", "turn_instr", "turn_tokens", "turn_instr_units", "turn_tokens_units", "delta_tokens_all", "full_tokens_all") if any(k in r for r in rs)}
                         | {"solved": sum(r["solved"] for r in rs if r["task"] == t), "reps": sum(1 for r in rs if r["task"] == t)}
                         for t in TASKS if any(r["task"] == t for r in rs)}
        res["row" + row] = s
    two = []
    for f in sorted(glob.glob(os.path.join(d, "two_*.json"))):
        run = json.load(open(f))
        c = run["model_calls"]
        two.append({"task": run["task"], "solved": run["solved"], "check": run["check"], "turns": run["turns"], "steps": run["steps"],
                    "tools": [x for k in c for x in k["tool_calls"]],
                    "calls": [{k: x[k] for k in ("call", "kind", "kept", "rolled_back", "full_tokens", "delta_tokens", "generated_tokens", "canonical", "common_prefix", "diverged")} for x in c],
                    "causes": dict(Counter(divergence_cause(x) for x in c[1:] if x["kind"] != "exact"))})
    res["two_turns"] = two
    every = [c for f in sorted(glob.glob(os.path.join(d, "r*_?.json"))) + sorted(glob.glob(os.path.join(d, "two_*.json"))) + sorted(glob.glob(os.path.join(d, "chat_*.json")))
             for c in json.load(open(f))["model_calls"]]
    res["all_generations"] = {"calls": len(every), "generated_tokens": sum(c["generated_tokens"] for c in every),
                              "noncanonical_calls": sum(1 for c in every if not c["canonical"]),
                              "stopped_cleanly": sum(1 for c in every if c["stopped"]), "parser_warnings": sum(len(c["warnings"]) for c in every)}
    second = [t["calls"][-1] for t in two if t["calls"]]
    res["two_turns_summary"] = {"runs": len(two), "solved": sum(t["solved"] for t in two), "kinds": dict(Counter(k["kind"] for t in two for k in t["calls"])),
                                "kinds_last_call": dict(Counter(x["kind"] for x in second)),
                                "last_call_full_tokens": sum(x["full_tokens"] for x in second), "last_call_delta_tokens": sum(x["delta_tokens"] for x in second),
                                "last_call_rolled_back": sum(x["rolled_back"] for x in second), "noncanonical": sum(1 for t in two for x in t["calls"] if not x["canonical"])}
    p = os.path.join(d, "chat_t1_bug.json")
    if os.path.exists(p):
        v = json.load(open(p))
        res["chat_route"] = {"solved": v["solved"], "steps": v["steps"], "turn": v["turn"], "kinds": dict(Counter(x["kind"] for x in v["model_calls"])), "generated": sum(x["generated_tokens"] for x in v["model_calls"]), "noncanonical": sum(1 for x in v["model_calls"] if not x["canonical"])}
    p = os.path.join(d, "merged.json")
    if os.path.exists(p):
        res["merged_check"] = json.load(open(p))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"two": res["two_turns_summary"], "chat": res.get("chat_route")}, indent=1))
    show = {k: {x: v for x, v in res[k].items() if x != "per_task"} for k in res if k.startswith("row")}
    print(json.dumps(show, indent=1))
    print("decode ms/token", None if decode_s is None else 1000 * decode_s, "feed tok/s", prefill)
    for r in rows:
        print(r["row"], r["task"], r["rep"], "SOLVED" if r["solved"] else "FAILED " + r["check"][:70], "steps", r["steps"], "gen", r["generated"],
              "kinds", r["kinds"], "noncanon", r["noncanonical_calls"], "instr", r["turn_instr"], "tok", r["turn_tokens"], "units", r.get("turn_tokens_units"))


if __name__ == "__main__":
    main()
