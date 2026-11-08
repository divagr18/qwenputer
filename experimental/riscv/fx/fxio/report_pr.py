"""docs/fx/pr.json from pr_driver.py's run summaries.

    python report_pr.py OUT_JSON SUMMARY.json [SUMMARY.json ...] [--extra KEY=FILE.json ...]
"""
from __future__ import annotations

import json
import sys


def row(s: dict) -> dict:
    calls = s.get("pr_tool_calls") or []
    hc = s.get("hidden_check_on_pr_branch") or {}
    tools = s.get("tool_calls") or []
    return {
        "run_id": s["run_id"], "issue": s["issue"], "task": s["task"], "seed": s.get("seed"),
        "pr_url": s.get("pr_url"),
        "solved_on_pr_branch": hc.get("solved"), "check": hc.get("why"),
        "pr_branch": hc.get("branch"), "pr_head": hc.get("head"),
        "pr_branch_equals_fx_store": hc.get("branch_equals_fx_store"),
        "solved_in_fx_store": s.get("store_check"), "stopped": s.get("stopped"),
        "steps": s.get("steps"), "tool_sequence": tools,
        "open_pull_request_calls": len(calls),
        "open_pull_request_ok": sum(1 for c in calls if not c["result"][1]),
        "open_pull_request_failed": [{"args": c["args"], "result": c["result"]} for c in calls if c["result"][1]],
        "open_pull_request_args": [c["args"] for c in calls],
        "chip_s": s.get("chip_s"), "qwen_s": s.get("qwen_s"),
        "chip_tokens": s.get("chip_tokens"), "instructions": s.get("instructions"),
        "audit": s.get("audit"),
        "rv32emu": {k: (s.get("emu") or {}).get(k) for k in ("requests_identical", "host_io_equal", "final_repo_equal")},
        "end": {k: (s.get("end") or {}).get(k) for k in ("registers_differing", "words_differing", "pages_changed_by_rv32emu")},
        "replayed": s.get("replayed"), "chip_file": s.get("chip_file"), "elf_sha256": s.get("elf_sha256"),
        "handler_calls": s.get("handler_calls"),
    }


def main() -> None:
    out, *rest = sys.argv[1:]
    extra = {}
    sums = []
    it = iter(rest)
    for x in it:
        if x == "--extra":
            k, _, f = next(it).partition("=")
            extra[k] = json.load(open(f))
        else:
            sums.append(json.load(open(x)))
    runs = [row(s) for s in sums]
    res = {"runs": runs, "summary": {
        "runs": len(runs), "prs_opened_by_fx": sum(1 for r in runs if r["pr_url"]),
        "solved_on_pr_branch": sum(1 for r in runs if r["solved_on_pr_branch"]),
        "called_once_and_ok": sum(1 for r in runs if r["open_pull_request_calls"] == 1 and r["open_pull_request_ok"] == 1),
        "wrong_chip_tokens": sum(0 if (r["audit"] or {}).get("wrong") is None else 1 for r in runs)}, **extra}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
