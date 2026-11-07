"""live.json from the live runs: python report_chip.py OUT_DIR live.json [EXTRA.json ...]

Per run: solved or not, steps, chip tokens, instructions, Qwen tokens, chip / Qwen / host seconds,
peak GPU memory, and the comparison with rv32emu and the reference. EXTRA files are merged in
under their own names (the GPU check, the regression, the restore and loaded-machine checks).
"""
import difflib
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "p0c"))
import tasks as TK  # noqa: E402


def repo_diff(files: dict) -> dict:
    """The final repo against the task's start: path -> a unified diff."""
    out = {}
    for rel in sorted(set(files) | set(TK.COMMON)):
        a, b = TK.COMMON.get(rel, ""), files.get(rel, "")
        if a != b:
            out[rel] = "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), rel, rel, n=1))
    return out


def row(p: str) -> dict:
    r = json.load(open(p))
    c, q, e, a = r["chip"], r["qwen"], r.get("emu", {}), r.get("audit", {})
    mc = r["model_calls"]
    if r["host_s"] > r.get("build_s", 1e9):  # (the first run's host seconds held the circuit's build)
        r["host_s"] = round(r["host_s"] - r["build_s"], 3)
    return {
        "file": os.path.basename(p), "task": r["task"], "rep": r["rep"], "turns": r.get("turns", 1),
        "solved": r["solved"], "check": r["check"], "stopped": r["stopped"], "steps": r["steps"],
        "tool_calls": sum(len(m["tool_calls"]) for m in mc),
        "chip_tokens_fx": c["tokens"]["fx"], "chip_tokens_io": c["tokens"]["io"], "chip_tokens_audit": c["tokens"]["audit"],
        # (runs from before the launch-boundary fix counted one instruction short per cut token;
        # rv32emu's count at the last ecall gives the true figure, the tokens themselves were right)
        "instructions": c["instructions"] + 1 + max((int(k) for k in e.get("ic_minus_chip", {"-1": 1})), default=-1),
        "instructions_counted": c["instructions"], "ecalls": c["ecalls"], "launches": c["launches"],
        "pages_mapped": c.get("pages_mapped"), "forced_positions_checked": c["forced_positions_checked"], "tok_s_fx": c["tok_s_fx"],
        "qwen_fed": q["fed"], "qwen_generated": q["generated"], "rollbacks": q.get("rollbacks", 0),
        "chip_s_fx": c["seconds"]["fx"], "chip_s_io": c["seconds"]["io"], "chip_s_audit": c["seconds"]["audit"],
        "qwen_s": q["link_s"], "qwen_seconds": q["seconds"], "host_s": r["host_s"], "shadow_s": r["shadow_s"], "wall_s": r["wall_s"],
        "gpu_peak_bytes": r["gpu_peak_bytes"], "gpu_reserved_peak_bytes": r.get("gpu_reserved_peak_bytes"),
        "max_reply_bytes": max((m["reply_bytes"] for m in mc), default=0),
        "max_generated": max((m["generated_tokens"] for m in mc), default=0),
        "max_request_bytes": max((m["request_bytes"] for m in mc), default=0),
        "call_kinds": [m["kind"] for m in mc], "delta_tokens": [m["delta_tokens"] for m in mc],
        "emu_requests_identical": e.get("requests_identical"), "emu_host_io_equal": e.get("host_io_equal"),
        "emu_repo_equal": e.get("final_repo_equal"), "emu_ic_minus_chip": e.get("ic_minus_chip"),
        "end": r.get("end"), "audit": {k: v for k, v in a.items()}, "final_repo_sha256": r["final_repo_sha256"],
        "final_repo_diff": repo_diff(r["final_repo"]),
        "calls": [{k: m.get(k) for k in ("call", "kind", "ctx_before", "kept", "rolled_back", "delta_tokens", "full_tokens",
                                         "generated_tokens", "reply_bytes", "request_bytes", "canonical", "tool_calls", "link_s",
                                         "chip_s_at_open", "rv_instr_at_open", "tokens_at_open", "snapshot_bytes", "stopped")}
                  | {"restore_s_so_far": m.get("seconds", {}).get("restore")} for m in mc],
        "elf_sha256": r["elf_sha256"], "boot": r["boot"],
    }


def main():
    d, out, *extra = sys.argv[1:]
    runs = [row(p) for p in sorted(glob.glob(os.path.join(d, "live_*.json")))]
    res = {"runs": runs, "summary": {
        "runs": len(runs), "solved": sum(r["solved"] for r in runs), "stopped": sum(bool(r["stopped"]) for r in runs),
        "chip_tokens_fx": sum(r["chip_tokens_fx"] for r in runs), "chip_tokens_io": sum(r["chip_tokens_io"] for r in runs),
        "wrong_tokens": sum(1 for r in runs if (r["audit"] or {}).get("wrong"))}}
    for p in extra:
        res[os.path.splitext(os.path.basename(p))[0]] = json.load(open(p))
    json.dump(res, open(out, "w"), indent=1)
    print("| task | rep | turns | solved | steps | chip tokens, fx | I/O tokens | instr | Qwen tokens, fed + generated | chip s, fx | "
          "chip s, I/O | Qwen s | host s | GPU peak, GB | rollbacks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        print(f'| {r["task"]} | {r["rep"]} | {r["turns"]} | {"yes" if r["solved"] else "NO"} | {r["steps"]} | {r["chip_tokens_fx"]:,} | '
              f'{r["chip_tokens_io"]:,} | {r["instructions"]:,} | {r["qwen_fed"]:,} + {r["qwen_generated"]:,} | {r["chip_s_fx"]:.1f} | '
              f'{r["chip_s_io"]:.1f} | {r["qwen_s"]:.1f} | {r["host_s"]:.2f} | {r["gpu_peak_bytes"] / 1e9:.2f} | {r["rollbacks"]} |')


if __name__ == "__main__":
    main()
