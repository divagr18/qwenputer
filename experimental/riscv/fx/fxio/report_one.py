"""one.json from the runs of docs/fx/one.md: python report_one.py OUT.json BASE_LIVE.json KEY=DIR_OR_FILE ...

KEY=DIR: every live_*.json in DIR, one row each, with the task's row of live.md (live.json) beside it.
KEY=FILE: any other result, merged in under KEY. A task's time is what a user waits: fx's and the I/O's
chip seconds, the model link's and the host's. It leaves out the circuit's build, the audit registers,
the lockstep emulators and the offline audit.
"""
import glob
import json
import os
import statistics as st
import sys


def row(p: str, base: dict) -> dict:
    r = json.load(open(p))
    c, q, e, o, a = r["chip"], r["qwen"], r.get("emu", {}), r.get("orig", {}), r.get("audit", {})
    host = r["host_s"]
    if host > r.get("build_s", 1e9):  # (the first run's host seconds hold the circuit's build)
        host = round(host - r["build_s"], 3)
    chip_s = c["seconds"]["fx"] + c["seconds"]["io"]
    b = base.get((r["task"], r["rep"])) if not r.get("chain") else None
    end, mem = r.get("end", {}), o.get("memory", {})
    return {
        "file": os.path.basename(p), "task": r["task"], "rep": r["rep"], "turns": r.get("turns", 1), "chain": r.get("chain"),
        "units": r.get("units"), "frames": r.get("frames"), "solved": r["solved"], "check": r["check"], "stopped": r["stopped"],
        "steps": r["steps"], "chip_tokens_fx": c["tokens"]["fx"], "chip_tokens_io": c["tokens"]["io"],
        "chip_tokens_audit": c["tokens"]["audit"], "chip_tokens_text": c["tokens"].get("text", 0),
        "instructions_chip": c["instructions"], "instructions_original": o.get("instr"), "ecalls": c["ecalls"],
        "frames_mapped": c.get("pages_mapped"), "linear_top": end.get("linear_top"), "tok_s_fx": c["tok_s_fx"],
        "chip_s_fx": c["seconds"]["fx"], "chip_s_io": c["seconds"]["io"], "chip_s_text": c["seconds"].get("text", 0),
        "qwen_s": q["link_s"], "host_s": host, "task_s": round(chip_s + q["link_s"] + host, 1),
        "qwen_fed": q["fed"], "qwen_generated": q["generated"], "rollbacks": q["rollbacks"],
        "gpu_peak_bytes": r["gpu_peak_bytes"], "gpu_reserved_peak_bytes": r["gpu_reserved_peak_bytes"],
        "snapshot_bytes": q.get("snapshot_bytes"),
        "emu_requests_identical": e.get("requests_identical"), "emu_host_io_equal": e.get("host_io_equal"),
        "emu_repo_equal": e.get("final_repo_equal"), "orig_requests_identical": o.get("requests_identical"),
        "orig_host_io_equal": o.get("host_io_equal"), "orig_repo_equal": o.get("final_repo_equal"),
        "orig_dead_registers_differing": o.get("registers_differing_dead"), "orig_memory": mem,
        "end_registers_differing": end.get("registers_differing"), "end_words_read": end.get("words_read"),
        "end_words_differing": end.get("words_differing"), "audit_chip_tokens": a.get("chip_tokens"),
        "audit_forced_tokens": a.get("forced_tokens"), "audit_wrong": a.get("wrong"),
        "audit_memory_words": a.get("memory_words_compared"), "audit_memory_diff": a.get("memory_diff"),
        "forced_positions_checked": c.get("forced_positions_checked"), "elf_sha256": r["elf_sha256"],
        "base_chip_s": round(b["chip_s_fx"] + b["chip_s_io"], 1) if b else None, "base_steps": b["steps"] if b else None,
        "base_task_s": round(b["chip_s_fx"] + b["chip_s_io"] + b["qwen_s"] + b["host_s"], 1) if b else None,
        "base_tokens": b["chip_tokens_fx"] if b else None, "base_frames": b["pages_mapped"] if b else None}


def summary(rows: list) -> dict:
    ok = [r for r in rows if r["solved"]]
    t = [r["task_s"] for r in rows]
    return {"runs": len(rows), "solved": len(ok), "stopped": sum(bool(r["stopped"]) for r in rows),
            "chip_tokens": sum(r["chip_tokens_fx"] + r["chip_tokens_io"] for r in rows),
            "audit_tokens": sum(r["audit_chip_tokens"] or 0 for r in rows),
            "wrong": sum(bool(r["audit_wrong"]) for r in rows), "tok_s_min": min(r["tok_s_fx"] for r in rows),
            "tok_s_max": max(r["tok_s_fx"] for r in rows), "task_s_min": min(t), "task_s_max": max(t),
            "task_s_median": st.median(t), "frames_max": max(r["frames_mapped"] or 0 for r in rows),
            "gpu_peak_max": max(r["gpu_peak_bytes"] for r in rows)}


def main():
    out, live, *items = sys.argv[1:]
    base = {(r["task"], r["rep"]): r for r in json.load(open(live))["runs"] if r["turns"] == 1}
    res = {}
    for it in items:
        k, _, p = it.partition("=")
        if os.path.isdir(p):
            rows = [row(f, base) for f in sorted(glob.glob(os.path.join(p, "live_*.json")))]
            res[k] = {"runs": rows, "summary": summary(rows) if rows else {}}
        else:
            res[k] = json.load(open(p))
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "summary" in v:
            print(k, v["summary"])


if __name__ == "__main__":
    main()
