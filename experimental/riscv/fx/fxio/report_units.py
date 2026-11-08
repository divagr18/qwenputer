"""docs/fx/units-file.json from the box's results directory (~/unitsres, docs/fx/units-file.md).

    python fx/fxio/report_units.py RESULTS_DIR docs/fx/units-file.json
"""
from __future__ import annotations

import json
import os
import re
import sys

LIVE_KEYS = ("task", "rep", "solved", "stopped", "steps", "frames", "units", "elf_sha256", "chip_file", "wall_s",
             "build_s", "host_s", "gpu_peak_bytes", "end", "orig", "audit", "final_repo_sha256")


def blocks(text: str) -> list:
    """Every top-level JSON object printed on its own lines (units_weights.py's indent=1 output)."""
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        j = text.find("\n{", i)
        if j < 0:
            return out
        try:
            o, k = dec.raw_decode(text[j + 1:])
            out.append(o)
            i = j + 1 + k
        except json.JSONDecodeError:
            i = j + 2


def weights(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    for o in blocks(open(path).read()):
        if "bad" in o:
            return {"layout": {k: o[k] for k in ("hidden", "m", "inv_c", "slots")},
                    "max_abs_weight": o["max_abs_weight"], "bad_count": len(o["bad"]),
                    "bad": [{k: b.get(k) for k in ("tensor", "index", "value", "node", "term", "gate_expr")}
                            for b in o["bad"]]}
    return None


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def live(path):
    r = load(path)
    if r is None:
        return None
    out = {k: r.get(k) for k in LIVE_KEYS}
    ch = r["chip"]
    out["chip"] = {k: ch.get(k) for k in ("tokens", "seconds", "instructions", "ecalls", "tok_s_fx", "pages_mapped")}
    out["qwen"] = {k: r["qwen"].get(k) for k in ("seconds", "link_s", "generated", "fed", "rollbacks")}
    out["emu"] = {k: v for k, v in r["emu"].items() if k != "ic_minus_chip"}
    out["task_s"] = round(ch["seconds"]["fx"] + ch["seconds"]["io"] + r["qwen"]["link_s"] + r["host_s"], 1)
    return out


def main() -> None:
    R, out = sys.argv[1:3]
    res = {"status": "2026-11-08, branch fx-units (from fx-int a604830), NVIDIA L4 q35-eval; M unless marked"}
    res["weights"] = {"fx_int_fxrun_map": weights(f"{R}/weights_base_fx.log"),
                      "fx_int_live": weights(f"{R}/weights_base_live.log"),
                      "fx_units_fxrun_map": weights(f"{R}/weights_u_fx.log")}
    da, db = load(f"{R}/digest_base.json"), load(f"{R}/digest_u.json")
    dig = {}
    for k, v in (da or {}).items():
        if isinstance(v, dict) and "sha256" in v:
            w = db[k]
            diff = sorted({x[0] for x, y in zip(v["nodes"], w["nodes"]) if x != y})
            dig[k] = {"identical": v["sha256"] == w["sha256"], "stats": v["stats"], "stats_equal": v["stats"] == w["stats"],
                      "nodes": len(v["nodes"]), "nodes_differing": len(diff),
                      "differing": diff if len(diff) <= 4 else [diff[0], "...", f"{len(diff)} names: h.ri and h.ri>0..83"]}
    res["digests_fx_int_vs_fx_units"] = dig
    files = {}
    for ln in open(f"{R}/chip_sha256.txt").read().split("\n"):
        if ln.strip():
            sha, p = ln.split()
            files[os.path.basename(p)] = {"sha256": sha}
    for log, name in (("write_live2.log", "fx-live-u13-f4096.safetensors"), ("write_fxrun2.log", "fx-u13-f4096.safetensors"),
                      ("write_text.log", "fx-live-u13-f4096-text.safetensors")):
        t = open(f"{R}/{log}").read() if os.path.exists(f"{R}/{log}") else ""
        m = re.search(r'"file_bytes": (\d+).*?"terms": (\d+)', t, re.S)
        if m and name in files:
            files[name].update(bytes=int(m.group(1)), terms=int(m.group(2)))
    res["files"] = files
    res["rewrite_same_arrays"] = open(f"{R}/rewrite_same_arrays.txt").read().strip().split("\n")
    res["lint_fxrun_map"] = load(f"{R}/lint_fxrun2.json")
    res["fxtf"] = load(f"{R}/fxtf.json")
    for k, f in (("session_from_file", "fx_session_file2.json"), ("session_python", "fx_session_python.json"),
                 ("session_from_file_first_write", "fx_session_file.json")):
        r = load(f"{R}/{f}")
        res[k] = {a: b for a, b in r.items() if a != "cum"} if r else None
    for k, f in (("units_tf", "units_tf.json"), ("units_tf_python", "units_tf_python.json")):
        r = load(f"{R}/{f}")
        if r:
            for c in r["circuits"].values():
                c["lint"] = {a: c["lint"][a] for a in ("ir_dims", "lifted_dims", "ir_dims_with_no_lifted_dim",
                                                       "lifted_dims_with_no_ir_dim", "head_diffs", "unit_group_diffs",
                                                       "bank_form_diffs")}
        res[k] = r
    res["live_t1_file"] = live(f"{R}/out/t1f/live_t1_bug_1.json")
    res["live_t1_text_file"] = live(f"{R}/out/t1ft/live_t1_bug_1.json")
    res["wide_file_check"] = load(f"{R}/wide_file_check.json")
    tests = {}
    for f in sorted(os.listdir(R)):
        if f.startswith("pytest_"):
            lines = [x for x in open(f"{R}/{f}").read().strip().split("\n") if x.strip()]
            tests[f[7:-4]] = lines[-1] if lines else ""
    res["tests_box"] = tests
    res["pr_local"] = load(f"{R}/out/prl/pr_local.json")
    res["one_md_13_units"] = {"session_tokens": 6666925, "session_gpu_s": 278.8, "session_tok_s": 23914,
                              "u13_t1_rep1": {"chip_tokens_fx": 3448213, "tok_s_fx": 23820, "task_s": 165.1,
                                              "audit_chip_tokens": 3552515, "steps": 3}}
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
