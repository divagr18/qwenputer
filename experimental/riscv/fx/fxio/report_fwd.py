"""docs/fx/fwd.json from the fx-fwd runs (docs/fx/fwd.md).

    python fx/fxio/report_fwd.py OUT_DIR docs/fx/fwd.json

OUT_DIR holds probe_before.json and probe_after.json (text_paged's probe with and without
go_on_tcc), fwd_q4p2.json (chipformer.rv.fwdcheck), q4diag.json and q4zero.json (rv-q4p2 residual
and code-removed checks), digest_old.json / digest_new1.json / digest_new2.json (circ_digest.py on
fx-int and fx-fwd), run13.json (fxrun's whole session), tl/ (live runs with --text), pa/ and pb/
(the probes' live runs) and out_smoke.log.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def probe_summary(p):
    names = [s["name"] for s in p["per_seq"]]
    mk = [i for i, n in enumerate(names) if not n.startswith("plain")]
    pl = [i for i, n in enumerate(names) if n.startswith("plain")]

    def m(key, idx):
        return float(np.mean([p[key][i] for i in idx]))

    out = {"go_on_tcc": p["go_on_tcc"], "sequences": len(names), "frames_mapped": p["frames_mapped"],
           "instructions": p["instructions"], "state_words_changed_outside_text_heads": p["state_words_changed"],
           "state_words_compared": p["state_words_compared"], "state_words_changed_by_head": p["state_words_changed_by_head"]}
    for grp, idx in (("all", range(len(names))), ("plain", pl), ("markup", mk)):
        out[grp] = {"n": len(idx), "kl_forward": m("kl_forward", idx), "kl_text_path": m("kl_text_last", idx),
                    "kl_noise_floor": m("kl_noise_floor", idx)}
    out["per_seq"] = [{
        "name": s["name"], "tokens": s["tokens"], "tcc": len(s["tcc_at"]),
        "positions_code_nonzero": s["positions_code_nonzero"], "code_nonzero_equals_tcc": s["code_nonzero_equals_tcc"],
        "code_names_there": sorted(set(s["code_names_there"])), "tau_there": sorted(set(s["tau_there"])),
        "kl_forward_there_min": min(s["kl_forward_there"]) if s["kl_forward_there"] else None,
        "kl_forward_there_max": max(s["kl_forward_there"]) if s["kl_forward_there"] else None,
        "forward_top1_there": sorted(set(s["forward_top1_there"])), "base_top1_there": sorted(set(s["base_top1_there"])),
        "kl_forward_elsewhere_mean": s["kl_forward_elsewhere_mean"],
        "kl_forward": p["kl_forward"][i], "kl_text_path": p["kl_text_last"][i], "kl_noise_floor": p["kl_noise_floor"][i],
        "forward_vs_text_path_max_logit": s["forward_vs_last_max_logit"],
        "state_words_changed_by_head": s["state_words_changed_by_head"]} for i, s in enumerate(p["per_seq"])]
    return out


def live(f):
    d = json.load(open(f))
    calls = [m for m in d["model_calls"] if m.get("tool_calls")]
    return {"task": d["task"], "rep": d["rep"], "backend": d["backend"], "solved": d["solved"], "stopped": d["stopped"],
            "steps": d["steps"], "elf_sha256": d["elf_sha256"], "audit_wrong": d["audit"]["wrong"],
            "audit_chip_tokens": d["audit"]["chip_tokens"], "requests_identical": d["emu"]["requests_identical"],
            "host_io_equal": d["emu"]["host_io_equal"], "final_repo_equal": d["emu"]["final_repo_equal"],
            "orig_final_repo_equal": d["orig"]["final_repo_equal"], "tool_calls": len(calls),
            "calls_ending_in_tcc_exactly": sum(m["raw_text"].endswith("</tool_call>") for m in calls),
            "calls_ending_in_nul": sum(m["raw_text"].endswith("\x00") for m in calls)}


def main(argv):
    o, out = argv
    J = lambda n: json.load(open(os.path.join(o, n)))  # noqa: E731
    res = {"probe_before": probe_summary(J("probe_before.json")), "probe_after": probe_summary(J("probe_after.json"))}
    q = J("fwd_q4p2.json")
    res["rv_q4p2_forward"] = {"plain": q["plain"], "tool": q["tool"], "vocab_merged": q["vocab_merged"], "per_seq": [
        {k: r[k] for k in ("name", "tokens", "kl_mean", "kl_noise_floor_mean", "kl_elsewhere_mean", "kl_floor_elsewhere_mean",
                           "chip_mass_elsewhere_max", "argmax_chip_equals_tcc", "top1_text_agree", "top1_floor",
                           "greedy_identical", "greedy_first_difference")}
        | {"tcc": len(r["tcc_at"]), "argmax_there": sorted(set(r["argmax_there"])),
           "chip_mass_there_min": min(r["chip_mass_there"]) if r["chip_mass_there"] else None,
           "kl_there_min": min(r["kl_there"]) if r["kl_there"] else None} for r in q["per_seq"]]}
    res["rv_q4p2_residual"] = {n: {k: v for k, v in r.items() if k != "kl_top_positions_off_tcc"} | {
        "kl_top5_off_tcc": r["kl_top_positions_off_tcc"][:5]} for n, r in J("q4diag.json").items()}
    res["rv_q4p2_code_removed"] = J("q4zero.json")
    dg = {k: J(k + ".json") for k in ("digest_old", "digest_new1", "digest_new2")}
    names = ["legacy", "legacy_text", "small_paged_text", "fx", "fx_text"]
    res["circuit_digests"] = {n: {"fx_int": dg["digest_old"][n]["sha256"], "fx_fwd": dg["digest_new1"][n]["sha256"],
                                  "fx_fwd_other_hash_seed": dg["digest_new2"][n]["sha256"], "stats": dg["digest_new1"][n]["stats"]}
                              for n in names}
    a, b = dict(dg["digest_new1"]["fx_text"]["nodes"]), dict(dg["digest_new1"]["fx_text_no_go"]["nodes"])
    res["circuit_digests"]["fx_text_no_go"] = {"fx_fwd": dg["digest_new1"]["fx_text_no_go"]["sha256"],
                                               "stats": dg["digest_new1"]["fx_text_no_go"]["stats"],
                                               "nodes_changed_against_fx_text": [k for k in a if a[k] != b.get(k)]}
    r13 = J("run13.json")
    res["whole_session"] = {k: r13[k] for k in ("frames", "tokens", "instructions", "segments", "wrong", "tok_s", "gpu_s")} | {
        "end": r13["end"]}
    res["live_text_after"] = [live(f) for f in sorted(glob.glob(os.path.join(o, "tl", "live_*.json")))]
    res["probe_runs"] = {"before": live(os.path.join(o, "pb", "live_t1_bug_1.json")),
                         "after": live(os.path.join(o, "pa", "live_t1_bug_1.json"))}
    res["test_parked_machine"] = open(os.path.join(o, "out_smoke.log")).read().strip().splitlines()[-1]
    if os.path.exists(os.path.join(o, "tests.log")):
        res["test_suite"] = open(os.path.join(o, "tests.log")).read().strip().splitlines()[-1]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("probe_before", "probe_after")}, indent=0)[:200])


if __name__ == "__main__":
    main(sys.argv[1:])
