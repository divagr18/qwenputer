"""Merge the study outputs into docs/fx/kvram.json.

  python merge.py OUT_JSON study.json probes.json hfcheck.json cost.json [flash_full.json]
"""
from __future__ import annotations

import json
import sys


def main():
    out, study, probes, hf, cost = sys.argv[1:6]
    res = {
        "labels": {"M": "measured on CPU (torch 2.14, transformers 5.19, bf16) on fx's real trace or probes",
                   "E": "estimated from shapes (cost section)"},
        "study_M": json.load(open(study)),
        "probes_M": json.load(open(probes)),
        "hf_module_M": json.load(open(hf)),
        "cost_E": json.load(open(cost)),
    }
    if len(sys.argv) > 6:
        res["flash_full_length_M"] = json.load(open(sys.argv[6]))
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
