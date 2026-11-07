"""Build size.json from the measurements in SCRATCH (rerun when a job finishes).

    python fx/size/report.py SCRATCH docs/fx/size.json

M = measured here; E = arithmetic on a file that was not built, or a judgement.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from qstream import SCHEMES

sp, out = sys.argv[1:3]
GB = 1e9


def load(n):
    p = f"{sp}/{n}"
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb")) if n.endswith(".pkl") else json.load(open(p))


def gb(x):
    return round(x / GB, 3)


H = pickle.load(open(sp + "/hdr.pkl", "rb"))
fam, n = M.qwen_text_families(H["base"])
qwen = sum(fam.values())
s = M.stock_bytes
res = {"note": "M measured, E estimated (see docs/fx/size.md)"}

an = {k: load(f"an_{k}.pkl") for k in ("q4p2", "fxint", "fx")}
res["circuits_M"] = {k: None if v is None else {x: v[x] for x in ("stats", "slots", "hidden", "m", "total", "distinct", "vocab", "peakC")}
                     | {"zero_mlp_layers": len(v["zero_mlp"])} for k, v in an.items()}
res["fx_circuit_without_text_memory_md_M"] = {"dims": 2987, "ffn_units": 45973, "heads": 850, "layers": 59}

V_fx = 550656
res["stock_build"] = {
    "rv_q4p2_M": {"bytes": s(3968, 11, 478656), "hidden": 3968, "chip_layers": 11, "vocab": 478656},
    "qwenpu_M": {"bytes": s(5120, 13, 642496), "hidden": 5120, "chip_layers": 13, "vocab": 642496},
    "fx_E": {"bytes": s(4096, 59, V_fx), "hidden": 4096, "chip_layers": 59, "vocab": V_fx,
             "per_layer_bytes": s(4096, 60, V_fx) - s(4096, 59, V_fx),
             "note": "not built; exact arithmetic from shapes; the formula reproduces both published files to the byte"},
}
pl = s(4096, 60, V_fx) - s(4096, 59, V_fx)
res["fx_growth_E"] = {
    "48_more_layers": 48 * pl,
    "hidden_3968_to_4096_on_everything": s(4096, 11, 478656) - s(3968, 11, 478656),
    "vocab_71936_more_rows": (V_fx - 478656) * 4096 * 2,
    "embedding_bytes": V_fx * 4096 * 2,
    "chip_layers_bytes": 59 * pl,
    "qwen_layers_bytes": s(4096, 0, 0) - 4096 * 2,
}
res["q4p2_breakdown_M"] = load("model.json")["q4p2"]
res["base_text_only_M"] = {"bytes": qwen, "families": fam, "weights": n}

# fx, stock, step by step
hzero = 4096
disk_zero = lambda h: 44 * 3 * 9216 * h * 2 + 3 * (8192 + 4096 + 4096 + 32) * h * 2 + 56 * 32 * h * 2
v = [("fx build, stock, bf16 (E)", s(4096, 59, V_fx), s(4096, 59, V_fx))]
V1 = V_fx - 65792 - 0
V1 = -(-(248320 + 302329 - 65792) // 64) * 64
v.append(("drop CW, EB (0 uses in fx; macros do not run on the paged map)", s(4096, 59, V1), s(4096, 59, V1)))
V2 = -(-(248320 + 302329 - 65792 - 65536) // 64) * 64
v.append(("R as a 2-token record (8,556 uses, +0.08% tokens)", s(4096, 59, V2), s(4096, 59, V2)))
v.append(("four-square ballast: hidden 3961, m 4", s(3961, 59, V2), s(3961, 59, V2)))
v.append(("all-zero tensors left out of the file", s(3961, 59, V2) - disk_zero(3961), s(3961, 59, V2)))
res["stock_waterfall_E"] = [dict(step=a, disk_gb=gb(b), vram_gb=gb(c)) for a, b, c in v]
res["stock_overlay_E"] = {h: dict(disk_gb=gb(s(h, 59, V2) - disk_zero(h)), vram_gb=gb(s(h, 59, V2)))
                          for h in (2816, 3072)}

store = {"q4p2": load("verify_q4p2.json"), "fx": load("verify_fx.json")}
res["sparse_chip_M"] = {k: None if v is None else v.get("store") for k, v in store.items()}
res["sparse_verify_M"] = {k: store["q4p2"].get(k) for k in ("tensors_identical", "tensors_different", "bytes_compared")} if store["q4p2"] else None
done = load("verify_q4p2.done.json")
if done:
    res["sparse_verify_progress_M"] = dict(tensors=len(done), identical=sum(v["same"] for v in done.values()),
                                           gb=gb(sum(v["bytes"] for v in done.values())), of_tensors=154)
res["sample_base_M"] = load("sample_base.json")
res["sample_published_M"] = "see docs/fx/size.md (nonzero and zstd by family)"
res["bench_cpu_M"] = load("bench.json")
res["proto_M"] = load("proto/proto.json")
res["four_square_M"] = load("foursq.json")
res["narrow_M"] = {"full_attention_layers_3_15": load("narrow_full.json"), "linear_layer_4": load("narrow_linear.json")}

kl = load("qstream.json")
partial = load("qpartial.json")
qs = {}
for name, sch in SCHEMES.items():
    if name in ("floor_eager",):
        continue
    if name.endswith(("_embin", "_lmhead")) or name in ("i8_ffn", "i8_gdn", "i8_attn", "i4_ffn", "i4_gdn", "i4_attn"):
        continue  # one family at a time: diagnostics, a real model ties emb and head
    qs[name] = dict(bytes=int(M.scheme_bytes(n, sch)), gb=gb(M.scheme_bytes(n, sch)))
    if kl:
        qs[name].update({k: kl["schemes"][name][k] for k in ("kl_mean", "top1", "kl_max_pos")})
res["quant_sizes_E_kl_M"] = qs
res["quant_kl_M"] = kl and {k: {x: v[x] for x in ("kl_mean", "top1", "kl_max_pos")} for k, v in kl["schemes"].items()}
res["quant_partial_M"] = partial
res["outside_anchor"] = {"gguf_q4_k_m_gb": [2.78, 2.81], "gguf_q8_0_gb": [4.3, 4.61], "gguf_bf16_gb": 8.67}
json.dump(res, open(out, "w"), indent=1, default=str)
print(json.dumps(res["stock_waterfall_E"], indent=0))
print(res["stock_overlay_E"])
print({k: v["gb"] for k, v in qs.items()})
print(res["fx_growth_E"])
