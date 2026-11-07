"""The size model: exact bytes of any stock build from its shapes, plus families and quantised sizes.

    python fx/size/model.py SCRATCH

stock_bytes(h, n_chip, vocab) is asserted against both published checkpoints to the byte. Anything it
gives for an unbuilt checkpoint is an estimate of a file that does not exist, exact only as arithmetic.
"""
import json
import pickle
import re
import sys
from collections import defaultdict

KH, VH, HD, INTER, CONV = 16, 32, 128, 9216, 4
NH, NKV, HEAD = 16, 4, 256
N_QWEN, N_FULL = 32, 8
GB = 1e9


def linear_params(h):
    """Tensor sizes of a linear_attention layer, by family."""
    return {"gdn": (2 * KH * HD + VH * HD) * h + VH * HD * h + h * VH * HD,  # qkv, z, out
            "ab": 2 * VH * h, "ffn": 3 * INTER * h,
            "small": (2 * KH * HD + VH * HD) * CONV + 2 * h + VH + VH + HD}


def full_params(h):
    return {"attn": 2 * NH * HEAD * h + NKV * HEAD * h * 2 + h * NH * HEAD,  # q (+gates), k, v, o
            "ffn": 3 * INTER * h, "small": 2 * h + 2 * HEAD}


def stock_bytes(h, n_chip, vocab):
    linear = n_chip + N_QWEN - N_FULL
    n = 2 * (sum(linear_params(h).values()) * linear + sum(full_params(h).values()) * N_FULL + vocab * h + h)
    return n + 2 * (VH + HD) * (N_QWEN - N_FULL)  # Qwen's A_log and gated-norm weights stay F32


def qwen_text_families(H):
    """Bytes of base Qwen3.5-4B's text model, by family, from the headers."""
    fam = defaultdict(int)
    n = defaultdict(int)
    for k, v in H.items():
        if not k.startswith("model.language_model."):
            continue
        s = v["b"] - v["a"]
        if "embed_tokens" in k:
            f = "emb"
        elif re.search(r"in_proj_(qkv|z)|out_proj", k):
            f = "gdn"
        elif re.search(r"in_proj_(a|b)", k):
            f = "ab"
        elif re.search(r"self_attn\.(q|k|v|o)_proj", k):
            f = "attn"
        elif re.search(r"mlp\.", k):
            f = "ffn"
        else:
            f = "small"
        fam[f] += s
        n[f] += s // (4 if v["dtype"] == "F32" else 2)
    return dict(fam), dict(n)


# cost of a method in bytes per weight (scales included): i8r one bf16 scale per row is negligible;
# i4g: 4 bits + one bf16 scale per 32 weights; nf4: 4 bits + 0.4 with double quant (bitsandbytes)
def per_weight(m, row_len):
    return {None: 2.0, "bf16": 2.0, "i8r": 1.0 + 2.0 / row_len, "i8t": 1.0, "i4g": 0.5 + 2.0 / 32,
            "nf4": 0.55}[m]


def scheme_bytes(n, s):
    """Bytes of the text model under a scheme (qstream.SCHEMES); 'emb' and 'head' are one tensor."""
    t = sum(n[f] * per_weight(s.get(f), 3000) for f in ("ffn", "gdn", "attn"))
    t += n["emb"] * per_weight(s.get("emb") or s.get("head"), 2560)
    t += n["ab"] * 2 + n["small"] * 2
    if s.get("ends"):  # the first and last two layers' ffn at ends
        t += 4 / 32 * n["ffn"] * (per_weight(s["ends"], 3000) - per_weight(s["ffn"], 3000))
    return t


def main():
    sp = sys.argv[1]
    H = pickle.load(open(sp + "/hdr.pkl", "rb"))
    q = sum(v["b"] - v["a"] for v in H["qp"].values())
    u = sum(v["b"] - v["a"] for v in H["qpu"].values())
    assert stock_bytes(3968, 11, 478656) == q, (stock_bytes(3968, 11, 478656), q)
    assert stock_bytes(5120, 13, 642496) == u, (stock_bytes(5120, 13, 642496), u)
    print("closed form matches both published checkpoints to the byte:", q, u)
    fam, n = qwen_text_families(H["base"])
    total = sum(fam.values())
    print("base text-only bytes", total, {k: round(v / GB, 3) for k, v in fam.items()})
    out = {"published_rv_q4p2": q, "published_qwenpu": u, "base_text_only": total, "base_families": fam,
           "base_weights": n, "base_repo_total": sum(v["b"] - v["a"] for v in H["base"].values())}
    # rv-q4p2 by origin
    h, nc, V = 3968, 11, 478656
    v0 = 248320
    chip_layers = sum(v["b"] - v["a"] for k, v in H["qp"].items()
                      if (m := re.search(r"layers\.(\d+)\.", k)) and int(m.group(1)) < nc)
    qwen_layers = sum(v["b"] - v["a"] for k, v in H["qp"].items()
                      if (m := re.search(r"layers\.(\d+)\.", k)) and int(m.group(1)) >= nc)
    emb = V * h * 2
    out["q4p2"] = dict(chip_layers=chip_layers, qwen_layers_widened=qwen_layers, embed=emb,
                       embed_qwen_part=v0 * 2560 * 2, embed_qwen_rows_chip_cols=v0 * (h - 2560) * 2,
                       embed_chip_rows=(V - v0) * h * 2,
                       qwen_layer_padding=qwen_layers - (total - fam["emb"]),
                       true_qwen=total, padding_or_chip=q - total)
    print(json.dumps(out["q4p2"], indent=0))
    for name, (hh, n_c, vv) in dict(fx_published_formula=(4096, 59, 550656)).items():
        print(name, stock_bytes(hh, n_c, vv) / GB)
    json.dump(out, open(sp + "/model.json", "w"), indent=1)


if __name__ == "__main__":
    main()
