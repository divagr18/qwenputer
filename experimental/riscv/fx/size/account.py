"""Exact byte accounting from the safetensors headers (hdr.pkl from fetch_headers.py)."""
import json, os, pickle, re, sys
from collections import defaultdict
sp = sys.argv[1]
H = pickle.load(open(sp + "/hdr.pkl", "rb"))
G = 1e9
for n in H:
    cfg = json.load(open(f"{sp}/cfg/{n}/config.json"))
    tc = cfg.get("text_config", cfg)
    print("==", n, {k: tc.get(k) for k in ("hidden_size", "num_hidden_layers", "vocab_size", "intermediate_size", "linear_num_key_heads",
          "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings", "max_position_embeddings")})
    print("   layer types", "".join("F" if x == "full_attention" else "L" for x in tc["layer_types"]), cfg.get("tie_word_embeddings"))
    dt = defaultdict(int)
    fam = defaultdict(int)
    per = defaultdict(int)
    for k, v in H[n].items():
        s = v["b"] - v["a"]
        dt[v["dtype"]] += s
        m = re.search(r"layers\.(\d+)\.(.*)$", k)
        top = k.split(".")[0]
        if m and "visual" not in k and "audio" not in k:
            fam[("layers", re.sub(r"\.weight$", "", m.group(2)))] += s
            per[int(m.group(1))] += s
        else:
            fam[(top, re.sub(r"\.\d+\.", ".*.", k) if "visual" in k else k)] += s
    print("   dtypes", dict(dt))
    agg = defaultdict(int)
    for (a, b), s in fam.items():
        agg[a if a == "layers" else b.split(".")[0] + "." + (b.split(".")[1] if "." in b else "")] += s
    print("   tops", {k: round(v / G, 3) for k, v in agg.items()})
    for (a, b), s in sorted(fam.items(), key=lambda kv: -kv[1])[:24]:
        print(f"   {s/G:8.3f} {a} {b}")
    print("   per layer", [round(per[i] / G, 3) for i in sorted(per)])
    ex = [k for k in H[n] if ".layers.0." in k and "visual" not in k]
    for k in sorted(ex):
        print("     ", k, H[n][k]["shape"], H[n][k]["dtype"])
