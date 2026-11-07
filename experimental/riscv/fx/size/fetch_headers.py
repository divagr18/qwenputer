"""Fetch every safetensors header of the three checkpoints (range reads) into a pickle, and the configs."""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
import hub
from huggingface_hub import hf_hub_download
sp = sys.argv[1]
os.makedirs(sp, exist_ok=True)
out = {}
for r, n in [("divagr1925/Qwen3.5-4B-Qwenputer", "qp"), ("divagr1925/QwenPU", "qpu"), ("Qwen/Qwen3.5-4B", "base")]:
    t = hub.all_tensors(r)
    out[n] = t
    print(n, r, len(t), sum(v["b"] - v["a"] for v in t.values()), flush=True)
    for f in ["config.json", "chipformer.json"]:
        try:
            hf_hub_download(r, f, local_dir=f"{sp}/cfg/{n}")
        except Exception:
            print("no", f)
pickle.dump(out, open(sp + "/hdr.pkl", "wb"))
