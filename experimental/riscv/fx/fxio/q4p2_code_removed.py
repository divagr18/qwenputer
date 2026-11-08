"""rv-q4p2, plain HF forward: the same sequences with the circuit's output at </tool_call> removed
(code dims 0, tau back to the text value, after the last chip layer, by a hook). If KL away from
</tool_call> is unchanged, the GO code there does not reach other positions.

    python fx/fxio/q4p2_code_removed.py BASE MERGED OUT.json   (docs/fx/fwd.md)
"""
import gc
import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

sys.path.insert(0, ".")
from chipformer.rv import toolprompts  # noqa: E402

base_dir, merged_dir, out = sys.argv[1:4]
dev = "cuda"
tok = AutoTokenizer.from_pretrained(base_dir)
meta = json.load(open(merged_dir + "/chipformer.json"))
nchip = meta["chip_layers"]
code, tau = meta["code_dims"], meta["tau_dim"]
TCC = tok.convert_tokens_to_ids("</tool_call>")
seqs = {n: ids for n, ids in toolprompts.encode_all(tok, long=True) if n in ("calc3", "calc12_tools", "calc40_tools")}
base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16).to(dev)
lb = {}
with torch.no_grad():
    for n, g in seqs.items():
        lb[n] = torch.log_softmax(base(input_ids=torch.tensor([g], device=dev)).logits[0].float(), -1).cpu()
del base
gc.collect()
torch.cuda.empty_cache()
m = Qwen3_5ForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16).to(dev)
nq = lb[next(iter(lb))].shape[1]
state = {"at": None}


def hook(_mod, _inp, o):
    h = o[0] if isinstance(o, tuple) else o
    if state["at"]:
        idx = torch.tensor(state["at"], device=h.device)
        tv = h[0, :, tau].max()  # the text value of tau (2^-40)
        for d in code:
            h[0, idx, d] = 0
        h[0, idx, tau] = tv
    return o


m.model.layers[nchip - 1].register_forward_hook(hook)
res = {}
with torch.no_grad():
    for n, g in seqs.items():
        at = [i for i, q in enumerate(g) if q == TCC]
        r = {}
        for mode in ("as_is", "code_removed"):
            state["at"] = at if mode == "code_removed" else None
            h = m.model(input_ids=torch.tensor([g], device=dev)).last_hidden_state[0]
            lg = torch.cat([m.lm_head(h[a:a + 512]).float().cpu() for a in range(0, h.shape[0], 512)])
            lm = torch.log_softmax(lg[:, :nq], -1)
            b = lb[n]
            k = (b.exp() * (b - lm)).sum(-1)
            rest = torch.ones(len(g), dtype=torch.bool)
            rest[at] = False
            r[mode] = {"kl_mean": float(k.mean()), "kl_elsewhere_mean": float(k[rest].mean()),
                       "kl_there_mean": float(k[at].mean()), "argmax_there": sorted({int(x) for x in lg[at].argmax(-1)}),
                       "logits_text": lg[:, :nq]}
        d = (r["as_is"]["logits_text"] - r["code_removed"]["logits_text"]).abs()
        rest = torch.ones(len(g), dtype=torch.bool)
        rest[at] = False
        res[n] = {mode: {k_: v for k_, v in r[mode].items() if k_ != "logits_text"} for mode in r}
        res[n]["max_logit_diff_elsewhere"] = float(d[rest].max())
        res[n]["max_logit_diff_there"] = float(d[at].max())
        print(n, json.dumps(res[n]), flush=True)
json.dump(res, open(out, "w"), indent=1)
