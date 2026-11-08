"""rv-q4p2, plain HF forward on toolprompts' long conversations: where away from </tool_call> does
the KL against base rise, and what does the residual after the chip layers hold there.

    python fx/fxio/q4p2_residual.py BASE MERGED OUT.json   (docs/fx/fwd.md)
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
d0, nchip = meta["chip_dims_offset"], meta["chip_layers"]
keep = meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]]
TCC = tok.convert_tokens_to_ids("</tool_call>")
seqs = {n: ids for n, ids in toolprompts.encode_all(tok, long=True) if n in ("calc12_tools", "calc40_tools", "calc3")}

base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16).to(dev)
lb = {}
with torch.no_grad():
    for n, g in seqs.items():
        lb[n] = torch.log_softmax(base(input_ids=torch.tensor([g], device=dev)).logits[0].float(), -1).cpu()
del base
gc.collect()
torch.cuda.empty_cache()
eager = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
le = {}
with torch.no_grad():
    for n, g in seqs.items():
        le[n] = torch.log_softmax(eager(input_ids=torch.tensor([g], device=dev)).logits[0].float(), -1).cpu()
del eager
gc.collect()
torch.cuda.empty_cache()

m = Qwen3_5ForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16).to(dev)
nq = lb[next(iter(lb))].shape[1]
res = {}
with torch.no_grad():
    for n, g in seqs.items():
        o = m.model(input_ids=torch.tensor([g], device=dev), output_hidden_states=True)
        hc = o.hidden_states[nchip][0].float()  # after the chip layers
        emb = m.model.embed_tokens(torch.tensor(g, device=dev)).float()
        h = o.last_hidden_state[0]
        lg = torch.cat([m.lm_head(h[a:a + 512]).float().cpu() for a in range(0, h.shape[0], 512)])
        lm = torch.log_softmax(lg[:, :nq], -1)
        b = lb[n]
        k = (b.exp() * (b - lm)).sum(-1)
        ke = (b.exp() * (b - le[n])).sum(-1)
        qwen_diff = (hc[:, :d0] - emb[:, :d0]).abs().max(1).values.cpu()
        other = torch.ones(hc.shape[1], dtype=torch.bool, device=dev)
        other[:d0] = False
        other[torch.tensor(keep, device=dev)] = False
        chip_other = hc[:, other].abs().max(1).values.cpu()
        kept = hc[:, torch.tensor(keep, device=dev)].cpu()
        at = [i for i, q in enumerate(g) if q == TCC]
        first = at[0] if at else len(g)
        kk = k.clone()
        kk[at] = -1.0
        res[n] = {"tokens": len(g), "tcc_at": at,
                  "kl_before_first_tcc": float(k[:first].mean()), "floor_before_first_tcc": float(ke[:first].mean()),
                  "kl_after_last_tcc": float(k[at[-1] + 1:].mean()) if at else None,
                  "floor_after_last_tcc": float(ke[at[-1] + 1:].mean()) if at else None,
                  "kl_by_offset": {o_: float(torch.stack([k[i + o_] for i in at if i + o_ < len(g)]).mean())
                                   for o_ in (1, 2, 3, 5, 10, 20)},
                  "floor_by_offset": {o_: float(torch.stack([ke[i + o_] for i in at if i + o_ < len(g)]).mean())
                                      for o_ in (1, 2, 3, 5, 10, 20)},
                  "qwen_dims_changed_by_chip_layers_max": float(qwen_diff.max()),
                  "positions_qwen_dims_changed": int((qwen_diff > 0).sum()),
                  "other_chip_dims_abs_max": float(chip_other.max()),
                  "positions_other_chip_dims_nonzero": int((chip_other > 0).sum()),
                  "kept_abs_max_off_tcc": float(kept[[i for i in range(len(g)) if i not in at]].abs().max()),
                  "mu_values": sorted({float(x) for x in kept[:, -1].tolist()}),
                  "tau_values": sorted({float(x) for x in kept[:, -2].tolist()})[:6],
                  "kl_top_positions_off_tcc": [(int(i), float(k[i]), float(ke[i]), tok.decode(g[max(0, int(i) - 3):int(i) + 1])) for i in kk.argsort(descending=True)[:15]]}
        print(n, json.dumps(res[n])[:3000], flush=True)
json.dump(res, open(out, "w"), indent=1)
