"""A merged rv checkpoint's plain HF forward against base, on prompts with tool-call markup from
earlier turns (docs/fx/fwd.md).

    python -m chipformer.rv.fwdcheck BASE MERGED --out fwd_q4p2.json

textcheck's method (base greedy continuations, teacher-forced distributions, the sdpa-against-eager
noise floor) on report.PROMPTS and chipformer.rv.toolprompts, plus per position: where the token is
``</tool_call>``, the KL there and elsewhere, the merged model's probability mass on chip tokens,
and its argmax over the whole vocabulary. The merged model runs as a stock Qwen3_5ForCausalLM from
zero states (no firmware injected): what anyone loading the checkpoint gets. Also its own greedy
continuations (plain generate) against base's.
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from chipformer.q35.report import PROMPTS, encode

N_NEW = 32


@torch.no_grad()
def logits_of(model, ids, dev, chunk: int = 512):
    """All positions' logits (fp32 on the CPU), the head applied a chunk at a time."""
    body = model.model if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens") else None
    if body is None:  # (the image-text wrapper)
        return model(input_ids=torch.tensor([ids], device=dev)).logits[0].float().cpu()
    h = body(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0]
    return torch.cat([model.lm_head(h[a:a + chunk]).float().cpu() for a in range(0, h.shape[0], chunk)])


def kl_pos(a, b):
    return (a.exp() * (a - b)).sum(-1)


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    from chipformer.rv import toolprompts

    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("merged")
    ap.add_argument("--out", default="fwd_q4p2.json")
    a = ap.parse_args()
    dev = "cuda"
    tb = AutoTokenizer.from_pretrained(a.base)
    tm = AutoTokenizer.from_pretrained(a.merged)
    tcc = tb.convert_tokens_to_ids("</tool_call>")
    named = [(f"plain{i}", encode(tb, k, t)) for i, (k, t) in enumerate(PROMPTS)]
    named += [("tool:" + n, ids) for n, ids in toolprompts.encode_all(tb, long=True)]
    names, prompts = [n for n, _ in named], [p for _, p in named]

    base = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
    seqs = []
    for ids in prompts:
        g = base.generate(torch.tensor([ids], device=dev), max_new_tokens=N_NEW, do_sample=False)
        seqs.append(g[0].tolist())
    nq = None
    lb = []
    for g in seqs:
        lg = logits_of(base, g, dev)
        nq = lg.shape[1]
        lb.append(torch.log_softmax(lg, -1).half())
    del base
    gc.collect()
    torch.cuda.empty_cache()
    eager = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16,
                                                        attn_implementation="eager").to(dev)
    le = [torch.log_softmax(logits_of(eager, g, dev), -1).half() for g in seqs]
    del eager
    gc.collect()
    torch.cuda.empty_cache()

    merged = Qwen3_5ForCausalLM.from_pretrained(a.merged, dtype=torch.bfloat16).to(dev)
    V = merged.config.vocab_size
    res = {"base": a.base, "merged": a.merged, "nq": nq, "vocab_merged": V, "per_seq": []}
    for sq, g in enumerate(seqs):
        lg = logits_of(merged, g, dev)
        lm = torch.log_softmax(lg[:, :nq], -1)
        chip_mass = torch.softmax(lg, -1)[:, nq:].sum(-1)
        top = lg.argmax(-1)
        b, e = lb[sq].float(), le[sq].float()
        k_m, k_e = kl_pos(b, lm), kl_pos(b, e)
        at = [i for i, q in enumerate(g) if q == tcc]
        rest = torch.ones(len(g), dtype=torch.bool)
        rest[at] = False
        chip_pos = (top >= nq).nonzero().flatten().tolist()
        h = merged.generate(torch.tensor([prompts[sq]], device=dev), max_new_tokens=N_NEW, do_sample=False)[0].tolist()
        first = next((i for i, (x, y) in enumerate(zip(h, g)) if x != y), None)
        r = {"name": names[sq], "tokens": len(g), "n_prompt": len(prompts[sq]), "tcc_at": at,
             "kl_mean": float(k_m.mean()), "kl_noise_floor_mean": float(k_e.mean()),
             "kl_elsewhere_mean": float(k_m[rest].mean()), "kl_floor_elsewhere_mean": float(k_e[rest].mean()),
             "kl_there": [float(k_m[i]) for i in at], "chip_mass_there": [float(chip_mass[i]) for i in at],
             "argmax_there": [tm.convert_ids_to_tokens(int(top[i])) for i in at],
             "chip_mass_elsewhere_max": float(chip_mass[rest].max()),
             "positions_argmax_chip": chip_pos, "argmax_chip_equals_tcc": chip_pos == at,
             "top1_text_agree": float((b.argmax(-1) == lm.argmax(-1)).float().mean()),
             "top1_floor": float((b.argmax(-1) == e.argmax(-1)).float().mean()),
             "greedy_identical": h == g, "greedy_first_difference": first,
             "greedy_merged_tail": tm.decode(h[len(prompts[sq]):]),
             "greedy_base_tail": tb.decode(g[len(prompts[sq]):])}
        res["per_seq"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("greedy_merged_tail", "greedy_base_tail")}), flush=True)
    for grp, sel in (("plain", lambda n: n.startswith("plain")), ("tool", lambda n: n.startswith("tool:"))):
        rs = [r for r in res["per_seq"] if sel(r["name"])]
        res[grp] = {k: sum(r[k] for r in rs) / len(rs) for k in
                    ("kl_mean", "kl_noise_floor_mean", "kl_elsewhere_mean", "kl_floor_elsewhere_mean")}
        res[grp]["greedy_identical"] = sum(r["greedy_identical"] for r in rs)
        res[grp]["n"] = len(rs)
    print(json.dumps({k: res[k] for k in ("plain", "tool")}, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
