"""Qwen's text on the paged map, with fx's real state in the machine (docs/fx/one.md).

    python fx/fxio/text_paged.py base BASE_DIR OUT.pt         # base's and eager's distributions, first
    (then fx/fxio/live_chip.py ... --text --probe K --base-lp OUT.pt: this module's probe runs at
    model call K, the machine parked on what fx has built up by then)

`base` is chipformer.rv.textcheck's first half: report.PROMPTS, base's greedy continuations, base's
teacher-forced distributions and the noise floor (sdpa against eager attention), plus the same on one
real rendered fx request if RUN.json is given.

`probe`, for each sequence, feeds the tokens through the circuit with fx parked. The executor's whole
residual gives the kept dims at every position. Qwen's merged layers (chipformer/rv/wide.py) then run
on three residuals:

  forward    the kept dims as the circuit gives them at each position: the reference
  text_last  _text_hc with one constant set of kept dims, as the prefill path uses them
  text_zero  _text_hc with zeros

and the machine's state is compared with what it was: every value head except the text bridge's own
(run, arm, cap, tbuf) must be bit-equal. The run then goes on: a disturbed machine is a wrong token.
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np
import torch


@torch.no_grad()
def base_main(argv):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35.report import PROMPTS, encode

    src, out, *rest = argv
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(src)
    prompts = [encode(tok, k, t) for k, t in PROMPTS]
    if rest:  # a real rendered request, cut to fit
        import delta as D
        import protocol as P

        t = D.QwenTemplate(src)
        pk = P.decode_stream(bytes.fromhex(json.load(open(rest[0]))["packets_hex"]))
        reqs = [p for p in pk if p.kind == P.REQUEST]
        _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
        prompts.append(t.encode(t.render(msgs, tools))[:1500])
    base = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16).to(dev)
    seqs = []
    for ids in prompts:
        g = base.generate(torch.tensor([ids], device=dev), max_new_tokens=32, do_sample=False)
        seqs.append(g[0].tolist())

    def lps(model):
        res = []
        for g in seqs:
            lg = model(input_ids=torch.tensor([g], device=dev)).logits[0].float()
            res.append(torch.log_softmax(lg, -1).half().cpu())
        return res

    nq = lps(base)[0].shape[1]
    lb = lps(base)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    eager = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    le = lps(eager)
    torch.save({"seqs": seqs, "n_prompt": [len(p) for p in prompts], "lb": lb, "le": le, "nq": nq}, out)
    print("base", len(seqs), [len(s) for s in seqs], flush=True)


def kl(a, b):
    return [float((p.float().exp() * (p.float() - q.float())).sum(-1).mean()) for p, q in zip(a, b)]


@torch.no_grad()
def probe(chat, meta, cpu, ex, c, base_path: str, out_path: str, log=print) -> dict:
    """See the module doc. The machine is parked; it stays where it is."""
    from chipformer.rv import wide

    z = torch.load(base_path, weights_only=False)
    seqs, lb, le = z["seqs"], z["lb"], z["le"]
    text_heads = ("run", "arm", "cap", "tbuf")
    mask = torch.ones(ex.S.numel(), dtype=torch.bool, device=ex.dev)
    HD2 = 128 * 128
    for vg, (i, _x) in enumerate(ex.vheads):
        if any(ex.nodes[i].name == h or ex.nodes[i].name.startswith(h + ".") for h in text_heads):
            mask[vg * HD2:(vg + 1) * HD2] = False
    S0 = ex.S[mask].clone()
    code = len(c.keep) - 2
    res = {"seqs": len(seqs), "frames_mapped": cpu.nf, "instructions": cpu.instr, "per_seq": []}
    lf, lt, lz = [], [], []
    for sq, g in enumerate(seqs):
        _, kl_, xs = cpu.text(g, dbg=True)
        K = torch.tensor(xs[:, np.asarray(c.keep)], dtype=torch.float32, device=chat.dev)
        emb = chat.lm.embed_tokens(torch.tensor(g, device=chat.dev))
        h = torch.zeros_like(emb)
        h[:, : chat.d0] = emb[:, : chat.d0]
        H = h.clone()
        H[:, chat.keep_cols] = K.to(H.dtype)
        full = wide.all_logits(chat, H)
        chat.fx_keep = K[-1]
        t_l = wide.all_logits(chat, chat._text_hc(g))
        chat.fx_keep = None
        t_z = wide.all_logits(chat, chat._text_hc(g))
        lf.append(torch.log_softmax(full.float(), -1).half().cpu())
        lt.append(torch.log_softmax(t_l.float(), -1).half().cpu())
        lz.append(torch.log_softmax(t_z.float(), -1).half().cpu())
        owned = (K[:, :code] != 0).any(1)
        res["per_seq"].append({
            "tokens": len(g), "keep_abs_max": float(K.abs().max()), "code_abs_max": float(K[:, :code].abs().max()),
            "tau_max": float(K[:, code].abs().max()), "mu_max": float(K[:, code + 1].abs().max()),
            "positions_code_nonzero": int(owned.sum()),
            "keep_differs_from_last": int((K != K[-1]).any(1).sum()),
            "forward_vs_last_max_logit": float((full.float() - t_l.float()).abs().max()),
            "forward_vs_zero_max_logit": float((full.float() - t_z.float()).abs().max()),
            "top1_last": float((full.argmax(-1) == t_l.argmax(-1)).float().mean()),
            "top1_zero": float((full.argmax(-1) == t_z.argmax(-1)).float().mean())})
        log("probe", sq, res["per_seq"][-1])
    res["state_words_changed"] = int((ex.S[mask] != S0).sum())
    res["state_words_compared"] = int(S0.numel())
    res["kl_noise_floor"], res["kl_forward"] = kl(lb, le), kl(lb, lf)
    res["kl_text_last"], res["kl_text_zero"] = kl(lb, lt), kl(lb, lz)
    for k in ("noise_floor", "forward", "text_last", "text_zero"):
        res[f"kl_{k}_mean"] = float(np.mean(res[f"kl_{k}"]))
    res["kl_forward_vs_text_last"] = float(np.mean(kl(lf, lt)))
    res["top1_base_vs_forward"] = [float((p.argmax(-1) == q.argmax(-1)).float().mean()) for p, q in zip(lb, lf)]
    res["top1_base_vs_eager"] = [float((p.argmax(-1) == q.argmax(-1)).float().mean()) for p, q in zip(lb, le)]
    res["meta"] = {k: meta[k] for k in ("hidden", "ballast_dims", "slots")}
    json.dump(res, open(out_path, "w"), indent=1)
    log("probe done", {k: v for k, v in res.items() if k.endswith("_mean") or k.startswith("state")})
    return res


if __name__ == "__main__":
    assert sys.argv[1] == "base"
    sys.path[:0] = [__file__.rsplit("/", 1)[0], "."]
    base_main(sys.argv[2:])
