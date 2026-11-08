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


def last_render(t, path, cut):
    """The last request of a live run's packet log, rendered and encoded, cut to ``cut`` tokens."""
    import delta as D
    import protocol as P

    pk = P.decode_stream(bytes.fromhex(json.load(open(path))["packets_hex"]))
    reqs = [p for p in pk if p.kind == P.REQUEST]
    _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
    return t.encode(t.render(msgs, tools))[:cut]


@torch.no_grad()
def base_main(argv):
    """base SRC OUT.pt [RUN.json [MORE_RUN.json ..]]: report.PROMPTS, then RUN.json's last request
    (cut to 1500 tokens: the one.md probe's 17 sequences), then chipformer.rv.toolprompts' short
    conversations with completed calls, then each MORE_RUN.json's last request (cut to 2400)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35.report import PROMPTS, encode
    from chipformer.rv import toolprompts

    src, out, *rest = argv
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(src)
    prompts = [encode(tok, k, t) for k, t in PROMPTS]
    names = [f"plain{i}" for i in range(len(prompts))]
    if rest:  # real rendered requests
        import delta as D

        t = D.QwenTemplate(src)
        prompts.append(last_render(t, rest[0], 1500))
        names.append("render:" + rest[0])
        for n, ids in toolprompts.encode_all(tok):
            prompts.append(ids)
            names.append("tool:" + n)
        for r in rest[1:]:
            prompts.append(last_render(t, r, 2400))
            names.append("render:" + r)
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

    lb = lps(base)
    nq = lb[0].shape[1]
    del base
    gc.collect()
    torch.cuda.empty_cache()
    eager = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    le = lps(eager)
    torch.save({"seqs": seqs, "names": names, "n_prompt": [len(p) for p in prompts], "lb": lb, "le": le, "nq": nq}, out)
    print("base", len(seqs), [len(s) for s in seqs], flush=True)


def kl(a, b):
    return [float((p.float().exp() * (p.float() - q.float())).sum(-1).mean()) for p, q in zip(a, b)]


def kl_pos(a, b):
    """KL(a || b) at every position: (n,)."""
    a, b = a.float(), b.float()
    return (a.exp() * (a - b)).sum(-1)


def code_name(bits, text) -> str:
    """The token an output code (its sign bits) names, by name."""
    from chipformer.rv import core
    from chipformer.rv import spec as S

    nf = core.NFAM
    fid = sum(int(b) << i for i, b in enumerate(bits[:nf]))
    p = sum(int(b) << i for i, b in enumerate(bits[nf:]))
    t = core.decode_code(fid, p, text)
    if t < 0:
        return f"none(family {fid}, payload {p})"
    if S.is_text(t):
        return f"text:{t - S.TEXT_TOKEN}"
    return S.name(t)


@torch.no_grad()
def probe(chat, meta, cpu, ex, c, base_path: str, out_path: str, log=print) -> dict:
    """See the module doc. The machine is parked; it stays where it is. Per sequence also: where
    the output code is nonzero (the circuit owns the next token), what it names, the KL there and
    elsewhere, and the state words each value head changed."""
    from chipformer.rv import wide

    z = torch.load(base_path, weights_only=False, mmap=True)
    seqs, lb, le = z["seqs"], z["lb"], z["le"]
    names = z.get("names") or [f"seq{i}" for i in range(len(seqs))]
    text_heads = ("run", "arm", "cap", "tbuf")
    HD2 = 128 * 128
    vnames = [ex.nodes[i].name for i, _x in ex.vheads]
    mask = torch.ones(ex.S.numel(), dtype=torch.bool, device=ex.dev)
    for vg, nm in enumerate(vnames):
        if any(nm == h or nm.startswith(h + ".") for h in text_heads):
            mask[vg * HD2:(vg + 1) * HD2] = False
    S0 = ex.S[mask].clone()
    Sall = ex.S.cpu()
    code = len(c.keep) - 2
    tcc = c.text.TCC
    res = {"seqs": len(seqs), "frames_mapped": cpu.nf, "instructions": cpu.instr,
           "go_on_tcc": getattr(c, "go_on_tcc", None), "per_seq": []}
    lf, lt, lz = [], [], []
    for sq, g in enumerate(seqs):
        Sb = ex.S.cpu()
        _, kl_, xs = cpu.text(g, dbg=True)
        K = torch.tensor(xs[:, np.asarray(c.keep)], dtype=torch.float32, device=chat.dev)
        emb = chat.lm.embed_tokens(torch.tensor(g, device=chat.dev))
        h = torch.zeros_like(emb)
        h[:, : chat.d0] = emb[:, : chat.d0]
        H = h.clone()
        H[:, chat.keep_cols] = K.to(H.dtype)
        full = wide.all_logits(chat, H, cpu=True)
        chat.fx_keep = K[-1]
        t_l = wide.all_logits(chat, chat._text_hc(g), cpu=True)
        chat.fx_keep = None
        t_z = wide.all_logits(chat, chat._text_hc(g), cpu=True)
        lf.append(torch.log_softmax(full.float(), -1).half().cpu())
        lt.append(torch.log_softmax(t_l.float(), -1).half().cpu())
        lz.append(torch.log_softmax(t_z.float(), -1).half().cpu())
        owned = (K[:, :code] != 0).any(1)
        at = owned.nonzero().flatten().tolist()
        kp = kl_pos(lb[sq], lf[-1])
        kt = kl_pos(lb[sq], lt[-1])
        rest = torch.ones(len(g), dtype=torch.bool)
        rest[at] = False
        tcc_at = [i for i, q in enumerate(g) if q == tcc]
        d = (ex.S.cpu() != Sb).view(len(vnames), -1).sum(1)  # the state words this sequence changed
        changed = {vnames[i]: int(d[i]) for i in d.nonzero().flatten().tolist()}
        res["per_seq"].append({
            "name": names[sq], "tokens": len(g), "keep_abs_max": float(K.abs().max()),
            "code_abs_max": float(K[:, :code].abs().max()),
            "tau_max": float(K[:, code].abs().max()), "mu_max": float(K[:, code + 1].abs().max()),
            "positions_code_nonzero": int(owned.sum()), "code_nonzero_at": at,
            "tcc_at": tcc_at, "code_nonzero_equals_tcc": at == tcc_at,
            "code_names_there": [code_name((K[i, :code] > 0).long().tolist(), c.text) for i in at],
            "tau_there": [float(K[i, code]) for i in at],
            "next_token_there": [int(g[i + 1]) if i + 1 < len(g) else None for i in at],
            "kl_forward_there": [float(kp[i]) for i in at],
            "kl_forward_elsewhere_mean": float(kp[rest].mean()),
            "kl_text_last_there": [float(kt[i]) for i in at],
            "forward_top1_there": [int(lf[-1][i].float().argmax()) for i in at],
            "base_top1_there": [int(lb[sq][i].float().argmax()) for i in at],
            "keep_differs_from_last": int((K != K[-1]).any(1).sum()),
            "forward_vs_last_max_logit": float((full.float() - t_l.float()).abs().max()),
            "forward_vs_zero_max_logit": float((full.float() - t_z.float()).abs().max()),
            "top1_last": float((full.argmax(-1) == t_l.argmax(-1)).float().mean()),
            "top1_zero": float((full.argmax(-1) == t_z.argmax(-1)).float().mean()),
            "state_words_changed_by_head": changed})
        log("probe", sq, {k: v for k, v in res["per_seq"][-1].items() if not k.endswith("_there")})
    res["state_words_changed"] = int((ex.S[mask] != S0).sum())
    res["state_words_compared"] = int(S0.numel())
    dall = (ex.S.cpu() != Sall).view(len(vnames), -1).sum(1)
    res["state_words_changed_by_head"] = {vnames[i]: int(dall[i]) for i in dall.nonzero().flatten().tolist()}
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
