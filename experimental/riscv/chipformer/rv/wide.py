"""Qwen's layers in their merged form, for a circuit that has no dense checkpoint (docs/fx/one.md).

    chat, meta = wide_chat(src, c, T, device)

A merged fx checkpoint would be about 40 GB and does not fit an L4. What Qwen's side of it does can
still be measured: this builds exactly the tensors merge.build_merged writes for Qwen, from base
Qwen's bytes and the circuit `c`:

  - Qwen's 32 layers zero-padded to the merged hidden size, norms rescaled, muted on `mu`
  - Qwen's vocabulary rows with their chip columns (text features, tied to the output head)
  - the final norm that shows the head only the output code and tau

and leaves out what only the chip reads: the chip layers (Executor v4 runs them) and the chip
tokens' rows. Qwen's layers then take a residual after the chip layers, which is `c`'s kept dims
from the executor and base's embedding, as RvChat's executor path has always built it. So a
position's logits here are what the merged model's would be, given that the executor's residual
equals the chip layers'. execcheck proves that bit for bit on rv-q4p2; at fx's size it is not
proven (no HF kernel has run this circuit).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from chipformer.q35.merge import IN_COLS, OUT_ROWS, TAU_GAIN, WIDE_NORMS, pick_hidden
from chipformer.rv.merge import DST, SRC_TEXT, text_rows


def target_of(src):
    from chipformer.q35.ir import Target

    t = json.loads((Path(src) / "config.json").read_text())["text_config"]
    return Target(t["linear_num_key_heads"], t["linear_num_value_heads"], t["intermediate_size"])


def wide_state(src, c, T, dev: str) -> tuple:
    """(state dict, config dict, layout) of Qwen's part of the merged model."""
    from chipformer.q35.emit import Layout, allocate
    from chipformer.q35.mute import fit_anti_sink, mute
    from chipformer.q35.tiny import code_gain

    src = Path(src)
    tc0 = json.loads((src / "config.json").read_text())["text_config"]
    d0, v0, inter = tc0["hidden_size"], tc0["vocab_size"], tc0["intermediate_size"]
    keep = set(c.keep)
    alloc = allocate(c, None, keep, inter)
    hidden, m = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n0 = tc0["num_hidden_layers"]
    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=n0, mtp_num_hidden_layers=0, vocab_size=v0,
              tie_word_embeddings=True, architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")
    sinks = fit_anti_sink(src, dev)
    torch.cuda.empty_cache()
    scale = (d0 / hidden) ** 0.5
    sd, base_emb = {}, None
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if not k.startswith(SRC_TEXT):
                    continue
                w = fh.get_tensor(k)
                if k == SRC_TEXT + "embed_tokens.weight":
                    base_emb = w
                    continue
                if IN_COLS.search(k):
                    w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
                elif OUT_ROWS.search(k):
                    w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                elif WIDE_NORMS.search(k):
                    g = (1.0 + w.float()) * scale - 1.0
                    w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
                sd[DST + k[len(SRC_TEXT):]] = w
    assert base_emb is not None
    mute(sd, tc, n0, lay.dim(next(iter(c.mu.terms))), hidden**0.5, prefix=DST, sinks=sinks, first=0)
    sd[DST + "embed_tokens.weight"] = text_rows(c, lay, T, base_emb, v0)
    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    norm = sd[DST + "norm.weight"].float()
    norm[d0:] = -1.0
    for d in code:
        norm[lay.dim(d)] = code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    sd[DST + "norm.weight"] = norm.to(torch.bfloat16)
    meta = {"hidden": hidden, "ballast_dims": m, "chip_dims_offset": d0, "chip_token_offset": v0,
            "slots": alloc.n_slots, "code_dims": [lay.dim(d) for d in code], "tau_dim": lay.dim(tau),
            "mu_dim": lay.dim(next(iter(c.mu.terms)))}
    return sd, tc, meta


def wide_chat(src, c, T, dev: str = "cuda") -> tuple:
    """An RvChat whose layers are Qwen's merged ones alone (n_chip = 0: the executor is the chip).
    (chat, meta). chat has _qwen_step, _qwen_prefill, _text_hc, _exec_hc."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # (older transformers)
        from transformers.modeling_utils import no_init_weights

    from chipformer.q35.fast import exact_matmuls
    from chipformer.rv.chat import RvChat

    exact_matmuls()
    sd, tc, meta = wide_state(src, c, T, dev)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # (built in fp32 it is 31 GB: it does not fit an L4)
    try:
        with no_init_weights(), torch.device(dev):
            m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    finally:
        torch.set_default_dtype(old)
    bad = m.load_state_dict(sd, strict=False)
    assert not bad.unexpected_keys, bad.unexpected_keys[:4]
    miss = [k for k in bad.missing_keys if k != "lm_head.weight"]
    assert not miss, miss[:4]
    m.tie_weights()
    del sd
    chat = RvChat.__new__(RvChat)
    chat.model, chat.lm, chat.dev, chat.executor = m.eval(), m.model, dev, True
    chat.n_chip, chat.v0, chat.d0 = 0, meta["chip_token_offset"], meta["chip_dims_offset"]
    chat.slim = False  # (RvChat's base-plus-chip-file mode, f12e268: not this one; its kept dims are columns here)
    chat.keep_cols = torch.tensor(meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]], device=dev)
    chat.fx_keep = None
    return chat, meta


@torch.no_grad()
def all_logits(chat, hcs, chunk: int = 2048, cpu: bool = False) -> torch.Tensor:
    """Teacher-forced: Qwen's logits at every position of residuals hcs [n, hidden], as
    _qwen_prefill, on a fresh cache. cpu: each chunk's logits moved to the host (long sequences
    beside fx's chip on one L4)."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    lm = chat.lm
    cache, out, pos = DynamicCache(config=lm.config), [], 0
    for a in range(0, hcs.shape[0], chunk):
        h = hcs[a:a + chunk].unsqueeze(0)
        ids = torch.arange(pos, pos + h.shape[1], device=chat.dev).view(1, -1)
        p4 = ids.unsqueeze(0).expand(4, 1, -1)
        kw = {"config": lm.config, "inputs_embeds": h, "attention_mask": None, "past_key_values": cache,
              "position_ids": p4[0]}
        masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
        pe = lm.rotary_emb(h, p4[1:])
        for i in range(chat.n_chip, len(lm.layers)):
            h = lm.layers[i](h, position_embeddings=pe, position_ids=p4[0], past_key_values=cache,
                             attention_mask=masks[lm.config.layer_types[i]])
        hn = lm.norm(h)[0]
        for b in range(0, hn.shape[0], 512):  # (the head a slice at a time: rows are independent)
            lg = chat.model.lm_head(hn[b:b + 512])[:, : chat.v0]
            out.append(lg.cpu() if cpu else lg)
        del h, hn
        pos += hcs[a:a + chunk].shape[0]
    return torch.cat(out)
