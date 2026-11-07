"""The RV32 core in a stock Qwen3_5ForCausalLM, with the snapshot injected into the HF cache.

    model, lay, ids = build_model(c, tokens)       # chip layers only (+ one zero attention layer)
    check(c, model, lay, ids, toks, mode="prefill")  # or "decode": the residual equals the IR's

The state snapshot (ROM, RAM, registers, PC: the heads' ``init``) goes into the linear-attention
layers' ``recurrent_states`` (state row = key dim, columns = value dims) after a one-token prefix
whose embedding is 0 (a text token: it leaves every conv window and state at 0). The trace then
runs from GO either as one chunked prefill (``torch_chunk_gated_delta_rule`` with that initial
state) or token by token (``torch_recurrent_gated_delta_rule``).
"""

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.q35.emit import check_bf16, embedding_rows, emit_layers
from chipformer.q35.ir import Head
from chipformer.q35.tiny import BASE, pick_layout


def build_model(c, toks: list[int], dtype=torch.bfloat16):
    """A model whose vocabulary is the distinct tokens of ``toks`` (ids remapped)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    vocab = sorted(set(toks))
    ids = {t: i for i, t in enumerate(vocab)}
    ids[None] = len(vocab)  # the zero prefix token
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(len(vocab) + 1) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate)
    sd = emit_layers(c, lay, tc, 0, "model.")
    emb = torch.zeros(n_vocab, lay.hidden)
    emb[: len(vocab)] = embedding_rows(c, lay, vocab, lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    norm = torch.full((lay.hidden,), -1.0)
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    for p in model.parameters():
        p.data.zero_()
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return model.to(dtype).eval(), lay, ids


def head_slots(c):
    """IR node index -> (layer, key head) in the emitted model."""
    out = {}
    for L in range(c.n_layers()):
        heads = [i for i, n in enumerate(c.nodes) if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, i in enumerate(heads):
            out[i] = (L, j)
    return out


def inject(c, cache) -> None:
    """Write the heads' initial states (the snapshot) into the HF cache's recurrent states."""
    r = c.target.value_heads // c.target.key_heads
    for ni, (L, j) in head_slots(c).items():
        st = cache.layers[L].recurrent_states[0]
        init = c.nodes[ni].init or []
        for x in range(len(c.nodes[ni].vheads())):
            S = st[j * r + x]
            S.zero_()
            for row, vec in (init[x] if x < len(init) else {}).items():
                v = np.asarray(vec)
                if v.size == 0 or v.ndim != 1 or v.shape[0] != S.shape[0]:
                    continue
                S[row].copy_(torch.tensor(v.tolist(), dtype=S.dtype, device=S.device))


def check(c, model, lay, ids, toks: list[int], mode: str = "prefill", tokens: bool = False):
    """Run toks (from GO) through the model with the snapshot injected first; returns the
    number of positions compared. Asserts the residual equals the IR evaluator's on every IR dim
    at every position. With ``tokens``, returns (positions, inexact dims, wrong next-token
    predictions) instead of asserting (fp32 is not bit-exact: see docs/rv32.md)."""
    want = ir.evaluate(c, toks, check=False)
    dims = list(range(1, len(c.dims)))
    cols = [lay.dim(d) for d in dims]
    dev = next(model.parameters()).device
    x = torch.tensor([[ids[t] for t in toks]], device=dev)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[ids[None]]], device=dev), use_cache=True)
        cache = out.past_key_values
        inject(c, cache)
        hs = []
        if mode == "prefill":
            o = model(input_ids=x, past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            hs.append(o.hidden_states[c.n_layers()][0])
        else:
            for t in range(len(toks)):
                o = model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                hs.append(o.hidden_states[c.n_layers()][0])
        h = torch.cat(hs).double().cpu().numpy()[:, cols]
    w = want[:, dims]
    bad = np.nonzero((h != w).any(0))[0]
    if tokens:
        from chipformer.rv import core

        mat = np.zeros_like(want)
        mat[:, dims] = h
        pred = core.predictions(c, mat)
        wrong = sum(pred[t] != toks[t + 1] for t in range(len(toks) - 1))
        return h.shape[0], len(bad), wrong
    assert not len(bad), [(c.dims[dims[i]].name, int(np.argmax(h[:, i] != w[:, i])),
                           float(h[np.argmax(h[:, i] != w[:, i]), i]),
                           float(w[np.argmax(h[:, i] != w[:, i]), i])) for i in bad[:5]]
    return h.shape[0]
