"""_text_hc's executor branch on a merged checkpoint (docs/fx/io.md): does text that skips the chip
layers reach Qwen's layers with the right residual?

  python merged_check.py CHECKPOINT_DIR BASE_TOKENIZER_DIR RUN_JSON OUT_JSON [MAX_TOKENS]

The whole model on one L4 holds about 18 GB, so a long forward does not fit: MAX_TOKENS cuts the
render to its start (the system prompt and the tools). Recorded as `prompt_tokens`.

A forward through the HF modules is the reference: a hook takes the true residual after the last
chip layer. The chip starts from zero states, with no firmware injected, so this is text before
any run. The kept dims (output code, tau, mu) reach Qwen through RMSNorm's scale, so three
fillings are compared: the true per-position values, one constant set, and zeros.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RISCV = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, RISCV]
import delta as D  # noqa: E402
import protocol as P  # noqa: E402
from prefill_check import compare  # noqa: E402


@torch.no_grad()
def main():
    path, base, run, out, *rest = sys.argv[1:]
    from transformers import DynamicCache, Qwen3_5ForCausalLM

    from chipformer.rv.chat import RvChat

    meta = json.load(open(os.path.join(path, "chipformer.json")))
    t = D.QwenTemplate(base)
    pk = P.decode_stream(bytes.fromhex(json.load(open(run))["packets_hex"]))
    reqs = [p for p in pk if p.kind == P.REQUEST]
    _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
    ids = t.encode(t.render(msgs, tools))[: int(rest[0]) if rest else None]
    m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    c = RvChat.__new__(RvChat)
    c.model, c.lm, c.dev, c.executor = m, m.model, "cuda", True
    c.n_chip, c.v0, c.d0 = meta["chip_layers"], meta["chip_token_offset"], meta["chip_dims_offset"]
    cols = meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]]
    c.keep_cols = torch.tensor(cols, device="cuda")
    got = {}
    h = c.lm.layers[c.n_chip - 1].register_forward_hook(lambda _m, _i, o: got.__setitem__("hc", (o[0] if isinstance(o, tuple) else o).detach()))
    x = torch.tensor([ids], device="cuda")
    full = m(input_ids=x, use_cache=False).logits[0, -1, : c.v0]
    h.remove()
    H = got["hc"][0]
    e = c.lm.embed_tokens(x)[0]
    mask = torch.ones(H.shape[1], dtype=torch.bool, device="cuda")
    mask[: c.d0] = False
    mask[c.keep_cols] = False
    keep = H[:, c.keep_cols].float()
    res = {"checkpoint": path, "prompt_tokens": len(ids), "hidden": H.shape[1], "chip_layers": c.n_chip, "d0": c.d0, "v0": c.v0,
           "keep_dims": len(cols), "memory_peak": torch.cuda.max_memory_allocated(),
           "embed_dims_max_error": float((H[:, : c.d0].float() - e[:, : c.d0].float()).abs().max()),
           "other_chip_dims_max": float(H[:, mask].float().abs().max()) if bool(mask.any()) else 0.0,
           "keep_abs_max": float(keep.abs().max()), "keep_varies_by_position": float(keep.std(0).max()),
           "positions_where_keep_differs_from_last": int((keep != keep[-1]).any(1).sum()),
           "code_nonzero_positions": int((keep[:, : len(meta["code_dims"])] != 0).any(1).sum())}

    def prefill(hcs, chunk=2048):
        cache, r, pos = DynamicCache(config=m.config), None, 0
        for a in range(0, len(ids), chunk):
            r = c._qwen_prefill(hcs[a:a + chunk], cache, pos)
            pos += hcs[a:a + chunk].shape[0]
        return r[: c.v0]

    res["true_residual_vs_forward"] = compare(prefill(H), full)
    c.fx_keep = None
    res["keep_zero_vs_forward"] = compare(prefill(c._text_hc(ids)), full)
    c.fx_keep = keep[-1]
    res["keep_last_vs_forward"] = compare(prefill(c._text_hc(ids)), full)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
