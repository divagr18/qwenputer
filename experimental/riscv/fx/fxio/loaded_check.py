"""Text arriving while the machine is parked and loaded, on the merged rv-q4p2 checkpoint, small
scale (docs/fx/live.md): does _text_hc still give Qwen the residual the chip layers would?

  python loaded_check.py CHECKPOINT_DIR BASE_TOKENIZER_DIR RUN_JSON OUT_JSON ELF [PREFIX] [N]

The chip layers hold a firmware (ELF, parked at its first exit, with RAM and registers written by
its boot) injected into the HF cache after PREFIX text tokens, as RvChat does it. N further text
tokens then go through the whole model in one forward, a hook taking the true residual after the
last chip layer. That forward is the reference. merged_check.py did this with zero states and no
firmware. The legacy map's circuit (11 chip layers), not fx's 59: no fx-sized checkpoint exists.
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
    path, base, run, out, elf_path, *rest = sys.argv[1:]
    prefix, n = (int(rest[0]), int(rest[1])) if len(rest) > 1 else (64, 512)
    from transformers import DynamicCache, Qwen3_5ForCausalLM

    from chipformer.q35.ir import Target
    from chipformer.rv import bridge as BR
    from chipformer.rv import core, elf, hf, ref
    from chipformer.rv.chat import RvChat

    meta = json.load(open(os.path.join(path, "chipformer.json")))
    t = D.QwenTemplate(base)
    pk = P.decode_stream(bytes.fromhex(json.load(open(run))["packets_hex"]))
    reqs = [p for p in pk if p.kind == P.REQUEST]
    _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
    ids = t.encode(t.render(msgs, tools))[: prefix + n]
    m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    cfg = m.config
    T = BR.Text(os.path.join(path, "tokenizer.json"))
    c0 = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.intermediate_size), T)
    mach = ref.Machine.from_program(elf.program(elf_path))
    BR.park(mach)
    regs_nz = sum(1 for v in mach.x if v)
    ram_nz = sum(1 for v in mach.ram if v)
    core.snapshot(c0, mach)
    c = RvChat.__new__(RvChat)
    c.model, c.lm, c.dev, c.executor = m, m.model, "cuda", True
    c.n_chip, c.v0, c.d0 = meta["chip_layers"], meta["chip_token_offset"], meta["chip_dims_offset"]
    cols = meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]]
    c.keep_cols = torch.tensor(cols, device="cuda")
    x = torch.tensor([ids], device="cuda")
    # the prefix, then the snapshot in the chip layers' states, then the text
    o = m(input_ids=x[:, :prefix], use_cache=True)
    cache = o.past_key_values
    hf.inject(c0, cache)
    saved = {}
    for L in range(c.n_chip, len(m.model.layers)):
        saved[L] = None
    import copy

    cache2 = copy.deepcopy(cache)
    got = {}
    h = c.lm.layers[c.n_chip - 1].register_forward_hook(
        lambda _m, _i, o_: got.__setitem__("hc", (o_[0] if isinstance(o_, tuple) else o_).detach()))
    full = m(input_ids=x[:, prefix:], past_key_values=cache, use_cache=True).logits[0, -1, : c.v0]
    h.remove()
    H = got["hc"][0]
    e = c.lm.embed_tokens(x[:, prefix:])[0]
    mask = torch.ones(H.shape[1], dtype=torch.bool, device="cuda")
    mask[: c.d0] = False
    mask[c.keep_cols] = False
    keep = H[:, c.keep_cols].float()
    res = {"checkpoint": path, "firmware": os.path.basename(elf_path), "prefix": prefix, "text_tokens": n,
           "machine_registers_nonzero": regs_nz, "machine_ram_words_nonzero": ram_nz,
           "hidden": H.shape[1], "chip_layers": c.n_chip, "memory_peak": torch.cuda.max_memory_allocated(),
           "embed_dims_max_error": float((H[:, : c.d0].float() - e[:, : c.d0].float()).abs().max()),
           "other_chip_dims_max": float(H[:, mask].float().abs().max()) if bool(mask.any()) else 0.0,
           "keep_abs_max": float(keep.abs().max()), "keep_varies_by_position": float(keep.std(0).max()),
           "positions_where_keep_differs_from_last": int((keep != keep[-1]).any(1).sum()),
           "code_nonzero_positions": int((keep[:, : len(meta["code_dims"])] != 0).any(1).sum())}

    def prefill(hcs):
        cc = copy.deepcopy(cache2)
        r = c._qwen_prefill(hcs, cc, prefix)
        return r[: c.v0]

    res["true_residual_vs_forward"] = compare(prefill(H), full)
    tx = ids[prefix:]
    c.fx_keep = None
    res["keep_zero_vs_forward"] = compare(prefill(c._text_hc(tx)), full)
    c.fx_keep = keep[-1]
    res["keep_last_vs_forward"] = compare(prefill(c._text_hc(tx)), full)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
