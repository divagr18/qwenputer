"""rv-tiny: the RV32 core as a standalone stock Qwen3_5ForCausalLM.

    python -m chipformer.rv.tiny --out dist/rv-tiny

Qwen3.5-4B's DeltaNet head shapes (16 key heads, 32 value heads, 128 dims) and MLP width (9216),
the circuit's layers, and one zero full-attention layer at the end (HF's cache counts positions
there). The vocabulary is the chip tokens (spec.py) plus PAD, a zero row: a run starts after a
one-token PAD prefix, with the program's snapshot written into the cache (firmware.py). The
output head is tied to the embedding: a row's code dims hold CODE_AMP * (2 * bit - 1) of its
token's code, and after the exit the residual holds the predicted code, so the argmax is the
token whose code bits all agree.
"""

import json
from pathlib import Path

import numpy as np
import torch

from chipformer.q35.emit import check_bf16, emit_layers
from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
from chipformer.rv import core
from chipformer.rv import spec as S


def embedding_matrix(c, lay, n_vocab: int) -> torch.Tensor:
    """Rows of every chip token (fast: a token's features are linear in its payload bits within
    its family), the PAD row (zero) and padding rows (zero)."""
    feats = sorted(c.token_fns)
    cols = [lay.dim(d) for d in feats]

    def vec(t):
        return np.array([float(c.feature_value(d, t)) for d in feats])

    e = np.zeros((n_vocab, lay.hidden), dtype=np.float32)
    for name, (base, size) in {**S.FAMILIES, **S.STORE_FAMILIES}.items():
        v0 = vec(base)
        nb = (size - 1).bit_length()
        deltas = np.stack([vec(base + (1 << i)) - v0 for i in range(nb)]) if nb else None
        p = np.arange(size)
        rows = np.tile(v0, (size, 1))
        if size <= 4096:  # small families: any features (e.g. QS / QD's digit index)
            rows = np.stack([vec(base + i) for i in range(size)])
        elif nb:
            bits = (p[:, None] >> np.arange(nb)[None, :]) & 1
            rows = rows + bits @ deltas
        e[base:base + size][:, cols] = rows
        e[base:base + size, lay.off:lay.off + lay.m] = lay.ballast
        for t in (base + size - 1, base + size // 3):  # spot checks of the linearity
            assert (vec(t) == rows[t - base]).all(), (name, t)
    for name, t in S.CONTROLS.items():
        e[t][cols] = vec(t)
        e[t, lay.off:lay.off + lay.m] = lay.ballast
    e = torch.tensor(e)
    check_bf16("embedding", e)
    return e


def build(c=None, dtype=torch.bfloat16):
    """(model, circuit): a Qwen3_5ForCausalLM; token id t is chip token t, id S.VOCAB is PAD."""
    from transformers import GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    c = c or core.build()
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(S.VOCAB + 1) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate, eos_token_id=S.tok("HALT"),
              pad_token_id=S.VOCAB, bos_token_id=S.VOCAB)
    sd = emit_layers(c, lay, tc, 0, "model.", None, set(c.keep))
    sd["model.embed_tokens.weight"] = embedding_matrix(c, lay, n_vocab)
    norm = torch.full((lay.hidden,), -1.0)  # (1 + w) = 0: the output head sees only the code
    for d in c.keep:
        norm[lay.dim(d)] = CODE_GAIN
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    for p in model.parameters():
        p.data.zero_()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(k.startswith(f"model.layers.{n_chip}.") or k == "lm_head.weight" for k in missing)
    model.tie_weights()
    model.generation_config = GenerationConfig(eos_token_id=S.tok("HALT"), pad_token_id=S.VOCAB,
                                               do_sample=False)
    model.layout = lay
    return model.to(dtype).eval(), c


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Save the RV32 core as a standalone checkpoint")
    ap.add_argument("--out", default="dist/rv-tiny")
    args = ap.parse_args()
    model, c = build()
    out = Path(args.out)
    model.save_pretrained(out)
    lay = model.layout
    meta = {"machine": "chipformer rv32 (RV32IM minus DIV)", "vocab": S.VOCAB, "pad": S.VOCAB,
            "families": {k: list(v) for k, v in S.FAMILIES.items()}, "controls": S.CONTROLS,
            "hidden": lay.hidden, "ballast_dims": lay.m, "chip_layers": c.n_layers(),
            "circuit": c.stats(), "rom_base": S.ROM_BASE, "rom_words": S.ROM_WORDS,
            "ram_base": S.RAM_BASE, "ram_words": S.RAM_WORDS}
    (out / "chipformer.json").write_text(json.dumps(meta, indent=1))
    n = sum(p.numel() for p in model.parameters())
    print(f"{out}: {c.stats()}, hidden {lay.hidden}, {n / 1e6:.0f}M parameters")


if __name__ == "__main__":
    main()
