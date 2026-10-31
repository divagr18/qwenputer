"""Merge the RV32 machine and its text bridge into Qwen3.5-4B (layout "rv1").

    python -m chipformer.rv.merge SRC --out dist/q4-rv1 [--no-mute]

The result is a text-only stock ``Qwen3_5ForCausalLM``:

- the chip layers come first (the circuit of ``core.build(text=...)``), then Qwen's layers,
  zero-padded to the new hidden size (q35.merge.widen's rules: Qwen never reads or writes chip
  dims; the full-residual norms are rescaled so its normalized inputs stay the same);
- every Qwen vocabulary row carries the ballast and its text features in the chip dims (is_text,
  tau, its first 16 UTF-8 bytes, the ``<tool_call>`` / ``</tool_call>`` flags, and for the 260
  printable tokens the output code of their echo); the RV32 tokens are appended after Qwen's
  vocabulary as special tokens;
- the tied output head reads only the output code and tau among the chip dims (q35.merge's
  final-norm gains), and Qwen's layers are muted on chip tokens (q35.mute, on mu).

``chipformer.json`` records the layout for the decoder (rv/chat.py).
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from chipformer.q35.merge import (IM_START, IN_COLS, OUT_ROWS, TAU_GAIN, WIDE_NORMS,
                                  pick_hidden)
from chipformer.rv import bridge as BR
from chipformer.rv import core
from chipformer.rv import spec as S

SRC_TEXT = "model.language_model."
DST = "model."


def chip_token_names() -> list[str]:
    names = []
    for t in range(S.VOCAB):
        fam, p = S.family(t)
        names.append(f"<|rv:{fam}|>" if fam in S.CONTROLS else f"<|rv:{fam}:{p}|>")
    return names


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    import shutil

    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in chip_token_names()]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(chip_token_names()[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    # tokenizer.json holds every added token; listing them in the config too makes it tens of
    # MB, too big for the hubs to parse
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tc.pop("added_tokens_decoder", None)
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def text_rows(c, lay, T: BR.Text, base_emb: torch.Tensor, v0: int) -> torch.Tensor:
    """Qwen's rows: the Qwen part, plus the chip features of each text token (a few special
    rows exactly, every other row = the plain text row plus its byte bits)."""
    from chipformer.q35.emit import embedding_rows

    h = lay.hidden
    d0 = base_emb.shape[1]
    full = torch.zeros(v0, h, dtype=torch.bfloat16)
    full[:, :d0] = base_emb[:v0, :d0]
    special = sorted({T.TCO, T.TCC, *T.printable})
    q_plain = next(q for q in range(T.n) if q not in set(special))
    plain = embedding_rows(c, lay, [BR.text_token(q_plain)], h)[0]  # (bytes zeroed below)
    byte_cols = [[lay.dim(next(iter(c.tb[k][i].terms))) for i in range(8)]
                 for k in range(S.TEXT_ROW)]
    for k in range(S.TEXT_ROW):
        for i in range(8):
            plain[byte_cols[k][i]] = 0.0
    full[:, d0:] = plain[d0:].to(torch.bfloat16)
    by = np.zeros((v0, S.TEXT_ROW), dtype=np.uint8)
    for q in range(min(v0, T.n)):
        b = T.row_bytes(q)
        by[q] = np.frombuffer(b, dtype=np.uint8)
    for k in range(S.TEXT_ROW):
        for i in range(8):
            full[:, byte_cols[k][i]] = torch.tensor((by[:, k] >> i) & 1, dtype=torch.bfloat16)
    rows = embedding_rows(c, lay, [BR.text_token(q) for q in special], h)
    for q, r in zip(special, rows):
        full[q, d0:] = r[d0:].to(torch.bfloat16)
    return full


def build_merged(src, dst, mute_qwen: bool = True, hidden: int | None = None,
                 m: int | None = None, vx: bool = False) -> dict:
    from safetensors.torch import save_file

    from chipformer.q35.emit import Layout, allocate, check_bf16, emit_layers
    from chipformer.q35.ir import Target
    from chipformer.q35.tiny import code_gain
    from chipformer.rv.tiny import embedding_matrix

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    tc0 = cfg["text_config"]
    d0, v0, inter = tc0["hidden_size"], tc0["vocab_size"], tc0["intermediate_size"]
    T = BR.Text(src / "tokenizer.json")
    c = core.build(Target(tc0["linear_num_key_heads"], tc0["linear_num_value_heads"], inter), T, vx=vx)
    keep = set(c.keep)
    alloc = allocate(c, None, keep, inter)
    if hidden is None:
        hidden, m = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    n0 = tc0["num_hidden_layers"]
    print(f"circuit {c.stats()}, slots {alloc.n_slots}, hidden {hidden} (m={m}), "
          f"{n_chip} chip layers", flush=True)

    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0, mtp_num_hidden_layers=0,
              layer_types=["linear_attention"] * n_chip + tc0["layer_types"],
              vocab_size=-(-(v0 + S.VOCAB) // 64) * 64, tie_word_embeddings=True,
              architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")
    vocab = tc["vocab_size"]

    sinks = None
    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sinks = fit_anti_sink(src, dev)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Qwen's weights, widened and renamed one tensor at a time (keeps peak memory low)
    scale = (d0 / hidden) ** 0.5
    sd = {}
    base_emb = None
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if not k.startswith(SRC_TEXT):
                    continue  # vision tower, mtp
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
                k2 = DST + k[len(SRC_TEXT):]
                k2 = re.sub(r"^(model\.layers\.)(\d+)\.", lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.",
                            k2)
                sd[k2] = w
    assert base_emb is not None
    if mute_qwen:
        from chipformer.q35.mute import mute

        mute(sd, tc, n0, lay.dim(next(iter(c.mu.terms))), hidden**0.5, prefix=DST, sinks=sinks,
             first=n_chip)

    # the chip layers (fp32 while emitted, bf16 stored)
    layers = emit_layers(c, lay, tc, 0, DST, alloc, keep)
    for k in list(layers):
        sd[k] = layers.pop(k).to(torch.bfloat16)

    # embeddings: Qwen's rows, then the RV32 tokens
    emb = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    emb[:v0] = text_rows(c, lay, T, base_emb, v0)
    del base_emb
    chip = embedding_matrix(c, lay, S.VOCAB)
    emb[v0:v0 + S.VOCAB] = chip[:S.VOCAB].to(torch.bfloat16)
    del chip
    check_bf16("embedding", emb[:, d0:].float()[: min(vocab, 300000)])
    sd[DST + "embed_tokens.weight"] = emb

    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    norm = sd[DST + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[DST + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shard, size, shards = {}, 0, []
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())},
             "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(tc, indent=2))
    extend_tokenizer(src, dst, v0)
    gen = {"eos_token_id": [248046, tc0.get("eos_token_id", 248044)],
           "pad_token_id": tc0.get("eos_token_id", 248044), "bad_words_ids": [[IM_START]],
           "do_sample": False}
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2))
    meta = {"layout": "rv1", "chip_token_offset": v0, "chip_vocab": S.VOCAB,
            "chip_layers": n_chip, "qwen_layers_from": n_chip, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": mute_qwen,
            "code_dims": [lay.dim(d) for d in code], "tau_dim": lay.dim(tau),
            "mu_dim": lay.dim(next(iter(c.mu.terms))), "nfam": core.NFAM, "npay": core.NPAY,
            "tool_call": [T.TCO, T.TCC], "printable": T.printable, "vx": vx}
    (dst / "chipformer.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge the RV32 machine into Qwen3.5 (layout rv1)")
    ap.add_argument("src")
    ap.add_argument("--out", default="dist/q4-rv1")
    ap.add_argument("--no-mute", action="store_true")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--ballast", type=int)
    ap.add_argument("--vx", action="store_true", help="with the VX vector unit (QwenPU)")
    args = ap.parse_args()
    meta = build_merged(args.src, args.out, not args.no_mute, args.hidden, args.ballast, args.vx)
    print(json.dumps({k: meta[k] for k in ("hidden", "chip_layers", "slots", "circuit")}))


if __name__ == "__main__":
    main()
