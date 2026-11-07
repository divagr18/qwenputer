"""Quantisation through stock transformers + torchao, on the small merged model (CPU).

    PYTHONPATH=. python fx/size/proto_quant.py OUTDIR

proto.py's P3 reconstructs quantised weights by hand. Here the real path: save the merged model,
from_pretrained with a TorchAoConfig that names Qwen's layers only, and check that the chip does not
move. Also: what the mute column does to a row quantised with a per-row scale.
"""
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto as P
from qstream import quant

DT = torch.bfloat16


def main():
    from safetensors.torch import save_file
    from torchao.quantization import FqnToConfig, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis, PerGroup
    from transformers import Qwen3_5ForCausalLM, TorchAoConfig

    out = sys.argv[1]
    d = out + "/merged"
    os.makedirs(d, exist_ok=True)
    c = P.circuit()
    n_chip = c.n_layers()
    qwen = P.random_qwen()
    cfg, sd, lay, O, _ = P.build(c, False, qwen)
    ids = [5, 7, 300, 11, 260, 1030, 1044, 1100, 12, 301, 1500, 300, 77, 310, 1033, 256, 200, 511]
    chip = [i for i, t in enumerate(ids) if t < P.TEXT]
    txt = [i for i, t in enumerate(ids) if t >= P.TEXT]
    dense = P.model_from(cfg, {k: v.clone() for k, v in sd.items()})
    hd, ld = P.run(dense, ids, n_chip)
    save_file({k: v.to(DT).contiguous() for k, v in sd.items()}, d + "/model.safetensors", metadata={"format": "pt"})
    cfg.save_pretrained(d)
    res = {"torch_ao": {}}

    # the mute column in a per-row int8: base-sized weights next to one large one
    W = sd[f"model.layers.{n_chip}.linear_attn.in_proj_z.weight"].float()
    q = quant(W, "i8r").float()
    qcols = W[:, :P.D0]
    res["mute_col_per_row_int8"] = dict(
        mute_weight=float(W[0, lay.cols[next(iter(c.mu_e.terms))]]), qwen_rms=float(qcols.pow(2).mean().sqrt()),
        qwen_weights_rounded_to_zero=float((q[:, :P.D0] == 0).float().mean()),
        row_rel_error=float((q[:, :P.D0] - qcols).norm() / qcols.norm()))
    print(res["mute_col_per_row_int8"], flush=True)

    names = r"(linear_attn\.(in_proj_qkv|out_proj)|mlp\.(gate_proj|up_proj|down_proj))"
    layers = "|".join(str(i) for i in range(n_chip, n_chip + 2))
    for tag, q_cfg in (("int8_per_row", IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0))),
                       ("int4_group32", IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)))):
        try:
            fq = FqnToConfig({f"re:model\\.layers\\.({layers})\\.{names}": q_cfg})
            t0 = time.time()
            m = Qwen3_5ForCausalLM.from_pretrained(d, dtype=DT, quantization_config=TorchAoConfig(quant_type=fq)).eval()
            kinds = {}
            for k, p in m.named_parameters():
                kinds.setdefault(type(p.data).__name__ if type(p).__name__ == "Parameter" else type(p).__name__, 0)
                kinds[type(p.data).__name__ if type(p).__name__ == "Parameter" else type(p).__name__] += 1
            hq, lq = P.run(m, ids, n_chip)
            sv = f"{out}/saved_{tag}"
            m.save_pretrained(sv)
            size = sum(os.path.getsize(f"{sv}/{f}") for f in os.listdir(sv) if f.endswith(".safetensors"))
            t1 = time.perf_counter()
            for _ in range(5):
                P.run(m, ids, n_chip)
            t_q = time.perf_counter() - t1
            t1 = time.perf_counter()
            for _ in range(5):
                P.run(dense, ids, n_chip)
            t_d = time.perf_counter() - t1
            res["torch_ao"][tag] = dict(ok=True, param_types=kinds, chip_residual_identical=P.same(hq, hd),
                                        chip_logits_identical=P.same(lq[chip], ld[chip]),
                                        text_logits_changed=not P.same(lq[txt], ld[txt]),
                                        safetensors_bytes=size, dense_bytes=sum(v.numel() * 2 for v in sd.values()),
                                        time_ratio=round(t_q / t_d, 2))
        except Exception as e:  # noqa: BLE001 - a failure here is the result
            res["torch_ao"][tag] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        print(tag, res["torch_ao"][tag], flush=True)
    json.dump(res, open(out + "/proto_quant.json", "w"), indent=1)


if __name__ == "__main__":
    main()
