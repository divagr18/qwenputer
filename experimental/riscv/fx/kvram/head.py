"""One KV-RAM bank (one kv head) built through Qwen3.5's own q_norm / k_norm / RoPE modules.

Keys and queries are formed exactly as Qwen3_5Attention.forward forms them after the projections:
x (bf16, the projection output) -> Qwen3_5RMSNorm (w from encoding.Enc) -> apply_rotary_pos_emb
with Qwen3_5TextRotaryEmbedding's cos/sin at the token's position. The projections themselves are
exercised in hfcheck.py (full module).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
                                                          apply_rotary_pos_emb)

from encoding import E, Enc

QWEN_CFG = "C:/Users/kesha/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json"
EV = 256  # Qwen3.5 v head dim (flash needs Ev == E)
V_BITS, V_HIT, V_BYTES = 0, 32, 33


def text_config():
    if os.path.exists(QWEN_CFG):
        tc = json.load(open(QWEN_CFG))["text_config"]
    else:  # the fields that matter, as in Qwen3.5-4B's config.json
        tc = {"head_dim": 256, "hidden_size": 2560, "num_attention_heads": 16, "num_key_value_heads": 4,
              "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "attn_output_gate": True,
              "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "default",
                                  "rope_theta": 10000000, "partial_rotary_factor": 0.25}}
    return Qwen3_5TextConfig(**tc)


class Head:
    def __init__(self, enc: Enc, cfg=None):
        self.enc = enc
        self.cfg = cfg or text_config()
        self.q_norm = Qwen3_5RMSNorm(E, eps=self.cfg.rms_norm_eps).to(torch.bfloat16)
        self.k_norm = Qwen3_5RMSNorm(E, eps=self.cfg.rms_norm_eps).to(torch.bfloat16)
        qw, kw = enc.norm_weights()
        with torch.no_grad():
            self.q_norm.weight.copy_(qw)
            self.k_norm.weight.copy_(kw)
        self.rope = Qwen3_5TextRotaryEmbedding(self.cfg)

    @torch.no_grad()
    def _rot(self, x, pos):
        """x: bf16 [N, E] after its norm, pos int64 [N] -> RoPE'd bf16 [N, E] (q and k use the same path)."""
        cos, sin = self.rope(x[None], torch.as_tensor(pos, dtype=torch.long)[None])
        y, _ = apply_rotary_pos_emb(x[None, None], x[None, None], cos, sin)
        return y[0, 0]

    @torch.no_grad()
    def keys(self, addr, t, pos, epoch=0, chunk=1 << 15):
        N = len(addr)
        out = torch.empty(N, E, dtype=torch.bfloat16)
        for i in range(0, N, chunk):
            x = torch.from_numpy(self.enc.key_in(addr[i:i + chunk], t[i:i + chunk],
                                                 epoch if np.isscalar(epoch) else epoch[i:i + chunk])).to(torch.bfloat16)
            out[i:i + chunk] = self._rot(self.k_norm(x), pos[i:i + chunk])
        return out

    @torch.no_grad()
    def queries(self, addr, pos, epoch=0):
        x = torch.from_numpy(self.enc.query_in(addr, epoch)).to(torch.bfloat16)
        return self._rot(self.q_norm(x), pos)

    @staticmethod
    def values(val, chunk=1 << 15, width=EV):
        """bf16 [N, width]: 32 data bits, hit flag, 4 bytes (all exact in bf16), zeros after."""
        val = np.asarray(val, dtype=np.uint32)
        N = len(val)
        v = torch.zeros(N, width, dtype=torch.bfloat16)
        for i in range(0, N, chunk):
            x = val[i:i + chunk].astype(np.int64)
            bits = (x[:, None] >> np.arange(32)) & 1
            by = (x[:, None] >> (8 * np.arange(4))) & 255
            v[i:i + chunk, V_BITS:V_BITS + 32] = torch.from_numpy(bits.astype(np.float32)).to(torch.bfloat16)
            v[i:i + chunk, V_HIT] = 1
            v[i:i + chunk, V_BYTES:V_BYTES + 4] = torch.from_numpy(by.astype(np.float32)).to(torch.bfloat16)
        return v

    @staticmethod
    def decode(o):
        """o: [L, EV] attention output -> (hit bool, value uint32) by thresholds (bits at 0.5)."""
        of = o.float()
        hit = of[:, V_HIT] > 0.5
        bits = (of[:, V_BITS:V_BITS + 32] > 0.5).to(torch.int64)
        val = (bits << torch.arange(32)).sum(1)
        return hit.numpy(), val.numpy().astype(np.uint32)
