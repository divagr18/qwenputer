"""Qwen in its merged form as a DeltaLink backend, reading through the parked circuit
(docs/fx/one.md, chipformer/rv/wide.py).

Every token Qwen reads or writes also goes through the chip as model input, with fx parked. Text fed
in one chunk takes the kept dims the circuit ends that chunk on (`_text_hc`, the prefill path); a token
Qwen writes takes the kept dims of its own position (`_exec_hc`). A snapshot or a rollback is Qwen's
alone: the circuit has read the rolled-back text, which only touches the text bridge's own heads.
"""
from __future__ import annotations

import time

import qwen_hf as QH
import torch


class WideBackend(QH.HfBackend):
    """HfBackend's interface. The circuit has to exist before Qwen's merged tensors, so the model
    is built at the first feed, after the boot snapshot."""

    def __init__(self, chip, model_dir, chunk: int = 2048, **kw) -> None:
        self.chip, self.model_dir, self.chunk = chip, model_dir, chunk
        self.cpu_of = None
        self.c, self.m = None, None
        self.kw = kw
        self.temperature, self.top_k, self.top_p = kw["temperature"], kw["top_k"], kw["top_p"]
        self.dev = "cuda"
        self.gen = torch.Generator(device=self.dev).manual_seed(kw.get("seed", 0))
        self.seconds = {"feed": 0.0, "generate": 0.0, "snapshot": 0.0, "restore": 0.0}
        self.snapshot_bytes = 0
        self.reset()

    def reset(self) -> None:
        self.cache = None
        self.pos, self.logits, self.snaps = 0, None, {}

    @torch.no_grad()
    def _run(self, ids: list[int]) -> None:
        from transformers import DynamicCache

        if self.c is None:
            self.c = self.chip.wide()[0]
            self.m = self.c.model
        if self.cache is None:
            self.cache = DynamicCache(config=self.m.config)
        cpu = self.cpu_of()
        if len(ids) == 1:
            _, keep = cpu.text(ids)
            self.logits = self.c._qwen_step(self.c._exec_hc(ids[0], keep), self.cache, self.pos)
            self.pos += 1
            return
        for a in range(0, len(ids), self.chunk):
            part = ids[a:a + self.chunk]
            _, keep = cpu.text(part)
            self.c.fx_keep = keep
            self.logits = self.c._qwen_prefill(self.c._text_hc(part), self.cache, self.pos)
            self.pos += len(part)

    def feed(self, ids: list[int]) -> None:
        if ids:
            t0 = time.perf_counter()
            self._run(ids)
            self.seconds["feed"] += time.perf_counter() - t0
