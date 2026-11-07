"""Qwen3.5 in HF transformers as a DeltaLink backend: one cache that only ever grows by what is
fed, with snapshots where generation starts. DeltaNet states cannot be rolled back, so going
back means restoring a snapshot. None is evicted: the oldest are the ones a new turn needs.

A snapshot is a host-side copy of the state, what a prefix cache is to a model server.
"""
from __future__ import annotations

import copy
import time

import torch


class HfBackend:
    def __init__(self, model, device=None, temperature=0.0, top_k=0, top_p=1.0, seed=0) -> None:
        self.m = model.eval()
        self.dev = device or next(model.parameters()).device
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.seconds = {"feed": 0.0, "generate": 0.0}
        self.reset()

    @classmethod
    def load(cls, name, dtype=torch.bfloat16, device="cuda", **kw):
        from transformers import AutoModelForCausalLM

        return cls(AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device), device, **kw)

    def reset(self) -> None:
        self.cache = None
        self.pos = 0
        self.logits = None
        self.snaps: dict[int, tuple] = {}

    @torch.no_grad()
    def _run(self, ids: list[int]) -> None:
        x = torch.tensor([ids], device=self.dev)
        out = self.m(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.logits = out.logits[0, -1]
        self.pos += len(ids)

    def feed(self, ids: list[int]) -> None:
        if not ids:
            return
        t0 = time.perf_counter()
        self._run(ids)
        self.seconds["feed"] += time.perf_counter() - t0

    def snapshot(self, n: int) -> None:
        assert n == self.pos
        self.snaps[n] = (copy.deepcopy(self.cache), self.logits.clone())

    def restore(self, n: int) -> None:
        cache, logits = self.snaps[n]
        self.cache, self.logits, self.pos = copy.deepcopy(cache), logits.clone(), n
        for k in [k for k in self.snaps if k > n]:
            del self.snaps[k]

    def _pick(self, logits: torch.Tensor) -> int:
        if self.temperature <= 0:
            return int(torch.argmax(logits))
        p = logits.float() / self.temperature
        if self.top_k:
            p[p < torch.topk(p, min(self.top_k, p.numel())).values[-1]] = -torch.inf
        if self.top_p < 1.0:
            v, i = torch.sort(p, descending=True)
            c = torch.softmax(v, -1).cumsum(-1)
            drop = c > self.top_p
            drop[1:] = drop[:-1].clone()
            drop[0] = False
            p[i[drop]] = -torch.inf
        return int(torch.multinomial(torch.softmax(p, -1), 1, generator=self.gen))

    def generate(self, max_new: int, stop_ids) -> list[int]:
        t0 = time.perf_counter()
        out: list[int] = []
        while len(out) < max_new:
            t = self._pick(self.logits)
            out.append(t)
            if t in stop_ids:
                break
            self._run([t])
        self.seconds["generate"] += time.perf_counter() - t0
        return out
