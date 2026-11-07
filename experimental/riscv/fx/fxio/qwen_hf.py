"""Qwen3.5 in HF transformers as a DeltaLink backend: one cache that only ever grows by what is
fed, with snapshots where generation starts. DeltaNet states cannot be rolled back, so going
back means restoring a snapshot. None is evicted: the oldest are the ones a new turn needs.

A snapshot is a host-side copy of the state, what a prefix cache is to a model server.
"""
from __future__ import annotations

import copy
import time

import torch


def size_of(obj, seen=None) -> int:
    """Bytes of the tensors an object holds, each counted once."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(size_of(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(size_of(v, seen) for v in obj)
    return sum(size_of(v, seen) for v in vars(obj).values()) if hasattr(obj, "__dict__") else 0


class HfBackend:
    def __init__(self, model, device=None, temperature=0.0, top_k=0, top_p=1.0, seed=0) -> None:
        self.m = model.eval()
        self.dev = device or next(model.parameters()).device
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.seconds = {"feed": 0.0, "generate": 0.0, "snapshot": 0.0, "restore": 0.0}
        self.snapshot_bytes = 0
        self.reset()

    @classmethod
    def load(cls, name, dtype=torch.bfloat16, device="cuda", temperature=None, top_k=None, top_p=None, **kw):
        """Sampling left as None comes from the model's generation_config, as vLLM's default."""
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        g = m.generation_config
        if temperature is None:
            temperature = (g.temperature or 0.0) if g.do_sample else 0.0
        return cls(m, device, temperature=temperature, top_k=(g.top_k or 0) if top_k is None else top_k,
                   top_p=(g.top_p or 1.0) if top_p is None else top_p, **kw)

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
        t0 = time.perf_counter()
        self.snaps[n] = (copy.deepcopy(self.cache), self.logits.clone())
        self.seconds["snapshot"] += time.perf_counter() - t0
        self.snapshot_bytes = size_of(self.cache)

    def restore(self, n: int) -> None:
        t0 = time.perf_counter()
        cache, logits = self.snaps[n]
        self.cache, self.logits, self.pos = copy.deepcopy(cache), logits.clone(), n
        self.seconds["restore"] += time.perf_counter() - t0
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
