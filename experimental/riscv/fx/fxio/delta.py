"""The model half of the fx packet path (docs/fx/io.md): fx's stateless model request to the
tokens Qwen does not have yet, and Qwen's completion back to the bytes fx reads.

    request bytes -> messages + tools -> Qwen3.5's chat template -> ids
    ids against what Qwen already holds -> only the new suffix is fed
    generated text -> reasoning, content, tool calls (vLLM 0.33's qwen3 engine parser, ported)
    -> one read of SSE (AI Gateway or OpenAI Chat Completions, whichever route fx used)

Who runs this: the host, today. Rendering, BPE and parsing are a model server's work and are
disclosed as host work; docs/fx/io.md gives what an in-weights version would cost.

No torch here; a backend (qwen_hf.py, or scripted.py) holds the model state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

STOP_TEXT = ("<|im_end|>", "<|endoftext|>")


class TemplateRaised(ValueError):
    """The chat template called raise_exception (e.g. a system message that is not first)."""


class QwenTemplate:
    """Qwen3.5's chat_template.jinja in the same jinja2 setup as transformers'
    apply_chat_template, and its tokenizer.json. tests/test_fx_io.py checks both against
    AutoTokenizer."""

    def __init__(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
            json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(msg):
            raise TemplateRaised(msg)

        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string((d / "chat_template.jinja").read_text(encoding="utf-8"))
        self.stop_ids = {self.tok.token_to_id(s) for s in STOP_TEXT}

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True) -> str:
        try:
            return self.template.render(messages=messages, tools=tools or None,
                                        add_generation_prompt=add_generation_prompt,
                                        enable_thinking=enable_thinking)
        except TemplateError as e:
            raise TemplateRaised(str(e)) from e

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ---- fx's request -> messages ---------------------------------------------------------------

def _text_of(c) -> str:
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def _merge_system(msgs: list[dict]) -> list[dict]:
    """As p0c/proxy.py: every system message merged into one at the start, text unchanged."""
    parts = [_text_of(m.get("content", "")) for m in msgs if m["role"] in ("system", "developer")]
    rest = [m for m in msgs if m["role"] not in ("system", "developer")]
    return ([{"role": "system", "content": "\n\n".join(parts)}] if parts else []) + rest


def _tool_output(out) -> str:
    if isinstance(out, str):
        return out
    kind = out.get("type")
    if kind in ("text", "error-text"):
        return out.get("value", "")
    if kind in ("json", "error-json"):
        return json.dumps(out.get("value"), ensure_ascii=False)
    if kind == "content":
        return "".join(p.get("text", "") for p in out.get("value", []))
    return json.dumps(out, ensure_ascii=False)


def _from_gateway(body: dict) -> tuple[list[dict], list[dict]]:
    msgs: list[dict] = []
    for m in body.get("prompt", []):
        role, content = m["role"], m.get("content")
        if role in ("system", "developer"):
            msgs.append({"role": "system", "content": _text_of(content)})
        elif role == "user":
            msgs.append({"role": "user", "content": _text_of(content)})
        elif role == "assistant":
            out: dict = {"role": "assistant", "content": ""}
            reasoning, calls = "", []
            for p in content if isinstance(content, list) else [{"type": "text", "text": content or ""}]:
                if p["type"] == "text":
                    out["content"] += p["text"]
                elif p["type"] == "reasoning":
                    reasoning += p["text"]
                elif p["type"] == "tool-call":
                    args = p.get("input", {})
                    calls.append({"type": "function", "id": p.get("toolCallId"), "function": {
                        "name": p["toolName"], "arguments": json.loads(args) if isinstance(args, str) else args}})
            if reasoning:
                out["reasoning_content"] = reasoning
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
        elif role == "tool":
            for p in content:
                msgs.append({"role": "tool", "tool_call_id": p.get("toolCallId"),
                             "content": _tool_output(p.get("output", ""))})
        else:
            raise ValueError(f"unexpected role {role}")
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("inputSchema", {})}}
             for t in body.get("tools", [])]
    return msgs, tools


def _from_chat(body: dict) -> tuple[list[dict], list[dict]]:
    msgs = []
    for m in body.get("messages", []):
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c["function"])
                if isinstance(f.get("arguments"), str):
                    f["arguments"] = json.loads(f["arguments"] or "{}")
                calls.append({**c, "function": f})
            m["tool_calls"] = calls
        if m.get("role") == "assistant" and "reasoning" in m and "reasoning_content" not in m:
            m["reasoning_content"] = m.pop("reasoning")
        if m.get("content") is None:
            m["content"] = ""
        msgs.append(m)
    return msgs, list(body.get("tools", []))


def request_messages(body: dict) -> tuple[str, list[dict], list[dict]]:
    """(protocol, messages, tools) of a request body: "gateway" or "chat"."""
    if "prompt" in body:
        proto, (msgs, tools) = "gateway", _from_gateway(body)
    else:
        proto, (msgs, tools) = "chat", _from_chat(body)
    return proto, _merge_system(msgs), tools


# ---- Qwen's text -> reasoning, content, tool calls -------------------------------------------
# A port of vLLM 0.33's Python engine for `--tool-call-parser qwen3_coder --reasoning-parser qwen3`
# (vllm/parser/qwen3.py's transitions, ParserEngine._single_pass_parse, _coerce_value). The engine
# also matches the four wrapper tokens by id; this works on the decoded text. The difference
# only shows if Qwen writes "</think>" as ordinary tokens instead of the special token.

_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
                       re.DOTALL)


def _trim_wrapping_newlines(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _coerce(value: str, schema: dict):
    """String values to the schema's type; anything that does not convert stays a string."""
    types = schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    if not types or "string" in types:
        return value
    v = value.strip()
    for t in types:
        try:
            if t == "integer":
                return int(v)
            if t == "number":
                return float(v) if not re.fullmatch(r"[+-]?\d+", v) else int(v)
            if t == "boolean" and v.lower() in ("true", "false"):
                return v.lower() == "true"
            if t == "null" and v.lower() in ("null", "none"):
                return None
            if t in ("object", "array"):
                r = json.loads(v)
                if isinstance(r, dict if t == "object" else list):
                    return r
        except ValueError:
            continue
    return value


def _args(raw: str, name: str, tools: list[dict]) -> dict:
    props = {}
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            props = (f.get("parameters") or {}).get("properties", {})
    out: dict = {}
    for m in _PARAM_RE.finditer(raw):
        k, v = m.group(1), _trim_wrapping_newlines(m.group(2))
        out[k] = _coerce(v, props[k]) if k in props else v
    return out


@dataclass
class Completion:
    reasoning: str = ""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "arguments": JSON text}
    warnings: list[str] = field(default_factory=list)

    @property
    def finish(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


# (state, terminal) -> (next state, what happens to the open call)
_TERMS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>", ">")
_GRAMMAR = {
    ("reasoning", "<think>"): ("reasoning", None),
    ("reasoning", "</think>"): ("content", None),
    ("content", "</think>"): ("content", None),
    ("reasoning", "<tool_call>"): ("preamble", "start"),
    ("content", "<tool_call>"): ("preamble", "start"),
    ("content", "<function="): ("name", "start"),
    ("preamble", "</tool_call>"): ("content", "end"),
    ("preamble", "<function="): ("name", None),
    ("name", ">"): ("args", None),
    ("name", "</function>"): ("between", "end"),
    ("name", "</tool_call>"): ("content", "end"),
    ("name", "<tool_call>"): ("preamble", "start"),
    ("name", "<function="): ("name", "start"),
    ("args", "</function>"): ("between", "end"),
    ("args", "<function="): ("name", "start"),
    ("between", "</tool_call>"): ("content", None),
    ("between", "<tool_call>"): ("preamble", "start"),
    ("between", "<function="): ("name", "start"),
}


def parse_completion(text: str, tools: list[dict] | None = None, thinking: bool = True) -> Completion:
    c = Completion()
    state = "reasoning" if thinking else "content"
    i = 0
    slots: list[dict] = []
    cur: dict | None = None
    while i < len(text):
        best, term = None, None
        for t in _TERMS:
            if (state, t) in _GRAMMAR:
                j = text.find(t, i)
                if j >= 0 and (best is None or j < best):
                    best, term = j, t
        end = len(text) if best is None else best
        chunk = text[i:end]
        if state == "reasoning":
            c.reasoning += chunk
        elif state == "content":
            c.content += chunk
        elif state == "name" and cur is not None:
            cur["name"] += chunk
        elif state == "args" and cur is not None:
            cur["raw"] += chunk
        if best is None:
            break
        i = best + len(term)
        state, act = _GRAMMAR[(state, term)]
        if act == "start":
            cur = {"name": "", "raw": ""}
            slots.append(cur)
        elif act == "end":
            cur = None
    known = {t.get("function", t).get("name") for t in tools or []}
    for s in slots:
        name = s["name"].strip()
        if not name or (known and name not in known):
            c.warnings.append(f"tool call dropped: {name!r}")
            continue
        c.tool_calls.append({"name": name,
                             "arguments": json.dumps(_args(s["raw"], name, tools or []), ensure_ascii=False)})
    if c.tool_calls and not c.content.strip():
        c.content = ""
    return c


# ---- the reply fx reads -----------------------------------------------------------------------

def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n"


def sse_reply(proto: str, c: Completion, idx: int, prompt_tokens: int, completion_tokens: int) -> list[bytes]:
    """The whole reply in one read (speed-iter rows 3-4, speedups §1)."""
    if proto == "gateway":
        ev: list[bytes] = []
        if c.reasoning.strip():
            ev += [_sse({"type": "reasoning-start", "id": "r0"}),
                   _sse({"type": "reasoning-delta", "id": "r0", "delta": c.reasoning}),
                   _sse({"type": "reasoning-end", "id": "r0"})]
        if c.content:
            ev += [_sse({"type": "text-start", "id": "t0"}),
                   _sse({"type": "text-delta", "id": "t0", "delta": c.content}),
                   _sse({"type": "text-end", "id": "t0"})]
        for k, t in enumerate(c.tool_calls):
            ev.append(_sse({"type": "tool-call", "toolCallId": f"call_{idx}_{k}", "toolName": t["name"],
                            "input": t["arguments"]}))
        reason = "tool-calls" if c.tool_calls else "stop"
        ev.append(_sse({"type": "finish", "finishReason": {"unified": reason, "raw": reason},
                        "usage": {"inputTokens": {"total": prompt_tokens}, "outputTokens": {"total": completion_tokens}}}))
        return [b"".join(ev)]
    delta: dict = {"role": "assistant", "content": c.content or None}
    if c.reasoning.strip():
        delta["reasoning_content"] = c.reasoning
    if c.tool_calls:
        delta["tool_calls"] = [{"index": k, "id": f"call_{idx}_{k}", "type": "function",
                                "function": {"name": t["name"], "arguments": t["arguments"]}}
                               for k, t in enumerate(c.tool_calls)]
    ev1 = _sse({"id": f"chatcmpl-{idx}", "model": "qwen3.5-4b",
                "choices": [{"index": 0, "delta": delta, "finish_reason": c.finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})
    return [ev1 + b"data: [DONE]\n\n"]


# ---- the delta --------------------------------------------------------------------------------

def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class DeltaLink:
    """One conversation's Qwen context as fx's requests extend it.

    `ctx` is every id the backend has run, in order. A request is rendered in full and compared
    with ctx id by id:

    - exact: the render starts with ctx. Only the rest is fed.
    - rollback: it does not. The backend goes back to the longest snapshot that is a prefix of
      the render (snapshots are taken where generation starts) and the rest is fed from there.
    - rebuild: no snapshot fits: the state is reset and the whole render is fed.

    So after feed Qwen's state is the state a stateless server would build from this request,
    up to chunking's float rounding. The check is on ids because identical text can tokenize
    differently from what Qwen generated.
    """

    def __init__(self, template: QwenTemplate, backend, thinking: bool = True, max_new: int = 8192) -> None:
        self.t, self.b, self.thinking, self.max_new = template, backend, thinking, max_new
        self.ctx: list[int] = []
        self.snaps: list[int] = []  # context lengths with a snapshot
        self.records: list[dict] = []

    def _align(self, ids: list[int]) -> tuple[str, int]:
        """Moves the backend to the longest usable prefix of ids; returns (kind, that length)."""
        if not self.ctx:
            return "first", 0
        lcp = common_prefix(self.ctx, ids)
        if lcp == len(self.ctx):
            return "exact", lcp
        usable = [s for s in self.snaps if s <= lcp]
        if usable:
            s = max(usable)
            self.b.restore(s)
            self.snaps = [x for x in self.snaps if x <= s]
            self.ctx = self.ctx[:s]
            return "rollback", s
        self.b.reset()
        self.snaps, self.ctx = [], []
        return "rebuild", 0

    def call(self, raw: bytes) -> tuple[list[bytes], dict]:
        """One model call: request bytes in, the bytes of one SSE read out."""
        proto, msgs, tools = request_messages(json.loads(raw))
        text = self.t.render(msgs, tools, enable_thinking=self.thinking)
        ids = self.t.encode(text)
        had = len(self.ctx)
        ctx_text = self.t.decode(self.ctx) if self.ctx else ""
        kind, keep = self._align(ids)
        suffix = ids[keep:]
        self.b.feed(suffix)
        self.ctx = list(ids)
        if not self.snaps or self.snaps[-1] != len(self.ctx):
            self.b.snapshot(len(self.ctx))
            self.snaps.append(len(self.ctx))
        gen = self.b.generate(self.max_new, self.t.stop_ids)
        stopped = bool(gen) and gen[-1] in self.t.stop_ids
        body_ids = gen[:-1] if stopped else gen
        out = self.t.decode(body_ids)
        self.ctx = self.ctx + body_ids  # the stop token is sampled, never run
        c = parse_completion(out, tools, self.thinking)
        idx = len(self.records)
        chunks = sse_reply(proto, c, idx, len(ids), len(body_ids))
        rec = {"call": idx, "proto": proto, "kind": kind, "text_extends": text.startswith(ctx_text),
               "ctx_before": had, "kept": keep, "rolled_back": had - keep,
               "full_tokens": len(ids), "full_bytes": len(text.encode()),
               "delta_tokens": len(suffix), "delta_bytes": len(self.t.decode(suffix).encode()),
               "generated_tokens": len(body_ids), "stopped": stopped, "request_bytes": len(raw),
               "reply_bytes": sum(map(len, chunks)), "tool_calls": [t["name"] for t in c.tool_calls],
               "finish": c.finish, "warnings": c.warnings, "n_messages": len(msgs)}
        self.records.append(rec)
        return chunks, rec
