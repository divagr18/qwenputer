"""Runs fx-core.wasm under wasmtime with fuel metering and a Python host.

The host mirrors libfx's wasm-runtime.js (WASI with no preopens, the `fx` imports)
and engine.js (ACP over stdin/stdout, host tools answered as libfx/tool_call).
The model is a mock OpenAI Chat Completions server reached through
fx_http_stream_open (FX_CONFIGURED_PROVIDERS, as libfx's customProvider sets it):
it streams a scripted conversation as real `data:` SSE chunks.

Usage: python -I host.py FX_CORE_WASM OUT_JSON [--instrument] [--turns N]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time

import wasmtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyrepo  # noqa: E402
import wasifs  # noqa: E402
import wasmbin  # noqa: E402

# Host-side knobs for the speed iteration (speed-iter.md). The defaults are P0's host.
#   stream:   token  one SSE event per ~4 characters (about one token), one read each
#             event  the whole reply as one SSE event (content or tool calls, finish, usage)
#             chunk  token-sized events, but all of a reply's bytes in one read
#             event_chunk  one event, one read
#   provider: configured  FX_CONFIGURED_PROVIDERS (OpenAI Chat Completions), as libfx's customProvider
#             gateway     no configured provider: fx's built-in AI Gateway route (Vercel protocol SSE)
#   clock:    ms (+1 ms per clock_time_get) | frozen (constant) | ns (+1 ns per call)
#   tools:    names of the host tools to advertise (default: all five)
#   tool_schemas: host (P0's schemas) | fx (fx's own built-in schemas, for the native-tool bridge)
#   env:      extra environment entries "K=V"
#   native:   0 | 1  run fx's own file tools in the guest (zbuild/rv_tools_main.zig's
#             fxplat_native_tool export); needs fs=repo. Tools it does not cover stay host tools.
#   fs:       none (P0: no preopens, as libfx) | repo (WASI preopen "/" = the repo, wasifs.py)
OPTS = {"stream": "token", "provider": "configured", "clock": "ms", "tools": None, "tool_schemas": "host", "env": [],
        "native": "0", "fs": "none"}


TRACE = os.environ.get("FXH_TRACE") == "1"


def set_opts(**kw):
    for k, v in kw.items():
        assert k in OPTS, k
        OPTS[k] = v


PROVIDER = "local"
MODEL = "local-model"
GATEWAY_MODEL = "openai/gpt-5"
PROVIDER_JSON = json.dumps({PROVIDER: {
    "protocol": "openai-chat-completions",
    "base_url": "http://localhost:8000/v1",
    "auth": {"type": "none"},
    "model_metadata": {MODEL: {"context_window": 131072, "max_output_tokens": 8192, "supports_tool_use": True}},
}})

TOOLS = [
    {"name": "read_file", "description": "Read a text file of the repository. Returns numbered lines.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "Path relative to the repository root"}}, "required": ["path"], "additionalProperties": False},
     "idempotent": True},
    {"name": "grep_files", "description": "Search files under a directory for a regular expression. Returns path:line:text matches.",
     "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False},
     "idempotent": True},
    {"name": "edit_file", "description": "Replace one exact occurrence of old_string with new_string in a file.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"], "additionalProperties": False},
     "writes": True},
    {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
     "writes": True},
    {"name": "glob_files", "description": "List repository files matching a glob pattern.",
     "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False},
     "idempotent": True},
]
FX_TOOLS = TOOLS  # replaced by fx's own schemas for the native-tool bridge (native_tools.py)
INSTRUCTIONS = "You are a coding agent working in a small Python repository. Use the tools to inspect and change files. Be concise."

PROMPTS = [
    "In tinycalc, make div raise ZeroDivisionError with a clear message, add a test for it, and list the Python files you touched.",
    "Thanks. Now summarize in two sentences what you changed.",
]

FINAL_TEXT = (
    "I made `div` in `tinycalc/calc.py` raise `ZeroDivisionError(\"division by zero: b must be non-zero\")` "
    "before doing floor division, replacing the TODO. I added `tests/test_div_zero.py`, which checks that "
    "`div(1, 0)` raises with that message, and confirmed with a glob that the package has 8 Python files. "
    "The files I touched are `tinycalc/calc.py` and `tests/test_div_zero.py`; the evaluator in `parse.py` "
    "now surfaces the same error for expressions like `4/0`. No other behaviour changed: floor division, "
    "left-to-right evaluation and the existing tests are untouched."
)
SUMMARY_TEXT = (
    "`div` now raises a clear ZeroDivisionError instead of failing inside Python's floor division. "
    "A new test in tests/test_div_zero.py covers it."
)

# One model response per agent step of turn 1, then the final answer; turn 2 answers directly.
SCRIPT = [
    [("tool", "read_file", {"path": "tinycalc/calc.py"})],
    [("tool", "grep_files", {"pattern": "div\\(", "path": "."})],
    [("tool", "edit_file", {"path": "tinycalc/calc.py",
                            "old_string": "    # TODO: reject division by zero with a clear error\n    return a // b\n",
                            "new_string": "    if b == 0:\n        raise ZeroDivisionError(\"division by zero: b must be non-zero\")\n    return a // b\n"})],
    [("tool", "write_file", {"path": "tests/test_div_zero.py",
                             "content": "import pytest\n\nfrom tinycalc.calc import div\n\n\ndef test_div_zero():\n    with pytest.raises(ZeroDivisionError, match=\"non-zero\"):\n        div(1, 0)\n"})],
    [("tool", "glob_files", {"pattern": "**/*.py"})],
    [("text", FINAL_TEXT)],
    [("text", SUMMARY_TEXT)],
]


def toks(text, n=4):
    """Split text into ~4-character pieces, roughly one model token each."""
    return [text[i:i + n] for i in range(0, len(text), n)]


def sse(obj):
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def model_response(step, idx):
    """SSE chunks (one per network read) for scripted step `idx`."""
    if OPTS["provider"] == "gateway":
        chunks, out_tokens = gateway_response(step, idx)
    else:
        chunks, out_tokens = chat_response(step, idx)
    if OPTS["stream"] in ("chunk", "event_chunk"):
        chunks = [b"".join(chunks)]
    return chunks, out_tokens


def gateway_response(step, idx):
    """The same reply in the Vercel AI Gateway stream protocol (fx's gateway/client.zig)."""
    one = OPTS["stream"] in ("event", "event_chunk")
    ev = []
    out_tokens = 0
    kind = step[0][0]
    if kind == "text":
        ev.append({"type": "text-start", "id": "t0"})
        pieces = [step[0][1]] if one else toks(step[0][1])
        for t in pieces:
            ev.append({"type": "text-delta", "id": "t0", "delta": t})
        out_tokens = len(toks(step[0][1]))
        ev.append({"type": "text-end", "id": "t0"})
        reason = "stop"
    else:
        for k, (_, name, args) in enumerate(step):
            call_id = f"call_{idx}_{k}"
            a = json.dumps(args)
            if not one:
                ev.append({"type": "tool-input-start", "id": call_id, "toolName": name})
                for t in toks(a):
                    ev.append({"type": "tool-input-delta", "id": call_id, "delta": t})
                ev.append({"type": "tool-input-end", "id": call_id})
            ev.append({"type": "tool-call", "toolCallId": call_id, "toolName": name, "input": a})
            out_tokens += len(toks(a))
        reason = "tool-calls"
    ev.append({"type": "finish", "finishReason": {"unified": reason, "raw": reason},
               "usage": {"inputTokens": {"total": 1000 + 200 * idx}, "outputTokens": {"total": out_tokens}}})
    return [sse(e) for e in ev], out_tokens


def chat_response(step, idx):
    if OPTS["stream"] in ("event", "event_chunk"):
        return chat_response_one(step, idx)
    cid = f"chatcmpl-{idx}"
    chunks = [sse({"id": cid, "model": MODEL, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})]
    out_tokens = 0
    kind = step[0][0]
    if kind == "text":
        for t in toks(step[0][1]):
            chunks.append(sse({"id": cid, "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}))
            out_tokens += 1
        reason = "stop"
    else:
        for k, (_, name, args) in enumerate(step):
            call_id = f"call_{idx}_{k}"
            chunks.append(sse({"id": cid, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": k, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": None}]}))
            for t in toks(json.dumps(args)):
                chunks.append(sse({"id": cid, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": k, "function": {"arguments": t}}]}, "finish_reason": None}]}))
                out_tokens += 1
        reason = "tool_calls"
    chunks.append(sse({"id": cid, "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                       "usage": {"prompt_tokens": 1000 + 200 * idx, "completion_tokens": out_tokens, "total_tokens": 1000 + 200 * idx + out_tokens}}))
    chunks.append(b"data: [DONE]\n\n")
    return chunks, out_tokens


def chat_response_one(step, idx):
    """The whole reply as one Chat Completions event: the content (or every tool call with its
    complete arguments), the finish reason and the usage, then [DONE]."""
    cid = f"chatcmpl-{idx}"
    kind = step[0][0]
    if kind == "text":
        delta = {"role": "assistant", "content": step[0][1]}
        out_tokens = len(toks(step[0][1]))
        reason = "stop"
    else:
        calls = []
        out_tokens = 0
        for k, (_, name, args) in enumerate(step):
            a = json.dumps(args)
            calls.append({"index": k, "id": f"call_{idx}_{k}", "type": "function", "function": {"name": name, "arguments": a}})
            out_tokens += len(toks(a))
        delta = {"role": "assistant", "content": None, "tool_calls": calls}
        reason = "tool_calls"
    ev = {"id": cid, "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
          "usage": {"prompt_tokens": 1000 + 200 * idx, "completion_tokens": out_tokens, "total_tokens": 1000 + 200 * idx + out_tokens}}
    return [sse(ev), b"data: [DONE]\n\n"], out_tokens


class RepoTools:
    def __init__(self, root):
        self.root = root

    def _p(self, rel):
        p = os.path.normpath(os.path.join(self.root, rel))
        assert p.startswith(os.path.normpath(self.root))
        return p

    def files(self):
        out = []
        for d, _, fs in os.walk(self.root):
            for f in fs:
                out.append(os.path.relpath(os.path.join(d, f), self.root).replace("\\", "/"))
        return sorted(out)

    def call(self, name, a):
        if name == "read_file":
            with open(self._p(a["path"]), encoding="utf-8") as f:
                lines = f.read().split("\n")
            return "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines)), False
        if name == "grep_files":
            rx = re.compile(a["pattern"])
            base = a.get("path", ".").strip("./")
            res = []
            for rel in self.files():
                if base and not rel.startswith(base):
                    continue
                with open(self._p(rel), encoding="utf-8") as f:
                    for i, l in enumerate(f.read().split("\n")):
                        if rx.search(l):
                            res.append(f"{rel}:{i + 1}:{l}")
            return "\n".join(res) or "no matches", False
        if name == "edit_file":
            p = self._p(a["path"])
            with open(p, encoding="utf-8") as f:
                s = f.read()
            if s.count(a["old_string"]) != 1:
                return "old_string must occur exactly once", True
            with open(p, "w", newline="\n", encoding="utf-8") as f:
                f.write(s.replace(a["old_string"], a["new_string"]))
            return f"Edited {a['path']}", False
        if name == "write_file":
            p = self._p(a["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", newline="\n", encoding="utf-8") as f:
                f.write(a["content"])
            return f"Wrote {len(a['content'].encode())} bytes to {a['path']}", False
        if name == "glob_files":
            pat = a["pattern"]
            res = [r for r in self.files() if fnmatch.fnmatch(r, pat) or fnmatch.fnmatch(r, pat.replace("**/", ""))]
            return "\n".join(res), False
        return f"unknown tool {name}", True


def repo_dir(tmp):
    """Where the scratch repo goes: the temp dir, or <temp>/w under the WASI preopen."""
    if OPTS["fs"] == "repo":
        d = os.path.join(tmp, "w")
        os.makedirs(d)
        return d
    return tmp


def host_env():
    env = [f"FX_MODEL={GATEWAY_MODEL if OPTS['provider'] == 'gateway' else MODEL}"]
    if OPTS["provider"] == "configured":
        env = [f"FX_PROVIDER={PROVIDER}", f"FX_CONFIGURED_PROVIDERS={PROVIDER_JSON}"] + env
    return env + list(OPTS["env"])


def host_tools():
    tools = TOOLS
    if OPTS["tool_schemas"] == "fx":
        tools = FX_TOOLS
    if OPTS["tools"]:
        tools = [t for t in tools if t["name"] in OPTS["tools"]]
    return tools


class Host:
    def _init_common(self, repo, turns, log):
        self.repo = RepoTools(repo)
        self.turns = turns
        self.log = log or (lambda *a: None)
        self.stdin = bytearray()
        self.stdout = bytearray()
        self.events = []  # (fuel, kind, detail)
        self.import_calls = {}
        self.peak_mem = 0
        self.mem = None
        self.clock_ns = 1_767_000_000_000_000_000
        self.rand = 0x12345678
        self.next_id = 1
        self.pending = {}  # id -> method
        self.state = "init"
        self.turn = 0
        self.model_calls = 0
        self.streams = {}
        self.snapshots = []
        self.messages_out = {}
        self.request_bytes = []
        self.exit_code = None
        self.env = host_env()
        self.args = ["fx", "acp"]
        self.new_method = os.environ.get("FX_NEW_METHOD", "libfx/new")
        # fs=repo: the preopen "/" is the repo's parent, the repo is /w (rv_tools_main.zig).
        self.fs = wasifs.WasiFs(self, os.path.dirname(os.path.abspath(repo))) if OPTS["fs"] == "repo" else None
        self.unsupported = {}
        self.native_calls = []
        self.native_fn = None  # wasmtime: the fxplat_native_tool export (RV32: imports.c calls it)

    def __init__(self, wasm_bytes, repo, instrument=False, turns=2, log=None):
        self._init_common(repo, turns, log)
        cfg = wasmtime.Config()
        cfg.consume_fuel = True
        self.engine = wasmtime.Engine(cfg)
        self.store = wasmtime.Store(self.engine)
        self.FUEL0 = 1 << 62
        self.store.set_fuel(self.FUEL0)
        src = wasmbin.Module(wasm_bytes)
        self.names = src.func_names
        self.n_imported = src.n_imported_funcs
        self.n_defined = len(src.code)
        self.instrumented = instrument
        if instrument:
            wasm_bytes = wasmbin.instrument(src)
        self.module = wasmtime.Module(self.engine, wasm_bytes)
        self.linker = wasmtime.Linker(self.engine)
        for imp in self.module.imports:
            fn = getattr(self, "w_" + imp.name if imp.module.startswith("wasi") else "x_" + imp.name, None)
            if fn is None:
                fn = self._make_unavailable(imp.module, imp.name)
            self.linker.define_func(imp.module, imp.name, imp.type, self._wrap(imp.name, fn))
        self.inst = self.linker.instantiate(self.store, self.module)
        ex = self.inst.exports(self.store)
        self.mem = ex["memory"]
        self.start = ex["_start"]
        self.native_fn = ex["fxplat_native_tool"] if "fxplat_native_tool" in ex else None
        if instrument:
            self.counters = []
            for fi in range(self.n_defined):
                idx = self.n_imported + fi
                self.counters.append((idx, ex[f"n{idx}"], ex[f"c{idx}"], ex[f"b{idx}"]))

    # ---- plumbing -------------------------------------------------------------------
    def fuel(self):
        return self.FUEL0 - self.store.get_fuel()

    def mark(self, kind, detail=None):
        self.events.append((self.fuel(), kind, detail))
        if TRACE and kind in ("tool_call", "tool_result"):
            print("mark", kind, detail, file=sys.stderr)
        if self.instrumented and kind in ("boot_end", "stream_open", "stream_close", "tool_call", "tool_result", "response", "deliver"):
            self.snapshots.append((len(self.events) - 1, self.snapshot()))

    def snapshot(self):
        return [(g.value(self.store), c.value(self.store), b.value(self.store)) for _, g, c, b in self.counters]

    def _wrap(self, name, fn):
        def w(*args):
            self.import_calls[name] = self.import_calls.get(name, 0) + 1
            if TRACE and name.startswith(("path_", "fd_")) and name not in ("fd_write", "fd_read") or (TRACE and args and args[0] >= 3 and name in ("fd_write", "fd_read")):
                r = fn(*args)
                print("wasi", name, args, "->", r, file=sys.stderr)
                return r
            n = self.mem.data_len(self.store) if self.mem else 0
            if n > self.peak_mem:
                self.peak_mem = n
            return fn(*args)
        return w

    def _make_unavailable(self, mod, name):
        def f(*args):
            self.unsupported[name] = self.unsupported.get(name, 0) + 1
            return 52 if mod.startswith("wasi") else -1
        return f

    def rd(self, p, n):
        return bytes(self.mem.read(self.store, p, p + n))

    def wr(self, p, b):
        self.mem.write(self.store, b, p)

    def u32(self, p):
        return struct.unpack("<I", self.rd(p, 4))[0]

    def put32(self, p, v):
        self.wr(p, struct.pack("<I", v & 0xFFFFFFFF))

    def text(self, p, n):
        return self.rd(p, n).decode("utf-8", "replace")

    # ---- WASI (wasm-runtime.js semantics) -------------------------------------------
    def _vec_sizes(self, vals, cp, sp):
        self.put32(cp, len(vals))
        self.put32(sp, sum(len(v.encode()) + 1 for v in vals))
        return 0

    def _vec(self, vals, ptrs, data):
        cur = data
        for i, v in enumerate(vals):
            b = v.encode() + b"\0"
            self.put32(ptrs + 4 * i, cur)
            self.wr(cur, b)
            cur += len(b)
        return 0

    def w_args_sizes_get(self, c, s): return self._vec_sizes(self.args, c, s)
    def w_args_get(self, p, d): return self._vec(self.args, p, d)
    def w_environ_sizes_get(self, c, s): return self._vec_sizes(self.env, c, s)
    def w_environ_get(self, p, d): return self._vec(self.env, p, d)
    def w_fd_close(self, fd): return self.fs.close(fd) if self.fs and fd >= 3 else 0
    def w_fd_sync(self, fd): return 0
    def w_fd_datasync(self, fd): return 0
    def w_fd_prestat_get(self, fd, out): return self.fs.prestat_get(fd, out) if self.fs else 8
    def w_fd_prestat_dir_name(self, fd, p, n): return self.fs.prestat_dir_name(fd, p, n) if self.fs else 8
    def w_fd_seek(self, fd, off, wh, out): return self.fs.seek(fd, off, wh, out) if self.fs and fd >= 3 else 29
    def w_fd_tell(self, fd, out): return self.fs.tell(fd, out) if self.fs and fd >= 3 else 29
    def w_fd_pread(self, fd, iovs, n, off, nread): return self.fs.read(fd, iovs, n, nread, off) if self.fs else 8
    def w_fd_pwrite(self, fd, iovs, n, off, nw): return self.fs.write(fd, iovs, n, nw, off) if self.fs else 8
    def w_fd_filestat_get(self, fd, out): return self.fs.filestat_get(fd, out) if self.fs else 8
    def w_fd_filestat_set_size(self, fd, size): return self.fs.filestat_set_size(fd, size) if self.fs else 8
    def w_fd_readdir(self, fd, buf, ln, cookie, used): return self.fs.readdir(fd, buf, ln, cookie, used) if self.fs else 8
    def w_path_open(self, *a): return self.fs.path_open(*a) if self.fs else 44
    def w_path_filestat_get(self, *a): return self.fs.path_filestat_get(*a) if self.fs else 44
    def w_path_create_directory(self, *a): return self.fs.mkdir(*a) if self.fs else 44
    def w_path_unlink_file(self, *a): return self.fs.unlink(*a) if self.fs else 44
    def w_path_remove_directory(self, *a): return self.fs.rmdir(*a) if self.fs else 44
    def w_path_rename(self, *a): return self.fs.rename(*a) if self.fs else 44
    def w_path_readlink(self, *a): return self.fs.readlink(*a) if self.fs else 44

    def w_fd_fdstat_get(self, fd, out):
        if self.fs and fd >= 3:
            return self.fs.fdstat_get(fd, out)
        self.wr(out, bytes(24))
        self.wr(out, bytes([2 if fd <= 2 else 0]))
        self.wr(out + 8, b"\xff" * 16)
        return 0

    def w_clock_res_get(self, cid, out):
        self.wr(out, struct.pack("<Q", 1_000_000))
        return 0

    def w_clock_time_get(self, cid, prec, out):
        # Deterministic: 1 ms per call (P0), so both metering modes see the same times.
        self.clock_ns += {"ms": 1_000_000, "ns": 1, "frozen": 0}[OPTS["clock"]]
        self.wr(out, struct.pack("<Q", self.clock_ns))
        return 0

    def w_random_get(self, p, n):
        b = bytearray()
        for _ in range(n):
            self.rand = (self.rand * 1103515245 + 12345) & 0x7FFFFFFF
            b.append((self.rand >> 16) & 0xFF)
        self.wr(p, bytes(b))
        return 0

    def w_proc_exit(self, code):
        self.exit_code = code
        self.mark("exit", code)
        raise SystemExit(code)

    def w_fd_write(self, fd, iovs, n, nwritten):
        if self.fs and fd >= 3:
            return self.fs.write(fd, iovs, n, nwritten)
        total = 0
        chunk = bytearray()
        for i in range(n):
            p, l = struct.unpack("<II", self.rd(iovs + 8 * i, 8))
            chunk += self.rd(p, l)
            total += l
        if fd == 1:
            self.stdout += chunk
            while b"\n" in self.stdout:
                line, _, rest = self.stdout.partition(b"\n")
                self.stdout = bytearray(rest)
                if line.strip():
                    self.on_message(json.loads(line), len(line))
        elif fd == 2:
            self.log("stderr:", chunk.decode("utf-8", "replace").rstrip())
        self.put32(nwritten, total)
        return 0

    def w_fd_read(self, fd, iovs, n, nread):
        if self.fs and fd >= 3:
            return self.fs.read(fd, iovs, n, nread)
        if fd != 0:
            return 8
        if not self.stdin:
            self.idle()
        total = 0
        for i in range(n):
            p, l = struct.unpack("<II", self.rd(iovs + 8 * i, 8))
            take = bytes(self.stdin[:l])
            del self.stdin[:len(take)]
            self.wr(p, take)
            total += len(take)
            if len(take) < l:
                break
        if total:
            self.mark("deliver", total)
        self.put32(nread, total)
        return 0

    def w_poll_oneoff(self, subs, events, n, nevents):
        if not self.stdin:
            # Only block (and so let the driver send) when nothing else is subscribed.
            has_clock = any(self.rd(subs + 48 * i + 8, 1)[0] == 0 for i in range(n))
            if not has_clock:
                self.idle()
        for i in range(n):
            base = subs + 48 * i
            typ = self.rd(base + 8, 1)[0]
            if (typ == 1 and self.stdin) or typ == 0:
                self.wr(events, bytes(32))
                self.wr(events, self.rd(base, 8))
                self.wr(events + 10, bytes([typ]))
                self.put32(nevents, 1)
                return 0
        self.wr(events, bytes(32))
        self.wr(events, self.rd(subs, 8))
        self.put32(nevents, 1)
        return 0

    # ---- fx imports ---------------------------------------------------------------
    def x_fx_http_stream_open(self, mp, ml, up, ul, hp, hl, bp, bl):
        body = self.rd(bp, bl)
        req = json.loads(body)
        idx = self.model_calls
        self.model_calls += 1
        step = SCRIPT[idx]
        chunks, out_tokens = model_response(step, idx)
        h = len(self.streams) + 1
        self.streams[h] = {"chunks": chunks, "pos": 0}
        self.request_bytes.append({"step": idx, "body_bytes": bl, "messages": len(req.get("messages", [])),
                                   "tools": len(req.get("tools", [])), "out_tokens": out_tokens,
                                   "sse_bytes": sum(len(c) for c in chunks), "sse_events": len(chunks),
                                   "url": self.text(up, ul)})
        if idx == 0:
            self.first_request = req
        self.mark("stream_open", idx)
        return h

    def x_fx_http_stream_status(self, h, out):
        self.wr(out, struct.pack("<H", 200))
        return 1

    def x_fx_http_stream_next(self, h, p, cap):
        st = self.streams[h]
        if st["pos"] >= len(st["chunks"]):
            return 0
        c = st["chunks"][st["pos"]]
        if len(c) > cap:
            st["chunks"][st["pos"]] = c[cap:]
            c = c[:cap]
        else:
            st["pos"] += 1
        self.wr(p, c)
        return len(c)

    def x_fx_http_stream_close(self, h):
        self.mark("stream_close", h)

    def x_fx_host_tool_call(self, np, nl, ap, al, op, ocap, sp, cp, cl):
        """wasm-runtime.js hostToolCall: run the host tool, copy its text result out."""
        name = self.text(np, nl)
        args = json.loads(self.rd(ap, al))
        self.mark("tool_call", {"name": name, "args_bytes": al})
        if OPTS["native"] == "1" and self.native_fn is not None:
            # fx's own tool code, in the guest: re-enter the module through the root's export.
            r = self.native_fn(self.store, np, nl, ap, al, op, ocap, sp, cp, cl)
            if r != -100:
                self.native_calls.append({"name": name, "status": self.rd(sp, 1)[0], "result": self.text(op, r)[:400] if r > 0 else r})
                self.mark("tool_result", {"name": name, "bytes": r, "native": True})
                return r
        t0 = time.perf_counter()
        content, is_error = self.repo.call(name, args)
        self.host_tool_seconds = getattr(self, "host_tool_seconds", 0) + time.perf_counter() - t0
        out = content.encode()
        self.wr(sp, bytes([1 if is_error else 0]))
        if len(out) > ocap:
            return -3
        self.wr(op, out)
        self.mark("tool_result", {"name": name, "bytes": len(out)})
        return len(out)

    def x_fx_http_stream_retry_after(self, h): return -1
    def x_fx_steering_state(self): return -1
    def x_fx_steering_take(self, p, cap): return 0
    def x_fx_steering_close(self): return None
    def x_fx_journal_flush(self): return 0
    def x_fx_attachment_size(self, i): return -1
    def x_fx_attachment_take(self, i, p, c): return -1
    def x_fx_attachment_put(self, p, l): return -1
    def x_fx_host_tool_result_release(self): return None
    def x_fx_host_tool_result_read(self, o, p, c): return -1

    # ---- ACP client (engine.js) -------------------------------------------------------
    def send(self, msg):
        self.stdin += (json.dumps(msg, separators=(",", ":")) + "\n").encode()

    def request(self, method, params):
        i = self.next_id
        self.next_id += 1
        self.pending[i] = method
        self.send({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        self.mark("request", method)

    def idle(self):
        """The core waits for stdin: send the next scripted client message."""
        if self.state == "init":
            self.mark("boot_end")
            self.request("initialize", {"protocolVersion": 1, "clientCapabilities": {"libfx": {"tools": host_tools(), "instructions": INSTRUCTIONS}}})
            self.state = "await_init"
        elif self.state == "ready_session":
            self.request(self.new_method, {} if self.new_method == "libfx/new" else {"cwd": "/", "mcpServers": []})
            self.state = "await_session"
        elif self.state == "ready_prompt":
            if self.turn >= self.turns:
                self.state = "closing"
                self.mark("close_stdin")
                return
            self.request("session/prompt", {"sessionId": self.session_id, "prompt": [{"type": "text", "text": PROMPTS[self.turn]}]})
            self.turn += 1
            self.state = "await_prompt"
        else:
            self.mark("idle_unexpected", self.state)

    def on_message(self, msg, size):
        method = msg.get("method")
        self.messages_out[method or "response"] = self.messages_out.get(method or "response", 0) + 1
        if method is None:
            m = self.pending.pop(msg.get("id"), None)
            self.mark("response", m)
            if "error" in msg:
                self.log("error response", m, msg["error"])
            if m == "initialize":
                self.state = "ready_session"
            elif m in ("session/new", "libfx/new"):
                self.session_id = msg["result"]["sessionId"]
                self.state = "ready_prompt"
            elif m == "session/prompt":
                self.log("turn end", msg.get("result"))
                self.state = "ready_prompt"
            return
        if method == "libfx/tool_call":
            p = msg["params"]
            self.mark("tool_call", p["name"])
            t0 = time.perf_counter()
            content, is_error = self.repo.call(p["name"], p["input"])
            self.host_tool_seconds = getattr(self, "host_tool_seconds", 0) + time.perf_counter() - t0
            self.send({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": content, "isError": is_error}})
            self.mark("tool_result", {"name": p["name"], "bytes": len(content.encode())})
        elif method == "session/request_permission":
            opts = msg["params"].get("options", [])
            pick = next((o["optionId"] for o in opts if "allow" in o.get("kind", "")), opts[0]["optionId"] if opts else None)
            self.mark("permission", pick)
            self.send({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": {"outcome": "selected", "optionId": pick}}})
        elif method == "libfx/journal_flush":
            self.send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        elif method == "session/update":
            pass
        elif "id" in msg:
            self.log("unhandled request", method)
            self.send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "not supported"}})

    def run(self):
        t0 = time.perf_counter()
        self.mark("start")
        try:
            self.start(self.store)
            self.mark("returned")
        except BaseException as e:  # proc_exit raises SystemExit through the trap
            if self.exit_code is None:
                self.mark("trap", f"{type(e).__name__}: {str(e)[:600]}")
        self.wall = time.perf_counter() - t0
        self.peak_mem = max(self.peak_mem, self.mem.data_len(self.store))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wasm")
    ap.add_argument("out")
    ap.add_argument("--instrument", action="store_true")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--opt", action="append")
    a = ap.parse_args()
    from rv.rvhost import parse_opts, turn_summary  # noqa: E402  (same option syntax as rvhost.py)
    set_opts(**parse_opts(a.opt))
    tmp = repo_dir(tempfile.mkdtemp(prefix="fxrepo-"))
    tinyrepo.make_repo(tmp)
    h = Host(open(a.wasm, "rb").read(), tmp, a.instrument, a.turns, log=lambda *x: print(*x, file=sys.stderr))
    h.run()
    res = {
        "wasm": os.path.abspath(a.wasm), "instrumented": a.instrument, "fuel_total": h.fuel(),
        "peak_linear_memory_bytes": h.peak_mem, "wall_s": h.wall, "exit_code": h.exit_code,
        "events": h.events, "import_calls": h.import_calls, "model_requests": h.request_bytes,
        "messages_out": h.messages_out, "opts": dict(OPTS), "summary": turn_summary(h.events),
        "unsupported_imports": h.unsupported, "native_calls": h.native_calls,
        "fs_calls": h.import_calls, "final_repo": {r: open(os.path.join(tmp, r), encoding="utf-8").read() for r in h.repo.files() if r.startswith("tinycalc/calc") or r.startswith("tests/test_div")},
    }
    if a.instrument:
        res["func_names"] = {str(k): v for k, v in h.names.items()}
        res["snapshots"] = [(ei, [(i, n, c, b) for (i, _, _, _), (n, c, b) in zip(h.counters, s) if n or c]) for ei, s in h.snapshots]
        final = h.snapshot()
        res["final_counts"] = [(i, n, c, b) for (i, _, _, _), (n, c, b) in zip(h.counters, final) if n or c]
    if hasattr(h, "first_request"):
        res["first_request"] = h.first_request
    with open(a.out, "w") as f:
        json.dump(res, f)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: res[k] for k in ("fuel_total", "peak_linear_memory_bytes", "wall_s", "exit_code")}))
    for e in h.events:
        if e[1] not in ("deliver",):
            print(e[0], e[1], str(e[2])[:150])


if __name__ == "__main__":
    main()
