"""fx's exact RV32 binary in rv32emu, with the packet path to a model backend, on one P0c task.

  python -I e2e.py DLL ELF IMPORTS MODEL_DIR OUT_JSON --task t1_bug
        --backend scripted | hf:NAME | replay:RUN_JSON  [--native] [--units all] [--seed N]

`--native` is speed-iter's row 24 (fx's own read / edit / write on the CPU); without it, row 11.
`--provider chat` is fx's configured Chat Completions route instead of its gateway route.
`--turns 2` adds a second prompt (fx rewrites history at a new turn: io.md).
`replay:RUN_JSON` answers from a recorded run's packets, and fails if fx's request differs from
the recording by a byte. It re-measures a run with units on or off without the model: the
units' contract is identical host I/O, and this run's hash chain checks it.

Reported as measured: RV32 instructions, trace tokens, the host-call hash chain, the repository
after the run and the hidden check of p0c/tasks.py. Whether a *model* solved the task depends on
the backend: see scripted.py.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
for p in (FX, os.path.join(FX, "rv"), os.path.join(FX, "circ"), os.path.join(FX, "p0c"), HERE):
    sys.path.insert(0, p)
import delta as D  # noqa: E402
import host as H  # noqa: E402
import protocol as P  # noqa: E402
import run as RUN  # noqa: E402  (circ/run.py)
import rvhost as RH  # noqa: E402
import tasks as TK  # noqa: E402  (p0c/tasks.py)
import units as U  # noqa: E402
import wasifs  # noqa: E402

GW = ["provider=gateway", "env=AI_GATEWAY_API_KEY=placeholder", "stream=event_chunk", "clock=frozen"]
CHAT = ["provider=configured", "stream=event_chunk", "clock=frozen"]
SECOND_PROMPT = "Thanks. Now summarize in two sentences what you changed."
MAX_CALLS = 40


class IoHost(RH.RvHost):
    """rvhost's host with the model mock replaced by the packet path."""

    def attach(self, link) -> None:
        self.link = link
        self.packets: list[bytes] = []
        self.calls_log: list[dict] = []

    def x_fx_http_stream_open(self, mp, ml, up, ul, hp, hl, bp, bl):
        if self.model_calls >= MAX_CALLS:
            self.log("too many model calls")
            return -1
        idx = self.model_calls
        self.model_calls += 1
        url, headers, body = self.rd(up, ul), self.rd(hp, hl), self.rd(bp, bl)
        req = P.request(idx, url, headers, body)
        i0, t0 = self.fuel(), self.e.rv_tokens()
        chunks, rec = self.link.call(body)
        rep = P.reply(idx, chunks)
        self.packets += [req.encode(), rep.encode()]
        h = len(self.streams) + 1
        self.streams[h] = {"chunks": list(chunks), "pos": 0}
        rec.update(rv_instr_at_open=i0, tokens_at_open=t0)
        self.calls_log.append(rec)
        self.request_bytes.append({"step": idx, "body_bytes": bl, "url": url.decode()})
        self.mark("stream_open", idx)
        return h


class ReplayLink:
    """Answers from recorded packets; a request that differs from the recording is an error."""

    def __init__(self, path: str) -> None:
        run = json.load(open(path))
        data = bytes.fromhex(run["packets_hex"])
        pk = P.decode_stream(data)
        self.pairs = list(zip(pk[0::2], pk[1::2]))
        self.records = run["model_calls"]
        self.i = 0
        self.mismatch = 0

    def call(self, raw: bytes):
        q, a = self.pairs[self.i]
        if P.parse_request(q)[2] != raw:
            self.mismatch += 1
        rec = {k: v for k, v in self.records[self.i].items() if k not in ("rv_instr_at_open", "tokens_at_open")}
        self.i += 1
        return P.parse_reply(a)[1], rec


def write_repo(root: str) -> None:
    for rel, text in TK.COMMON.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="\n", encoding="utf-8") as f:
            f.write(text)


def solved(root: str, task: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c", TK.TASKS[task]["check"]], cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    return r.returncode == 0, tail


def make_link(args, template):
    kind, _, arg = args.backend.partition(":")
    if kind == "scripted":
        import scripted as SC

        return D.DeltaLink(template, SC.ScriptedBackend(template, args.task, extra=args.turns - 1)), None
    if kind == "replay":
        return ReplayLink(arg), None
    if kind == "hf":
        import qwen_hf as QH

        b = QH.HfBackend.load(arg, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
        return D.DeltaLink(template, b), b
    raise SystemExit(f"unknown backend {args.backend}")


def session(args, template) -> dict:
    opts = list(GW if args.provider == "gateway" else CHAT) + (["native=1", "fs=repo"] if args.native else [])
    H.set_opts(**RH.parse_opts(opts))
    H.PROMPTS[:] = [TK.TASKS[args.task]["prompt"], SECOND_PROMPT][:args.turns]
    dll = RUN.setup_dll(ctypes.CDLL(os.path.abspath(args.dll)))
    tmp = H.repo_dir(tempfile.mkdtemp(prefix="fxio-"))
    write_repo(tmp)
    h = IoHost(dll, args.elf, json.load(open(args.imports)), tmp, args.turns, log=lambda *x: None, prof=False, memtrace=False)
    if h.fs is not None:  # a deterministic inode, as circ/run.py
        root = os.path.dirname(os.path.abspath(tmp))

        def _filestat(path, out, fs=h.fs):
            try:
                st = os.stat(path)
            except OSError:
                return wasifs.ENOENT
            ino = zlib.crc32(os.path.relpath(path, root).replace(os.sep, "/").encode())
            ft = wasifs.FT_DIR if os.path.isdir(path) else wasifs.FT_REG
            fs.h.wr(out, struct.pack("<QQBxxxxxxxQQQQQ", 1, ino, ft, 1, st.st_size, 1_767_000_000_000_000_000,
                                     1_767_000_000_000_000_000, 1_767_000_000_000_000_000))
            return wasifs.ESUCCESS

        h.fs._filestat = _filestat
    link, backend = make_link(args, template)
    h.attach(link)
    rec = RUN.Recorder(h)
    specs, skipped = U.specs_for(args.elf, args.units)
    ids = RUN.install(dll, specs)
    dll.rv_units_mode(1 if specs else 0)
    marks = []
    mark0 = h.mark

    def mark(kind, detail=None):
        marks.append((h.fuel(), dll.rv_tokens(), kind, detail))
        return mark0(kind, detail)

    h.mark = mark
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    start = next(((i, t) for i, t, k, d in marks if k == "request" and d == "session/prompt"), None)
    end = next(((i, t) for i, t, k, d in marks if k == "response" and d == "session/prompt"), None)
    ok, why = solved(tmp, args.task)
    starts = [(i, t) for i, t, k, d in marks if k == "request" and d == "session/prompt"]
    ends = [(i, t) for i, t, k, d in marks if k == "response" and d == "session/prompt"]
    repo = {r: open(os.path.join(tmp, r), encoding="utf-8").read() for r in h.repo.files() if "__pycache__" not in r}
    res = {
        "task": args.task, "backend": args.backend, "units": args.units, "native": args.native, "elf": os.path.basename(args.elf),
        "elf_sha256": hashlib.sha256(open(args.elf, "rb").read()).hexdigest(),
        "solved": ok, "check": why, "model_calls": h.calls_log, "steps": len(h.calls_log),
        "turn": {"instructions": end[0] - start[0], "tokens": end[1] - start[1]} if start and end else None,
        "turns": [{"instructions": e[0] - s[0], "tokens": e[1] - s[1]} for s, e in zip(starts, ends)],
        "provider": args.provider,
        "boot": next(({"instructions": i, "tokens": t} for i, t, k, d in marks if k == "boot_end"), None),
        "session_end": {"instructions": h.fuel(), "tokens": dll.rv_tokens()}, "emulator_wall_s": wall,
        "host_calls": rec.calls, "host_io_sha256": rec.sha.hexdigest(), "import_calls": h.import_calls,
        "unsupported_imports": h.unsupported, "native_calls": h.native_calls,
        "trap": [e for e in h.events if e[1] == "trap"], "misaligned": h.e.rv_misaligned(),
        "final_repo_sha256": hashlib.sha256(json.dumps(repo, sort_keys=True).encode()).hexdigest(),
        "packets_hex": b"".join(h.packets).hex() if hasattr(h, "packets") else "",
        "unit_stats": RUN.unit_stats(dll, specs, ids), "units_not_installed": skipped,
    }
    if backend is not None:
        res["backend_seconds"] = backend.seconds
    if isinstance(link, ReplayLink):
        res["replay_mismatches"] = link.mismatch
        res["replay_used"] = f"{link.i}/{len(link.pairs)}"
    shutil.rmtree(os.path.dirname(tmp) if args.native else tmp, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    for a in ("dll", "elf", "imports", "model_dir", "out"):
        ap.add_argument(a)
    ap.add_argument("--task", required=True, choices=list(TK.TASKS))
    ap.add_argument("--backend", default="scripted")
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--units", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="gateway", choices=("gateway", "chat"))
    ap.add_argument("--turns", type=int, default=1, choices=(1, 2))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    res = session(a, D.QwenTemplate(a.model_dir))
    json.dump(res, open(a.out, "w"))
    show = {k: res[k] for k in ("task", "solved", "check", "steps", "turn", "trap", "misaligned", "unsupported_imports")}
    show["kinds"] = [c["kind"] for c in res["model_calls"]]
    show["replay"] = (res.get("replay_mismatches"), res.get("replay_used"))
    print(json.dumps(show, default=str))


if __name__ == "__main__":
    main()
