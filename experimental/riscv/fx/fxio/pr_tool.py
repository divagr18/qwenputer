"""fx's one extra tool, `open_pull_request(title, body)` (docs/fx/pr.md). The box side.

fx-core advertises a host tool the ACP client declares in `initialize`, under
`clientCapabilities.libfx.tools`, and routes the model's calls to it to the host import
`fx_host_tool_call`. On the RV32 build (row 24) the guest first offers the call to fx's own tool
code (read / edit / write); for any other name that returns -100 and the call leaves the guest as
an ecall. On the chip that ecall is the live I/O path of docs/fx/live.md: the name and the JSON
arguments are read out of fx's memory through the read sled (tokens the chip prints), and the result
goes back through the write sled (tokens the host feeds). Nothing here changes fx's source or binary.

What the host does with the call is mechanics only: it sends fx's file store (the repository fx
edited with its own tools) and fx's title and body to the pull-request handler (pr_server.py, on the
laptop that holds the GitHub login, through an SSH reverse tunnel), and hands GitHub's answer back to
fx as the tool result: the pull request's URL, or the error text.

The lockstep rv32emu hosts never call the handler: they get the chip host's recorded answer, and
their arguments must equal the chip host's byte for byte (ReplayPr).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

NAME = "open_pull_request"

# What Qwen sees (fx puts it in the request's tool list). Kept short; any change is disclosed in pr.md.
TOOL = {
    "name": NAME,
    "description": ("Open a pull request on GitHub with the changes you made to the repository's files. "
                    "Call it once, when your change is complete. Returns the pull request's URL."),
    "inputSchema": {"type": "object", "properties": {
        "title": {"type": "string", "description": "A short title for the pull request"},
        "body": {"type": "string", "description": "What you changed and why"}},
        "required": ["title", "body"], "additionalProperties": False},
    "writes": True,
}

# The user prompt for an issue: the issue as GitHub has it, and one plain instruction (disclosed).
PROMPT = ("Fix this GitHub issue in the repository.\n\n"
          "Issue #{number}: {title}\n\n{body}\n\n"
          "When your change is complete, open a pull request for it with the open_pull_request tool.")


def issue_prompt(number: int, title: str, body: str) -> str:
    return PROMPT.format(number=number, title=title.strip(), body=body.strip())


SKIP_PARTS = ("__pycache__", ".git")


def store_files(root: str) -> dict[str, bytes]:
    """fx's file store as it stands: every file under the repository root, by relative path."""
    out = {}
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_PARTS)
        for f in sorted(fs):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = open(p, "rb").read()
    return out


def changes(base: dict[str, bytes], now: dict[str, bytes]) -> dict[str, bytes | None]:
    """The diff of the store against the base tree: path -> new bytes, or None for a deletion."""
    out: dict[str, bytes | None] = {p: b for p, b in now.items() if base.get(p) != b}
    out.update({p: None for p in base if p not in now})
    return dict(sorted(out.items()))


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class PrClient:
    """The chip host's side of the tool: posts the call to the handler, returns (text, is_error)."""

    def __init__(self, url: str, run_id: str, timeout: float = 180.0) -> None:
        self.url, self.run_id, self.timeout = url.rstrip("/"), run_id, timeout
        self.calls: list[dict] = []

    def __call__(self, root: str, args: dict) -> tuple[str, bool]:
        rec = {"args": args}
        self.calls.append(rec)
        title, body = args.get("title"), args.get("body")
        if not isinstance(title, str) or not isinstance(body, str):
            rec["result"] = ("open_pull_request needs a string title and a string body.", True)
            return rec["result"]
        files = store_files(root)
        req = {"run_id": self.run_id, "title": title, "body": body,
               "files": {p: b64(b) for p, b in files.items()}}
        data = json.dumps(req).encode()
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                self.url + "/open_pull_request", data=data, headers={"Content-Type": "application/json"}),
                timeout=self.timeout)
            ans = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                ans = json.loads(e.read())
            except Exception:  # noqa: BLE001
                ans = {"ok": False, "text": f"the pull request service answered HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001  (the tunnel is down, a timeout: fx is told, not crashed)
            ans = {"ok": False, "text": f"could not reach the pull request service: {type(e).__name__}"}
        rec.update(answer=ans, store=sorted(files))
        rec["result"] = (str(ans.get("text", "")), not ans.get("ok", False))
        return rec["result"]


class PrTools:
    """RepoTools with open_pull_request added (the chip host's). Every other tool is unchanged."""

    def __init__(self, inner, client) -> None:
        self.inner, self.client = inner, client
        self.root = inner.root
        self.log: list[tuple[str, dict, tuple]] = []

    def files(self):
        return self.inner.files()

    def call(self, name, a):
        if name != NAME:
            return self.inner.call(name, a)
        r = self.client(self.root, a)
        self.log.append((name, a, r))
        return r


class ReplayPr:
    """A lockstep host's open_pull_request: the chip host's answer, after its arguments are checked
    equal to the chip host's. It never reaches GitHub."""

    def __init__(self, inner, chip_tools: PrTools, who: str) -> None:
        self.inner, self.chip, self.who, self.i = inner, chip_tools, who, 0
        self.root = inner.root

    def files(self):
        return self.inner.files()

    def call(self, name, a):
        if name != NAME:
            return self.inner.call(name, a)
        if self.i >= len(self.chip.log):
            raise RuntimeError(f"{self.who}: open_pull_request call {self.i} that the chip host did not make")
        _, a0, r = self.chip.log[self.i]
        if a0 != a:
            raise RuntimeError(f"{self.who}: open_pull_request call {self.i}: arguments differ from the chip's")
        self.i += 1
        return r
