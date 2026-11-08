"""The pull-request handler for fx's open_pull_request tool (docs/fx/pr.md). Runs on the laptop that
holds the GitHub login; the box reaches it through an SSH reverse tunnel to a localhost-only port.

    python pr_server.py serve --port 8765 --run RUN_ID --issue N --base SHA --log CALLS.json [--cache DIR]

What it does for one call, and nothing else:
  1. the diff of fx's file store (sent by the box) against the base commit's tree;
  2. a commit of exactly those file contents on the base commit, message = fx's title, author and
     committer `fxputer <114850579+divagr18@users.noreply.github.com>`;
  3. a push of that commit to a new branch `fx/issue-<N>-<run>` (never an existing branch, never a
     force push, never main);
  4. a pull request into main with fx's title, and fx's body plus a footer this host adds;
  5. the pull request's URL, or GitHub's error text, as the tool result.

Hard limits, in code: one repository (REPO), branches under fx/ only, one pull request per run (and a
few attempts at most), only the run ids the driver registered. The GitHub token never leaves `gh`:
git asks `gh auth git-credential`, the API goes through `gh api`, and neither is logged.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = "divagr18/fxputer-sandbox"  # the only repository this handler ever writes to
BASE_BRANCH = "main"
AUTHOR = ("fxputer", "114850579+divagr18@users.noreply.github.com")
BRANCH_RE = re.compile(r"^fx/[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
MAX_PRS_PER_RUN = 1
MAX_ATTEMPTS_PER_RUN = 3
MAX_TITLE = 200
MAX_BODY = 20000
MAX_FILES_BYTES = 1 << 20
GH_CRED = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]

FOOTER = ("\n\n---\n"
          "*Added by the host, not by fx:* this pull request was opened by "
          "[fx](https://github.com/vercel-labs/fx) running on the in-weights CPU of "
          "[Qwenputer](https://github.com/divagr18/qwenputer) (Executor v4), with Qwen3.5-4B as its model. "
          "The title, the text above this line and the change are fx's. Run `{run_id}`.")


class PrError(Exception):
    """Text that goes back to fx as the tool's error result."""


def footer(run_id: str, issue: int | None, body: str) -> str:
    f = FOOTER.format(run_id=run_id)
    if issue is not None and not re.search(rf"#{issue}\b", body):
        f += f"\nFixes #{issue}"
    return f


def safe_path(p: str) -> bool:
    if not p or p.startswith("/") or "\\" in p or "\0" in p:
        return False
    parts = p.split("/")
    return all(x not in ("", ".", "..") for x in parts) and parts[0] != ".git" and ".git" not in parts


class Git:
    """git plumbing on a bare cache clone: blob, tree and commit built without a work tree, so the
    bytes are fx's bytes (no line-ending filters)."""

    def __init__(self, cache: str, remote: str, cred: list[str]) -> None:
        self.cache, self.remote, self.cred = cache, remote, cred
        if not os.path.isdir(os.path.join(cache, "objects")):
            os.makedirs(cache, exist_ok=True)
            self.run("init", "--bare", "-q", cache, cwd=None, bare=False)
        self.run("config", "core.autocrlf", "false")

    def run(self, *args, inp: bytes | None = None, env=None, cwd="", bare=True, check=True) -> bytes:
        cmd = ["git"] + self.cred + (["--git-dir", self.cache] if bare else []) + list(args)
        r = subprocess.run(cmd, input=inp, capture_output=True, env={**os.environ, **(env or {})},
                           cwd=None if cwd in ("", None) else cwd)
        if check and r.returncode:
            raise PrError(f"git {args[0]} failed: {r.stderr.decode('utf-8', 'replace').strip()[-400:]}")
        return r.stdout

    def fetch_base(self) -> str:
        self.run("fetch", "-q", self.remote, f"+refs/heads/{BASE_BRANCH}:refs/remotes/origin/{BASE_BRANCH}")
        return self.run("rev-parse", f"refs/remotes/origin/{BASE_BRANCH}").decode().strip()

    def tree(self, commit: str) -> dict[str, tuple[str, str]]:
        out = {}
        for ent in self.run("ls-tree", "-r", "-z", commit).split(b"\0"):
            if ent:
                meta, path = ent.split(b"\t", 1)
                mode, typ, sha = meta.decode().split()
                if typ == "blob":
                    out[path.decode()] = (mode, sha)
        return out

    def files(self, commit: str) -> dict[str, bytes]:
        return {p: self.run("cat-file", "blob", sha) for p, (_, sha) in self.tree(commit).items()}

    def blob(self, data: bytes) -> str:
        return self.run("hash-object", "-w", "--no-filters", "--stdin", inp=data).decode().strip()

    def commit(self, base: str, changes: dict[str, bytes | None], message: str) -> str:
        fd, idx = tempfile.mkstemp(prefix="fxpr-index-")
        os.close(fd)
        os.unlink(idx)
        env = {"GIT_INDEX_FILE": idx}
        try:
            self.run("read-tree", base, env=env)
            lines = [f"0 {'0' * 40}\t{p}" if data is None else f"100644 {self.blob(data)}\t{p}"
                     for p, data in changes.items()]  # (mode 0: the path leaves the index)
            self.run("update-index", "--index-info", inp="".join(x + "\n" for x in lines).encode(), env=env)
            tree = self.run("write-tree", env=env).decode().strip()
        finally:
            if os.path.exists(idx):
                os.unlink(idx)
        who = {"GIT_AUTHOR_NAME": AUTHOR[0], "GIT_AUTHOR_EMAIL": AUTHOR[1],
               "GIT_COMMITTER_NAME": AUTHOR[0], "GIT_COMMITTER_EMAIL": AUTHOR[1]}
        return self.run("commit-tree", tree, "-p", base, "-m", message, env=who).decode().strip()

    def branch_exists(self, branch: str) -> bool:
        return bool(self.run("ls-remote", "--heads", self.remote, f"refs/heads/{branch}").strip())

    def push_new_branch(self, commit: str, branch: str) -> None:
        if not BRANCH_RE.match(branch):
            raise PrError(f"refusing to push to {branch!r}: only fx/... branches")
        if self.branch_exists(branch):
            raise PrError(f"branch {branch} already exists; not overwriting it")
        # a plain push of a new ref: no --force, no + in the refspec, never main
        self.run("push", "-q", self.remote, f"{commit}:refs/heads/{branch}")


def gh_create_pr(repo: str, head: str, base: str, title: str, body: str) -> dict:
    """POST /repos/{repo}/pulls through gh (its stored login; nothing of it is printed)."""
    if repo != REPO:
        raise PrError(f"refusing to open a pull request on {repo}")
    req = json.dumps({"title": title, "head": head, "base": base, "body": body, "draft": False})
    r = subprocess.run(["gh", "api", "-X", "POST", f"repos/{repo}/pulls", "--input", "-"],
                       input=req.encode(), capture_output=True)
    try:
        ans = json.loads(r.stdout or b"{}")
    except json.JSONDecodeError:
        ans = {}
    if r.returncode or "html_url" not in ans:
        msg = ans.get("message") or r.stderr.decode("utf-8", "replace").strip()
        errs = "; ".join(e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in ans.get("errors", []))
        raise PrError(f"GitHub: {msg}" + (f" ({errs})" if errs else ""))
    return {"url": ans["html_url"], "number": ans["number"]}


class Handler:
    """One driver run's handler. remote / create_pr are injectable for the tests only; the command
    line always uses GitHub's REPO through gh."""

    def __init__(self, cache: str, runs: dict, remote: str | None = None, create_pr=None, cred=None,
                 log_path: str | None = None) -> None:
        self.git = Git(cache, remote or f"https://github.com/{REPO}.git", GH_CRED if cred is None else cred)
        self.create_pr = create_pr or gh_create_pr
        self.runs = runs  # run_id -> {"issue": N or None, "base": sha}
        self.count: dict[str, int] = {}
        self.attempts: dict[str, int] = {}
        self.calls: list[dict] = []
        self.log_path = log_path
        self.lock = threading.Lock()

    def handle(self, req: dict) -> dict:
        with self.lock:
            t0 = time.time()
            rec = {"t": round(t0, 3), "run_id": req.get("run_id"), "title": req.get("title"),
                   "body": req.get("body")}
            try:
                out = self._handle(req, rec)
                ans = {"ok": True, "text": out["url"], **out}
            except PrError as e:
                ans = {"ok": False, "text": str(e)}
            rec.update(answer=ans, seconds=round(time.time() - t0, 2))
            self.calls.append(rec)
            if self.log_path:
                json.dump(self.calls, open(self.log_path, "w"), indent=1)
            return ans

    def _handle(self, req: dict, rec: dict) -> dict:
        run_id = req.get("run_id")
        if run_id not in self.runs:
            raise PrError("this run is not registered with the pull request service")
        run = self.runs[run_id]
        if self.count.get(run_id, 0) >= MAX_PRS_PER_RUN:
            raise PrError(f"a pull request was already opened in this run: {run.get('url')}")
        if self.attempts.get(run_id, 0) >= MAX_ATTEMPTS_PER_RUN:
            raise PrError("too many attempts to open a pull request in this run")
        title, body = req.get("title"), req.get("body")
        if not isinstance(title, str) or not isinstance(body, str):
            raise PrError("open_pull_request needs a string title and a string body")
        title = " ".join(title.split())
        if not title:
            raise PrError("the title is empty")
        if len(title) > MAX_TITLE or len(body) > MAX_BODY:
            raise PrError(f"the title (at most {MAX_TITLE} characters) or the body (at most {MAX_BODY}) is too long")
        files = req.get("files")
        if not isinstance(files, dict):
            raise PrError("no file store in the request")
        store, total = {}, 0
        for p, v in files.items():
            if not safe_path(p):
                raise PrError(f"refusing the path {p!r}")
            store[p] = base64.b64decode(v)
            total += len(store[p])
        if total > MAX_FILES_BYTES:
            raise PrError("the file store is too large")
        base = run["base"]
        btree = self.git.tree(base)
        changes: dict[str, bytes | None] = {}
        for p, data in sorted(store.items()):
            old = btree.get(p)
            if old is None or old[1] != self.git.blob(data):
                changes[p] = data
        for p in sorted(btree):
            if p not in store:
                changes[p] = None
        rec["changes"] = {p: (None if d is None else len(d)) for p, d in changes.items()}
        if not changes:
            raise PrError("there are no changes to commit: the files are the same as on main")
        # an attempt is a push: checks that fail before here cost nothing and reach no remote
        n = self.attempts[run_id] = self.attempts.get(run_id, 0) + 1
        commit = self.git.commit(base, changes, title)
        branch = f"fx/{run['slug']}" + ("" if n == 1 else f"-{n}")
        self.git.push_new_branch(commit, branch)
        rec.update(commit=commit, branch=branch)
        pr = self.create_pr(REPO, branch, BASE_BRANCH, title, body + footer(run_id, run.get("issue"), body))
        self.count[run_id] = self.count.get(run_id, 0) + 1
        run["url"] = pr["url"]
        return {"url": pr["url"], "number": pr["number"], "branch": branch, "commit": commit,
                "changed": sorted(changes)}


def serve(handler: Handler, port: int, ready=None) -> HTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # (requests are in the handler's own log, without bodies)
            pass

        def _send(self, code: int, obj: dict) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/open_pull_request":
                return self._send(404, {"ok": False, "text": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 4 * MAX_FILES_BYTES:
                return self._send(413, {"ok": False, "text": "request too large"})
            try:
                req = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                return self._send(400, {"ok": False, "text": "bad request"})
            self._send(200, handler.handle(req))

    srv = HTTPServer(("127.0.0.1", port), H)  # localhost only: the box comes in through ssh -R
    if ready:
        ready()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("serve",))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run", required=True)
    ap.add_argument("--slug", required=True, help="the branch is fx/<slug>")
    ap.add_argument("--issue", type=int, default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "fxpr-cache.git"))
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    h = Handler(a.cache, {a.run: {"issue": a.issue, "base": a.base, "slug": a.slug}}, log_path=a.log)
    if os.path.exists(a.log):  # a restarted handler keeps the run's cap: earlier calls count
        h.calls = json.load(open(a.log))
        for c in h.calls:
            if c.get("branch"):
                h.attempts[a.run] = h.attempts.get(a.run, 0) + 1
            if c["answer"].get("ok"):
                h.count[a.run] = h.count.get(a.run, 0) + 1
                h.runs[a.run]["url"] = c["answer"]["url"]
    head = h.git.fetch_base()
    if head != a.base:
        print(json.dumps({"note": "main moved", "main": head, "base": a.base}), flush=True)
    srv = serve(h, a.port)
    print(json.dumps({"serving": f"127.0.0.1:{a.port}", "run": a.run, "repo": REPO}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
