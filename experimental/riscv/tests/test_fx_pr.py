"""fx's open_pull_request tool (docs/fx/pr.md): the handler against a fake git remote, the box's
client through the HTTP service, the lockstep replay, the tool in ACP initialize, and (on a GPU box
with fx's ELF) a whole session on the chip where the call arrives as host-call tokens."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FXIO = os.path.join(os.path.dirname(HERE), "fx", "fxio")
sys.path.insert(0, os.path.dirname(FXIO))  # fx/ (host.py)
sys.path.insert(0, FXIO)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fx", "p0c"))
import pr_server as PS  # noqa: E402
import pr_tool as PR  # noqa: E402
import tasks as TK  # noqa: E402

README = b"# fxputer-sandbox\n\nThe sandbox's own README.\n"
BASE = {**{p: s.encode() for p, s in TK.COMMON.items()}, "README.md": README}


def git(*a, cwd=None, env=None) -> str:
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode()


@pytest.fixture()
def remote(tmp_path):
    """A bare repository standing in for GitHub, with main = the sandbox's tree."""
    bare = tmp_path / "remote.git"
    git("init", "--bare", "-q", "-b", "main", str(bare))
    w = tmp_path / "seed"
    git("init", "-q", "-b", "main", str(w))
    git("config", "core.autocrlf", "false", cwd=w)
    for p, b in BASE.items():
        f = w / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
    git("add", "-A", cwd=w)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base", cwd=w)
    git("push", "-q", str(bare), "main", cwd=w)
    base = git("rev-parse", "HEAD", cwd=w).strip()
    return str(bare), base


class FakeGitHub:
    def __init__(self, fail: str | None = None) -> None:
        self.prs, self.fail = [], fail

    def __call__(self, repo, head, base, title, body):
        if self.fail:
            raise PS.PrError(self.fail)
        self.prs.append({"repo": repo, "head": head, "base": base, "title": title, "body": body})
        n = len(self.prs)
        return {"url": f"https://github.com/{repo}/pull/{n}", "number": n}


def handler(tmp_path, remote, gh=None, issue=1):
    bare, base = remote
    gh = gh or FakeGitHub()
    h = PS.Handler(str(tmp_path / "cache.git"), {"r1": {"issue": issue, "base": base, "slug": "issue-1-r1"}},
                   remote=bare, create_pr=gh, cred=[])
    assert h.git.fetch_base() == base
    return h, gh


def req(files, title="Fix total()", body="The loop skipped the last price.", run="r1"):
    return {"run_id": run, "title": title, "body": body,
            "files": {p: base64.b64encode(b).decode() for p, b in files.items()}}


FIXED = dict(BASE)
FIXED["inventory/money.py"] = BASE["inventory/money.py"].replace(b"range(len(prices) - 1)", b"range(len(prices))")


def test_opens_one_pull_request_with_exactly_the_changed_file(tmp_path, remote):
    bare, base = remote
    h, gh = handler(tmp_path, remote)
    ans = h.handle(req(FIXED))
    assert ans["ok"], ans
    assert ans["text"] == "https://github.com/divagr18/fxputer-sandbox/pull/1"
    assert ans["branch"] == "fx/issue-1-r1" and ans["changed"] == ["inventory/money.py"]
    # the commit on the remote: on the base, fx's title, fxputer's identity, only that file changed
    log = git("--git-dir", bare, "log", "-1", "--format=%P|%an|%ae|%cn|%ce|%B", "refs/heads/fx/issue-1-r1")
    parent, an, ae, cn, ce, msg = log.split("|", 5)
    assert parent == base and msg.strip() == "Fix total()"
    assert (an, ae, cn, ce) == ("fxputer", PS.AUTHOR[1], "fxputer", PS.AUTHOR[1])
    diff = git("--git-dir", bare, "diff", "--name-only", base, "refs/heads/fx/issue-1-r1").split()
    assert diff == ["inventory/money.py"]
    got = subprocess.run(["git", "--git-dir", bare, "show", "refs/heads/fx/issue-1-r1:inventory/money.py"],
                         capture_output=True).stdout
    assert got == FIXED["inventory/money.py"]
    # main is untouched
    assert git("--git-dir", bare, "rev-parse", "refs/heads/main").strip() == base
    pr = gh.prs[0]
    assert pr["repo"] == PS.REPO and pr["head"] == "fx/issue-1-r1" and pr["base"] == "main"
    assert pr["title"] == "Fix total()"
    assert pr["body"].startswith("The loop skipped the last price.\n\n---\nThis pull request was opened by [fx](")
    assert "Added by the host" not in pr["body"]
    assert "The title, the text above this line and the change are fx's." in pr["body"]
    assert "Run `r1`." in pr["body"] and pr["body"].endswith("Fixes #1")


def test_one_pull_request_per_run(tmp_path, remote):
    h, gh = handler(tmp_path, remote)
    assert h.handle(req(FIXED))["ok"]
    again = h.handle(req(FIXED, title="Again"))
    assert not again["ok"] and "already opened" in again["text"] and "pull/1" in again["text"]
    assert len(gh.prs) == 1


def test_no_changes_is_an_error_for_fx(tmp_path, remote):
    h, gh = handler(tmp_path, remote)
    ans = h.handle(req(BASE))
    assert not ans["ok"] and "no changes" in ans["text"] and not gh.prs


def test_unregistered_run_and_bad_paths_are_refused(tmp_path, remote):
    h, gh = handler(tmp_path, remote)
    assert "not registered" in h.handle(req(FIXED, run="other"))["text"]
    for bad in ("../x.py", ".git/config", "/etc/passwd", "a/../../b", "inventory\\x.py"):
        ans = h.handle(req({**FIXED, bad: b"x"}))
        assert not ans["ok"] and "refusing the path" in ans["text"], bad
    assert not gh.prs


def test_only_new_fx_branches_are_pushed(tmp_path, remote):
    bare, base = remote
    h, _ = handler(tmp_path, remote)
    c = h.git.commit(base, {"x.txt": b"x"}, "t")
    for br in ("main", "fx/../main", "feature/x", "fx/"):
        with pytest.raises(PS.PrError, match="only fx/"):
            h.git.push_new_branch(c, br)
    h.git.push_new_branch(c, "fx/one")
    c2 = h.git.commit(base, {"y.txt": b"y"}, "t2")
    with pytest.raises(PS.PrError, match="already exists"):
        h.git.push_new_branch(c2, "fx/one")
    assert git("--git-dir", bare, "rev-parse", "refs/heads/fx/one").strip() == c
    assert git("--git-dir", bare, "rev-parse", "refs/heads/main").strip() == base


def test_new_files_deletions_and_exact_bytes(tmp_path, remote):
    bare, base = remote
    h, gh = handler(tmp_path, remote)
    store = dict(FIXED)
    store["inventory/text.py"] = b"def slugify(t):\r\n    return t\n"  # bytes as fx wrote them, CRLF kept
    del store["tests/__init__.py"]
    ans = h.handle(req(store))
    assert ans["ok"], ans
    assert ans["changed"] == ["inventory/money.py", "inventory/text.py", "tests/__init__.py"]
    br = ans["branch"]
    got = subprocess.run(["git", "--git-dir", bare, "show", f"refs/heads/{br}:inventory/text.py"],
                         capture_output=True).stdout
    assert got == store["inventory/text.py"]
    names = git("--git-dir", bare, "ls-tree", "-r", "--name-only", f"refs/heads/{br}").split()
    assert "tests/__init__.py" not in names


def test_github_error_text_goes_back_to_fx(tmp_path, remote):
    h, _ = handler(tmp_path, remote, gh=FakeGitHub(fail="GitHub: Validation Failed (A pull request already exists)"))
    ans = h.handle(req(FIXED))
    assert ans == {"ok": False, "text": "GitHub: Validation Failed (A pull request already exists)"}
    # an attempt that failed at GitHub does not count as the run's pull request; the cap on attempts holds
    h.create_pr = FakeGitHub()
    ok = h.handle(req(FIXED))
    assert ok["ok"] and ok["branch"] == "fx/issue-1-r1-2"


def test_footer_does_not_repeat_an_issue_link(tmp_path, remote):
    h, gh = handler(tmp_path, remote)
    h.handle(req(FIXED, body="Fixes #1: the loop skipped the last price."))
    assert gh.prs[0]["body"].count("#1") == 1
    assert "Fixes #3" in PS.footer("r", 3, "nothing") and "Fixes" not in PS.footer("r", None, "x")


def test_title_and_body_checks(tmp_path, remote):
    h, gh = handler(tmp_path, remote)
    assert "string title" in h.handle({"run_id": "r1", "title": 3, "body": "", "files": {}})["text"]
    assert "empty" in h.handle(req(FIXED, title="  \n "))["text"]
    ans = h.handle(req(FIXED, title="Fix\ntotal()"))
    assert ans["ok"] and gh.prs[0]["title"] == "Fix total()"


def test_client_through_the_service(tmp_path, remote):
    """PrClient (the box) -> HTTP on 127.0.0.1 -> Handler (the laptop); the store is read from disk."""
    h, gh = handler(tmp_path, remote)
    srv = PS.serve(h, 0)
    port = srv.server_address[1]
    assert srv.server_address[0] == "127.0.0.1"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        root = tmp_path / "store"
        for p, b in FIXED.items():
            f = root / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b)
        (root / "inventory" / "__pycache__").mkdir()
        (root / "inventory" / "__pycache__" / "money.cpython-312.pyc").write_bytes(b"\0")
        c = PR.PrClient(f"http://127.0.0.1:{port}", "r1")
        text, err = c(str(root), {"title": "Fix total()", "body": "b"})
        assert (text, err) == ("https://github.com/divagr18/fxputer-sandbox/pull/1", False)
        assert c.calls[0]["store"] == sorted(FIXED)
        text, err = c(str(root), {"title": "x"})
        assert err and "string title" in text
        down = PR.PrClient("http://127.0.0.1:9", "r1", timeout=2)
        text, err = down(str(root), {"title": "t", "body": "b"})
        assert err and text.startswith("could not reach the pull request service")
    finally:
        srv.shutdown()


class Inner:
    root = "/x"

    def __init__(self):
        self.seen = []

    def files(self):
        return []

    def call(self, name, a):
        self.seen.append(name)
        return "inner", False


def test_lockstep_hosts_replay_the_chip_hosts_answer():
    calls = []

    def client(root, args):
        calls.append(args)
        return "https://github.com/x/pull/7", False

    chip = PR.PrTools(Inner(), client)
    shadow = PR.ReplayPr(Inner(), chip, "rv32emu")
    assert chip.call("read_file", {"path": "a"}) == ("inner", False)
    a = {"title": "t", "body": "b"}
    r = chip.call(PR.NAME, a)
    assert shadow.call(PR.NAME, dict(a)) == r and len(calls) == 1
    with pytest.raises(RuntimeError, match="did not make"):
        shadow.call(PR.NAME, a)
    other = PR.ReplayPr(Inner(), chip, "orig")
    with pytest.raises(RuntimeError, match="arguments differ"):
        other.call(PR.NAME, {"title": "t", "body": "B"})


def test_store_diff():
    base = {"a": b"1", "b": b"2"}
    assert PR.changes(base, {"a": b"1", "b": b"3", "c": b"4"}) == {"b": b"3", "c": b"4"}
    assert PR.changes(base, {"a": b"1"}) == {"b": None}


def test_prompt_wording():
    p = PR.issue_prompt(4, " Low stock ", "Items should count as low stock...\n")
    assert p == ("Fix this GitHub issue in the repository.\n\nIssue #4: Low stock\n\n"
                 "Items should count as low stock...\n\n"
                 "When your change is complete, open a pull request for it with the open_pull_request tool.")


def test_initialize_declares_the_tool():
    """The tool reaches fx the way fx's own source reads host tools: clientCapabilities.libfx.tools in
    ACP initialize (fx/src/acp/server.zig parseInitializeRequest; host_tool_runtime.zig)."""
    H = pytest.importorskip("host", reason="host.py needs wasmtime")
    old = H.TOOLS
    try:
        H.TOOLS = list(old) + [PR.TOOL]
        h = H.Host.__new__(H.Host)
        h._init_common("/tmp/x/w", 1, None)
        h.mark = lambda *a: None
        h.idle()
        msg = json.loads(bytes(h.stdin).decode())
        assert msg["method"] == "initialize"
        tools = msg["params"]["clientCapabilities"]["libfx"]["tools"]
        t = next(x for x in tools if x["name"] == "open_pull_request")
        assert set(t) == {"name", "description", "inputSchema", "writes"}
        assert t["inputSchema"]["required"] == ["title", "body"]
        assert len(t["description"]) < 1000
        assert [x["name"] for x in tools][:5] == [x["name"] for x in old][:5]
    finally:
        H.TOOLS = old


# ---- on the chip: a whole fx session (GPU box with fx's ELF) ---------------------------------------
ELF = os.environ.get("FXPR_ELF")  # row24/fx-fastcopy.elf
DLL = os.environ.get("FXPR_DLL")  # rv32emu.so
MODEL = os.environ.get("FXPR_MODEL")  # Qwen3.5-4B's directory (the tokenizer and template)


@pytest.mark.skipif(not (ELF and DLL and MODEL), reason="needs FXPR_ELF, FXPR_DLL, FXPR_MODEL (a GPU box)")
def test_the_call_arrives_as_host_call_tokens(tmp_path):
    """fx on Executor v4 with a scripted model (not Qwen) that ends with open_pull_request. The name
    and the arguments the handler gets are the bytes the chip printed through the read sled, the
    answer is in fx's memory as the chip prints it, rv32emu replays it, every chip token audits."""
    import live_chip as LC

    from chipformer.rv import fxlive as FL

    job = {"run_id": "t", "issue": 1, "task": "t1_bug", "prompt": "Fix total(). Then open a pull request.",
           "pr_url": "http://127.0.0.1:9", "files": {p: base64.b64encode(b).decode() for p, b in BASE.items()}}
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    a = LC.parse([DLL, ELF, os.path.join(os.path.dirname(ELF), "imports.json"), MODEL, str(tmp_path / "out"),
                  "--job", str(jp), "--backend", "scripted-pr", "--full-regs", "--tmp", str(tmp_path),
                  "--frames", "4096"])
    seen = {}
    url = "https://github.com/divagr18/fxputer-sandbox/pull/99"

    def client(root, args):
        seen["args"] = args
        seen["store"] = PR.store_files(root)
        return url, False

    client.calls = []
    a.pr_client = client
    words = []  # every word the chip printed through the read sled
    rw = FL.LiveCPU.read_words

    def read_words(self, addr, n, kind="io"):
        v = rw(self, addr, n, kind)
        words.append((addr, v))
        return v

    tool = {}
    host_call = LC.ChipHost.x_fx_host_tool_call

    def x_fx_host_tool_call(self, np_, nl, ap, al, op, ocap, sp, cp, cl):
        n0 = len(words)
        r = host_call(self, np_, nl, ap, al, op, ocap, sp, cp, cl)
        name = self.cpu.read(np_, nl)
        if name == PR.NAME.encode():
            tool.update(name_at=np_, args=(ap, al), out=(op, r), words=words[n0:],
                        printed=self.cpu.read_words(op & ~3, (((op + r + 3) & ~3) - (op & ~3)) // 4))
        return r

    FL.LiveCPU.read_words = read_words
    LC.ChipHost.x_fx_host_tool_call = x_fx_host_tool_call
    try:
        chip, template, backend = LC.setup(a)
        r = LC.session(a, chip, template, backend, "t1_bug", 1, a.out, print)
    finally:
        FL.LiveCPU.read_words = rw
        LC.ChipHost.x_fx_host_tool_call = host_call
    assert r["stopped"] is None and r["solved"], r.get("stopped")
    assert "open_pull_request" in r["tools_declared"]
    assert seen["args"] == {"title": "Scripted: t1_bug", "body": "A scripted test of the tool.\nLine 2."}
    assert seen["store"]["inventory/money.py"] != BASE["inventory/money.py"]

    def mem(addr, n, ws):
        """addr..addr+n from the words the chip printed during the call."""
        m = {}
        for a0, vs in ws:
            for i, v in enumerate(vs):
                m[a0 + 4 * i] = v
        out = b"".join(m[w].to_bytes(4, "little") for w in range(addr & ~3, addr + n + 3 & ~3, 4))
        return out[addr & 3:(addr & 3) + n]

    # the handler's name and arguments are bytes the chip printed as read-sled value records
    assert mem(tool["name_at"], len(PR.NAME), tool["words"]) == PR.NAME.encode()
    raw_args = mem(*tool["args"], tool["words"])
    assert json.loads(raw_args) == seen["args"]
    # the answer went in through the write sled: the chip prints it back from fx's memory
    op, n = tool["out"]
    assert n == len(url)
    assert mem(op, n, [(op & ~3, tool["printed"])]) == url.encode()
    assert r["pr"]["replayed"] == {"rv32emu": 1}
    assert r["emu"]["requests_identical"] == r["steps"] and r["emu"]["host_io_equal"]
    assert r["audit"]["wrong"] is None and r["audit"]["chip_tokens"] > 1_000_000
