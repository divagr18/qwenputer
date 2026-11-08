"""docs/fx/units-file.md: pr.md's issue flow on the box with no GitHub: pr_server's Handler (the same
code the laptop runs) against a bare git repository standing in for the sandbox, and a fake PR call in
place of `gh api` (as tests/test_fx_pr.py does). Nothing is pushed anywhere and no PR is opened.

    python fx/fxio/pr_local.py OUT_DIR README.md TASK SEED LIVE_CHIP_ARGS..

The job is pr_driver's for that task's issue: pr_tool.PROMPT with the issue's title (pr_driver.ISSUE_TITLES)
and the task's prompt as its body, the repository = P0c's files with the sandbox's README. live_chip runs
it (LIVE_CHIP_ARGS: dll elf imports model out ... without --job); afterwards the fx/ branch on the bare
repository is checked out and the task's hidden check runs on it.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path[:0] = [HERE, FX, os.path.join(FX, "p0c")]

import pr_driver as PD  # noqa: E402
import pr_server as PS  # noqa: E402
import pr_tool as PR  # noqa: E402
import tasks as TK  # noqa: E402

PORT = 8771


def git(*a, cwd=None) -> str:
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode()


class FakeGitHub:
    """`gh api POST /pulls`, recorded and never sent."""

    def __init__(self) -> None:
        self.prs = []

    def __call__(self, repo, head, base, title, body):
        self.prs.append({"repo": repo, "head": head, "base": base, "title": title, "body": body})
        return {"url": f"https://example.invalid/{repo}/pull/{len(self.prs)} (not opened: pr_local)",
                "number": len(self.prs)}


def main() -> None:
    out, readme, task, seed = sys.argv[1:5]
    live_args = sys.argv[5:]
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    files = {p: s.encode() for p, s in TK.COMMON.items()}
    files["README.md"] = open(readme, "rb").read()
    bare, w = os.path.join(out, "remote.git"), os.path.join(out, "seed")
    git("init", "--bare", "-q", "-b", "main", bare)
    git("init", "-q", "-b", "main", w)
    git("config", "core.autocrlf", "false", cwd=w)
    for p, b in files.items():
        f = os.path.join(w, p)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(b)
    git("add", "-A", cwd=w)
    git("-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "base", cwd=w)
    git("push", "-q", bare, "main", cwd=w)
    base = git("rev-parse", "HEAD", cwd=w).strip()
    issue = 1
    run_id, slug = "local1", "issue-1-local"
    gh = FakeGitHub()
    h = PS.Handler(os.path.join(out, "cache.git"), {run_id: {"issue": issue, "base": base, "slug": slug}},
                   remote=bare, create_pr=gh, cred=[], log_path=os.path.join(out, "calls.json"))
    assert h.git.fetch_base() == base
    srv = PS.serve(h, PORT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    job = {"run_id": run_id, "issue": issue, "issue_title": PD.ISSUE_TITLES[task], "task": task,
           "repo": PS.REPO + " (a local bare copy)", "base_sha": base,
           "prompt": PR.issue_prompt(issue, PD.ISSUE_TITLES[task], TK.TASKS[task]["prompt"]),
           "pr_url": f"http://127.0.0.1:{PORT}", "seed": int(seed),
           "files": {p: base64.b64encode(b).decode() for p, b in files.items()}}
    jp = os.path.join(out, "job.json")
    json.dump(job, open(jp, "w"))
    rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, "live_chip.py"), *live_args, "--job", jp,
                          "--reps", str(seed)])
    srv.shutdown()
    res = {"live_chip_rc": rc, "calls": h.calls, "fake_prs": gh.prs}
    heads = [ln.split()[1] for ln in git("ls-remote", bare, cwd=out).splitlines() if "refs/heads/fx/" in ln]
    res["branches"] = heads
    if heads:
        d = os.path.join(out, "branch")
        git("clone", "-q", "-b", heads[0][len("refs/heads/"):], bare, d)
        r = subprocess.run([sys.executable, "-c", TK.TASKS[task]["check"]], cwd=d, capture_output=True, text=True)
        res["solved_on_branch"] = r.returncode == 0
        res["branch_diff"] = git("diff", "--stat", "origin/main", "HEAD", cwd=d)
        res["main_unchanged"] = git("rev-parse", "origin/main", cwd=d).strip() == base
    json.dump(res, open(os.path.join(out, "pr_local.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "calls"}), flush=True)


if __name__ == "__main__":
    main()
