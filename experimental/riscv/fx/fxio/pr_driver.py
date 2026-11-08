"""Issue in, pull request out (docs/fx/pr.md). Runs on the laptop that holds the GitHub login.

    python pr_driver.py issues                      # one issue on the sandbox per P0c task (once)
    python pr_driver.py run N [--seed S] [--box q35-batch] [--port P] [--out DIR]
    python pr_driver.py check RUN_SUMMARY.json      # the hidden check on the PR branch, again

`run` fetches issue N and main's tree from GitHub, writes the job (the prompt, the repository at that
commit, the run id), starts pr_server.py on 127.0.0.1, opens `ssh -R` so the box reaches it on its own
127.0.0.1, and runs fx on the chip (live_chip.py --job) on the box under nohup. When the run ends it
copies the run file back, closes the tunnel and the handler, and runs the task's hidden check on the
pull request branch as GitHub has it.

The prompt is pr_tool.PROMPT: the issue's number, title and body, and one sentence asking fx to open a
pull request with the open_pull_request tool when the change is complete. Nothing else is added.
Credentials stay here: the box only ever sees the tunnel's port.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FX, "p0c"))
import pr_server as PS  # noqa: E402
import pr_tool as PR  # noqa: E402
import tasks as TK  # noqa: E402

REPO = PS.REPO
BOX_DIR = "/opt/dlami/nvme/pr"
# the live configuration of docs/fx/one.md (4,096 frames, the 16 MB window, all registers printed), with the
# circuit lifted from a chip file. No units: the 13-unit circuit has weights outside bf16 (pr.md), so it has no file.
BOX_RUN = ("cd {d} && . venv/bin/activate && export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 "
           "TMPDIR={d}/tmp && M=$(python -c 'from huggingface_hub import snapshot_download as s; "
           "print(s(\"Qwen/Qwen3.5-4B\"))') && mkdir -p out/pr && "
           "{{ setsid nohup sh -c 'python -u fx/fxio/live_chip.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json "
           "\"$0\" out/pr --job jobs/{run}.json --reps {seed} --full-regs --tmp tmp --frames 4096 {units} "
           "{chip} > out/pr/{run}.log 2>&1; echo $? > out/pr/{run}.done' \"$M\" < /dev/null > /dev/null 2>&1 & }} && echo launched")

ISSUE_TITLES = {
    "t1_bug": "total() in inventory/money.py gives the wrong sum",
    "t2_rename": "Rename Store.get_count to count",
    "t3_feature": "Add slugify(text) in a new inventory/text.py",
    "t4_config": "Low stock should mean fewer than 10 units, not 5",
    "t5_parse": "parse_line breaks on surrounding whitespace",
}


def gh(*args, inp: dict | None = None) -> dict | list:
    r = subprocess.run(["gh", "api", *args] + (["--input", "-"] if inp is not None else []),
                       input=json.dumps(inp).encode() if inp is not None else None, capture_output=True)
    if r.returncode:
        raise SystemExit(f"gh api {args[0]}: {r.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return json.loads(r.stdout or b"null")


def task_of(body: str) -> str | None:
    """The P0c task whose prompt is this issue's text (its hidden check then applies)."""
    return next((t for t, v in TK.TASKS.items() if v["prompt"].strip() == body.strip()), None)


def cmd_issues(a) -> None:
    have = {i["title"]: i["number"] for i in gh(f"repos/{REPO}/issues?state=all&per_page=100")
            if "pull_request" not in i}
    out = {}
    for t, title in ISSUE_TITLES.items():
        if title in have:
            out[t] = have[title]
            continue
        i = gh("-X", "POST", f"repos/{REPO}/issues", inp={"title": title, "body": TK.TASKS[t]["prompt"]})
        out[t] = i["number"]
    print(json.dumps(out))


def wait_line(p: subprocess.Popen, key: str, timeout: float = 60) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline().decode()
        if key in line:
            return line
        if p.poll() is not None:
            raise SystemExit(f"process ended: {line}")
    raise SystemExit(f"no {key!r} within {timeout} s")


def ssh(box: str, cmd: str, check: bool = True, timeout: float = 120) -> str:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", box, cmd], capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise SystemExit(f"ssh {cmd[:60]}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    return r.stdout.decode("utf-8", "replace")


def hidden_check(cache: str, branch: str, task: str | None, store: dict | None) -> dict:
    """The task's hidden check on the pull request branch, fetched from GitHub."""
    g = PS.Git(cache, f"https://github.com/{REPO}.git", PS.GH_CRED)
    g.run("fetch", "-q", g.remote, f"+refs/heads/{branch}:refs/remotes/pr/{branch}")
    head = g.run("rev-parse", f"refs/remotes/pr/{branch}").decode().strip()
    files = g.files(head)
    out = {"branch": branch, "head": head, "files": len(files)}
    if store is not None:
        out["branch_equals_fx_store"] = {p: b.decode("utf-8", "replace") for p, b in files.items()} == store
    if not task:
        out["solved"] = None
        out["why"] = "no hidden check: the issue is not one of the P0c tasks"
        return out
    d = tempfile.mkdtemp(prefix="fxpr-check-")
    for p, b in files.items():
        q = os.path.join(d, p)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        open(q, "wb").write(b)
    r = subprocess.run([sys.executable, "-c", TK.TASKS[task]["check"]], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out["solved"] = r.returncode == 0
    out["why"] = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    return out


def cmd_run(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    cache = a.cache
    if a.resume:  # a run already launched on the box: handler, tunnel, wait, check
        jp = os.path.join(a.out, f"job_{a.resume}.json")
        job = json.load(open(jp))
        run_id, task, base, a.seed = job["run_id"], job["task"], job["base_sha"], job["seed"]
        slug = "issue-" + run_id[1:]
        assert job["issue"] == a.issue, "the job is another issue's"
    else:
        issue = gh(f"repos/{REPO}/issues/{a.issue}")
        if "pull_request" in issue or issue["state"] != "open":
            raise SystemExit(f"#{a.issue} is not an open issue")
        task = task_of(issue["body"] or "")
        g = PS.Git(cache, f"https://github.com/{REPO}.git", PS.GH_CRED)
        base = g.fetch_base()
        files = g.files(base)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id, slug = f"i{a.issue}-{stamp}", f"issue-{a.issue}-{stamp}"
        job = {"run_id": run_id, "issue": a.issue, "issue_title": issue["title"], "task": task, "repo": REPO,
               "base_sha": base, "prompt": PR.issue_prompt(a.issue, issue["title"], issue["body"] or ""),
               "pr_url": f"http://127.0.0.1:{a.port}", "seed": a.seed,
               "files": {p: base64.b64encode(b).decode() for p, b in files.items()}}
        jp = os.path.join(a.out, f"job_{run_id}.json")
        json.dump(job, open(jp, "w"))
    calls = os.path.join(a.out, f"calls_{run_id}.json")
    print(json.dumps({"run": run_id, "issue": a.issue, "task": task, "base": base}), flush=True)
    srv = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "pr_server.py"), "serve", "--port", str(a.port),
                            "--run", run_id, "--slug", slug, "--issue", str(a.issue), "--base", base,
                            "--log", calls, "--cache", cache], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tun = None
    try:
        wait_line(srv, "serving")
        tun = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                "-R", f"127.0.0.1:{a.port}:127.0.0.1:{a.port}", a.box],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(30):
            time.sleep(2)
            if tun.poll() is not None:
                raise SystemExit(f"tunnel failed: {tun.stderr.read().decode()[-300:]}")
            if '"ok": true' in ssh(a.box, f"curl -s -m 5 http://127.0.0.1:{a.port}/health", check=False):
                break
        else:
            raise SystemExit("the tunnel did not come up")
        print(json.dumps({"tunnel": f"box 127.0.0.1:{a.port} -> laptop 127.0.0.1:{a.port}"}), flush=True)
        if not a.resume:
            ssh(a.box, f"mkdir -p {BOX_DIR}/jobs")
            subprocess.run(["scp", "-q", jp, f"{a.box}:{BOX_DIR}/jobs/{run_id}.json"], check=True)
            chip = f"--chip-file {shlex.quote(a.chip_file)}" if a.chip_file else ""
            # (only the job is backgrounded, with its own fds, so this ssh returns at once)
            print(ssh(a.box, BOX_RUN.format(d=BOX_DIR, run=run_id, seed=a.seed, chip=chip,
                                            units=f"--units {a.units}" if a.units else "")).strip(), flush=True)
        t0 = time.time()
        while True:
            time.sleep(30)
            if tun.poll() is not None:  # the tunnel dropped: bring it back while fx runs
                tun = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                        "-R", f"127.0.0.1:{a.port}:127.0.0.1:{a.port}", a.box],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(json.dumps({"tunnel": "reopened", "at_s": round(time.time() - t0)}), flush=True)
            st = ssh(a.box, f"cat {BOX_DIR}/out/pr/{run_id}.done 2>/dev/null; tail -c 300 {BOX_DIR}/out/pr/{run_id}.log",
                     check=False)
            first = st.split("\n", 1)[0].strip()
            if first.isdigit():
                print(json.dumps({"box_exit": int(first), "wall_s": round(time.time() - t0)}), flush=True)
                break
            if time.time() - t0 > a.timeout:
                raise SystemExit("the run did not finish in time; it is still running on the box")
    finally:
        if tun is not None and tun.poll() is None:
            tun.terminate()
        srv.terminate()
    local = os.path.join(a.out, f"pr_{run_id}.json")
    subprocess.run(["scp", "-q", f"{a.box}:{BOX_DIR}/out/pr/pr_{run_id}.json", local], check=False)
    subprocess.run(["scp", "-q", f"{a.box}:{BOX_DIR}/out/pr/{run_id}.log", os.path.join(a.out, f"{run_id}.log")],
                   check=False)
    summary = summarize(a, run_id, task, local, calls, cache)
    sp = os.path.join(a.out, f"summary_{run_id}.json")
    json.dump(summary, open(sp, "w"), indent=1)
    with open(os.path.join(a.out, "runs.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


def summarize(a, run_id, task, local, calls_path, cache) -> dict:
    s = {"run_id": run_id, "issue": a.issue, "task": task, "seed": a.seed}
    calls = json.load(open(calls_path)) if os.path.exists(calls_path) else []
    s["handler_calls"] = [{k: c.get(k) for k in ("title", "changes", "branch", "commit", "seconds")}
                          | {"ok": c["answer"]["ok"], "text": c["answer"]["text"]} for c in calls]
    opened = [c for c in calls if c["answer"]["ok"]]
    s["pr_url"] = opened[0]["answer"]["url"] if opened else None
    if not os.path.exists(local):
        s["run_file"] = None
        return s
    r = json.load(open(local))
    s.update(stopped=r.get("stopped"), steps=r.get("steps"), store_check=r.get("solved"),
             chip_s=round(sum(v for k, v in r["chip"]["seconds"].items() if k != "text"), 1),
             qwen_s=r["qwen"]["link_s"], chip_tokens=r["chip"]["tokens"], instructions=r["chip"]["instructions"],
             audit={k: r.get("audit", {}).get(k) for k in ("wrong", "chip_tokens", "forced_tokens", "memory_diff",
                                                           "memory_words_compared")},
             emu=r.get("emu"), end=r.get("end"), chip_file=r.get("chip_file"),
             tools_declared=r.get("tools_declared"),
             tool_calls=[x.get("name") if isinstance(x, dict) else x for x in r.get("pr", {}).get("tool_calls_seen", [])],
             pr_tool_calls=r.get("pr", {}).get("calls"), replayed=r.get("pr", {}).get("replayed"),
             elf_sha256=r.get("elf_sha256"))
    if opened:
        s["hidden_check_on_pr_branch"] = hidden_check(cache, opened[0]["answer"]["branch"], task, r.get("final_repo"))
    return s


def cmd_check(a) -> None:
    s = json.load(open(a.summary))
    br = next(c["branch"] for c in s["handler_calls"] if c["ok"])
    print(json.dumps(hidden_check(a.cache, br, s["task"], None)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("issues")
    r = sub.add_parser("run")
    r.add_argument("issue", type=int)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--box", default="q35-batch")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--units", default="", help="hot-code units (the chip file has none)")
    r.add_argument("--chip-file", default="chip/fx-live-f4096.safetensors",
                   help="on the box, relative to its run directory; '' builds the circuit in Python")
    r.add_argument("--timeout", type=float, default=4 * 3600)
    r.add_argument("--resume", default=None, help="a RUN_ID already launched: serve, tunnel, wait, check")
    r.add_argument("--out", default="pr_out")
    c = sub.add_parser("check")
    c.add_argument("summary")
    for p in (r, c):
        p.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "fxpr-cache.git"))
    a = ap.parse_args()
    {"issues": cmd_issues, "run": cmd_run, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
