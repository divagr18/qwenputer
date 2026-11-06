# P0c: can base Qwen3.5-4B drive native fx?

Setup: unmodified fx v0.0.16 (linux x86_64 release binary, sha256 checked), base Qwen3.5-4B served by
vLLM 0.33 (`--tool-call-parser qwen3_coder --reasoning-parser qwen3`, max len 65,536) on one A10G,
`fx ask --json --no-save`. Five tasks on a tiny Python repo, each checked by a hidden script the agent
never sees (`fx/p0c/tasks.py`), 3 reps each, two conditions:

- **full:** every fx tool, `--full-access`.
- **noshell:** `permission.bash."*" = "deny"` (auto mode): the shell is offered but every call errors,
  so only the five file tools work, as in the in-weights design.

| | full | noshell |
|---|---|---|
| Solved | **15 / 15** | **15 / 15** |
| Seconds per task (mean / max) | 20.1 / 34 | 18.6 / 35 |
| Model steps per task (mean / max) | 7.8 / 18 | 6.3 / 14 |
| Output tokens per task (mean) | 946 | 870 |
| Tool calls | read 42, edit 23, grep 22, shell 16, glob 10, write 4 | read 36, edit 26, grep 21, shell 7 (all denied), write 3, glob 2 |
| Failed calls | edit 2, shell 5, glob 1 | edit 2, shell 7, grep 1 |

All failed calls were followed by a recovery. All numbers measured.

**Compatibility.** fx sends 7 system messages per request (about 9 KB in total) and exposes 13 tools.
Qwen3.5's chat template rejects a system message that isn't first, so `fx/p0c/proxy.py` merges the
system messages into one at the start without changing the text. The model saw exactly fx's prompt
otherwise.

**Caveat.** These are small tasks. The result shows that Qwen3.5-4B can drive fx's loop and file
tools, not that it can handle large repos.

Raw per-run results: `p0c-summary.jsonl`. Scripts: `fx/p0c/` (`run.sh`, `tasks.py`, `proxy.py`, `stats.py`).
