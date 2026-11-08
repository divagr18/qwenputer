# fx opens the pull request: issue in, PR out, on the in-weights CPU

Status 2026-11-08, branch `fx-pr` (from `fx-int` 048c114). NVIDIA L4 (q35-batch), Executor v4 running
the circuit lifted from a chip file, real Qwen3.5-4B. Code: `fx/fxio/pr_tool.py` (the tool, the box
side), `fx/fxio/pr_server.py` (the handler, the laptop), `fx/fxio/pr_driver.py` (issue in, PR out),
`fx/fxio/report_pr.py`, hooks in `fx/fxio/live_chip.py`, and `tests/test_fx_pr.py`. Raw numbers:
`pr.json`. **M** = measured, **E** = estimated. Everything here is M unless it says E.

## Result

**Yes. fx opened all five pull requests itself: one per issue, t1-t5, on
[divagr18/fxputer-sandbox](https://github.com/divagr18/fxputer-sandbox), running on Executor v4 with
real Qwen3.5-4B, and every PR branch passes its task's hidden check.**

| issue | PR | solved on the PR branch | fx's title |
|---|---|---|---|
| [#1](https://github.com/divagr18/fxputer-sandbox/issues/1) total() gives the wrong sum | [#6](https://github.com/divagr18/fxputer-sandbox/pull/6) | yes | Fix total() function to include all prices in the sum |
| [#2](https://github.com/divagr18/fxputer-sandbox/issues/2) rename get_count | [#7](https://github.com/divagr18/fxputer-sandbox/pull/7) | yes | Rename Store.get_count to count |
| [#3](https://github.com/divagr18/fxputer-sandbox/issues/3) slugify | [#8](https://github.com/divagr18/fxputer-sandbox/pull/8) | yes | Add slugify(text) function to inventory/text.py |
| [#4](https://github.com/divagr18/fxputer-sandbox/issues/4) low stock 10 | [#9](https://github.com/divagr18/fxputer-sandbox/pull/9) | yes | Fix low stock threshold: fewer than 10 units (not 5) |
| [#5](https://github.com/divagr18/fxputer-sandbox/issues/5) parse_line whitespace | [#10](https://github.com/divagr18/fxputer-sandbox/pull/10) | yes | Fix parse_line to trim surrounding whitespace on names |

- **The decision, the change, the title and the body are fx's**, driven by Qwen; the host did the git
  and GitHub mechanics (Who does what). The call reached the host as host-call tokens through the
  live I/O path, like fx's other host tools.
- **0 wrong chip tokens** over 65,724,917 audited chip tokens; rv32emu identical in lockstep on every
  run (requests, host I/O hash chain, final repository, all registers, every changed page).
- **The circuit came from a chip file**, lifted from its weights at load: a 161-layer file built for
  this configuration, not the shipped 51 MB one, and without units (Running from the chip file).
- **fx called the tool once and correctly on issues 1-4.** On issue 5 it called it three times: the
  first call was lost because I restarted the handler while it was serving it (my error, below), the
  second was refused because that first call had already pushed its branch, and fx changed its title
  and called again, which opened #10. A model-side recovery from a host-side fault, not a model error.
- One run per issue, seed 1, on 5 tiny tasks: it shows the path works, not a success rate.

## The tool

One tool, `open_pull_request(title, body)`. Its whole declaration, as Qwen sees it in fx's request:

```json
{"name": "open_pull_request",
 "description": "Open a pull request on GitHub with the changes you made to the repository's files. Call it once, when your change is complete. Returns the pull request's URL.",
 "inputSchema": {"type": "object", "properties": {
   "title": {"type": "string", "description": "A short title for the pull request"},
   "body": {"type": "string", "description": "What you changed and why"}},
   "required": ["title", "body"], "additionalProperties": false},
 "writes": true}
```

- **No paths, no diff, no branch name.** The change is whatever fx already wrote with its own
  `edit_file` / `write_file` into its file store. The model only says when, and what to call it.
- **It is fx's own mechanism for host tools.** The ACP client (the host) declares it in
  `initialize`, under `clientCapabilities.libfx.tools`, next to the five tools of `live.md`. fx-core
  parses it there (`fx/src/acp/server.zig`, `parseInitializeRequest`; `core/tooling/host_tool_runtime.zig`),
  advertises it to the model, and routes the model's call to the host import `fx_host_tool_call`
  (`acp/prompt.zig`, `hostToolProvider`, on wasm `core/hosts/js_host_tools.zig`).
- **On the chip the call is host-call tokens**, exactly like `grep_files`. Row 24's guest first offers
  the call to fx's own tool code (read, edit, write); for this name that returns -100 and the call
  leaves as an ecall (`fx/rv/gen_imports.py --native-tools`). The host reads the name and the JSON
  arguments out of fx's memory through the read sled (tokens the chip prints), and writes the result
  back through the write sled (tokens the host feeds), as `live.md` describes. A test checks this
  token by token (Tests).
- fx's source and binary are unchanged: `fx/check_src.sh`, and the ELF's sha256 `282e53d4…` is in
  every run file.

The prompt (`pr_tool.PROMPT`, disclosed in full). The issue's number, title and body are GitHub's;
the first and last sentences are ours:

```
Fix this GitHub issue in the repository.

Issue #{number}: {title}

{body}

When your change is complete, open a pull request for it with the open_pull_request tool.
```

fx-core's own system instructions are `live.md`'s, unchanged ("You are a coding agent working in a
small Python repository...").

## Who does what

This split is the basis of "fx raised the PR".

| | who | how |
|---|---|---|
| Deciding to open a pull request, and when | **fx, driven by Qwen** | Qwen's generation calls `open_pull_request`; fx's agent loop, running on the chip, parses the call and dispatches it |
| The change | **fx, driven by Qwen** | fx's own `edit_file` / `write_file` code, on the chip, into fx's file store |
| The title and the body | **fx, driven by Qwen** | the call's arguments, as fx wrote them into its memory |
| The call reaching the host | the chip | the `fx_host_tool_call` ecall: name and arguments printed by the chip through the read sled |
| Asking for a PR at all | the user prompt | one sentence, above. Without it fx would not know a tool for PRs is wanted |
| Diff, commit, branch, push, PR creation | the host | `pr_server.py` on the laptop: git plumbing and `gh api`, nothing else |
| The footer on the PR body, `Fixes #N` | the host | added by `pr_server.py` below a rule; it says the title, the text above it and the change are fx's (it is the host's own text, disclosed here) |
| The commit identity | the host | `fxputer <114850579+divagr18@users.noreply.github.com>`, author and committer |
| The tool result fx sees | GitHub, via the host | the PR's URL, or GitHub's error text |
| The hidden check | the driver | the task's check of `p0c/tasks.py`, on the PR branch as GitHub serves it |

What the host never does: choose files, edit content, write the title or the body's text, decide
whether to open a PR, retry a call fx did not make.

## The flow: issue in, PR out

```
laptop                                                    q35-batch (L4)
pr_driver.py run N
  gh: issue N, main's sha and tree
  job = prompt + the tree at main + run id      --scp-->  jobs/RUN.json
  pr_server.py on 127.0.0.1:8765
  ssh -R 127.0.0.1:8765:127.0.0.1:8765         <=======>  127.0.0.1:8765 (box side)
                                                          live_chip.py --job --chip-file
                                                            fx on Executor v4, Qwen beside it
                                                            ... fx calls open_pull_request
                                                            ecall -> read sled -> PrClient
  Handler.handle  <------------- HTTP POST (fx's store, title, body) -----------
    diff vs main's tree, commit, push fx/issue-N-RUN, gh api POST /pulls
  ------------------------------------- {"ok", "text": URL} ---------------->
                                                            write sled -> fx -> Qwen
  copy the run file back, close tunnel and handler
  hidden check on the PR branch (fetched from GitHub)
```

- **The file store is the repository at the issue's base commit.** The driver ships main's tree,
  byte for byte, as fx's store; the handler diffs fx's store against that commit's tree. The
  sandbox's README differs from P0c's (the sandbox's own text); every other file is P0c's `COMMON`.
- **The lockstep rv32emu hosts never reach GitHub.** They get the chip host's recorded answer
  (`pr_tool.ReplayPr`), and their call's arguments must equal the chip host's, byte for byte, or the
  run stops. So the comparison with rv32emu (requests, host I/O hash chain, final repository,
  registers, memory) still covers the PR call and its result.
- **Sustainable.** `pr_driver.py issues` makes the issues once; `pr_driver.py run N` is the whole
  loop for any open issue on the sandbox. An issue whose text is a P0c task's prompt gets that task's
  hidden check; any other issue runs the same way with no hidden check. `--resume RUN` reattaches the
  handler and the tunnel to a run already on the box.

## Credentials and limits

- **The token stays on the laptop.** `gh` is logged in there. git pushes with
  `credential.helper=!gh auth git-credential`, the API goes through `gh api`. The box sees a
  localhost port and nothing else; no token is in the job, the prompt, the model's context, the box
  or any log (the handler logs titles, bodies, changed paths, branches, commits and GitHub's answers).
- **Hard-coded in `pr_server.py`:** the one repository `divagr18/fxputer-sandbox`; pushes only to a
  new branch matching `fx/[A-Za-z0-9._-]+` (checked absent with `ls-remote` first, a plain push, no
  `+` refspec, no `--force`); PRs only into main; no merge, close or delete anywhere in the code.
- **Caps:** one PR per run; at most 3 attempts per run that reach a push; only the run id the driver
  registered. A restarted handler reloads its log and keeps the cap.
- The handler listens on 127.0.0.1 only. The box binds its end of the tunnel to 127.0.0.1 too.

## Runs

All runs: L4, `--frames 4096`, all registers printed at every ecall, no units, the circuit from
`chip/fx-live-f4096.safetensors`, base Qwen3.5-4B in HF transformers beside it (not `--text`: Qwen's
tokens did not go through the circuit's text bridge; the file has none). Temperature 1, seed 1.

| issue | task | PR | solved on PR branch | steps | fx's tool calls, in order | open_pull_request calls | chip s | Qwen s | chip tokens (fx code) | frames mapped |
|---|---|---|---|---|---|---|---|---|---|---|
| #1 | t1_bug | #6 | yes | 4 | read, edit, **PR** | 1, ok | 299.5 | 28.5 | 7,756,533 | 575 |
| #2 | t2_rename | #7 | yes | 8 | grep x2, read x3, edit x3, read x3, **PR** | 1, ok | 773.1 | 56.3 | 20,023,483 | 921 |
| #3 | t3_feature | #8 | yes | 4 | glob, write, **PR** | 1, ok | 371.0 | 48.0 | 9,627,602 | 617 |
| #4 | t4_config | #9 | yes | 7 | grep x2, read x2, edit, **PR** | 1, ok | 455.5 | 39.5 | 11,794,176 | 690 |
| #5 | t5_parse | #10 | yes | 7 | read, edit, read, **PR** x3 | 3: error (lost), error (branch exists), ok | 600.9 | 53.3 | 15,588,615 | 776 |

- Steps are model calls; the last of each run is fx's final answer, with no tool call. Each run's
  final answer reports the PR's URL back to the user (`pr.json`, `final_answer_tail`).
- Chip seconds are launches of fx's code and its I/O (26,066-26,091 tok/s); Qwen seconds are the
  whole model link. The handler took 3.96-4.44 s per successful call (commit, push, PR), inside the
  host's time. Plus 123-130 s at the start of each process for the lift and Executor v4's plan.
- Qwen's tool calls, every one, end exactly in `</tool_call>` with no NUL (`tool_calls_end_in_tool_call_tag`).
- Each PR changes only the files fx changed: #6 `money.py`, #7 `store.py`, `report.py`,
  `tests/test_store.py`, #8 the new `inventory/text.py`, #9 `config.py`, #10 `parse.py`.

**Issue 5, in detail (M).** fx's first call reached the handler process the driver had started. I
was restarting that process to load a changed footer (below) and killed it while it was serving the
call: the branch `fx/issue-5-20261108-081817` had been pushed, the PR not yet opened. fx got
`could not reach the pull request service: RemoteDisconnected`. It called again with the same title
and body; the restarted handler refused, `branch fx/issue-5-20261108-081817 already exists; not
overwriting it`. fx then changed its title ("... on names") and called a third time; the handler
pushed `fx/issue-5-20261108-081817-2` and opened #10. The lost first branch stays on the sandbox
with no PR (no deletes). The handler now recovers this case itself: a branch that already holds
exactly the same change (same tree, parent and author) is reused instead of refused
(`test_a_lost_answer_is_recovered_without_a_second_branch`). Qwen also rolled back one generation in
this run (Qwen's state, `live.md`), with no effect on the result.

## Checks, every run

| check | when | result, all 5 runs |
|---|---|---|
| PC, all 31 registers, instruction count at every ecall, against rv32emu | live | identical |
| Every request body to Qwen, byte for byte | live | identical: 30 requests |
| The PR call's arguments on rv32emu against the chip's | live | equal; rv32emu replayed 7 calls (1, 1, 1, 1, 3) |
| Host I/O hash chain | end | identical |
| Final repository | end | identical |
| All 31 registers at the end | end | 0 differ |
| Every page rv32emu changed, read through the chip | end | 3,268 pages, 0 words differ |
| Every chip token against the reference machine (offline) | after | 65,724,917 chip tokens, 0 wrong; then 313,803 memory words, 0 differ |
| The PR branch as GitHub serves it equals fx's final file store | after | yes, all 5 |
| The task's hidden check on the PR branch | after | 5 of 5 pass |
| The task's hidden check on fx's store, in the run | end | 5 of 5 pass |
| fx's source (`check_src.sh`) | | 717 files, 0 changed against 757bdb9 |
| fx's ELF | | sha256 `282e53d4...` in every run file |

## Running from the chip file

**Yes, the PR runs run from a chip file: the circuit is lifted from the file's weights, not built in
Python.** But not the shipped 51 MB file, and not with units:

| | |
|---|---|
| The file | `chip/fx-live-f4096.safetensors`, 132,640,220 B, sha256 `61aa878f…`, 20,657,783 terms, 161 chip layers |
| Built by | `live_chip.py --write-chip-file` (`chipbuild.standalone`), from this configuration's boot snapshot, once |
| Configuration | 4,096 frames, the 16 MB linear window (`one.md`), the I/O ROM in the ROM state, no units, no text bridge |
| Lift at load | 14.6-14.8 s; Executor v4's plan from the lifted nodes 108.8-115.4 s: 14 phases, 708,256 term slots, the same as the Python build in the same configuration (the token test's build: 14 phases, 708,256 slots) |
| Refused at load if | its boot state's hash is not this run's boot snapshot, its ELF hash differs, or its memory map is not this run's |

- **Why not the shipped file.** `chipfile.md`'s fx file is 59 layers: 1,024 frames, the 4 MB window,
  and fxrun's boot without the I/O ROM. The live path of `one.md` needs 4,096 frames and the 16 MB
  window (the chained session outgrew 4 MB), and the I/O ROM in its state. The circuit is a different
  size, so that file cannot run these sessions. The new file is the same format and loader.
- **Why no units.** `one.md`'s 13-unit circuit does not fit in weights: chipbuild's emission finds 6
  weights of -3.4e38 to -1.4e39, past bf16's and fp32's range (`why_units.py`, M). Exec4 runs that
  circuit from the IR, where it is fine; as a file it would be `inf`. So the PR runs have no units,
  as `live.md`'s, and pay for it: about 26,100 tok/s on fx's code, the same rate, but more tokens
  per step than `one.md`'s units runs.
- What still comes from Python on this path: the memory map (`cfg`) the host's LiveCPU uses, and the
  reference machine of the offline audit. Neither is the circuit.

## Tests

`tests/test_fx_pr.py`, 16 tests.

| test | what it shows |
|---|---|
| `test_opens_one_pull_request_with_exactly_the_changed_file` | against a bare git repository standing in for GitHub: the commit sits on the base, carries fx's title and fxputer's identity, changes only the file fx changed, bytes equal; main does not move; the PR call gets the title and fx's body plus the host's footer and `Fixes #1` |
| `test_one_pull_request_per_run` | a second call is refused with the first PR's URL |
| `test_no_changes_is_an_error_for_fx` | no diff, no push: fx gets an error it can act on |
| `test_unregistered_run_and_bad_paths_are_refused` | unknown run ids; `../`, `.git/`, absolute and backslash paths |
| `test_only_new_fx_branches_are_pushed` | `main`, `fx/../main`, `feature/x` refused; an existing `fx/` branch is not overwritten |
| `test_new_files_deletions_and_exact_bytes` | a new file (CRLF bytes kept as fx wrote them), a deletion |
| `test_github_error_text_goes_back_to_fx` | GitHub's error text is the tool result; the retry opens the PR on the branch already pushed |
| `test_a_lost_answer_is_recovered_without_a_second_branch` | an identical change already on the run's branch is reused; a different one is never touched (`-2`) |
| `test_footer_does_not_repeat_an_issue_link`, `test_title_and_body_checks` | |
| `test_client_through_the_service` | the box's client, over HTTP on 127.0.0.1, the store read from disk (`__pycache__` left out); a down tunnel gives fx an error, not a crash |
| `test_lockstep_hosts_replay_the_chip_hosts_answer` | rv32emu gets the chip host's answer; a call it did not make, or different arguments, stop it |
| `test_store_diff`, `test_prompt_wording` | |
| `test_initialize_declares_the_tool` | the ACP `initialize` message carries the tool in `clientCapabilities.libfx.tools`, the five tools before it unchanged |
| **`test_the_call_arrives_as_host_call_tokens`** (GPU) | a whole fx session on Executor v4 (4,096 frames, the circuit from `core.build`), with a scripted model (not Qwen) whose last call is `open_pull_request`. The name and the JSON arguments the handler got are exactly bytes the chip printed as read-sled value records during that ecall; the URL fed back is in fx's memory as the chip prints it; rv32emu replayed the call; the task solved; the offline audit found 0 wrong chip tokens. 1 passed, 827 s on the L4 |

Laptop, final code: 14 passed, 2 skipped (no wasmtime, no GPU). Box, before the last two handler changes (the reuse of an identical branch, the closing-keyword rule): 15 of 15 passed, the GPU test included; the box was stopped after the runs, so the final code's box run is not done.

A scripted session from the chip file too (`out/q2`, not Qwen): 7 steps, 13,447,342 chip tokens,
0 wrong; rv32emu identical; 0 of 82,688 end words differ. Its PR call had no handler behind it and
got `could not reach the pull request service: URLError` as an error result; fx went on to its final
answer, as it should.

## Disclosures

- **The prompt asks for the PR** (above). fx was not left to guess that a tool call is wanted.
- **The footer and the issue link are the host's**, and the PR says so.
- **The commit identity is the host's choice** (`fxputer`, divagr18's noreply email), so GitHub links
  the commits to divagr18's account. The PR is opened by divagr18's `gh` login.
- **The title is whitespace-normalised** by the handler (newlines to spaces): GitHub titles are one
  line. It did not change any title in these runs. It changed two: issue 2's title came with a leading space and issue 5's with a trailing newline; GitHub shows them trimmed (the raw arguments are in `pr.json`).
- **The issues' titles are ours** (`pr_driver.ISSUE_TITLES`); their bodies are the P0c prompts
  verbatim.
- **Host work, as `live.md`:** template rendering, BPE, parsing, SSE, the file store, grep and glob,
  the import handlers, Qwen's snapshots. Boot ran on rv32emu; its state is the snapshot (and the
  chip file's state).
- Sampling as `live.md`: temperature 1, no top-k or top-p, seed 1 for every run. One run per issue:
  it shows the path works, not a rate.
- One process, one GPU, two sets of weights: the chip's (from the file) and base Qwen's. Not one
  merged model (`live.md`, What is in one model).
- **Operations, honestly.** (a) On issue 1 the driver's launch ssh hung (the whole command line
was backgrounded) and the driver exited before fx reached the tool; I restarted the handler and the
tunnel by hand before fx's first model call, then reattached the driver (`--resume`). A leftover tunnel
held the box's port, so the driver's own tunnel kept failing and reopening; the call went through the
leftover one to the same handler. Both are fixed in the driver (only the job is backgrounded; a busy
port is refused). (b) On issue 5, the lost call above.
- **The footer's wording changed during the runs.** #6-#9 were opened with a footer that began
  "*Added by the host, not by fx:*"; the user asked for that line to go and is editing those bodies
  by hand. #10 and every PR from now on start "This pull request was opened by fx ...". Either way
  the footer is written by the host, not by fx, and only the text above its rule is fx's.
- The `Fixes #N` rule: the host adds it unless the body already closes the issue with a GitHub
  keyword. During the runs a bare mention of `#N` also counted, so #9 (whose body says "as
  requested in issue #4") went out without `Fixes #4`; the rule is fixed for future PRs.

## Reproduce

On the laptop with `gh` logged in, and q35-batch set up as `fx/fxio/live.sh` (venv, `rv32emu.so`,
`row24/`) under `/opt/dlami/nvme/pr`:

1. The chip file, once, on the box: `python fx/fxio/live_chip.py ./rv32emu.so row24/fx-fastcopy.elf
   row24/imports.json MODEL out/cf --job JOB.json --backend scripted-pr --full-regs --frames 4096
   --write-chip-file chip/fx-live-f4096.safetensors`.
2. The issues, once: `python fx/fxio/pr_driver.py issues`.
3. Each issue: `python fx/fxio/pr_driver.py run N --seed 1 --out OUT`.
4. `python fx/fxio/report_pr.py docs/fx/pr.json OUT/summary_*.json`.
5. Tests: `pytest tests/test_fx_pr.py`; on the box with `FXPR_ELF`, `FXPR_DLL`, `FXPR_MODEL` set for
   the GPU test.
