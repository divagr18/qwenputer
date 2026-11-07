# fx on Qwenputer: the I/O packet path

Status 2026-11-07. Follows `speed-iter.md` (rows 11 and 24), `circuits.md` (the token model and
units) and `speedups.md` (§1, §4, §5, §9, §13). Raw numbers: `io.json`. Code: `fx/fxio/`,
`chipformer/rv/chat.py` (`FxQwen`, `fx_session`, `_qwen_prefill`), `tests/test_fx_io.py`.
**M** = measured, **E** = estimated.

## Read this first: real Qwen has not driven fx yet

**The end-to-end run with Qwen did not happen.** No Qwen3.5-4B weights were run in any number
here. The GPU boxes could not be reached (Blockers), and this machine has neither the weights nor
the RAM. So task 3's solve rate is **not measured**, and P0c's 15/15 has no counterpart yet.

What ran in its place is `fxio/scripted.py`: hand-written tool-call plans for the P0c tasks,
written out in Qwen's raw format and tokenized with Qwen's real tokenizer. Everything except the
model is real: fx's exact binaries, its agent loop and tools, the chat template, the delta, the
parser, the SSE, the instruction and token counts.

**The "pass" column below is not a solve rate.** A plan that is right by construction passing the
hidden check shows that the packet path carries a correct plan through fx. It says nothing about
whether Qwen would write that plan. Each plan copies the tool order of a real P0c run, and its
reasoning is filler sized to that run's recorded output tokens per step, so request bodies grow
as they did. P0c did not record arguments or reasoning, so neither is Qwen's.

The numbers that should survive a live run are the per-byte costs, the delta's mechanism and the
protocol. The per-task instruction and token counts will move with what Qwen actually does.

| | result | |
|---|---|---|
| Packet protocol, all 56 imports classified (the brief says 55) | 30 OS layer, 5 model I/O, 21 stubbed; 29 ever called | M |
| Delta on the 28 later calls of the 5 tasks | **69 tokens a call against 1,323: 94.8% saved** (270 B against 5,630 B) | M, scripted |
| Delta exactness, one turn | 28 of 28 calls exact, 0 tokens rolled back | M, scripted |
| Delta at a second fx turn | history is rewritten: rollback, 876 of 1,520 tokens re-fed | M, scripted |
| fx's RV32 binaries driving the packet path | 10 of 10 sessions complete, plans pass the checks; **not Qwen** | M, scripted |
| Instructions and tokens per task, row 24 | 4.2-10.6M instr, 6.0-15.5M tokens; with units 2.1-5.2M, 3.3-8.2M | M; units' tokens E |
| Text work on the RV32 core instead of the host | 32.5k instr per render, 12.0k per reply; output byte-identical on every call | M |
| Projected wall time per task, row 24 + units + delta | 3.2-6.8 min (23k tok/s, 65 ms / token) | E |
| **Solve rate with real Qwen** | **not measured** | |

## Exactness

- **fx's source:** `fx/check_src.sh`: 717 files, 0 changed against `757bdb9` (M).
- **The binaries:** the images of speed-iter and circuits.md, not rebuilt. sha256 `282e53d4…`
  (row 24) and `209f2a87…` (row 11), recorded in every run file (M).
- **Across every run:** 0 misaligned accesses, 0 traps, 0 imports outside what the host answers (M).
- **Units:** each run was replayed from its packets with all units on. On all 10: 0 request
  bytes differ, the host-I/O hash chain is identical, the final repository is identical, and no
  unit check failed (M).

**Disclosed: what moves to the host, or changes which fx code runs.**

| | who does it | status |
|---|---|---|
| Chat-template rendering, BPE, reasoning and tool-call parsing, SSE construction | the host | the model server's job. Render, parse and SSE are also measured on RV32 (below); BPE is not |
| Qwen state snapshots | the host | a prefix cache by another name |
| grep_files, glob_files | the host | **fx's work, done by the host.** fx's own do not compile for WASM (speed-iter item 1) |
| The file store | the host (`wasifs.py`) | fx's code reaches files only through WASI; the store itself is not in the weights |
| The ACP client (initialize, prompt) | the host | stands in for the user |
| fx's read / edit / write, row 24 | the guest, through a platform root | changes which fx code runs, toward more of it (speed-iter row 24) |
| Gateway route, one SSE read, frozen clock | configuration | speed-iter rows 2-5 |

**This is not the prompt P0c used.** Native fx sent 7 system messages (9 KB) and 13 tools.
fx-core under the libfx contract sends one short system message and the 5 host tools. A live run
tests Qwen on a smaller prompt than P0c, and a different tool set; its solve rate would not be
directly comparable to 15/15.

## 1. The packet protocol

Spec and table: `fxio/protocol.py`. Below the packets is the call layer that exists today: every
import is an `ecall` with a7 = 0x400 + id, a0 = pointer to the u32 argument words, a1 = word count,
result in a0, buffers by DMA into wasm linear memory.

**Frames**, little-endian: `"FXIO" | version u8 | kind u8 | flags u16 | seq u32 | length u32 | payload`.

| kind | direction | payload |
|---|---|---|
| REQUEST | guest to Qwen | u32 url length, url, u32 headers length, headers JSON, body |
| REPLY | Qwen to guest | u16 status, then each read as u32 length + bytes |
| USER | user to guest | ACP client bytes for stdin |
| TOOL_REQ / TOOL_RES | guest to host and back | grep_files, glob_files only |

**Onto RvChat's token stream:**
- **Chip to Qwen:** a REQUEST never reaches Qwen as bytes. The mapping turns it into the ids Qwen
  lacks, fed as one chunked prefill. Qwen never sees fx's JSON.
- **Qwen to chip:** Qwen's ids decode to text, the text parses into reasoning, content and calls,
  and those become one SSE read into fx's buffer.
- **Silent:** ACP output on stdout is never echoed to Qwen.

**Classification**, with call counts over the 5 tasks (M). Imports with zero calls are in `io.json`;
every one of them has a fixed answer or an implementation in the host.

| group | import | row 24 | row 11 | what it becomes |
|---|---|---|---|---|
| a | `clock_time_get` | 479 | 472 | frozen constant |
| a | `fd_write` | 170 | 163 | stdout / stderr: silent sink; fd >= 3: file store |
| a | `path_open`, `fd_close` | 179 each | 0 | file store |
| a | `fd_fdstat_get`, `fd_filestat_get` | 172, 169 | 0 | file store |
| a | `path_filestat_get`, `fd_pread`, `path_readlink` | 108, 77, 39 | 0 | file store |
| a | `fd_read` | 20 | 20 | stdin: USER packets; fd >= 3: file store |
| a | `fx_host_tool_call` | 7 | 28 | see the disclosure: row 24 leaves only grep and glob |
| a | `fd_prestat_get`, `fd_prestat_dir_name` | 20, 10 | 10, 0 | ROM constants |
| a | `path_rename`, `fd_sync` | 7, 7 | 0 | file store |
| a | `args_*`, `environ_*`, `random_get` | 5 each | 5 each | ROM constants; a seeded PRNG |
| b | `fx_http_stream_open` / `status` / `retry_after` / `next` / `close` | 33 each | 33 each | one model call: REQUEST and REPLY |
| c | `fx_steering_state`, `fx_steering_take` | 38 each | 38 each | fixed answers: no steering |
| c | `fx_host_tool_result_release`, `fx_steering_close` | 28, 5 | 28, 5 | no-ops |

The rest of (c), never called: attachments, oauth, the session store, `fx_http_request`, journal,
set-times, link, symlink. (a) also holds file-store calls these sessions never reached, plus
`poll_oneoff` and `proc_exit`.

## 2. The model call mapping

`fxio/delta.py`, class `DeltaLink`. Each request:

1. fx's body (AI Gateway or Chat Completions) becomes messages and tools, with system messages
   merged as `p0c/proxy.py` does.
2. Qwen3.5's own template renders it. `QwenTemplate` matches `apply_chat_template` in text and ids
   (test, M).
3. The render's **ids** are compared with every id Qwen has run. If it extends them, only the
   suffix is fed. If not, the backend restores the longest snapshot that is a prefix.
4. Qwen's text is parsed and returned as one read.

So Qwen's state is what a stateless server would build from that request, up to chunking's float
rounding (`text_extends` in the runs records the text-level result too). The check is on ids
because the same text can tokenize differently from what Qwen generated; a test shows a case
where the text matches, the ids do not, and the rollback fires.

**Why a tool round is exact.** The template rebuilds an assistant turn from fx's echo of it:
reasoning, content, then calls. It is exact only because fx echoes reasoning back. That was
checked on the binary on both routes (M). If fx dropped it, every step would roll back.

| task (row 24) | full tokens, all calls | delta tokens | saved | full bytes | delta bytes | saved | calls |
|---|---|---|---|---|---|---|---|
| t1_bug | 7,186 | 1,124 | 84.4% | 29,795 | 4,395 | 85.2% | 1 first, 5 exact |
| t2_rename | 12,435 | 1,218 | 90.2% | 52,326 | 5,065 | 90.3% | 1 first, 8 exact |
| t3_feature | 8,111 | 1,070 | 86.8% | 35,334 | 4,327 | 87.8% | 1 first, 5 exact |
| t4_config | 8,438 | 973 | 88.5% | 35,799 | 4,155 | 88.4% | 1 first, 7 exact |
| t5_parse | 4,204 | 867 | 79.4% | 18,396 | 3,624 | 80.3% | 1 first, 3 exact |
| **all** | **40,374** | **5,252** | **87.0%** | **171,650** | **21,566** | **87.4%** | **33 calls** |

- These include each session's first prompt, which has to go in whole. On the 28 later calls
  alone: **1,323 against 69 tokens a call, 94.8% saved** (5,630 against 270 bytes). All M, scripted.
- Native fx's request is much larger (9 KB of system prompt, 13 tools), and it repeats on every
  call, so the saving there should be higher (E).

**The limit of that result: generations here are canonical.** `scripted.py` produces exactly the
tokens the tokenizer would. Real Qwen may not, and it may write a tool call in a layout the
template will not rebuild byte for byte (`<parameter=x>value</parameter>` without newlines, say).
Each such step costs a rollback of that step's generation, and only that. How often this happens
is the first thing a live run measures. **"28 of 28 exact" is an upper bound.**

**A second turn is not a pure extension (M, scripted, rows 24 and 11).** Two things happen together
when a new prompt arrives:
- fx inserts a "Session file evidence" user message before the last answer;
- the template stops rendering think blocks before the newest question.

This hits every multi-turn session. Row 24: 1,232 tokens rolled back, 876 of 1,520 re-fed, kept 644.
With snapshots only at generation starts it was a full rebuild, which is why `DeltaLink` snapshots
at every message boundary. In weights this needs Qwen's state to be restorable, and each snapshot
holds the DeltaNet states (about 50 MB each, E; not measured on 4B).

**Routes.** Row 11 and row 24 use fx's gateway route, as those rows are defined. Chat Completions
also runs and is also exact (t1: 1 first, 5 exact), but fx pays much more for it: 16.46M tokens
against 9.47M on the same plan (M, no units). This matches speed-iter row 2. So this work returns
the reply as gateway events rather than the OpenAI event the brief names; both are implemented.

**The parser** is a port of the Python engine vLLM 0.33 serves with
(`vllm/parser/qwen3.py`, `ParserEngine`), read from the v0.33.0 tag. P0c's outputs keep a leading
`\n\n` in content, which the Python engine does and the repo's Rust parser would not, so that is
the right one to port.
- Its cases come from reading vLLM's grammar and tests. **It has not been run against vLLM.**
- It works on text; the engine also matches the wrapper tokens by id.
- Schema coercion is simplified. All five tools take only strings, so it did not matter here.

### Who renders, and what doing it in weights would cost

Today the host renders, tokenizes and parses, so the delta is exact but not in weights. To price
that, `fxio/render_rv.c` does the text work on RV32, run in rv32emu on the recorded requests:

| job | calls | instructions per call | per byte | trace tokens per call | output vs the host | |
|---|---|---|---|---|---|---|
| render: fx's body to the new template text | 28 | 32.5k | 6.5 per body byte | 42.3k | **identical on all 28** | M |
| reply: Qwen's text to SSE | 33 | 12.0k | 21.5 per text byte | 17.6k | **identical on all 33** | M |
| BPE encode of the delta | | 5-11k | 20-40 per byte, 270 B a call | | not built | E, speedups §4 |
| token ids to bytes | | small | a table lookup | | not built | E |

- **In all, about 50-55k instructions a call** (M + E), about 60k trace tokens without BPE.
- That is **about 7% of a step** with units (0.84M tokens a step) and about 4% without (1.54M).
- Code: 8.8 KB of RV32 text, 233 lines of C.
- BPE also needs a vocabulary trie in DROM, 2-4 MB (E, speedups §4). Greedy longest-match is not
  exact BPE, so an in-weights tokenizer would need merge ranks to keep the id check meaningful.
- **Scope of `render_rv.c`:** Qwen's normal format and tool rounds only. It has no malformed-call
  recovery, no type coercion and no second-turn template. It scans the whole body; a provider
  bound in the platform root (speedups §5) would make that O(new bytes).

## 3. End to end, fx's binary with the scripted stand-in

Row 24 is fx's own read / edit / write on the CPU; row 11 has all five tools in the host. One run
of each plan, since a fixed plan gives the same result every time; the brief's two reps need the
live run. Instructions and tokens are one prompt to its response, without boot (0.53M instr,
0.91M tokens, which a snapshot removes). "units" is the same run replayed with all units.

| row | task | steps | plan check | instr | tokens | instr, units | tokens, units | generated | delta tok / later call | full tok / later call |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | t1_bug | 6 | pass | 6.56M | 9.47M | 3.31M | 5.14M | 752 | 95 | 1,307 |
| 24 | t2_rename | 9 | pass | 10.58M | 15.49M | 5.24M | 8.17M | 823 | 71 | 1,473 |
| 24 | t3_feature | 6 | pass | 6.62M | 9.33M | 3.25M | 5.02M | 1,078 | 74 | 1,482 |
| 24 | t4_config | 8 | pass | 7.43M | 10.58M | 3.96M | 6.19M | 555 | 45 | 1,112 |
| 24 | t5_parse | 4 | pass | 4.19M | 6.04M | 2.11M | 3.26M | 750 | 67 | 1,180 |
| 11 | t1_bug | 6 | pass | 5.90M | 8.26M | 3.05M | 4.71M | 752 | 77 | 1,250 |
| 11 | t2_rename | 9 | pass | 9.30M | 13.00M | 4.78M | 7.42M | 823 | 62 | 1,426 |
| 11 | t3_feature | 6 | pass | 6.31M | 8.76M | 3.10M | 4.79M | 1,078 | 64 | 1,455 |
| 11 | t4_config | 8 | pass | 7.16M | 10.10M | 3.82M | 5.98M | 555 | 40 | 1,102 |
| 11 | t5_parse | 4 | pass | 3.81M | 5.32M | 1.97M | 3.02M | 750 | 55 | 1,154 |

- Instructions, CPU tokens, steps and checks: M. Units' tokens: E per circuits.md's record formats.
- **Steps and generated tokens are the plan's, not Qwen's.** Generated tokens were sized from one
  P0c run per task; P0c's noshell mean was 870 output tokens a task and 6.3 steps (M).
- t1's plan includes the failed first edit of its P0c run. fx's real error text came back through
  the guest on both rows, and the plan recovered.
- Row 24's t1-t5 cost 5-19% more tokens than row 11's (M).

**Projected wall time, row 24 (E).** The tokens are measured, the rates are not this system's:
- 23k chip tok/s is Executor v4 in RvChat on calc firmware. exec-v4.md puts fx's memory layers at
  about 15-19k (E), so 15k is the row to plan on.
- 65 ms / token is the HF path; 30 ms is speedups §9's target.
- Delta ingest assumes a chunked prefill at 5k tok/s (speedups §4).
- The last column prints every render byte by byte through today's echo path, at 2 chip tokens a
  byte, with no units.

| task | chip, no units | chip, units | generate | delta ingest | **total, units + delta** | no units, no delta, bytewise |
|---|---|---|---|---|---|---|
| **23k tok/s, 65 ms** | | | | | | |
| t1_bug | 6.8 min | 3.7 min | 49 s | 0.22 s | **4.5 min** | 40.0 min |
| t2_rename | 11.2 min | 5.9 min | 53 s | 0.24 s | **6.8 min** | 68.8 min |
| t3_feature | 6.7 min | 3.6 min | 70 s | 0.21 s | **4.8 min** | 46.2 min |
| t4_config | 7.6 min | 4.5 min | 36 s | 0.19 s | **5.1 min** | 47.1 min |
| t5_parse | 4.4 min | 2.4 min | 49 s | 0.17 s | **3.2 min** | 25.1 min |
| **15k tok/s, 65 ms** | | | | | | |
| t1_bug | 10.5 min | 5.7 min | 49 s | 0.22 s | **6.5 min** | 43.7 min |
| t2_rename | 17.2 min | 9.1 min | 53 s | 0.24 s | **10.0 min** | 74.9 min |
| t3_feature | 10.4 min | 5.6 min | 70 s | 0.21 s | **6.7 min** | 49.9 min |
| t4_config | 11.8 min | 6.9 min | 36 s | 0.19 s | **7.5 min** | 51.2 min |
| t5_parse | 6.7 min | 3.6 min | 49 s | 0.17 s | **4.4 min** | 27.5 min |

- **What dominates:** the chip. Qwen's generation is 8-26% of the total, and ingest
  vanishes once the delta is batched. The expensive thing left is fx's own glue, as speedups said.
- **What the delta buys:** nearly all of the gap in the last column is byte-by-byte ingest of the
  whole render. Ingest falls from about 57 minutes on t2 to a quarter of a second.
- **Not in these totals:** the 50-55k instructions a call of in-weights text work, because the
  host did it here. Add about 7% for the pure version.

## 4. RvChat integration

In `chipformer/rv/chat.py`; `core.py` and `exec4.py` are untouched.
- `_qwen_prefill`: a run of text through Qwen's layers as one chunk (§4), with the masks a chunk
  needs. `_qwen_step` takes one token and needs none.
- `FxQwen`: RvChat's Qwen as a `DeltaLink` backend, with reset, snapshot and restore.
- `fx_session(chat, chip, template)`: the loop. `chip` gives REQUEST packets and takes REPLY packets.

**Tests** (`tests/test_fx_io.py`, 26 tests; with `test_rv_bridge.py`, 35 pass, M):
- chunked prefill equals token-by-token on a tiny real Qwen3_5, within 2e-3;
- a chip replaying recorded fx packets (`tests/data/fx/t1_bug.fxio`, 35 KB): with the recorded text
  forced, every later call is exact;
- with a random model the history never matches, and each call rolls back to its generation start
  and still answers. That is the path real divergence takes.

**Not exercised:** none of this has run on a merged checkpoint. `_text_hc` fills the kept dims from
the park, and those dims reach Qwen through RMSNorm's scale, so whether batched ingest matches
`_exec_hc` needs a run on the real checkpoint. No chip that runs fx on the circuit exists yet.

## Blockers

1. **No access to a GPU box. This is what stopped task 3.**
   - Port 22 on both boxes admits a fixed list of addresses, and this machine's current address,
     `106.222.210.5`, is on neither list.
   - Adding it to q35-eval2's security group was attempted once and **denied by the session's
     permission policy**. Nothing was changed, and no workaround was tried. The boxes have no
     instance profile, so SSM is not available.
   - **To unblock:** someone with the authority adds that address to port 22 of
     `sg-09b8c20eadfa4a8b3` (q35-eval2, us-west-2) or `sg-0dc8a50ea3fb3f5c3` (q35-batch, us-east-1),
     or runs `fx/fxio/live.sh` from an allowed address. It does tasks t1-t5 × 2 reps on row 24, row
     11, and the units replays.
   - `live.sh` and `qwen_hf.py` have **never run on real Qwen**; the backend is tested on a tiny
     model. Expect a fix or two.
2. **Capture window.** Qwen to the CPU is 80 tokens in the circuit; this needs at least 195 tokens
   and 1.4 KB a call, and real calls with tool arguments will need more (M, scripted). speedups
   §13's 4-8k-token buffer is required before any of this runs in weights.
3. **The file store and grep / glob** are still on the host (Disclosed).
4. **The in-weights tokenizer** is unbuilt, and exact BPE needs merge ranks.
5. **Snapshots in weights:** whether Qwen's state can be saved and restored on the real path, and
   what it costs per message, is not measured.

## Reproduce

Paths as in speed-iter.md (`it/env.sh`). `MODEL` is the Qwen3.5-4B HF snapshot; only its tokenizer
and template are needed for the scripted runs.

1. The emulator: as in circuits.md.
2. One session: `python -I fx/fxio/e2e.py DLL ELF IMPORTS MODEL OUT.json --task t1_bug [--native]
   [--provider chat] [--turns 2] [--backend scripted | hf:Qwen/Qwen3.5-4B | replay:RUN.json]
   [--units all]`.
3. The RV32 text work: build `fxio/render_rv.c` as `fxio/render_cost.py` says, then
   `python -I fx/fxio/render_cost.py DLL render_rv.elf MODEL OUT.json RUN.json ..`.
4. `io.json`: `python -I fx/fxio/report.py RUNS IMPORTS RENDER.json io.json`.
5. Tests: `pytest tests/test_fx_io.py`.
6. The live run, from an allowed address: `fx/fxio/live.sh 2`.
