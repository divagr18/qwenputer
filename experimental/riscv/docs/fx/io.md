# fx on Qwenputer: the I/O packet path

Status 2026-11-07. Follows `speed-iter.md` (rows 11 and 24), `circuits.md` (the token model and
units) and `speedups.md` (§1, §4, §5, §9, §13). Raw numbers: `io.json`. Code: `fx/fxio/`,
`chipformer/rv/chat.py` (`FxQwen`, `fx_session`, `_qwen_prefill`), `tests/test_fx_io.py`.
**M** = measured, **E** = estimated. Figures from the scripted stand-in are labelled scripted.

## Result

**fx's exact RV32 binaries, in rv32emu, drove real Qwen3.5-4B and solved every P0c task they were
given: 20 of 20, 10 on each row.** Native fx in P0c: 15 of 15. No task failed, so no failures are
listed. All on one L4 with HF transformers, sampling at vLLM's defaults (see Limits).

| | real Qwen | scripted (see §3) | |
|---|---|---|---|
| **Solved**, row 24 (t1-t5 x 2) | **10 / 10** | plans correct by construction, not a solve rate | M |
| **Solved**, row 11 (t1-t5 x 2) | **10 / 10** | as above | M |
| Solved in the two-turn and chat-route runs | 6 / 6 | | M |
| Steps per task, mean, row 24 | 4.1 | 6.6 | M |
| Generated tokens per task, mean, row 24 | 681 | 792 | M |
| Instructions per task, row 24 | 2.8-9.8M; with units 1.3-4.8M | 4.2-10.6M; 2.1-5.2M | M |
| Tokens per task, row 24 | 4.2-14.7M; with units 2.0-7.6M | 6.0-15.5M; 3.3-8.2M | M; units' tokens E |
| **Delta on later calls, row 24** | **102 tokens a call against 1,271: 92.0% saved** | 69 against 1,323: 94.8% | M |
| Delta on later calls, row 11 | 88 against 1,280: 93.1% saved | | M |
| Rollbacks inside a turn | **0 of 80** later calls (31 row 24, 32 row 11, 17 in two-turn runs) | 0 of 28 | M |
| Rollbacks at a new fx turn | **5 of 5** second prompts; 47% of tokens still saved there | 1 of 1 | M |
| Canonically tokenized generations | **114 of 115** | all, by construction | M |
| Generations ending in a stop token; parser warnings | 115 of 115; 0 | | M |
| Projected wall time per task, row 24, units + delta | 2.2-6.0 min at 23k tok/s; **3.0-8.8 min at 15k** (task means) | 3.2-6.8 min at 23k | E |
| Protocol: 56 imports (the brief says 55) | 30 OS layer, 5 model I/O, 21 stubbed; 29 ever called | | M |
| Text work on the RV32 core, not the host | | 32.5k instr a render, 12.0k a reply; byte-identical on every call | M, scripted sessions |

**Limits of this result.**
- **Two reps per task** on small tasks. 20 of 20 shows the path works and that Qwen can drive these
  tools through it. It does not bound the failure rate.
- **Not the same prompt as P0c** (see Exactness), so 20 of 20 and 15 of 15 are not a like-for-like
  comparison. Steps and tokens are lower here partly because the prompt is smaller.
- **Sampling is inferred.** Qwen3.5-4B ships no `generation_config.json` and fx sends no
  temperature, so P0c ran at vLLM's defaults; these runs set temperature 1, no top-k or top-p, to
  match. The inference is from the code path, not an observation of the P0c server.
- **Not vLLM.** Qwen ran in HF transformers, and the parser is a port. In 115 generations it gave no
  warnings and fx accepted every call, but it was never compared with vLLM's output directly.
- **Host work remains** (disclosed below). This is the fxputer loop minus the in-weights CPU, as
  briefed: the emulator stands in for the circuit, and nothing here ran on the chip layers.

## Exactness

- **fx's source:** `fx/check_src.sh`: 717 files, 0 changed against `757bdb9` (M).
- **The binaries:** the images of speed-iter and circuits.md, not rebuilt. sha256 `282e53d4…`
  (row 24) and `209f2a87…` (row 11), recorded in every run file and checked again on the GPU box (M).
- **Across every run, real and scripted:** 0 misaligned accesses, 0 traps, 0 imports outside what
  the host answers (M).
- **Units:** each run was replayed from its packets with all units on. On all 20 real runs and all
  10 scripted ones: 0 request bytes differ, the host-I/O hash chain is identical, the final
  repository is identical, and no unit check failed (M).

**Disclosed: what moves to the host, or changes which fx code runs.**

| | who does it | status |
|---|---|---|
| Chat-template rendering, BPE, reasoning and tool-call parsing, SSE construction | the host | the model server's job. Render, parse and SSE are also measured on RV32 (below, on scripted sessions); BPE is not |
| Qwen state snapshots | the host | a prefix cache by another name; 84-128 MB each on real Qwen (M) |
| grep_files, glob_files | the host | **fx's work, done by the host.** fx's own do not compile for WASM (speed-iter item 1) |
| The file store | the host (`wasifs.py`) | fx's code reaches files only through WASI; the store itself is not in the weights |
| The ACP client (initialize, prompt) | the host | stands in for the user |
| fx's read / edit / write, row 24 | the guest, through a platform root | changes which fx code runs, toward more of it (speed-iter row 24) |
| Gateway route, one SSE read, frozen clock | configuration | speed-iter rows 2-5 |

**This is not the prompt P0c used.** Native fx sent 7 system messages (9 KB) and 13 tools.
fx-core under the libfx contract sends one short system message and the 5 host tools. The live
runs test Qwen on a smaller prompt and a different tool set, so 20 of 20 is not directly
comparable to 15 of 15.

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

**Classification**, with call counts over the 5 scripted sessions (M). Imports with zero calls are in `io.json`;
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

**Scripted**, row 24. The real figures follow.

| task (row 24, scripted) | full tokens, all calls | delta tokens | saved | full bytes | delta bytes | saved | calls |
|---|---|---|---|---|---|---|---|
| t1_bug | 7,186 | 1,124 | 84.4% | 29,795 | 4,395 | 85.2% | 1 first, 5 exact |
| t2_rename | 12,435 | 1,218 | 90.2% | 52,326 | 5,065 | 90.3% | 1 first, 8 exact |
| t3_feature | 8,111 | 1,070 | 86.8% | 35,334 | 4,327 | 87.8% | 1 first, 5 exact |
| t4_config | 8,438 | 973 | 88.5% | 35,799 | 4,155 | 88.4% | 1 first, 7 exact |
| t5_parse | 4,204 | 867 | 79.4% | 18,396 | 3,624 | 80.3% | 1 first, 3 exact |
| **all** | **40,374** | **5,252** | **87.0%** | **171,650** | **21,566** | **87.4%** | **33 calls** |

- These include each session's first prompt, which has to go in whole. On the 28 later calls
  alone: 1,323 against 69 tokens a call, 94.8% saved (5,630 against 270 bytes). All M, scripted.

**Real Qwen (M).**

| | row 24 | row 11 |
|---|---|---|
| Model calls, of which later | 41, 31 | 42, 32 |
| Later calls exact | 31 of 31 | 32 of 32 |
| Tokens rolled back | 0 | 0 |
| Delta / full tokens, per later call | 102 / 1,271 | 88 / 1,280 |
| Saved on later calls | **92.0%** | **93.1%** |
| Delta / full bytes, per later call | 386 / 5,100 | 345 / 5,165 |
| Saved including first prompts | 78.7% | 80.2% |

- The saving is close to the scripted 94.8%. It is lower because real sessions are shorter (4.1
  steps against 6.6, so the render a delta is measured against is smaller), and because Qwen often
  batches tool calls, so one step brings in several results.
- **Parallel calls are exact too.** Qwen sent several tool calls in one reply on t2 (three reads,
  three edits), and the template rebuilt them byte for byte. The scripted plans never did this.
- Native fx's request is much larger (9 KB of system prompt, 13 tools), and it repeats on every
  call, so the saving there should be higher (E).

**Are real generations canonically tokenized? Almost always (M).** 114 of 115 are: encoding the
generated text gives back Qwen's own ids.
- The one exception (row 11, t5, rep 2) has 198 ids where the tokenizer gives 197.
- **It was a session's last answer, so no later request ever compared against it.** It shows that
  non-canonical output happens. It does not show what it costs; mid-session it would have forced a
  rollback of that step's generation. So 0 rollbacks inside a turn means none were triggered. The
  rollback path itself did run on real Qwen, on the 5 second prompts below, but a non-canonical
  trigger for it has only happened in a test.
- Every real tool call used the template's own layout. The layout mismatch this section feared
  (`<parameter=x>value</parameter>` with no newlines) did not occur.

**A second turn is not a pure extension (M, real Qwen, 5 of 5).** Two things happen together when a
new prompt arrives:
- fx inserts a "Session file evidence" user message before the last answer;
- the template stops rendering think blocks before the newest question.

| task | solved | calls | in-turn calls exact | second prompt: full tokens | re-fed | saved | rolled back | kept |
|---|---|---|---|---|---|---|---|---|
| t1_bug | yes | 4 | 2 / 2 | 1,032 | 388 | 62% | 537 | 644 |
| t2_rename | yes | 7 | 5 / 5 | 2,272 | 1,628 | 28% | 1,767 | 644 |
| t3_feature | yes | 3 | 1 / 1 | 1,103 | 409 | 63% | 825 | 694 |
| t4_config | yes | 8 | 6 / 6 | 1,359 | 708 | 48% | 794 | 651 |
| t5_parse | yes | 5 | 3 / 3 | 1,189 | 529 | 56% | 834 | 660 |

- Every second prompt rolled back, and all 5 diverged at the same place: the first assistant turn's
  `<think>`, which the render no longer has. So this is structural, not a model behaviour.
- Tokens re-fed: 3,662 of 6,955, 47% saved on those calls. The whole first turn goes back in.
- Snapshots only at generation starts made this a full rebuild, which is why `DeltaLink` snapshots
  at every message boundary. In weights this needs Qwen's state to be restorable, at 84-128 MB per
  snapshot on these runs (M). That is per message, so a long session holds many of them.
- All 5 tasks stayed solved, and the second prompt's rendering is what a stateless server would see.

**Routes.** Row 11 and row 24 use fx's gateway route, as those rows are defined. Chat Completions
also runs and is also exact (scripted t1: 1 first, 5 exact; real Qwen t1: 1 first, 4 exact, solved in
5 steps), but fx pays much more for it: 16.46M tokens against 9.47M on the same plan (M, scripted, no
units); the real run took 13.06M. This matches speed-iter row 2. So this work returns
the reply as gateway events rather than the OpenAI event the brief names; both are implemented.

**The parser** is a port of the Python engine vLLM 0.33 serves with
(`vllm/parser/qwen3.py`, `ParserEngine`), read from the v0.33.0 tag. P0c's outputs keep a leading
`\n\n` in content, which the Python engine does and the repo's Rust parser would not, so that is
the right one to port.
- Its cases come from reading vLLM's grammar and tests. **It has not been run against vLLM.**
- On real Qwen: 115 generations, 0 warnings, and every tool call was accepted by fx (M).
- It works on text; the engine also matches the wrapper tokens by id.
- Schema coercion is simplified. All five tools take only strings, so it did not matter here.

### Who renders, and what doing it in weights would cost

Today the host renders, tokenizes and parses, so the delta is exact but not in weights. To price
that, `fxio/render_rv.c` does the text work on RV32, run in rv32emu on the scripted sessions' requests:

| job | calls | instructions per call | per byte | trace tokens per call | output vs the host | |
|---|---|---|---|---|---|---|
| render: fx's body to the new template text | 28 | 32.5k | 6.5 per body byte | 42.3k | **identical on all 28** | M |
| reply: Qwen's text to SSE | 33 | 12.0k | 21.5 per text byte | 17.6k | **identical on all 33** | M |
| BPE encode of the delta | | 5-11k | 20-40 per byte, 270 B a call | | not built | E, speedups §4 |
| token ids to bytes | | small | a table lookup | | not built | E |

- **In all, about 50-55k instructions a call** (M + E), about 60k trace tokens without BPE.
- That is **about 7% of a step** with units on the scripted sessions (0.84M tokens a step), and
  about 6% on the real ones (0.97M); without units 3-4%.
- Code: 8.8 KB of RV32 text, 233 lines of C.
- BPE also needs a vocabulary trie in DROM, 2-4 MB (E, speedups §4). Greedy longest-match is not
  exact BPE, so an in-weights tokenizer would need merge ranks to keep the id check meaningful.
- **Scope of `render_rv.c`:** Qwen's normal format and tool rounds only. It has no malformed-call
  recovery, no type coercion and no second-turn template. It scans the whole body; a provider
  bound in the platform root (speedups §5) would make that O(new bytes).

## 3. End to end

### Real Qwen

fx's exact binaries in rv32emu, on `q35-batch` (one L4), with Qwen/Qwen3.5-4B in HF transformers,
bf16. `fxio/live.sh 2`. Task means below, real against scripted. Each run's replay with all units
gave identical host I/O (M).

| row | task | solved | steps, real / scripted | generated, real / scripted | instr, real / scripted | tokens, real / scripted | instr, units | tokens, units |
|---|---|---|---|---|---|---|---|---|
| 24 | t1_bug | 2 / 2 | 4.5 / 6 | 466 / 752 | 4.47M / 6.56M | 6.52M / 9.47M | 2.30M | 3.57M |
| 24 | t2_rename | 2 / 2 | 5.5 / 9 | 907 / 823 | 9.37M / 10.58M | 14.06M / 15.49M | 4.59M | 7.23M |
| 24 | t3_feature | 2 / 2 | 2.0 / 6 | 789 / 1,078 | 2.96M / 6.62M | 4.38M / 9.33M | 1.38M | 2.11M |
| 24 | t4_config | 2 / 2 | 5.5 / 8 | 422 / 555 | 4.87M / 7.43M | 6.97M / 10.58M | 2.57M | 4.02M |
| 24 | t5_parse | 2 / 2 | 3.0 / 4 | 824 / 750 | 3.86M / 4.19M | 5.70M / 6.04M | 1.87M | 2.87M |
| 11 | t1_bug | 2 / 2 | 3.0 / 6 | 314 / 752 | 2.37M / 5.90M | 3.33M / 8.26M | 1.28M | 1.98M |
| 11 | t2_rename | 2 / 2 | 5.0 / 9 | 876 / 823 | 7.35M / 9.30M | 10.41M / 13.00M | 3.79M | 5.94M |
| 11 | t3_feature | 2 / 2 | 4.0 / 6 | 1,046 / 1,078 | 5.50M / 6.31M | 7.67M / 8.76M | 2.56M | 3.93M |
| 11 | t4_config | 2 / 2 | 6.0 / 8 | 445 / 555 | 5.05M / 7.16M | 7.12M / 10.10M | 2.67M | 4.18M |
| 11 | t5_parse | 2 / 2 | 3.0 / 4 | 651 / 750 | 3.29M / 3.81M | 4.61M / 5.32M | 1.67M | 2.56M |

Every run:

| row | task | rep | solved | steps | generated | instr | tokens | tokens, units | tool calls (failed) |
|---|---|---|---|---|---|---|---|---|---|
| 11 | t1_bug | 1 | yes | 3 | 321 | 2.41M | 3.40M | 2.02M | 2 (0) |
| 11 | t1_bug | 2 | yes | 3 | 306 | 2.32M | 3.26M | 1.94M | 2 (0) |
| 11 | t2_rename | 1 | yes | 5 | 919 | 7.40M | 10.47M | 5.96M | 10 (0) |
| 11 | t2_rename | 2 | yes | 5 | 832 | 7.30M | 10.34M | 5.91M | 10 (0) |
| 11 | t3_feature | 1 | yes | 6 | 1,375 | 8.44M | 11.73M | 5.89M | 5 (0) |
| 11 | t3_feature | 2 | yes | 2 | 716 | 2.57M | 3.60M | 1.96M | 1 (0) |
| 11 | t4_config | 1 | yes | 5 | 364 | 3.96M | 5.58M | 3.28M | 4 (0) |
| 11 | t4_config | 2 | yes | 7 | 526 | 6.14M | 8.65M | 5.09M | 6 (0) |
| 11 | t5_parse | 1 | yes | 3 | 695 | 3.39M | 4.76M | 2.63M | 2 (0) |
| 11 | t5_parse | 2 | yes | 3 | 607 | 3.18M | 4.46M | 2.48M | 2 (0) |
| 24 | t1_bug | 1 | yes | 3 | 366 | 2.83M | 4.18M | 2.29M | 2 (0) |
| 24 | t1_bug | 2 | yes | 6 | 565 | 6.11M | 8.85M | 4.85M | 5 (1) |
| 24 | t2_rename | 1 | yes | 5 | 924 | 8.89M | 13.41M | 6.84M | 10 (0) |
| 24 | t2_rename | 2 | yes | 6 | 890 | 9.84M | 14.70M | 7.63M | 11 (0) |
| 24 | t3_feature | 1 | yes | 2 | 969 | 3.12M | 4.58M | 2.21M | 1 (0) |
| 24 | t3_feature | 2 | yes | 2 | 609 | 2.81M | 4.18M | 2.01M | 1 (0) |
| 24 | t4_config | 1 | yes | 4 | 318 | 3.28M | 4.72M | 2.72M | 3 (0) |
| 24 | t4_config | 2 | yes | 7 | 526 | 6.46M | 9.22M | 5.32M | 6 (0) |
| 24 | t5_parse | 1 | yes | 3 | 859 | 3.96M | 5.84M | 2.94M | 2 (0) |
| 24 | t5_parse | 2 | yes | 3 | 788 | 3.76M | 5.56M | 2.80M | 2 (0) |

- One tool call failed across the 20 runs (row 24, t1, rep 2: an edit). Qwen read fx's real error,
  re-read the file and recovered, as it did in P0c.
- Instructions and CPU tokens: M. Units' tokens: E per circuits.md's record formats.
- Row 24 still costs more than row 11 per run, for fx's own tool code. Task means don't show it
  cleanly, because Qwen's paths differ between runs (t3, 2 steps on row 24 against 2 and 6 on row 11).
- **Per-task spread is wide:** t1 took 3 or 6 steps, and t3 on row 11 took 2 or 6. Two reps do not
  pin these means down.
- Chip-emitted text per later call: 102 BPE tokens on row 24, 88 on row 11.
- **Largest single generation: 795 tokens and a 3,314-byte reply** (M). The capture window is 80 tokens.

**Qwen's cost on this box (M).** HF transformers, one L4, with HF falling back to its reference
PyTorch kernels (`flash-linear-attention` and `causal_conv1d` not installed), and on the base
model, not the padded checkpoint.

| | measured | assumed in the scripted projection |
|---|---|---|
| Decode | **50.1 ms / token** | 65 ms (E), 30 ms target (E) |
| One chunked prefill, 2,272 tokens | **2,635 tok/s** | 5,000 tok/s (E) |
| Delta feeds as they really are, many short suffixes | **682 tok/s** | |
| Snapshot | 113 ms a session at most | |

**Projected wall time, row 24, task means (E).** Tokens, decode and feed rates are measured; the
chip rate is not this system's.
- 23k tok/s is Executor v4 in RvChat on calc firmware (M there). exec-v4.md puts fx's memory layers
  at about 15-19k (E), so **15k is the row to plan on**.
- The last column prints every render byte by byte at 2 chip tokens a byte, with no units, and
  feeds each byte at the measured decode rate.
- In-weights text work is not included (about 6% of a real step; see §2). Boot is not included (0.53M
  instructions, removed by a snapshot).

| task | chip, no units | chip, units | generate | delta feed | **total, units + delta** | no units, no delta, bytewise |
|---|---|---|---|---|---|---|
| **23k tok/s** | | | | | | |
| t1_bug | 4.7 min | 2.6 min | 23 s | 1.4 s | **3.0 min** | 20.8 min |
| t2_rename | 10.1 min | 5.2 min | 45 s | 2.3 s | **6.0 min** | 37.6 min |
| t3_feature | 3.2 min | 1.5 min | 40 s | 1.1 s | **2.2 min** | 10.7 min |
| t4_config | 5.0 min | 2.9 min | 21 s | 1.3 s | **3.3 min** | 24.0 min |
| t5_parse | 4.1 min | 2.1 min | 41 s | 1.1 s | **2.8 min** | 14.7 min |
| **15k tok/s** | | | | | | |
| t1_bug | 7.2 min | 4.0 min | 23 s | 1.4 s | **4.4 min** | 23.3 min |
| t2_rename | 15.6 min | 8.0 min | 45 s | 2.3 s | **8.8 min** | 43.2 min |
| t3_feature | 4.9 min | 2.3 min | 40 s | 1.1 s | **3.0 min** | 12.4 min |
| t4_config | 7.7 min | 4.5 min | 21 s | 1.3 s | **4.8 min** | 26.7 min |
| t5_parse | 6.3 min | 3.2 min | 41 s | 1.1 s | **3.9 min** | 16.9 min |

- The chip dominates: generation is 7-34% of the total, and the delta feed is about 1-2 s.
- Range across single runs, row 24: 2.0-6.3 min at 23k, 2.8-9.3 min at 15k.
- A second turn adds the re-feed, a few seconds, plus its own chip time.

### Scripted stand-in

**Not Qwen.** Hand-written plans in `fxio/scripted.py`, in Qwen's raw format, using Qwen's real
tokenizer. Each copies the tool order of a real P0c run, with reasoning as filler sized to that
run's recorded output tokens. A "pass" here means a correct plan got through fx, not that a model
solved anything. These figures are kept because they exercised the path before any GPU was
available, and because the plans include things real runs did not (a planned failing edit in both
rows).

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

**Scripted projection, row 24 (E).** Superseded by the real one above. The tokens are measured, the rates are not this system's:
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

**Chunked prefill on real base weights (M, `fxio/prefill_check.py`, 2,272-token real render).**

| comparison | max logit difference | top-1 equal | KL |
|---|---|---|---|
| `_qwen_prefill` vs the model's forward | 0.125 | True | -2.2e-08 |
| chunks of 64 vs one chunk | 0.1875 | True | 3.1e-05 |
| `_qwen_step` token by token vs one chunk | 0.1875 | True | 1.3e-06 |

- It behaves: top-1 agrees everywhere and the differences are at bf16 rounding scale.
- **Not bit-exact:** the top-10 order differed once (token-by-token vs chunk), and a greedy run from
  chunks of 64 left a run from one chunk after 63 of 64 tokens. So chunking can change
  which tokens Qwen writes, the same class of effect as batching. speedups §4 said so; this is the
  measurement.
- A snapshot restore reproduced the greedy continuation exactly: True.
- `_qwen_prefill` was 133x faster than `_qwen_step` here: 0.86 s against 115 s.

**`_text_hc` on the merged RV checkpoint (M, `fxio/merged_check.py`,
`divagr1925/Qwen3.5-4B-Qwenputer`).** A hook took the true residual after the 11 chip layers as
the reference.

| filling for the 23 kept dims | max logit difference vs forward | top-1 equal |
|---|---|---|
| the true residual, per position | 0.0 | True |
| `_text_hc`, one constant set (the last position's) | 0.0 | True |
| `_text_hc`, zeros | 0.015625 | True |

- Qwen's embedding dims come through the chip layers unchanged (max error 0.0), and every other
  chip dim is 0.0 after them.
- The kept-dims risk from RMSNorm did not show: those dims peak at 9.5e-07 here, too small to matter.
- **Scope, and it matters:** a zero-state chip with no firmware, on the first 640 tokens, because the
  full model peaks at 18 GB and a 2,272-token forward ran out of the L4's 22 GB. The real
  case is text arriving while fx is parked, with a loaded machine; the kept values there were not
  measured, and tau and mu may be larger. Executor v4 was not used either.

No chip that runs fx on the circuit exists yet, so `fx_session` still runs against a replaying chip.

## Blockers

1. **Capture window.** Qwen to the CPU is 80 tokens in the circuit. Real Qwen needed 795 tokens and
   3,314 bytes in one call (M), so a `write_file` is well past it. speedups §13's 4-8k-token
   buffer is required before any of this runs in weights.
2. **The file store and grep / glob** are still on the host (Disclosed). Real Qwen used grep on t2
   and t4 and never used glob, so those tasks still leave the guest for fx's own tool work.
3. **The in-weights tokenizer** is unbuilt, and exact BPE needs merge ranks.
4. **`render_rv.c`** was checked on scripted sessions only. It is meant to handle parallel tool
   calls, which real Qwen uses, but that case has not been compared against the host.
5. **Snapshots in weights:** real snapshots are 84-128 MB, one per message. Whether Qwen's state can be
   saved and restored on the real path is not measured, and every new fx turn needs it.
6. **A non-canonical generation mid-session** has not occurred, so its real rollback is untested.
7. **`_text_hc` with a loaded machine:** see §4. The merged-checkpoint check has no firmware in it.
8. **Sample size:** two reps per task, on tiny repos. Larger repositories are untested.

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
6. The live run: `fx/fxio/live.sh 2` on the GPU box (setup, row 24, row 11, units replays), then
   `python -I fx/fxio/report_live.py RUNS live.json`.
7. Qwen's text path: `python fx/fxio/prefill_check.py Qwen/Qwen3.5-4B MODEL RUN.json OUT.json` and
   `python fx/fxio/merged_check.py CHECKPOINT MODEL RUN.json OUT.json 640`, with `PYTHONPATH` at
   `experimental/riscv`.
