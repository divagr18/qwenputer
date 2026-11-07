# fx on the in-weights CPU with real Qwen, live

Status 2026-11-08, branch `fx-live`. Follows `memory.md` (the paged map, Executor v4) and `io.md`
(the packet path). Code: `chipformer/rv/fxlive.py`, `fxlivecheck.py`, `fx/fxio/live_chip.py`,
`restore_check.py`, `loaded_check.py`, `report_chip.py`, `tests/test_fx_live.py`. Raw numbers:
`live.json`. **M** = measured, **E** = estimated. Everything here is M unless it says E.

## Result

**Yes. fx's exact binary, running on Executor v4 with real I/O, solved P0c tasks with real
Qwen3.5-4B on one L4: 15 of 15 single-turn runs, every one checked by the task's hidden check.** All five tasks, 3 reps each. Two two-turn runs were solved too. No run
stopped and no chip token was wrong.

**This is not one merged model.** The chip is Executor v4 running the 59-layer paged circuit. Qwen
is base Qwen3.5-4B in HF transformers. They share one process and one GPU, and they are two sets of
weights: nothing of fx's chip is in Qwen's checkpoint. No fx-sized merged checkpoint exists. See
[What is in one model](#what-is-in-one-model).

| | | |
|---|---|---|
| Regression: fx's recorded session, log replay | 10,508,989 tokens, 7,174,817 instr, 0 wrong, final memory 0 of 1,897,586 words differ, registers equal | M |
| Solved, t1-t5 | 15 of 15 single-turn runs: rep 1: 5 of 5, rep 2: 5 of 5, rep 3: 5 of 5 | M |
| Solved in two-turn runs (t1, t4), each with a real rollback | 2 of 2 | M |
| Chip tokens the GPU generated, all runs | 147,254,049, 0 wrong | M |
| Chip rate, fx's code between ecalls | 25,813-25,896 tok/s (one launch per stretch) | M |
| I/O tokens as a share of fx's tokens | 0.49% (714,990 tokens); audit registers a further 0.18% | M |
| Pokes into the state after the boot snapshot | **0** | M |
| Identical with rv32emu on the same replies | every run: PC, all 31 registers and instruction count at every ecall, every request body, host I/O hash chain, final repo, final registers, every page rv32emu changed | M |
| Largest capture in a run | 2,844 bytes in one read; the check carried 3,314 and 9,000 | M |
| Typical run time | t1: 203 s chip + 27 s Qwen. Longest: 1,018 s chip, 48 s Qwen (t2, 15 steps) | M |

**Limits of this result.**
- 3 reps per task on small tasks. It shows the path works; it does not bound a failure rate.
- These are the runs of `io.md` moved onto the chip: fx-core's short prompt and 5 host tools, not
  P0c's native prompt. Not comparable to P0c's 15 of 15.
- Sampling as `io.md`: temperature 1, no top-k or top-p, inferred from vLLM's defaults. The seed is
  the rep's number. Reps 1 and 2 reuse the seeds of `io.md`'s rv32emu runs, so they are not fully
  independent of them: t2 rep 1 and t4 rep 1 gave every reply of the recording, the other 8 left it
  at some call. Rep 3 uses a new seed.
- The file store, grep, glob, template rendering, tokenizing, parsing and SSE are host-side
  (Disclosures).

## t1_bug, end to end

`t1_bug` rep 1, the first live run: 204.6 s on the chip, 26.9 s of Qwen and 0.24 s of host, from the
boot snapshot to the end of the session. The hidden check passed on the repo in the host's file
store. fx's own read and edit tools ran on the chip.

| call | kind | Qwen fed | generated | fx called | reply bytes | Qwen s | chip s when it opened |
|---|---|---|---|---|---|---|---|
| 0 | first | 649 | 67 | read_file | 576 | 13.6 | 58.2 |
| 1 | exact | 134 | 199 | edit_file | 1,088 | 8.8 | 101.6 |
| 2 | exact | 27 | 100 | none, final answer | 792 | 4.5 | 185.0 |

Call 0 holds the process's first Qwen forward: 10.9 s of feed here, against under 2 s for a whole
later run.

The final repository (every run's is in `live.json`, sha256 and diff):

```diff
--- inventory/money.py
+++ inventory/money.py
@@ -3,3 +3,3 @@
     s = 0
-    for i in range(len(prices) - 1):
+    for i in range(len(prices)):
         s += prices[i]
```

## How a call works now

Between ecalls the GPU runs alone, as in `memory.md`. What changed is the ecall: the log and the
pokes are gone. The host talks to the machine only by feeding tokens, the way text is fed to Qwen.
Three facts of the circuit make that enough, and none of them is new:

1. A forced `J` or `V JF` moves the PC. It writes rd of the instruction the PC stands on, so the
   host jumps only from an instruction with no rd: the ecall, `BPRD`, a store.
2. A forced value record (`F0`, `V F1`, `V V F2`) on an instruction with an rd writes that
   register: a commit takes its value from the token's payload.
3. A forced `V V V SSx` on an `SH` writes the whole word from the V payloads into the slot the `SH`'s
   EA names.

**No change to the circuit, the kernel or the binary.** `core.py` and `exec4.py` are untouched. Two
additions sit around them:

- **An I/O ROM**: 3,073 words of ordinary instructions at ROM word 0x1BE000, after fx's code in the
  fetch space. It is in the ROM, not in the ELF. Loads never see it.
- **A stop in the decode table**: HALT and BYTE have no embedding row in the live table, so the
  kernel ends a launch when one is predicted, with that token as its last prediction, unfed.
  fxrun's table is not patched.

| I/O ROM | instructions | who supplies the tokens |
|---|---|---|
| emit | `addi rX, rX, 0` | the chip prints a register as its value record |
| read sled, 1,024 words | `lw a7, imm(a0)` | the chip prints a memory word as its value record |
| write sled, 1,024 words | `sh x0, imm(a0)` | the host's `V V V SSx` writes a word |
| set | `addi rX, rX, 0` | the host's forced value record sets a register |
| probe | `lw a7, 0(a0); sw a7, 0(a0)` | the chip: `PA(f)` if unmapped, then `ST0` / `ST1` |
| stop | `BPRD` | predicted, never fed, never commits |

**One ecall**, as tokens: the chip prints `FLT(7) HEX x 8` and predicts HALT, which ends the launch.
The host feeds `V JF` into the I/O ROM, the chip prints a7, a0 and a1, and the host reads the
argument words and any buffer through the read sled. It runs the import with the same handler as
rvhost. Writes go through the write sled. Then it feeds a7, a0 = the result and `V JF` to PC + 4,
and the launch goes on in fx's code.

**The first host write to a page needs a probe.** `SSx` names the frame's value-head parity, and
only the circuit's page table knows the frame. So the chip reads one word and stores it back, which
maps the page as a guest store would and prints the parity. The word it stores is the word it read.
After that the host keeps the parity. In a forced write stretch the host checks the circuit's own
prediction at every `SSx` and `R` position: 51,685 positions across the runs, 0 wrong.

Registers survive because the sleds use only a0 and a7, and the host restores both.

### What goes through the token stream

| transfer | direction | how | tokens | |
|---|---|---|---|---|
| Ecall, its PC | guest to host | `FLT HEX x 8` | 9 | chip's |
| a7, a0, a1 | guest to host | emit | 1-3 each | chip's |
| All 31 registers (audit, every ecall) | guest to host | emit | 1-3 each | chip's |
| Arguments, paths, iovecs, request bodies | guest to host | read sled | 1-3 a word | chip's |
| A page's frame parity | guest to host | probe | about 6 | chip's |
| Resume at PC + 4 | host to guest | `V JF` | 2 | forced |
| a0, a7 | host to guest | forced value record | 1-3 each | forced |
| Reply bytes, stdin, file reads, stat structs | host to guest | write sled | 4 a word, `R` every 63 | forced |

**Pokes that remain: one.** The boot snapshot is loaded into the state once, at the start of a run:
rv32emu's memory at the first stdin read as the image, and its registers. After that nothing
writes `Exec4.S` or the lag history. There is no `ex.poke` and no `ex.reset` in the live path, and
one token stream runs from `GO` to the end of the session.

A forced token is model input, like a prefilled text token: it goes through the embedding and
every layer, and the circuit's prediction at that position is ignored. In a merged model these are
chip-vocabulary input ids that Qwen does not see.

### Cost

| | | |
|---|---|---|
| I/O tokens, per run | 22,520-134,300 | M |
| Audit tokens (all registers, every ecall), per run | 7,182-40,726, not needed in production | M |
| Launches, I/O | 2.4-2.6 per ecall | M |
| Chip seconds on I/O | 0.9-5.4 s of a run | M |
| Host seconds at ecalls | 0.18-0.51 s a run: the handlers and their Python | M |

## The capture window

**Solved for the bytes fx reads.** A reply no longer goes through the 80-row text buffer. It is a
stream of SS records into paged RAM, as long as fx's buffer, at 1 token a byte plus one `R` per 63
words. It is the same path as every other host write.

| GPU check at fx's size (`fxlivecheck`) | result |
|---|---|
| Registers at the boot ecall, through the chip | equal |
| 8 KB out of the image through the read sled | 2,048 words, 0 differ |
| Capture of 3,314 bytes (the largest real reply), unaligned, across pages | read back equal, 0.15 s |
| Capture of 9,000 bytes (past one sled) | read back equal, 0.41 s |
| Every chip token against the reference | 10,795 chip tokens, 12,531 forced, 0 wrong |
| Teacher-forced residual against `ir.evaluate` | 23,326 positions, 2,987 dims, 0 differing |
| The same protocol in the IR, free-running on the big test map | `tests/test_fx_live.py`, 6 passed |

**What this does not do.** Qwen's text reaches the chip through the host's detokenizer, parser and
SSE builder. The 80-token path that lets the circuit capture Qwen's own tokens is unchanged and
unused here. The capture is a host writing bytes with chip tokens, not the circuit reading Qwen's
text.

## Qwen's state

**Save and restore work on the real path, exactly.** `restore_check.py`: HfBackend on real weights,
through `DeltaLink`'s own feed of a real 2,335-token request.

| | |
|---|---|
| Snapshots | 13, one per message boundary, on the GPU |
| Size | 72-128 MB each, 1.3 GB in all (M). A long session holds more |
| Restore | 5-6 ms |
| After restore: logits | bit-equal on all 13 |
| After restore: every cache tensor | bit-equal on all 13 |
| Sampled continuation from the same seed, twice | equal on all 13 |

**Through a live run, on the chip:**

| run | call | kind | full tokens | re-fed | rolled back | kept | restore |
|---|---|---|---|---|---|---|---|
| t1_bug rep 1 (2 turns) | 3 | rollback | 1,076 | 432 | 532 | 644 | 5.3 ms |
| t4_config rep 1 (2 turns) | 4 | rollback | 1,010 | 359 | 432 | 651 | 5.4 ms |
| t2_rename rep 2  | 6 | rollback | 1,197 | 82 | 47 | 1,115 | 6.1 ms |

- **A mid-session non-canonical generation has now happened** (`io.md` blocker 6). On t2 rep 2, call
  5's generation did not re-tokenize to its own ids (M). Call 6 rolled back those 47 tokens to the
  snapshot at the generation's start, fed 82 and went on. The run was solved and stayed identical
  with rv32emu. The restore took 6 ms. This was the only non-canonical generation in 80.
- Scope: this is HF's cache on base Qwen, not a state inside a merged model.

## What is in one model

| | in this run |
|---|---|
| One GPU, one process | yes |
| One set of weights, one forward | **no** |
| The fx circuit in a checkpoint | **no**: Executor v4 builds it from `core.build`. It has not been merged or run on the HF kernels at fx size (`memory.md`) |
| Qwen's layers seeing chip state | **no**: base Qwen has no chip layers, so no kept dims reach it |
| Text through the chip layers | **no**: the standalone circuit has no text path. fx's circuit here is built with `text=None` |
| Chip tokens in Qwen's context | no, as in a merged model: the chip is transparent to Qwen |

So the interaction the brief asks about, text arriving while fx is parked, cannot happen in these
runs. Merging is estimated at 35 GB with 58 GB of RAM (`memory.md`, E) and was not attempted.

**The kept dims with a loaded machine, at small scale (M, `loaded_check.py`).** The rv-q4p2 merged
checkpoint (11 chip layers, the legacy map), with `calc-bridge.elf` parked and injected into the
chip layers' states after 64 tokens, then 512 tokens of a real fx render through the whole model.
A hook takes the true residual; that forward is the reference.

| filling for the 23 kept dims | max logit difference vs forward | top-1 equal |
|---|---|---|
| the true residual, per position | 0.0 | True |
| `_text_hc`, one constant set (the last position's) | 0.0 | True |
| `_text_hc`, zeros | 0.03125 | True |

- Qwen's embedding dims come through unchanged (max error 0.0), and every other chip dim is 0.0.
- The kept dims peak at 9.5e-07, the same as `merged_check.py` found with zero states.
- **Scope, and it matters.** A small firmware: 11 nonzero registers, 49 nonzero RAM words, a parked
  calc, 512 tokens. Not fx's 59 layers, not its map, not a machine with fx's state. It says the
  chip layers keep to themselves with a real program loaded; it does not check fx's own checkpoint,
  which does not exist.

## Runs

| task | rep | solved | steps | chip tokens | instr | Qwen tokens, fed + generated | chip s | Qwen s | host s, ecalls | GPU peak GB | calls exact / rollback | largest reply B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1_bug | 1 | yes | 3 | 5,254,840 | 3,541,691 | 810 + 366 | 204.6 | 26.9 | 0.24 | 9.7 | 2 / 0 | 1,088 |
| t1_bug | 2 | yes | 3 | 5,069,131 | 3,411,404 | 810 + 312 | 197.3 | 14.9 | 0.22 | 9.7 | 2 / 0 | 790 |
| t1_bug | 3 | yes | 5 | 8,415,082 | 5,734,489 | 994 + 471 | 327.6 | 22.7 | 0.27 | 10.3 | 4 / 0 | 990 |
| t2_rename | 1 | yes | 5 | 14,556,511 | 9,622,596 | 1,530 + 924 | 565.2 | 41.3 | 0.40 | 10.6 | 4 / 0 | 1,404 |
| t2_rename | 2 | yes | 15 | 26,287,074 | 18,119,193 | 1,577 + 1,047 | 1023.4 | 48.0 | 0.51 | 13.6 | 13 / 1 | 893 |
| t2_rename | 3 | yes | 4 | 9,345,933 | 6,085,334 | 797 + 558 | 363.9 | 25.2 | 0.29 | 10.0 | 3 / 0 | 1,291 |
| t3_feature | 1 | yes | 2 | 4,868,495 | 3,325,654 | 726 + 769 | 188.9 | 33.2 | 0.18 | 9.5 | 1 / 0 | 2,844 |
| t3_feature | 2 | yes | 4 | 8,593,130 | 5,792,896 | 1,010 + 660 | 333.8 | 28.8 | 0.21 | 10.1 | 3 / 0 | 1,456 |
| t3_feature | 3 | yes | 4 | 8,485,424 | 5,761,060 | 973 + 792 | 329.9 | 35.8 | 0.21 | 10.1 | 3 / 0 | 1,422 |
| t4_config | 1 | yes | 4 | 5,806,127 | 4,002,497 | 765 + 318 | 225.5 | 14.3 | 0.21 | 10.0 | 3 / 0 | 669 |
| t4_config | 2 | yes | 7 | 10,375,298 | 7,208,262 | 999 + 580 | 403.8 | 26.1 | 0.26 | 10.9 | 6 / 0 | 741 |
| t4_config | 3 | yes | 5 | 7,590,775 | 5,256,998 | 963 + 539 | 295.6 | 24.9 | 0.23 | 10.3 | 4 / 0 | 1,027 |
| t5_parse | 1 | yes | 4 | 7,010,625 | 4,780,243 | 883 + 540 | 272.2 | 23.9 | 0.22 | 10.0 | 3 / 0 | 1,159 |
| t5_parse | 2 | yes | 3 | 6,488,512 | 4,371,701 | 779 + 657 | 252.3 | 28.6 | 0.20 | 9.8 | 2 / 0 | 1,727 |
| t5_parse | 3 | yes | 3 | 5,546,366 | 3,759,448 | 779 + 416 | 215.8 | 19.3 | 0.20 | 9.7 | 2 / 0 | 1,249 |
| t1_bug | 1 (2 turns) | yes | 4 | 6,455,212 | 4,405,898 | 1,242 + 448 | 251.2 | 21.6 | 0.24 | 9.8 | 2 / 1 | 1,088 |
| t4_config | 1 (2 turns) | yes | 5 | 7,105,514 | 4,934,794 | 1,124 + 422 | 276.6 | 19.9 | 0.23 | 10.0 | 3 / 1 | 818 |

- Chip tokens, instructions: fx's own code from the boot snapshot to the end of the session. Boot's
  0.53M instructions ran on rv32emu (Disclosures). I/O and audit tokens are counted apart.
- Qwen seconds are the whole model link: render, tokenize, feed, generate, parse.
- Host seconds are what is left of the wall time: handlers and Python, without Qwen, the chip or
  the lockstep rv32emu.
- Chip seconds are launches for fx's code and its I/O, without the audit registers.
- GPU peak is `torch.cuda.max_memory_allocated`: base Qwen, the chip's plan and state, and Qwen's
  snapshots. The worst run held 13.6 GB of the L4's 23 GB.
- Spread is wide because Qwen's paths differ. t2 took 5 steps on one rep and 15 on another (many
  greps, then a read of each file).
- Instructions in 3 runs are corrected by a counter bug, fixed since (Honesty below).

## Comparison with rv32emu

rv32emu ran beside the chip on the same model replies, stepping to the same ecalls, with its own
host and its own copy of the repo. It feeds nothing to the chip.

| check, each run | when | result, all 17 runs |
|---|---|---|
| PC at every ecall | live | identical |
| All 31 registers at every ecall, printed by the chip | live | identical |
| Instruction count at every ecall | live | identical up to the fix below; constant offset 1 (the chip counts the ecall) |
| Every import's a0 | live | identical |
| Every request body, byte for byte | live | identical: 80 requests |
| Host I/O hash chain (every import, read and write) | end | identical |
| Final repository | end | identical |
| All 31 registers at the end | end | 0 differ |
| Every page rv32emu changed, read through the chip | end | 1,158,912 words, 0 differ |

A difference on any live row stops the run and is reported. None did.

**Every chip token against the reference (offline).** Each run's whole stream went through
`PagedMachine` after the run: 149,681,339 chip tokens and 333,120 forced ones, 0 wrong. The
reference's memory then equalled rv32emu's on every word it wrote (881,597 words, 0 differ).
The audit also caught a planted wrong token (test). Nothing in the production path reads the
reference.

## Disclosures

**Host work that stays on the host.**

| | who does it | |
|---|---|---|
| Chat-template rendering, BPE, reasoning and tool-call parsing, SSE construction | the host | a model server's job |
| The file store (`wasifs.py`) | the host | fx reaches files through WASI |
| grep_files, glob_files | the host | fx's work, done by the host: they do not compile for WASM |
| The ACP client (initialize, prompt) | the host | stands in for the user |
| Import handlers: clock, random, env, args, fixed answers | the host | `protocol.py`'s (a) and (c) groups |
| Qwen state snapshots | the host | on the GPU, a prefix cache by another name |

**What changes around fx.**
- fx's source: `fx/check_src.sh`, 717 files, 0 changed against `757bdb9` (M). The binary: sha256
  `282e53d4…`, row 24, recorded in every run file.
- **The I/O ROM and the decode-table stop are new**, and they are not fx's. The stop is a change to
  how a launch ends, not to any token's meaning.
- **fx's boot did not run on the chip in these runs.** rv32emu ran 526,882 instructions to the first
  stdin read and that state is the snapshot. `memory.md` ran boot on the GPU on a larger config.
- Configuration as `io.md`: gateway route, one SSE read, frozen clock, row 24 (no macro units).

**Exactness.**
- No Python reference in the production path. rv32emu (C) runs beside it in lockstep as a check,
  and `PagedMachine` runs after it. Neither feeds the chip. The lockstep costs 0.06-0.33 s a run.
- Exact zeros: only those of `exec-v4.md` and `memory.md`. Forced rows are the table's embedding rows,
  checked equal to `Exec4.embed` on sample tokens, and the residual check covers them.
- A probe stores a word back, so that slot's valid column is set with the value it already read.
  Memory values do not change, and the reference does the same.

**Honesty.**
- **A counter bug, fixed.** The token that crosses a launch boundary (1M tokens) was missing from the
  stretch the host counts instructions over. It changed nothing fed to the chip: the log, the audit
  and every comparison saw it. rv32emu's count showed the drift on 3 runs (1-3 instructions), which is
  how it was found. Those runs are reported with rv32emu's true count. The fix and its test are in.
- The regression ran with `core.py`, `exec4.py` and `fxrun.py` as on `fx-int`: this work does not
  touch them. It ran on the commit's tree before the counter fix, which is host code.
- t1 rep 1's raw host seconds held the circuit's build (33.8 s). The table has it taken out.
- Other work shared the box: IR tests on its CPU during some runs, and a short GPU test at the end of
  the regression. The chip's rate stayed at 25,813-25,896 tok/s, as in `memory.md`.

## Blockers

1. **One merged model.** Everything above is two sets of weights on one GPU. The paged circuit has
   no text path built, has not been run on the HF kernels at fx size, and its checkpoint does not
   exist (about 35 GB, E).
2. **Text on the paged map.** `core.build(text=..., cfg=...)` is not tested, and `TEXT_ADDR` (0x8000)
   sits inside fx's linear memory, so the text buffer must move before the two can be built together.
3. **The frame budget.** t2 rep 2 mapped 968 pages of 1,023. One longer session runs out, and
   then faults. 2,048 frames cost layers but no measured speed (`memory.md`).
4. **The host's text work**, the file store and grep / glob (Disclosures). `io.md` prices the render
   at about 7% of a step.
5. **The chip dominates a run:** 77-95% of the wall time at row 24 with no macro units, 5,698 s
   against Qwen's 455 s over all 17 runs. The units are other work, and they do not run on the
   paged map yet.
6. **Sample size:** 3 reps per task on tiny repositories.

## Reproduce

On a GPU box, with `P` holding `experimental/riscv`'s `chipformer`, `fx`, `tests`, and row 24's
`fx-fastcopy.elf` and `imports.json` in `row24/`:

1. `gcc -O2 -shared -fPIC fx/rv/rv32emu.c -o rv32emu.so`.
2. Regression: `python -m chipformer.rv.fxrun r24log.npz row24/fx-fastcopy.elf run --out regress.json`.
3. The GPU check: `python -m chipformer.rv.fxlivecheck r24log.npz row24/fx-fastcopy.elf --out check.json --tf 30000`.
4. The runs: `python fx/fxio/live_chip.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json MODEL out/live
   --tasks t1_bug t2_rename t3_feature t4_config t5_parse --reps 1 2 3 --full-regs --tmp tmp`. Add
   `--turns 2` for two turns. `--cpu-ref --backend replay:RUN.json` runs the host with the reference
   as the chip, with no GPU.
5. `python fx/fxio/restore_check.py Qwen/Qwen3.5-4B MODEL RUN.json restore.json`.
6. `python fx/fxio/loaded_check.py rv-q4p2 MODEL RUN.json loaded.json tests/data/rv/calc-bridge.elf 64 512`.
7. `python fx/fxio/report_chip.py out/live live.json regress.json check.json restore.json loaded.json`.
8. Tests: `pytest tests/test_fx_live.py` (14 min on CPU, the protocol in the IR).
