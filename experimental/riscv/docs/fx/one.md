# fx on the chip with the circuits on: frames, units, text, long sessions

Status 2026-11-08, branch `fx-one` (from `fx-int` 1689d0d). NVIDIA L4 (q35-eval), Executor v4, real
Qwen3.5-4B. Raw numbers: `one.json` next to this file, built by `fx/fxio/report_one.py` from the runs
listed at the end. **M** = measured, **E** = estimated. Every number below is M unless marked.

The runs were made by the `fx-one` agent; it stopped before writing this doc, so this was written from
its committed code and the results copied off the box (`report_one.py` output, logs, JSON).

## Result

- **15 of 15 runs solved with all 13 units on** (t1-t5 x 3 reps), with the hidden checks; 0 wrong
  chip tokens, no run stopped.
- **The same 71 steps cost 34.7% fewer chip tokens**: 87.24M against `live.md`'s 133.69M for its 15
  runs (which also took 71 steps in total). Median task wall time 248.8 s (range 155.7-758.8 s).
- **One long session: all five tasks as five turns of one fx session, solved, 15 steps, 1,113 s**,
  with 4,096 frames and a 16 MB linear-memory window. The first attempt with a 4 MB window faulted
  (below); that was the window, not the frames.
- **Qwen's text through the circuit (`--text`): 5 of 5 solved.**
- **Text on a machine loaded with fx state stays at about the noise floor** on the production
  text path; a plain one-shot forward does not, on one long sequence (see Text).

## Per task, units on (u13) against `live.md` (no units)

Means over the 3 reps. Steps vary between the two samples, so per-task times are rough; the
71-step total above is the like-for-like comparison.

| task | steps | task s, units | task s, live.md | Qwen s | frames mapped |
|---|---|---|---|---|---|
| t1_bug | 3 | 159.3 | 264.9 | 16.9 | 485 |
| t2_rename | 8 | 482.6 | 689.4 | 42.6 | 785 |
| t3_feature | 4 | 294.1 | 317.0 | 44.0 | 622 |
| t4_config | 5.3 | 245.5 | 330.3 | 22.0 | 579 |
| t5_parse | 3.3 | 202.4 | 270.9 | 26.2 | 527 |

- Chip rate 23,804-23,831 tok/s in every run. Peak GPU memory 14.0 GB (17.3 GB with `--text`).
- Task time is what a user waits: chip (fx and I/O), Qwen, host. It leaves out the circuit build,
  the lockstep emulators and the offline audit.

## Checks, every run

- rv32emu with the units' twins in lockstep: requests, host I/O and final repo identical; all
  registers equal at every ecall; 0 words differing at the end.
- A second rv32emu running fx's original code (no units): requests identical, host I/O hash equal,
  repo equal. Only ABI-dead registers differ, and memory differs in 27 words, all on the C stack
  below sp, which is dead state (as in `circuits2.md`).
- Offline audit against the reference machine: 0 wrong chip tokens over 89.4M audited tokens (u13),
  27.3M (text) and 24.1M (chain).
- fx's source: `check_src.sh` 717 files, 0 changed; ELF sha256 `282e53d4...` on every run.
- Whole session from the log, 13 units, 4,096 frames: 6,666,925 tokens, 0 wrong, 0 of 1,897,552
  end words differing, 23,914 tok/s, 278.8 s on the GPU.
- Tests on the box on the final code: `test_fx_live.py`, `test_rv_hot.py`, `test_rv_mem.py`: 93
  passed, 10 skipped.

## Frames and the long-session fault

- **4,096 frames** (the most a 12-bit `PA` payload names): 156 layers (162 with the text bridge),
  still 14 Executor v4 phases. Chip state 312 MB (325 MB with 16 MB of linear window).
- **Speed does not drop with frames**: the same 940,743-token segment ran at 23,651 tok/s with
  1,024 frames and 23,912 tok/s with 4,096 (`sp1024.json`, `sp4096.json`).
- **The chained session's first run faulted** after 7 steps: `ChipFault: store out of range`. fx's
  wasm linear memory had grown past the map's 4 MB window (`live.md`'s and `circuits2.md`'s map). The
  fix widens the linear window to 16 MB (`fxrun.RW`; the old map is `RW_4MB`); frames, not the window,
  now bound RAM. The rerun used 1,129 frames and reached 4.34 MB of linear memory. Faulting cleanly
  is the designed behaviour: nothing wrapped.
- Frame use: the longest single task mapped 968 frames; the five-task session 1,129 of 4,096.

## Text

`fx/fxio/text_paged.py`, on fx's machine after 1.16M instructions (258 frames mapped), with Qwen's
merged layers (`wide.py`) and the chip on the executor. 17 sequences: 16 short ones and a real
1,108-token fx render.

| path | mean KL against base | noise floor (same positions) |
|---|---|---|
| production text path (`_text_hc`, the machine's last state) | 9.77e-4 | 8.33e-4 |
| the same with the chip state zeroed | 9.77e-4 | |
| one-shot forward of the whole sequence | 0.524 | |

- On the 16 short sequences the forward equals the text path exactly (max logit difference 0.0),
  kept dims peak at 9.1e-13, and code dims stay 0.
- **On the 1,108-token render the one-shot forward diverges** (KL 8.9, max logit difference 2,555)
  while the production path stays near the floor (7.92e-3 against 6.04e-3). Code dims went nonzero
  at 4 positions there. Our reading, not yet checked: that render contains tool-call markup from
  earlier turns, so in a plain forward the chip's bridge reacts to it as if a call had arrived. The
  production path never feeds the chip that way. **A stock one-shot forward of a merged model would
  therefore not be faithful on such prompts.** This needs a direct check before any merged-form claim.
- Of 81.3M state words compared, 4 changed while text was fed.

## What is still host-side

As in `live.md`: template rendering, BPE, parsing, SSE, the file store, grep and glob, the import
handlers and Qwen's snapshots. Added around fx: the I/O ROM and a decode-table stop on HALT and
BYTE. Boot ran on rv32emu; its state is the snapshot. The units are specialised to this ELF's PCs.

## Not done

- wyhash is not built (E: saves about 3.4% of tokens, costs about 2% of step time).
- The divergence of the one-shot forward on the long render is unexplained beyond the reading above.
- No merged fx-sized checkpoint; the chip ships as a sparse file (`chipfile.md`), and the HF kernels
  have not run this circuit at fx size.
- The first `--text` attempt ran out of GPU memory (`text_oom` in `one.json`); the wide model is
  now built in bf16 (`wide.py`) and the rerun is the `text` row.
- Sample: tiny repos, 3 reps; reps 1-2 reuse `io.md`'s seeds.

## Runs

`one.json` keys, all from q35-eval's `/opt/dlami/nvme/one/out`, copied to the scratchpad:
`u13` (15 live runs, 13 units, 4,096 frames, 4 MB window), `u16` (t2 rep 2 with the 16 MB window),
`chain` (the faulting chained run), `chain16` (the chained run with the 16 MB window), `text`
(`--text`), `text_oom` (the first `--text` attempt), `probe` (the text probe), `run13_16` (whole
session from the log), `sp1024` / `sp2048` / `sp4096` (speed against frames). The per-run token
logs (2 GB of `.npz`) are on q35-eval's root disk in `~/one_keep/npz_pt.tar`.
