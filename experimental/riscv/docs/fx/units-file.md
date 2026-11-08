# fx's 13 hot-code units as weights: the unit circuit in a chip file

Status 2026-11-08, branch `fx-units` (from `fx-int` a604830). NVIDIA L4 (q35-eval), Executor v4.
Raw numbers: `units-file.json` next to this file. **M** = measured, **E** = estimated. Every number
below is M unless marked.

## Result

**Yes. The 13-unit circuit of `one.md` and `circuits2.md` now emits as finite bf16 weights. It is
written as chip files, and fx runs from them exactly, with no added executor phase (14, as before).**

- **The cause was one node family: SHA-256's step thermometer.** The 6 bad weights are the gates of
  `h.ri>81`, `h.ri>82` and `h.ri>83`, two terms each. They sit in layer 0's `gate_proj`, all on the
  step counter's column. Their values are -2^128, -2^129 and -2^130. That is the gate weight
  K * 8 * 2^m / c (K = 2^7, 1/c = 2^37) for m = 81, 82 and 83: a thermometer that reads 84 steps down
  a 2^-k counter. The same 6 weights appear in both configurations checked: fxrun's 13-unit map and
  the live configuration.
- **The fix is a constant moved from the gates into the counter.** The step counter's head writes
  2^16 * 2^-k instead of 2^-k (its out weight, 2^7 to 2^23), and the thermometer's coefficients are
  8 * 2^(m - 16). Every gate input is the same number as before, so the circuit computes the same
  values. It has the same nodes, dims and levels, so the same phases. The largest weight in the
  circuit is now 2^114 (2.08e34), against bf16's limit of about 2^128.
- **A second, smaller blocker sat behind the first.** The units' token family M (16,384 rows) has
  tag features that are not linear in its payload bits. `chipbuild` stored such a family row by row
  only up to 4,096 rows, so it refused M. It now stores up to 65,536 rows whole; the whole embedding
  basis is 0.99 MB of columns in the live file. The loader already read any size.
- **A third fix is for loading a program onto a unit file.** `core.snapshot` gives the units' 23
  heads (30 value heads) their ballast row only when the circuit has its Python `Hot`. A file now lists those heads
  (`meta.snapshot.hot_tables`), so `core.snapshot` on the lifted circuit writes the same state as on
  the Python one. The files' own boot states were right either way: they come from the Python build.
- **Other builds are node for node identical** (`circ_digest.py`'s digest). These were checked:
  legacy, legacy + text (rv-q4p2's), the small paged map + text, and fx's map without units at
  4,096 frames (`pr.md`'s circuit, 161 layers). The three fx circuits with units (fx, fx + text,
  fx + text without GO) differ in exactly 85 nodes: the `h.ri` head and its 84 thermometer nodes.
  Their dims, node counts and stats are equal.

| | `fx-int` | `fx-units` |
|---|---|---|
| weights outside bf16 / fp32 (13-unit circuit) | **6** (-3.4e38, -6.8e38, -1.4e39, two each) | **0** |
| largest weight | 1.36e39 (inf in bf16 and fp32) | 2.08e34 (2^114) |
| chip file with units | cannot be written | written, lifted, exact |
| Executor v4 phases, from the file | n/a | **14** (the Python build: 14) |

| from the unit chip files (GPU) | result |
|---|---|
| fx's whole session (fxrun map, 13 units, 4,096 frames, 16 MB) | **6,666,925 tokens, 0 wrong**; 0 of 1,897,552 end words differ, registers equal; the file's state equals `core.snapshot`'s (0 values differ) |
| residual against `ir.evaluate`, fx's first 4,000 positions | **4,981 dims, 0 positions differing** |
| the units' test programs (8 programs, every unit kind) | free-running: **13,069 tokens, 0 programs differing**; residual: 0 differing on 7, and on FIXED a counter below its declared `tiny` ([below](#exactness-from-the-file-gpu-m)) |
| live t1, real Qwen, from the live unit file | **solved**, 0 wrong over 3,552,515 audited chip tokens, rv32emu lockstep clean |
| live t1 with `--text`, from the `--text` unit file | **solved**, 0 wrong over 3,550,642, lockstep clean, 2 of 2 calls end in `</tool_call>` |
| speed, the whole session | **24,008 tok/s, 277.7 s** from the file; 23,910 tok/s, 278.8 s from `core.build` on the same box; `one.md` 23,914 tok/s, 278.8 s |

## The six weights (M)

`fx/fxio/units_weights.py` emits the circuit through chipbuild's sparse path without writing a file,
and lists every weight that is not a finite bf16 value with the node and term it came from. On
`fx-int`:

| tensor | row, column | value | node | term |
|---|---|---|---|---|
| `model.layers.0.mlp.gate_proj.weight` | 253, 3146 | -3.40e38 (-2^128) | `h.ri>81` | 0 (`7 - 8 * 2^81 x`) |
| same | 254, 3146 | -3.40e38 | `h.ri>81` | 1 (`6 - 8 * 2^81 x`) |
| same | 256, 3146 | -6.81e38 (-2^129) | `h.ri>82` | 0 |
| same | 257, 3146 | -6.81e38 | `h.ri>82` | 1 |
| same | 259, 3146 | -1.36e39 (-2^130) | `h.ri>83` | 0 |
| same | 260, 3146 | -1.36e39 | `h.ri>83` | 1 |

- Column 3146 is the residual column of `h.ri[2]`, the SHA-256 step counter's output: x = 2^-k,
  where k is the number of MN tokens since the unit's MP.
- The layout: hidden 4096, 64 ballast dims, so c = sqrt(hidden / m) / 2^40 = 2^-37. A gate row
  carries K / c times the IR coefficient (`emit.row`).
- The same six weights, at the same rows and column, are in both the live configuration
  (`live_chip.py --units all+ --frames 4096`, I/O ROM, 16 MB window; 1,460 slots) and fxrun's
  13-unit map (`fxrun.load(..., frames=4096, units=all+)`; 1,459 slots).
- On the test map (`tests/test_rv_hot.py`, hidden 4864, 1/c = 2^36) it is 4 weights: `h.ri>82` and
  `h.ri>83`.
- **Why only SHA-256.** `core.thermo` reads a 2^-k counter with coefficients 8 * 2^m. A weight
  K * 8 * 2^m / c with 1/c <= 2^37 (`pick_hidden` at Qwen's 2,560 columns gives hidden = m * 4^k
  with k >= 3) passes 2^128 only when m > 80. The other thermometers are the record counter j (66
  steps), the refresh count n (63) and the text bridge's capture (80, m up to 79: 2^126, in range).
  SHA-256's has 84 steps.

## The fix, and why it is exact

```python
# core.py
def counter(self, name, reset, step, scale=0):    # out weight 2^7 * 2^scale: the head writes 2^scale * 2^-k
def thermo(self, x, reset, n, name, scale=0):     # gates 7 - 8 * 2^(m - scale) x and 6 - 8 * 2^(m - scale) x
# hot.py
SHA_SCALE = 16
xs = c.counter("h.ri", self.tMP, self.tMN, SHA_SCALE)
self.st = c.thermo(xs, self.tMP, SHA_END + 1, "h.ri", SHA_SCALE)
```

- **Same values everywhere.** The counter's state rows are unchanged (A_BIT * 2^-k: a snapshot
  writes them as before). Only its read is scaled, by the head's out weight. Each thermometer gate
  is 7 - 8 * 2^(m-16) * (2^16 x) = 7 - 8 * 2^m x, the same number. Nothing else reads `h.ri[2]`.
  `ir.evaluate`'s exactness checks pass (bf16 at every materialized tensor, silu's exact range):
  `test_rv_hot.py` 18 passed on the laptop, including `test_ir_sha` and the new weight test.
- **The residual stays small.** The counter's largest value is now 2^16 (at k = 0), against
  `ir.RESIDUAL_MAX` 2^28. Its square, 2^32, vanishes next to the ballast's 64 * 2^80 in the norm.
- **Why 16.** It has to be at least 4 (m - scale <= 80). 16 leaves 2^13 of headroom on the largest
  weight and 2^12 on the residual bound.
- **No phase.** The same nodes at the same levels, one head's out weight and 168 gate coefficients
  changed. The plan from the file is 14 phases and 849,730 term slots, exactly the Python build's
  (fxrun map).
- **Other circuits do not move.** `thermo` keeps int coefficients when `scale` is 0, and only
  SHA-256 passes a scale. The digests are above.

The options the brief listed, and why this one:
- splitting a term across units or layers costs units or a level;
- a different encoding of the step (a one-hot ring) costs a head and its write chain;
- "only compared" is what a thermometer already does, but the comparison's own coefficient was the
  problem.

Moving a power of 2 between the two sides of one product costs nothing.

## The chip files

Built on the box with `fx-units`' code (sha256 in `units-file.json`):

| file | configuration | bytes | terms | layers | phases | sha256 |
|---|---|---|---|---|---|---|
| `fx-live-u13-f4096.safetensors` | live: 4,096 frames, 16 MB window, I/O ROM, 13 units, boot state | 124,960,497 | 19,071,656 | 162 | 14 (850,058 slots) | `48ddb06c...` |
| `fx-live-u13-f4096-text.safetensors` | the same with the text bridge (`--text`, `go_on_tcc=False`) | 129,060,613 | 19,065,792 | 162 | 14 (843,880 slots) | `fda824ef...` |
| `fx-u13-f4096.safetensors` | fxrun's map: 4,096 frames, 16 MB, 13 units, fxrun's boot (no I/O ROM) | 125,004,105 | 19,071,309 | 162 | 14 (849,730 slots) | `62f212f1...` |

- The live file is what `live_chip.py --chip-file` loads. It is refused unless its boot hash is this
  run's boot snapshot, its ELF hash is `282e53d4...`, and its memory map is this run's.
- The fxrun-map file is fxrun's boot without the I/O ROM, as in `chipfile.md`'s fx file, at 4,096
  frames, with the 16 MB window and the 13 units. It is what `chipcheck fx` / `fxtf` run fx's
  logged session on.
- Smaller than `pr.md`'s no-units live file (132,640,220 B, 20,657,783 terms): with units on,
  `core.build` prunes the logic of uops the ROM does not hold (`hot.rom_absent`, `circuits2.md`).
- The three files are kept on q35-eval's root disk, `~/units_chip/` (the NVMe is wiped on stop), and
  are not in git.
- The files were first written before the `hot_tables` fix. The rewrite has the same arrays (equal
  `content_hash`); only `meta.snapshot.hot_tables` was added.

## Exactness from the file (GPU, M)

| check | file | result |
|---|---|---|
| lift against `core.build` (`chipcheck lint`) | fxrun map | 2,496 heads, 3,924 unit groups, 17 banks: **0 differences**; every IR dim that has a column is paired |
| Executor v4's plan from the lifted nodes | fxrun map | **14 phases, 849,730 slots**, as the Python build's (`fxrun`, same box: 14, 849,730) |
| fx's whole session from the log (`chipcheck fx --frames 4096 --units all+`) | fxrun map | **6,666,925 tokens, 0 wrong**, 4,142,400 instructions, 400 segments; end: **0 of 1,897,552 words differ, registers equal** |
| the file's boot state against `core.snapshot` of the reference's boot machine | fxrun map | **0 values differ** (325 MB of state) |
| residual, the session's first 4,000 positions (`chipcheck fxtf`) | fxrun map | **4,981 dims, 0 positions differing** |
| the units' test programs (`fx/fxio/units_tf.py`), free-running from GO | a file per program | **8 programs, 13,069 tokens, 0 different**; 982 M-family tokens among them |
| the same programs, residual against `ir.evaluate` | a file per program | 7 programs: **0 positions differing**. FIXED: 1,280 of 8,254 positions differ in one counter dim, below its declared `tiny` (below) |
| live t1, real Qwen, `--chip-file` | live | **solved**, 3 steps; audit **3,552,515 chip tokens, 0 wrong**; 42,512 memory words, 0 differ |
| rv32emu with the units' twins, in lockstep | live | requests 3 of 3 identical, host I/O hash equal, final repo equal; end: 0 registers and 0 of 54,528 words differ |
| rv32emu on fx's original code | live | requests, host I/O and repo equal; 425 ABI-dead register values and 27 words on the C stack below sp differ, as in `one.md` |
| live t1 with `--text`, real Qwen through the circuit's text bridge | live + text | **solved**, 3 steps; audit **3,550,642, 0 wrong**; lockstep clean; 2 of 2 calls end in `</tool_call>`, none in NUL |
| Executor v4 from the file against Executor v4 from `core.build`, residual (`units_tf.py --python`) | FIXED, SHA-256, random0 | **bit for bit equal** on all 10,259 positions |
| Qwen's merged side built from the `--text` file against the Python build (`wide_file_check.py`) | live + text | **426 tensors, 0 differing** (bit for bit; 2.16 GB embedding included), config equal, code / tau / mu columns equal. The anti-sink keys were fitted once and used by both |
| `tests/` on the box | | `test_rv_hot` 18 passed; `test_rv_chipfile` 13 passed, 1 skipped (rv-q4p2's RvChat test: no `CHIP_Q4P2` on this box); `test_rv_mem` 66 passed, 10 skipped (`RV_SLOW` not set); `test_fx_live` 11 passed; `test_fx_pr` 16 passed (its GPU token test included) |

**The residual difference in FIXED.** All 1,280 differing positions are in one dim, `h.k[2]`, the byte loops' 2^-k commit counter (`hot.Hot`, `c.counter("h.k", ...)`). There the IR holds at most 5.9e-39 (2^-127, a subnormal in fp32) and the executor holds 0. That is below the counter head's declared `tiny`, 2^-90 (outputs below it may round, `ir.Head.tiny`). Its only reader, a 3-step thermometer, gives the same answer for both. Executor v4 built from the Python circuit gives the same residual as Executor v4 from the file, bit for bit, on FIXED, SHA-256 and random0 (`units_tf.py --python`, M). Both free-run FIXED's 8,253 tokens exactly. So this is the executor's fp32 against the IR's float64 far below a declared tolerance, not the file.

- **Why the test programs ran pruned.** They cannot run on Executor v4 unpruned: the test map's
  circuit with every unit is 5,410 dims wide and asks for 101,648 B of shared memory, more than an
  L4's 101,376. So each program's circuit is ROM-specialized to its own ROM (`hot.rom_absent`), as
  fx's is. That leaves 3,702 to 3,754 dims, with 32 uops pruned.
- **The IR dims without a column** (191 on fx's map) are dims that pruning left behind. No node
  writes or reads them, so they have no residual slot (M on fx's map: 0 written, 0 read, 0 slotted).

## Speed and tokens against `one.md`

All on q35-eval's L4, Executor v4; `one.md`'s figures are from the same box.

| | `one.md` (13 units, Python circuit) | **from the unit chip file** | `core.build`, this box, this code |
|---|---|---|---|
| whole session, tokens | 6,666,925 | **6,666,925** | 6,666,925 |
| whole session, GPU s | 278.8 | **277.7** | 278.8 |
| whole session, tok/s | 23,914 | **24,008** | 23,910 |
| us / step | 41.82 | **41.65** | 41.82 |
| phases | 14 | **14** | 14 |
| live t1 rep 1: fx chip tokens | 3,448,213 | **3,448,213** | |
| live t1 rep 1: chip tok/s (fx code) | 23,820 | **23,600** | |
| live t1 rep 1: task s (chip + I/O + Qwen + host) | 165.1 | **168.4** (146.1 + 1.2 + 20.9 + 0.2) | |
| live t1 `--text`: audited chip tokens | 3,550,642 (`fwd.md`) | **3,550,642** | |
| live t1 `--text`: chip tok/s, GPU peak | 23,715, 17.26 GB (`one.md`'s text run, before `fwd.md`'s fix) | **23,736, 17.26 GB** | |

- **The token saving is unchanged, because the circuit is unchanged.** The session is the same
  6,666,925 tokens, and t1 the same 3,448,213. Against `live.md`'s no-units figures that saving is
  `one.md`'s 34.7% on 71 steps (R). It was not re-measured here: this run is one task.
- **Speed from the file is the Python build's, within noise.** The whole session ran 0.4% faster by
  tok/s, one run each, not interleaved. `chipfile.md` found file runs 0.2-3% slower on other
  circuits, so read either direction as noise. Live t1 from the file ran at 23,600 tok/s against
  `one.md`'s 23,820 (-0.9%, one run each).
- **Load.** The lift and the plan take 15.6 s and 158.6 s (live file); the plan dominates. For
  comparison, `fxrun`'s `core.build` + snapshot + plan took 233.6 s to its `loaded` line (M).
- **Wall time.** The live t1 wall time, 343 s, includes the 174 s load and the lockstep emulators
  running beside the chip.

## What changed in the code

- **`chipformer/rv/core.py`.**
  - `counter(..., scale)` and `thermo(..., scale)`. The coefficients stay ints when the scale is 0,
    so every other circuit is node for node unchanged.
  - `snapshot` reads the units' heads from `c.hot` or, on a lifted circuit, from `c.hot_tables`.
- **`chipformer/rv/hot.py`.** `SHA_SCALE = 16` on SHA-256's step counter and its thermometer.
- **`chipformer/rv/chipbuild.py`.**
  - A family that is not linear in its bits is stored row by row up to 65,536 rows (was 4,096).
  - `meta.snapshot.hot_tables` is written when the circuit has units.
  - `merged_fx(..., extra)` carries the run's metadata.
  - `standalone_layout()` is factored out.
  - `fx --frames N --units ...`.
- **`chipformer/rv/chipfile.py`.** `ChipCircuit.hot_tables`, checked against the file's tables.
- **`chipformer/rv/chipcheck.py`.** `fx` / `fxtf` take `--frames` and `--units` (`fx_load`).
- **`chipformer/rv/wide.py`.** `wide_state(..., cf=)` builds Qwen's merged side from a chip file with
  the text bridge. That covers the layout, the mute column, the code and tau columns and their
  gains, and the text rows (`file_text_rows`, the inverse of `chipbuild.text_arrays`).
- **`fx/fxio/live_chip.py`.**
  - `--text --write-chip-file` writes the text-bridged file (`chipbuild.merged_fx`).
  - `--text --chip-file` runs it, with both the circuit and Qwen's merged side from the file.
  - A file with the text bridge is refused without `--text`, and one without it is refused with it.
- **New tools in `fx/fxio/`.** `units_weights.py` (the diagnosis), `units_tf.py` (the test programs
  from files), `wide_file_check.py`, `pr_local.py` (the PR flow with a local remote), and
  `report_units.py`.
- **New tests.**
  - `test_rv_hot.py::test_unit_weights_fit_bf16`: the units' circuit emits, and every weight is
    bf16 and below 2^120. On `fx-int` it fails with an overflow.
  - `test_rv_chipfile.py::test_units_circuit_is_a_file`: write, lift and lint the file. Then check
    that `core.snapshot` on the file equals the Python circuit's for every unit head. With a GPU, a
    program of copies, SHA-256 and eqlBytes runs free from the file to the reference's tokens.

## The PR flow from the unit file, with no GitHub (optional, M)

`fx/fxio/pr_local.py` runs `pr.md`'s issue #1 job on the box from the live unit file, with real
Qwen. It uses pr_server's `Handler`, the same code the laptop runs, but in place of GitHub there is a
local bare git repository holding P0c's files and the sandbox's README, and a recorded fake in place
of `gh api POST /pulls`. **Nothing was pushed anywhere, and no PR was opened.**

| | `pr.md` #1 (no units, its 161-layer file) | **this run (13-unit live file)** |
|---|---|---|
| fx's tool calls | read, edit, **PR** | read, edit, **PR** (one `open_pull_request`, ok) |
| fx's title | "Fix total() function to include all prices in the sum" | the same title |
| change | `money.py` | `inventory/money.py`, 1 line, committed on `fx/issue-1-local`; main unchanged |
| hidden check on the branch | pass | **pass** |
| steps | 4 | 4 |
| chip tokens (fx code) | 7,756,533 | **5,031,008** |
| chip tok/s (fx code) | about 26,100 | 23,599 |
| audit | 0 wrong | **0 wrong** over 5,163,675; 49,881 memory words, 0 differ |
| rv32emu lockstep | identical | requests 4 of 4, host I/O and repo equal, the PR call replayed (rv32emu 1, original-code rv32emu 1); end: 0 registers and 0 of 65,536 words differ |

- One run each, seed 1. `pr.md`'s run was on q35-batch, this one on q35-eval (both L4s). The 35% fewer
  chip tokens is these two runs (M), not a rate. Both made the same calls with the same title.
- The prompt is `pr_tool.PROMPT` with `pr_driver.ISSUE_TITLES["t1_bug"]` and the task's prompt as
  the body, as `pr_driver.py` builds it from the sandbox's issue. Here it is built offline.

## Disclosures

- **fx's source and binary are unchanged.** `fx/check_src.sh` finds 717 files, 0 changed against
  757bdb9. The ELF's sha256 `282e53d4...` is in every run file.
- **One run per configuration**, seed 1, t1 only, with real Qwen. That shows the files drive the
  live loop exactly; it is not a rate. `one.md`'s 15-run sample was not repeated from the file.
- **The `--text` path from a file is new code** (`wide.py`'s file branch). It has run one live task
  and the tensor check above. Qwen's merged layers are still `wide.py`'s, not a merged checkpoint:
  as in `one.md` and `fwd.md`, no fx-sized dense checkpoint exists.
- **Some inputs still come from Python on these paths**, as `pr.md` says. These are the memory map
  (`cfg`) the host's LiveCPU uses, and the reference machine of the offline audit. With `--text`
  there is also the anti-sink fit of the mute column, from base Qwen, as `wide.py` always did. None
  of these is the circuit.
- **One field of `wide.py`'s returned metadata differs from the file:** `slots`. A file stores the
  layout, not the slot count, so the file branch reports an upper bound (1,774 against 1,719). No
  tensor depends on it.
- **The test programs ran on pruned circuits**, one file each, because the unpruned test circuit
  does not fit an L4's shared memory under Executor v4. The unpruned circuit's emission is checked by
  `test_unit_weights_fit_bf16`, and its IR exactness by `test_rv_hot.py`.
- **Not done:**
  - an HF-kernel run of the unit circuit (there is no merged fx checkpoint);
  - t2-t5 from the file;
  - speed interleaved in one process.

## Reproduce

On q35-eval, with `fx-units` under `/opt/dlami/nvme/units/u` beside `row24/` (`fx-fastcopy.elf`,
`imports.json`), `rv32emu.so` and `l13.npz` (`one.md`'s 13-unit log, from `~/onres`). `M` is base
Qwen3.5-4B's snapshot. Run everything from `experimental/riscv`.

1. **The weights before and after.** Run `python fx/fxio/units_weights.py fx l13.npz ELF 4096` on
   each tree. For the live configuration: `... live ./rv32emu.so ELF row24/imports.json M OUT
   --tasks t1_bug --backend scripted --full-regs --frames 4096 --units all+ --write-chip-file X`.
2. **Digests.** `python fx/fxio/circ_digest.py M l13.npz ELF OUT.json` on both trees.
3. **The live file.** `python fx/fxio/live_chip.py ./rv32emu.so ELF row24/imports.json M out/cf
   --tasks t1_bug --backend scripted --full-regs --frames 4096 --units all+ --write-chip-file
   chip/fx-live-u13-f4096.safetensors`. Add `--text` for the text file.
4. **The fxrun-map file.** `python -m chipformer.rv.chipbuild fx --log l13.npz --elf ELF --frames
   4096 --units all+ --out chip/fx-u13-f4096.safetensors`.
5. **Lift, residual and session checks.** `python -m chipformer.rv.chipcheck {lint,fxtf,fx}
   chip/fx-u13-f4096.safetensors --log l13.npz --elf ELF --frames 4096 --units all+`.
6. **The test programs.** `python fx/fxio/units_tf.py OUT.json [--only fixed sha --python]`.
7. **Live t1.** `python fx/fxio/live_chip.py ./rv32emu.so ELF row24/imports.json M out/t1f --tasks
   t1_bug --reps 1 --full-regs --frames 4096 --units all+ --chip-file
   chip/fx-live-u13-f4096.safetensors`. Add `--text` with the text file.
8. **Qwen's side from the file.** `python fx/fxio/wide_file_check.py M
   chip/fx-live-u13-f4096-text.safetensors OUT.json` followed by step 7's live_chip arguments, with
   `--backend scripted --text`.
9. **The JSON.** `python fx/fxio/report_units.py RESULTS docs/fx/units-file.json`.
