# The chip as a file: Executor v4 runs from `chip.safetensors`

Status 2026-11-08, branch `fx-chip` (from `fx-int` 11dc532). Code: `chipformer/rv/chipfile.py` (the
format and the loader: production), `chipbuild.py` (writes the file from the Python circuit: build
time only), `expand.py` (the dense checkpoint back from base plus the file), `chipcheck.py` (the
audits), `tests/test_rv_chipfile.py`, and small hooks in `exec4.py`, `chat.py` and `rvbatch.py`.
Raw numbers: `chipfile.json` next to this file. **M** = measured here, **E** = estimated, **R** = read
from another doc. Every GPU number is an NVIDIA A10G (q35-eval2, 80 SMs), so the speeds are not
comparable with `exec-v4.md`'s L4 figures; each speed below has an A10G baseline from the same box.

## Result

**Production runs from the file.** `RvChat` loads base Qwen3.5-4B's own text weights and
`chip.safetensors`, and Executor v4's circuit is lifted from the weights in that file. Nothing on
that path builds the circuit in Python. A test runs `RvChat` in a process where
`chipformer.rv.core`, `chipformer.q35.emit` and `chipformer.rv.hf` cannot be imported at all. In
that process the calc firmware is loaded from the file, Qwen calls `calc`, the chip computes, and the
answer is right (M, `test_rvchat_runs_from_the_file_beside_base`).

| | before (`core.build`, merged checkpoint) | **now (base + `chip.safetensors`)** |
|---|---|---|
| what the executor's circuit comes from | Python (`core.build`), checkpoint chip weights unread | **the chip file's weights, lifted** |
| files | 18.72 GB merged checkpoint | **8.41 GB base Qwen (its own files) + 12.2 MB chip file** (M) |
| chip file, rv-q4p2 | | 12,222,537 B on disk, **1.28 MB zstd-19** (M) |
| chip file, fx (boot image included) | | 47,465,656 B on disk, 2.82 MB zstd-19 (M); 51.4 MB / 3.91 MB with the text bridge (M) |
| VRAM after load, with Qwen | 18.95 GB allocated (M) | **8.63 GB** allocated (M) |
| VRAM peak in a calc conversation | 19.46 GB allocated, 19,389 MiB `nvidia-smi` (M) | **8.93 GB** allocated, 9,181 MiB `nvidia-smi` (M) |
| load time | 76.5 s (M) | **19.4 s** (M) |
| chip tokens/s inside RvChat (4 calc prompts x 2) | 13,978 (M) | **13,978** (M) |
| dense checkpoint rebuilt from base + file | | **all 8 shards' SHA-256 equal the Hub's LFS hashes: 18,715,279,016 B, 580 tensors** (M) |

- **Exact.** On the GPU, running from the file, with 0 differences everywhere (M):
  - rv-q4p2, residual: every dim against `ir.evaluate` on 63 programs, 64,735 positions.
  - rv-q4p2, free-running: the 63 programs, 6,424,726 tokens.
  - The paged circuit at fx's RAM size (1,024 frames, 42 layers): residual on 64,690 positions;
    free-running 6,424,294 tokens.
  - fx's whole session: 10,508,989 tokens, 7,174,817 instructions, 400 segments, 0 wrong tokens.
    At the end, 0 of rv32emu's 1,897,586 memory words differ.
  - HF verify mode on rv-q4p2: 5 records, 53,663 chip tokens. The residual after the chip layers
    is bit-identical at every position, and the executor's tokens equal the HF path's on all 5.
- **No quantisation.** Qwen's weights are base's bytes. After cutting the merged checkpoint
  down, all 427 of its tensors are bit-identical to base's (M).
- **Speed.** Running from the file is 0.2-0.4% slower per step than running from `core.build`,
  interleaved in one process (M: c-sieve 69.92 against 69.67 us, c-qsort 70.04 against 69.91 us).
  Inside RvChat the rate is the same (13,978 tok/s both). See [Speed](#speed) for fx.

**A finding on the way.** On `fx-int`, `core.build()` is no longer rv-q4p2's circuit. It has 3,055
dims, 104 heads and 44,490 units against the checkpoint's 2,918, 103 and 42,114, because the macro
units came in. `spec.py` also inserted the CW and EB families before the control tokens, so the
control ids moved. So on `fx-int`, before this change, `RvChat` on the published checkpoint rebuilt
the wrong circuit; its code columns did not even match the checkpoint's. Running from the file
removes that failure: the circuit comes from the checkpoint's own weights. For the same reason,
every rv-q4p2 audit here runs on `fx-exec`'s tree, the one whose `core.build` is rv-q4p2's circuit
(the lint below proves that). That tree carries this branch's `exec4.py`, `chipfile.py`, `chat.py`
and tools copied over it. `memory.md`'s "the legacy map and the rv-q4p2 circuit are unchanged"
holds for `fx-mem`'s base, not for `fx-int`.

## The file

One safetensors file. Its `__metadata__` has `format` (`qwenputer-chip`), `version` (1), `meta` (a
JSON object), `config_hash` (the SHA-256 of `meta`), and `content_hash` (the SHA-256 of every
array). The loader refuses a file when the version is wrong or either hash does not match (M, tests).

| arrays | what | rv-q4p2 | fx |
|---|---|---|---|
| `weights.rows`, `.cols` (u16), `.values` (BF16), `.off` | **the nonzeros of all 14 tensors of every chip layer, at their row and column, bf16 as in the checkpoint**; `meta.tensors` gives each tensor's name, shape and fill value | 1,044,349 terms, 6.27 MB | 6,485,826 terms, 38.9 MB |
| `emb.family`, `emb.basis.*` | the chip tokens' embedding rows: a family stores its base row and one delta per payload bit when its rows are linear in the bits (checked on every row of families up to 4,096 and 2,048 sampled rows above), else every row; the ballast is in `meta.layout` | 21.5 KB | 22.4 KB |
| `text.*` | Qwen's rows' chip columns, as `merge.text_rows` writes them: one plain row, a byte table (16 bytes of each of base's tokens), the byte columns, and 262 special rows (`<tool_call>`, `</tool_call>`, the printable tokens) whole | 4.0 MB | 4.0 MB (text bridge only) |
| `norm.*` | the final norm's chip gains (the output code, tau) | 88 B | 84 B |
| `mute.sinks` | the anti-sink keys in `k_proj`'s mute column on Qwen's 8 full-attention layers: fitted when the checkpoint was merged, so base cannot give them | 16 KB | (no dense file) |
| `state.*` (optional) | a firmware: every nonzero state row, ballast as 2 bits, the rest as bytes, any other value apart | calc-bridge parked, 1.9 MB | fx's boot image: ROM, image, registers, 8.4 MB |

The firmware is state, not weights: ROM, RAM and registers are what `core.snapshot` writes into the
heads' recurrent state. The file carries one so a release has a program to run. Any other ELF still
loads through `core.snapshot` and the reference loader, as an OS would load it.

**What the executor derives from the weights** (`chipfile.ChipCircuit`):
- **Nodes.** Each head is one key head's rows: `q`, `k`, `v` (with conv taps), `z`, `a`, `beta`, and
  its `out_proj` columns. Each MLP unit is a `gate`, `up` and `down` triple. The emitter's scaling is
  undone: `x * c` on the input side, `K` on gates, `1 / K` on the a-form, constants on the ballast
  columns, and `beta` from `sigmoid(256 g - 128)`.
- **Dims.** A residual column hands its slot from one value to the next with a unit
  `-x * relu(ONE)` out of `x`'s own column. Each such handover starts a new version of the column,
  and the versions are exactly the IR's dims. rv-q4p2: 2,918 lifted against 2,918 in the IR, every
  one paired, none left over (M). The exit's clears are handovers too, and the units that remove the
  ballast write only ballast columns, so they drop out.
- **Banks** (fx's ROM, image, page tables and RAM). A bank's address form (gate, row, head, key
  gate) is solved from row 0 of head 0 and the rows and heads at powers of 2. Every other row of
  every head must then match the form's prediction, or the load fails. So no memory weight goes
  unread. A changed RAM query weight is caught at load (M, test). The forms equal `core.build`'s
  (M: 17 banks on fx with the text bridge, 0 differences).
- **The decode table.** Each token's output code is in its own embedding row, which is how the tied
  output head decodes. The table is that map inverted, and a control's family id covers every
  payload. It equals `exec4.build_tables` on all 2,097,152 codes (M).
- **Phases.** `exec4.schedule`, run on the lifted nodes. The plan comes out the same size as from
  `core.build`: 13 phases, 980,863 term slots on 80 SMs (M).
- **Checks.** Every term must be read, so weights on an unused value head fail the load. The
  constants the kernel keeps in code must hold in the file: chip-layer norms 0, `dt_bias` -110,
  `A_log` 0, the gated norm 1. A changed `dt_bias` is refused (M, test).

**What cannot come from the weights, so `meta` stores it (disclosed):**
- the layout: hidden size, the first chip column, the ballast dims and the ballast's value;
- the output code's columns in bit order, and how many are family bits;
- the chip vocabulary size and the chip token offset;
- head names and bank membership: hints only, checked against the rows;
- what a snapshot writes: register and PC columns, the ROM and image plans, and the memory map
  (`MemCfg`);
- tensor shapes and fill values.

The kernel's semantics are in code, not in the file: relu for silu, decay by 1, 0.5 or 0, the
one-row write, the 2-hot read, and the ballast norm.

## The proof that production does not build the circuit

`tests/test_rv_chipfile.py`, on the box (M):

| test | what it shows |
|---|---|
| `test_production_path_needs_no_builder` | a fresh process with `chipformer.rv.core`, `q35.emit`, `chipbuild`, `q35.prims` and `rv.ref` all unimportable loads the file, builds Exec4's plan and tables, takes the state from the file, and runs a program: every token equals the reference's. With a GPU it runs on Exec4, else in `ir.evaluate` |
| `test_rvchat_runs_from_the_file_beside_base` | `RvChat(base, chip=...)` in a process where `core`, `emit`, `chipbuild`, `prims` and `hf` cannot be imported: Qwen's weights under 9 GB, the calc firmware from the file, 123456789 * 987654321 answered right |
| `test_lift_is_the_circuit` / `test_paged_banks_are_solved_from_the_rows` | every head, unit and bank form equals `core.build`'s under the dim map |
| `test_plan_is_the_same_size` | the same phases, terms per phase and heads per phase from the file as from Python |
| `test_tables_equal_the_python_tables` | the decode table and embedding rows equal `exec4.build_tables` |
| `test_residual_in_the_ir_equals_the_python_circuit` | `ir.evaluate` on the lifted circuit equals it on the Python one, every dim |
| `test_state_round_trip` | the file's firmware is `core.snapshot`'s state, row for row |
| `test_a_bad_file_is_refused`, `test_changed_arrays_are_refused`, `test_a_changed_constant_is_refused`, `test_a_changed_memory_weight_is_caught` | forged metadata, a changed weight, a changed constant, a changed bank row: all refused |

`core.build` stays as the build tool (`chipbuild.py`) and the audit (`chipcheck.py`,
`RvChat(python_circuit=True)`, `replay`). Production imports `core` only to load a new program with
`core.snapshot`.

## Step 1: the executor frees what it does not read

`RvChat(merged, executor=True, base=...)` loads the rv-q4p2 checkpoint, frees its 11 chip layers,
and cuts Qwen's layers and embedding back to 2,560 columns. The norm gains come from base's files,
since the widened gains do not invert exactly. That leaves Qwen's 427 tensors, bit-identical to
base's (M). After load it holds 8.63 GB allocated (M), against 18.95 GB before (M). From base
directly, `RvChat(base, executor=True)`, it is the same 8.63 GB with a 19.4 s load, against 29.0 s
through the merged file (M). A text token's residual is its embedding row: the output code stays
in the executor's kept values, and Qwen's layers never see a chip token, so the mute columns never
fire.

The HF chip path (no executor, or `verify`) keeps the whole merged model.

| VRAM, A10G (M) | params | allocated after load | peak allocated | `nvidia-smi` at the end |
|---|---|---|---|---|
| merged checkpoint, `core.build` (before) | 18.715 GB | 18.948 GB | 19.460 GB | 19,389 MiB |
| merged checkpoint, cut down, chip file | 8.412 GB | 8.629 GB | 8.929 GB | 9,181 MiB |
| base + rv-q4p2 chip file | 8.412 GB | 8.629 GB | 8.929 GB | 9,181 MiB |
| base + fx chip file (text bridge), no conversation | 8.412 GB | 8.744 GB | 8.744 GB | 8,655 MiB |

`size.md` estimated 8.5 GB for this step. The measured 8.63 GB is Qwen's 8.41 GB plus about 0.22 GB
for the executor: the decode table, the embedding rows, the term slots and the state. The fx
chip's state adds 110 MB more (R, `memory.md`; 8.74 GB here, M). The peak adds Qwen's KV cache and
activations.

## Step 2: `expand.py`, the dense checkpoint back

`expand.py CHIP BASE --hub divagr1925/Qwen3.5-4B-Qwenputer` rebuilds every shard, header included, a
tensor at a time and in chunks. It hashes each shard and compares the SHA-256 with the Git LFS
hash the Hub publishes for that file, so no shard is downloaded.

**All 8 shards equal the published files: 18,715,279,016 bytes, 580 tensors** (M, `main`). Byte for
byte against the box's local copy of the checkpoint too (M).

It covers every tensor of the checkpoint:
- the 154 chip-layer tensors, from the file;
- Qwen's 32 layers with their zero padding, from base's bytes by `q35.merge`'s widening rules;
- the mute column, by `q35.mute`'s rule, plus the stored anti-sink keys;
- the rescaled norms, the final norm with the chip gains, and the embedding (Qwen's rows, their
  chip columns, the chip rows, the zero padding rows).

It does not cover `config.json`, the tokenizer files or `chipformer.json`, which are not weights.
So the file holds exactly the chip's weights: together with base's bytes, nothing more is needed
to reproduce the published checkpoint.

## Exactness

All on the A10G, the executor running from the file (M):

| check | circuit | result |
|---|---|---|
| lint: lifted nodes against `core.build` | rv-q4p2, paged 1024, fx, fx + text | 0 head and 0 unit differences; dims 2,918 / 2,886 / 2,987 / 3,226, all paired |
| residual (`chipcheck tf`): every dim against `ir.evaluate` of the Python circuit, 4,000 positions or the whole stream | rv-q4p2, 63 programs | **64,735 positions, 2,917 dims (plus ONE), 0 differing** |
| free-running (`chipcheck free`) | rv-q4p2, 63 programs | **6,424,726 tokens, 0 different** |
| HF verify (`execcheck`): HF chip layers drive, the file-loaded executor beside them | rv-q4p2, 5 records (`check_recs.jsonl`) | **53,663 chip tokens, 0 residual mismatches; the executor's tokens equal the HF path's on 5/5** |
| residual | paged, 1,024 frames, 42 layers, 63 programs | **64,690 positions, 2,885 dims, 0 differing** |
| free-running | the same | **6,424,294 tokens, 0 different** |
| fx's whole session (`chipcheck fx`), boot state from the file | fx, 59 layers | **10,508,989 tokens, 0 wrong; memory at the end 0 of 1,897,586 words differ, registers equal** |
| residual on the session's first 4,000 tokens | fx | **2,986 dims, 0 differing** |
| the file's boot state against `core.snapshot` of the same machine | fx | 0 of 27.6M state values differ (110 MB) |
| decode table and embedding rows against `exec4.build_tables` | rv-q4p2 | 2,097,152 codes and 230,584 rows, 0 different |

The 5 records were decoded on another GPU at another batch size, so their own tokens diverge from
this box's at text positions (the first differences at 36 to 254 tokens), in both the HF path and
the executor alike. That is Qwen's bf16 arithmetic on another kernel, as `exec-v4.md` notes, and
it is why the comparison is HF against the executor on the same box.

"Paged 1024" is `spec.small_cfg(1024)`: the paged machinery at fx's frame count (32 RAM layers), with
the legacy addresses, so the 63 existing ELFs run on it. fx's own map puts ROM at 0x02000000, so
those ELFs do not run there; fx's session is its test.

**Qwen's text** (`chipcheck qwen`, `textcheck.py`'s prompts and method, M). Base's greedy
continuations are teacher-forced, and KL(base || this) is compared with KL(base sdpa || base eager):

| | KL mean | noise floor (sdpa against eager) | greedy identical | top-1 agrees |
|---|---|---|---|---|
| RvChat's Qwen (base's files, `chipfile.load_base`), a whole sequence at once | **0.0** | 5.16e-4 | **16 / 16** | **100%** |
| RvChat's text path: the prompt at once, then `_qwen_step` a token at a time | 4.82e-4 | 4.79e-4 on the same positions | | 99.02% |

The first row is exactly zero: these are base's weights and base's arithmetic. The second row is
the usual prefill-against-decode rounding (one token at a time against a whole chunk), at the floor.

An earlier pass of this check gave KL 4.6e-4 and 12 / 16 greedy identical. The cause: my loader cast
the rotary embedding's `inv_freq` to bf16, which `from_pretrained` does not do. The loader now
leaves buffers alone and casts only base's few fp32 parameters, as `from_pretrained(dtype=bf16)`
does. The text checks above are after that fix.

## Speed

A10G, the same box, best of runs (M):

| | from `core.build` | from the file | |
|---|---|---|---|
| rv-q4p2 c-sieve, interleaved in one process, best of 6 | 69.67 us / step | 69.92 us | +0.4% |
| rv-q4p2 c-qsort, the same | 69.91 us | 70.04 us | +0.2% |
| paged 1024 c-sieve / c-crc32, file first then Python | 45.28 / 45.03 us | 46.34 / 45.72 us | +2.3% / +1.5% (not interleaved) |
| fx's whole session, separate processes (`fxrun run` against `chipcheck fx`) | 48.15 us, 20,769 tok/s | 49.64 us, 20,144 tok/s | +3.1% (not interleaved) |
| fx, the session's first 300,000 tokens, interleaved in one process, best of 6 | 47.56 us | 47.97 us | +0.9% |
| inside RvChat, calc, 4 prompts x 2, chip tokens/s | 13,978 | 13,978 | 0 |

The plan is the same size: the same phases, term slots, shared memory and value heads. Two things
do differ: the dims' order in the residual, and so which shared-memory banks the gathers hit. The
lift orders dims by the level where each version begins, then by column. That is close to the IR's
creation order but not equal to it. The non-interleaved rows mix in run order: the first program
pays for warm-up.

Load: the lift takes 0.85 s on rv-q4p2 and 5.4 s on fx (M). For comparison, `chipbuild`, which is
`core.build` plus the sparse emission, takes 13.6 s and 50 s (M). Exec4's plan then takes 9.2 s and
33-36 s (M), the same work as before. `RvChat` from base loads in 19.4-26.0 s over two runs. The old
path took 76.5 s: the 18.7 GB merged file, then `core.build`, with Exec4's tables from its disk
cache (M).

## Honesty: what is true now

- **The file contains** every nonzero weight of the chip's 11 (or 59) layers, the chip part of
  the embedding (in a lossless compact form that `expand.py` turns back into the checkpoint's exact
  bytes), the final norm's chip gains, the anti-sink keys of the mute column, the layout and field
  metadata listed above, and optionally a firmware's state.
- **What comes from it on the production path:**
  - every head, unit and bank, every dim and phase;
  - the embedding rows the executor feeds back;
  - the decode table, and the firmware state when the file carries one.
- **What does not come from it:**
  - Qwen's weights: base's own files, unchanged.
  - The tokenizer: base's.
  - The kernel's semantics: code.
  - The metadata above: stored, not derived, and disclosed.
  - A new program's state, through `core.snapshot` and the reference ELF loader (an OS's job).
  - fx's host side: serving an import from the log.
- **The release is a patch on base Qwen's weights** when distributed by reference: base
  Qwen3.5-4B's files from `Qwen/Qwen3.5-4B`, plus a 12 MB chip file (1.3 MB compressed) for
  rv-q4p2. The weights of the dense Qwenputer checkpoint are exactly base's bytes plus this file;
  `expand.py` proves it against the published hashes.
- **What is lost against the dense file:** one checkpoint that stock `from_pretrained` loads with
  no code. The executor (or `expand.py`) is needed. The `trust_remote_code` HF class from
  `size.md`'s step 2 was not built.
- **Not done here:** a dense fx file (none exists to compare with), fx with the text bridge on a
  real session (its file lints and loads; `fxlive.py` is another branch's), L4 timings, and any
  quantisation (the user ruled it out).
- `fx-int`'s `core.build` cannot rebuild rv-q4p2 (above), so the rv-q4p2 audits ran on `fx-exec`'s
  core, spec and ref. The production path needs neither tree's builder.
