# Executor v4: the RV Qwenputer's chip layers as one persistent kernel

Status 2026-11-07. NVIDIA L4 (q35-eval, 58 SMs), torch 2.14.1 + NVRTC 13.0, the rv-q4p2
checkpoint (`divagr1925/Qwen3.5-4B-Qwenputer`: 11 chip layers, then Qwen3.5-4B's 32). Code:
- `chipformer/rv/exec4.py`: the executor.
- `exec4check.py`: the checks and the standalone timing.
- `exec4chat.py`: v2 against v4 inside RvChat.
- The v4 paths in `chat.py`, `rvbatch.py`, `execcheck.py` and `batchcheck.py`.

Raw numbers: `exec-v4.json` next to this file. **M** = measured, **E** = estimated.

## Result

| | Executor2 (v2) | **Executor v4** | |
|---|---|---|---|
| Inside RvChat, chip tokens/s (production: no lockstep reference) | 3,145 | **23,104** | M, 24 calc records in one run, text tokens interleaved (waterfall row 10; the final kernel: 23,283 on 4 of them) |
| Inside RvChat, RV32 instr/s (calc firmware, 1.50 tokens per instruction) | 2,100 | **15,430** | M |
| Standalone, per step | 190-192 us (graph replay, no decode) | **36.5-37.0 us** (free-running) | M, 4 programs |
| Standalone, tokens/s | 5,200-5,250 | **27,000-27,400** | M |
| Standalone, instr/s | 3,355-4,351 | **17,400-22,650** (1.21-1.56 tok/instr) | M |
| fx turn 1, 4.97M instr, at fx's 1.45 tok/instr | 36 min (at 2.3k/s) | **5.2 min at RvChat's rate (15.9k instr/s); 4.4 min at the standalone rate (18.9k instr/s)** | E (tok/s M, tok/instr E) |
| fx turn 1 with fx's own tools on the CPU, 5.65M instr | 41 min | **5.9 min / 5.0 min** | E |

- **The 15k instr/s target is met** (M).
  - Inside RvChat on the calc firmware: 15,430 instr/s (23,104 chip tokens/s at 1.50 tokens
    per instruction), 7.3x v2 in the same run.
  - Standalone: 17,400-22,650 instr/s.
  - At fx's estimated 1.45 tokens per instruction, RvChat's rate is 15.9k instr/s (E).
- **Exact.**
  - Free-running on the GPU equals the reference machine on 63 programs: 6.42M tokens, 5.63M
    instructions, 0 differences (M).
  - The IR predicts every owned token of those streams: 1.17M tokens over 10 of the programs,
    0 wrong predictions, 0 owner mismatches (M).
  - The residual equals `ir.evaluate` on all 2,918 dims at every position: 64,735 positions,
    63 programs, 0 differences (M).
  - Inside RvChat, v4's token streams equal v2's on all 24 records, with a clean lockstep (M).
  - Plus batchcheck, the HF verify mode and the IR replay of the v4 records (below).
- **No kernel numerics changes a bit**: all values are exact, and the checks above compare
  bitwise (details under Honesty).
- **Memory layers.**
  - Flattening keeps the barrier count at 13 whatever the number of RAM layers.
  - Each further word-slot layer still costs its own work: about 0.75 us per step (M, a fit over
    0-6 extra layers), against about 3.9 us a layer without flattening (M, 0-2 layers).
  - So fx's 23-41 extra layers are not free (E): about 54-67 us a step, 18.6k-14.9k tok/s,
    12.8k-10.3k instr/s at 1.45 tokens per instruction. Without flattening it would be
    143-214 us.
  - Today's per-block tables would not fit in shared memory past about 8 extra layers (see
    Scaling).

## What it is

The plan is `speedups.md` §2. Executor2 (`tritonexec.py`) runs 2 Triton launches per sublayer
(20 sublayers, so 40 launches) under one CUDA graph per token, and RvChat runs Python between
chip tokens. v4 is one cooperative CUDA kernel for a whole chip burst, compiled with NVRTC from
source that is generated for the circuit.

1. **The chip-token loop on the GPU (plan step 1).**
   - After the last phase every block decodes the output code into the next token itself:
     `core.decode_code` as a table over all 2^21 codes.
   - It writes that token's embedding into its residual: `Executor2.embed`'s rows, stored
     sparse.
   - A launch runs until the prediction is not a chip token: a text token, no token, or the
     budget.
   - RvChat calls the GPU once per chip burst (`ExecRow4.burst`) and once per text token.
     RvBatch does the same per row.
2. **One persistent kernel, a grid barrier per phase (step 2).**
   - One block of 512 threads per SM, launched with `cuLaunchCooperativeKernel`, so all blocks
     are co-resident.
   - The barrier is an arrival counter: a fence, an atomic add, then an acquire spin.
   - Each block keeps its own copy of the residual in shared memory.
   - A phase's outputs go into that phase's own buffer, by atomic adds. At the next phase's
     start each block adds in the dims that phase wrote and the block reads.
3. **Heads whose query matches no row (step 3).** A value head whose query is all 0 reads no
   state row and adds nothing (see Exact zeros).
4. **Memory sublayers in one phase (step 4).**
   - The circuit's nodes are list-scheduled by the dims they read and write. A node runs in the
     first phase after every lower-level node that writes a dim it reads. It runs no earlier
     than any node at its level or below that reads a dim it writes.
   - So, on the dims it reads, its input in that phase equals its input at its own level, at
     every step, and its conv (lag) history is the same too.
   - `exec4.check_schedule` asserts this for every pair of nodes.
   - The 20 sublayers become **13 phases**:

   | phase | levels | heads in it |
   |---|---|---|
   | 0 | 1, 2 | pc, count, j, vcnt, run, arm |
   | 1 | 2 | |
   | 2 | 3, 4, 22 | cap |
   | 3 | 4, 5, 7, 9 | **rom0-15 and rom16-31** (both ROM sublayers) |
   | 4 | 8 | |
   | 5 | 9, 10, 14 | regs.rs1 / rs2 / rd |
   | 6 | 10, 12, 16 | |
   | 7 | 11, 12, 14 | bn1, bn2 |
   | 8 | 12-17, 19 | **ram0-15, ram16-31, drom0-15, tbuf** (both RAM sublayers, the data ROM, the text buffer) |
   | 9-12 | 14-22 | |

5. **Sparse weights in L2 (step 5).**
   - Every sparse row is cut into pieces of at most 8 terms: an MLP unit's a and b, a head's
     q / k / v / z rows and its a / beta scalars, over the residual and the lag history.
   - Per phase and block, the pieces are packed best-fit-decreasing onto as few threads as
     hold them at the phase's J terms each.
   - Each term is one 8-byte slot, (shared-memory index | piece << 16, fp32 weight), laid out
     thread-major.
   - That is 966,054 terms in 979,842 slots: 7.8 MB per step, all in L2 (48 MB).
   - The next phase's slots load into registers while the block waits at the barrier.
   - The weights are Executor2's fp32 values, the same `float(cf) * tap`. So the repack is
     lossless by construction, and the residual comparison below checks it.
   - The per-block tables live in shared memory: units, head records and dim lists, about
     45 KB of the 67 KB used. Unit coefficients and output weights are stored as bf16 (all are
     bf16-exact; asserted at load).
6. **Lockstep off (step 6).**
   - `RvChat.generate(lock=None)` runs no reference machine.
   - The lockstep reference is still there (`lock=`).
   - `replay.py` (the IR) audits records offline.

**Per phase and block**, in order:
- the gathers: each thread's pieces, one after another, into shared-memory piece sums;
- a fix-up pass for rows of several pieces, and for the empty rows of a head's 128;
- the lag history moves on;
- the MLP units add `coef * a * relu(b)`;
- the value heads, one per warp, 4 channels per lane, each doing Executor2's head step:
  - the decay;
  - the key's row (argmax, the smallest index on ties) and the one-row write;
  - the query's rows, at most QHOT of them (argmax rounds, each chosen row zeroed);
  - the ballast norm and the gated outputs.
- An argmax round is two warp reductions: `redux.sync` max of the value bits, then min of the
  index.
- Each lane decays, writes and reads only its own 4 channels, so a written row is forwarded
  in registers.

## Waterfall

Standalone: c-sieve (1.393 tokens per instruction), free-running from GO on the GPU, 20k steps
per launch, best of 3. RvChat: the calc records, chip tokens over the time spent on them,
Python included. Every row was checked exact; "tf" is the residual against the IR on every dim
(c-crc32, 3,000 positions, at each step; all 63 programs for the final rows).

| # | change | us / step | tok/s | instr/s | check |
|---|---|---|---|---|---|
| 0a | **Executor2 in RvChat** (baseline: execcheck, 4 records, no lock) | 312 | **3,206** | 2,300 | records' tokens |
| 0b | Executor2 standalone (CUDA graph replay per token, no decode) | 192 | 5,202 | 3,735 | IR, 300 positions |
| 1 | **v4**: one cooperative kernel, nodes flattened into 13 phases, the token loop on the GPU, 256 threads, slots read in the loop | 74.7 | 13,390 | 9,610 | tf; free == reference |
| 2 | + next phase's slots prefetched into registers, tables in shared memory, 512 threads, 8-term pieces | 55.9 | 17,890 | 12,840 | tf |
| 3 | + code generated per phase (sizes built in), only the dims a block reads added | 54.2 | 18,450 | 13,250 | tf |
| 4 | + dense slot packing (no padded slots loaded) | 47.7 | 20,960 | 15,050 | tf |
| 5 | + rows as piece sums with a fix-up, one value head per warp | 45.6 | 21,930 | 15,750 | tf |
| 6 | + argmax rounds as warp reductions | 42.2 | 23,700 | 17,010 | tf; RvChat 20,200-21,600 tok/s on 4 records |
| 7 | + head writes forwarded, read-only ROM loads, head warps skip the units, slots loaded during the barrier, unrolled fix-ups | 38.6 | 25,870 | 18,570 | tf; 63 programs free |
| 8 | + 16-byte head loads | 38.2 | 26,170 | 18,790 | tf |
| 9 | + `o / norm` as a product when norm is a power of 2 (exact) | **36.2** | **27,620** | **19,830** | tf on 63 programs, free on 63, IR 1.17M |
| 10 | inside RvChat (row 9's kernel): RvChat takes each burst whole (one launch, one copy back) | | **23,104** | 15,430 (calc) | 24 records == v2, lock clean |
| 11 | + unit and output tables packed (bf16, asserted lossless; 76 -> 67 KB of shared memory, for the scaling runs): **the final kernel** | 36.5 | 27,390 | 19,660 | free on 63, tf on 63, batchcheck, 4 RvChat records == v2 (23,283 tok/s on them) |
| - | tried and reverted: split each phase's terms into early ones (computed while the grid waits) and late ones | 44.6 | 22,440 | | tf (exact, slower: the late chain stayed 8 terms and the slots grew 1.5x) |
| - | row 11 with step 4 off (one phase per level: 20 phases) | 53.9 | 18,560 | 13,330 | free == reference |

Per-phase profile of the final kernel (M, `prof.py`, block clocks): yadd 0.25 us, gathers
0.4 us, fix-up and lag history 0.25-0.5 us, MLP units 0.2 us, value heads 0.4-1.5 us on the
head phases. The barrier is 0.6-0.7 us from the last arrival to the release, the decode and
embedding 1.3 us per step.

## Exactness

All on the L4. The kernel of waterfall row 9 (commit `5ae07b3`) ran every check; the final
kernel (row 11, which only packs two tables) re-ran free-running on all 63 programs, the
teacher-forced residual on all 63, batchcheck and 4 RvChat records, all exact.

| check | what | result |
|---|---|---|
| teacher-forced residual (`exec4check tf`) | every dim (2,918) of the residual after every step against `ir.evaluate` (float64), 4,000 positions or the whole stream of each of 63 programs (rv32ui 39, rv32um 8, rv32uzba 3, 10 C programs, MicroPython s1 and s5) | **64,735 positions, 0 differing** (M) |
| free-running (`exec4check free`) | each program from GO on the GPU alone (chip tokens and echoes, one launch), against `bridge.Session`'s stream | **63 / 63 identical: 6,424,726 tokens, 5,626,517 instructions** (c-sudoku 3.06M tokens and c-fib 2.17M in one launch each) (M) |
| IR on those streams (`exec4check ir`) | the IR, teacher-forced, predicts every owned next token and owns exactly the reference's positions | **1,165,952 tokens (mpy-s5, c-qsort, c-sieve4, mpy-s1, c-gmpmul, c-sieve, c-sort, c-crc32p, c-crc32, c-strops): 0 wrong, 0 owner mismatches** (M) |
| decode and embedding tables (`exec4check tables`) | the decode table against `core.decode_code` over all 2,097,152 codes; the sparse embedding rows against `Executor2.embed`, bitwise | **0 differences: 2,097,152 codes, 230,584 embedding rows** (M) |
| inside RvChat (`exec4chat --lock`) | v2 and v4 decode the same prompts with no lock; then v4 again with the lockstep reference | **24 / 24 records identical: 156,182 chip tokens in 1,562 bursts and 7,809 text tokens (4,096 of them in arith-0, 42 calc calls); with the lock: 0 lock errors, 0 head mismatches, the same tokens** (M) |
| execcheck (`--exec v4 --no-lock`) | the records' tokens, v4 with and without the lock, against v2's run on this GPU | v4 == v2 on 4 records (dmx_mul_big-5081 also == its record; arith-5 / -39 / -0 differ from their records at the same token as v2: those records came from batch-8 decoding) (M) |
| HF verify mode (`execcheck`, verify) | the HF chip layers drive; v4 runs beside them; the residual after the chip layers compared bit for bit at every position | **0 mismatches** on dmx_mul_big-1618 and arith-18 (2,150 and 2,304 chip tokens): the residual after the chip layers is bit-identical at every position (M) |
| batchcheck (`--exec v2/v4 --batch 4`) | RvBatch, 6 records, 4 rows | **v4's tokens equal v2's on all 6 records** (both dumped and compared); the same 4 of 6 match their batch-8 records with both executors; 0 lock errors (M) |
| IR replay (`replay.py`) | the v4 RvChat records, offline | **24 / 24 ok: 174,472 positions, 157,787 owned, 0 wrong predictions, 0 owner mismatches** (M) |

## Honesty

- **The same arithmetic over the same weights.**
  - v4 runs the circuit's own graph (`core.build`), as Executor2 does, with the same fp32
    weights and the same per-node semantics:
    - relu for silu;
    - decay by 1, 0.5 or 0;
    - the one-row write and the k-hot read;
    - the ballast norm and the gated output.
  - The checkpoint's HF chip layers are the compiled form of the same graph. `execcheck`'s
    verify mode runs both and compares the residual bit for bit (above).
- **No host computes CPU results in production.**
  - With `lock=None` there is no reference machine; the GPU decides every chip token.
  - The decode table is `core.decode_code`, the output code's meaning, not a CPU emulation.
  - The embedding rows are the circuit's token features, as Executor2 and the HF embedding
    have them.
  - The host does what it did before: Qwen's layers for text tokens, the tokenizer.
- **Exact zeros used.** Each is exact, so the results are bit-identical.
  1. An MLP unit whose `coef * a * relu(b)` is +-0 adds nothing. Executor2 adds it, but adding
     +-0 to an output buffer that starts at +0 changes no bit.
  2. A value head whose query is all 0 reads no row and adds nothing: its read is the zero
     row, `ref = 0`, so its output is 0 (Executor2 computes the same 0).
  3. An output column whose weight or gate z is 0 skips its division and its add: its output
     is `x * 0 * w`, +-0.
  4. Empty rows are not stored; their value is 0. A fully empty 128-row block of a head reads
     a shared zero block.
  5. A block adds into its residual copy only the dims it reads (block 0 adds them all, for
     the debug dump). Dims a block never reads may be stale there, and nothing reads them.
  6. Padding slots go to a dummy piece sum that is never read.
- **The repack is lossless.**
  - Each slot holds Executor2's fp32 weight bit for bit.
  - The slots and tables are built from the graph at load time; the checkpoint is unchanged.
  - The decode and embedding tables are cached in `~/.cache/chipformer`, keyed by a hash of
    their sources.
- **Kernel numerics changes.** All values are exact, so none of these changes a bit (checked):
  - Summation order: a row's terms are summed in pieces of up to 8 (in order), then the pieces
    in order. Executor2 sums a padded row with `tl.sum`, a tree. Outputs into the residual are
    atomic adds in any order, as in Executor2.
  - `acc += w * x` may compile to an FMA. w, x and w * x are exactly representable, so the FMA
    rounds the same as a multiply then an add.
  - The read's `o / norm` is a product by `1 / norm` when norm is a power of 2. That is always
    so for this circuit's ballast rows: norm = 2^(21 - n) or 2^22. Both are exact scalings
    with one rounding, so they give the same float. Otherwise v4 uses IEEE division
    (`__fdiv_rn`), as Triton's `div_rn`. `sqrt` is `__fsqrt_rn` (IEEE), as `sqrt_rn`.
  - The argmax rounds compare the bits of non-negative floats as integers (the same order),
    and break ties by the smallest index, as `tl.argmax` does.

## Scaling with memory layers

**How it was measured (M).** c-sieve standalone (as in the waterfall), with N further RAM
layers: each a copy of the 16 heads of the first RAM sublayer (ram0-15). They have the same
weights, so the same queries, keys and row reads; their outputs go nowhere. Each sits at its own
new level, as a further word-slot layer would. The free-running output still equals the
reference on every run.

The kernel here is the final one, whose unit and output tables are packed into fewer words to
fit shared memory (bf16 coefficients and weights, asserted lossless).

| extra RAM layers | flattened (step 4): phases | us / step | tok/s | not flattened: phases | us / step | tok/s |
|---|---|---|---|---|---|---|
| 0 | 13 | 36.5 | 27,390 | 20 | 53.9 | 18,560 |
| 1 | 13 | 37.1 | 26,920 | 21 | 58.1 | 17,210 |
| 2 | 13 | 38.6 | 25,910 | 22 | 61.7 | 16,200 |
| 4 | 13 | 39.0 | 25,660 | (does not fit in shared memory) | | |
| 6 | 13 | 41.3 | 24,240 | (does not fit) | | |

- **Flattened, a layer costs about 0.75 us** (M, a least-squares fit over 0-6 layers), about 2%
  of today's step.
  The barrier count stays at 13. What remains per layer is its work:
  - 123k terms, 1 MB of term slots streamed from L2 every step (979,842 slots become
    1,102,594 with one layer);
  - 32 more value heads over the blocks.
- **Not flattened, a layer costs about 3.9 us** (M, 0-2 layers): a phase each, with its
  barrier, its gathers and the latency of its heads. Flattening alone is worth 1.48x at 0
  extra layers (53.9 -> 36.5 us).
- **So step 4 keeps the phase count constant but not the cost.**
  - fx's RAM needs 23-41 extra layers (E: `speedups.md` §12). At 0.75 us a layer that is
    about 54-67 us a step: 18.6k-14.9k tok/s, 12.8k-10.3k instr/s at fx's 1.45 tokens per
    instruction.
  - Without flattening it would be 143-214 us a step.
  - These are linear extrapolations from 6 layers (E). The slot traffic of 41 layers
    (42 MB a step) would also crowd the L2 that today holds everything (48 MB).
- **Shared memory caps the flattened RAM phase at about 8 extra layers** today. The per-block
  tables (rows, head records) grow about 3.5 KB a layer: 69 KB at 1, 90 KB at 6, against
  99 KB. For fx's RAM, the RAM phase has to run in passes, or keep its head tables in L2.
- **The per-layer work is the lever for fx.**
  - Every RAM bank recomputes the same address decode (q / k rows of 25-28 terms over the
    address bits) for its own 128 rows. Computing the shared part once per phase and only the
    bank selection per bank would cut most of a layer's 123k terms (E).
  - Bigger slots (2-3 words a slot, `speedups.md` §12) cut the number of layers itself.

## What is left

The target is met for today's circuit, so this is about margin and about fx's bigger memory.
The final kernel's step is 36.5 us (M). Where it goes, from the per-phase profile of row 9 (M):

| part | per step | lever |
|---|---|---|
| 13 grid barriers, 0.6-0.7 us each | ~8.5 us | fewer phases: compute the tiny phases (0, 1, 4, 11, 12: 0.4k-13k terms) redundantly in every block, with no barrier; re-place nodes by criticality (`q35.sched`) to cut the depth |
| value heads on the 6 head phases (one warp per head: the S-row read from L2, the norm, the output adds) | ~7 us over the other blocks | keep small heads' state rows (counters, regs, cap, quo) in shared memory: no L2 round; split a block's heads over more warps |
| gathers, fix-ups and Y adds | ~11 us | shorter late chains (split pieces by the terms that read the previous phase, with the fix-up folded in); fewer smem bank conflicts by ordering slots |
| decode and next-token embedding | 1.3 us | the decode table and the frequent tokens' rows in shared memory |
| inside RvChat: Python and one launch per burst (~100 chip tokens per burst on calc) | 27.6k -> 23.7k tok/s | longer bursts (fx's are long), a CUDA-graph launch, or a launch that also runs the next text token's chip step |

- Two conversation rows in one launch (RvBatch) would use the idle lanes of the latency-bound
  phases: throughput, not latency.
- For fx's 23-41 extra word-slot layers see Scaling: the per-layer work has to shrink, and the
  per-block tables must leave shared memory or the RAM phase must run in passes.
