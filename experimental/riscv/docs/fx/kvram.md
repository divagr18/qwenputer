# fx on Qwenputer: RAM in a full-attention layer's KV cache

Status 2026-11-07, branch `fx-kvram`. CPU only (torch 2.14, transformers 5.19, bf16).
**M** = measured, **E** = estimated. Raw numbers: `kvram.json`. Code: `experimental/riscv/fx/kvram/`.

## Verdict

| question | answer | |
|---|---|---|
| Exact on the HF path? | **Yes**, with kernels that keep fp32 scores and give the max exactly weight 1 (torch SDPA math, torch CPU flash). On fx's whole session (5.40M stores, 8.74M loads) there are **0 wrong loads**, with **4 banks and a store counter per bank**. This is proven for every load from exact integer scores, and the kernels give bit-exact outputs on sampled loads | M |
| Largest exact log | **2^21 = 2.10M stores per bank (and per epoch)**. Proven: every score that matters is a multiple of g below 2^24·g, so fp32 adds it with no rounding. Positions do not matter (tested to 2^30) | M |
| What breaks first | Rounding in the fp32 q·k sum. The first wrong loads come at a 2^23-store counter range: the 128-nat step between consecutive stores rounds to a tie. On fx's trace with one 23-bit global counter, 22 of 1,691 sampled loads are wrong. Other kernels: eager attention (bf16 scores) breaks at 2^8 stores. GPU FlashAttention-2 numerics (P rounded to bf16 after folding in log2 e) never pick the wrong entry below 2^23, but are not bit-exact: bits are off by up to 2^-8 and bytes by 1 | M (FA2: CPU replica) |
| Fixes that work | **Per-bank counters**: 4 kv heads = 4 banks; fx's largest bank has 1.50M stores. **Epochs with compaction**: 3 compactions, 257k re-stores. **Executor-only**: drop the entries whose weight is exactly 0 (non-store positions, superseded stores), or use a hash-indexed read. Both give identical outputs | M / E |
| Viable for fx's session? | **Not on the stock HF path.** Every position adds 4 KB per KV layer, about **200 GB** for the session's ~49M positions. An L4 has ~5 GB left after the 18.7 GB model, which is 1.29M positions. **Viable on the custom executor.** There the exact shortcuts shrink the log to the live words (165k entries, 169 MB, 0.14 ms per load), or to an O(1) gather | E |
| Against DeltaNet word slots | **Word slots:** 23-41 extra layers, 6.9-12.3 GB of checkpoint, a fixed cost per token. **KV-RAM:** 1 layer (0.3 GB), but its runtime memory and its cost per token grow with the log | E |

## 1. The encoding, inside Qwen3.5's attention as it is

What the module does (transformers `Qwen3_5Attention` and the Qwen3.5-4B `config.json`):
- head_dim is 256, with 16 q heads and 4 kv heads (GQA 4:1). The scaling is 1/16.
- q_proj gives q and a per-head output gate. The output is o_proj(attn · sigmoid(gate)).
- q_norm and k_norm are RMSNorms over the head's 256 dims: `bf16(x / rms(x) · (1 + w))`. The w is
  shared by all heads of the layer.
- RoPE (partial_rotary_factor 0.25) rotates dims 0..63 only (pairs i, i+32); dims 64..255 pass through.
- With `eager`, QKᵀ comes out in bf16. SDPA math and CPU flash keep the scores in fp32.

| part | dims | key (a store, before k_norm) | query (a load, before q_norm) | norm multiplier (1 + w) |
|---|---|---|---|---|
| address: 4 one-hot digit groups, bases 64/32/32/32 (21-bit word address) | 160 | one-hot digit | +1 at the digit, -1 at the others | α = 2,164,260,864 (bf16) |
| recency: store counter t, 21 bits | 21 | ±1 per bit | 1 | g·2^i, g = 1024 |
| offset W (store flag) | 3 | 1 | 1 | W split into 2 bf16 parts, -6,492,782,592 in total |
| ballast | in dims 0..63 | integers with Σk² = 256 (15,1,1,1) | Σq² = 256 (8,2,2) | 0 |
| non-store positions | all | 0 (k_proj of a non-store residual) | | |

**Ballast and RoPE.**
- Both norms see rms exactly 1, so they only apply the multiplier. Its bf16 rounding (1 + w within
  2^-10 of the target) disappears in the norm's own bf16 output cast. q and k come out exactly as
  designed (checked through the real modules, `head.py`).
- The ballast's multiplier is 0, so dims 0..63 are exactly 0 in both q and k. RoPE has nothing to
  rotate, so position never enters a score.

**Scores** (q·k units; nats = q·k / 16):

| entry | score | at n_rec = 21 |
|---|---|---|
| store, same address, counter t | W + 4α + g(2t − 2^21 + 1) | ≥ +1,048,640 nats |
| store, ≥ 1 digit differs | ≤ W + 2α + g(2^21 − 1) | ≤ −1,048,640 nats |
| non-store position (k = 0) | exactly 0 | |
| consecutive stores to one word | differ by 2g | **128 nats** |

**What this gives:**
- **Last write wins** by ≥ 128 nats.
- **Any match beats any non-match** by ≥ 2.1M nats.
- **Non-stores get weight exactly 0:** e^-128 underflows fp32, even as a subnormal (below e^-103.3).
- **The winner gets weight exactly 1.0** (`exp(0)`).

**Exactness argument.**
- Every product is a multiple of g. The positive terms of a hit, of an older same-address store or
  of a one-digit miss sum to ≤ 4α + g·2^21 = 10.55M·g.
- That is less than 2^24·g, so every partial sum in any order is an exact fp32 integer. The score
  each kernel sees is the exact integer: 0 mismatches in configs A and E (M).
- With epoch bits (config E), each bit also weighs α. The bound is then 6α + g·2^21 = 14.8M·g,
  still below 2^24·g.

**Values.**
- 32 data bits (0/1), a hit flag, and the 4 bytes (0..255).
- Because the weights are exactly 1 and 0, SDPA math gives back the winner's row bit for bit, bytes
  included.

**Misses and the base image.**
- A word never stored in this log matches nothing. The max is then the non-store score 0, so all 37
  outputs are exactly 0: hit = 0, value = 0. That is fx's value for unwritten memory.
- .rodata (317 KB) is never stored; its 107,690 loads go to a DeltaNet DROM, as in P2, chosen by
  address or by hit = 0.
- The wasm data segments (292 KB) are ordinary stores in the log (the instantiate phase), or
  snapshot entries injected into the cache.

**Banks.**
- Each kv head is a bank, chosen by word-address bits 0-1. The key of kv head j is gated by
  "store to bank j".
- q head 4j is bank j's load port. Its gate is +16 (sigmoid = 1.0 in bf16) at loads to bank j and
  −128 (sigmoid = 0.0) everywhere else. o_proj adds the one open head into fresh residual dims.

**Store counter.**
- The key needs t's bits at the store token. Positions cannot supply them: RoPE is bf16 and periodic.
- The circuit has to carry the counter (not built). Proposal:
  - ST tokens carry the low k bits of their bank's counter;
  - a per-bank GDN register holds the high bits;
  - a CNT token every 2^k stores updates it;
  - the token before a store reads the bank's register and emits ST(c+1).
- With k = 12: 8k extra vocabulary rows and +0.02% tokens (E).

## 2. Precision on fx's real trace

**The trace (M).**
- `gen_trace.py` reruns P0's RV32 session (same host and script) with a full word-level tracer:
  `MT_TRACE` in `rv/memtrace.h`, off by default.
- It records 14.1M events: 5,402,052 word stores (incl. 12,151 host writes) and 8,740,595 word
  loads, over the whole address space (native stack included).
- Last-write-wins on that log reproduces every loaded value:
  - 8,602,368 hits;
  - 30,537 reads of never-written memory, all 0;
  - 107,690 .rodata reads.
- Addresses are remapped to 21-bit word addresses (`trace.compact`).

**Configurations.**
- **A**: 4 banks, a counter per bank, n_rec = 21.
- **G**: one global counter, n_rec = 23, with no fix.
- **E**: global counter, n_rec = 21, epochs with compaction, epoch number as 2 more ±1 key bits.

**Kernels.**
- `math`: torch SDPA MATH, through a chunked replica. The replica is checked bit-identical to
  `torch.sdpa` MATH and FLASH (CPU) on a 200k log.
- `fa2`: a CPU replica of FlashAttention-2's forward numerics (`kernels.fa2`).

**The sample.**
- Per bank: about 420 loads, the full bank log (1.29-1.69M entries incl. one zero-key filler per 8
  stores).
- The loads are the session's last ones, random hits, the oldest winners, the tightest
  same-address gaps, misses, and (for E) compaction loads.

| | A: 4 banks, per-bank counter | G: global counter (no fix) | E: global counter + epochs |
|---|---|---|---|
| KV entries (stores + re-stores) | 5,402,052 | 5,402,052 | 5,659,225 (257,173 re-stores) |
| largest counter / bits built for | 1,498,507 / 21 | 5,402,051 / 23 | 2,084,309 / 21 |
| exact bound / 2^24 | **0.63** | 2.52 | **0.88** |
| analytic, all 8.6-8.8M hits: min margin, latest vs older same-address | 128 nats | 128 nats (exact ints) | 128 nats |
| analytic: hits ≥ / misses ≤ (vs non-store 0) | +1.05M / −1.05M nats | +4.19M / −4.19M | +1.05M / −1.05M |
| sampled loads against the full bank log | 1,691 | 1,691 | 2,048 |
| **math (= torch SDPA math = CPU flash): wrong / bit-exact** | **0 / 1,691** | **22 wrong** / 1,657 | **0 / 2,048** |
| real torch CPU flash (bank 1, 1.29M entries, 256-dim V) | **0 / 420** | | |
| FA2 replica: wrong / bit-exact | 0 / 1,450 (bits ≤ 2^-8 off, bytes ≤ 1 off) | 22 wrong / 1,385 | 0 / 1,693 (same error sizes) |
| sampled hits whose fp32 winner score is not the exact integer | 0 | 1,433 | 0 |
| min fp32 margin, winner vs runner-up | **128 nats** (= exact) | **0** (ties) | **128 nats** |
| highest store score on a miss | −81M nats | −396M | −3.1M |

All numbers are M.
- In G the wrong loads are ties between consecutive stores to one word. The read averages them:
  7 → 6, 980480 → 979968.
- They appear early in the session too. The weights are sized for 2^23 counters, so every score
  is ~2^35 and fp32's spacing there is wider than the 128-nat step.
- In E the first version of the encoding gave the epoch bits half the address weight (a flipped
  epoch bit cost 1·α). The kernel runs caught it: 87 of 512 wrong in bank 0, exact margins
  negative. With the epoch bits at α it is 0 wrong.

## 3. What breaks first: adversarial probes (`probes.py`, M)

**The probe.** 48 queries per counter width n_rec, with counters at the top of the range:
- consecutive same-address stores;
- the 4 one-digit neighbours at the top counters above a same-address run;
- an old winner at counter 0 under top-counter neighbours.

There are 4k random entries and zero-key fillers, at positions up to 2^30. Each cell gives wrong
loads (by threshold) / bit-exact outputs.

| n_rec (log 2^n) | exact bound / 2^24 | math | CPU flash | FA2 replica | eager | min fp32 margin |
|---|---|---|---|---|---|---|
| 7 (128) | 0.00004 | 0 / 48 | 0 / 48 | 0 / 48 | 0 / 48 | 128 nats |
| 8 (256) | 0.00008 | 0 / 48 | 0 / 48 | 0 / 48 | **32 wrong** | 128 |
| 11 (2k) | 0.0006 | 0 / 48 | 0 / 48 | 0 wrong, 32 / 48 exact | 32 wrong | 128 |
| 16 (65k) | 0.02 | 0 / 48 | 0 / 48 | 0 wrong, 45 / 48 exact | | 128 |
| 21 (2.1M) | 0.63 | **0 / 48** | **0 / 48** | 0 / 48 | | 128 |
| 22 (4.2M) | 1.26 | 0 / 48 | 0 / 48 | 0 wrong, 32 / 48 exact | | 256 (rounded) |
| 23 (8.4M) | 2.52 | **32 wrong** | **32 wrong** | **32 wrong** | | 0 (ties) |

**Over n = 2..22 in total (M):**
- math and CPU flash: 0 wrong, all bit-exact.
- FA2 replica: 0 wrong in 1,008 loads, 902 bit-exact.
  - The others are off by 2^-8 on a bit, or by 1 on a byte.
  - Why: FA2 folds log2(e) into the scale, so the winner's p = exp2(rounding error of S·c), not
    exactly 1. Then P is cast to bf16 for P·V, while the row sum stays fp32.

**What fails, in order:**
1. **eager**: from 2^8 stores (bf16 scores).
2. **FA2**: bit-exactness, from 2^11. It still decodes correctly.
3. **fp32 accumulation** in every kernel: from 2^23. Consecutive stores tie and the read averages
   them (on fx's trace in G: e.g. 7 read as 6).
4. 2^22 is past the proven bound but survived the probe: the rounding moved margins to 256 or 0
   nats, and none tied.

The limit depends on the counter width the weights are built for (α ∝ 2^n_rec), not on how far a
session has got: G fails on loads early in the session too.

**Positions** (to 2^30) and the number of non-store entries never mattered, because RoPE dims are
zeroed and non-stores weigh exactly 0.

## 4. The full HF module, gate, o_proj and the bf16 residual (`hfcheck.py`, M)

The test module:
- a stock `Qwen3_5Attention` built from Qwen3.5-4B's text config, with a 1,088-dim synthetic
  residual (bf16) and hand-set q/k/v/o, gate and norm weights;
- all 76,259 instantiate-phase stores injected into a `DynamicCache` through the module's own
  k_proj / k_norm / RoPE / v_proj;
- then fx's events from the start of boot, one filler position per event, chunk by chunk;
- per-bank counters offset so the window ends at counter 2^21 − 1.

| attention (HF `_attn_implementation`) | window | positions bit-exact (output dims of the bf16 residual after o_proj + residual add) | loads | wrong | other residual dims changed | |
|---|---|---|---|---|---|---|
| sdpa, CPU flash backend | 4,000 events, 8,000 positions, after 76,259 cached stores | **8,000 / 8,000** | 2,288 (all bit-exact) | **0** | 0.0 | M |
| eager (bf16 QKᵀ) | 300 events, 600 positions, same prefix | 573 / 600 | 99 (76 bit-exact) | **23** | 0.0 | M |

**At loads**, the residual's 37 output dims hold the winner's 32 bits, hit = 1 and the 4 bytes,
bit for bit: the gate's sigmoid(16) is 1.0 in bf16, and o_proj and the residual add are exact on
0/1 and 0..255.

**At every other position** they are exactly 0: sigmoid(−128) is 0.0.

**What this test simplifies.** The test feeds hidden_states straight into the attention module.
In the merged model the input is the layer's input_layernorm(residual). That norm is the circuit's
business (ballast dims, as in every chip layer), not this layer's.

## 5. Fixes

**Keep the HF module exact:**

| fix | how | cost | fx | |
|---|---|---|---|---|
| **banks with per-bank counters** | kv head j holds word-address bits 0-1 = j; each bank counts its own stores; the gate picks the bank | 4 counter registers instead of 1; the key-side residual dims ×4 (≈ 880 dims, E) | largest bank 1,498,508 stores < 2^21: **the whole session in one epoch** (A) | M |
| more banks (2 KV layers = 8 banks) | the same, over 2 layers | +1 layer (0.3 GB) | largest bank 969,663 | M |
| **epochs / compaction** | at a model-call boundary the machine re-stores every live non-zero word with epoch e+1 and a fresh counter; queries match the epoch exactly, so old entries become misses | tokens: about 1.5M if the live list is known, ~6 tokens per word (E); about 14M (≈ 30% of the session) if found by walking fx's ~800k-word touched range (E) | 3 compactions (before model calls 4, 6, 7): 80,581 + 83,885 + 92,707 re-stores (M); the largest epoch has 2,084,309 stores < 2^21 | M / E |
| run attention as SDPA math or a fp32-score kernel | force the backend for this layer | the KV layer cannot use FA2 | | |

**Executor-only, identical outputs.** These are exact because the losers' weights are exactly 0
(margins ≥ 128 nats):

| shortcut | why it is exact | memory |
|---|---|---|
| drop non-store positions; keep one zero-key sink | on a hit they weigh 0; on a miss the sink alone gives 0/1 = 0 | 1 KB per store (one head, 256-dim K and V) |
| drop superseded stores (dedup by address) | the older entry weighs exactly 0 for its own address and for every other address. One entry per address also means the recency term is never compared, so the counter may wrap and **no epochs are needed** | the live set: 165,315 entries, 169 MB for fx (E) |
| hash-indexed read (address → latest value) | SDPA math gives the winner's row exactly, so the map gives the same output | 8 B per word, O(1) |

What stays on the HF path is the module math (any fp32-score kernel). The cache contents
(DynamicCache) must keep every position, and that is the cost problem below.

## 6. Cost (E, from shapes; L4: 24 GB, ~300 GB/s, 121 TFLOPS bf16)

**KV memory per stored word.**
- HF path: 4 kv heads × 256 × (K + V) × 2 B = **4,096 B per position**, and fx has ~9.1 positions
  per store at 1.3 tokens per instruction. That is **37 KB per stored word** (29-45 KB at 1.0-1.55
  tokens/instr).
- The session: 156-241 GB. An L4 holds 1.29M positions beside the 18.7 GB model.

**Attention per token** (HF: every token runs all 16 q heads over the whole cache):

| log | HF cache (N positions): KV, read per token, FLOPs, ms | store-only, 4 banks (N stores): KV, read per load, ms |
|---|---|---|
| 10^5 | 0.41 GB, 0.41 GB, 1.6 GFLOP, **1.4 ms** | 0.10 GB, 26 MB, 0.09 ms |
| 10^6 | 4.1 GB, 4.1 GB, 16 GFLOP, **14 ms** | 1.0 GB, 0.26 GB, 0.85 ms |
| 10^7 | 41 GB (does not fit), 164 GFLOP, **137 ms** | 10 GB (does not fit beside the model), 2.6 GB, 8.5 ms |
| fx dedup (165k live) | | 169 MB, 42 MB, **0.14 ms** |

All of these are bandwidth-bound. Compute is ≤ 1.4 ms even at 10^7.

**For comparison:**
- The current chip step is 0.18-0.3 ms per token on the executor and ~4 ms on the HF path
  (rv32.md).
- fx's session is ~49M positions. On the HF path the cache alone would be ~200 GB, and at the
  session's end each token would read 200 GB, ~0.7 s.

| | DeltaNet word slots | KV-cache RAM |
|---|---|---|
| layers | 23 (live 93,278 words) to 41 (165,315 written), at 4,096 words per layer | 1 (4 banks), or 2 for 8 banks |
| checkpoint | **6.9-12.3 GB** | 0.3 GB |
| runtime state | 48-86 MB of fp32 recurrent state, fixed | grows with the log: 200 GB (HF), 5.5 GB (store-only), 169 MB (dedup) |
| cost per token | fixed | grows with the log, unless deduped or hashed |
| exact | yes (one-hot slots) | yes, with fp32-score kernels and 2^21 stores per bank-epoch |
| caveat | slots are direct-mapped by address bits, so this count needs a compacted address map | needs a store counter in the circuit |

## 7. Verdict

- **Exactness is not the problem.**
  - With the encoding above, the stock attention module reads fx's memory exactly for the whole
    session: 4 banks with per-bank counters, and SDPA math or CPU flash.
  - The precision limits are understood: 2^21 stores per bank-epoch proven; first failures at
    2^23; eager unusable beyond 2^7; FA2 decode-correct but not bit-exact.
- **Memory and bandwidth are the problem on the HF path.**
  - Every token position, not every store, adds 4 KB per KV layer.
  - fx's 5.4M stores ride on ~49M positions. That is ~200 GB of cache and ~0.7 s of reads per
    token by the end, against 5 GB free on an L4. The HF path is viable only for short sessions
    (≲ 1.3M positions on an L4).
- **On the custom executor it is viable and cheap,** because the exact shortcuts apply:
  - dedup makes the log equal to the live words: 169 MB, 0.14 ms per load;
  - a hash-indexed read is O(1).
  - But at that point the executor's RAM is an ordinary array whose semantics are defined by
    attention.
- **Recommendation for fx:**
  - Keep DeltaNet word slots (or the executor's array) as the RAM of record.
  - Use KV-RAM only where the HF path must run short sessions with more RAM than the slots give.
  - If KV-RAM is built, it needs: one full-attention layer above a relay of address(t−1) and
    data(t−1) (+0-1 layer, E); per-bank store counters carried by ST tokens; SDPA math (not FA2)
    for that layer; and the dedup cache on the executor.

## Reproduce

Scripts are in `experimental/riscv/fx/kvram/`; the P0 scratchpad paths are as in `p0-report.md`.
1. Build the traced emulator:
   `zig cc -target x86_64-windows-gnu -O2 -shared rv32emu_trace.c -o rv32emu_trace.dll`.
2. Record the trace: `python -I gen_trace.py rv32emu_trace.dll fx--Os-...unaligned_scalar_mem.elf imports.json fxtrace.npz`
   (venv with pyelftools + numpy). It takes 2 s and writes 113 MB.
3. Run `python study.py fxtrace.npz study.json --configs A,G,E --per-bank 512 --kernels math,fa2`
   (one process per bank: `--banks B`; it resumes from `study.json.partial.json`).
4. Run `python probes.py probes.json`, the two `hfcheck.py` runs below
   and `python cost.py cost.json`.
5. Run `python study.py fxtrace.npz flash.json --configs A --banks 1 --kernels flash,math` (real CPU flash at full bank length). Then merge: `python merge.py kvram.json study.json probes.json hf.json cost.json flash.json`, where hf.json holds the two hfcheck runs (`--events 4000 --impl flash`, `--events 300 --impl eager`).
