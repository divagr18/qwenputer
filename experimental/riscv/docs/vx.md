# VX: the vector unit in the circuit (design B, early layers)

This is the plan the `vx` branch builds. `chipformer/rv/vx.py` is the reference: its token costs
and semantics are the contract, and the circuit has to match it token for token.

## Why the early layers

The core's datapaths sit in layers 5–9 and are nearly full there (MLP free: L5 0.6k, L6 1.7k, L8
1.2k, L9 0.7k; heads: L3–L9 full). The free room is:

- heads: L1 11, L2 16, L10 16, L11 16
- MLP units: L1 about 8.9k, L2 7.8k, L3 8.8k, L4 4.9k, L7 5.1k, L10 7.9k, L11 8.1k
- live dims: low in L1–L4 (about 120–240) against a peak of about 1,090 at L8–L9

Exact arithmetic costs one MLP stage, and so one layer, per carry or threshold level. A multiply
takes about 5 stages and a 64-lane dot product about 6. A unit that reads its operands in layers
2–3 has 8–9 MLP stages before the exit (L11). A unit on top of the core would add one layer per
stage.

## The rule the design follows

1. **Memories:** the unit's memories are its own heads in L1–L3, so its addresses never need the
   register file (L5) or RAM (L7–8).
   - L1 attention queries can use only token features.
   - L2 queries can use L1 outputs, the `j` thermometers included, but only in linear (andpre)
     forms.
   - L3 queries can use L2 MLP outputs, adders included.
2. **Writes:** every write takes its value from token payloads (lags 0..3: up to 64 bits), its key
   from L1/L2 dims at t-1 (the record's config state and `j`), and its beta from the token.
3. **Configuration:** the unit can't see the registers or the decoded instruction. Each vector op's
   record starts with VC tokens (16-bit payloads) that the exit fills from the instruction's
   registers. L1 state heads keep them for the rest of the record.
4. **Depth:** a token does at most one dot product or one multiply chain that fits the stages
   before the exit. An op that needs more takes more tokens per element, and the intermediate goes
   through the token stream.
5. **Trackers:** a stream token is VB instead of VE when its element is the new max. The tracker's
   value is then written from that token's payloads.

## Memories

| Memory | Contents | Rows | Heads, layer |
|---|---|---|---|
| VM | 16 vector registers × 64 int32; bank A = v0–v7, bank B = v8–v15, one read port each | element pairs (64 bits), written every 4th stream token | 2, L2 |
| VA | 256 int8 activations, 4 chunks of 64 | byte pairs in pair-specific columns, read a chunk at a time (multi-hot) | 1–2, L2 |
| KC | K cache: rows (l, g, t) of 8 int8, 1024 rows | 8 bytes written from 4 tokens' payloads | 3, L2 |
| VC | V cache: rows (l, g, t) of 8 int8, laid out so a read of column j over t < 32 is one query | as KC | 3, L2 |
| WROM | int4 weights, one row-chunk (8 words, 64 cells) plus the row scale S_r per value-head row, 4,152 row-chunks | read-only (snapshot) | 17, L3 (+1 L2) |
| state | op config, trackers, RoPE cos/sin | single rows | L1 |

To free L3, the ROM shrinks to 128 KB (16 heads, L4 only). The ROM fetch timing doesn't change.

## Ops (what `vx.py` v3 models)

- **Record:** VC tokens (config), then the stream (VE/VB), then NT.
- **Stream limit:** 128 tokens (JMAX goes from 66 to 130 or more).

| Op | Does | Tokens |
|---|---|---|
| VMAT | out = acc × S_r into a VM register (k = 1: 64 lanes); argmax mode | 2 a row (k = 1), 6 a row (k = 3) |
| VMATR | scores: VA q · K rows (signed max tracked); p·V: VA p · V column j, × r0 >> 12 | 2 an output |
| VQ | a VM register's pairs → int8 pairs into VA or a cache row | 1 per 2 elements |
| VEW | ADD (window × factor, then add); ROPE (pairs, cos/sin from state); GLU (two multiplies chained) | ADD, ROPE: 2 an element; GLU: 4 an element |
| VEXP | exp-table softmax numerators, VM scores → VA | 1 an element |
| VRED | Σ VA² or Σ VA over a chunk | 1–2 |
| VGET | rd = tracker | value record |
| VPUT | VM[v][i] = register | 1 VC + 2 + NT |
| VSETR | RoPE cos/sin into state | VC + payload |

SiLU and exp are fixed-point function units in the MLP (generic activation units, not model data).
The model's weights, scales and caches stay data.

## Build order

1. `vx.py` v3 and `stories_vx.c` on it: tokens and the story.
2. Infrastructure: VE/VB/VC token features and output codes; JMAX; the ROM move; L1 state heads;
   VU decode in the exit; snapshot of the unit's state.
3. Ops one at a time, each teacher-forced against the reference on random programs (the
   `run_check` harness), starting with VPUT/VGET, then VMAT, VQ, VEW, VMATR/VEXP/VRED.
4. Whole-program checks (the IR on prefixes, the executor free-running) and the layer, hidden-size
   and checkpoint measurements.

## Circuit facts that shaped the format (found while building)

- **Runtime writes store bits.** A head's written row holds 0 or A_BIT per column; byte-valued
  rows are allowed only in snapshot-loaded tables (ROM, DROM, WROM). So VM, VA and the caches
  store bits: a 31-bit word is 31 columns.
- **Reads:** keys are one-hot. Queries are multi-hot with equal values, and exact only at
  power-of-4 counts (the kernel scales by 1 / sqrt(count)). A multi-hot read sums the rows, so
  rows must use disjoint columns unless the sum is wanted, and a count stays exact up to 256.
- **Gates:** beta is 1 or an integer <= 0. It can use dims computed below the head at t (a head
  in L2 can gate on L1's counters and config). A head in L1 can use token features only.
- **Keys:** a key comes from dims at t-1 below the head's layer, so an L2 head's key can use L1
  outputs (counters, thermometers, config) in linear (andpre) forms, but nothing from an MLP above
  L1. That is why addresses are concatenations: an element offset is aligned to the op's
  power-of-two length, so element = offset | count.
- **Counters:** a thermometer reaches m < 88 (bf16), so streams are at most 64 tokens. JMAX is 66.
- **Vocabulary:** a 65,536-token family costs about 0.5 GB in the tied embedding. The unit adds
  VE and VH (16 bits each); VC uses the freed WB family (15 bits).
- **Config:** 8 slots of 12 bits, one single-row value head each (4 key heads in L1). A VC token
  writes the slot named by its top 3 bits, gated by those token bits.
- **Counters for the unit (L1):** vel counts VH tokens since the last VC or record start (the
  element of the next token), and vsub counts VE tokens since the last VH or VC (its position
  inside the element).
- **VM:** one element per row (31 bit columns): 2 banks � 512 rows = 4 key heads. A read is the
  2-hot pair containing the element; a write is one element at its VH, from lags 1 (VE, bits
  0-15) and 0 (VH, bits 16-30).
- **VA:** 6 chunks of 32 byte pairs. A row is a pair (16 columns), 4 rows per value head, so a
  chunk read is 4-hot in each of 4 key heads. A VQ / VEXP byte goes in at the odd element's VH
  together with the even byte from lag 1.
- **The V cache for p.V** needs every slot's byte j in one read. Layout idea: one value head per
  slot t (32 value heads = 16 key heads), rows (l, g), and z gates that pick byte j's 8 columns,
  giving 256 dims. Not built yet. The head budget is tight: L2 has 16, L1 11; L3 frees 16 if the
  ROM shrinks to 128 KB.

## Status

- **Reference v6:**
  - element units, aligned offsets, 6-chunk VA, slotted VC, VE / VH, 31-bit words
  - write-target codes in slot 0; trackers flagged on their own VH; argmax rows as words
  - stories260K: 2.22M trace tokens for the 60-token story (12.53M today), same story
- **Circuit (vxcirc.py), still 11 layers:**
  - config slots, element counters, VM, VA, trackers
  - VPUT, VQ, VRED and VGET, teacher-forced exact on random programs (tests/test_rv_vx.py)
- **Lesson:** every literal in a case or t_and must be exactly 0 or 1 at every position,
  including where the case doesn't fire. A sum of two register bits broke ordinary instructions.

## Next: VMAT

- **Weights.** WROM needs 9 heads: row-chunks packed 2 int4 cells a byte column, 2 chunks a row,
  S in 2 byte columns. L2 has 7 free. Plan: a 128 KB ROM in the NPU build (a build option;
  MicroPython images need 205 KB), pinned to L4 so the fetch timing stays, which frees L3's 16
  heads for the WROM.
- **One tracker.** Programs always read a tracker right after the op that set it, so vmx, vsum
  and vbest could share one head (frees 2 in L2). The reference would follow.
- **4 tokens a row.** The dot product (about 7 stages from the L3 read) and the row-scale
  multiply (5) don't fit one token. A row becomes acc lo, acc hi, then o = acc * S lo, hi; the
  multiply reads acc from the previous tokens through an L1 "lag" head (written every token from
  the lagged payloads, read at once). Expect about 44k trace tokens per story token (4.7-5x
  fewer than today) at about +0 layers.

### Found building VMAT

- **Byte tables need 2-hot reads.** A byte-valued snapshot table read one-hot isn't exact
  through the HF kernel: it scales by 2^-3.5, and only power-of-two data survives the bf16 cast.
  The ROM and DROM read row pairs (p, p + 64), which scales by 2^-4 and is exact. Counts of 2, 8
  and 32 rows scale by powers of 2. Bit tables are exact at any count, because the data and the
  ballast round alike.
- **So a key head holds 126 byte columns × 64 row pairs × 2 value heads = 16 KB.** The WROM (4,152
  chunks × 34 bytes) needs about 12 heads with 3 chunks a row pair.
- **New WROM addressing.** The WROM goes in L3, where queries may use L2's MLP outputs, so the row
  pair = base + (k = 1: r // 3, k = 3: r) by a one-stage threshold adder, and the slot = r % 3 (k
  = 1) or the chunk c (k = 3). The CPU passes the row-pair base; each op starts at slot 0.
- **Layer count.** The first VMAT build (one-hot WROM, 4 comparison adders) placed at 13 layers:
  L5, L6, L8 and L9's MLPs hit the 9,216-unit cap, so the dot product spilled upward.
