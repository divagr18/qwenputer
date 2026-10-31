# VX: the vector unit in the circuit (design B, early layers)

This is the plan the `vx` branch builds. `chipformer/rv/vx.py` is the reference: its token costs
and semantics are the contract, and the circuit has to match it token for token.

## Why the early layers

The core's datapaths sit in layers 5â€“9 and are nearly full there (MLP free: L5 0.6k, L6 1.7k, L8
1.2k, L9 0.7k; heads: L3â€“L9 full). The free room is:

- heads: L1 11, L2 16, L10 16, L11 16
- MLP units: L1 about 8.9k, L2 7.8k, L3 8.8k, L4 4.9k, L7 5.1k, L10 7.9k, L11 8.1k
- live dims: low in L1â€“L4 (about 120â€“240) against a peak of about 1,090 at L8â€“L9

Exact arithmetic costs one MLP stage, and so one layer, per carry or threshold level. A multiply
takes about 5 stages and a 64-lane dot product about 6. A unit that reads its operands in layers
2â€“3 has 8â€“9 MLP stages before the exit (L11). A unit on top of the core would add one layer per
stage.

## The rule the design follows

1. **Memories:** the unit's memories are its own heads in L1â€“L3, so its addresses never need the
   register file (L5) or RAM (L7â€“8).
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
| VM | 16 vector registers Ã— 64 int32; bank A = v0â€“v7, bank B = v8â€“v15, one read port each | element pairs (64 bits), written every 4th stream token | 2, L2 |
| VA | 256 int8 activations, 4 chunks of 64 | byte pairs in pair-specific columns, read a chunk at a time (multi-hot) | 1â€“2, L2 |
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
| VMAT | out = acc Ã— S_r into a VM register (k = 1: 64 lanes); argmax mode | 2 a row (k = 1), 6 a row (k = 3) |
| VMATR | scores: VA q Â· K rows (signed max tracked); pÂ·V: VA p Â· V column j, Ã— r0 >> 12 | 2 an output |
| VQ | a VM register's pairs â†’ int8 pairs into VA or a cache row | 1 per 2 elements |
| VEW | ADD (window Ã— factor, then add); ROPE (pairs, cos/sin from state); GLU (two multiplies chained) | ADD, ROPE: 2 an element; GLU: 4 an element |
| VEXP | exp-table softmax numerators, VM scores â†’ VA | 1 an element |
| VRED | Î£ VAÂ² or Î£ VA over a chunk | 1â€“2 |
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
- **VM:** one element per row (31 bit columns): 2 banks × 512 rows = 4 key heads. A read is the
  2-hot pair containing the element; a write is one element at its VH, from lags 1 (VE, bits
  0-15) and 0 (VH, bits 16-30).
- **VA:** 6 chunks of 32 byte pairs. A row is a pair (16 columns), 4 rows per value head, so a
  chunk read is 4-hot in each of 4 key heads. A VQ / VEXP byte goes in at the odd element's VH
  together with the even byte from lag 1.
- **The V cache for p.V** needs every slot's byte j in one read. Layout idea: one value head per
  slot t (32 value heads = 16 key heads), rows (l, g), and z gates that pick byte j's 8 columns,
  giving 256 dims. Not built yet. The head budget is tight: L2 has 16, L1 11; L3 frees 16 if the
  ROM shrinks to 128 KB.

## Status (QwenPU, 2026-10-31)

Built and exact: every op in `tests/test_rv_vx.py`, the story program through the IR (first
110,000 steps) and on the GPU executor (all 3,184,921 steps), and the merged checkpoint's own
layers (`vxcheck`). What changed from the plan above, while building:

- **Weight memory:** a 2-hot row-pair table in L3, 3 row-chunks a pair (12 key heads). A k = 1
  op's row r is chunk 3 base + start + r (ops packed back to back, a start slot in the config);
  a k = 3 op's row r is pair base + r, chunk c in slot c. The pair index is an adder in L2's
  MLP.
- **Dot products:** by bit planes (VMAT) and bit-pair counts (VRED, VMATR), so the first
  reduction round splits small sums.
- **One multiply-and-shift datapath** serves VQ and VEW: a window of a VM element (a one-stage
  shifter, clamp16 folded into the products) times a factor, then a signed shift.
- **Streams:** VEW and VMATR p.V stream a raw value first, then the word from the lags, so the
  add and the tracker compare stay shallow. VEXP streams e, e' then p, p'.
- **Caches:** K rows in 5 key heads (halves in the two value heads); V with one value head per
  slot (16 key heads), so p.V reads byte j of every slot at once. VQ writes 4 bytes at a time.
- **Placement:** `q35/sched.py` re-places the built graph by criticality (15 to 12 layers before
  the last ops; 13 now), with a level checker.
- **Gotchas:** counter functions must hold at reset tokens; literals from config slots must stay
  0/1 under every op's config (fold them into gates); executors write a zero key's row 0, so
  every write is gated by its own key head; the executors round MLP gates to bf16 as the model
  does.

Next: the hidden size (5120, from about 2,250 live dims at L8-L9) and with it the checkpoint
(26.7 GB).
