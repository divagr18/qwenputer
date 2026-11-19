# Latent CPU control, Phase 0e: operand order and op for the no-call command

Phase 0d read the operand pair from attention at Qwen's direct answer, with no call text. Order and
op were the weak parts. Phase 0e works on both, CPU only, from the Phase 0d data
(`chipformer/rv/latentorder.py`).

## Setup (measured)

- **Data.** Phase 0d's direct-answer (D) rows from two of the three packs (840 rows).
  - The third pack is on q35-latent-4, which had no capacity when this ran.
  - Candidate values come from the Phase 0c meta (the question's numbers, in order).
- **Evaluation.** Every number is leave-one-source-out.
  - The op probe uses layer 20, fixed rather than chosen on the test source.
  - Full numbers: `order-report.json`.
- **Pair.**
  - One learned weight per full-attention head.
  - Loss: both operands' mass at the answer's first digit.
- **Order: antisymmetric features of an ordered pair, fit on - and / only.**
  - *text:* the earlier number first.
  - *cues:* text order, which value is larger, which divides the other.
  - *attn:* the difference in each head's log attention mass between the two numbers.
- **Op: logistic regression on the residual at the answer.**
  - The 0d setting.
  - Stronger weight decay (0.1).
  - Plus the attention or value features.
  - Per-source centering, which was dropped (below).

## Results (measured)

| held-out source | n | pair | order, text | order, attention | op, 0d | op, wd 0.1 | command, 0d method | **command, 0e** | Qwen's own answer |
|---|---|---|---|---|---|---|---|---|---|
| synth | 386 | 97.2 | 59.5 | 85.3 | 99.2 | 99.2 | 77.7 | **89.9** | 47.7 |
| svamp | 138 | 83.3 | 50.5 | 75.2 | 92.0 | 93.5 | 47.8 | 65.9 | **73.9** |
| dmx_mul_big | 92 | 100 | - | - | 72.8 | 98.9 | 72.8 | **98.9** | 47.8 |
| dmx_div_big | 82 | 100 | 100 | 100 | 98.8 | 97.6 | 98.8 | **97.6** | 46.3 |
| dmx_add_or_sub_big | 94 | 100 | 81.6 | 83.7 | 66.0 | 63.8 | 56.4 | **55.3** | 31.9 |
| arith | 46 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 34.8 |

How to read the columns:
- **Order** is measured on - and / commands, given the true pair.
- **Command** means the pair, order and op are all predicted, and all right.

Findings:
- **Order from attention beats text order:**
  - synthetic 60 → 85, where roles are reversed ("subtract b from a");
  - SVAMP 51 → 75.
  - Value and divisibility cues alone are weaker: 63–65 on the word problems.
- **Stronger regularisation fixes the op on big-number multiplication** (73 → 99) and costs at most
  2 points elsewhere.
- **Extra features don't help the op.** Adding attention or value features to the op probe does not
  help (mul 57–74, add/sub ≤ 66).
- **Per-source centering of the residual breaks the op** on the single-op DM sets (div 99 → 38),
  because there the source mean is the op direction. Dropped.
- **Signed numbers limit big add/subtract.**
  - 100 of its 146 calls have a negative operand (`-5 - -2608731043409517`), and the pointers are
    unsigned, so the op label is ambiguous there.
  - On its 32 sign-free rows the command is right 90.6%, against Qwen's 34.4%.
  - The same caveat applies to the signed rows of the mul and div sets. There the command is right
    in magnitude, but the CPU's answer sign is not covered.
- **The confidence-gated fallback doesn't help yet.** "CPU when confident, else Qwen" uses a
  threshold set on the training sources. On SVAMP it leaves 67% to the CPU and scores 65.9, the same
  as the command alone, while picking whichever of the two is right would reach 81.2. The confidence
  (pair mass × op probability × order probability) doesn't separate good commands from bad ones.

## Reading

- **With no call text, the full command is right much more often than Qwen's own direct answer on
  every large-number source:**

  | source | latent command | Qwen alone |
  |---|---|---|
  | synthetic | 90 | 48 |
  | mul | 99 | 48 |
  | div | 98 | 46 |
  | arith | 100 | 35 |
  | sign-free add/sub | 91 | 34 |

- **SVAMP is still below Qwen alone** (66 vs 74). Its errors are split between the pair (83%) and
  the order (75%).
- **Not covered:**
  - signed operands;
  - multi-step problems;
  - an end-to-end run where the CPU writes the digits. So far this is a readout scored against
    Qwen's own tool-call labels.

## Next (estimated, not tested)

1. **Signed pointers.**
   - Count a leading "-" as part of a number run, so the CPU copies signed values and the op label
     is defined.
   - Needs a new collection run.
2. **Wider attention capture.**
   - Collect attention over all question tokens, not only number tokens, so the op (and order) can
     be read from where the heads look at "minus", "divided by" or "from".
   - Same run.
3. **Direct-mode demo end to end.**
   - At the answer's first digit, read the command and let the CPU write the exact digits.
   - Score against gold, next to Qwen alone and Qwen with the text tool.
