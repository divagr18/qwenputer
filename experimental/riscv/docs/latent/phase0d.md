# Latent CPU control, Phase 0d: read the command at the answer, with no call

Phase 0c read the operands from attention while Qwen wrote a tool call. Phase 0d removes the call:
Qwen answers with no tools, and the command is read at the position where Qwen starts writing a
computed number.

## Setup (measured)

- **Model.** Base Qwen3.5-4B, bf16, no tools.
- **Problems.** The same 2,050 as Phase 0c, in two conditions:
  - **D, direct answer.** System prompt "Answer with only the final number. Do not show any
    working."
  - **C, step by step.** The evals' "Do every arithmetic step exactly" prompt.
- **What was saved.** For every number Qwen writes, at the position that predicts its first digit:
  - the hidden states at layers 8–24;
  - the attention of all 128 full-attention heads over every number in the context.
- **Collection.** `latentprobe anspos` in 6 parts on 3 boxes. Decoding uses sdpa; one eager
  forward pass captures the attention.
- **Labels: which two numbers and which op Qwen means.** These come from Phase 0c's tool calls on
  the same problems.
  - **D:** the first number Qwen writes, on problems whose Phase 0c run made exactly one binary call
    whose operands are in the question.
  - **C:** each number Qwen writes that isn't a copy of an earlier number and equals the result of
    one of the problem's calls.
    - *C-written:* the operands and "=" come right before the number.
    - *C-bare:* they don't.
- **Labelled rows:**

  | condition | synth | svamp | gsm8k | div | add/sub | mul | arith | multi-step |
  |---|---|---|---|---|---|---|---|---|
  | D | 580 | 191 | 3 | 128 | 146 | 150 | 57 | 0 |
  | C-bare | 258 | 114 | 54 | 69 | 51 | 30 | 8 | 12 |
  | C-written | 36 | 98 | 149 | 5 | 6 | 0 | 0 | 2 |

  GSM8K's tool calls are mostly whole multi-op expressions, so it has almost no D labels.
- **Evaluation.** Every number below is leave-one-source-out: heads ranked, and weights and probes
  fit, on the other sources only.
  - The op probe's layer is the best of layers 8–24 on the held-out source, as in Phase 0c. That
    slightly favours the probe.
  - Full numbers: `ansfit-report.json`.

## Results (measured)

### D: direct answer, no call text at all

| held-out source | n | pair, best head | pair, top 4 heads | pair, learned | ordered, learned per slot | ordered, last two | op probe | command right | Qwen's own answer right |
|---|---|---|---|---|---|---|---|---|---|
| synth | 580 | 91.0 | 94.1 | 96.0 | 86.6 | 43.6 | 99.0 | **85.9** | 47.4 |
| svamp | 191 | 83.2 | 85.9 | 82.2 | 72.8 | 42.4 | 84.8 | 60.2 | **76.4** |
| dmx_div_big | 128 | 100 | 100 | 100 | 99.2 | 100 | 97.7 | **96.9** | 51.6 |
| dmx_mul_big | 150 | 100 | 100 | 100 | 100 | 100 | 76.7 | **76.7** | 48.7 |
| dmx_add_or_sub_big | 146 | 100 | 100 | 100 | 89.7 | 89.7 | 63.0 | **52.7** | 35.6 |
| arith | 57 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | 38.6 |

Column definitions:
- **Pair:** the two most-attended numbers are the operands, in either order.
- **Ordered:** operand order also right (+ and * accept either order).
- **Command right:** ordered pointers and op both right, so the CPU's exact result equals the
  result of Qwen's own tool call.
- **Qwen's own answer right:** the number Qwen wrote equals that result.

Findings:
- **The operand pair is in attention when Qwen starts its answer, with no call text.**
  - Synthetic set: 96%, against 44–46% for the order heuristics, in questions with distractor
    numbers and reversed roles.
  - SVAMP: 82–86%.
  - Two-number sets: 100%.
- **The pointer comes from different heads than Phase 0c's.**
  - In every split the top heads are 19.14, 23.8 and 23.5 (also 23.14 and 19.11), mid-stack rather
    than layer 31.
  - The best single head finds the pair on 95–98% of the training sources.
- **Order and op are the weak parts.**
  - Order is mostly right with one learned weight vector per slot: synth 87%, SVAMP 73%.
  - The op probe transfers to synth (99%) and div (98%), but not to the DM mul (77%) and add/sub
    (63%) sets, whose phrasing is unlike the other sources.
- **Command vs Qwen's own direct answer:**
  - The full command beats Qwen's own direct answer on every set with large numbers: synth 86 vs
    47, div 97 vs 52, mul 77 vs 49, add/sub 53 vs 36, arith 100 vs 39.
  - SVAMP is the exception: small numbers, so Qwen's own answer is better (76 vs 60).

### C: inside step-by-step text

- **C-written:** reading the operands and "=" Qwen has just written is trivial. Layer-3 heads 3.0,
  3.2 and 3.3 find them 93–99% of the time, the same as "last two".
- **C-bare: fails** (13–62%; first-two does better on the DM sets).
  - The labels are loose: a number that matches a call's result, written after long working, is
    computed from the intermediate numbers, not from the question's operands.
  - So this neither confirms nor rules out a bare pointer inside reasoning text.

### When to fire

Whether the next number will be computed or copied: a hidden-state probe at layer 24, held out,
gives AUC 0.88–0.94 (gsm8k 0.90, svamp 0.93, synth 0.92). That is useful but not a trigger by
itself. In D the trigger is trivial: the answer's first token.

## Reading

- With no tool call, no forced tokens and no extra forward pass, Qwen's own forward pass on its
  answer carries which two context numbers it is combining. Mid-stack attention heads read them out
  held-out.
- The best of these heads finds the right pair about 91–100% of the time, depending on the source
  (pair, best head column).
- **What is missing for a full latent command:**
  - **Operand order:** 73–87% on word problems.
  - **The op:** off-style sources at 63–77%.
- **The latent command is already more accurate than Qwen's own direct arithmetic on every
  large-number set.** It is not more accurate on SVAMP's small numbers.
- **Not demonstrated:**
  - multi-step problems (GSM8K has no single-call labels in D);
  - computed numbers inside free reasoning (C-bare labels are too loose to say).

## Next (estimated, not tested)

1. **Order and op from the attention itself.**
   - Read the op from the same heads' values, or from a probe at the operand-pointed positions,
     instead of the residual at the answer.
   - Fit order with features of the two pointed numbers (text order, role words), still held out.
2. **Silent CPU in direct mode, end to end.** At the answer's first digit, read (op, pointer,
   pointer) and let the CPU write the exact digits. Report accuracy against gold, compared with
   Qwen direct and with Qwen plus the text tool.
3. **Multi-step chaining.** Prompt Qwen to state intermediate results without the arithmetic ("so
   she has … apples"), label each from the calls, and test whether a pointer at each result finds
   the earlier result plus a question number.
4. **In weights.** The heads sit at layers 19 and 23, so a compiled CPU block placed after layer 23
   could reuse their query/key projections, fetch the operands exactly and write the result logits.
   That needs no feedback wire.
