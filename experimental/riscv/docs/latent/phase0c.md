# Latent CPU control, Phase 0c: read the operands where Qwen writes them

Phase 0b showed that probes on the hidden state before `<tool_call>` do not carry a command across
question styles: held out, they score at or below the "last two numbers" heuristic. Phase 0c tests
two changes suggested by the arithmetic-interpretability literature, where operands are fetched by
attention at the point of use:

1. the same probes, with the query at the position right before each operand's first digit ("arg")
   instead of before `<tool_call>` ("pre");
2. no probe at all: the attention of Qwen's full-attention heads at the position right before each
   operand's first digit, summed over each number in the context.

## Setup (measured)

- Base Qwen3.5-4B, bf16, eager attention, the calc tool prompt (condition T only). Qwen writes its
  own calls; the host answers. 2,050 problems: synth 600, the four DM sets and arith 150 each,
  gsm8k 400, svamp 300 (`latentprobe argpos`, 4 parts).
- Full-attention layers: 3, 7, 11, 15, 19, 23, 27, 31 (16 heads each). Head ids below are
  layer.head.
- Commands: binary calls with both operands found as numbers in the call; targets are the context
  numbers (prompt, earlier generated text or tool responses) equal to what Qwen wrote. Per source:
  synth 624, svamp 303, gsm8k 338 (of 365 with the operands in the context), div 191, add/sub 151,
  mul 150, arith 57, multi-step 51.
- Attention pointer for a slot: at the position that predicts the operand's first digit (Qwen has
  not written the operand yet), sum the head's attention over each candidate number's tokens and
  take the argmax. "Pointer" = both operands right (+ and * accept the swap).
- Every number is leave-one-source-out: heads are ranked, and per-head weights fit, on the other
  sources only.
- Full numbers: `posfit-report.json`.

## Results (measured)

Pointer accuracy (both operands right), held-out source:

| held-out source | n | attention, best head | attention, top 4 heads | attention, learned weight per head | probe at "arg" (best layer) | probe at "pre" (best layer) | last two |
|---|---|---|---|---|---|---|---|
| synth | 624 | 95.5 | 97.3 | **98.7** | 55.4 | 42.0 | 51.4 |
| svamp | 303 | 84.8 | 90.8 | **96.7** | 76.9 | 70.3 | 58.7 |
| gsm8k | 338 | 82.5 | 84.6 | **87.3** | 50.3 | 47.3 | 50.3 |
| dmx_add_or_sub_big | 151 | 98.0 | 100.0 | 100.0 | 90.1 | 94.0 | 96.7 |
| dmx_div_big | 191 | 99.0 | 99.5 | 98.4 | 99.5 | 99.5 | 97.9 |
| dmx_mul_big | 150 | 99.3 | 100.0 | 100.0 | 100.0 | 99.3 | 100.0 |
| dmx_mul_div_multiple_longer | 51 | 100.0 | 94.1 | 100.0 | 47.1 | 49.0 | 60.8 |
| arith | 57 | 98.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

- The same heads rank first whichever source is held out: 31.1, 27.13, 31.0, 31.2 (first-digit hit
  rate on the training sources 95-97%). Held out, the best head's first-digit hit rate is 90.5%
  (gsm8k), 92.1% (svamp), 97.7% (synth), 99-100% elsewhere.
- Continuation digits: the best head points at the matching digit of the source number 92-100% of
  the time.
- "Learned" is 128 weights (one per full-attention head) per slot, trained on the other sources.
- Probes, layers 8-24: moving the query from "pre" to "arg" helps a little on word problems (gsm8k
  50 vs 47, svamp 77 vs 70, synth 55 vs 42) but stays near the heuristics. The op is much easier at
  "arg" (synth 100%, gsm8k 87-92%), since Qwen has begun the call.
- Abstention: a share-margin threshold on the attention (top number's share of the attention on
  numbers minus the runner-up's) reaches 100% on accepted calls in every source, but on gsm8k and
  svamp only at 3-17% coverage. Misses are mostly diffuse attention (peak mass 0.01-0.07) on
  operands Qwen computes or recalls rather than copies (e.g. `10 + 1`, `1500 * 1.20` with 1.20
  written as 120 in the context).

## Reading

- The operand choice is not linearly readable from the residual stream, at either position, in a
  way that carries across styles. It is readable from where Qwen's late full-attention heads look
  when it is about to write the operand. That pointer is training-free (one head, or a sum of four)
  and transfers across every held-out source: 82-100% with no fitting, 87-100% with 128 weights,
  against 50-61% for the heuristics on the word problems and synthetic set.
- The >99% gate is met on the numeric sets and nearly on synth (98.7%), not on gsm8k (87%) or svamp
  (97%).
- This is a teacher-forced readout of Qwen's own call: it says which context number Qwen is about to
  copy. Used as an interface, Qwen still begins the call (at least up to the first operand's position
  for slot 0, and through the first operand and the op for slot 1), so it is latent in the output,
  not in the mechanism.

## Next (estimated, not tested)

1. Silent scaffold end to end: fork the state, force the call opener, read slot 0 at the position
   before the first digit, let Qwen write operand 1 and the op privately, read slot 1, hand the CPU
   (op, pointer, pointer), discard the fork. Compare exact-operand rate and end-to-end accuracy with
   the text tool call on held-out gsm8k/svamp.
2. Abstention that keeps coverage on word problems: a learned confidence from all 128 heads, or a
   consistency check that the pointed-at number's digits equal the digits Qwen writes, with fallback
   to the text call.
3. In weights: copy the query/key projections of heads 31.1/27.13/31.0/31.2 into a compiled CPU
   layer so the CPU's operand fetch reuses Qwen's own pointer.
