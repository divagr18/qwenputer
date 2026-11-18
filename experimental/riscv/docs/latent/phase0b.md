# Latent CPU control, Phase 0b: leave one source out

Phase 0 found that a pointer probe trained on synthetic questions dropped to 71% on benchmark
questions, below the "last two numbers" heuristic (91%). Phase 0b asks whether that is a data
problem (too little variety) or a method problem (frozen-state probes do not carry a portable
command).

## Setup (measured)

- Base Qwen3.5-4B, bf16, the calc tool prompt from `evalarith` (condition T), plus 1,200 synthetic
  questions with no tools (condition N, training data only).
- 3,750 sequences, hidden states at layers 8, 12, 16 and 20 (`latentprobe collect`, 8 parallel
  parts). Two-number commands per source: synth 623, dmx_div_big 285, dmx_mul_big 252,
  dmx_add_or_sub_big 249, svamp 226, gsm8k 214 (198 with both operands in the context),
  arith 97, dmx_mul_div_multiple_longer 91.
- For each source: train the op probe and the two pointer probes on every other source (and N,
  except when synth is held out), test on the held-out source. Pointer variants: content only
  (the Phase 0 bilinear pointer) and content plus a learned order bias (rank from the end, rank from
  the start, in the prompt or not). Pointers are trained with padded minibatches of 32
  (`refit --bs 32`).
- "Command" = op right and both operands right (+ and * accept either order).
- Full numbers: `refit-report.json`.

## Results (measured, layer 20, the best layer overall)

| held-out source | n | pointer | command | op | last-two | first-two |
|---|---|---|---|---|---|---|
| arith | 97 | 99.0 | 97.9 | 97.9 | 99.0 | 99.0 |
| dmx_add_or_sub_big | 249 | 95.2 | 81.5 | 85.5 | 98.0 | 87.6 |
| dmx_div_big | 285 | 97.5 | 96.5 | 98.9 | 96.8 | 96.1 |
| dmx_mul_big | 252 | 100.0 | 90.5 | 90.5 | 100.0 | 100.0 |
| dmx_mul_div_multiple_longer | 91 | 56.0 | 45.1 | 76.9 | 64.8 | 25.3 |
| gsm8k | 198 | 59.6 | 35.4 | 53.3 | 61.1 | 13.6 |
| svamp | 225 | 80.4 | 55.6 | 65.0 | 64.9 | 36.9 |
| synth | 623 | 42.7 | 40.0 | 87.6 | 51.7 | 46.1 |
| mixed random split (all sources seen) | 419 | 93.1 | 89.3 | 94.1 | 76.4 | 64.4 |

The order bias moves pointer accuracy a few points either way (at layer 20: -6.5 to +0.5) and closes
no gap. Layer 20 is the best layer, or within 3 points of it, on every held-out source.

## Reading

- With every source in training (mixed split), the probes are well above the heuristics (pointer
  93% against 76%). That matches Phase 0's in-distribution numbers, so the minibatch change is not
  hiding anything.
- Held out, the pointer is at or below "last two numbers" on 6 of 8 sources. It beats the best
  heuristic clearly only on svamp (80.4 against 64.9). The op probe also drops (53–65% on gsm8k
  and svamp).
- More varied training data did not fix transfer. A linear op probe plus a bilinear pointer read from
  frozen states at the position before `<tool_call>` does not give a command that carries across
  question styles. This is a method problem, not a data problem. The >99% gate is far off on every
  source where the heuristics are not already near 100%.

## What this leaves

Reading a finished command out of base Qwen's state before it writes anything is not reliable.
Directions that remain open (estimated, not tested):

1. Read operands while Qwen writes them: the copy/induction attention at the first digit of each
   operand points at its source, training-free. This needs states or attention saved at positions
   inside the call (not collected here; `--own-sample 0.1` kept only 10% of them).
2. Train the interface: an adapter on Qwen so a command is made explicit. This changes Qwen's
   weights or adds trained ones.
3. Keep tool-call text as the interface (today's design) and make the rest of the path latent.
