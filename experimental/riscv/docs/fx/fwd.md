# Why a plain forward of the merged model left base on tool-call text, and the fix

Status 2026-11-08, branch `fx-fwd` (from `fx-int` 048c114). NVIDIA L4 (q35-eval), Executor v4, real
Qwen3.5-4B. Raw numbers: `fwd.json` next to this file, built by `fx/fxio/report_fwd.py` from the
runs listed at the end. **M** = measured, **E** = estimated or a reading we have not checked. Every
number below is M unless marked.

## Result

- **The cause is found.** The text bridge's `owner` term made the circuit own the position after
  every `</tool_call>` token while the machine is not running. There the output code names **GO**
  and tau goes to 0. It fires on every `</tool_call>` in the text, wherever it is: the two in the
  system prompt's format example, and one per completed call in the history. In a plain forward
  that code reaches the tied head. A large code gain then turns it into logits that rank tokens by
  code agreement. Over Qwen's vocabulary the winner is `\x00` (printable token 0, GO's payload),
  more than 2,000 logits above base's next token. That is the one.md divergence: the 1,108-token
  render has 4 `</tool_call>` and code fired at exactly those 4 positions.
- **The production `--text` path had the same fault in generation.** Every call Qwen generated in
  one.md's `--text` runs ended `</tool_call>\x00`: 14 of 14 calls (`tl2`). Prefill hid it,
  because `_text_hc` gives a whole chunk the kept dims of the chunk's last token. A generated token
  uses its own (`_exec_hc`). The runs were still solved because fx's parser drops the NUL.
- **Fixed for fx.** `core.build(..., go_on_tcc=False)` drops that one term (and GO's case at
  `</tool_call>`). fx starts a run with the GO the host feeds at unpark, so it never needed the
  trigger. On 26 sequences (16 plain, 10 with tool-call markup, 101 to 2,304 tokens) the plain
  forward now equals the production text path **bit for bit** (max logit difference 0.0). Mean KL
  against base on the markup sequences fell from **13.7 to 8.57e-3**, against a noise floor of
  7.16e-3.
- **Exact after the fix.** These all pass:
  - fx's whole session: 6,666,925 tokens, 0 wrong.
  - t1-t5 live with `--text`: 5 of 5 solved, 0 wrong chip tokens, and 16 of 16 calls end exactly
    in `</tool_call>`.
  - The probe's own t1 run: 0 wrong.
  - Every circuit built without the new flag is node for node the same as fx-int's.
- **rv-q4p2 (the published model) has the same trigger, and there it is the design.** A plain HF
  forward predicts `<|rv:GO|>` at every `</tool_call>` in a history, with probability 1.0 (to
  print precision), and nowhere else. That cannot be fixed in the weights. See
  [What a merged model can claim](#what-a-merged-model-can-claim).

## The cause, with the evidence

`core.build` (P3 bridge), before this change:

```python
owner = running + HALT + isTCC * (1 - running)     # who predicts the next token
cases.append(([isTCC], "GO", 0))                   # at </tool_call>, predict GO
tau += mu - TAU_TEXT * owner * is_text             # owned text: tau 0
```

The `isTCC * (1 - running)` term is rv-q4p2's calc trigger: Qwen writes `</tool_call>`, the
circuit predicts GO, GO runs the firmware. It has no notion of a call that was already served.

The probe is `fx/fxio/text_paged.py`. It ran on fx's machine after 1,158,538 instructions (258
frames), on 26 sequences:
- the 17 from one.md;
- 6 short and mid-length conversations with completed calls (`chipformer/rv/toolprompts.py`);
- 3 more real fx renders.

It reports, for each position, whether the output code is set, what token it names, tau, the KL
there and elsewhere, and the state words each value head changed.

| check (go_on_tcc on, the old bridge) | result |
|---|---|
| positions where the code is nonzero, against positions holding `</tool_call>` | **equal on all 10 markup sequences** (51 positions); 0 on the 16 plain ones |
| the token the code names there | GO, at all 51 |
| tau there | 0.0 (2^-40 elsewhere) |
| KL(base ‖ forward) at those positions | 565 to 2,688 nats |
| forward's top text token there | 188 (`\x00`) at all 51; base's is `<|im_end|>`, `\n` or `\n\n` |
| mean KL over the other positions | the production text path's level (render t1: 7.92e-3, against the text path's 7.90e-3 over all positions) |
| the one.md render: positions | 497, 531, 715, 1048: the system prompt's two, then two completed calls |

So the merged stack does predict a chip token in the middle of text: GO, after `</tool_call>`.
Nothing else triggers it: `<tool_call>`, `<function=` and the rest of a call's text never set the
code in these runs. The 1,108-token render's mean KL of 8.9 is 4 positions at about 2,450 nats each,
averaged over 1,108.

**Why production differed.** Both paths feed the chip the same tokens: `WideBackend` runs every
token Qwen reads or writes through `cpu.text` with fx parked. Nothing is reset or masked on the
chip side. The difference is on Qwen's side:
- **Prefill** (`_text_hc` with `fx_keep`) writes the kept dims of the chunk's last token into
  every position of the chunk. An owned position in the middle of a chunk is overwritten by the
  chunk end's zero code. A render never ends on `</tool_call>` (it ends on
  `<|im_start|>assistant\n`), so prefill was always clean.
- **A generated token** takes its own kept dims (`_exec_hc`). So after Qwen wrote `</tool_call>`,
  the next sample came from the GO-coded logits: `\x00`. Then `<|im_end|>`.

**The four state words.** one.md reported 4 changed state words outside the bridge's own heads.
They are the `j` and `vcnt` counters, 2 words each. They change on the first text sequence after
the park (plain0, which has no markup), and on no later sequence. This happens with and without
the trigger, and the run's audit afterwards is 0 wrong over 3,552,515 chip tokens. They are
unrelated to the divergence. Why a counter word moves on the first text token after HALT is not
traced (E: a counter settling after the park's HALT).

## Bound: is it only fx?

No. It is the P3 bridge, in both circuits that have it.

**rv-q4p2.** The box's `dist/rv-q4p2` is the published `divagr1925/Qwen3.5-4B-Qwenputer`: all 8
shards match the Hub's LFS sha256 at revision b14ec565. `chipformer/rv/fwdcheck.py` runs
textcheck's method on report.PROMPTS and toolprompts (calc and read_file histories, 101 to 2,177
tokens, up to 42 `</tool_call>`). It runs the checkpoint as a stock `Qwen3_5ForCausalLM`, from zero
states, which is what anyone loading it gets.

| | 16 plain prompts | 7 prompts with completed calls |
|---|---|---|
| KL(base ‖ merged), Qwen's vocabulary | 6.12e-4 (floor 4.96e-4): the card's numbers reproduce | **26.1** (floor 7.16e-3) |
| argmax over the whole vocabulary at `</tool_call>` | | `<|rv:GO|>` at all 67, chip mass 1.0 |
| positions where the argmax is a chip token | 0 | exactly the 67 `</tool_call>` positions |
| KL at every other position | | 1.24e-2 (floor 7.22e-3), see below |
| stock greedy generate equal to base's | 10 of 16 | 4 of 7, plus 2 equal up to the merged config's EOS |

**The GO code reaches no other position.** `fx/fxio/q4p2_code_removed.py` puts a hook after the
last chip layer that zeroes the code and restores tau at the `</tool_call>` positions. With it:
- the logits at every other position are bit-identical (max difference 0.0) on calc3, calc12 and
  calc40;
- at the `</tool_call>` positions KL drops to 4.6e-4 to 1.0e-3, and the argmax becomes base's
  `<|im_end|>` or newline.

The residual after the chip layers (`fx/fxio/q4p2_residual.py`) is base's embedding exactly, at
every position, on those three:
- Qwen's 2,560 dims differ from the embedding nowhere;
- every chip dim outside the kept ones is 0;
- off `</tool_call>` the kept dims are at most 2^-40 (tau's text value), and mu is 0.

**Away from `</tool_call>`, long markup is above the floor in rv-q4p2. The bridge is not the
cause.**
- Short prompts (101 to 459 tokens) are within 1.4x of the floor (0.88x to 1.36x; e.g. calc1 6.7e-4
  vs 7.5e-4).
- calc12 (876 tokens, 14 calls) is 2.89e-2 vs 1.96e-2.
- calc40 (2,177 tokens, 42 calls) is 4.02e-2 vs 1.43e-2.
- In calc40 the excess sits on a few positions inside the repeated call blocks: three at 14 to 16
  nats where eager-vs-sdpa gives 0.05 to 6.5 there.
- fx's merged form on calc12 is at its floor (1.74e-2 vs 1.93e-2).
- Since the residual entering Qwen's layers is base's own embedding, this is Qwen's merged layers'
  numerics (hidden 3968, rescaled norms, the mute) on long repetitive text. **E, not isolated
  further.**

**Plain greedy differences** (6 of 16) start at reply tokens 0 to 29. **E:** bf16 near-ties (on
those prompts the top-1 agreement with base is also below 1 for eager attention). textcheck uses
the same method; the card does not quote greedy agreement.

## The fix

**fx (done).** `core.build(target, text, cfg, go_on_tcc=False)`:
- `owner = running + HALT`;
- the circuit predicts nothing at `</tool_call>`;
- the capture into the text buffer (arm, cap, tbuf) is unchanged.

fx's two text builds pass it: `fx/fxio/live_chip.py` (`Chip`) and `chipbuild fx --base`.
`live_chip.py --go-on-tcc` rebuilds the old bridge, which is how the "before" column was measured
on the same code. The default stays `True`, so rv-q4p2's circuit, the legacy map and every test
circuit are unchanged.

The change in the fx text circuit is three nodes: `owner`, `exit.code2` and `exit.code4` (GO's
code bits). 44,996 FFN units instead of 44,999 (`circ_digest.py`).

**Before and after, fx's merged form on fx's machine** (the probe; Qwen's merged layers from
`wide.py`; the chip on Executor v4):

| sequences | KL forward, before | KL forward, after | KL production text path | noise floor |
|---|---|---|---|---|
| 16 plain (39-65 tokens) | 5.01e-4 | 5.01e-4 | 5.01e-4 | 5.07e-4 |
| 6 short and mid conversations with calls (101-876) | 11.1 to 31.7 each | 4.6e-4 to 1.74e-2 | equal to after | 6.0e-4 to 1.93e-2 |
| 4 fx renders (1,108-2,304) | 4.8 to 8.9 each | 6.2e-3 to 2.62e-2 | equal to after | 5.5e-3 to 1.85e-2 |
| all 10 with markup | **13.7** | **8.57e-3** | 8.57e-3 | 7.16e-3 |
| all 26 | 5.28 | 3.60e-3 | 3.60e-3 | 3.06e-3 |

- **After the fix** the forward equals the text path bit for bit on 26 of 26 sequences. The code
  dims are 0 at every position, and the kept dims never differ from the last position's.
- **Against the floor**, sequence by sequence, the after column is within 1.45x
  (`render:r_tl2_t4`: 7.9e-3 vs 5.5e-3). That excess is the merged layers' own (it is in the
  production text path, which this change does not touch). As in one.md, this is a measurement on
  Qwen's side given the executor's residual; there is no fx-sized checkpoint.
- **Rounding:** to fit the L4 the head now runs as 2-D calls of 512 rows, not one batched call.
  On the first 17 sequences the floor is identical to one.md's, and the merged KLs move by up to
  14% relative (render t1: 8.9067 before the fix, as in one.md; the text path 7.90e-3 against
  one.md's 7.92e-3). E: the GEMM's rounding with the new call shape. Within this code the probe
  repeats exactly: the before and after runs give identical KLs on the 16 plain sequences.

## Exactness after the fix

| check | result |
|---|---|
| circuit digests, fx-int vs fx-fwd (`circ_digest.py`: dims, every node's terms, gates and outputs, sorted so hash order cannot matter; two hash seeds agree): legacy, legacy + text, the test's small paged map + text, fx, fx + text | **identical, all 5** |
| fx's whole session (`fxrun`, 13 units, 4,096 frames, the 16 MB map), no text | 6,666,925 tokens, **0 wrong**, 0 of 1,897,552 end words differing |
| live t1-t5 with `--text`, real Qwen (hf backend), fixed bridge | **5 of 5 solved**; audit 0 wrong over 3.55M / 9.06M / 4.90M / 7.65M / 3.98M chip tokens; rv32emu requests, host I/O and repo identical; 16 of 16 calls end in `</tool_call>`, none in `\x00` (one.md's `tl2`: 14 of 14 did) |
| the probe's t1 run (replay backend), before and after | solved both, audit 0 wrong over 3,552,515 chip tokens each |
| `test_a_parked_machine_ignores_text`, now for both flag values (IR) | 2 passed: with the flag the circuit owns exactly the `</tool_call>` position and predicts GO; without it, it owns no text position, its code stays 0, and the host's GO still unparks and the run audits clean |
| fx's source and ELF | sha256 `282e53d4...` on every run; nothing under fx's source changed |
| the riscv suite on the box (`pytest tests`, the final code) | **423 passed, 25 skipped** (fx-int: 422 and 25; the one more is the parked-machine test's new case) |
| the 63 programs (`chipcheck free` / `tf` on rv-q4p2's chip file) | **not rerun.** They run on fx-exec's tree (fx-int's `core.build` is not rv-q4p2's circuit, `chipfile.md`), and none of the files they use changed here |

## What a merged model can claim

The trigger is inherent to rv-q4p2's design. A causal model at a `</tool_call>` cannot know
whether that call was already served: in a rendered history, the token after it is `<|im_end|>`,
which comes later. Whatever the model predicts there is either GO (and it diverges on every
history) or base's token (and its own calls never run in a stock `generate`). Choosing GO is the
product. So:

**rv-q4p2, as published (M on the prompts above):**
- **Plain prompts.** Without `</tool_call>` in them, its teacher-forced distributions over Qwen's
  vocabulary match base's: KL 6.1e-4 against a floor of 5.0e-4 (16 prompts). That is the card's
  claim, and it holds as worded for such prompts.
- **Prompts with completed calls.**
  - At each position holding `</tool_call>` while the machine is not running, it puts essentially
    all its probability on `<|rv:GO|>`, by design. This includes the system prompt's two format
    examples whenever tools are declared.
  - Nothing else in the residual differs from base, and removing the code there leaves every other
    logit bit-identical.
  - Away from those positions, KL is within 1.4x of the floor on short prompts. It is 1.5x on 876
    tokens with 14 calls and 2.8x on 2,177 tokens with 42 calls, from the merged layers' numerics,
    not the bridge (E).
- **Don't claim** that it scores, or computes the perplexity of, transcripts containing tool calls
  the way base does. The card's KL line should say "on prompts without tool-call markup".
- **Generation from a history is unaffected.** A prompt never ends on `</tool_call>`, so the next
  token after it comes from base's distribution, up to those numerics. The trigger fires when the
  model writes `</tool_call>` itself.
- **After that GO, a stock `generate` is not the calc tool.** It injects no firmware; RvChat does
  (the card says so). What a stock run prints after GO was not measured.

**fx's merged form (with this fix):**
- A plain forward over any text equals the production text path bit for bit (26 sequences, M).
- It is within 1.45x of the noise floor, including renders with 12 completed calls (M).
- `</tool_call>` is ordinary text to it. Only the host's GO and HALT move the machine.
- All of this is measured on `wide.py`'s Qwen layers with the executor as the chip. The HF chip
  layers have not run at fx size, and no fx checkpoint exists (E for a checkpoint).

**Ways rv-q4p2 could shrink the divergence (not done).**
- Not own a `</tool_call>` inside the system turn. That removes the two format-example positions
  but not the history's.
- Or render served calls in their chip-native form, `</tool_call>` GO ... echo, so that the history
  matches what the model would have predicted.

## Also found

- **fx-int's `--text` and probe paths were broken.** `wide.wide_chat` builds an `RvChat` with
  `__new__` and never set `slim`, which f12e268 added: `AttributeError` at the first text feed.
  Fixed (`chat.slim = False`). one.md's text runs predate that merge.

## Runs

- **Box and directory:** q35-eval, `/opt/dlami/nvme/fwd`. Results copied to `~/fwdres` on the root
  disk.
- **Inputs:**
  - `row24/fx-fastcopy.elf` and `imports.json` (sha256 `282e53d4...`);
  - `u1_t1.json` (one.md's replay);
  - three runs' packet logs for the extra renders: `u13` t2, and `tl2` t4 and t3;
  - `~/onres/l13.npz`.

1. `python fx/fxio/text_paged.py base MODEL out/base_fwd.pt u1_t1.json r_u13_t2.json r_tl2_t4.json r_tl2_t3.json`
   - The first 17 sequences and base's distributions on them are bit-identical to one.md's
     `base_lp.pt`.
2. The before probe:
   `python fx/fxio/live_chip.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json MODEL out/pb --tasks t1_bug --reps 1 --full-regs --frames 4096 --units all+ --text --go-on-tcc --backend replay:u1_t1.json --probe 1 --base-lp out/base_fwd.pt --probe-out out/probe_before.json`
3. The after probe: the same without `--go-on-tcc` (`out/pa`, `probe_after.json`).
4. Live with `--text`: `live_chip.py ... out/tl --tasks t1_bug t2_rename t3_feature t4_config t5_parse --reps 1 --full-regs --frames 4096 --units all+ --text`
5. rv-q4p2:
   - `python -m chipformer.rv.fwdcheck MODEL ~/weights/dist/rv-q4p2 --out out/fwd_q4p2.json`
   - `python fx/fxio/q4p2_residual.py MODEL ~/weights/dist/rv-q4p2 out/q4diag.json` and
     `python fx/fxio/q4p2_code_removed.py MODEL ~/weights/dist/rv-q4p2 out/q4zero.json`, run from
     `experimental/riscv` (`fwd.json`: `rv_q4p2_residual`, `rv_q4p2_code_removed`).
6. The whole session: `python -m chipformer.rv.fxrun l13.npz row24/fx-fastcopy.elf run --units all+ --frames 4096`
7. Digests: `python fx/fxio/circ_digest.py MODEL l13.npz row24/fx-fastcopy.elf OUT`, on fx-fwd
   (two hash seeds) and on `git archive fx-int`.
8. `python fx/fxio/report_fwd.py OUT_DIR docs/fx/fwd.json`
