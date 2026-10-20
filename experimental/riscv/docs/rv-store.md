# In-weights object store v2 on the RV32 Qwenputer (status)

Branch `rv-store` (from `rv-p3b` 8488330). Plan: `lets-make-our-own-effervescent-chipmunk.md`.

Code: `chipformer/rv/store.py` (reference, protocol), `chipformer/rv/store_rv.py` (circuit),
`chipformer/rv/core.py` (`build(text=T, store=Config())`), tests `tests/test_rv_store.py`.

## Status

| Phase | State |
|---|---|
| F0 port + F1 hashed directory (IR) | done, gate passed |
| F2 capacity and fit (merge, executor, batch) | done |
| F3 persistence (`save_store` / `load_store`) | done (chat, rvbatch; GPU round trip) |
| F4 merge and GPU checks | done, gate reached (see "GPU results") |
| F5 benchmarks | done (see "Benchmarks") |
| F6 write-up | not started |

Checkpoint `rv-q4store` (q35-eval `~/weights/dist/rv-q4store`, not uploaded): `python -m
chipformer.rv.merge BASE --out DIR --store`, 4 min on the g6.4xlarge, peak RAM 44.7 GB,
22.3 GB on disk (hidden 4608, 12 chip layers + 32 Qwen layers).

## What it is

Five tools next to `calc`: `put(name, text)`, `append(name, text)`, `get(name)`, `list()`,
`delete(name)`. The chip layers capture the name and text while Qwen writes the call, keep them
in Gated DeltaNet heads, and answer at `</tool_call>` with a few chip records and a
`<tool_response>` they print themselves. A `calc` call (any other function) still starts the
RV32 machine.

The answer ends like calc's: `\n<tool_response>\n` body `</tool_response>\n<think>\n\n</think>\n\n`
(each `\n\n` printed as two `\n`); Qwen predicts again after the last `\n`.

## Design as built

**Dispatch.** The op is the token after `function` `=` and before `>` (`=get` / `=list` are
single tokens), so `get_weather` is not `get`. A store call's `</tool_call>` turns `S` on, which
gates the core's GO case and owner term; `calc` calls are unchanged.

**Directory (content-addressed).** A name's key is its first 6 tokens (18 id bits each) and its
length (3 bits). The hash is a fixed random GF(2) map of those 111 bits to 10 bits (parity units,
one MLP sublayer). Collisions probe the next slot (linear probing). A delete leaves a tombstone
(used, not alive) that probes walk past; a re-put of the same name revives it. `list` walks an
order table (slot per name, in order of first creation).

**Records** (one chip token each, `spec.STORE_FAMILIES`, 10-bit payloads):

| Record | Meaning | Writes |
|---|---|---|
| `SP(s)` | probe slot s (the hash, then s + 1, ...) | (reads dir[s]) |
| `SN(g)` | put / append that creates or replaces | dir[s] = (name, alive, head g, tail g), len[g], NEXTG = g + 1 |
| `SA(g)` | append to a live file | next[tail] = g, tail = g, len[g], NEXTG = g + 1 |
| `SO(n)` | the slot is new | order[n] = s, NORD = n + 1 |
| `SD` | delete | dir[s] becomes a tombstone |
| `SG(c)` | get: chunk c is printed next | GCUR = c |
| `SL(i)` | list: order entry i | LCUR = i |

The slot s of SN / SA / SD / SO is the last SP's: those heads take it through a conv tap (lag 1
or 2) of the SP payload's group features, so no register is read for it.

**Tables** (one-hot keys, exact overwrites, no decay; at the 4B size):

| Table | Rows | Key heads |
|---|---|---|
| content | 512 chunks x 8 rows of 14 tokens (2 value heads x 7) | 32 |
| dir | 1024 slots: name, length, alive, used, head \| tail | 8 |
| next | 512 chunks: next chunk, has next | 4 |
| order | 512 entries (value head = index bit 0) | 2 |
| len | 512 chunks (value head = chunk bit 0) | 2 |
| regs | NEXTG, NORD, GCUR, LCUR | 1 |

Capacity: 512 chunks x 104 tokens = **53,248 tokens**, 512 names (ever created), 1024 slots
(at most 50% load). A put / append is one chunk; a file is a chain of chunks.

**Content rows.** A staging head holds the current 7-token value-head row so far (one row per
slot, the slot-0 write carries the ballast and clears the rest); every text token rewrites its
content row with it, so the row's last write holds all 7 tokens. A get's print reads chunk GCUR,
row H // 14, and the output gates pick value head (H // 7) % 2 and token H % 7 into 18 dims.

**Any token in, any token out.** Every Qwen row carries its 18 id bits (input) and an id code
`128 * (2 bit - 1)` (output). To print token X the exit leaves the class code at 0 and
`KAPPA * (2 bit - 1)` of X's id in the id dims (as v1). Special tokens (`<|endoftext|>` and up)
are stored as token 0.

## Changes from the plan

- **Records are single RV-style tokens** (7 store families after the controls) instead of v1's
  ADDR / VAL pairs: one token per record.
- **Hash:** a seeded random GF(2) linear map instead of a plain XOR fold (same parity units, better
  spread for names that differ in few bits). Tests use `hash_bits` 0 to 3 to force collisions.
- **Tombstones** are revived only by the same name; a different name never takes a dead slot. So
  at most 512 names can ever be created (deleted ones count), and `list` keeps a re-created name
  in its first place.
- **Content packing: 14 tokens per key row (7 per value head) instead of v1's 8**, through a
  staging head. That takes 32 content heads instead of 52 for 512 chunks. With v1's packing the
  build had 15 chip layers; with this one, 12.
- **len and next are separate tables** (v1 had one meta table), so an append writes its chunk's
  length and the old tail's link at one token without reading the old length first.
- **`full`** is answered when the chunks or the names run out (v1 answered `ok` and dropped the
  text).
- **Name dispatch** uses token detectors on the row features (the id bits are there anyway), not
  byte features.
- The any-print id code is computed in the exit sublayer itself (no extra sublayers).
- `ir.kernel_check` exempts values below a head's `tiny` bound, as `evaluate` already did (the
  core's own `cap` counter underflows there on long calls).

## Costs (IR, Qwen3.5-4B target)

| | core + bridge | + store |
|---|---|---|
| chip layers | 11 | 12 |
| key heads | 103 | 164 |
| IR dims | 2918 | 3753 |
| residual slots | 1265 | 1806 |
| FFN units | 42,114 | 47,532 |
| hidden (`pick_hidden`) | 3968 | 4608 |
| checkpoint (estimate) | 18.7 GB | about 22 GB |

The store's live dims at the core's peak (sublayers 12 to 18) are 454; the name buffer (108) and
the staging head (126) are most of it. Hidden 4352 needs at most 1775 slots. Placing the
late-read FFNs just before their readers was tried: the peak fell to 1530 live dims, but the
slots went up (1822 to 2258, and 13 layers when they were forced into sublayer 18, whose MLP is
nearly full, so the allocator had no room for handover units). Reverted.

Chip tokens per op (reference, 4B size, random names; mean at 256 / 448 names in 1024 slots):

| op | chip tokens |
|---|---|
| put, new name (SP... SN SO) | 3.15 / 3.36 |
| put, existing (SP... SN) | 2.15 / 2.5 |
| append, existing (SP... SA) | 2.15 / 2.2 |
| get, found, 1-2 chunks (SP... SG...) | 2.15 / 2.25 |
| get, not found | 1.38 / 1.73 |
| delete, found (SP... SD) | 2.05 / 2.8 |
| list | 1 per name ever created |

v1 needed 2 per directory entry up to the match for every op.

## Tests (`tests/test_rv_store.py`, IR, every bf16 rounding checked)

- Token ids match the tokenizer; `call_tokens` spells calls as the chat template does.
- Reference vs a plain dict: 1000 random ops at the 4B size, plus 400 with a 2-bit hash, 200 with
  a store that fills up, and 300 with every name colliding.
- Circuit vs reference, token for token and owner at every position:
  - all ops, empty texts, a revived tombstone;
  - calc calls next to store calls;
  - random calls (tiny hashes, overwrites, append chains, deletes and re-puts, names equal in
    their first 6 tokens, empty and over-long texts, any token, full stores, calc in between);
  - the full 4B size with the HF kernels' head reads replayed (`kernel=True`);
  - thousands of text tokens between calls;
  - a fresh session whose store heads start from session 1's state (the snapshot).

- Extended fuzz (not in the suite): 2 x 150 random ops with every name colliding (`hash_bits` 0,
  16 slots) and a 3-bit hash (64 slots), calc calls in between: 39k tokens, 26,249 owned
  positions, 0 mismatches.
- Existing suites on this branch: test_rv_bridge 9, test_rv_ref 130, test_rv_fuzz 4,
  test_rv_core 94, test_q35_ir 3, test_q35_ir_vheads 1, all pass. test_rv_hf crashes with a
  Windows access violation in torch 2.14.1+cpu here, in the untouched rv-p3b worktree too; it
  is run on the GPU box in F2.

## GPU results (q35-eval, L4; `rv/storecheck.py`)

- **Teacher-forced exactness** (a reference stream of 60 random store calls with random tokens,
  7 calc runs, 17,612 positions, 14,564 owned): 0 wrong and 0 owner mismatches on the HF chip
  modules decoding token by token, on the executor, and (for information) as chunked prefills.
  The store heads' states after the stream are bitwise equal on both paths. The executor also
  equals the IR on every dim at every position of that stream (`tritonexec` vs `ir.evaluate`).
- **Live, Qwen deciding** (6 prompts: save; save + append + get; two puts + list; put +
  delete + get; calc then save the result; put + get), HF and executor side by side
  ("verify"): 0 lock errors, 0 head mismatches, 0 verify mismatches; the executor alone gives
  the same tokens; IR replay of every stream clean. Qwen called the tools on its own and
  restated the results after each response.
- **Save / restore:** session 1 stored 4 facts with `put` (30 state rows saved); 3 fresh
  sessions (empty context, the store loaded) read them back with `get` / `list`: the printed
  texts are the stored ones exactly, Qwen answered from them, lock and IR replay (from the
  snapshot) clean.
- **Batch:** 4 live prompts decoded together in a 4-row rvbatch and each alone in it: the same
  tokens, 0 lock errors, 0 head mismatches.
- **HF modules vs IR** (`test_store_in_hf`, chip-only model, prefill and decode, every dim at
  every position): pass; `test_rv_hf` passes on the Linux box.

- **Executor after the bf16 change** (`rv/execsuite.py`, q35-eval): the executor equals the IR on
  every dim at every position of the 51 rv32u* programs (19,609 positions) and of a bridge
  stream with 9 calc runs on the store circuit (25,251 positions). Both arithmetic regressions
  above (160 problems) ran on the new executor: 0 lock errors.
- **Text fidelity** (`rv/textcheck.py`, 16 prompts): mean KL vs base 5.48e-4, noise floor
  (base sdpa vs eager) 5.07e-4; greedy text identical on 10/16 (rv-q4p2: 6.1e-4, 10/16);
  tokenizer identical 4/4.
- **Arithmetic regression** (`evalarith`, n = 40, batch 8, the fixed calc firmware on every
  side; exact rational match):

| Set | rv-q4store (L4) | rv-q4p2 (L4) | rv-q4p2 (A10G) | rv-q4p2 published (L4, batch 1, calc v1) |
|---|---|---|---|---|
| arith | 36/40 | 38/40 | 37/40 | 39/40 |
| dmx_mul_big | 39/40 | 39/40 | 39/40 | 39/40 |

  Every miss is Qwen's: 0 lock errors, 0 head mismatches, and every machine result exact.
  The misses are 42 to 50-call loops on a result the machine returned correctly, and one
  wrong final answer after correct results. The store model and rv-q4p2 on the same L4 give
  identical token streams on 23 of 40 arith problems (the two rv-q4p2 runs on different GPUs:
  25 of 40), among them three 100k-token loops; they split only on arith-18 and arith-39, in
  Qwen's text. So the store moves Qwen's rounding about as much as a change of GPU does.
- **IR replay** of the 40 rv-q4store arith records (the reference with the store, the IR from
  the snapshot): 40/40 clean, 453,152 positions, 417,834 circuit-owned, 0 wrong predictions,
  0 owner mismatches.

Bugs found on the GPU and fixed:

- `rv/tiny.embedding_matrix` built rows only for the core's families: the store's chip tokens
  had zero rows in the merge (no ballast), so the HF path dropped every store record.
- The executor computed MLP units in plain fp32, where `7 - 2^-22` (a threshold unit) is a
  rounding tie that 7 and 6 break in opposite directions: 0/1 dims carried 2^-21, and a decay
  selector of 238 * 2^-21 cleared the staging head (1 wrong content token in 14,564). Its MLP
  now rounds the gate, the up value and their product to bf16, as the HF model and the IR's
  hardware model do, and beta / decay are read by nearest value (as sigmoid / softplus are in
  HF). Exact values are unchanged.

## Benchmarks (F5; `rv/storebench.py`, tables by `rv/benchsum.py`)

Greedy decoding at batch 1, budget 512 tokens per answer (2048 for a needle save turn). The
Qwenputer runs on the executor with every circuit token checked in lockstep against the
store's reference: 0 lock errors and 0 head mismatches in every run below. The host-tool
baseline is base Qwen3.5-4B with the same tools, system prompt and messages; its tool calls are
answered by the store's reference in Python, which prints the same responses token for token.

**How Qwen was prompted (a nudge, in both tool conditions).** System prompt:

> You have a memory that keeps texts across conversations, with the tools put, append, get, list
> and delete, and a calc tool for exact arithmetic. When the user tells you something about
> themselves (facts, preferences, changes), save it in your memory with put, one short name per
> fact; when a fact changes, put the new value under the same name. Before you answer a
> question about the user, look in your memory first (list, then get).

The tools are the five store tools plus calc (`store.TOOLS`, `chat.CALC_TOOL`). The base and
context conditions get "You are a helpful assistant." and no tools.

### A. Multi-session recall (synthetic, LongMemEval-style)

30 cases (seed 0): 4 past sessions each, one user message per session, chit-chat around 2 facts
about the user (city, job, dog, food, partner, car, hobby, allergy, sister, gym day; 8 facts per
case); the 4th session also changes 2 earlier facts. Then 3 questions, each in a new session
with an empty context: a fact from sessions 1 and 2, a changed fact (the new value must appear,
the old one must not), a fact from sessions 3 and 4. n = 90 questions (60 facts, 30 changed).
Scored on Qwen's own words (tool calls and responses removed), case-insensitive.

| Condition | All | Facts | Updated facts |
|---|---|---|---|
| Base Qwen3.5-4B, fresh session, no memory | 7/90 | 6/60 | 1/30 |
| **Qwenputer + in-weights store** | **75/90** | 48/60 | **27/30** |
| Base Qwen + the same tools run by Python on the host | 71/90 | 46/60 | 25/30 |
| Base Qwen, the past sessions pasted into context | 69/90 | 60/60 | 9/30 |

- Qwen saved on its own in 93 of the 120 past sessions with the store (318 put / append
  calls) and in 94 of 120 with the host tools (376 calls). The store's 15 misses: 11 facts
  never saved, 3 saved but not read back, 1 read back and answered wrongly. Host: 14, 4, 1.
- The store and the host baseline are both right on 70 questions; 5 only with the store, 1
  only with the host. They run on different GPUs (L4 / A10G) and the merged model's rounding
  differs, so Qwen's choices of what to save drift apart.
- With the sessions in context Qwen gets every plain fact but mostly answers a changed fact
  with its old value, or with both.
- Time: the store run took 79 min of generation (L4); host 48 min (A10G); base and context a
  few minutes each.

### B. Long-context needle

N named strings (`s1`..: two words, a code, a word, a decimal; about 15 tokens). Save phase: 10
per user message, "Save each of these in your memory with put, one call per item, the name and
the text exactly as given:"; each message in a session of its own (the store carried over).
Then one question per string: plain "What is the text for s17? Reply with the text only." or
tool "Use the get tool to read s17, then reply with the text only.", either in a fresh session
(empty context, the store loaded) or in the same conversation after P tokens of unrelated chat
(GSM8K problems and answers as user / assistant turns). Base in-context: the strings are given
in the same messages (each answered with a fixed "Got it, I'll keep them in mind.") and only
the plain ask. Exact match of the string in Qwen's own words. n = 50 questions per cell.

N = 50 (all 50 saved by both tool conditions):

| Setting | Base in-context | Store, plain ask | Store, tool ask | Host tools, plain | Host tools, tool ask |
|---|---|---|---|---|---|
| fresh session (empty context) | (cannot know) | **50/50** | **50/50** | 50/50 | 50/50 |
| same conversation, 10k tokens of padding | 49/50 | 44/50 | 42/50 | 44/50 | 42/50 |
| same conversation, 100k tokens of padding | 37/50 | 30/50 | 12/50 | 30/50 | 12/50 |

N = 200, all 200 saved: fresh session store 50/50 (plain) and 50/50 (tool), host tools 50/50
and 50/50; base with the 200 strings in context and 10k of padding: 44/50.

- In the same conversation the strings are still in the context (the user's save messages),
  and Qwen answers from there: it called `get` on at most 3 of 50 questions even when told to.
  At 100k tokens Qwen no longer uses tools defined that far back (as in v1): the store's
  100k cells give 96 of 100 answers identical to the host baseline's, all from context.
- So the store wins where the context does not hold the strings (a new session: 50/50 at
  N = 50 and N = 200, against nothing for base), and inside one long conversation it adds
  nothing over the context, because Qwen does not call it there.
- The store's answers are its own text read back exactly; the store's per-question time is 5-6
  s (L4 / A10G, a `get` round trip), the host's 2.3 s (A10G / L4).

Infrastructure notes:
- The first needle runs kept the save phase in one growing conversation whose earlier
  assistant turns showed only their replies (tool calls stripped): Qwen then answered "saved"
  without calling put from the 5th message on (40 of 50 and 60 of 200 saved). Those runs were
  discarded; each save message now has its own session.
- A 100k-token prompt does not fit the merged model (18 GB with the chip layers compacted) and
  its KV cache on a 24 GB GPU. HF's offloaded cache (`offloading=True`) gave wrong answers (a
  20k-token check: it answered a different string than the same prompt without it), so the
  100k store cells ran on an L40S (q35-tc), with the checkpoint rebuilt there (tensor for
  tensor the same as q35-eval's: 594 of 594 sha1 equal).
- A shared prompt prefix is prefilled once per cell (`RvChat.make_prefix`, the base model's
  equivalent) and each question runs only its own tokens; the prefix and no-prefix paths give
  the same tokens (checked at 20k).

GPU time for F5 (box start to stop): q35-eval (L4) 2.1 h, q35-eval2 (A10G) 1.9 h, q35-inline
(A10G) 1.4 h, q35-batch (L4) 1.2 h, q35-tc (L40S) 0.3 h: 6.9 GPU-hours in all (F2-F4 before it: about 2.8, q35-eval and q35-inline).

## Found along the way

- `calc-bridge.elf` (rv-p3b) answered wrongly when calls followed each other (`12*34` then `7*6`
  gave `2`): digits left in the bignum registers. Fixed on rv-p3b (456c947) and cherry-picked
  here (05ad59b); test_circuit_with_calc now checks all three answers.
- `fastexec.Generator` iterated `OUT_NAMES` with `S.tok("Q", 0)` (a KeyError since P3 added
  `Q`); it now skips `Q`.
