# fx on a dedicated Qwenputer: ways to cut wall time

Design and analysis only; no circuit code changed. Status 2026-11-06.

The question: run Vercel's fx (its unmodified Zig source, compiled to RV32) on a Qwenputer that
exists only to run fx, with Qwen as fx's model, and find everything that cuts wall time. Each
idea is rated against the claim **"exact fx code"**:

- fx's own source compiles unmodified;
- the platform may add a root file, build flags and config, and the host functions / OS layer;
- the compiled program's semantics stay exact;
- the weights, not the host, do fx's computation.

**Inputs.**
- The circuit and executors: `docs/rv32.md`, `chipformer/rv/{spec,core,tritonexec,chat}.py`; the
  vx and rv-store branches (`docs/vx.md`, `docs/rv-store.md`).
- fx's source (`src/wasm_core_main.zig`, `src/core/agent`, `src/gateway`, `src/acp`,
  `src/tools/filesystem`).
- The P0 agent's raw runs: fx-core.wasm under a fuel-counting runtime, with a scripted model.
  - Files: `runs/stock.json` (stock build) and `runs/names_instr.json` (per-function counts) in
    the P0 scratchpad.
  - Scope: 7 model calls (6 in a first prompt with read / grep / edit / write / glob, 1 in a
    second prompt) on a toy repo.
  - `docs/fx/p0-report.md` was not written yet when this was. Its RV32 numbers should replace
    the conversion factors below.

## Ranked summary

Rank = how much slower the recommended stack gets without the idea, on the reference task
below. The baseline is fx as is, on today's machine, at today's speeds.

| # | Idea | Speedup | Cost | Exactness risk | "Exact fx code" | Depends on |
|---|---|---|---|---|---|---|
| 1 | One SSE event per model reply (server side) | ~100x on the stream term (64% of baseline); 2.7x on the task alone; **9.5x** leave-one-out | none in the circuit; a server flag | none | honest | the model server (host or in-weights) |
| 2 | Executor v4: one persistent step kernel, GPU-resident decode loop, memory layers flattened, lockstep off | 3-6x on all chip time; **4.4x** l-o-o | 0 layers; 2-4 weeks of Triton/CUDA | low (bit-exact checks exist) | honest | none (needed by #12) |
| 3 | Secret-scan unit (fx's `maskSecrets` loop) + compare-with-constant unit | ~100x on post-stream cost per tool-argument byte; **4.2x** l-o-o (more on write-heavy tasks) | 3-7k MLP units, 0-1 heads, maybe +1 layer | medium (loop equivalence) | honest (ROM specialization) | P0 call-site attribution; #7 payloads |
| 4 | Batch circuit-emitted text into Qwen as a chunked prefill, re-tokenized to BPE | 169 s -> ~0.4 s of Qwen time per step; **3.0x** l-o-o | host decoder change | none for the circuit | honest; host-side BPE is the model server's tokenizer (disclose) | none |
| 5 | A Qwenputer model provider bound in the platform root (deltas only, no request body) | removes ~150 wasm instr per body byte per step, the SSE parse and the configured-provider SHA-256; **2.3x** l-o-o | ~1-2k lines of platform Zig | none | honest with disclosure (fx's provider contract; gateway serializer not exercised) | Qwen chat template + tool-call parser |
| 6 | Streaming units for the glue: SHA-256, wyhash, JSON scanning, token estimate, UTF-8, memcpy / memset / memchr | ~1.7x on remaining CPU tokens; **1.55x** l-o-o | ~15-30k units, 2-6 heads, +1-2 layers | medium | honest | P0 profile; #7 |
| 7 | Record formats and address map: 1-token LUI / AUIPC, 16-bit payloads, SB / SH in 2 tokens, hot code first | 1.25-1.4x tokens per instruction; **1.26x** l-o-o | vocab +100-200k rows (+1-1.5 GB), ~1-3k units | low | honest | linker symbol ordering (#8) |
| 8 | Build flags + fx config | 1.15x instructions + 1.1x work avoided; **1.23x** l-o-o | flags; Zbb ~2-4k units | low | honest | — |
| 9 | Fast Qwen decode (live width, Qwen-only lm_head, fla / conv kernels, CUDA graph) + thinking budget | 65 -> ~30 ms per text token; **1.17x** l-o-o (1.5x with long thinking) | host decoder work | none for the circuit | honest | — |
| 10 | Speculative trace decoding with a drafter that never runs fx (n-gram / stride over the circuit's own trace) | 1.5-3x on chip time (**1.7x** on the stack) | chunked verify mode in the executor; 3-6 weeks | low-medium | honest if the drafter is semantics-free (disclose) | #2 |
| 11 | Boot snapshot through `initialize` + `libfx/new` | removes ~2M wasm instr per task (11 min now, ~2 min after #2) | days | none | honest (the calc firmware's park is the precedent) | memory fits; deterministic clock / random |
| 12 | Memory architecture for MBs (dense ROM, word-slot RAM sized to the measured peak, bulk regions, file store) | enabling; avoids a ~2x slowdown | +30-100 layers, +12-40 GB checkpoint | low | honest | #2 flattening |
| 13 | OS layer as circuits: Zig `std.Io` on in-weights stores, silent ACP sink, wide capture window, any-token print | ~1.1-1.3x, mostly enabling | 1-3 layers of buffers | low | honest; host-run tools would not be | #12 |
| 14 | Profile-chosen fused pairs (auipc+jalr, slt+branch, ...) | 1.05-1.15x tokens | 1-3k units, maybe +1 layer | low-medium | honest | P0 trace |
| 15 | Pruning unused circuit features and code | ~1.0-1.1x (frees room for #3 / #6) | removes ~9k units, 16 heads | low | honest | — |
| 16 | DIV64 for compiler_rt's `__udivdi3` family | <= 1.05x unless u64 formatting is hot | 8-15k units | low | honest | P0 profile |
| 17 | An RV64 core | 1.1-1.3x instructions, 2x bytes per RAM slot | a new core, months | medium | honest | — |
| 18 | Longer R refresh period | < 1% | trivial | none | honest | bf16 thermometer cap (m < 88) |
| 19 | Whole-basic-block records | net <= 1.1x after the extra depth | +4-8 layers | medium | honest | — |
| 20 | Speculative decoding with the host's reference CPU drafting | 10-50x | moderate | none | **violates**: the host runs fx | — |
| 21 | Batching several fx sessions | throughput 5-8x, no latency gain | exists (`rvbatch`) | none | honest | — |

**Combined estimate.** The recommended stack is 1-9 and 11, plus 12-13 as enablers.
- Per agent step: about **83 s** (range about 1-2.5 min), against **4.3 h** as is.
- Per 20-step task: about **28 min**, against **86 h**. That is about 190x (range 100-300x).
- Adding #10: about 48 s per step and 16 min per task.
- Keeping fx's own wire serializer (no #5): about 2-3 min per step.

## Where the time goes (the cost model)

### What P0 measured (wasm fuel, stock fx-core.wasm)
- **Build:** 2.35 MB of code, 0.30 MB of data, a 1 MB stack reserve.
- **Session:** 50.0M fuel in all. Peak linear memory 3.2 MB.
- **Before the first prompt:** about 2.0M.
  - 0.3M before `main`, 0.2M of boot to the first read.
  - `initialize` 1.2M, `libfx/new` 0.3M.
- **Streaming:** **about 72k wasm instructions per SSE event**, over all 7 streams:
  - 11 events take 0.78M; 67 take 5.06M; 145 take 10.4M.
  - With chat-completions streaming, one event is one token.
  - The cost: parsing each event into `std.json` dynamic values (hash maps, so wyhash and
    `mem.eqlBytes`), `json_depth.exceeds`, the SSE reader, and an ACP `session/update` per chunk
    (198 of them).
- **After the stream, before the tool runs:** about 1.4M plus **about 16k per byte of tool
  arguments**.
  - edit_file with 249 bytes: 5.7M. write_file with 206: 4.7M. glob with 22: 1.8M.
  - The per-byte part is mostly `mem.eqlBytes` (63%) and `maskSecrets`. `maskSecrets` runs
    `findSecretSpan` at every byte: about 40 prefix probes (`secret_prefixes`, inline keys,
    AWS, GitHub, basic auth), and tool presentation runs it on the arguments for display.
  - The fixed part is about 60% SHA-256. fx calls `configured_provider.binding_identity` many
    times per step (1,340 `hash_part` calls in the session).
- **The request:** about 0.3M plus **about 150 per request-body byte**, every step.
  - Steps 1-5 grow from 0.63M at 2.4 KB to 0.88M at 4.1 KB.
  - The work: building the body, then `measureProviderRequest` / `textTokens` over all of it
    (24%), JSON string encoding (13%), UTF-8 validation (11%).
- **Hot functions** over the session (names build, counted instructions):

| Function | Share |
|---|---|
| `mem.eqlBytes` | 20.6% |
| SHA-256 block | 11.8% |
| `Wyhash.hash` | 8.1% |
| `json.Scanner.next` (+ `nextAllocMax`, `peekNextTokenType`, `skipWhitespace`: +3.8%) | 5.1% |
| `gateway.json_depth.exceeds` | 3.0% |
| `mem.findScalarPos` | 2.7% |
| `token_estimate.consume` | 2.6% |
| `ArenaAllocator.alloc` (+ resize 0.9%, `malloc.realloc` 1.5%) | 2.3% |
| `muldXi` + `__multi3` (wyhash's 128-bit products) | 3.4% |
| `ArrayHashMap(json Value).get*` | ~3.8% |
| `encodeJsonString`, ACP `writeJsonStr`, `Writer.writeByte` | 4.3% |
| `maskSecrets`, `utf8ValidateSlice` | 2.8% |

  The top 20 functions are about 75% of the session.
- **Not in P0:** P0 used libfx's host tools (`fx_host_tool_call`), so native tool costs (fx's
  Zig `read_file` / `edit_file` over WASI) are not in these numbers.

### Conversion to the Qwenputer (assumptions, to be replaced by P0's RV32 numbers)
- **RV32 instructions per wasm instruction: 0.75** (range 0.6-1.0).
  - Wasm's local.get / set are register moves on RV32.
  - Against that, i64 operations cost 2-4 instructions, rotates cost 3 without Zbb, and
    `memory.copy` becomes a loop.
- **1.45 tokens per instruction.** That is MicroPython's 1.24 plus the effect of a MB heap: most
  pointers become 2-token values, and Zig's byte stores are 4-token SB records.
- **Chip speed:** 3,200 chip tokens per second (RvChat with Executor2 on an L4).
- **Qwen:** 65 ms per text token at batch 1. Circuit-emitted text goes through Qwen one byte
  token at a time (`chat.generate`; `bridge.Session.run` echoes every putchar).

### The reference task
- One prompt, 20 agent steps.
- Request body 13 KB plus 3.8 KB per step (mean 49 KB, last 85 KB).
- Qwen writes 400 tokens per step with its thinking capped. One SSE event is one token.
- Tool arguments 400 bytes per step on average; 2.6 KB of new context per step.
- **As is: 34M RV32 instructions per step** = 49M chip tokens = **4.3 h per step, 86 h per task**.
  - Shares: the stream 64%, the request 17%, after the stream 17%.
  - Qwen: 1.3% (26 s of generation plus 169 s of byte-by-byte ingest per step).
- Without the executor's flattening, the extra memory layers (#12) would halve the chip speed:
  172 h.

The model is `tools_fxspeed/model2.py` and `loo.py` in the scratchpad. All factors below are its
inputs.

## 1. One SSE event per model reply
- **What:** the model server (the `fx_http_stream_*` host functions) sends Qwen's whole reply as
  one chat-completions event:
  - the content, then the tool call with its complete `arguments`, then the finish event and
    `[DONE]`;
  - no reasoning deltas.

  fx reduces any valid chunking to the same completion. It just sends fewer ACP chunk updates.
- **Speedup:** the stream term falls from 72k x 400 to about 0.3M wasm instructions per step
  (about 100x). That is 2.7x on the as-is task, and the biggest single item in the stack.
- **Cost:** none in the circuit or in fx; a server flag.
- **Exactness:** none.
- **Honesty:** honest. The chunking is the server's choice, and fx runs exactly on the bytes it
  receives.
  - If the host is the model server (it parses the request and builds the SSE), it is doing a
    model server's job, not fx's.
  - The pure version runs that server as platform code on the CPU, O(new bytes) per step.
- **Depends on:** none.

## 2. Executor v4
- **Today:**
  - Executor2 runs about 0.177 ms per step standalone, 0.31 ms inside `RvChat`, where Python
    work per token (`_exec_step`, `code_token`, the loop) brings it to 3,200 tok/s.
  - FreeRun3 (vx branch: 24 launches per step, decode and embed on the GPU) does about
    5,500 tok/s on 13 layers.
- **Steps:**
  1. Keep the chip-token loop on the GPU until the predicted token is text or HALT: a CUDA-graph
     while node, or a persistent kernel. No Python per chip token.
  2. One persistent kernel per step with a grid barrier per sublayer: about 25 barriers at
     1-2 us plus 1-2 us of work each, so 50-100 us per step.
  3. Skip heads whose query matches no row: their output is exactly 0, as the top-k gathers
     already assume.
  4. Run consecutive memory sublayers in one phase when they only read addresses computed below
     them (ROM banks read `w`, RAM banks read `EA`). The sum of one live head and exact zeros is
     exact. This is what makes #12's tens of memory layers cheap.
  5. bf16-coded sparse weights with delta-coded indices, about 2-4 MB per step, all in L2.
  6. Lockstep off in production. The Python reference machine would cap the speed, and it
     means the host runs fx. Keep IR replay as an offline audit.
- **Speedup:** 3,200 -> 10-20k tok/s (3-6x); 16k is used in the model.
- **Cost:** 0 layers or units; 2-4 weeks.
- **Exactness:** low risk. Check it with the existing `execcheck` / verify mode and free-running
  checks against the IR.
- **Honesty:** honest. It is the same arithmetic over the weights, checked bit-exact.
- **Depends on:** none.

## 3. Secret-scan unit and compare-with-constant unit
- **What fx does:** `text_utils.maskSecrets` walks the text a byte at a time. At each byte,
  `findSecretSpan` tries:
  - `matchesBasicAuthUrl`, `matchesAwsAccessKey`, `matchesSensitiveAssignment`;
  - every `secret_prefixes` entry, through `eqlIgnoreCase`;
  - `matchesInlineKey` / `matchesGithubToken`.

  Then it `writeByte`s the byte into an Allocating writer. Tool presentation runs it on
  `arguments_json` and on results (`tool_presentation.zig:958, 1479`). P0 puts it, with the
  `mem.eqlBytes` it drives, at about 16k wasm instructions per argument byte.
  - A 4 KB `write_file` costs about 64M wasm instructions here: about 73 min at 16k tok/s
    without the unit.
- **(a) Compare-with-constant unit:** `mem.eql` / `startsWith` / `eqlIgnoreCase` against a DROM
  constant, as one record of 1-3 tokens.
  - A RAM word and a DROM word come from separate read ports in the same position, so a compare
    needs no data tokens.
  - It replaces about 300-700 instructions per call.
  - ~2k units.
- **(b) maskSecrets loop macro:** a ROM macro-uop at the loop head's PC.
  - It copies runs of bytes that cannot start any of the fixed patterns, judged by the (previous
    byte, byte) class.
  - The bytes reach the writer's buffer through the stream at 2 tokens a word.
  - It exits to the original code at every candidate position, and before the buffer's capacity
    ends (so `drain` / realloc run as compiled).
  - ~3-5k units, 0-1 heads.
- **Speedup:** about 100x on the per-byte term (the model assumes a conservative 150 wasm
  instructions per byte left, for the candidates' fallback). 4.2x leave-one-out; larger on
  write-heavy tasks.
- **Exactness:** medium. The macro's architectural effect must equal the loop's: the buffer
  bytes, `end`, every register the loop leaves live. Check it by differential fuzzing against the
  reference stepping the original instructions, and by IR teacher-forcing.
- **Honesty:** honest. The binary is unchanged, and the CPU is specialized to its fixed ROM (a
  first Futamura projection). The reference decodes the same macro-uop.
- **Depends on:**
  - P0 call-site counts, to confirm that `eqlBytes` comes from this path;
  - 16-bit payloads (#7) for 2-token words;
  - stream records of at most 64 tokens (the j thermometer's bf16 cap), so long runs re-dispatch.

## 4. Batched ingest of circuit-emitted text, re-tokenized
- **What:**
  - Echo tokens are circuit-owned, Qwen predicts nothing during a run, and Qwen's layers never
    feed the chip layers. So `chat.generate` can buffer each echo's `hc` and send the run's
    text through Qwen's 32 layers as one chunked prefill at HALT. This can overlap the next
    chip burst on a second stream.
  - Today 2.6 KB is 2,600 byte tokens at 65 ms: 169 s per step. A prefill at 4-7k tok/s takes
    0.4-0.7 s.
  - Re-tokenizing to Qwen's BPE (about 3.5 bytes per token) gives about 750 positions in about
    0.15 s.
  - It also matters for quality: fx's tool results arrive as single-byte tokens today, which is
    far from what Qwen was trained on.
- **Three ways to get BPE:**
  - (a) The host re-tokenizes the echoed bytes before Qwen's layers.
  - (b) **Any-token print:** rv-store already prints any Qwen id from the circuit. The CPU-side
    server picks ids by greedy longest match over a vocab trie in DROM (2-4 MB, 4-8 dense
    layers, ~20-40 instructions per byte).
  - (c) Keep bytes.
- **Speedup:** about 170 s per step; 3.0x leave-one-out.
- **Cost:** host decoder work, or (b)'s DROM layers.
- **Exactness:** none for the circuit. Qwen's numerics move at bf16 rounding (prefill vs decode),
  the same class as batching.
- **Honesty:** honest for fx. With (a) the host does the model's tokenizer, as any server does
  (disclose). (b) is the pure version.
- **Depends on:** none.

## 5. A Qwenputer model provider bound in the platform root
- **What:**
  - The root binds its own `stream_provider.Provider`, as `wasm_core_main.zig` binds
    `js_host_stream_provider`.
  - It sets `build_request_fn = null`. The orchestrator then skips building and measuring a body
    (`orchestrator.zig` around 7818-7870).
  - Its `stream_fn` renders only the messages added since its last call into Qwen's chat
    template, prints them to Qwen, and returns a `ModelCompletion` parsed from Qwen's
    `<tool_call>`.
  - No JSON body, no SSE, no `FX_CONFIGURED_PROVIDERS`, so no `binding_identity` hashing.
  - If fx rewrites history (compaction, truncation, steering), the provider detects the
    divergence by comparing message identities and re-ingests from there. The host keeps Qwen
    state snapshots at message boundaries, because DeltaNet states can't be rolled back.
- **Speedup:**
  - Removes the request term (about 150 wasm instructions per body byte per step: 17% of the
    baseline, about 60% of the step once #1-#4 are in).
  - Removes the stream parse left after #1 and most of the fixed SHA-256.
  - 2.3x leave-one-out, growing with session length.
- **Cost:** ~1-2k lines of platform Zig. Per step it costs O(new bytes) of template rendering
  and tool-call parsing.
- **Exactness:** none.
- **Honesty:** honest with disclosure.
  - fx's contract says "Providers own validation, wire serialization, endpoint selection,
    headers, HTTP, and stream reduction" (`stream_provider.zig`). Binding one is what the WASM
    and NAPI roots do.
  - fx's agent runtime, tools, history and presentation run unmodified.
  - Not exercised: the gateway / chat-completions serializers, the SSE parser, and request-size
    measurement. Without a serializer, fx's capacity checks fall back to its own defaults, which
    is fx's designed behaviour for such providers.
- **Strict alternative:** keep fx's chat-completions provider (`host_chat_completions`) with a
  one-entry provider config, plus #1. Streaming units for the request path (token estimate, JSON
  escaping, UTF-8, copies) halve the 150 per byte. That costs about 2-3 min per step instead of
  about 1.4.
- **Depends on:** the Qwen tool-call parser and chat template, in platform code.

## 6. Streaming units for the glue
- **The rule they follow:**
  - A streaming record is one token per element.
  - Loop-carried state either fits a token's payload (16 bits) or lives in a state head written
    from payloads, as in vx.
  - Data that moves RAM to RAM passes through the stream, because a write takes its value from
    below at t-1. So a copied word costs 2 tokens with 16-bit payloads.
- **The units:**

| Unit | Now (RV32 loop) | With the unit | Notes |
|---|---|---|---|
| memcpy / memmove (RAM to RAM) | ~1.75 tok/B aligned, ~8 misaligned | 0.5 tok/B | funnel from the previous word via a lag tap |
| memset | ~1.5-6 tok/B | 0.25 tok/B | data from the register (SW-like) |
| memchr / `findScalarPos`, escape scan, whitespace token estimate, UTF-8 validation, `json_depth` | ~8-15 tok/B | ~0.25-0.4 tok/B | read-only scans, SC-like continue tokens; UTF-8 carries <= 10 bits |
| compare with RAM operand (`eqlBytes`, map keys) | ~2-3 tok/B | ~0.6 tok/B | one RAM port, so one operand rides the stream |
| SHA-256 block (no Zbb) | ~70 tok/B | ~6 tok/B | 2 new words a round plus W_t; state head for b..h and W |
| wyhash round / 128-bit `mum` | ~40 instr per mum | ~4 tokens | 64-bit product through the existing multiplier columns, twice |
| `Writer.writeByte` / Allocating appends | ~6-10 tokens a byte | 1-2 | an append-to-buffer record that keeps `end` |

  - libc / compiler_rt symbols (`memcpy`, `memset`, `memmove`, `memcmp`) are provided by the
    platform with custom ops.
  - Inlined std loops (`std.mem`, `std.json`, `std.unicode`, `token_estimate`) are ROM
    macro-uops at their PCs in fx's binary.
- **Speedup:** about 1.7x on the remaining glue tokens (0.6x in the model); 1.55x leave-one-out.
- **Cost:**
  - ~1-3k units per unit, SHA ~6k plus 2-4 heads: about 15-30k units in all.
  - The free room is in L1-L4 and L10-L11 (vx.md's table), but units that read registers need
    L5-L9, which are nearly full. So +1-2 layers.
  - 6-10 weeks.
- **Exactness:** medium per unit. Same differential fuzzing as #3.
- **Honesty:** honest. Runtime-library symbols are platform code with fixed semantics; ROM
  macro-uops specialize the CPU to the binary.
- **Depends on:** the P0 profile (which loops, at which PCs); #7.

## 7. Record formats and the address map
- **1-token LUI / AUIPC.**
  - Both values are ROM constants: LUI's upper immediate, and AUIPC's precomputed PC +
    (imm << 12).
  - Commit them like the JAL link, through register value channels from L4 dims at t-1, behind
    a payload-free control token. No V / F tokens.
  - Fuse li / la pairs (lui+addi, auipc+addi on the same rd) into one such record when the ROM
    fetch can return both words.
  - Expected -6-10% tokens (LUI / AUIPC are ~5-8% of instructions, mostly 2-3-token values).
  - Register rows are close to full (P0 counted 109 of 128 columns). The 30-column static
    channel fits once 16-bit payloads drop the F2 / sign-extension channels.
- **16-bit V / F payloads.**
  - No 3-token values; SB / SH become `V16 SS16`, 2 tokens instead of 4.
  - Vocab +100-200k rows (+0.8-1.5 GB; about 0.5 GB per 65k-row family).
  - -4-8% tokens.
- **Address map.**
  - RAM at 0, the stack and hot globals in the 1-token window (+-32K with 16-bit payloads).
  - Code: J / R carry a 16-bit word index (256 KB), and fx's ROM is 2.5-4 MB. Two fixes:
    - order hot functions first with a linker symbol-ordering file from the P0 trace, so more
      than 95% of taken jumps land in the first 256 KB with 1-token J;
    - a 2-token far J for the rest.

    The alternative, 20-bit J, needs a 2^20-row family: about 8 GB of embedding.
- **Combined:** 1.25-1.4x tokens per instruction.
- **Cost:** ~1-3k units, vocab growth.
- **Exactness:** low risk; extend the compliance and fuzz suites.
- **Honesty:** honest.

## 8. Build flags and fx config
- **Flags (honest; build config):**
  - ReleaseFast instead of the wasm build's ReleaseSmall: -10-25% dynamic instructions (fewer
    calls and prologue spills, which are SW / LW tokens). Code grows +30-80%, which costs
    checkpoint size, not speed, with #2's flattening.
  - `+zbb` (and `+zbs`): clz / ctz / cpop / rotates / andn / min / max / orc.b in Zig's
    bitsets, hashes, allocators, log2 and UTF-8. -2-6%; ~2-4k units, maybe +1 layer.
  - GP linker relaxation; the symbol ordering from #7.
  - Compare ReleaseFast against the shipped wasm build's behaviour on the P0 task set (UB can
    differ between modes).
- **Config (honest; fx's own settings):**
  - `minimal_kernel = true` (no cache-warming pings, no extra steering).
  - No journal, no `fx_session_commit`, no libfx checkpoint or snapshot requests. Persistence
    serializes and SHA-256s the whole history per prompt.
  - Model metadata with a context window large enough that auto-compaction never fires in a
    task. A compaction is a whole extra model call plus re-ingest.
  - fx's five file tools only (read / glob / grep / edit / write), so schemas stay small.
  - A `max_tool_result_bytes` of 16-32 KB instead of the wasm root's 64 KB. This changes agent
    behaviour within fx's own knobs.
  - `allow_acp_mcp = false`.
  - Low `FX_EFFORT`, which the server maps to Qwen's thinking budget.
  - No `FX_CONFIGURED_PROVIDERS` (its identity is SHA-256'd repeatedly), or keep it to one entry
    if strict mode needs it.
- **Combined:** about 1.15x and 1.1x; 1.23x leave-one-out.

## 9. Fast Qwen decode and a thinking budget
- **Why it is slow now:** the HF path pays for
  - the padded hidden (3968 against Qwen's 2560);
  - a 478k-row lm_head;
  - torch's linear-attention fallback and the causal_conv1d reference.

  That is close to its bandwidth floor at the padded size.
- **Steps:**
  - Run Qwen's 32 layers at their live width: 2,560 plus the 23 kept dims, with the RMSNorm
    divisor kept. The zero columns contribute nothing.
  - Score only Qwen's 248k rows at Qwen-owned positions; circuit tokens come from the code dims.
  - Add fla and causal-conv1d, a static cache and a CUDA graph.
  - About 7-8 GB per token in bf16 gives about 25-30 ms on an L4: 2-2.5x.
- **It also fits the model on the GPU.** fx-sized checkpoints (#12) pass a 24 GB L4 in their
  dense form, and the executor needs only the sparse chip weights.
- **Thinking budget:** 2-4x fewer generated tokens.
- **Speedup:** 1.17x leave-one-out at 400 tokens per step, about 1.5x with 1,500.
- **Exactness:** none for the circuit; text numerics move at bf16 rounding.
- **Honesty:** honest.
- **Optional:** int8 / fp8 Qwen weights are about another 2x, but they change the model.

## 10. Speculative trace decoding with a semantics-free drafter
- **What:**
  - A drafter predicts the next K chip tokens from the circuit's own past output only: replay
    of the trace after the same PC, with stride prediction for address-like payloads. It never
    executes or emulates fx.
  - The executor runs the K positions as one chunk. The chunked delta rule was bit-exact on the
    GPU in P0's `gpucheck`; the executor needs a chunk mode with a sequential state scan.
  - It keeps the longest prefix where its own predictions agree, then continues from its own
    token.
- **Speedup:** #3 and #6 already absorb the regular loops, so acceptance is perhaps 40-70%. An
    accepted run of 2-3 at a chunk cost of 1-1.5 steps gives 1.5-3x on chip time.
- **Cost:** 3-6 weeks.
- **Exactness:** low-medium (chunk numerics).
- **Honesty:** honest if the drafter is publishable and semantics-free. Every kept token is the
  circuit's own prediction. Disclose it.
- **Depends on:** #2.

## 11. Boot snapshot
- **What:** run fx from reset to its park in `fd_read` after `initialize` and `libfx/new` with
  a fixed workspace config, and inject RAM, registers and ROM as the snapshot. calc already
  works this way: `bridge.park`, then `core.snapshot`.
- **Speedup:** P0 has the whole pre-prompt part at about 2.0M wasm instructions: 11 min at
  3,200 tok/s, about 2 min after #2. About 1.08x on a 20-step task, 20-30% on 3-step tasks.
- **Cost:** days.
- **Exactness:** none.
- **Honesty:** honest.
  - The boot is input-independent, so its result is a constant of the program, like `.data`.
  - Show it once: free-run the boot in the weights and compare the state.
  - Never snapshot past user input (the prompt, the repo).
- **Depends on:** deterministic `clock_time_get` / `random_get` (#13), and the memory fitting
  (#12).

## 12. The memory architecture fx needs
**Needs (P0, wasm):**
- 2.35 MB of code; RV32 is probably 2.5-4 MB.
- 0.30 MB of data.
- A 1 MB stack reserve; the depth fx touches is unmeasured.
- Peak linear memory 3.2 MB in a 7-call toy session, so a heap high-water of about 1.8 MB.
- Plus the repository fx edits, and a capture buffer.

**Hard facts of the medium:**
- **Slots:** a value head's slot is the unit of writing. 32 value heads x 128 slots = **4,096
  independently writable slots per layer**.
- **Runtime writes store bits:** at most ~126 bit-cells per slot (vx.md).
- **No partial writes:** a write's value must come from token payloads or dims below at t-1, so
  a slot can't be partly rewritten in place. The old contents would have to pass through the
  stream: that is why SB / SH cost 4 tokens.

**Capacities:**

| Region | Layout | Capacity per layer | Store cost | fx layers |
|---|---|---|---|---|
| ROM / DROM today | pre-decoded bytes, 2-hot row pairs, each row 32 of 128 columns | 128 KB | snapshot only | 20-30 (code) |
| Dense ROM / DROM | 2-hot with an all-zero partner row Z (carrying the ballast); each row uses ~124 byte columns, 31 words | ~500 KB | snapshot only | 5-8 (code) + 1 (rodata) |
| Word-slot RAM (today's) | 1 word per slot | 16 KB | SW 1, SB/SH 2-4 | 2 MB -> 128 |
| 2-word slots | partner word through the stream | 32 KB | SW 3 | 64 |
| 3-word slots | two partner words through the stream; a staging head (one value head per word, L6) makes sequential stores into a slot 1 token after the first | 48 KB | SW 5 first, 1 after | 43 |

- **Layer size:** each layer at hidden 4608 is about 0.4 GB of checkpoint, from rv-store's
  22.3 GB for 44 layers.
- **The heap has to shrink.**
  - #5 removes request bodies and SSE parse trees.
  - Use a compact platform `malloc`: size classes with reuse, arenas returned.
  - Cap tool results.
  - Map only the touched stack depth.
  - Then provision word slots for the measured peak, and put large write-once blocks (history
    strings, file buffers) in a 2- or 3-word bulk region chosen by allocation size.
  - Target <= 0.5-1 MB writable: about 25-60 RAM layers. With the dense ROM, about 35-75
    memory layers on about 15 core layers: 20-35 GB.
- **The repository** goes in an rv-store-like hashed object store: content in packed rows,
  about 64 KB per layer at 126 bits per slot.
- **Effect on speed:**
  - With #2's flattening, independent memory sublayers cost about one phase in all.
  - Without it, about 15 us each (Executor2), and tens of them halve the speed.
  - Bulk regions add tokens only on stores: +5-10% at a 2-word bulk.
- **Rejected:**
  - A full-attention KV cache as RAM: it grows with the trace (millions of tokens, about 4 KB
    per token per layer), and softmax exactness is hard.
  - The embedding table as ROM through fetch tokens: +1 token per fetch or per block, and
    20-bit block ids mean multi-GB embeddings.
- **Open question:** byte-valued runtime cells would not add slots, so they don't lift the
  word-slot ceiling. Only coarser slots or fewer bytes do.

## 13. The OS layer as circuits
- **The interface:**
  - The platform root provides Zig's `std.Io` (the vtable fx reaches through
    `io_mod.setIo`), plus the WASI-like and `fx_*` externs it needs, on in-weights stores.
  - Use a non-WASI target OS tag. Under `is_wasm` fx compiles out its own tools
    (`tool_call_presentation.activeToolSet` returns host tools or none), and host-run tools
    would be the host doing fx's work.
  - With a non-WASI tag, fx's Zig `read_file` / `edit_file` / `grep_files` / `glob_files` /
    `write_file` run on the CPU.
- **Files:** a hashed object store in the weights. `fd_read` / `fd_pread` / `fd_write` copy
  with #6's streaming copy (2 tokens a word), not byte loops.
- **stdout (ACP JSON-RPC):** a silent sink, one ecall per `fd_write` (P0: 202 writes), never
  echoed to Qwen. The host may display it by reading RAM; that is observation, not computation.
- **Time, randomness, waits:**
  - `clock_time_get` is a deterministic virtual clock (P0: 429 calls); `random_get` is a seeded
    PRNG. The reference and the circuit must agree, and today's CSR stubs read 0.
  - `poll_oneoff` and steering waits park instead of spinning.
- **Qwen to the CPU:** the 80-token capture window is far too small for `write_file` /
  `edit_file`. Use a two-level counter (the thermometer caps at m < 88 in bf16) and a
  4-8k-token buffer: 1-2 layers.
- **The CPU to Qwen:** any-token print (rv-store) and a PUTS record of one position per token.
- **Speedup:** ~1.1-1.3x, and enabling.
- **Honesty:** honest. The OS layer is allowed; the flag is only for running tools or the
  filesystem's logic on the host.

## 14. Profile-chosen fused pairs
- **Candidates:**
  - `auipc ra; jalr ra` far calls. The AUIPC value is dead once JALR writes the link, so the
    pair is one J record (-2 tokens).
  - `slt` / `sltu` followed by a branch on it, when the slt result is dead: one J / NT.
  - li / la pairs (#7).
- **Limit:** every live register result computed at or above L5 still needs its own value
  tokens, so fusion only saves dead values and control tokens.
- **Speedup:** -3-8% tokens.
- **Cost:** 1-3k units, maybe +1 layer for a compare after an ALU op.
- **Honesty:** honest (static liveness on the fixed binary; commit every live value).

## 15. Pruning
- **The circuit:** fx needs none of the following, which free L5-L9 room for #3 / #6:
  - the digit file (BMAC / BMSB / BLD / BSCAN / BGET / BPRD: about 9k units and 16 heads in
    L5-L6);
  - calc's byte-row text path, if #13 replaces it;
  - the NPU / VX units.
- **DIV stays:** Zig emits div / rem for runtime divisors. Its compare chain sets the depth at
  L10.
- **Code:** only comptime choices prune code. Zig compiles what the root references, so a lean
  root (ACP server, file tools, the provider) shrinks the ROM. Runtime flags like
  `allow_acp_mcp` do not.
- **Speedup:** ~1.0-1.1x.
- **Honesty:** honest.

## 16. DIV and DIV64
- **What:** Zig's u64 / i64 division on RV32 calls `__udivdi3` / `__umoddi3` / `__divdi3`
  (about 50-300 instructions each), mostly from formatting u64 timestamps and counts.
- **The fix:** extend the QS / QD micro-code to 16 hex digits behind a custom op that only the
  platform's compiler_rt uses.
- **Speedup:** <= 1.05x unless P0's RV32 profile shows u64 formatting.
- **Cost:** 8-15k units. A radix-16 software loop over the existing DIVU is cheaper to try
  first.
- **Honesty:** honest (a runtime-library symbol).

## 17. An RV64 core
- **Gain:**
  - Native u64 arithmetic: timestamps, hashing, wyhash's products.
  - 8-byte copies.
  - 64-bit SD slots, so 32 KB of RAM per layer.
  - About -10-25% instructions.
- **Against:** pointers double the pointer-heavy heap.
- **Cost:** a new core, months. Defer until the RV32 profile is known.
- **Honesty:** honest.

## 18. R refresh
The refresh comes only after 63 commits without a taken jump: under 1% of tokens. The bf16
thermometer cap (m < 88) bounds it anyway. Leave it.

## 19. Whole-basic-block records
- **Why it doesn't pay:**
  - Each live result computed after L5 still needs its own tokens, so a block record saves only
    dead intermediates and NT / J tokens.
  - Chaining 3-5 dependent ALU ops costs 2 MLP stages each: +4-8 layers.
- **Net:** <= 1.1x. Not recommended; #14 gets most of it.

## 20. Speculative decoding with the host's reference CPU drafting
- **What:** the host's reference machine runs fx and drafts the exact trace, and the circuit
  verifies it in prefill-sized chunks.
- **Speedup:** about 100% acceptance, so 10-50x.
- **Honesty: this violates the point.** The host computes every result and the weights only
  check them: it is running fx on the host and replaying its trace.
- **Allowed only** as a labelled verification mode (today's lockstep is the same kind of thing,
  which is why #2 turns it off in production).

## 21. Batching sessions
`rvbatch` / Executor2 batch rows give 5-8x throughput at B = 8-32 for independent fx tasks, and
no latency gain for one task. Honest.

## The middle ground between a CPU and HLS
- **The two ends:**
  - The CPU is the fully shared extreme: about 1-1.5 tokens per instruction, every value
    through the stream.
  - C-to-gates for about a million instructions of fx code is out of reach: 9,216 units per
    sublayer, and 8-9 MLP stages per token before the exit.
- **The rules that place the sweet spot:**
  - Depth per token is fixed: a 32-bit add is 2 stages, a multiply 5.
  - Loop-carried state must fit a token's payload (16 bits) or live in a state head written
    from payloads.
  - Writable memory moves data through the stream.
  - So a unit pays off only for a loop whose body fits about 6 stages, carries at most 16-32
    bits per iteration besides its induction variables, and runs often.
- **fx's profile has such a head:** memory, string, scan, hash and format loops, plus fx's own
  `maskSecrets`. The top 20 functions are 75% of P0.
- **The sweet spot is the CPU plus about 10-20 streaming units (#3, #6) and a few fused pairs
  (#14):** about 20-35k units, +1-2 layers. That covers an estimated 50-70% of the dynamic
  instructions left after #1 and #5.
- **Beyond about 20 units the returns collapse.** What is left is flat glue (dispatch,
  allocator calls, hash-map bookkeeping), and each piece runs too rarely to pay for its
  circuit.
- **An "LLVM IR to streaming unit" compiler** (Zig emits LLVM IR with `-femit-llvm-ir`) is
  worth building as the generator for these units. Its pattern: a single-block loop body, at
  most one load stream and one store stream, carried state of at most 16 bits besides the
  induction variables. Everything else stays on the CPU.

## Recommended stack and turn time
**Order of work:**
1. One SSE event per reply (#1).
2. fx config and build flags (#8).
3. The boot snapshot (#11).
4. Batched ingest with host BPE (#4).
5. Executor v4 with flattening (#2), together with the memory plan (#12) and the OS layer
   (#13).
6. The Qwenputer provider (#5).
7. The secret-scan and compare units (#3).
8. The glue units (#6).
9. Record formats and the address map (#7, #14).
10. Fast Qwen decode (#9).
11. Optionally, semantics-free speculation (#10).

**Every item in the stack is honest.** Two need a disclosure:
- the provider (#5): fx's agent runtime is exact, the model wire format is the platform's;
- host-side BPE (#4a), if used instead of any-token print.

Excluded: host-reference speculation (#20) and host-run tools.

**Estimate on the reference task (20 steps, 400 Qwen tokens a step):**

| | Per agent step | Per task |
|---|---|---|
| fx as is, today's machine | 4.3 h | 86 h (+ 11 min boot) |
| + #1 | 97 min | 32 h |
| + #2 | 22 min | 7.3 h |
| + #3 | 15 min | 4.9 h |
| + #4 | 12 min | 4.0 h |
| + #5 | 3.6 min | 1.2 h |
| + #6 | 2.4 min | 48 min |
| + #7, #8, #9 | **83 s** | **28 min** (boot 0 with #11) |
| + #10 | 48 s | 16 min |

- **Combined: about 190x** (100-300x across the assumptions), about 330x with #10.
- **Sensitivity of the final step:**
  - RV32 / wasm factor 0.6-1.0: 68-105 s.
  - Executor at 8k-25k tok/s: 57-152 s.
  - 1,500 thinking tokens per step: 116 s.
  - Strict wire (no #5): about 2-3 min.
- **The final step's 83 s:** about 70 s of chip time (1.0M RV32 instructions at 1.1 tokens each,
  at 16k tok/s) and about 12 s of Qwen generation.
- **What is left is fx's own glue:** about 1M instructions per step of tool admission,
  presentation, ACP updates and history bookkeeping. Below that, the only levers are executor
  speed, #10, and a longer tail of units.

## What P0 needs to measure next
- The RV32 build:
  - code size and section sizes;
  - tokens per instruction on fx's own trace;
  - instructions per wasm instruction;
  - the stack depth touched;
  - the heap high-water with the platform allocator.
- **Native tools:** P0 used host tools, so native tool costs are missing, especially the edit
  path's SHA-256 over file contents.
- **Call sites:** of `mem.eqlBytes` (is it all `maskSecrets`?) and of SHA-256 (how much is
  `binding_identity` under `FX_CONFIGURED_PROVIDERS`?).
- **With base Qwen on the p0c tasks:** output tokens and tool-argument bytes per step, and how
  often fx compacts or truncates history.
