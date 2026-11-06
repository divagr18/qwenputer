# fx on Qwenputer, P0: code size, instruction counts, profile, memory

Status 2026-11-07. Vercel fx at commit `757bdb9` (version string 0.0.16), Zig 0.17.0, CPU only.
**M** = measured, **E** = estimated. Raw numbers: `p0-results.json` next to this file. Scripts:
`experimental/riscv/fx/`.

**How it was measured**
- fx's source does not compile for riscv32 unmodified (see Blockers).
- So the RV32 program is fx's own headless WebAssembly build, `fx-core.wasm`. fx's
  `build.zig` builds it with `-Dwasm-surface=core` (root `src/wasm_core_main.zig`, ReleaseSmall).
  It is translated by wasm2c (wabt 1.0.42) and compiled by `zig cc` for
  `riscv32 +m +zba +unaligned_scalar_mem`, `-Os`. The bare-metal runtime is about 200 lines
  (`rv/rt.c`); every wasm import is an `ecall` into the host.
- None of fx's files changed: the src tree was hash-checked against `757bdb9` after every build.
- The same Python host drives both runs:
  - fx-core.wasm under wasmtime 50 with fuel metering;
  - the RV32 ELF in a small RV32IM+Zba+Zalrsc interpreter (`rv/rv32emu.c`) that counts every
    retired instruction by mnemonic and by PC.
- The host mirrors libfx's `wasm-runtime.js` and `engine.js` (WASI with no preopens; tools are
  libfx host tools). The ACP sequence is `initialize`, `libfx/new` and two `session/prompt`s.
- The mock model is an OpenAI Chat Completions server (`FX_CONFIGURED_PROVIDERS`, as libfx's
  `customProvider` sets it). It streams real `data:` SSE, one chunk per about 4 characters
  (about 1 token).
- **Session:** 7 model calls.
  - Turn 1 has 6: `read_file`, `grep_files`, `edit_file`, `write_file`, `glob_files`, then a
    142-token answer.
  - Turn 2 is a 36-token answer.
- Both runs leave the tiny repo (10 files) in the same final state.
- Fuel check: the per-function counters of the instrumented module add up to the uninstrumented
  fuel within 27 units of 49.68M (M). This holds once wasmtime's extra charges are counted: 1
  per call and 1 per byte of `memory.copy`/`fill`.

## Headline

| | value | |
|---|---|---|
| RV32 code (.text), ReleaseSmall, Zba, unaligned | **3.98 MB** (fx code 3.96 MB); .rodata 317 KB; .data 0; .bss 131 KB | M |
| Boot to the first ACP read | **0.39M RV32 instr** (0.23M of it copies the 292 KB data segments; a snapshot removes it) | M |
| `initialize` + `libfx/new` | 1.09M + 0.26M | M |
| Turn 1 (5 tool calls + answer) | **32.5M RV32 instr**, i.e. 33-50M trace tokens, about **3.9 h** at 2.3k instr/s | M / E |
| Turn 2 (36-token answer, no tools) | 3.74M RV32 instr (about 27 min) | M / E |
| Streaming cost per model output token | **about 63k RV32 instr** (27 s of machine time) | M |
| Peak linear memory (grown) / bytes actually written | 3.2 MB / **626 KB** (stack 1 MB reserved, 375 KB used) | M |
| RV32 instr per wasm instruction (fuel) | **0.76** (whole run, -Os, unaligned); 1.21 with aligned-only loads | M |

## 1. Code size

| build | .text | .rodata | .data | .bss | wasm code / data | |
|---|---|---|---|---|---|---|
| **RV32 -Os, Zba, unaligned scalar mem** (ReleaseSmall wasm) | **3,982,568** | 316,848 | 0 | 131,160 | 2,345,842 / 299,316 | M |
| RV32 -Os, Zba, aligned only (every wasm load/store becomes byte ops) | 6,966,052 | 316,848 | 0 | 131,160 | same | M |
| RV32 from the ReleaseFast wasm | 7,837,116 | 324,032 | 0 | 131,160 | 4,315,115 / 294,784 | M |
| RV32, stripped root (one provider: configured Chat Completions; no Codex / Grok / Gateway / catalogs) | 3,693,532 | 303,108 | 0 | 131,160 | 2,177,101 / 286,220 | M |
| published libfx 0.0.16 `fx-core.wasm` (older commit, for reference) | - | - | - | - | 2,037,033 / 216,770 | M |
| native RV32 compile of fx's source | blocked | | | | | |

- .bss is the runtime's static funcref table (8192 entries; fx uses about 700).
- The wasm data segments end up in .rodata (copied into linear memory at boot).
- At ROM's pre-decoded 8 bytes per instruction, 0.996M static instructions need about **8 MB of
  ROM** (E), against today's 256 KB.
- The stripped root (`fx/zbuild/rv_core_main.zig`) saves only 7%. The provider code is small;
  the bulk is the ACP server, the agent runtime, JSON and std.

## 2. Instructions per phase

The RV32 column is the main result: ReleaseSmall, -Os, Zba, unaligned. The ReleaseFast build is
shown for comparison.

| phase | wasm fuel (M) | **RV32 instr (M)** | RV32 from ReleaseFast wasm (M) |
|---|---|---|---|
| instantiate (data segments) | 303,105 | 231,311 | (in boot) |
| boot: `_start` to the first stdin read | 216,606 | 161,708 | 354,435 incl. instantiate |
| `initialize` (5 host tool descriptors) | 1,166,008 | 1,090,819 | 754,254 |
| `libfx/new` | 319,454 | 255,420 | 190,739 |
| **turn 1** | 43,519,508 | **32,505,371** | 20,524,792 |
| turn 2 | 4,411,444 | 3,736,134 | 2,555,766 |
| whole session | 49,983,821 | 38,024,738 | 24,414,461 |

### Per agent step (turn 1), RV32 instructions (M)

- The tool itself runs in the host: in fx-core, `read_file` and the rest are libfx host tools
  (`fx_host_tool_call`), so their own guest cost is 0.
- The guest pays for the work around each call: taking in the previous result, building the
  request, the stream, and dispatching the call.

| step (output tokens) | tool | take in last result + build request | stream consume | after stream to tool call | **step total** | wasm fuel |
|---|---|---|---|---|---|---|
| 1 (7) | read_file | 825,427 | 651,015 | 1,301,654 | **2.78M** | 3.53M |
| 2 (9) | grep_files | 528,069 | 778,265 | 1,387,126 | **2.69M** | 3.45M |
| 3 (63) | edit_file | 566,570 | 4,223,833 | 2,829,444 | **7.62M** | 11.45M |
| 4 (52) | write_file | 622,478 | 3,523,795 | 2,398,587 | **6.54M** | 9.65M |
| 5 (6) | glob_files | 670,376 | 587,881 | 1,308,709 | **2.57M** | 3.31M |
| 6 (142) | final answer | 731,489 | 8,946,605 | 624,048 (to turn end) | **10.30M** | 12.13M |
| 7, turn 2 (36) | answer | 1,154,018 | 2,362,578 | 219,538 | 3.74M | 4.41M |

**What it costs (M unless marked)**
- Stream consume is about 63-67k RV32 instructions per output token. Each SSE event costs
  about 13% wyhash, 8% eqlBytes, 8% findScalar and 6% JSON scan. It is also re-serialized as an
  ACP `session/update` (198 of them in the session).
- Fixed cost per tool call: about 1.3M RV32 for dispatch.
  - 57% of that is SHA-256 over about 8 KB per call: 126 blocks, from the configured provider's
    binding identity (`configured_provider.hash_part`, 60 calls).
  - The rest is mainly `mem.eqlBytes` and `maskSecrets`. `maskSecrets` grows with the size of
    the tool arguments (edit/write: +1-1.4M).
- The request body grows from 1.9 to 5.2 KB over the session (empty system prompt: fx-core's
  `prompt_policy` is `""`). The request build grows with it, by about 150 wasm instr per body
  byte (E, from the steps).

## 3. WASM to RV32 ratio

| measure | ratio | |
|---|---|---|
| whole session, RV32 instr / wasm fuel (-Os, Zba, unaligned) | **0.761** | M |
| same, aligned-only loads | 1.208 | M |
| ReleaseFast wasm to RV32 | 0.864 | M |
| per function (top 30) | 0.27 (`json_depth.exceeds`) to 2.41 (`__multi3`, 128-bit multiply); `Io.Writer.writeByte` 1.85; the JSON scanner 0.9-1.2 | M |
| fuel = wasm ops 47.81M + calls 0.45M + bulk-memory bytes 1.42M | | M |
| a native RV32 compile of fx (if it built) vs the wasm2c path | about 0.7-0.9x (wasm2c keeps wasm's shadow stack and 64-bit address arithmetic) | E, not measured |

## 4. Hot functions

These are shares of wasm fuel, from per-function counters in the instrumented
`fx-core.wasm` (with names). The RV32 instruction counts come from the PC profile of the RV32
build; "inlined" means clang folded the function into its callers.

| category | share of wasm fuel | share of RV32 instr |
|---|---|---|
| memcpy / memset / memcmp (`mem.eqlBytes`, `findScalarPos`, memmove) | 23.2% | 20.4% |
| hashing (SHA-256, wyhash) | 20.5% (23.9% with wyhash's 128-bit multiply) | 16.0% |
| JSON parse (Scanner, dynamic Value, string hash maps) | 18.5% | 24.7% |
| allocator (arena, dlmalloc realloc, ArrayList growth) | 9.6% | 11.7% |
| formatting / `Io.Writer` / compiler_rt multiply | 7.2% | 10.7% |
| SSE / model stream (`sse.Reader`, `json_depth`, chat-completions reducer) | 6.5% | 4.4% |
| token estimate / `maskSecrets` / compactor lint | 4.6% | 3.9% |
| JSON write / escape | 3.2% | 2.6% |
| UTF-8 / text scan | 2.5% | 1.0% |
| agent loop | 1.8% | 1.3% |
| ACP / session / journal | 1.1% | 0.6% |
| other | 1.5% | 2.6% |

| # | function | category | fuel share | calls | RV32 instr | RV/wasm |
|---|---|---|---|---|---|---|
| 1 | `mem.eqlBytes` | memcmp | 19.81% | 14,579 | 3,653,724 | 0.37 |
| 2 | `crypto.sha2.Sha2x32` (SHA-256 rounds) | hashing | 11.34% | 707 | 3,399,963 | 0.60 |
| 3 | `hash.wyhash.Wyhash.hash` | hashing | 7.85% | 9,651 | 2,169,054 | 0.56 |
| 4 | `json.Scanner.next` | JSON parse | 4.96% | 9,740 | 2,203,053 | 0.89 |
| 5 | `gateway.json_depth.exceeds` | SSE | 2.93% | 359 | 396,817 | 0.27 |
| 6 | `mem.findScalarPos` | memchr | 2.64% | 1,484 | inlined | |
| 7 | `token_estimate.StreamingEstimator.consume` | token estimate | 2.52% | 194 | 844,310 | 0.67 |
| 8 | `heap.ArenaAllocator.alloc` | allocator | 2.25% | 9,230 | 1,039,445 | 0.93 |
| 9 | `compiler_rt.mulXi3.muldXi` (wyhash's 128-bit multiply) | multiply | 2.11% | 20,127 | inlined | |
| 10 | `json.Stringify.encodeJsonString` | JSON write | 1.61% | 1,548 | 323,446 | 0.41 |
| 11 | `gateway.sse.Reader.next` | SSE | 1.50% | 341 | 341,153 | 0.46 |
| 12 | `Io.Writer.writeByte` | writer | 1.46% | 21,971 | 1,342,906 | 1.85 |
| 13 | `text_utils.maskSecrets` | secrets | 1.46% | 34 | 494,263 | 0.68 |
| 14 | `c.malloc.realloc` | allocator | 1.45% | 445 | 398,374 | 0.55 |
| 15 | `json.Scanner.nextAllocMax` | JSON parse | 1.38% | 8,746 | 819,161 | 1.19 |
| 16 | `json.Scanner.peekNextTokenType` | JSON parse | 1.37% | 8,746 | 841,163 | 1.23 |
| 17 | `__multi3` | multiply | 1.30% | 20,127 | 1,549,779 | 2.41 |
| 18 | `unicode.utf8ValidateSlice` | UTF-8 | 1.29% | 1,746 | 242,436 | 0.38 |
| 19 | `StringArrayHashMap(json.Value).get` | JSON parse | 1.22% | 6,677 | 683,012 | 1.13 |
| 20 | `StringArrayHashMap(json.Value).getIndexAdapted` | JSON parse | 1.18% | 6,679 | 487,560 | 0.83 |
| 21 | `json.dynamic.handleCompleteValue` | JSON parse | 1.02% | 2,370 | 502,845 | 0.99 |
| 22 | `acp.jsonrpc.writeJsonStr` | JSON write | 1.02% | 840 | 307,665 | 0.61 |
| 23 | `json.Scanner.skipWhitespace` | JSON parse | 1.01% | 22,743 | 475,794 | 0.95 |
| 24 | `ArrayList(u8).ensureTotalCapacity` | allocator | 0.92% | 5,389 | 472,813 | 1.04 |
| 25 | `heap.ArenaAllocator.resize` | allocator | 0.91% | 10,064 | 195,776 | 0.43 |
| 26 | `compactor.lint.isWordByte` | text scan | 0.72% | 14,988 | inlined | |
| 27 | `json.static.parseFromTokenSourceLeaky` | JSON parse | 0.68% | 464 | 356,768 | 1.06 |
| 28 | `gateway.host_chat_completions.stream` | SSE | 0.62% | 7 | 284,865 | 0.92 |
| 29 | `StringArrayHashMap(json.Value).getOrPutAssumeCapacityAdapted` | JSON parse | 0.61% | 2,885 | 265,930 | 0.88 |
| 30 | `Io.Writer.write` | writer | 0.60% | 5,985 | 355,288 | 1.20 |

The top 60 with all counts are in `p0-results.json` (`hot_functions_top60`). Per-phase deltas
can be rebuilt from the instrumented run's snapshots (`host.py --instrument`).

## 5. Opcode census (RV32 -Os, Zba, unaligned)

| | RV32I | M | Zba | A / Zalrsc | Zicsr | F / D | C | |
|---|---|---|---|---|---|---|---|---|
| static (995,642 instr) | 989,413 | 510 (MUL, MULHU, DIVU only) | 5,719 | 0 | 0 | 0 (soft-float) | 0 | M |
| dynamic (38.02M) | 99.4% | 344k (MUL 219k, MULHU 123k, DIVU 1k) | 187k | 0 | 0 | 0 | 0 | M |

**Dynamic mix (M)**
- Classes:
  - loads 22.7% (LW 7.04M, LBU 1.46M);
  - stores 14.1% (SW 4.83M, SB 0.47M);
  - branches 9.6%;
  - JAL / JALR 2.9%;
  - M 0.9%, Zba 0.5%;
  - other ALU 49% (ADDI alone 22%).
- Misaligned scalar accesses: 128,221.
  - Where: `mem.eqlBytes` 68.6k, wyhash 26.0k, string hash maps 23.2k.
  - That is about 0.9% of the 14.0M loads and stores.
  - Built aligned-only, the same run takes 60.4M instructions: 1.59x more, with 1.75x more code.
- No atomics remain, because the wasm build is single-threaded.
- On the wasm side, 121 distinct opcodes are used statically: MVP, sign-ext, bulk-memory
  `memory.copy`/`fill`, and nontrapping float-to-int. There is no SIMD.

## 6. Memory access (RV32 build, wasm linear memory only)

**How it was measured**
- Measured on the RV32 run, not wasmtime: wasm2c maps wasm addresses 1:1 onto linear memory,
  so the addresses and bytes are the wasm program's (clang may drop a few redundant accesses).
- wasm2c's C stack is excluded.
- Host DMA (stdin, tool results, SSE chunks: 47.5 KB) marks words as written but is not counted
  as stores.

**Phases**
- "Boot" is the data-segment copy plus `_start` up to the first stdin read.
- Model call k runs from its `stream_open` to the next one: the stream, the tool, and the next
  request's build.
- Words are 4 bytes; lines are 128 bytes. RO reads are words not written since boot; RW reads
  are words written since boot.

| phase | RV32 instr | bytes loaded | bytes stored | distinct words written | 128-B lines written | new words (cumulative KB) | distinct words read: RO / RW |
|---|---|---|---|---|---|---|---|
| instantiate (data segments) | 231,311 | 0 | 299,296 | 74,824 (292 KB) | 2,339 | 74,824 (292) | 0 / 0 |
| boot (`_start`) | 161,708 | 97,298 | 109,628 | 23,638 (92 KB) | 766 | 23,631 (385) | 18,598 / 0 |
| setup (initialize, new, prompt 1, request 1) | 2,184,963 | 605,389 | 441,288 | 20,856 (81 KB) | 823 | 19,484 (461) | 1,803 / 16,622 |
| model call 1 (read_file) | 2,480,738 | 706,183 | 497,256 | 17,886 (70 KB) | 723 | 12,908 (511) | 1,686 / 15,573 |
| model call 2 (grep_files) | 2,731,961 | 786,251 | 557,279 | 19,355 (76 KB) | 791 | 3,394 (524) | 1,648 / 17,044 |
| model call 3 (edit_file) | 7,675,755 | 2,252,409 | 1,591,072 | 23,206 (91 KB) | 922 | 5,501 (546) | 1,661 / 20,705 |
| model call 4 (write_file) | 6,592,758 | 1,934,519 | 1,368,213 | 22,828 (89 KB) | 913 | 2,464 (555) | 1,677 / 20,440 |
| model call 5 (glob_files) | 2,628,079 | 747,306 | 526,980 | 20,770 (81 KB) | 838 | 2,490 (565) | 1,645 / 18,381 |
| model call 6 (answer, 142 tok) | 10,726,501 | 3,050,185 | 2,227,574 | 31,977 (125 KB) | 1,264 | 13,890 (619) | 1,507 / 31,049 |
| model call 7 (turn 2) | 2,610,964 | 745,739 | 550,315 | 9,807 (38 KB) | 551 | 1,596 (626) | 1,150 / 10,285 |

All figures below are measured.

**Write working set**
- Per step: 70-125 KB of distinct words, 551-1,264 lines.
- Whole session: 161,281 distinct words (630 KB), spread over 5,489 lines (686 KB of lines).
- Boot-only image: 98,534 words (385 KB), of which 292 KB are the data segments.
- Written after boot: 64,771 words (253 KB). Only 2,024 words are written both at boot and
  after.
- By region:
  - stack: 46.5k words (181 KB);
  - data segments: 74.8k words;
  - heap: 39.9k words (156 KB), spread across 1.6 MB of address space (per-64 KB page counts are
    in the JSON).

**Reads**
- Per model call, 90-95% of the distinct words read were written after boot.
- Only about 1.5-1.8k words (6-7 KB) per call come from never-rewritten boot state, so a
  read-only post-boot image would serve very little.

**Store runs**
- These are consecutive stores to adjacent ascending addresses, binned by run length in bytes:

  | run length | runs |
  |---|---|
  | 1 | 88.8k |
  | 2-3 | 71.7k |
  | 4-7 | 421k |
  | 8-15 | 367k |
  | 16-31 | 71.9k |
  | 32-63 | 7.7k |
  | 64-127 | 4.0k |
  | 128-255 | 1.7k |
  | 256 B and up | 695 |

- Most stores are a single word or a pair of words: i64 values become 2 x SW.

**Whole-line writes**
- Of 793k write episodes (a line written, then read), only 4,322 (0.5%) fully overwrote the line
  before it was next read.
- 1,519 of 5,489 written lines (28%) had at least one such episode. These are mostly bulk copies
  and memsets.
- Read-modify-write is the normal case.

**High-water marks**
- Stack: 375,488 B of the 1 MB reserved, by the deepest store.
- Heap: stores reach the top of the grown memory (3.21 MB; heap base 0x149120), but only 156 KB
  of heap words are ever written.

**Rewrites per word**
- Over the session:

  | times written | words |
  |---|---|
  | 1 | 109k |
  | 2-3 | 9.4k |
  | 4-7 | 23.0k |
  | 8-15 | 8.8k |
  | 16-63 | 7.3k |
  | 64-255 | 2.5k |
  | 256-1k | 0.8k |
  | 1k-4k | 0.17k |
  | 4k and up | 0.1k |

- The hottest 3 words were written more than 32k times each.
- Live at session end: 89,882 of the 161,281 written words are non-zero (351 KB).

## Blockers

1. **fx's source does not compile for riscv32** (Zig 0.17.0, `riscv32-linux-musl`, ReleaseSmall,
   single-threaded, fx's own wasm-core flags).
   - **64-bit atomics:** `std.atomic.Value(u64)` loads, stores and RMWs in
     `core/mcp_host/runtime.zig:598`, `core/subagent/agent_adapter.zig:997`,
     `gateway/client.zig:1145` and others. Zig refuses atomics wider than 32 bits on rv32, even
     single-threaded.
   - **Thread.spawn** in `acp/server.zig:2420` (`startPrompt`) and 7 more sites. fx picks its
     WASM code paths with `os.tag == .wasi`, so any riscv32 OS target compiles the native,
     threaded paths, and a single-threaded build rejects `spawn`.
   - **musl for riscv32 needs Zalrsc** (`lr.w` / `sc.w` in `atomic_arch.h`); `+zalrsc` fixes
     that part.
   - **Workaround used:** compile fx's own `fx-core.wasm`, translate it with wasm2c, and compile
     the C for RV32. fx's code is the unmodified wasm module; only the translation and the
     runtime are ours.
   - wasm2c costs: wasm's shadow stack lives in linear memory, and addresses are computed in
     64 bits.
2. **Alignment.** wasm2c loads through `memcpy`.
   - Without `+unaligned_scalar_mem`, every access becomes byte ops: 60.4M instructions and
     7.0 MB of code.
   - With it, 128k word accesses are misaligned.
   - Qwenputer needs misaligned LW/SW in the circuit, a trap-and-emulate path, or wasm2c patched
     to honor each memarg's alignment hint.
3. **clang -O2 crashes** (LLVM fatal error) on the 47 MB wasm2c C file, so the RV32 numbers are
   -Os. `wasm2c --num-outputs` could split it.
4. **Tools are host tools in fx-core.**
   - The ACP wasm build only advertises the client's tools (`tool_call_presentation.zig`
     `activeToolSet`: host tools, or none on WASI).
   - libfx's host also answers `path_open` with ENOSYS. So no WASI preopen is used, and fx's own
     `src/tools/filesystem` code never runs.
   - On Qwenputer, read / grep / edit / write / glob would be extra RV code behind
     `fx_host_tool_call`. That code is not measured here (its guest cost is 0 above).
5. **Wall clock.** fx keeps deadlines in real time, and at 2.3k instr/s a turn takes hours. The
   host must give a virtual clock (here: +1 ms per `clock_time_get`).
6. **Not measured:** a native RV32 compile (blocked), -O2 / -O3, a non-empty system prompt, and
   longer histories (the request build grows with the body).

## Reproduce

These are the commands, in order; the paths are the session scratchpad's.

1. Build the wasm:
   `zig build -Dfx=FX -Darch=wasm -Doptimize=small -Dstrip=false` in `fx/zbuild` (or
   `-Doptimize=fast`, or `-Droot=rv_core_main.zig` after copying it into `FX/src`).
2. Run it under wasmtime: `python -I fx/host.py fx-core.wasm out.json [--instrument]`.
3. Build the RV32 image:
   `ZIG=... WABT=... sh fx/rv/build_rv.sh fx-core.wasm OUT -Os generic_rv32+m+zba+unaligned_scalar_mem`.
   `zig cc -target x86_64-windows-gnu -O2 -shared fx/rv/rv32emu.c -o rv32emu.dll` builds the
   interpreter.
4. Run it: `python -I fx/rv/rvhost.py rv32emu.dll fx-*.elf imports.json out.json`.
5. Collect the results: `python -I fx/analyze.py RUNS builds.json p0-results.json`.

Binaries are not committed. They are under the P0 scratchpad
`C:/Users/kesha/AppData/Local/Temp/claude/8450bc6024565175/d1310d09-a8d6-4b26-affc-4f555296c2cb/scratchpad/`:
- `build-stock`, `build-wasm-names`, `build-wasm-fast`, `build-wasm-stripped`: the wasm builds;
- `rvb`, `rvbf`, `rvbs`: the RV32 ELFs and wasm2c C;
- `runs/`: raw run JSON;
- `dl-zig`, `dl-wabt`: the toolchains.
