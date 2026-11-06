# fx on Qwenputer: speed iteration

Status 2026-11-07. CPU only. Follows `p0-report.md` (the baseline) and `speedups.md` (the
estimates this measures). Raw numbers: `speed-iter.json`. Scripts: `experimental/riscv/fx/`.
**M** = measured, **E** = estimated.

**What is measured**
- The same program and session as P0:
  - Vercel fx at `757bdb9`, as its own WASM build, through wasm2c and clang for RV32;
  - run in `rv/rv32emu.c`, which counts every retired instruction;
  - P0's scripted session: turn 1 has 5 tool calls (read / grep / edit / write / glob) and a
    142-token answer; turn 2 is a 36-token answer.
- Every run leaves the repository in P0's final state (`same_repo_as_p0` in the JSON).
- fx's sources were hash-checked after the last build: the 717 files under `src/`, `build.zig`
  and `build.zig.zon` equal commit `757bdb9` (`fx/check_src.sh`).
- Wall time is instructions divided by 2.3k instr/s (today's machine) and 15k instr/s (a future
  executor), as the brief asks.

## Result

| | turn 1 | turn 2 | turn 1 wall, 2.3k/s / 15k/s |
|---|---|---|---|
| P0 | 32.51M | 3.74M | 3.9 h / 36 min |
| **Best on today's ISA** (RV32IM+Zba, aligned-only; row 11) | **4.97M (6.5x)** | **0.80M (4.7x)** | **36 min / 5.5 min** |
| Same + Zbb + Zbs (row 12) | 4.87M (6.7x) | 0.78M (4.8x) | 35 min / 5.4 min |
| Row 11 + fx's own read / edit / write running on the CPU (row 24) | 5.65M | 0.80M | 41 min / 6.3 min |

- Every row in the waterfall runs fx's exact code.
- Row 24 is the most honest one: the host no longer does fx's file work, except grep and
  glob (see item 1).
- Turn 2 at 15k/s takes 53 s.

## Waterfall

Rows 1-12 are cumulative. Rows 13 on are single changes or variants, against the row they name.
"x" is P0 / row (higher is better). All measured.

| # | change | turn 1 (M) | turn 2 (M) | x turn 1 | x turn 2 | turn 1 at 2.3k/s | at 15k/s | turn 2 at 2.3k/s | at 15k/s | honesty verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **P0 baseline**: ReleaseSmall wasm, one 47 MB C file at clang -Os, unaligned scalar access, configured Chat Completions provider, one SSE event per ~token | 32.51 | 3.74 | 1.00 | 1.00 | 3.9 h | 36 min | 27 min | 4 min | exact fx code (P0) |
| 2 | + fx's built-in AI Gateway route instead of `FX_CONFIGURED_PROVIDERS` | 19.86 | 2.68 | 1.64 | 1.39 | 2.4 h | 22 min | 19 min | 3 min | exact fx code, same semantics; config selects fx's default provider route |
| 3 | + the whole model reply as one SSE event | 10.18 | 1.24 | 3.19 | 3.00 | 1.2 h | 11 min | 9 min | 1.4 min | exact fx code, same semantics; model server's chunking |
| 4 | + that event in one network read | 10.18 | 1.24 | 3.19 | 3.01 | 1.2 h | 11 min | 9 min | 1.4 min | exact fx code, same semantics; transport |
| 5 | + frozen virtual clock | 10.18 | 1.24 | 3.19 | 3.01 | 1.2 h | 11 min | 9 min | 1.4 min | exact fx code, same semantics; OS clock |
| 6 | + ReleaseFast wasm (Zig build mode) | 5.78 | 0.92 | 5.63 | 4.08 | 42 min | 6.4 min | 6.6 min | 61 s | exact fx code, same semantics; build flag |
| 7 | same, built as 96 wasm2c files + ThinLTO (what -O2 needs) | 5.80 | 0.92 | 5.61 | 4.06 | 42 min | 6.4 min | 6.6 min | 61 s | exact fx code, same semantics; build pipeline |
| 8 | + clang -O2 | 5.68 | 0.90 | 5.72 | 4.17 | 41 min | 6.3 min | 6.5 min | 60 s | exact fx code, same semantics; C compiler flag |
| 9 | + wasm linear memory at address 0 | 5.04 | 0.81 | 6.45 | 4.60 | 37 min | 5.6 min | 5.9 min | 54 s | exact fx code, same semantics; wasm2c runtime layout |
| 10 | **aligned-only CPU**: each access keeps its memarg alignment hint, under-aligned ones check the address | 5.26 | 0.84 | 6.18 | 4.45 | 38 min | 5.8 min | 6.1 min | 56 s | exact fx code, same semantics; wasm2c patch; 0 misaligned accesses |
| 11 | + word-at-a-time memcpy / memmove in the runtime for any alignment | **4.97** | **0.80** | **6.54** | **4.68** | **36 min** | **5.5 min** | **5.8 min** | **53 s** | exact fx code, same semantics; platform libc routine |
| 12 | + Zbb + Zbs | 4.87 | 0.78 | 6.67 | 4.77 | 35 min | 5.4 min | 5.7 min | 52 s | exact fx code, same semantics; ISA extension (costs circuit, below) |
| 13 | P0 + one SSE event per reply only | 14.71 | 1.51 | 2.21 | 2.47 | 1.8 h | 16 min | 11 min | 1.7 min | exact fx code, same semantics |
| 14 | P0 + per-token events, one read per reply | 32.24 | 3.70 | 1.01 | 1.01 | 3.9 h | 36 min | 27 min | 4 min | exact fx code, same semantics |
| 15 | P0 + frozen clock only | 32.42 | 3.73 | 1.00 | 1.00 | 3.9 h | 36 min | 27 min | 4 min | exact fx code, same semantics |
| 16 | P0 + ReleaseFast only | 20.52 | 2.56 | 1.58 | 1.46 | 2.5 h | 23 min | 19 min | 2.8 min | exact fx code, same semantics |
| 17 | P0 as 96 files + ThinLTO, -Os | 31.95 | 3.68 | 1.02 | 1.02 | 3.9 h | 36 min | 27 min | 4 min | exact fx code, same semantics |
| 18 | P0 + -O2 only (96 files + ThinLTO) | 27.05 | 3.31 | 1.20 | 1.13 | 3.3 h | 30 min | 24 min | 3.7 min | exact fx code, same semantics |
| 19 | row 11 at -O3 | 4.97 | 0.80 | 6.54 | 4.68 | 36 min | 5.5 min | 5.8 min | 53 s | same instruction count as -O2 |
| 20 | row 11 without the address check: under-aligned accesses as byte ops | 5.20 | 0.83 | 6.25 | 4.51 | 38 min | 5.8 min | 6.0 min | 55 s | exact fx code, same semantics |
| 21 | P0 on an aligned-only CPU, hints honoured, under-aligned = byte ops | 33.60 | 3.78 | 0.97 | 0.99 | 4.1 h | 37 min | 27 min | 4 min | exact fx code, same semantics |
| 22 | P0 on an aligned-only CPU without hints (P0's 60.4M build) | 51.78 | 5.94 | 0.63 | 0.63 | 6.3 h | 58 min | 43 min | 6.6 min | exact fx code, same semantics |
| 23 | P0 advertising only `read_file` (turn 2 comparable only) | 31.09 | 3.51 | 1.05 | 1.06 | 3.8 h | 35 min | 25 min | 3.9 min | changes fx's behaviour: turn 1's other calls fail |
| 24 | row 11 + fx's own read / edit / write tools in the guest | 5.65 | 0.80 | 5.76 | 4.66 | 41 min | 6.3 min | 5.8 min | 53 s | changes which fx code runs, toward more of fx: its tool code moves from the host onto the CPU |
| 25 | P0 + fx's own read / edit / write tools in the guest | 32.92 | 3.65 | 0.99 | 1.02 | 4.0 h | 37 min | 26 min | 4 min | as row 24 |

**Would a skeptic accept it?**
- **Rows 2-12:** yes.
  - No row moves fx's computation to the host, and none changes a byte of fx's code.
  - Rows 2-5 change only what fx is fed (configuration, the model server's framing, a clock).
  - Rows 6-12 change only the build after fx's source, the runtime library and the CPU.
- **Row 2:** fx-core's own default route, the one libfx uses when the app configures no
  provider. Notes:
  - the request goes through fx's gateway serializer (`vercel_protocol`) and its SSE reducer
    (`gateway/client.zig`), instead of the Chat Completions ones;
  - `AI_GATEWAY_API_KEY` is a placeholder;
  - the platform's `fx_http_stream_*` serves the gateway URL with the model.

  Disclose it as "fx's AI Gateway provider, served locally".
- **Rows 3-4:** the model server may chunk a stream as it likes, and fx reduces every valid
  chunking to the same completion. It sends fewer ACP `session/update` chunks: 22 instead of
  198.
- **Row 24:** a skeptic should prefer it.
  - It needs a patched platform libc and std, and a platform root that exports one function.
  - Its caveats are in item 1.

## 1. fx's own file tools

**The blocker for `.allow_native_tools = true` on WASI is in fx's source, so a different root
doesn't help.**
- `acp/tool_call_presentation.zig:91-94` (`activeToolSet`):
  - host tools win when the client sends any;
  - otherwise `if (comptime host_target.is_wasm) return tool_set_contract.empty;`.
- `acp/prompt.zig:671-674` (`hostToolProvider`): `if (comptime host_target.is_wasm) return
  js_host_tools.provider();`.
- `is_wasm` is `os.tag == .wasi` (`core/hosts/target.zig`).
- So on WASI, `allow_native_tools` is never read.
- Any other OS tag compiles fx's threaded native paths, which riscv32 can't build (P0: u64
  atomics, `Thread.spawn`).
- A different wasm32 OS tag would select the same native paths, and no Zig std I/O exists for
  them.

**What was built instead: a platform bridge** (`fx/zbuild/rv_tools_main.zig`).
- The root is fx's `wasm_core_main.zig`, imported and called unchanged, plus one
  `export fn fxplat_native_tool`. It runs fx's own built-in tools:
  - `builtin_tools.read_file` through `tool_dispatch.dispatchAuthorizedToolCall`;
  - `edit_file` / `write_file` through `tool_admission.preflightFileMutation` and
    `file_mutation_execution.execute`, fx's core mutation path, with permission mode `yolo`.
- `fx_host_tool_call` (libfx's host-tool import) is served on the CPU:
  - `imports.c --native-tools` calls the export first;
  - only a tool that the bridge doesn't cover (-100) goes to the host;
  - files are reached through a WASI preopen "/" served by `fx/wasifs.py`.
- fx's agent loop, admission, presentation and history are the host-tool path, as in libfx.
- **What does not run:** the native-tool runtime around a call (permission review, read
  tracker, change tracker, undo snapshots), because the ACP server never selects it on WASI.
- **Verdict:** changes which fx code runs, toward more of it.
  - The host stops doing fx's read, edit and write.
  - A skeptic accepts it with that disclosure.

**What else stood in the way, and how it was handled:**
1. `grep_files` and `glob_files` don't compile into any fx WASM build.
   - Both list files through `core/workspace/tool_files.zig:52`, which calls
     `core/indexer/scan.zig`.
   - `scan.zig:418` may start `std.Thread` workers.
   - fx's WASM builds are single-threaded: "Cannot spawn thread when building in single-threaded
     mode".
   - Fixing that needs fx's source or a multi-threaded wasm (wasi-threads, atomics); not done.
   - **They stay host tools.**
2. `edit_file` refuses every existing file on WASI. Two std / libc facts cause it, not fx:
   - wasi-libc's `fstat` reports `st_mode = S_IFREG` with no permission bits. Zig's std uses
     libc's `fstat` when libc is linked. fx's preimage check (`core/tooling/file_mutation.zig:1018`,
     `permissions.toMode() & 0o222 == 0`) then fails with "file mutation failed before commit".
   - Zig std's `File.setPermissions` returns `error.Unexpected` on WASI
     (`lib/std/Io/Threaded.zig:9261`). fx stages the new file with the old file's permissions
     (`file_mutation.zig:532`).
   - **Fix:** a platform patch in a copy of Zig's lib dir (`ZIG_LIB_DIR`), in
     `fx/zbuild/zig-lib-wasi-platform.patch`:
     - stat reports 0644 / 0755;
     - setPermissions is a no-op on WASI.
   - fx is untouched. This is an OS-layer choice: the preopen grants write.
3. With workspace root "/", fx's display paths lose their first character ("edited
   inycalc/calc.py"), so the bridge uses root `/w`.

**Per-call cost of fx's native tools (M, RV32 instructions in the guest):**

| tool | row 24 stack | P0 stack | wasm fuel (ReleaseSmall) |
|---|---|---|---|
| read_file (266 B file) | 51,301 | 74,067 | 95,359 |
| edit_file (249 B of args, 250 B file) | 399,641 | 556,224 | 850,697 |
| write_file (147 B) | 216,997 | 291,670 | 388,720 |
| grep_files / glob_files | host (can't compile) | host | - |

- Turn 1 rises from 4.97M to 5.65M (+14%) when the three tools run in the guest.
- edit is the expensive one: it reads the file, hashes the preimage with SHA-256, writes a staged
  file, re-reads and hashes it, renames it, and builds the diff for the result.

## 2. The misaligned-access blocker: fixed

- **What P0 found:**
  - wasm2c turns every load and store into `memcpy` and drops the wasm alignment hint;
  - so an aligned-only RV32 needs byte ops everywhere (row 22: 1.59x);
  - with unaligned access allowed, P0 made 128,221 misaligned accesses.
- **The fix** (`fx/rv/alignsplit.py` + `build.py --align`) carries the hint through wasm2c.
  - An access whose memarg alignment is below min(natural, 4 bytes) is moved to a second wasm
    memory, M1. An i64 at 4-byte alignment is two aligned words.
  - The runtime aliases M1 onto memory 0. `build.py` renames those calls to `*_ua` helpers on
    memory 0.
  - Every other access becomes a plain aligned LW / SW / LH (`__builtin_assume_aligned`).
  - An under-aligned (`_ua`) access either:
    - checks the address and takes the word path when it happens to be aligned (`check`, row 10);
    - or always does byte ops (`bytes`, row 20).
- **Counts:**
  - ReleaseFast wasm: 10,804 of 244,527 static accesses are under-aligned;
  - ReleaseSmall: 4,756 of 124,698.
- **Results:**
  - **0 misaligned accesses** in every aligned build, so the hints held for this session.
  - The aligned-only CPU costs **+4%** (row 9 to 10) instead of P0's +59%.
  - With the faster copy (row 11) it ends below the unaligned row 9.
  - On P0's own build the hints alone recover 51.8M to 33.6M (row 22 to 21).
- **Honesty:** exact fx code, same semantics.
  - The hint is the compiler's promise; LLVM emits natural alignment only for aligned
    addresses.
  - rv32emu counts any access that breaks it; there were none.
  - On real hardware a broken hint would trap instead of being counted. For a run that is not
    this one, a skeptic would want a trap-and-emulate fallback.
- **What misaligned LW / SW in the circuit would still buy** (the ISA alternative): row 9
  against row 10, about 4% (M). Its circuit cost isn't estimated here.

## 3. clang -O2: the "crash" was out of memory

- **The cause:**
  - Rerun here, P0's LLVM fatal error shows as `LLVM ERROR: out of memory`.
  - On this machine C: was 99-100% full during this work, so the page file couldn't grow, and
    another job held about 3 GB.
  - The same error hit -Os and -O1 on the largest wasm2c function.
- **The fix:**
  - `wasm2c --num-outputs=96` splits the C.
  - `build.py` moves the largest function into a file of its own and compiles it alone first,
    at the requested level (falling back to -O1 / -O0 only if clang runs out of memory). The
    function is fx's `orchestrator.processQueuedPromptLoop`: 137k lines of C for ReleaseSmall,
    about 195k for ReleaseFast, and hot.
  - ThinLTO links the files, which keeps clang's cross-function view.
- **The split + ThinLTO pipeline matches the single file:** 2% fewer instructions on P0's build
  (row 17), +0.3% on ReleaseFast (row 7). -O2 on top gives 1.20x on ReleaseSmall (row 18), but
  only 2% after ReleaseFast (row 8).
- **Two pitfalls, both fixed in `build.py`:**
  - wasm2c's split files `#include "fxcore-impl.h"` with quotes. That finds the unpatched copy
    next to the .c before any `-I`, which left bounds checks on and doubled the count.
  - Compiling `processQueuedPromptLoop` at -O0 (a fallback during the memory trouble) cost 2.8M
    on turn 1 of row 7's build (8.57M against 5.80M). No row uses an -O0 fallback
    (`fallbacks.json` is empty for every build).
- **ReleaseFast** is the bigger lever: 1.58x alone on P0 (row 16), 1.79x after the host changes
  (row 5 to 6).
- **-O3** gives the same count as -O2 (row 19).
- **Code size (.text):**
  - P0: 3.98 MB;
  - P0's ReleaseFast: 7.84 MB;
  - row 11: 6.87 MB (ReleaseFast, -O2, aligned-only with hints).

## 4. Host and configuration

| change | alone on P0 | in the stack | why |
|---|---|---|---|
| AI Gateway route (no `FX_CONFIGURED_PROVIDERS`) | 1.64x turn 1 (row 2) | first step | see the list below |
| one SSE event per reply | 2.21x (row 13) | 1.95x (row 2 to 3) | turns stream cost per token into a fixed cost per reply (steps 3 and 6 of turn 1 drop from 1.8M / 6.1M of streaming to 0.15M / 0.27M) |
| one network read per reply | 1.01x | 1.000x | the per-read cost is small next to the per-event cost |
| frozen virtual clock (`clock_time_get` constant) | 1.003x | 1.000x | fx reads the clock about 430 times a session, and nothing it does is O(elapsed) |
| journal / persistence off | 0 | 0 | already off: fx-core journals only when the host opens a journal (no `libfx/journal_append` in any run) |
| compaction off | 0 | 0 | never fires here: the context window is far from full (`FX_AUTO_COMPACT_PERCENT` exists if needed) |
| smaller tool set | ~56k per advertised tool per request (turn 2: 3.74M to 3.51M with 1 of 5) | not applied | the session uses all five tools, so the set is already minimal; fx-core adds no tools of its own |

**Why the gateway route is cheaper:**
- It avoids the configured provider's SHA-256 `binding_identity`. That cost about 0.87M per tool
  step.
- The gateway reducer is also cheaper per event: about 42k instructions per output token
  instead of 63k.

**No configuration avoids the per-call SHA-256 inside the configured provider.**
- `binding_identity` hashes id, protocol, base_url and auth.
- The fixed 25-byte prefix plus four length-framed parts are always over 55 bytes, so it is two
  blocks whatever the strings are.
- `host_chat_completions.bundle` recomputes it for both sides of a comparison on every lookup.
- Only a different provider route removes it, as in row 2.

**What a turn-1 tool step costs at row 11 (M, instructions):**

| step | take in the last result + build the request | stream | stream end to tool call | total |
|---|---|---|---|---|
| read | 358k | 66k | 249k | 0.67M |
| grep | 322k | 70k | 289k | 0.68M |
| edit | 345k | 91k | 574k | 1.01M |
| write | 379k | 87k | 495k | 0.96M |
| glob | 403k | 66k | 256k | 0.73M |
| answer (142 tokens) | 430k | 144k | 348k | 0.92M |

## 5. What remains (row 11, whole session 7.01M)

**Where it goes (M), by function group.** `fx/profile_iter.py` builds this from `--prof` runs:
- per-PC counts and bytes moved;
- mnemonic pairs and triples;
- inclusive cost.

| group | instructions | share | bytes moved (load + store) | instr / byte |
|---|---|---|---|---|
| memcpy / memmove (runtime, wasm `memory.copy`) | 1.60M | 22.9% | 2.53 MB | 0.6 |
| JSON write / string escape (`json.Stringify`, ACP `writeJsonStr`) | 1.03M | 14.8% | 1.49 MB | 0.7 |
| JSON scanner (`json.Scanner.*`) | 1.03M | 14.7% | 1.42 MB | 0.7 |
| token estimate (`measureProviderRequest`, where it is inlined) | 0.61M | 8.8% | 81 KB | 7.6 |
| `text_utils.maskSecrets` | 0.44M | 6.2% | 241 KB | 1.8 |
| SHA-256 | 0.16M | 2.2% | 60 KB | 2.6 |
| wyhash + 128-bit multiply | 0.14M | 2.0% | 147 KB | 1.0 |
| `mem.eqlBytes` and other compares | 0.12M | 1.7% | 47 KB | 2.5 |

**Notes on the groups:**
- memmove's 7,754 calls average 80 bytes.
- The largest callers are fx's own struct and buffer copies:
  - `acp.prompt.AcpContext.toolContext`: 83 KB in 322 calls;
  - `acp.server.runWithTransport`: 72 KB;
  - `prompt_context.buildProviderPrompt`: 47 KB;
  - `Io.Writer.Allocating.drain`: 43 KB.
- After ReleaseFast, `mem.eqlBytes` falls from P0's top function (3.65M) to 0.12M.
- maskSecrets is still 6%.

**Superinstruction candidates (M, dynamic share of row 11):**

| pair | share | dependent (second reads the first's rd) | what it is |
|---|---|---|---|
| LW LW | 9.7% | 1.5% of those | two words: i64 values, copy loops |
| SW SW | 9.2% | 0 | the same, stores |
| ADDI ADDI | 7.5% | 0.2% | pointer and counter bumps |
| LW ADDI | 2.8% | 16% | load, then index |
| ADDI SW | 2.6% | 64% | address, then store |
| ADDI BLTU | 2.3% | 45% | loop step and test |

- Triples: SW SW SW 6.7%, LW LW LW 6.5%, ADDI ADDI ADDI 2.4%.
- The pairs are mostly independent, so they are one "two-word load / store" record (an RV32
  LD/SD-like pair) rather than fused dependent ops.
- Most of them sit in the copy loop, which item 5's top hardware candidate takes over.

**Top 5 exact-hardware candidates.**
- **The model:** each group becomes one streaming op at 1 token per 16 bytes moved, against
  1.45 tokens per instruction today.
- **Instructions saved (E, instruction-equivalents, whole session 7.01M):**

| # | candidate | now | at 1 tok / 16 B | saved | share of session |
|---|---|---|---|---|---|
| 1 | streaming copy (memcpy / memmove) | 1.60M | 0.11M-equiv | 1.49M | 21% |
| 2 | JSON string escape + write | 1.03M | 0.06M | 0.97M | 14% |
| 3 | JSON scanner | 1.03M | 0.06M | 0.97M | 14% |
| 4 | token estimate (scan for word / script classes) | 0.61M | 0.003M | 0.61M | 9% |
| 5 | secret-mask scan (`maskSecrets`) | 0.44M | 0.01M | 0.43M | 6% |
| | **all five** | 4.71M | | **4.45M** | **64% (session to about 2.6M, 2.7x)** |

- These are upper bounds.
- They count whole functions, and 2-5 contain control flow, allocation and calls that a
  streaming unit would hand back to the CPU.
- speedups.md's rule of thumb (a unit pays for a loop of 6 MLP stages or fewer, with 16-32 bits
  of carried state) fits 1, 4 and 5 directly. It fits 2 and 3 only for their inner byte loops.

## ISA extensions and their cost

| extension | measured gain | circuit cost |
|---|---|---|
| Zbb + Zbs | 2.0% (row 11 to 12); dynamic use: ROR 22k, SEXT.B 15k, BSET 3.7k, MAXU 2.4k, ANDN 2.4k, CLZ 1.5k, REV8 0.7k | about 2-4k MLP units, maybe +1 layer (E, speedups.md #8) |
| misaligned LW / SW | about 4% over the hint path (row 9 vs 10, before the copy change) | not estimated (byte-lane merge in the load / store records) |
| a two-word load / store record (LW LW, SW SW pairs) | up to about 9% of records (E, from the pair table: half of 9.7% + 9.2% pairs) | a 64-bit value path in the record format (E, not costed) |

rv32emu now decodes Zbb and Zbs. The interpreter has no Zbc / Zbkb, and they were not tried.

## Blockers left

1. **Native grep / glob:** they don't compile into a single-threaded WASM build
   (`core/indexer/scan.zig:418` `std.Thread.spawn`). They need a wasi-threads build of fx or an
   fx source change. They stay host tools.
2. **Native edit on WASI:** it needs the platform libc / std patch (above). Stock Zig 0.17 + wasi-libc
   can't run fx's `edit_file` on any existing file.
3. **Memory on this machine:** clang on the hot 195k-line function needs several GB, and C: is
   nearly full.
   - Builds use 3 parallel jobs, delete their C and objects after linking (`--clean`), and the
     largest file compiles alone.
   - Something removed Zig's 129 empty std stub files (`test.zig` etc.) from `dl-zig` during
     the run. They were restored from the release zip.
4. **The alignment proof is per run:** zero misaligned accesses in these sessions. A hint
   broken by other inputs would need a trap path on Qwenputer.
5. **Not measured:**
   - token counts on the Qwenputer itself (this is RV32 instructions × the P0 factor);
   - longer sessions, where the request build grows with history;
   - a boot snapshot (speedups #11).

     Boot plus `initialize` is 0.5M + 0.6M at row 11.
     - At row 9 (linear memory at 0), boot is 2.1M. The data segments' source and destination
       there are differently aligned, and P0's memcpy copies such buffers a byte at a time.
     - Row 11's copy brings it to 0.52M.
     - A snapshot removes all of it.

## Reproduce

Paths are the session scratchpad's (`.../scratchpad`); `it/env.sh` sets `ZIG`, `WABT` and `PY`.

1. The wasm builds:
   - P0's ReleaseSmall (`build-wasm-names`) and ReleaseFast (`build-wasm-fast`) builds.
   - The native-tool root:
     `ZIG_LIB_DIR=<patched lib> zig build -Dfx=FX -Darch=wasm -Doptimize=ReleaseFast -Dstrip=false -Droot=rv_tools_main.zig -Drdynamic=true`
     in `fx/zbuild`.
     - Copy `rv_tools_main.zig` into `FX/src` first, and remove it after.
     - Build the patched lib by applying `zig-lib-wasi-platform.patch` to a copy of Zig's
       `lib/`, and use a fresh global cache so wasi-libc is rebuilt.
2. The RV32 images: `python -I fx/rv/build.py --wasm W --out D --opt=-O2 --cpu generic_rv32+m+zba
   --split 96 -j 3 --lto --membase 0 --align check --both-copies [--native-tools] [--clean]`.
   - Row 12 adds `--cpu generic_rv32+m+zba+zbb+zbs`.
   - The emulator: `zig cc -target x86_64-windows-gnu -O2 -shared fx/rv/rv32emu.c -o rv32emu.dll`.
3. One run: `python -I fx/rv/rvhost.py rv32emu.dll D/fx-fastcopy.elf D/imports.json out.json
   --opt provider=gateway --opt env=AI_GATEWAY_API_KEY=placeholder --opt stream=event_chunk
   --opt clock=frozen [--opt native=1 --opt fs=repo] [--prof]`.
4. The table: `python -I fx/speed_iter.py rv32emu.dll builds.json speed-iter.json`.
   - `fx/builds.iter.example.json` lists the labels.
   - The profile: `python -I fx/profile_iter.py PROF_RUN.json out.json` (`RV_WATCH=memmove` adds
     call sites).
5. fx's sources: `sh fx/check_src.sh FX`.
