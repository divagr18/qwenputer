# fx on Qwenputer: exact dedicated units ("circuits") for fx's hot code

Status 2026-11-07. CPU only. Follows `speed-iter.md` (rows 11 and 24 are the starting points)
and `speedups.md` (§3, §6, §7, §14, §19, "the middle ground"). Raw numbers: `circuits.json`.
Code: `experimental/riscv/fx/circ/` (units, driver, fuzzer, reference), `fx/rv/circ_units.h` and
`fx/rv/circ_loops.h` (the units in rv32emu), `chipformer/rv/core.py` (the two units built as
circuits), `tests/test_rv_circ.py`. **M** = measured, **E** = estimated.

**The rule followed (fx-exact).**
- fx's source is byte-identical: `fx/check_src.sh` on the fx checkout after this work: 717 files,
  0 changed against `757bdb9` (M).
- The binaries are byte-identical: every unit ran on the row-11 and row-24 images of the speed
  iteration, unchanged since that run (sha256 `209f2a87…` and `282e53d4…`, M).
- A unit is a macro-uop that fires at a fixed ROM PC of that image (a function entry or a loop
  head) and has exactly the architectural effect of running the original instructions: memory
  bytes, every register live at its exit, the PC. Then the CPU resumes at the exit PC.
- The CPU is specialized to the ROM: `fx/circ/units.py` installs a unit only where the exact
  instruction words of its code region appear (a signature from the row-11 image; only
  layout-dependent words are masked: LUI/AUIPC immediates, call offsets, the `addi` of a
  `lui/addi` address pair, and jumps that leave the region). Row 24's build matched all 15
  triggers.
- Fallbacks run as compiled. A unit declines (no state change) on inputs it does not cover, or
  stops at a loop head before an iteration it does not cover (an escape byte, a candidate
  secret, the end of a buffer). Then the original code runs.
- Nothing moves to the host. The host is the same `host.py` / `rvhost.py`; units are part of
  the CPU.

**Two disclosed modelling choices.**
- **Dead registers and dead stack bytes.** A unit leaves alone the registers that are dead at
  its exit:
  - inside a function, from a static liveness analysis of that function (`units.liveness`;
    conservative: an indirect jump makes everything live);
  - after a return, by the calling convention (caller-saved registers other than a0, and a1
    for wyhash's u64).

  Bytes the original writes below the stack pointer at a return are dead too (the native stack
  below `sp`). Both are checked dynamically, not assumed: in verify mode every register and
  byte that differs from the original's is poisoned, and a later read counts as a violation.
  The poison check found one real case: ChaCha20 reads stale wasm shadow-stack bytes that
  SHA-256's and wyhash's frames had left. So those two units also write their frames exactly
  (the 64-word message schedule; the 64-bit products), and the wasm shadow stack is not
  treated as dead.
- **Baked constants.** The maskSecrets unit compares the text against the 12
  `secret_prefixes` as constants (as the circuit would, from DROM). The table and the strings
  live in wasm linear memory, so the emulator store-watches them for the whole run. 0 stores
  hit them (M).

## Result

| | turn 1 instr | turn 1 tokens | turn 2 tokens | session instr | session tokens | saved (tokens) | turn 1 wall at 2.3k / 15k tok/s |
|---|---|---|---|---|---|---|---|
| row 11 (speed-iter's best), no units | 4.97M | 6.98M | 1.09M | 7.01M | 10.01M | 0.0% | 50.6 min / 7.8 min |
| **row 11 + every unit** | 2.55M | 4.00M | 0.65M | 3.48M | 5.60M | 44.1% | 29.0 min / 4.4 min |
| row 11 + only the two IR-built circuits | 4.84M | 6.79M | 1.07M | 6.86M | 9.79M | 2.2% | 49.2 min / 7.5 min |
| row 24 (fx's own read / edit / write on the CPU), no units | 5.65M | 8.30M | 1.09M | 7.70M | 11.36M | 0.0% | 60.2 min / 9.2 min |
| **row 24 + every unit** | 2.80M | 4.40M | 0.65M | 3.73M | 6.01M | 47.1% | 31.9 min / 4.9 min |
| row 24 + only the two IR-built circuits | 5.51M | 8.10M | 1.07M | 7.54M | 11.12M | 2.0% | 58.7 min / 9.0 min |

- **Measured:** the CPU's own tokens and instructions (the instructions that still run), host
  I/O, the final repository, and every check.
- **Estimated:** the tokens of each unit's records. They are counted per firing by the
  record format below, for designs that are not built, except MMOVE and MEQL. The rows "only
  the two IR-built circuits" are measured end to end: their record formats exist as circuits.
- **Wall time is tokens / (2.3k or 15k tokens/s)**: tokens are the time unit, and the units
  emit tokens, not instructions.
  - speed-iter.md divided instructions by 2.3k / 15k instr/s instead. On that clock row 11's
    turn 1 was 36 min / 5.5 min; the same 4.97M instructions are 6.98M tokens: 50.6 / 7.8 min
    at 2.3k / 15k tokens/s.
  - With every unit, turn 1 takes **29.0 min / 4.4 min** (row 11) and **31.9 / 4.9 min**
    (row 24): 1.74x and 1.89x fewer tokens.
- **Against the 64% upper bound** (speed-iter.md item 5: five groups, whole session, at 1 token
  per 16 bytes):
  - the measured saving is **44.1% of row 11's session tokens** (47.1% on row 24): **69%
    of the bound**;
  - in instructions it is 7.01M -> 3.48M (50.4%).
  - The bound's five groups: copy, JSON write, JSON scan, token estimate and maskSecrets.
    The units take the copies and the token estimate whole: every memmove / memcpy / memset
    call, all 14 request bodies. Of the other three they take only the tight loops. Left
    after the units (M, units-on profile): 0.90M tokens in the JSON scanner's state machine,
    0.30M in encodeJsonString's escape path, 0.24M in maskSecrets' other matchers.
  - SHA-256 and wyhash, outside the bound, add 5.9% on row 11 and 12.4% on row 24 (there
    edit_file hashes the file, 122 blocks instead of 37).
- **Every step leaves host I/O identical:** the hash of every host call with its arguments and
  result, and of every byte the host read from or wrote to guest memory. The final repository
  is identical too, so fx sends the same ACP messages, requests and tool calls.


## How the token count is measured

- **The CPU's tokens** (M): rv32emu now charges every retired instruction the tokens of its
  record in today's trace format (`chipformer/rv/spec.py`):
  - a value record of 1, 2 or 3 tokens by the value's size (15-bit chunks);
  - SB / SH 4 tokens (V V V SS);
  - SW, branches, x0 writes 1;
  - taken jumps J;
  - DIV-class QS + digits + value;
  - an R refresh after 63 commits without a jump;
  - an ecall 3 (the call and a0's value from the OS).

  Row 11 comes out at 1.43 tokens per instruction (10.01M tokens, 7.01M instructions),
  inside speedups.md's assumed 1.0-1.55.
- **A unit's tokens** (E, by its record format, counted per firing): the data tokens of its
  records, at most 64 per record, then the commits. The rules:
  - Each live register the unit changes is committed as a value record (1-3 tokens).
  - The last token is a J to the exit PC.
  - A run longer than 64 data tokens re-dispatches: each further record costs the commits
    again plus a J.
  - Payloads are 16 bits, and data that moves RAM to RAM goes through the stream (a word
    written is 2 tokens).
  - A unit reads RAM / DROM in a position for free, and one RAM word per position.
  - Loop state above 16 bits costs a token per 16 bits.

  Per unit:

| unit | record per firing (tokens) |
|---|---|
| memmove / memcpy (full model) | 2 per aligned source word + 2 per partial destination word (its merged value) |
| memset | 1 per full destination word (data from the register, as SW) + 2 per partial word |
| eqlBytes (full model) | 2 per aligned word of operand a streamed (b is read through the RAM port), up to the mismatch, + a0 |
| SHA-256 block | W0..15 2 each, W16..63 2 each, new a / e of every round 4, the state's 8 words 2 each: 400 |
| wyhash | 2 per aligned input word + 20 per 64 x 64 -> 128-bit multiply |
| token estimate | 2 per aligned input word (run and total, 16 bits each) |
| JSON string scan, escape scan, ASCII run, memchr | 1 per aligned word read (a 16-bit cursor), + the cursor's SW where the loop stores it |
| maskSecrets prefix probe | 1 per aligned text word the 12 compares need (a 12-bit still-matching mask) |
| realloc byte copy | as memmove |
| **MOVE as built in the IR** | MP, 3 per word (CW lo, CW hi, CWW), J |
| **EQL as built in the IR** | MP, 2 per byte compared (EB, EC), J (it commits a0), then the `ret` |

  These are what the units emit. The original instructions they replace are counted with the
  CPU's own record model; no unit is charged "1 token per 16 bytes".

## Selection

The hot code of row 11 (M, `--prof`, per-PC counts):

| group (speed-iter share) | where the time is | unit | trigger(s), row 11 |
|---|---|---|---|
| memcpy / memmove 23% | rt.c `copy_fwd` loops (16-byte aligned 0.47M, 16-byte funnel 0.56M, bytes 0.10M); 7,763 memmove + 558 memcpy calls, 80 bytes on average | `move`: whole call | memmove 0x208f4d8, memcpy 0x208f2fc |
| JSON scan 15% | `Scanner.next` string loop: 16 instr / byte, the cursor stored each byte (14k iterations); the rest is the state machine (prologue saves 12 registers, dispatch) | `jscan`: run of plain string bytes | loop head 0x263ffbc |
| JSON write 15% | `encodeJsonString` run of unescaped bytes (8 instr / byte, 25k); `write` ASCII run (6 / byte, 19k); `writeJsonStr` byte appends (24 / byte, 2.9k: not done) | `jesc`, `ascii` | 0x263dea4, 0x263d664 |
| token estimate 9% | `measureProviderRequest`'s inlined whitespace-run estimate: 15 instr / byte over every request body (35k bytes) | `tokest`: whole loop | 0x22c4574 (a second copy at 0x22c5318 never runs) |
| maskSecrets 6% | 1,091 positions at 390 instr each; 270 of them are the 12-prefix `eqlIgnoreCase` probe | `maskp`: the probe at one position | 0x24d9e44 |
| SHA-256 2% | `Sha2x32.round`, fully unrolled, 37 blocks; 2.42 tokens / instr (32-bit rotate / xor values are 3-token records): 3.7% of tokens | `sha`: whole block | 0x25a8ad4 |
| wyhash 2% | `Wyhash.hash`, 892 calls (hash-map keys); 1.98 tokens / instr | `wyhash`: whole call | 0x262b4c0 |
| compare 1.7% | `mem.eqlBytes`, 2,637 calls | `eql`: whole call | 0x266616c |
| (small) | `writeOneLine` newline / CR searches (5 / byte), `realloc` byte copy (5 / byte), `memset` | `memchr` x2, `bcopy`, `set` | 0x20ce7a8, 0x20ce7d0, 0x21e44a0, 0x208f610 |

The top pairs, LW LW and SW SW (9.7% and 9.2% of row 11): about 186k of each (27-29%) are
`copy_fwd`'s unrolled 16-byte loops, which `move` replaces (E, from the per-PC counts). Much of
the rest is prologue / epilogue saves and restores of callee-saved registers (for example
`valueStartAssumeTypeOk` saves 12 registers per call). No fused-pair or whole-block records
were built (§14, §19): every live value still needs its own tokens, so they would only save NT /
J tokens and dead intermediates.

### The units (exact semantics, triggers, exits)

Register names are those of fx's compiled code at that PC (row 11; row 24's copies use the same
registers). "Dead" = registers the unit leaves alone (see above). Every unit's semantics is
written twice, independently: in C (`fx/rv/circ_units.h`, `circ_loops.h`) and in Python
(`fx/circ/ref_units.py`).

**Function units** (trigger = the entry; exit = the return to `ra` with `sp` restored; dead =
caller-saved registers except the return value):
- **move** (memmove a0 = d, a1 = s, a2 = n; memcpy the same). Bytes `[d, d + n)` become
  `s[0..n)` with memmove semantics (rt.c's copy_fwd / copy_bwd give exactly that whenever
  memmove picks them; their read-ahead only reads). memcpy declines when `s < d < s + n` (an
  overlapping memcpy has copy_fwd's own byte order). Declines if the copy reaches into the
  frames the original would push. a0 unchanged.
- **set** (memset a0 = d, a1 = c, a2 = n): bytes `[d, d + n)` = `c & 0xff`.
- **eql** (mem.eqlBytes a1 = a.ptr, a2 = a.len, a3 = b.ptr, a4 = b.len): a0 = lengths equal and
  (len 0, same pointer, or all bytes equal).
- **sha** (`Sha2x32(...,256).round` a0 = wasm instance, a1 = state, a2 = block): the SHA-256
  compression of the 64-byte block into the 8-word state; also the 64-word message schedule at
  `__stack_pointer - 0x100` (the original leaves it there). Declines when state, block and
  schedule overlap.
- **wyhash** (`Wyhash.hash` with seed 0 folded in: a0 = instance, a1 = ptr, a2 = len): the u64
  in a0 / a1, and the products the original leaves in its 0x60-byte shadow-stack frame (the
  final mix's halves at -0x60, final2's product at -0x50, the last final1 mix at -0x40, the last
  round's three mixes at -0x30..-0x08; found by probing, then fuzzed).

**Loop units** (dead = from liveness at the exit):
- **tokest** (head 0x22c4574 of `measureProviderRequest`'s inlined estimate). For each of the
  a6 bytes at s5, the loop does this:
  - a byte in 9..13 or ' ' adds ceil(run / 4) to the saturating u64 total a0:a1 and clears
    the run a3:a2;
  - any other byte increments the saturating run.

  It needs a4 = 5 and a5 = 0x20 (checked). Exit 0x22c45f0. It declines unless run and total
  stay below 2^16 (16-bit payloads).
- **jscan** (head 0x263ffbc of `json.Scanner.next`). Each iteration does a3 += 1, stores a3 at
  0x24(a2), stops at a4, and continues while the byte at a0 + a3 is in the plain set. The
  plain set is 0x20..0x7f except `"` and `\`, computed from the binary's jump table. The unit
  runs k >= 1 iterations and exits at the head before the first other byte. Dead at the head:
  a1 a5 a6.
- **jesc** (head 0x263dea4 of `encodeJsonString`). Each iteration does s8 += 1 and stops at the
  length at -0x3c(s0). For the byte b at s5 + s8:
  - b == s3 leaves the loop;
  - b != s7 continues if s7 < b, or b & 0x7e == 0x20, or (b + 0xdd) & 0xff <= 0x38;
  - anything else leaves the loop.

  fx keeps `"` and `\` in s3 / s7 in either order; the unit requires both below 256. The path
  effects on s11, a1, a2, a3 are reproduced. Exits at the head.
- **ascii** (head 0x263d664 of `json.Stringify.write`). Each iteration does a0 += 1; a0 >= s3
  exits to 0x263d7f4; a negative byte at s4 + a0 exits to 0x263d67c. Sets t4 / t5 as the code
  does.
- **maskp** (0x24d9e44 in `maskSecrets`): findSecretSpan's loop over the 12 `secret_prefixes` at
  one position (`text_utils.zig:844`). A prefix is skipped when the byte before is an
  assignment-key character (s6 == 0), or when `start + len >= text.len`. Otherwise it is
  compared with eqlIgnoreCase. Exits:
  - 0x24d9f10: none matched; s11 = 12;
  - 0x24d9eb8: prefix s11 matched.

  a4, a5 and a6 are live at 0x24d9f10 and are reproduced. It needs s5 = the table, s4 = '_'
  and a7 = 10 (checked).
- **memchr** (heads 0x20ce7a8 / 0x20ce7d0 in `writeOneLine`): `do { a3 = *a1; a3 == a0 -> found;
  a2--; a1++ } while (a2)`; two exits.
- **bcopy** (head 0x21e44a0 in `realloc`): `do { *a2++ = *s11++ } while (s11 != a1)`. Declines when
  the destination starts inside the source (+4 bytes), where the forward byte order matters.

**The two units built in the IR** (emulator twins `move_ir`, `eql_ir`, same conditions and
token counts):
- **MMOVE** sits on memmove's entry `beq a0, a1, ret`. It fires when d, s and n are 4-aligned,
  4 <= n <= 84, d != s, and d < s or d - s >= n (so a forward word copy is memmove's result). In
  the test machine it also needs both buffers below 0x7F80 (RAM is 32 KB there). Otherwise the
  `beq` runs and the original memmove follows.
- **MEQL** sits on eqlBytes' `beq a1, a3, ret`, after its length and zero checks (a0 = 1
  there). It fires when a1 != a3 and 1 <= n <= 31.
- Both exit with a J to the beq's target. That target is a `ret` in both functions, so the
  twins are checked after the return.

## Waterfall (one unit at a time)

Cumulative, in the order of each unit's saving on row 11. Instructions are M; unit tokens are E (see above).

### Row 11

| # | step | turn 1 instr | turn 1 tokens | turn 2 tokens | session instr | session tokens | session tokens saved | turn 1 at 2.3k tok/s | at 15k tok/s | host I/O = baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 4.97M | 6.98M | 1.09M | 7.01M | 10.01M | 0.0% | 50.6 min | 7.8 min | yes |
| 1 | + memmove / memcpy (streaming copy) | 4.14M | 6.06M | 0.91M | 5.46M | 8.15M | 18.6% | 43.9 min | 6.7 min | yes |
| 2 | + token estimate (whitespace-run scan) | 3.71M | 5.61M | 0.80M | 4.92M | 7.59M | 24.2% | 40.6 min | 6.2 min | yes |
| 3 | + SHA-256 block | 3.55M | 5.25M | 0.80M | 4.77M | 7.23M | 27.8% | 38.1 min | 5.8 min | yes |
| 4 | + JSON scanner string scan | 3.37M | 5.01M | 0.78M | 4.51M | 6.90M | 31.1% | 36.3 min | 5.6 min | yes |
| 5 | + maskSecrets prefix probe (compare with constants) | 3.10M | 4.72M | 0.78M | 4.24M | 6.61M | 34.0% | 34.2 min | 5.2 min | yes |
| 6 | + wyhash | 3.01M | 4.57M | 0.75M | 4.10M | 6.37M | 36.4% | 33.1 min | 5.1 min | yes |
| 7 | + JSON escape scan (encodeJsonString) | 2.86M | 4.41M | 0.71M | 3.90M | 6.16M | 38.5% | 32.0 min | 4.9 min | yes |
| 8 | + mem.eqlBytes (compare) | 2.75M | 4.24M | 0.70M | 3.78M | 5.97M | 40.4% | 30.7 min | 4.7 min | yes |
| 9 | + JSON write ASCII-run scan | 2.67M | 4.14M | 0.67M | 3.67M | 5.84M | 41.7% | 30.0 min | 4.6 min | yes |
| 10 | + realloc byte copy | 2.64M | 4.08M | 0.66M | 3.61M | 5.72M | 42.9% | 29.6 min | 4.5 min | yes |
| 11 | + writeOneLine newline / CR search | 2.59M | 4.03M | 0.65M | 3.53M | 5.64M | 43.7% | 29.2 min | 4.5 min | yes |
| 12 | + memset | 2.55M | 4.00M | 0.65M | 3.48M | 5.60M | 44.1% | 29.0 min | 4.4 min | yes |

Verify run (row 11, every unit, every firing checked against the original code): host I/O identical, final repo identical, check failures 0, poisoned register reads 0, poisoned byte reads 0, stores into baked constants 0.

| unit (trigger) | firings | declines | instructions replaced | their tokens | unit tokens | tokens saved |
|---|---|---|---|---|---|---|
| memmove | 7,763 | 0 | 983,326 | 1,478,021 | 346,288 | 1,131,733 |
| memcpy | 558 | 0 | 565,102 | 947,029 | 215,233 | 731,796 |
| memset | 473 | 0 | 55,408 | 67,675 | 25,560 | 42,115 |
| mem.eqlBytes | 2,637 | 0 | 118,036 | 202,149 | 10,572 | 191,577 |
| crypto.sha2 SHA-256 round (block compression) | 37 | 0 | 153,069 | 370,983 | 15,059 | 355,924 |
| hash.wyhash.Wyhash.hash | 892 | 0 | 141,442 | 280,515 | 46,819 | 233,696 |
| measureProviderRequest whitespace-run loop | 14 | 0 | 545,263 | 580,211 | 18,326 | 561,885 |
| measureProviderRequest whitespace-run loop (2nd copy) | 0 | 0 | 0 | 0 | 0 | 0 |
| json.Scanner.next string loop | 1,047 | 1,133 | 251,089 | 336,325 | 7,393 | 328,932 |
| json.Stringify.encodeJsonString plain-run loop | 1,908 | 1,975 | 202,404 | 225,697 | 11,409 | 214,288 |
| json.Stringify.write ASCII-run loop | 907 | 0 | 113,049 | 131,437 | 6,109 | 125,328 |
| text_utils.maskSecrets secret_prefixes probe | 1,091 | 0 | 271,637 | 301,085 | 3,156 | 297,929 |
| acp.jsonrpc.writeOneLine find-newline loop | 26 | 0 | 35,375 | 42,450 | 1,818 | 40,632 |
| acp.jsonrpc.writeOneLine find-CR loop | 26 | 0 | 35,375 | 42,450 | 1,818 | 40,632 |
| c.malloc.realloc byte-copy loop | 86 | 0 | 62,835 | 125,670 | 6,567 | 119,103 |

IR-built circuits only (row 11): session tokens 9.79M (2.2% saved), turn 1 6.79M tokens; host I/O identical.
  - memmove (circuit built in the core IR): 677 firings, 7,086 declines, 72,907 -> 23,557 tokens
  - mem.eqlBytes (circuit built in the core IR): 962 firings, 662 declines, 185,002 -> 14,662 tokens

Each unit alone (row 11): move 18.6%, tokest 5.6%, sha 3.6%, jscan 3.3%, maskp 3.0%, wyhash 2.3%, jesc 2.1%, eql 1.9%, ascii 1.3%, bcopy 1.2%, memchr 0.8%, set 0.4%.

### Row 24

| # | step | turn 1 instr | turn 1 tokens | turn 2 tokens | session instr | session tokens | session tokens saved | turn 1 at 2.3k tok/s | at 15k tok/s | host I/O = baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 5.65M | 8.30M | 1.09M | 7.70M | 11.36M | 0.0% | 60.2 min | 9.2 min | yes |
| 1 | + memmove / memcpy (streaming copy) | 4.78M | 7.32M | 0.92M | 6.10M | 9.42M | 17.0% | 53.0 min | 8.1 min | yes |
| 2 | + token estimate (whitespace-run scan) | 4.34M | 6.87M | 0.80M | 5.55M | 8.85M | 22.0% | 49.8 min | 7.6 min | yes |
| 3 | + SHA-256 block | 3.83M | 5.69M | 0.80M | 5.05M | 7.68M | 32.4% | 41.2 min | 6.3 min | yes |
| 4 | + JSON scanner string scan | 3.64M | 5.44M | 0.78M | 4.79M | 7.34M | 35.4% | 39.5 min | 6.0 min | yes |
| 5 | + maskSecrets prefix probe (compare with constants) | 3.37M | 5.15M | 0.78M | 4.52M | 7.04M | 38.0% | 37.3 min | 5.7 min | yes |
| 6 | + wyhash | 3.28M | 5.00M | 0.75M | 4.37M | 6.81M | 40.1% | 36.2 min | 5.6 min | yes |
| 7 | + JSON escape scan (encodeJsonString) | 3.12M | 4.83M | 0.71M | 4.17M | 6.59M | 42.0% | 35.0 min | 5.4 min | yes |
| 8 | + mem.eqlBytes (compare) | 3.01M | 4.65M | 0.70M | 4.04M | 6.38M | 43.8% | 33.7 min | 5.2 min | yes |
| 9 | + JSON write ASCII-run scan | 2.92M | 4.55M | 0.68M | 3.93M | 6.26M | 44.9% | 33.0 min | 5.1 min | yes |
| 10 | + realloc byte copy | 2.89M | 4.49M | 0.66M | 3.86M | 6.14M | 46.0% | 32.5 min | 5.0 min | yes |
| 11 | + writeOneLine newline / CR search | 2.84M | 4.44M | 0.66M | 3.79M | 6.06M | 46.7% | 32.1 min | 4.9 min | yes |
| 12 | + memset | 2.80M | 4.40M | 0.65M | 3.73M | 6.01M | 47.1% | 31.9 min | 4.9 min | yes |

Verify run (row 24, every unit, every firing checked against the original code): host I/O identical, final repo identical, check failures 0, poisoned register reads 0, poisoned byte reads 0, stores into baked constants 0.

| unit (trigger) | firings | declines | instructions replaced | their tokens | unit tokens | tokens saved |
|---|---|---|---|---|---|---|
| memmove | 8,474 | 0 | 1,025,108 | 1,543,014 | 354,743 | 1,188,271 |
| memcpy | 571 | 0 | 572,989 | 961,961 | 217,364 | 744,597 |
| memset | 567 | 0 | 65,370 | 79,824 | 29,800 | 50,024 |
| mem.eqlBytes | 2,844 | 0 | 125,928 | 214,829 | 11,312 | 203,517 |
| crypto.sha2 SHA-256 round (block compression) | 122 | 0 | 504,714 | 1,224,661 | 49,654 | 1,175,007 |
| hash.wyhash.Wyhash.hash | 907 | 0 | 143,780 | 285,112 | 47,394 | 237,718 |
| measureProviderRequest whitespace-run loop | 14 | 0 | 551,717 | 587,052 | 18,533 | 568,519 |
| measureProviderRequest whitespace-run loop (2nd copy) | 0 | 0 | 0 | 0 | 0 | 0 |
| json.Scanner.next string loop | 1,069 | 1,156 | 258,349 | 345,983 | 7,575 | 338,408 |
| json.Stringify.encodeJsonString plain-run loop | 1,974 | 1,987 | 206,003 | 229,659 | 11,650 | 218,009 |
| json.Stringify.write ASCII-run loop | 907 | 0 | 115,335 | 134,104 | 6,197 | 127,907 |
| text_utils.maskSecrets secret_prefixes probe | 1,091 | 0 | 271,637 | 301,085 | 3,156 | 297,929 |
| acp.jsonrpc.writeOneLine find-newline loop | 26 | 0 | 35,400 | 42,480 | 1,819 | 40,661 |
| acp.jsonrpc.writeOneLine find-CR loop | 26 | 0 | 35,400 | 42,480 | 1,819 | 40,661 |
| c.malloc.realloc byte-copy loop | 86 | 0 | 62,835 | 125,670 | 6,567 | 119,103 |

IR-built circuits only (row 24): session tokens 11.12M (2.0% saved), turn 1 8.10M tokens; host I/O identical.
  - memmove (circuit built in the core IR): 714 firings, 7,760 declines, 77,157 -> 24,843 tokens
  - mem.eqlBytes (circuit built in the core IR): 1,037 firings, 693 declines, 194,757 -> 15,999 tokens

Each unit alone (row 24): move 17.0%, tokest 5.0%, sha 10.3%, jscan 3.0%, maskp 2.6%, wyhash 2.1%, jesc 1.9%, eql 1.8%, ascii 1.1%, bcopy 1.0%, memchr 0.7%, set 0.4%.



## Verification

**1. Whole sessions, every firing checked (M).** `run.py --mode verify` runs each firing in
this order:
- run the unit;
- undo it;
- step the original instructions from the same state to the same exit (for a unit that exits
  at its own loop head: until the induction register reaches the unit's value);
- compare PC, every register not dead at that exit, and every byte either side wrote. Dead
  stack bytes below `sp` at a return are not compared, but they are poisoned.

Then the run continues from the unit's state. Results, rows 11 and 24 (in the tables above):
- 0 check failures;
- 0 reads of a poisoned register or byte;
- 0 stores into the baked constants;
- host I/O, ACP stdout and the final repository identical to the run without units.

Each unit's "instructions replaced" and "their tokens" are counted by stepping those original
instructions.

The check found three real problems during the work, all fixed before these runs:
- a loop-head exit compared too early (now stops at the induction value);
- the comparison of a byte written twice;
- the stale shadow-stack frames ChaCha20 reads.

**2. Differential fuzzing (M), `fuzz.py`.** Each unit gets 10^5 random input states on row
11's image and 2 x 10^4 on row 24's. Each case is checked three ways:
- C unit against the original instructions (`rv_unit_check`);
- the independent Python reference (`ref_units.py`) against the C unit's final state;
- a decline must leave the state untouched, and must agree with the reference's decline.

The generators cover these edge cases:
- zero lengths;
- overlap both ways;
- every alignment;
- `end` right after the cursor;
- the cursor inside the scanned bytes;
- run / total near 2^16;
- s3 / s7 in either order and out of range;
- inserted and case-changed secret prefixes;
- the preceding-byte flag;
- lengths across wyhash's 16 / 48-byte paths and up to 1,500-byte scans;
- inputs a unit must decline.

Result: **0 unit errors in 1.98M cases.** The one "error" in the first row-11 run was the
harness: a generated cursor 4 bytes below the compared window. It was fixed and jscan was
rerun with 10^5 cases, 0 errors.

The table rows, in order: the row-11 run, the row-11 rerun (jscan and the two IR twins), the row-24 run.

| unit | cases | fired | declined | errors |
|---|---|---|---|---|
| memmove | 100,000 | 100,000 | 0 | 0 |
| memcpy | 100,000 | 87,890 | 12,110 | 0 |
| memset | 100,000 | 100,000 | 0 | 0 |
| mem.eqlBytes | 100,000 | 100,000 | 0 | 0 |
| crypto.sha2 SHA-256 round (block compression) | 100,000 | 100,000 | 0 | 0 |
| hash.wyhash.Wyhash.hash | 100,000 | 100,000 | 0 | 0 |
| measureProviderRequest whitespace-run loop | 100,000 | 75,246 | 24,754 | 0 |
| json.Scanner.next string loop | 100,000 | 62,759 | 37,241 | 1 |
| json.Stringify.encodeJsonString plain-run loop | 100,000 | 74,939 | 25,061 | 0 |
| json.Stringify.write ASCII-run loop | 100,000 | 100,000 | 0 | 0 |
| text_utils.maskSecrets secret_prefixes probe | 100,000 | 93,282 | 6,718 | 0 |
| acp.jsonrpc.writeOneLine find-newline loop | 100,000 | 99,016 | 984 | 0 |
| acp.jsonrpc.writeOneLine find-CR loop | 100,000 | 99,009 | 991 | 0 |
| c.malloc.realloc byte-copy loop | 100,000 | 84,087 | 15,913 | 0 |
| memmove (circuit built in the core IR) | 100,000 | 33,259 | 66,741 | 0 |
| mem.eqlBytes (circuit built in the core IR) | 100,000 | 66,945 | 33,055 | 0 |
| json.Scanner.next string loop | 100,000 | 63,565 | 36,435 | 0 |
| memmove | 20,000 | 20,000 | 0 | 0 |
| memcpy | 20,000 | 17,534 | 2,466 | 0 |
| memset | 20,000 | 20,000 | 0 | 0 |
| mem.eqlBytes | 20,000 | 20,000 | 0 | 0 |
| crypto.sha2 SHA-256 round (block compression) | 20,000 | 20,000 | 0 | 0 |
| hash.wyhash.Wyhash.hash | 20,000 | 20,000 | 0 | 0 |
| measureProviderRequest whitespace-run loop | 20,000 | 15,063 | 4,937 | 0 |
| json.Scanner.next string loop | 20,000 | 12,653 | 7,347 | 0 |
| json.Stringify.encodeJsonString plain-run loop | 20,000 | 14,929 | 5,071 | 0 |
| json.Stringify.write ASCII-run loop | 20,000 | 20,000 | 0 | 0 |
| text_utils.maskSecrets secret_prefixes probe | 20,000 | 18,665 | 1,335 | 0 |
| acp.jsonrpc.writeOneLine find-newline loop | 20,000 | 19,792 | 208 | 0 |
| acp.jsonrpc.writeOneLine find-CR loop | 20,000 | 19,817 | 183 | 0 |
| c.malloc.realloc byte-copy loop | 20,000 | 16,897 | 3,103 | 0 |

**3. The IR circuits (M), `tests/test_rv_circ.py`, 51 tests, all pass.**
- fx's own memcpy / copy_fwd / memmove block (197 words) and eqlBytes (198 words), from row
  11's image, run in the core's test machine.
- The traced machine with the macros leaves the same RAM and results as the plain ISA
  emulator stepping fx's instructions:
  - 7 fixed memmove calls;
  - 4 eqlBytes calls (a == b, mismatch after 4 bytes, 31 equal bytes, an immediate
    mismatch);
  - 46 random programs of 5-6 calls, fired and declined: aligned and unaligned, overlapping
    both ways, lengths 0..96, equal and one-byte-different operands.
- The circuit, teacher-forced, predicts every token of 8 of these programs: the 2 fixed
  ones and 6 random ones, macro records and the original code's decline paths together.

**4. The existing suites on the final core (M):**
- rv32ui / rv32um / rv32uzba and the compiled C programs, teacher-forced through the circuit
  (`test_rv_core.py`, with its depth and capacity check);
- `test_rv_ref.py`, `test_rv_bridge.py`, `test_rv_fuzz.py`, `test_rv_hf.py`.

Result, with test_rv_circ.py in the same run: **287 passed, 14 skipped, 4 failed** (24 min).
- The skips are the suites' own: RV_SLOW traces, and programs checked free-running on the
  GPU executor.
- The 4 failures fail identically on the unmodified `fx` branch, so they are not from this
  work:
  - three `test_rv_hf.py` tests: this machine's transformers has no `Qwen3_5ForCausalLM`;
  - `test_fuzz_free_running_cpu`: `fastexec.FreeRun` looks up the bridge family `Q`
    as a chip family (`KeyError: 'Q'`). The suites caught one bug of mine on the way: a variable in the MEQL
code shadowed the digit file's zero-length check (`test_faults[illegal]`). It was fixed
before the final run.


## Circuit cost

**Built (M, `chipformer/rv/core.py`, `core.build().stats()` against the unmodified core):**

| | base core | with MMOVE + MEQL | delta |
|---|---|---|---|
| layers | 11 | 11 | **0** |
| FFN units | 42,014 | 44,530 | **+2,516** |
| heads | 99 | 100 | **+1** (`eqa`, MEQL's latest operand byte) |
| fullest MLP sublayer | 8,149 | 9,214 of 9,216 (L5) | |
| RAM value columns per value head | 66 | 98 of 128 | +32 (block C: the CWW word from two CW payloads) |
| vocabulary | | +65,797 rows | families CW (16-bit), EB (8-bit); controls MP, CWW0/1, EC0/1 |

The +2,516 units break down as follows:
- 928: two new modes of the BLD / BSCAN streaming adder;
- 532: the exit-target adder;
- 160: the offset selection that keeps it at two modes;
- 328: decode, fire conditions and the decline path;
- 370: MMOVE's schedule;
- 198: MEQL's compare.

The first build used four adder modes. That filled L5's MLP and spilled one layer (12
layers). Picking the offsets at L4 brought it back to 11. **L5 is now full**, so the next unit
that reads registers needs a layer (speedups.md's +1-2 layers).

**The others are emulator-only, costed (E).** The estimate scales the built units' parts:
- an adder mode about 460 units;
- a 16-bit add about 250;
- a lane class test about 10;
- a state head 1 head.

| unit | what the circuit needs | FFN units | heads | layers |
|---|---|---|---|---|
| move, full (any alignment, partial words, backward, re-dispatch) | funnel by (s - d) & 3 from a two-row src-word head, merge with streamed old end words, write from t-1 dims; d + n / s + n adders for the schedule; a backward mode; a continuation token | 3.5-4.5k | 3 | +1 (L5 full) |
| set | the fill word from the register (SW-like), partial-word merge | 0.8k | 0 | shared |
| eql, full (word-streamed a, any alignment) | two-row a-word head, funnel + 4-lane compare, early exit | 1.5-2k | 2 | shared |
| tokest | 4 lane classes, ceil(run / 4) per whitespace byte, two 16-bit adds, run / total through payload + a one-row head | 1.5k | 1 | shared |
| jscan, jesc, ascii, memchr (one shared word scanner) | lane class tables (ranges), first-stop priority over 4 lanes, a 16-bit cursor add, the cursor SW | 1.5k for all four | 1 | shared |
| maskp | 12 constant prefixes x 4 lanes case-folded byte compares, a 12-bit mask in the payload, the preceding-byte class, 12 `start + len < text.len` compares | 2k | 1 | shared |
| bcopy | the move unit at another PC | 0.2k | 0 | - |
| sha | per round two 5- and 2-operand 32-bit adds, Ch / Maj, rotations (wiring), W schedule adds; state head for b..h and W | 5-7k | 3-4 | +1 (add chains) |
| wyhash | 64 x 64 -> 128 through the multiplier columns in 4 passes, 128-bit adds, partial-product head | 2-3k | 1-2 | shared |
| **all emulator-only units** | | **18-23k** | **12-15** | **+1-2** |

This matches speedups.md's 15-30k units and +1-2 layers for 10-20 units. It also matches its
rule of thumb: a unit pays when its loop body fits about 6 MLP stages, with at most 16-32
bits of carried state. All of these do, except SHA-256's round, which needs a layer of its own
for the adds.


## Honesty verdicts

| unit | verdict | what a skeptic should know |
|---|---|---|
| move, set (memmove / memcpy / memset entries) | **exact fx code, same semantics** | rt.c's routines in the unchanged image; the macro fires at their entry PCs. Caller-saved registers are dead after the return by the ABI; checked per run (0 poisoned reads). |
| eql, sha, wyhash (fx / Zig std functions) | **exact fx code, same semantics** | Function-level macros. sha and wyhash also write the stack-frame bytes the original leaves: fx's ChaCha20 reads such stale bytes later, which the poison check caught. |
| tokest, jscan, jesc, ascii, memchr, bcopy (fx's own compiled loops) | **exact fx code, same semantics** | A first Futamura projection: macros at loop heads of fixed ROM words; dead registers from static liveness. Every iteration they do not cover (escape, special byte, end, overlap) runs the original. |
| maskp (secret-prefix probe) | **exact fx code, same semantics, one disclosure** | It compares against `secret_prefixes` as constants: fx's const data, in wasm linear memory. The emulator proves per run that nothing stores into it (0 stores). A circuit would hold the constants in DROM. Only the probe is covered: the other matchers and the writeByte path at each position stay on the CPU. |
| MMOVE, MEQL (built in the IR) | **exact fx code, same semantics** | Exact in the IR against the reference and against fx's own instructions (tests/test_rv_circ.py). Coverage is a subset: MMOVE only aligned copies of up to 84 bytes, MEQL up to 31 bytes. |
| token counts of the emulator-only units | **estimate** | Their record formats are designs that follow the core's constraints, not built circuits. The two built ones show the formats can be met. MMOVE costs 3 tokens per word where the full model assumes 2, because writes do not yet ride the stream tokens. |

No unit moves fx's computation to the host, and none changes a byte of fx's code or binary.

## Blockers and what is left

1. **The full streaming copy is not built in the IR.** MMOVE as built fires on 677 of row
   11's 7,763 memmove calls (aligned and at most 84 bytes). The modelled unit covers all 8,321
   memmove / memcpy calls; its saving (1.86M tokens, the largest) is E until built.
   - What it needs: the funnel from a two-row source-word head, streamed old end words for
     the partial-word merges, a backward mode, and a continuation token for long copies
     (records are limited to j <= 66).
   - It also needs the CWW write to ride the CW tokens (2 tokens per word, not 3). That
     takes the write key from a second address, but the core's RAM keys are the previous
     position's read address.
2. **L5's MLP is full** (9,214 of 9,216 units). Any further unit that reads registers adds a
   layer.
3. **The IR tests run on the core's test machine** (32 KB RAM, 256 KB ROM), not on fx's memory
   map (speedups.md §12). The macros' range check (`< 0x7F80`) is that machine's. At fx's
   scale it would be the RAM map's.
4. **Liveness and dead stack bytes are checked per run, not proven.** This is like
   speed-iter's alignment proof. A different session could read a register a unit treats as
   dead; the poison check would flag it. Real hardware has no such check.
5. **Today's record format (15-bit V chunks) for the CPU, 16-bit payloads for the units.**
   With 16-bit V / F (speedups.md §7) the CPU's own count would drop 4-8%, and the units'
   share would drop with it.
6. **What is left after the units** (row 11, M, `--pcdump` with units on):
   - the JSON scanner's state machine (Scanner.next, nextAllocMax, peekNextTokenType,
     appendSlice: about 0.9M tokens);
   - `encodeJsonString`'s escape path (0.3M);
   - ArenaAllocator.alloc (0.27M);
   - the rest of maskSecrets' matchers (0.24M);
   - the orchestrator (0.2M);
   - writeJsonStr's byte appends (0.11M; a unit for them was not done);
   - prologue / epilogue saves.

   This is flat glue (§ "middle ground"): each piece runs too rarely to pay for a circuit.
7. **Toolchain:** nothing was rebuilt, so Zig was not used. The `dl-zig` tree was checked
   against its release zip at the start: 0 files missing. The 129 stub files did not go
   missing again.
8. **Turn 2 at 15k tok/s** and longer sessions were not separately optimized. The units scale
   with request size (tokest, JSON write and scan), so longer sessions should gain more (E).

## Reproduce

Paths as in speed-iter.md (`.../scratchpad`, `it/env.sh`).

1. The emulator: `zig cc -target x86_64-windows-gnu -O2 -shared fx/rv/rv32emu.c -o rv32emu.dll`.
2. The signatures (once, from row 11's image): `python -I fx/circ/units.py sigs ROW11.elf`.
   To list the triggers in any image: `python -I fx/circ/units.py specs ELF [all|move_ir,eql_ir]`.
3. One session:
   `python -I fx/circ/run.py rv32emu.dll ELF imports.json out.json --units all|none|NAME,.. --mode on|verify --opt provider=gateway --opt env=AI_GATEWAY_API_KEY=placeholder --opt stream=event_chunk --opt clock=frozen [--opt native=1 --opt fs=repo]`.
   - `--mode verify` checks every firing against the original code.
   - `out.json` has per-turn instructions and tokens, the host-I/O hash, the final repo,
     violations and per-unit counters.
   - With `fs=repo` the driver gives files a deterministic inode, since the real one changes
     with the temp directory.
4. The waterfall: `python -I fx/circ/waterfall.py rv32emu.dll builds.json waterfall.json --jobs 2`.
5. The fuzzer: `python -I fx/circ/fuzz.py rv32emu.dll ELF fuzz.json --cases 100000 [--units ..]`.
6. The tables and `circuits.json`: `python -I fx/circ/report.py waterfall.json FUZZ.json,.. ircost_base.json ircost_new.json circuits.json`.
7. The IR: `pytest tests/test_rv_circ.py` (51 tests, about 90 s), and the existing suites
   `pytest tests/test_rv_core.py tests/test_rv_ref.py tests/test_rv_bridge.py tests/test_rv_fuzz.py tests/test_rv_hf.py`.
