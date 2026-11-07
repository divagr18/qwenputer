# fx's hot-code units on the paged map

Status 2026-11-08, branch `fx-circ2`. Follows `circuits.md` (the emulator's estimates) and `memory.md`
(the 59-layer map). Code: `chipformer/rv/hot.py` (reference and circuit), hooks in `core.py`, `ref.py`,
`fxrun.py`; `fx/circ/fuzz2.py`, `fx/rv/circ_loops.h` (the emulator's twins), `tests/test_rv_hot.py`. Raw
numbers: `circuits2.json`. **M** = measured, **E** = estimated.

Row 24's image and session (`fx-fastcopy.elf`, sha256 `282e53d4`), from the boot snapshot. Every session
figure below is the GPU's (NVIDIA L4, Executor v4) unless marked E. `fx/check_src.sh`: 717 files, 0
changed. The binary is unchanged, and now so is the ROM: a unit is a PC comparison, not a ROM word.

## Result

| # | step | session tokens | saved | turn 1 tokens | saved | instructions | tok/s | us / step | session | turn 1 | wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline, no units | 10,508,989 | 0.0% | 7,620,307 | 0.0% | 7,174,817 | 25,947 | 38.54 | 405 s | 294 s (4.9 min) | 0 |
| 1 | + memmove, memcpy | 9,288,793 | 11.6% | 6,799,730 | 10.8% | 6,099,150 | 25,099 | 39.84 | 370 s | 271 s (4.5 min) | 0 |
| 2 | + token estimate | 8,813,638 | 16.1% | 6,498,929 | 14.7% | 5,600,690 | 25,053 | 39.92 | 352 s | 259 s (4.3 min) | 0 |
| 3 | + JSON string scan | 8,497,728 | 19.1% | 6,248,827 | 18.0% | 5,343,410 | 24,972 | 40.04 | 340 s | 250 s (4.2 min) | 0 |
| 4 | + eqlBytes | 8,311,672 | 20.9% | 6,131,551 | 19.5% | 5,229,736 | 25,048 | 39.92 | 332 s | 245 s (4.1 min) | 0 |
| 5 | + ASCII run | 8,210,570 | 21.9% | 6,064,511 | 20.4% | 5,118,020 | 24,918 | 40.13 | 330 s | 243 s (4.1 min) | 0 |
| 6 | + realloc copy | 8,091,782 | 23.0% | 5,979,833 | 21.5% | 5,055,271 | 24,877 | 40.2 | 325 s | 240 s (4.0 min) | 0 |
| 7 | + writeOneLine (2 loops) | 8,022,690 | 23.7% | 5,934,351 | 22.1% | 4,984,783 | 24,603 | 40.65 | 326 s | 241 s (4.0 min) | 0 |
| 8 | + memset | 8,004,339 | 23.8% | 5,921,631 | 22.3% | 4,929,183 | 24,652 | 40.56 | 325 s | 240 s (4.0 min) | 0 |
| 9 | + SHA-256 | 7,003,277 | 33.4% | 4,966,947 | 34.8% | 4,490,873 | 23,521 | 42.51 | 298 s | 211 s (3.5 min) | 0 |

- **Whole session, every step:** the GPU generated every token between ecalls, 0 wrong against the
  reference. Session: 10,508,989 -> 7,003,277 tokens (33.4%), 7,174,817 -> 4,490,873 instructions, 405 s -> 298 s. Turn 1: 7,620,307 -> 4,966,947 tokens, 294 s -> 211 s (4.9 -> 3.5 min). The target was 3 minutes: see Blockers.
- **Exact against rv32emu at every step:** registers equal at all ecalls and final memory equal
  (0 of 1,897,580 words differing), against a log taken with the same units on in rv32emu from
  boot's end (`snaplog.py --units`).
- **Against the original code (M, all units on, against the units-off log):** in 400 ecalls only
  caller-saved registers differ (t0-t6, a2-a6: dead by the ABI or by liveness), and 22 memory words,
  all on the C stack below sp (the original's dead frames). Host I/O is the same log.
- **Turn 1 here is from the boot snapshot**, up to the last ecall before rv32emu's turn-1 boundary
  (segment 320). `memory.md`'s 5.4 min and `circuits.md`'s turn-1 figures include boot. Reading the
  same session from reset adds 906,558 tokens (M, `memory.md`), about 40 s at these rates, and
  the units are off during it in these runs.
- **Speed:** 14 phases at every step, as at baseline. The units do not lose the token saving to a slower
  step: 25,947 -> 23,521 tok/s, -9.3%, against -33.4% tokens.
- **The emulator's 44-47%** was a from-reset figure with all 12 units. The saving is less from the
  snapshot (boot's copies are gone: it had 0.9M tokens, mostly copies) and 3 units are not built.

## How a unit works

- **Trigger.** A constant comparison of the PC word (one MLP unit per PC), so nothing in the ROM
  changes and a decline costs one token: M.MD, then the original instruction runs as compiled.
- **Addressing.** No new adder. The ports read the base into rs1 and the unit's offset is added into
  the ROM immediate's bits (the word's own fields are gated off at decode), so the core's ADDI form
  gives EA. At the trigger the same adder gives d - s (SUB form), which every overlap test uses.
- **Tokens.** One new family, M (16,384 rows), so the output code keeps its 5 family bits and the
  legacy ids do not move. Data uses the existing CW, EB, EC0, EC1, SC. See `hot.py`'s module doc.
- **Why state goes through tokens.** A value crosses a position only by climbing a sublayer, so loop
  state that a lower layer needs must come down in a token. The record counter j gives the position
  in a chunk (31 words, or 31 bytes in a byte loop); the cursor comes down in MC or CW between chunks.
- **Copy (any alignment, partial words, descending, long).**
  - 2 tokens a word: position a reads the source word and emits CW(lo); position b sets EA to the
    destination and emits CW(hi); the hi token stands on the next position, which writes the word
    from the two tokens (key from b).
  - The funnel takes two source words. They sit in two stash heads by word parity, so they survive a
    chunk boundary, and a byte mask merges the header's old word for partial words.
  - A chunk ends at 31 words, at a destination page edge, or before a partial last word. Its header
    is MC [PA] MH: PA allocates the page, MH's bit is the frame's value head, which the write needs
    from a token.
  - Copy-on-write words need nothing: a read goes through the normal slot / image path.
  - A descending copy (memmove, s < d < s + n) is the same with the index counting down. It never
    happens in the session; it is covered by the IR tests and the fuzzer.
- **eqlBytes.** 2 tokens a byte (EB(a[k]), then EC1), F0(0) at a mismatch. It has written nothing
  until then, so a byte outside every load region ends in MD and the original faults as compiled.
- **Byte loops (token estimate, JSON string scan, ASCII run, writeOneLine).** 1 token a byte. Each runs
  the loop's plain iterations and exits at the loop's own head, so the iteration that leaves runs as
  compiled. That is weaker than `circuits.md`'s model for the token estimate: whitespace bytes run
  the compiled path (1,972 of 35,335 bytes).
- **SHA-256.** The stream is the 8 state words, 64 x (W, a', e') and the 8 new state words: about 530
  tokens a block against 10,036 (M). Rings of a.. / e.. / W, written from the CW tokens, sit in L2's
  attention; the rounds are carry-save adders in the MLPs under the ROM's heads, closed by one adder,
  all linear in a step thermometer, so the constants K are free.
- **Pruning.** With units on, `core.build` leaves out the logic of uops no ROM word holds (`hot.rom_absent`:
  digit file, legacy macros, signed DIV / REM, MULH, SRA, ...). ROM-specialized, like the units. It
  removes more than all the units add.

## What costs time (M)

Executor v4 runs every MLP unit every step, active or not, and each barrier costs far more than
its terms. The first copy build (a separate 2-mode address adder, 2 stash heads above the funnel) went
from 14 to 16 phases and from 25,890 to 21,960 tok/s: more than the copy saved. The byte loops took it
to 21 phases (18,155 tok/s). After moving the address onto the core's adder, computing the offset in one
stage, carrying the unit's class in MP, and leaving nothing between the funnel and the exit, it is back
at 14 phases.

| build | units on | phases | slots | us / step |
|---|---|---|---|---|
| baseline | 0 | 14 | 707,664 | 38.63 |
| first copy | 2 | 16 | 793,264 | 45.54 |
| 10 units, first | 10 | 21 | 668,291 | 55.08 (first segment) |
| 10 units, 15 phases | 10 | 15 | 742,918 | 44.57 |
| 10 units, final | 10 | 14 | 741,150 | 40.56 |
| 11 units, final | 11 | 14 | 819,082 | 42.51 |

So the rule for a new unit is: it must not lengthen the chain from the registers to the exit.

## Checkpoint and headroom

| step | saved, measured (M) | emulator estimate (E, from reset) | layers | IR dims | heads | FFN units | exec phases | residual slots | hidden | checkpoint (E) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline, no units | 0 | 0 | 59 | 3,002 | 850 | 45,973 | 14 | 1137 | 3840 | 35.0 GB |
| + memmove, memcpy | 1,220,196 | 1,932,868 | 59 | 3,065 | 846 | 37,648 | 14 | 1159 | 3840 | 35.1 GB |
| + token estimate | 475,155 | 568,519 | 59 | 3,185 | 849 | 38,491 | 14 | 1214 | 3840 | 35.1 GB |
| + JSON string scan | 315,910 | 338,408 | 59 | 3,208 | 849 | 38,858 | 14 | 1258 | 3840 | 35.1 GB |
| + eqlBytes | 186,056 | 203,517 | 59 | 3,241 | 850 | 39,367 | 14 | 1277 | 3968 | 36.3 GB |
| + ASCII run | 101,102 | 127,907 | 59 | 3,252 | 850 | 39,482 | 14 | 1283 | 3968 | 36.3 GB |
| + realloc copy | 118,788 | 119,103 | 59 | 3,320 | 852 | 39,692 | 14 | 1289 | 3968 | 36.3 GB |
| + writeOneLine (2 loops) | 69,092 | 81,322 | 59 | 3,344 | 852 | 39,847 | 14 | 1304 | 3968 | 36.3 GB |
| + memset | 18,351 | 50,024 | 59 | 3,347 | 852 | 39,930 | 14 | 1307 | 3968 | 36.3 GB |
| + SHA-256 | 1,001,062 | 1,175,007 | 60 | 5,135 | 863 | 45,458 | 14 | 1427 | 4096 | 37.8 GB |

- **Layers:** the memory-only path uses none. SHA-256 takes one: its rings need query rows from the
  step thermometer, a write gate that is not linear in it (it has two cases) needs an MLP stage, so the
  rings land on L3 and push the ROM's heads up.
- **Hidden:** what grows the checkpoint here is residual slots, not layers. Each step of 128 is 3.3%,
  about 1.2 GB (E). The peak is DIV's 692 live dims (17 DIVU sites), lifted above the multiplier
  (`at_least(48)`); that alone took the build down one step. `hidden` is `merge.pick_hidden` on these
  slots; the merged config with text will differ a little (not merged here).
- **Heads and MLP room:** L5's old limit does not apply on this map. L18 is the only full MLP that
  matters, and it is below full since pruning. L24-L55's 32 MLPs are empty, for functions of registers
  and tokens only. L1, L2, L57-59 have attention slots.
- **Vocabulary:** +16,384 rows, about 0.13 GB at hidden 3968 (E).

## Where to stop

Read the two tables together: a unit pays if the session's GPU seconds fall, and the checkpoint column says what it
costs to keep. Step time is 39.8 to 40.6 us across steps 1-8, so about 0.3 us per 1,000 MLP units (M, a fit
over those steps), and a run's rate carries about 1% of noise.

- **No checkpoint cost beyond the M family (0.13 GB, E):** copy, token estimate, JSON string scan. Hidden stays
  3840, layers 59. They give most of the time saving: 405 -> 340 s.
- **One hidden step (+1.2 GB, E):** eqlBytes crosses it (1,277 slots; 3840 holds about 1,265); ASCII run and
  realloc copy then ride on it. These still pay in seconds: 340 -> 325 s.
- **The stop, on time:** writeOneLine (+0.45 us a step against 69k tokens: 325 -> 326 s) and memset
  (324.7 s) are break-even. Neither pays for its units within the noise; both are cheap and harmless. Below
  about 70k tokens a unit does not pay at these rates.
- **SHA-256 pays despite its size:** 1.0M tokens, 325 -> 298 s, for one layer, a second hidden step, 5,500 MLP
  units and +2 us a step, about +1.5 GB (E). If the checkpoint is what matters most, this is the one to weigh.
- **The size-first choice:** stop after the JSON string scan: 340 s and no growth. **The speed-first choice:**
  everything built here: 298 s and about 38 GB (E).
- **Not built, and not because they stop paying:** JSON escape scan (0.22M tokens, E), maskSecrets probe (0.30M, E),
  wyhash (0.24M, E) are each above the 70k line. The first needs a loaded bound and a path-dependent a3, the
  second exact leftovers in a4-a6, the third 64 x 64 multiplies. They were left for scope, and their
  saving is still the emulator's estimate.

## Verification

**Whole session, GPU (M):** above. `fxrun run --units`; a mismatch stops the run. Every step: 0 wrong tokens,
registers equal at every ecall (the replay asserts it), final memory 0 of 1,897,580 words differing, final registers
equal.

**fx in the residual, 11 units, 60 layers (M):** `fxrun tf`: 5,135 dims, 4,000 positions, 0 differing against
`ir.evaluate`.

**IR (M), `tests/test_rv_hot.py`:** 17 tests pass; a further sweep of 108 random programs (copies, memset, eqlBytes, byte loops), 98,004 positions, 0 wrong, exactness asserts on. Circuit against `ref.py`, teacher-forced, exactness asserts on,
on fx's own memmove / eqlBytes words and fx-form loops, with every changed word read back through loads
at the end of the program, so a wrong register or a wrong word becomes a wrong token. (The read-back
exists because the session found what the first tests could not see: a key left on J cleared a0 after a
matching eqlBytes.)

**Fuzz against stepping the original instructions on fx's image (M), `fx/circ/fuzz2.py`:** random
machine states at the trigger: registers, written words, which pages have frames, frames left. The
reference with the unit against the plain paged machine stepping the original. Compared: fault, PC,
every live register, every word either run wrote except the C stack. Covered: page crossings, unmapped
pages, image-backed words, overlap both ways, zero length, every alignment, the 16 KB and window
edges, the frame budget's edge. A loop unit is compared at its exit head against the iterations it
stands for. "fired" counts records started, including those that ended in MD.

| unit | cases | fired | declined | faulted (both ways) | errors | tokens: unit / original (fired cases) |
|---|---|---|---|---|---|---|
| ASCII run | 100,000 | 43,674 | 56,326 | 0 | 0 | 34,292,950 / 238,189,577 |
| realloc copy | 100,000 | 82,232 | 17,768 | 261 | 0 | 101,382,090 / 350,495,828 |
| writeOneLine, newline | 100,000 | 85,918 | 14,082 | 0 | 0 | 7,123,541 / 38,148,090 |
| writeOneLine, CR | 100,000 | 85,918 | 14,082 | 0 | 0 | 7,123,541 / 38,148,090 |
| eqlBytes | 100,000 | 88,201 | 11,799 | 20,205 | 0 | 24,622,671 / 102,443,095 |
| JSON string scan | 100,000 | 35,308 | 64,692 | 0 | 0 | 777,619 / 10,174,332 |
| memcpy | 100,000 | 68,536 | 31,464 | 479 | 0 | 10,505,799 / 46,785,014 |
| memmove | 100,000 | 79,502 | 20,498 | 421 | 0 | 14,134,380 / 74,883,806 |
| memset | 100,000 | 86,222 | 13,778 | 387 | 0 | 15,111,903 / 20,242,049 |
| SHA-256 | 100,000 | 89,558 | 10,442 | 8,033 | 0 | 83,720,314 / 882,330,389 |
| token estimate | 100,000 | 68,426 | 31,574 | 0 | 0 | 3,477,917 / 49,276,837 |


The fuzzer found one real difference: eqlBytes could fire with no frames left, where the original
faults on its own stack frame. It now declines near the frame limit (34 frames), as the copies do,
and that row is the rerun.

**Existing suites (M):** `tests/test_rv_hot.py` + `tests/test_rv_circ.py`: 68 passed; `tests/test_rv_hot.py` + `tests/test_rv_mem.py`: 83 passed, 10 skipped. `test_fuzz_free_running_cpu` left to the user as asked.
63 programs free-running on Executor v4 (no units, `small_cfg`): 63 of 63 identical to the reference, 6,424,294 tokens, 0 differing (M).

## Honesty verdicts

| unit | built | verdict | what a skeptic should know |
|---|---|---|---|
| memmove, memcpy | yes | exact | Fires on all 9,029 session calls. Caller-saved registers stay stale (dead by the ABI). The C stack is assumed outside window 0: asserted in the reference, not checked by the circuit (a fourth register port) |
| memset | yes | exact | 2 tokens a word, against 2.7 for the original: little saved. A 1-token path from the register was not built |
| eqlBytes | yes | exact | Byte-wise, not word-wise |
| token estimate | yes | exact, narrower than E | Only runs of non-whitespace. The whitespace path stays compiled |
| JSON string scan | yes | exact | Declines if the cursor could lie in the scanned text, by a margin of 40 bytes |
| ASCII run, writeOneLine | yes | exact | Exit at the head; the leaving iteration runs compiled |
| realloc copy | yes | exact | n comes down in a token: 1 token more than a memcpy |
| SHA-256 | yes | exact, one layer | Declines unaligned message blocks (16 of 122 in the session) |
| pruning | yes | ROM-specialized | Exact on this image only. A rebuilt fx needs a rebuild of the circuit, as the PCs do |
| maskSecrets probe | **no** | emulator estimate only | Registers a4-a6 at its exit depend on the last prefix compared |
| wyhash | **no** | emulator estimate only | Needs 64 x 64 -> 128 through the multiplier in 4 passes |
| JSON escape scan | **no** | emulator estimate only | Needs len from memory and a path-dependent a3 |

## Blockers and what is left

1. **3 minutes for turn 1 is not reached:** 3.5 min from the snapshot (M). What is left of a turn is flat glue (`circuits.md`
   item 6): the JSON scanner's state machine, the allocator, prologues. The three unbuilt units are
   worth about 7% more (E). The token format's 16-bit values (`speedups.md` §7) would act on all of it.
2. **Three units are not built:** maskSecrets probe, wyhash, JSON escape scan. Their figures are the
   emulator's.
3. **The C stack outside window 0** is assumed by the copy units and asserted by the reference, not
   checked by the circuit. It holds by fx's link map; a fourth register port would check it.
4. **Dead registers** are dead by the ABI and by `circuits.md`'s liveness, checked per run by its poison
   mode on rv32emu's units, not proven. The twins used here for tok2, chr2, asc2 were not run in poison
   mode; their dead sets are empty, and the fuzzer compares every register but the loop's own dead ones.
5. **Hugging Face path:** the unit circuit has not been merged or run on the HF kernels. Checkpoint
   and hidden figures are estimates; the merged config with the bridge's text will differ.
6. **fx at IR scale:** the IR has run the units on the test map (21 layers), and on fx's map only through
   the 4,000-position residual check under Verification; the rest of fx's session is checked by Executor v4 against the reference.
7. **The units are specialized to this image.** PCs and the pruned uop list come from it; a rebuilt fx
   needs them regenerated. `hot.fx_units` asserts the words at memmove's, eqlBytes' triggers only.

## Reproduce

Paths as in `memory.md`. `S` = the scratchpad.

1. rv32emu: `zig cc -target x86_64-windows-gnu -O2 -shared fx/rv/rv32emu.c -o rv32emu.dll`.
2. A log with the twins on from boot's end: `python -I fx/rv/snaplog.py rv32emu.dll ELF imports.json LOG.npz
   --opt ... --units move,set,eql,bcopy,jscan,sha,tok2,chr2a,chr2b,asc2`.
3. The session: `python -m chipformer.rv.fxrun LOG.npz ELF run --units all+ --out r.json` (`ref` for the
   reference alone; `all` leaves SHA out).
4. Fuzz: `python fx/circ/fuzz2.py r24log.npz ELF out.json --unit NAME --cases 100000`.
5. Tables: `python fx/circ/report2.py RUNS OUT.json`.
