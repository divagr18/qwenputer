# fx's memory in the weights: the paged map

Status 2026-11-07, branch `fx-mem`. Code: `chipformer/rv/spec.py` (`MemCfg`), `ref.py`
(`PagedMachine`), `core.py` (`build(cfg=...)`, `dense_plan`), `exec4.py` (banks), `fxrun.py`,
`memcheck.py`, `fx/rv/snaplog.py`, `tests/test_rv_mem.py`. Raw numbers: `memory.json`.
**M** = measured, **E** = estimated.

The legacy map and the rv-q4p2 circuit are unchanged: the paged path is `core.build(cfg=...)`, and
new tokens sit after the legacy ids.

## What fx needs (M, row 24: fx's own tools on the CPU)

`fx-fastcopy.elf`, ReleaseFast, -O2, aligned-only, RV32IM+Zba; session as in `speed-iter.md`.

| region | size | note |
|---|---|---|
| .text | 6,973,628 B, 1,743,407 words | at 0x02000000 |
| .rodata | 353,756 B | right after .text |
| .bss | 131,208 B | the funcref table; 1,503 words nonzero after boot |
| boot-written words | 98,004 words (383 KB) in linear memory, 0xD9270..0x1A00B0 | |
| words written after boot | 63,225 (247 KB), spread over 3 MB of addresses | linear 0xA8540..0x300000; C stack 2,819 words; .bss 7 words |
| C stack depth | 11.5 KB | P0's trace did not cover it |
| pages written after boot | 704 of 512 B (352 KB) | 396 at 1 KB; 2,411 at 128 B |
| pages written ever | 1,458 of 512 B (729 KB) | from reset |
| words read after boot that were not written after boot | 3,130 in linear memory | |

The hard number is the address spread: 247 KB of words over 3 MB. Direct-mapped word slots would
need 192 layers (E: 3 MB at 16 KB a layer). A dense store cannot take the writes either.

## The design

**Why word slots stay the writable unit.** A store's new slot must come from dims below the
writing layer at t-1, or from tokens; the slot's own read is above it. So a slot holding more than
one word needs its neighbours through the token stream, on every store. Nothing cheaper exists:
no pure-add writes (q/k are L2-normalized), and a 2-hot key mixes rows. 14% of fx's instructions are
stores, so word slots keep SW at 1 token. Density goes where nothing is written.

**What tokens add: a path down.** A token is seen by every layer at t, so a decision taken at the
top at t-1 reaches any layer at t. That is what makes allocation possible at all: the frame number
goes down in a `PA` token.

| region | structure | capacity (fx config) | layers |
|---|---|---|---|
| ROM | dense byte rows + spill tables, key head = 32 KB | 1.835M words (7.34 MB) | 15 |
| image | the same layout over 32 KB windows, read-only | 35 windows (1.12 MB) | shared with RAM |
| page tables | one slot per page, 4 windows | 4 MB + 3 x 256 KB of address space | 3 |
| RAM frames | word slots: a value head is a 512-byte frame | 1,023 frames (524 KB) | 32 |

**ROM and image: the dense layout.** A byte read needs a 2-hot query (scale 2^-4, exact), so every
read pairs a data row with an all-zero ballast row Z (row 127). A data row holds 31 words: 124 of
its 126 data columns. That is 504 KB a layer against today's 128 KB.
- 31 slots and 127 rows do not bit-slice. Dividing the PC by 31 would take 2 layers and an adder, on
  the link and JALR paths too.
- So addresses stay bit-sliced (32 slots, 128 rows), and the words that do not fit, slot 31 and row
  127, go to smaller tables of the same kind (`core.dense_plan`, recursive). Their gates are address
  bits, so no mux is needed: exactly one table holds a word, and the others read 0 into the same
  dims. Nothing is added to the fetch depth.
- A base table has 16 words a row, no spill.

**RAM: demand paging.** Frames are handed out in order. A store to an unmapped page puts `PA(f)`
before its record, with f = the allocator head + 1. At the PA token:
- the page table's key comes from t-1 (the store's EA), the value from the token, and beta from
  the token and the EA at t (the record has not committed, so the EA is still there);
- the table is read after the write at the same position, so the store then finds its frame.

A store's key comes from the frame bits at t-1 and its value head from the token (`ST0/1`,
`SS0/1`: the frame's low bit), as the legacy RAM uses EA[6]. Running out of frames faults (`store
out of range`). Nothing wraps silently.

**Copy on write.** A slot's valid column says it was written. A load takes the slot if valid, else
the image word, in one MLP stage with no extra layer (the image's byte-to-bit units are gated by
`1 - valid`). So the boot image sits in dense read-only rows, and only pages written after boot
take frames: 704 instead of 1,458.

**PC.** 21 bits. The low 16 stay in the `pc` head. A new `pch` head holds the high bits.
- `J` keeps `pch`; a jump that leaves the 64K-word window is `V(lo) JF(hi)`: V carries bits 0..14, so
  there is no second 65K-row family.
- A sequential crossing is `RF`: one token (the low bits past a crossing are at most 62).
- JAL holds a signed word offset, added by the branch adder: a 21-bit target did not fit the word.

**Why not the alternatives.**

| option | layers for fx | why not |
|---|---|---|
| direct-mapped word slots | 192+ | the address spread |
| set-associative word slots with tags | about 24 | every load reads and compares all ways of a set |
| 2- or 3-word slots (`speedups.md` §12) | 64 / 43 for 2 MB | SW costs 3 / 5 tokens |
| KV cache (`kvram.md`) | | rejected on the HF path |
| paged word slots + image (built) | 35 | one row read per load |

**Disclosures.**
- The production config uses a boot snapshot: linear memory at boot's end is the image (head `init`),
  taken at the first stdin read, before any user input. A from-reset config needs 2,048 frames
  (90 layers): `fxrun --from-reset`.
- The image and the frame budget are sized on this program. The image windows are built from its
  snapshot; 1,023 frames is 1.45x the 704 this session mapped. A longer session can run out, and then
  faults.
- The macro units (MMOVE / MEQL) do not run on the paged map; `circuits2.md` has the units that do. Their limit (both buffers below 0x7F80)
  already excludes fx's buffers, so nothing changes here; widening them is separate work.
- Between ecalls the machine runs alone. At an ecall the host serves the import from a recorded log
  and resumes the CPU, as an OS would, by writing registers and memory into the state. The real I/O
  path is the bridge's, not part of this work.

## Results

**fx's whole session runs on Executor v4 with 0 differences**, both from the boot snapshot and from
reset (M, NVIDIA L4). The GPU generates every token between ecalls, and each is compared with the
reference machine's; a wrong token stops the run. The reference's registers are compared with
rv32emu's at every ecall, and its memory on all nonzero words at the end. rv32emu's memory is not
read back from the GPU: the GPU is tied to it through the token stream.

| whole session | from the boot snapshot | from reset (no snapshot) |
|---|---|---|
| layers | 59 | 90 |
| frames provisioned / pages mapped | 1,024 / 704 | 2,048 / 1,458 |
| tokens the GPU generated | 10,508,989 | 11,415,547 |
| instructions | 7,174,817 | 7,701,699 (rv32emu's count) |
| segments (between ecalls) | 400 | 412 |
| wrong tokens | 0 | 0 |
| final memory differing from rv32emu | 0 of 1,897,586 words | 0 of 1,897,586 words |
| final registers | equal | equal |
| GPU time | 405.91 s | 442.25 s |
| us / step | 38.63 | 38.74 |
| **tok/s** | **25,890** | **25,812** |
| **instr/s** | **17,676** | **17,415** |

GPU time counts the launches only. It leaves out the reference machine beside it (34.94 s /
37.1 s of Python, only there for the comparison) and the state pokes at ecalls. The
from-reset run includes boot: fx boots in the weights with no snapshot, on the larger config. The
snapshot run is the production config. Its starting state is rv32emu's at boot's end, not the GPU's
own: the two runs were not compared with each other at that point.

| other checks | result | |
|---|---|---|
| fx residual against `ir.evaluate`, 59 layers | 2,987 dims, 4,000 positions, 0 differing | M |
| fx in the IR, teacher-forced, 59 layers | 57,344 positions predicted, 0 wrong, exactness asserts on the first 4,000 (first segment only: about 55 tok/s, so the session would take about 53 h, E) | M |
| existing programs, residual (paged circuit, 14 layers) | 63 programs, 64,690 positions, 0 differing | M |
| existing programs, free-running | 63 / 63 identical, 6,424,294 tokens, 5,626,517 instructions (MicroPython s1, s5 included) | M |
| big map: far jumps, windows, bank edges | 2 programs, residual and free-running exact | M |
| `tests/test_rv_mem.py` | 66 passed, 10 skipped (the long programs; they are in the free-running row) | M |
| legacy suites | 287 passed, 14 skipped, 1 failed. `test_fuzz_free_running_cpu` fails the same way on clean `fx-int` (`KeyError: 'Q'` in `fastexec.py`) | M |
| HF merge of the paged circuit | not attempted | - |

**Speed against memory layers** (M, fx from the boot snapshot, 902,180 tokens on the L4):

| frames | layers | us / step | tok/s | wrong |
|---|---|---|---|---|
| 128 | 31 | 39.04 | 25,616 | 0 |
| 256 | 35 | 38.92 | 25,696 | 0 |
| 512 | 43 | 38.9 | 25,709 | 0 |
| 1024 | 59 | 38.95 | 25,674 | 0 |
| 2048 | 91 | 38.86 | 25,732 | 0 |

Flat from 31 to 91 layers. `exec-v4.md` estimated 54-67 us for fx's memory.

**Banks on and off** (M, c-sieve, 20,000 tokens, best of 3):

| frames | layers | banks | slots | shared memory (B) | us / step | tok/s |
|---|---|---|---|---|---|---|
| 8 | 11 | on | 667,588 | 56,944 | 39.49 | 25,323 |
| 8 | 11 | off | 748,891 | 64,432 | 40.64 | 24,603 |
| 16 | 11 | on | 667,636 | 56,944 | 39.12 | 25,565 |
| 16 | 11 | off | 774,555 | 64,448 | 40.41 | 24,746 |
| 32 | 11 | on | 667,644 | 56,960 | 39.26 | 25,474 |
| 32 | 11 | off | 827,571 | 64,480 | 40.72 | 24,558 |
| 64 | 12 | on | 667,676 | 56,992 | 39.56 | 25,277 |
| 64 | 12 | off | 938,347 | 65,232 | 41.04 | 24,369 |
| 128 | 14 | on | 667,708 | 57,024 | 39.92 | 25,048 |
| 128 | 14 | off | 1,182,573 | 75,936 | 42.58 | 23,483 |

The same plan at full size, 59 layers (M, plan sizes, no timing):

| | banks | no banks | today's 11 layers |
|---|---|---|---|
| terms | 702,768 | 6,142,509 | 966,054 |
| value-head records | 62 | 1684 | |
| shared memory | 61,152 B | 175,760 B (limit 99 KB: does not fit) | 67 KB |

**Size.**

| | value | |
|---|---|---|
| layers | 59 (11 today); 90 from reset | M |
| IR dims | 2,987 | M |
| heads, MLP units | 850, 45,973 | M |
| exec4 state | 110 MB | M |
| vocabulary | +6,208 rows | M |
| checkpoint | about 35 GB, +16.5 GB: 48 layers at about 0.34 GB, scaled from rv-tiny's 244 MB a layer at hidden 2816 | E |
| HF merge RAM | about 58 GB against the box's 60 GB, scaled from 31 GB | E |

**Tokens** (M, whole session from the snapshot): 1.465 tokens per instruction. `JF` 55,364 (7.9% of
jumps, 0.5% of tokens), `PA` 692, `RF` 0.

**fx turn 1** (M): 5,649,313 instructions and 8,350,275 tokens, counted on
the reference at segment boundaries (rv32emu: 5,648,040 instructions), 1.478 tokens per instruction.
At the session's 25,890 tok/s that is **323 s, 5.4 min**. The
time is tokens / measured rate, because the GPU run did not log a clock at the turn's edges. It
counts neither the host's time at ecalls nor Qwen's text. This is row 24 with no macro units.

## Exact zeros added to `exec-v4.md`'s list
7. A bank head that the address does not select has an all-zero key and query: it neither writes nor
   reads. Exec4 computes the selected head and row from binary forms over the same address bits.
   `test_rv_mem.check_bank_forms` asserts that they fire exactly the rows of the 128 channels, at
   every position, and the residual checks compare every dim.

## Blockers
1. Real I/O: imports are served from a recorded log, and each resume pokes registers and memory into
   the state. That is the bridge's work. Every number above is the CPU's, with no host time.
2. The frame budget is 1.45x one short session; a longer one faults when frames run out.
3. HF: the paged circuit has not been merged or run on the HF kernels at fx size. The IR's kernel
   emulation covers the small and big test configs only on traces under 1,500 tokens. The results
   above are Executor v4's, the IR-equivalent path.
4. The IR has checked fx over only the positions above; the rest of the session is checked by
   the reference.
5. MMOVE / MEQL do not run on this map. Their successors do: `circuits2.md`.
