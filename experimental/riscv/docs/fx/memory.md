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
- The macro units (MMOVE / MEQL) do not run on the paged map. Their limit (both buffers below 0x7F80)
  already excludes fx's buffers, so nothing changes here; widening them is separate work.
- Between ecalls the machine runs alone. At an ecall the host serves the import from a recorded log
  and resumes the CPU, as an OS would, by writing registers and memory into the state. The real I/O
  path is the bridge's, not part of this work.
