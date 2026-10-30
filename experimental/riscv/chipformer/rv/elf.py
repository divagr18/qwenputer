"""Load a static RV32 ELF (gcc, env/rom.ld or env/qemu.ld) into an asm.Program or an Rv32."""

import struct
from pathlib import Path

from chipformer.rv import spec as S
from chipformer.rv.asm import Program


def segments(path) -> tuple[int, list[tuple[int, bytes]]]:
    """(entry, [(vaddr, bytes incl. zero-filled bss)]) of the PT_LOAD segments."""
    b = Path(path).read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, "ELF32 only"
    entry, phoff = struct.unpack_from("<II", b, 0x18)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    segs = []
    for i in range(phnum):
        p_type, off, vaddr, _, filesz, memsz, _, _ = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type == 1 and memsz:
            segs.append((vaddr, b[off:off + filesz] + bytes(memsz - filesz)))
    return entry, segs


def program(path) -> Program:
    """A ROM-map ELF as a Program: code words from ROM_BASE, read-only data from DROM_BASE,
    data bytes from RAM_BASE, the weight memory's cells from WROM_BASE."""
    entry, segs = segments(path)
    regions = {"rom": (S.ROM_BASE, 4 * S.ROM_WORDS), "drom": (S.DROM_BASE, 4 * S.DROM_WORDS),
               "ram": (S.RAM_BASE, 4 * S.RAM_WORDS),
               "wrom": (S.WROM_BASE, max(S.WROM_CELLS * S.WROM_WORDS, 72 * S.VX_CHUNKS))}
    bufs = {k: bytearray() for k in regions}
    for vaddr, blob in segs:
        for k, (base, size) in regions.items():
            if base <= vaddr and vaddr + len(blob) <= base + size:
                b = bufs[k]
                end = vaddr - base + len(blob)
                b += bytes(max(0, end - len(b)))
                b[vaddr - base:end] = blob
                break
        else:
            raise AssertionError(f"segment at {vaddr:#x} (+{len(blob)}) outside ROM/DROM/RAM")
    text = bufs["rom"] + bytes(-len(bufs["rom"]) % 4)
    words = [int.from_bytes(text[i:i + 4], "little") for i in range(0, len(text), 4)]
    return Program(words, bytes(bufs["ram"]), {"_start": entry}, data_base=S.RAM_BASE,
                   rodata=bytes(bufs["drom"]), wrom=bytes(bufs["wrom"]))


def load_into(cpu, path) -> None:
    """Load any static ELF into a ref.Rv32 (flat memory) and set its PC."""
    entry, segs = segments(path)
    for vaddr, blob in segs:
        for i, x in enumerate(blob):
            cpu.mem[vaddr + i] = x
    cpu.pc = entry
