"""rv32emu with an fx image loaded but fx not running: for differential fuzzing of the units
(fuzz.py) and for probing what the original code leaves behind."""
from __future__ import annotations

import ctypes
import os
import struct
import sys

from elftools.elf.elffile import ELFFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as RUN  # noqa: E402
import units as U  # noqa: E402

SENTINEL = 0x07000000  # an ebreak the code under test returns to
EBREAK = 0x00100073


class Machine:
    def __init__(self, dll_path, elf_path, units="all"):
        self.e = RUN.setup_dll(ctypes.CDLL(os.path.abspath(dll_path)))
        e = self.e
        e.rv_mem.restype = ctypes.c_void_p
        e.rv_regs.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_get_pc.restype = ctypes.c_uint32
        e.rv_set_pc.argtypes = [ctypes.c_uint32]
        e.rv_run.argtypes = [ctypes.c_uint64]
        e.rv_icount.restype = ctypes.c_uint64
        f = open(elf_path, "rb")
        elf = ELFFile(f)
        t = elf.get_section_by_name(".text")
        self.text_lo, self.text_hi = t["sh_addr"], t["sh_addr"] + t["sh_size"]
        assert e.rv_init(self.text_lo, self.text_hi) == 0
        self.base = e.rv_mem()
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD" and seg["p_filesz"]:
                d = seg.data()
                ctypes.memmove(self.base + seg["p_vaddr"], d, len(d))
        self.regs = e.rv_regs()
        self.specs, self.skipped = U.specs_for(elf_path, units)
        self.ids = RUN.install(e, self.specs)
        self.by_label = {s["label"]: (s, i) for s, i in zip(self.specs, self.ids)}
        self.w32(SENTINEL, EBREAK)

    # memory
    def wr(self, a, b):
        ctypes.memmove(self.base + a, bytes(b), len(b))

    def rd(self, a, n):
        return ctypes.string_at(self.base + a, n)

    def w32(self, a, v):
        self.wr(a, struct.pack("<I", v & 0xFFFFFFFF))

    def r32(self, a):
        return struct.unpack("<I", self.rd(a, 4))[0]

    def get_regs(self):
        return [self.regs[i] for i in range(32)]

    def set_regs(self, x):
        for i in range(1, 32):
            self.regs[i] = x[i] & 0xFFFFFFFF

    def check(self, pc):
        """One verified firing at pc: (status, steps, original tokens, unit tokens, fail info)."""
        self.e.rv_set_pc(pc)
        out = (ctypes.c_uint64 * 4)()
        st = self.e.rv_unit_check(out)
        f = self.e.rv_unit_fail()
        return st, out[0], out[1], out[2], [f[i] for i in range(8)]

    def run_original(self, pc, limit=10**7):
        """Run the original code from pc (units off) until it reaches the sentinel ebreak."""
        self.e.rv_units_mode(0)
        self.e.rv_set_pc(pc)
        r = self.e.rv_run(limit)
        return r, self.e.rv_get_pc()
