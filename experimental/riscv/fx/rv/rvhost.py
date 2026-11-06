"""Runs the RV32 build of fx-core (wasm2c, see build_rv.sh) in rv32emu and drives it
with exactly the host and scripted conversation of ../host.py. Every executed RV32
instruction is counted, by mnemonic and by PC.

Usage: python -I rvhost.py RV32EMU_DLL FX_ELF IMPORTS_JSON OUT_JSON [--turns N]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import struct
import sys
import tempfile
import time

from elftools.elf.elffile import ELFFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import host as H  # noqa: E402
import tinyrepo  # noqa: E402

LIN = 0x02000000
STACK_TOP = 0x100000  # __stack_pointer initial value (stack first, data at 1 MB)
HEAP_BASE = 0x149120  # end of the data segments
MNEMONICS = ("LUI AUIPC JAL JALR BEQ BNE BLT BGE BLTU BGEU LB LH LW LBU LHU SB SH SW ADDI SLTI SLTIU XORI ORI "
             "ANDI SLLI SRLI SRAI ADD SUB SLL SLT SLTU XOR SRL SRA OR AND FENCE ECALL EBREAK MUL MULH MULHSU MULHU "
             "DIV DIVU REM REMU SH1ADD SH2ADD SH3ADD LR.W SC.W CSR").split()


class RvHost(H.Host):
    def __init__(self, dll, elf_path, imports, repo, turns=2, log=None):
        self.repo = H.RepoTools(repo)
        self.turns = turns
        self.log = log or (lambda *a: None)
        self.instrumented = False
        self.stdin = bytearray()
        self.stdout = bytearray()
        self.events = []
        self.import_calls = {}
        self.peak_mem = 0
        self.mem = None
        self.clock_ns = 1_767_000_000_000_000_000
        self.rand = 0x12345678
        self.next_id = 1
        self.pending = {}
        self.state = "init"
        self.turn = 0
        self.model_calls = 0
        self.streams = {}
        self.snapshots = []
        self.messages_out = {}
        self.request_bytes = []
        self.exit_code = None
        self.env = [f"FX_PROVIDER={H.PROVIDER}", f"FX_CONFIGURED_PROVIDERS={H.PROVIDER_JSON}", f"FX_MODEL={H.MODEL}"]
        self.args = ["fx", "acp"]
        self.new_method = "libfx/new"

        e = dll
        e.rv_mem.restype = ctypes.c_void_p
        e.rv_regs.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_get_pc.restype = ctypes.c_uint32
        e.rv_set_pc.argtypes = [ctypes.c_uint32]
        e.rv_icount.restype = ctypes.c_uint64
        e.rv_misaligned.restype = ctypes.c_uint64
        e.rv_ops.restype = ctypes.POINTER(ctypes.c_uint64)
        e.rv_pccounts.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_mispc.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_run.argtypes = [ctypes.c_uint64]
        e.rv_fault_addr.restype = ctypes.c_uint32
        self.e = e
        f = open(elf_path, "rb")
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        self.text_lo, self.text_hi = text["sh_addr"], text["sh_addr"] + text["sh_size"]
        assert e.rv_init(self.text_lo, self.text_hi) == 0
        assert e.mt_init() == 0
        e.mt_set_watch(0, STACK_TOP)  # wasm shadow stack: [0, 1 MB), grows down
        e.mt_host_store.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        e.mt_phases.restype = ctypes.POINTER(ctypes.c_uint64)
        e.mt_run_hist.restype = ctypes.POINTER(ctypes.c_uint64)
        self.phase = 0
        self.phase_names = ["instantiate (data segments)", "boot (_start to first stdin wait)"]
        self.base = e.rv_mem()
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD" and seg["p_filesz"]:
                data = seg.data()
                ctypes.memmove(self.base + seg["p_vaddr"], data, len(data))
        e.rv_set_pc(elf.header["e_entry"])
        sym = elf.get_section_by_name(".symtab")
        self.linear_bytes_addr = sym.get_symbol_by_name("rv_linear_bytes")[0]["st_value"]
        self.funcs = sorted((s["st_value"], s["st_size"], s.name) for s in sym.iter_symbols()
                            if s["st_info"]["type"] == "STT_FUNC" and s["st_size"])
        self.imports = imports
        self.regs = e.rv_regs()
        self.handlers = []
        for imp in imports:
            nm = ("w_" if imp["module"].startswith("wasi") else "x_") + imp["name"]
            fn = getattr(self, nm, None) or self._make_unavailable(imp["module"], imp["name"])
            self.handlers.append((imp, fn))

    # memory and counters
    def fuel(self):
        return self.e.rv_icount()

    def rd(self, p, n):
        return ctypes.string_at(self.base + LIN + p, n)

    def wr(self, p, b):
        ctypes.memmove(self.base + LIN + p, bytes(b), len(b))
        self.e.mt_host_store(LIN + p, len(b))

    def mark(self, kind, detail=None):
        super().mark(kind, detail)
        if kind == "boot_end":
            self.next_phase("setup (initialize, libfx/new, prompt 1 intake, request 1 build)")
        elif kind == "stream_open":
            self.next_phase(f"model call {detail + 1} (stream, tool, next request)")

    def next_phase(self, name):
        self.phase += 1
        self.phase_names.append(name)
        self.e.mt_phase(self.phase)

    def linear_bytes(self):
        return struct.unpack("<I", ctypes.string_at(self.base + self.linear_bytes_addr, 4))[0]

    def hostcall(self, ident, argp, nwords):
        imp, fn = self.handlers[ident]
        words = struct.unpack(f"<{nwords}I", ctypes.string_at(self.base + argp, 4 * nwords)) if nwords else ()
        args, i = [], 0
        for t in imp["params"]:
            if t in ("u64", "s64"):
                args.append(words[i] | (words[i + 1] << 32))
                i += 2
            else:
                v = words[i]
                args.append(v - (1 << 32) if t == "s32" and v >= 1 << 31 else v)
                i += 1
        self.import_calls[imp["name"]] = self.import_calls.get(imp["name"], 0) + 1
        self.peak_mem = max(self.peak_mem, self.linear_bytes())
        r = fn(*args)
        return 0 if r is None else r & 0xFFFFFFFF

    def run(self):
        t0 = time.perf_counter()
        self.mark("start")
        while True:
            r = self.e.rv_run(1 << 40)
            pc = self.e.rv_get_pc()
            if r == 1:
                a7 = self.regs[17]
                if a7 >= 0x400:
                    try:
                        self.regs[10] = self.hostcall(a7 - 0x400, self.regs[10], self.regs[11])
                    except SystemExit:
                        break
                    self.e.rv_set_pc(pc + 4)
                    continue
                if a7 == 95:
                    self.next_phase(self.phase_names[1])
                    self.phase_names.pop()
                    self.e.rv_set_pc(pc + 4)
                    continue
                if a7 == 93:
                    self.mark("returned", self.regs[10])
                    break
                if a7 == 94:
                    self.mark("trap", f"wasm trap {self.regs[10] - 1000}")
                    break
                self.mark("trap", f"unknown ecall {a7}")
                break
            word = struct.unpack("<I", ctypes.string_at(self.base + pc, 4))[0] if pc < 0x04000000 else 0
            self.mark("trap", f"rv_run={r} pc={pc:#x} insn={word:#010x} fault={self.e.rv_fault_addr():#x} sym={self.sym(pc)}")
            break
        self.wall = time.perf_counter() - t0
        self.peak_mem = max(self.peak_mem, self.linear_bytes())

    def sym(self, pc):
        import bisect
        i = bisect.bisect_right(self.funcs, (pc, 1 << 40, "")) - 1
        if i >= 0 and self.funcs[i][0] <= pc < self.funcs[i][0] + self.funcs[i][1]:
            return self.funcs[i][2]
        return "?"

    def profile(self, which="pc"):
        n = (self.text_hi - self.text_lo) // 4
        ptr = self.e.rv_pccounts() if which == "pc" else self.e.rv_mispc()
        arr = (ctypes.c_uint32 * n).from_address(ctypes.addressof(ptr.contents))
        counts = memoryview(arr).cast("B").cast("I")
        per = {}
        for addr, size, name in self.funcs:
            lo = (addr - self.text_lo) // 4
            hi = lo + size // 4
            if lo < 0 or hi > n:
                continue
            s = sum(counts[lo:hi])
            if s:
                per[name] = per.get(name, 0) + s
        return per


PH_FIELDS = ["icount_start", "icount_end", "loads", "stores", "bytes_loaded", "bytes_stored", "words_written",
             "lines_written", "words_read_ro", "words_read_rw", "new_words_written"]


def memory_stats(h):
    e = h.e
    p = e.mt_phases()
    phases = []
    for k in range(h.phase + 1):
        row = {f: p[k * len(PH_FIELDS) + i] for i, f in enumerate(PH_FIELDS)}
        row["name"] = h.phase_names[k]
        row["rv_instructions"] = row.pop("icount_end") - row.pop("icount_start")
        phases.append(row)
    rh = e.mt_run_hist()
    out = (ctypes.c_uint64 * 32)()
    e.mt_summary(out)
    pages = (ctypes.c_uint32 * 512)()
    e.mt_page_hist(pages)
    return {
        "written_words_per_64k_page": {hex(i << 16): pages[i] for i in range(512) if pages[i]},
        "phases": phases,
        "store_run_hist_bytes_log2": {f">={1 << i}": rh[i] for i in range(24) if rh[i]},
        "distinct_words_written_ever": out[0], "distinct_words_written_boot": out[1],
        "distinct_words_written_after_boot": out[2], "distinct_words_written_both": out[3],
        "words_nonzero_at_end_among_written": out[4], "distinct_lines128_written_ever": out[5],
        "line_write_episodes_closed_by_read": out[6], "line_episodes_fully_overwritten_before_read": out[7],
        "lines_with_a_full_overwrite_episode": out[8], "line_episodes_open_at_end": out[9],
        "min_store_offset": out[10], "max_store_end_offset": out[11], "min_stack_store_offset": out[12],
        "host_written_bytes": out[13],
        "word_write_count_hist_log2": {f">={1 << i}": out[16 + i] for i in range(16) if out[16 + i]},
        "stack_reserved_bytes": STACK_TOP, "stack_high_water_bytes": STACK_TOP - out[12],
        "heap_base": HEAP_BASE, "heap_high_water_bytes": out[11] - HEAP_BASE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("elf")
    ap.add_argument("imports")
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=2)
    a = ap.parse_args()
    dll = ctypes.CDLL(os.path.abspath(a.dll))
    tmp = tempfile.mkdtemp(prefix="fxrepo-")
    tinyrepo.make_repo(tmp)
    h = RvHost(dll, a.elf, json.load(open(a.imports)), tmp, a.turns, log=lambda *x: print(*x, file=sys.stderr))
    h.run()
    nops = dll.rv_nops()
    opsv = h.e.rv_ops()
    ops = {MNEMONICS[i]: opsv[i] for i in range(nops) if opsv[i]}
    res = {"elf": os.path.abspath(a.elf), "rv_instructions_total": h.fuel(), "peak_linear_memory_bytes": h.peak_mem,
           "wall_s": h.wall, "events": h.events, "import_calls": h.import_calls, "model_requests": h.request_bytes,
           "dynamic_ops": ops, "misaligned_accesses": h.e.rv_misaligned(), "per_function": h.profile(), "misaligned_by_function": h.profile("mis"),
           "final_repo": {r: open(os.path.join(tmp, r), encoding="utf-8").read() for r in h.repo.files()
                          if r.startswith("tinycalc/calc") or r.startswith("tests/test_div")}}
    res["memory_access"] = memory_stats(h)
    json.dump(res, open(a.out, "w"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: res[k] for k in ("rv_instructions_total", "peak_linear_memory_bytes", "wall_s", "misaligned_accesses")}))
    for ev in h.events:
        if ev[1] != "deliver":
            print(ev[0], ev[1], str(ev[2])[:200])


if __name__ == "__main__":
    main()
