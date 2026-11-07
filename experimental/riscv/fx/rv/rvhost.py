"""Runs the RV32 build of fx-core (wasm2c, see build_rv.sh) in rv32emu and drives it
with exactly the host and scripted conversation of ../host.py. Every executed RV32
instruction is counted, by mnemonic and by PC.

Usage: python -I rvhost.py RV32EMU_DLL FX_ELF IMPORTS_JSON OUT_JSON [--turns N]
           [--opt KEY=VALUE ...] [--prof] [--no-memtrace]
--opt sets host.py's OPTS (stream, provider, clock, tools, tool_schemas, env); --prof adds
mnemonic pairs/triples and bytes moved per function. The wasm linear-memory base is read from
the ELF's `rv_linear_base` symbol (P0's images: 0x02000000).
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

LIN = 0x02000000  # default; RvHost reads the image's rv_linear_base
STACK_TOP = 0x100000  # __stack_pointer initial value (stack first, data at 1 MB)
HEAP_BASE = 0x149120  # end of the data segments
MNEMONICS = ("LUI AUIPC JAL JALR BEQ BNE BLT BGE BLTU BGEU LB LH LW LBU LHU SB SH SW ADDI SLTI SLTIU XORI ORI "
             "ANDI SLLI SRLI SRAI ADD SUB SLL SLT SLTU XOR SRL SRA OR AND FENCE ECALL EBREAK MUL MULH MULHSU MULHU "
             "DIV DIVU REM REMU SH1ADD SH2ADD SH3ADD LR.W SC.W CSR "
             "ANDN ORN XNOR CLZ CTZ CPOP SEXT.B SEXT.H ZEXT.H MAX MAXU MIN MINU ROL ROR ORC.B REV8 "
             "BCLR BEXT BINV BSET").split()


class RvHost(H.Host):
    def __init__(self, dll, elf_path, imports, repo, turns=2, log=None, prof=False, memtrace=True):
        self._init_common(repo, turns, log)
        self.instrumented = False
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
        for f in (e.rv_pairs, e.rv_pairs_dep, e.rv_trips):
            f.restype = ctypes.POINTER(ctypes.c_uint64)
        e.rv_pcbytes.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_incl.restype = ctypes.POINTER(ctypes.c_uint64)
        e.rv_wcalls.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_wbytes.restype = ctypes.POINTER(ctypes.c_uint64)
        e.rv_set_watch.argtypes = [ctypes.c_uint32]

        e.rv_set_lin.argtypes = [ctypes.c_uint32]
        self.e = e
        f = open(elf_path, "rb")
        elf = ELFFile(f)
        text = elf.get_section_by_name(".text")
        self.text_lo, self.text_hi = text["sh_addr"], text["sh_addr"] + text["sh_size"]
        sym = elf.get_section_by_name(".symtab")
        lb = sym.get_symbol_by_name("rv_linear_base")
        assert lb or self.text_lo < 0x02000000, "image without rv_linear_base: linear-memory base unknown"
        self.lin = LIN
        if lb:
            off = lb[0]["st_value"]
            for seg in elf.iter_segments():
                if seg["p_type"] == "PT_LOAD" and seg["p_vaddr"] <= off < seg["p_vaddr"] + seg["p_filesz"]:
                    self.lin = struct.unpack("<I", seg.data()[off - seg["p_vaddr"]:off - seg["p_vaddr"] + 4])[0]
        e.rv_set_lin(self.lin)
        e.rv_set_prof(1 if prof else 0)
        assert e.rv_init(self.text_lo, self.text_hi) == 0
        if memtrace:
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
        self.linear_bytes_addr = sym.get_symbol_by_name("rv_linear_bytes")[0]["st_value"]
        self.funcs = sorted((s["st_value"], s["st_size"], s.name) for s in sym.iter_symbols()
                            if s["st_info"]["type"] == "STT_FUNC" and s["st_size"])
        if prof and os.environ.get("RV_WATCH"):  # call sites of one function (e.g. memmove)
            w = [ad for ad, sz, nm in self.funcs if nm == os.environ["RV_WATCH"]]
            if w:
                e.rv_set_watch(w[0])
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
        return ctypes.string_at(self.base + self.lin + p, n)

    def wr(self, p, b):
        ctypes.memmove(self.base + self.lin + p, bytes(b), len(b))
        self.e.mt_host_store(self.lin + p, len(b))

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

    def arg_bytes(self, argp, nwords):
        """An import's argument words (argp is a machine address)."""
        return ctypes.string_at(self.base + argp, 4 * nwords)

    def hostcall(self, ident, argp, nwords):
        imp, fn = self.handlers[ident]
        words = struct.unpack(f"<{nwords}I", self.arg_bytes(argp, nwords)) if nwords else ()
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

    def dispatch(self, r, pc):
        """One stop of the machine (rv_run's result r at pc); False ends the run."""
        if r == 1:
            a7 = self.regs[17]
            if a7 >= 0x400:
                try:
                    self.regs[10] = self.hostcall(a7 - 0x400, self.regs[10], self.regs[11])
                except SystemExit:
                    return False
                self.e.rv_set_pc(pc + 4)
                return True
            if a7 == 96:  # native tool call begins (imports.c --native-tools)
                self.native_start = (self.e.rv_icount(), self.text(self.regs[10], self.regs[11]))
                self.e.rv_set_pc(pc + 4)
                return True
            if a7 == 97:  # native tool call ended; a0 = its result (-100: not native)
                r = self.regs[10]
                if r != 0xFFFFFF9C:
                    st, nm = self.native_start
                    self.events.append((st, "tool_call", {"name": nm, "native": True}))
                    self.mark("tool_result", {"name": nm, "bytes": r, "native": True})
                    self.native_calls.append(nm)
                self.e.rv_set_pc(pc + 4)
                return True
            if a7 == 95:
                self.next_phase(self.phase_names[1])
                self.phase_names.pop()
                self.e.rv_set_pc(pc + 4)
                return True
            if a7 == 93:
                self.mark("returned", self.regs[10])
                return False
            if a7 == 94:
                self.mark("trap", f"wasm trap {self.regs[10] - 1000}")
                return False
            self.mark("trap", f"unknown ecall {a7}")
            return False
        word = struct.unpack("<I", ctypes.string_at(self.base + pc, 4))[0] if pc < 0x08000000 else 0
        self.mark("trap", f"rv_run={r} pc={pc:#x} insn={word:#010x} fault={self.e.rv_fault_addr():#x} sym={self.sym(pc)}")
        return False

    def run(self):
        t0 = time.perf_counter()
        self.mark("start")
        while True:
            r = self.e.rv_run(1 << 40)
            if not self.dispatch(r, self.e.rv_get_pc()):
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
        ptr = {"pc": self.e.rv_pccounts, "mis": self.e.rv_mispc, "bytes": self.e.rv_pcbytes}[which]()
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


def parse_opts(items):
    kw = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k in ("tools", "env"):
            kw[k] = [x for x in v.split(",") if x] if k == "tools" else kw.get("env", []) + [v]
        else:
            kw[k] = v
    return kw


def turn_summary(events):
    """Turn totals (prompt sent to its response) and per-step splits, from the event list."""
    ev = [(f, k, d) for f, k, d in events if k != "deliver"]
    req = [(f, d) for f, k, d in ev if k == "request"]
    resp = [(f, d) for f, k, d in ev if k == "response"]
    prompts = [f for f, d in req if d == "session/prompt"]
    ends = [f for f, d in resp if d == "session/prompt"]
    out = {"boot": next(f for f, k, d in ev if k == "boot_end"),
           "turns": [e - p for p, e in zip(prompts, ends)]}
    init = [(f, d) for f, d in req if d == "initialize"]
    if init:
        out["initialize"] = next(f for f, d in resp if d == "initialize") - init[0][0]
    steps = []
    opens = [f for f, k, d in ev if k == "stream_open"]
    closes = [f for f, k, d in ev if k == "stream_close"]
    calls = [(f, d) for f, k, d in ev if k == "tool_call"]
    results = [f for f, k, d in ev if k == "tool_result"]
    marks = sorted([(p, "prompt") for p in prompts] + [(r, "result") for r in results])
    for i, fo in enumerate(opens):
        prev = max((m for m, _ in marks if m <= fo), default=fo)
        fc = closes[i] if i < len(closes) else fo
        nxt = min([f for f, _ in calls if f >= fc] + [e for e in ends if e >= fc] + [1 << 62])
        steps.append({"intake_and_request": fo - prev, "stream": fc - fo, "after_stream": nxt - fc})
    out["steps"] = steps
    out["tool_in_guest"] = [r - f for (f, _), r in zip(calls, results)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("elf")
    ap.add_argument("imports")
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--opt", action="append")
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--no-memtrace", action="store_true")
    a = ap.parse_args()
    H.set_opts(**parse_opts(a.opt))
    dll = ctypes.CDLL(os.path.abspath(a.dll))
    tmp = H.repo_dir(tempfile.mkdtemp(prefix="fxrepo-"))
    tinyrepo.make_repo(tmp)
    h = RvHost(dll, a.elf, json.load(open(a.imports)), tmp, a.turns, log=lambda *x: print(*x, file=sys.stderr),
               prof=a.prof, memtrace=not a.no_memtrace)
    h.run()
    nops = dll.rv_nops()
    opsv = h.e.rv_ops()
    ops = {MNEMONICS[i]: opsv[i] for i in range(nops) if opsv[i]}
    res = {"elf": os.path.abspath(a.elf), "rv_instructions_total": h.fuel(), "peak_linear_memory_bytes": h.peak_mem,
           "wall_s": h.wall, "events": h.events, "import_calls": h.import_calls, "model_requests": h.request_bytes,
           "dynamic_ops": ops, "misaligned_accesses": h.e.rv_misaligned(), "per_function": h.profile(), "misaligned_by_function": h.profile("mis"),
           "final_repo": {r: open(os.path.join(tmp, r), encoding="utf-8").read() for r in h.repo.files()
                          if r.startswith("tinycalc/calc") or r.startswith("tests/test_div")}}
    if not a.no_memtrace:
        res["memory_access"] = memory_stats(h)
    res["opts"] = dict(H.OPTS)
    res["native_calls"] = h.native_calls
    res["unsupported_imports"] = h.unsupported
    res["summary"] = turn_summary(h.events)
    res["linear_base"] = h.lin
    if a.prof:
        n = nops
        pr = h.e.rv_pairs()
        pd = h.e.rv_pairs_dep()
        tr = h.e.rv_trips()
        res["pairs"] = sorted(([MNEMONICS[i // n], MNEMONICS[i % n], pr[i], pd[i]] for i in range(n * n) if pr[i]), key=lambda r: -r[2])[:400]
        res["triples"] = sorted(([MNEMONICS[i // (n * n)], MNEMONICS[(i // n) % n], MNEMONICS[i % n], tr[i]] for i in range(n * n * n) if tr[i]), key=lambda r: -r[3])[:400]
        res["bytes_by_function"] = h.profile("bytes")
        n_ = (h.text_hi - h.text_lo) // 4
        inc = (ctypes.c_uint64 * n_).from_address(ctypes.addressof(h.e.rv_incl().contents))
        wc = (ctypes.c_uint32 * n_).from_address(ctypes.addressof(h.e.rv_wcalls().contents))
        wb = (ctypes.c_uint64 * n_).from_address(ctypes.addressof(h.e.rv_wbytes().contents))
        callers = {}
        for i in range(n_):
            if wc[i]:
                f = h.sym(h.text_lo + 4 * i)
                c = callers.setdefault(f, [0, 0])
                c[0] += wc[i]
                c[1] += wb[i]
        res["watched_callers"] = {"function": os.environ.get("RV_WATCH"), "callers": callers}
        res["inclusive_by_function"] = {name: inc[(addr - h.text_lo) // 4] for addr, size, name in h.funcs
                                        if h.text_lo <= addr < h.text_hi and inc[(addr - h.text_lo) // 4]}
    json.dump(res, open(a.out, "w"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: res[k] for k in ("rv_instructions_total", "peak_linear_memory_bytes", "wall_s", "misaligned_accesses")}))
    print("turns", res["summary"]["turns"], "messages", h.messages_out)
    for ev in h.events:
        if ev[1] != "deliver":
            print(ev[0], ev[1], str(ev[2])[:200])


if __name__ == "__main__":
    main()
