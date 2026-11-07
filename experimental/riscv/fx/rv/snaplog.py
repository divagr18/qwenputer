"""Records an fx session for replay on the Qwenputer (chipformer/rv/fxrun.py): the host's side of
every import (the a0 it returned, the bytes it wrote into guest memory), the registers at every
ecall, and the machine state at boot's end (the first stdin read).

    python -I snaplog.py RV32EMU_DLL FX_ELF IMPORTS_JSON OUT.npz [--opt KEY=VALUE ...] [--units a,b]

Runs the same host as rvhost.py. The replay checks the CPU's registers against this log at every
ecall, so it also proves the log complete. --units: the named circ units (the emulator's twins of
the units built in the circuit, chipformer/rv/hot.py) go on at boot's end, where the replay starts,
so the log holds the registers and memory of a run with those units.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import host as H  # noqa: E402
import rvhost as RH  # noqa: E402
import tinyrepo  # noqa: E402

MEMSZ = 0x08000000


class LogHost(RH.RvHost):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.elog = []
        self.cur = None
        self.snap = None

    def mem_words(self):
        mem = np.ctypeslib.as_array((ctypes.c_uint8 * MEMSZ).from_address(self.base)).view("<u4")
        ad = np.nonzero(mem)[0]
        return ad.astype(np.uint32), mem[ad].copy()

    def wr(self, p, b):
        super().wr(p, b)
        if self.cur is not None:
            self.cur["w"].append((self.lin + p, bytes(b)))

    def mark(self, kind, detail=None):
        super().mark(kind, detail)
        if kind == "boot_end" and self.snap is None and self.cur is not None:
            self.snap = len(self.elog)

    def hostcall(self, ident, argp, nwords):
        ic = self.e.rv_icount()
        x = [self.regs[i] for i in range(32)]
        pc = self.e.rv_get_pc()
        self.cur = {"ic": ic, "pc": pc, "x": x, "w": [], "imp": self.imports[ident]["name"]}
        snap_ad = None
        if self.snap is None:  # (the state before this call, if it turns out to be boot's end)
            snap_ad = self.mem_words()
        try:
            r = super().hostcall(ident, argp, nwords)
        except SystemExit:
            self.cur["exit"] = 1
            self.elog.append(self.cur)
            self.cur = None
            raise
        self.cur["a0"] = r
        self.elog.append(self.cur)
        self.cur = None
        if self.snap == len(self.elog) - 1 + 0 and snap_ad is not None and not hasattr(self, "boot"):
            self.boot = dict(ic=ic, pc=pc, x=x, ad=snap_ad[0], val=snap_ad[1], index=len(self.elog) - 1)
            if getattr(self, "unit_specs", None):
                self.e.rv_units_mode(1)
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("elf")
    ap.add_argument("imports")
    ap.add_argument("out")
    ap.add_argument("--opt", action="append")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    H.set_opts(**RH.parse_opts(a.opt))
    dll = ctypes.CDLL(os.path.abspath(a.dll))
    tmp = H.repo_dir(tempfile.mkdtemp(prefix="fxrepo-"))
    tinyrepo.make_repo(tmp)
    h = LogHost(dll, a.elf, json.load(open(a.imports)), tmp, 2, log=lambda *x: None, memtrace=False)
    if a.units:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "circ"))
        import run as CR  # noqa: E402
        import units as CU  # noqa: E402

        names = a.units.split(",")
        h.unit_specs, skipped = CU.specs_for(a.elf, ",".join(n for n in names if n not in CU.TWINS))
        assert not skipped, skipped
        h.unit_specs += CU.twin_specs(names)
        CR.setup_dll(dll)
        CR.install(dll, h.unit_specs)
        dll.rv_units_mode(0)
    h.run()
    end_ad, end_val = h.mem_words()
    x_end = [h.regs[i] for i in range(32)]
    b = h.boot
    np.savez_compressed(a.out, boot_ad=b["ad"], boot_val=b["val"], boot_x=np.array(b["x"], np.uint32),
                        boot_pc=b["pc"], boot_ic=b["ic"], boot_index=b["index"], end_ad=end_ad,
                        end_val=end_val, end_x=np.array(x_end, np.uint32), end_ic=h.e.rv_icount(),
                        log=np.frombuffer(__import__("pickle").dumps(h.elog), np.uint8))
    print(json.dumps({"ecalls": len(h.elog), "boot_ic": b["ic"], "boot_index": b["index"],
                      "end_ic": h.e.rv_icount(), "summary": RH.turn_summary(h.events)["turns"],
                      "imports": sorted({e["imp"] for e in h.elog}),
                      "written_bytes": sum(len(x[1]) for e in h.elog for x in e["w"])}))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
