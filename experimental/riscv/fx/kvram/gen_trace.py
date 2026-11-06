"""Records fx's full word-level memory trace on the RV32 build (P0's run, same host and script).

Builds on ../rv/rvhost.py; the emulator is ../rv/rv32emu.c compiled with MT_TRACE
(rv32emu_trace.c). Output: an .npz with
  rec  uint32 [n, 2]: word index | kind << 28 (0 load, 1 store, 2 host store, 3 phase), value
  phase_names
Usage: python -I gen_trace.py RV32EMU_TRACE_DLL FX_ELF IMPORTS_JSON OUT_NPZ
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rv"))
sys.path.insert(0, os.path.dirname(HERE))
import rvhost  # noqa: E402
import tinyrepo  # noqa: E402


def main():
    dll_p, elf, imports, out = sys.argv[1:5]
    dll = ctypes.CDLL(os.path.abspath(dll_p))
    dll.mt_trace_enable.argtypes = [ctypes.c_uint64]
    dll.mt_trace_buf.restype = ctypes.POINTER(ctypes.c_uint32)
    dll.mt_trace_len.restype = ctypes.c_uint64
    tmp = tempfile.mkdtemp(prefix="fxrepo-")
    tinyrepo.make_repo(tmp)
    h = rvhost.RvHost(dll, elf, json.load(open(imports)), tmp, 2, log=lambda *x: None)
    cap = 80_000_000
    assert dll.mt_trace_enable(cap) == 0
    h.run()
    n = dll.mt_trace_len()
    assert n < cap, "trace buffer full"
    buf = np.ctypeslib.as_array(dll.mt_trace_buf(), shape=(n * 2,)).reshape(n, 2).copy()
    np.savez(out, rec=buf, phase_names=np.array(h.phase_names))
    shutil.rmtree(tmp, ignore_errors=True)
    kind = buf[:, 0] >> 28
    print(json.dumps({"records": int(n), "loads": int((kind == 0).sum()), "stores": int((kind == 1).sum()),
                      "host_stores": int((kind == 2).sum()), "phases": int((kind == 3).sum()),
                      "rv_instructions_total": int(h.fuel())}))


if __name__ == "__main__":
    main()
