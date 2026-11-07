"""Runs render_rv.c on rv32emu against a recorded session: every output byte must equal delta.py's.

  zig cc -target riscv32-freestanding-none -mcpu=generic_rv32+m+zba -O2 -ffreestanding -nostdlib
         -fno-builtin -Wl,-T,rv/link.ld fxio/render_rv.c -o render_rv.elf
  python -I render_cost.py DLL render_rv.elf MODEL_DIR OUT_JSON RUN_JSON [RUN_JSON ..]

The runs must be scripted ones: their raw completions are rebuilt from the plan.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import sys

from elftools.elf.elffile import ELFFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import delta as D  # noqa: E402
import protocol as P  # noqa: E402
import scripted as SC  # noqa: E402

MB, IN, OUT = 0x01F00000, 0x02000000, 0x04000000


class Emu:
    def __init__(self, dll, elf_path):
        e = self.e = ctypes.CDLL(os.path.abspath(dll))
        e.rv_mem.restype = ctypes.c_void_p
        e.rv_regs.restype = ctypes.POINTER(ctypes.c_uint32)
        e.rv_get_pc.restype = ctypes.c_uint32
        e.rv_set_pc.argtypes = [ctypes.c_uint32]
        e.rv_icount.restype = ctypes.c_uint64
        e.rv_tokens.restype = ctypes.c_uint64
        e.rv_run.argtypes = [ctypes.c_uint64]
        e.rv_set_lin.argtypes = [ctypes.c_uint32]
        elf = ELFFile(open(elf_path, "rb"))
        text = elf.get_section_by_name(".text")
        e.rv_set_prof(0)
        e.rv_set_lin(IN)
        assert e.rv_init(text["sh_addr"], text["sh_addr"] + text["sh_size"]) == 0
        self.base = e.rv_mem()
        self.segs = [(s["p_vaddr"], s.data()) for s in elf.iter_segments() if s["p_type"] == "PT_LOAD" and s["p_filesz"]]
        self.entry = elf.header["e_entry"]
        self.regs = e.rv_regs()

    def run(self, mode, data, *mail):
        for va, d in self.segs:
            ctypes.memmove(self.base + va, d, len(d))
        ctypes.memmove(self.base + IN, data + b"\0", len(data) + 1)
        m = struct.pack("<5I", mode, len(data), *(list(mail) + [0, 0, 0])[:3])
        ctypes.memmove(self.base + MB, m, len(m))
        self.e.rv_set_pc(self.entry)
        i0, t0 = self.e.rv_icount(), self.e.rv_tokens()
        r = self.e.rv_run(1 << 34)
        assert r == 1 and self.regs[17] == 93, (r, hex(self.e.rv_get_pc()), self.regs[17])
        out = ctypes.string_at(self.base + OUT, self.regs[10])
        return out, self.e.rv_icount() - i0, self.e.rv_tokens() - t0


def main():
    dll, elf, model, out, *runs = sys.argv[1:]
    t = D.QwenTemplate(model)
    emu = Emu(dll, elf)
    rows = []
    for path in runs:
        run = json.load(open(path))
        pk = P.decode_stream(bytes.fromhex(run["packets_hex"]))
        sb = SC.ScriptedBackend(t, run["task"])
        prev_text = ""
        for i, (q, a) in enumerate(zip(pk[0::2], pk[1::2])):
            body = P.parse_request(q)[2]
            _, msgs, tools = D.request_messages(json.loads(body))
            text = t.render(msgs, tools)
            ids = t.encode(text)
            gen = sb.generate(0, None)[:-1]
            raw = t.decode(gen)
            want = P.parse_reply(a)[1][0]
            row = {"task": run["task"], "call": i, "body_bytes": len(body), "raw_bytes": len(raw.encode()), "reply_bytes": len(want)}
            got, n, tk = emu.run(2, raw.encode(), len(ids), len(gen), i)
            row.update(sse_instr=n, sse_tokens=tk, sse_equal=got == want)
            if i:
                suffix = text[len(prev_text):]
                got, n, tk = emu.run(1, body)
                row.update(delta_bytes=len(suffix.encode()), render_instr=n, render_tokens=tk, render_equal=got == suffix.encode())
            rows.append(row)
            prev_text = text + raw
    res = {"rows": rows}
    calls = [r for r in rows if "render_instr" in r]
    res["render"] = {"calls": len(calls), "all_equal": all(r["render_equal"] for r in calls),
                     "instr_total": sum(r["render_instr"] for r in calls),
                     "tokens_total": sum(r["render_tokens"] for r in calls),
                     "body_bytes": sum(r["body_bytes"] for r in calls), "delta_bytes": sum(r["delta_bytes"] for r in calls)}
    res["sse"] = {"calls": len(rows), "all_equal": all(r["sse_equal"] for r in rows),
                  "instr_total": sum(r["sse_instr"] for r in rows), "tokens_total": sum(r["sse_tokens"] for r in rows),
                  "raw_bytes": sum(r["raw_bytes"] for r in rows), "reply_bytes": sum(r["reply_bytes"] for r in rows)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("render", "sse")}, indent=1))
    bad = [r for r in rows if not r["sse_equal"] or not r.get("render_equal", True)]
    if bad:
        print("first differing row:", bad[0])


if __name__ == "__main__":
    main()
