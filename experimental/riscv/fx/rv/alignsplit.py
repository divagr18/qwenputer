"""Carries each wasm load/store's alignment hint through wasm2c, which ignores it.

wasm2c turns every access into a memcpy, so without unaligned scalar access in the CPU
every i32 load becomes four byte loads. This pass moves each load/store whose memarg
alignment is below what an aligned RV32 access needs (the natural alignment, capped at 4
bytes: an i64 is two words) (fx's code has them where it reads integers out of
byte slices: std.mem.readInt, wyhash, mem.eqlBytes) to a second memory, index 1, that the
runtime maps onto the same bytes as memory 0. Then build.py rewrites wasm2c's C so that
memory-0 accesses are plain aligned LW/SW and memory-1 accesses keep the byte-safe path.

Nothing in fx's code changes: the module is the same program with its accesses tagged by
their own alignment hint. The hint is a promise the compiler made (LLVM emits the natural
alignment only when the address is known aligned); rv32emu counts any access that breaks
it (misaligned_accesses), and the measured runs report that count.

Usage: python -I alignsplit.py IN.wasm OUT.wasm
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wasmbin as W  # noqa: E402

NATURAL = {0x28: 2, 0x29: 3, 0x2A: 2, 0x2B: 3, 0x2C: 0, 0x2D: 0, 0x2E: 1, 0x2F: 1, 0x30: 0, 0x31: 0, 0x32: 1,
           0x33: 1, 0x34: 2, 0x35: 2, 0x36: 2, 0x37: 3, 0x38: 2, 0x39: 3, 0x3A: 0, 0x3B: 1, 0x3C: 0, 0x3D: 1, 0x3E: 2}


def rewrite(m: W.Module):
    d = m.data
    bodies = []
    moved = kept = 0
    for s, e in m.code:
        i = W.body_code_start(d, s)
        out = bytearray(d[s:i])
        for st, en, op, _ in W.ops(d, i, e):
            if 0x28 <= op <= 0x3E:
                flags, j = W.uleb(d, st + 1)
                mem = 0
                if flags & 0x40:
                    mem, j = W.uleb(d, j)
                off, j = W.uleb(d, j)
                assert j == en and mem == 0
                align = flags & ~0x40
                if align < min(NATURAL[op], 2):  # RV32 words: an i64 needs only 4-byte alignment
                    out += bytes([op]) + W.enc_uleb(align | 0x40) + W.enc_uleb(1) + W.enc_uleb(off)
                    moved += 1
                    continue
                kept += 1
            out += d[st:en]
        bodies.append(W.enc_uleb(len(out)) + bytes(out))
    res = bytearray(d[:8])

    def emit(sid, payload):
        res.append(sid)
        res.extend(W.enc_uleb(len(payload)))
        res.extend(payload)

    for sid, st, s, e, cname in m.sections:
        if sid == 5:
            n, j = W.uleb(d, s)
            assert n == 1
            emit(5, W.enc_uleb(2) + d[j:e] + b"\x00\x01")
        elif sid == 10:
            emit(10, W.enc_uleb(len(bodies)) + b"".join(bodies))
        elif sid == 0 and cname != "name":
            continue  # debug info would point at old code offsets
        else:
            res.extend(d[st:e])
    return bytes(res), moved, kept


if __name__ == "__main__":
    m = W.Module(open(sys.argv[1], "rb").read())
    out, moved, kept = rewrite(m)
    open(sys.argv[2], "wb").write(out)
    print(f"under-aligned accesses moved to memory 1: {moved}; naturally aligned kept: {kept}")
