"""Trace a stories ELF on the reference Machine; trace tokens and instructions per model token.

python trace.py ELF [max_model_tokens]   (run from the worktree root)
The DROM is widened to 1 MB at 0x100000 for this measurement only (weights are 260 KB int8).
"""
import collections
import sys
import time

from chipformer.rv import elf, ref, spec as S

import os
if not os.environ.get("REALMAP"):
    S.DROM_BASE, S.DROM_WORDS = 0x100000, 1 << 18

path = sys.argv[1]
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
m = ref.Machine.from_program(elf.program(path))
toks = steps = 0
last_out, marks = 0, []
by_op = collections.Counter()
by_op_tok = collections.Counter()
t0 = time.time()
while not m.halted:
    w = m.rom[(m.pc - S.ROM_BASE) >> 2] if m.n != S.N_MAX else None
    op = ref.decode(w).op if w is not None else "R"
    r = m.record()
    toks += len(r)
    by_op[op] += 1
    by_op_tok[op] += len(r)
    if len(m.out) > last_out:
        if not marks or m.out[last_out:last_out + 1] != b"" and len(marks) == 0 or True:
            pass
        last_out = len(m.out)
        if not marks or toks - marks[-1][0] > 1000:  # first byte of a new word
            marks.append((toks, m.steps, m.out.decode(errors="replace")))
            if len(marks) >= nmax:
                break
print(f"{time.time() - t0:.0f} s python")
prev = (0, 0)
for k, (t, s, o) in enumerate(marks):
    print(f"model token {k}: {t - prev[0]:,} trace tokens, {s - prev[1]:,} instructions, "
          f"{(t - prev[0]) / (s - prev[1]):.2f} tok/instr  out={o!r}")
    prev = (t, s)
tot = sum(by_op_tok.values())
print("top ops by trace tokens:")
for op, n in by_op_tok.most_common(14):
    print(f"  {op:6s} {n / tot:6.1%}  {by_op[op]:>9,} instr  {n / by_op[op]:.2f} tok each")
