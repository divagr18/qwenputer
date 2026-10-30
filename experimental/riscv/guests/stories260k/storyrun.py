"""stories260K on the circuit's executor (GPU, batch 1): free-running, checked against the reference.

python storyrun.py ELF [--exec fused|triton|compile] [--steps N] [--chunk N]
--exec: fusedexec.FreeRun3 (default), the Triton Executor2 or the torch.compile executor;
--steps: only run the first N steps (still checked against the reference prefix). The story is
printed as it comes, every --chunk steps.
"""
import argparse
import time

import numpy as np
import torch

from chipformer.rv import core, elf, fastexec, fusedexec, ref
from chipformer.rv import spec as S

ap = argparse.ArgumentParser()
ap.add_argument("elf")
ap.add_argument("--exec", default="fused", choices=("fused", "triton", "compile"))
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--chunk", type=int, default=50000)
args = ap.parse_args()

c = core.build()
prog = elf.program(args.elf)
m = ref.Machine.from_program(prog)
t = time.time()
want = m.run(10**9)
st = ref.trace_stats(want)
print(f"reference: {st['tokens']:,} tokens, {st['instructions']:,} instructions "
      f"({time.time() - t:.0f} s): {m.out.decode(errors='replace')!r}", flush=True)
n = (args.steps or len(want) - 1)
if args.exec == "fused":
    fr = fusedexec.FreeRun3(c, 1, "cuda")
else:
    fr = fastexec.FreeRun(c, 1, "cuda", compile_=args.exec == "compile", v2=args.exec == "triton")
core.snapshot(c, ref.Machine.from_program(prog))
fr.load(0, S.tok("GO"))
fr.capture()
torch.cuda.synchronize()
t = time.time()
got = []
while len(got) < n:
    k = min(args.chunk, n - len(got))
    part = [int(x) for x in np.asarray(fr.run(k)[0])]
    text = bytes(S.family(x)[1] for x in part if S.family(x)[0] == "BYTE").decode(errors="replace")
    print(text, end="", flush=True)
    got += part
dt = time.time() - t
print()
bad = next((i for i in range(n) if got[i] != want[i + 1]), None)
print("exact" if bad is None else f"MISMATCH at step {bad}: {S.name(got[bad])} vs {S.name(want[bad + 1])}")
print(f"{args.exec}: {n:,} steps in {dt:.1f} s = {n / dt:,.0f} tok/s ({dt / n * 1e3:.3f} ms/step)",
      flush=True)
