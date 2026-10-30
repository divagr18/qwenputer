import bisect, collections, sys
from chipformer.rv import elf, ref, spec as S
import os
if not os.environ.get("REALMAP"):
    S.DROM_BASE, S.DROM_WORDS = 0x100000, 1 << 18
path, syms = sys.argv[1], sys.argv[2]
sy = [(int(a, 16), n) for a, _, n in (l.split() for l in open(syms))]
addrs = [a for a, _ in sy]
m = ref.Machine.from_program(elf.program(path))
c = collections.Counter()
while not m.halted:
    pc = m.pc
    r = m.record()
    c[sy[bisect.bisect_right(addrs, pc) - 1][1]] += len(r)
tot = sum(c.values())
for k, v in c.most_common(15):
    print(f"{k:22s} {v:>9,} {v / tot:6.1%}")
print("total", tot)
