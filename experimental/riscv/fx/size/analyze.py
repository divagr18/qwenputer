"""Exact sparse-term and overlay statistics of one circuit.

    PYTHONPATH=<riscv tree> python fx/size/analyze.py TOKDIR legacy|fx OUT.pkl
"""
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse as SP

from chipformer.q35.emit import Layout, allocate
from chipformer.q35.merge import pick_hidden
from chipformer.rv import bridge as BR
from chipformer.rv import core
from chipformer.rv import spec as S

tokdir, which, out = sys.argv[1:4]
T = BR.Text(tokdir + "/tokenizer.json")
cfg = None
if which == "fx":
    from chipformer.rv import fxrun

    image = tuple(sorted(set(range(0xD9270 >> S.WIN_BITS, (0x1A00B0 >> S.WIN_BITS) + 1))
                         | set(range(100, 119))))[:35]
    cfg = fxrun.fx_cfg(image)
c = core.build(core.T4B, T, cfg=cfg) if cfg else core.build(core.T4B, T)
print("stats", c.stats())
import numpy as np
from chipformer.q35.ir import Head as _H
# taps of every channel: a single positive lag can be gated, a mixed one cannot
from collections import Counter as _C
tp = _C()
for n in c.nodes:
    if isinstance(n, _H):
        chs = list(n.q.values()) + list(n.k.values())
        for vh in n.vheads():
            chs += list(vh.v.values())
        for ch in chs:
            nz = tuple(i for i, x in enumerate(ch.taps) if x)
            tp[(nz, all(x >= 0 for x in ch.taps))] += 1
print("taps (lags, all >= 0)", dict(tp))
keep = set(c.keep)
al = allocate(c, None, keep, core.T4B.intermediate)
h, m = pick_hidden(2560, 1 + al.n_slots)
print("slots", al.n_slots, "hidden", h, "m", m, "needs", 2560 + m + 1 + al.n_slots)
lay = Layout(hidden=h, off=2560, m=m, slots=al.slots)
tc = dict(hidden_size=h, linear_num_key_heads=16, linear_num_value_heads=32,
          intermediate_size=9216, linear_conv_kernel_dim=4)
t = time.time()
sd = SP.emit_sparse(c, lay, tc, al, keep)
print("emitted", round(time.time() - t, 1))
nnz = defaultdict(int)
per = defaultdict(lambda: defaultdict(int))
total = 0
values = Counter()
for k, v in sd.items():
    L = int(k.split(".")[2])
    fam = ".".join(k.split(".")[3:]).replace(".weight", "")
    nz = sum(1 for x in v.d.values() if x != v.fill)
    nnz[fam] += nz
    per[L][fam] = nz
    total += nz
    values.update(x for x in v.d.values() if x != v.fill)
print("total nnz", total, dict(nnz))
print("distinct values", len(values), values.most_common(8))
zero_mlp = [L for L in per if not (per[L]["mlp.gate_proj"] or per[L]["mlp.up_proj"]
                                   or per[L]["mlp.down_proj"])]
print("layers with an all-zero MLP", len(zero_mlp), zero_mlp)
print("per layer nnz", [sum(per[L].values()) for L in sorted(per)])
print("all-zero tensors", Counter(".".join(k.split(".")[3:]).replace(".weight", "")
                                  for k, v in sd.items()
                                  if v.fill == 0 and not any(x != 0 for x in v.d.values())))
Tinit, Tset, C, why = SP.overlay_sets(c, T)
top = max(n.level for n in c.nodes) + 1
pk = SP.peak(c, C, keep, top)
print("token features nonzero on text", len(Tinit), "dims active on text", len(Tset),
      "chip-only dims", len(C), "of", len(c.dims), "why", dict(why))
print("peak live: chip-only", pk, "all", SP.peak(c, range(1, len(c.dims)), keep, top))
pickle.dump(dict(stats=c.stats(), slots=al.n_slots, hidden=h, m=m, nnz=dict(nnz),
                 per={L: dict(v) for L, v in per.items()}, zero_mlp=zero_mlp, Tinit=len(Tinit),
                 Tset=len(Tset), C=len(C), why=dict(why), peakC=pk, vocab=getattr(c, "vocab", S.VOCAB), total=total,
                 distinct=len(values)), open(out, "wb"))
