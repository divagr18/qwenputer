"""Circuit statistics from the IR, no dense tensors: dims, slots, sparse terms per layer.

    python fx/size/circ.py TOKENIZER_DIR legacy|fx OUT.pkl
Run with PYTHONPATH pointing at the riscv tree whose core.py you want.
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
import numpy as np
from chipformer.q35.emit import allocate, reads, Layout
from chipformer.q35.ir import FFN, Head, Target
from chipformer.q35.merge import pick_hidden
from chipformer.rv import bridge as BR, core
from chipformer.rv import spec as S

tokdir, which, out = sys.argv[1:4]
t0 = time.time()
T = BR.Text(tokdir + "/tokenizer.json")
cfg = None
if which == "fx":
    from chipformer.rv import fxrun
    # image windows: 35 as the memory doc says; which windows does not change the structure
    image = tuple(range(0xD9270 >> S.WIN_BITS, (0x1A00B0 >> S.WIN_BITS) + 1)) + tuple(range(100, 100 + 19))
    image = tuple(sorted(set(image)))[:35]
    print("image windows", len(image))
    cfg = fxrun.fx_cfg(image)
c = core.build(core.T4B, T, cfg=cfg)
print("built", c.stats(), round(time.time() - t0, 1), flush=True)
keep = set(c.keep)
al = allocate(c, None, keep, core.T4B.intermediate)
print("slots", al.n_slots, "pick_hidden", pick_hidden(2560, 1 + al.n_slots), flush=True)
kinds = Counter(d.kind for d in c.dims)
print("dims by kind", kinds, "token fns", len(c.token_fns), "chip vocab", c.vocab if hasattr(c, "vocab") else None)
L = c.n_layers()
per = []
for l in range(L):
    heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * l + 1]
    ffn = [n for n in c.nodes if isinstance(n, FFN) and n.level == 2 * l + 2]
    units = sum(len(n.terms) for n in ffn)
    per.append(dict(layer=l, heads=len(heads), vheads=sum(len(n.vheads()) for n in heads), units=units,
                    names=sorted({n.name.split(".")[0].split("[")[0] for n in heads})[:6]))
print(json.dumps(per))
pickle.dump(dict(stats=c.stats(), slots=al.n_slots, per=per, vocab=getattr(c, "vocab", None)), open(out, "wb"))
