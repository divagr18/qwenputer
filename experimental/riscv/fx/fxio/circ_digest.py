"""Digests of core.build's circuits, to show which ones a change to core.py moves (docs/fx/fwd.md).

    python fx/fxio/circ_digest.py MODEL_DIR LOG.npz ELF OUT.json

Run it on two trees and compare: every circuit whose digest is equal is node for node the same
(dims, and every FFN and head with its terms, gates and outputs; not the snapshot in `init`).
Circuits: the legacy map alone, the legacy map with the text bridge (rv-q4p2's), test_fx_live's
small paged map with the bridge, fx's map alone, and fx's map with the bridge (both go_on_tcc
values where the tree has the flag).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import fields, is_dataclass, replace


def canon(o):
    """A form of o that does not depend on hash order: an Expr's terms and every dict sorted."""
    if hasattr(o, "terms") and isinstance(o.terms, dict):
        return ("Expr", tuple(sorted((int(k), repr(v)) for k, v in o.terms.items())))
    if is_dataclass(o):
        return (type(o).__name__, tuple((f.name, canon(getattr(o, f.name))) for f in fields(o) if f.name != "init"))
    if isinstance(o, dict):
        return tuple(sorted(((repr(k), canon(v)) for k, v in o.items())))
    if isinstance(o, (list, tuple)):
        return tuple(canon(x) for x in o)
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(repr(x) for x in o))
    return repr(o)


def node_repr(n) -> str:
    return repr(canon(n))


def digest(c) -> dict:
    h = hashlib.sha256()
    h.update(repr([(d.name, d.kind, d.level) for d in c.dims]).encode())
    names = []
    for n in c.nodes:
        r = node_repr(n)
        h.update(r.encode())
        names.append((n.name, hashlib.sha256(r.encode()).hexdigest()[:16]))
    return {"sha256": h.hexdigest(), "stats": c.stats(), "nodes": names}


def main(argv):
    model_dir, log, elf, out = argv
    sys.path[:0] = ["."]
    from chipformer.rv import bridge as BR
    from chipformer.rv import core, fxrun, wide
    from chipformer.rv import spec as S

    T = BR.Text(os.path.join(model_dir, "tokenizer.json"))
    tg = wide.target_of(model_dir)
    has_flag = "go_on_tcc" in inspect.signature(core.build).parameters
    big = S.MemCfg("big", 22, 19, 40, ((0, 15), (0x100000, 16), (0x3F0000, 15)), 64, (2, 3, 32), 0x3F8000)
    small = replace(big, text_addr=0x7F0000)  # tests/test_fx_live.py's BIG and text_cfg
    cfg = fxrun.load(log, elf, frames=4096, units=["all+"])[0]  # (fx's map, the 13 units)
    todo = {"legacy": lambda: core.build(), "legacy_text": lambda: core.build(tg, T),
            "small_paged_text": lambda: core.build(text=T, cfg=small), "fx": lambda: core.build(cfg=cfg),
            "fx_text": lambda: core.build(tg, T, cfg)}
    if has_flag:
        todo["fx_text_no_go"] = lambda: core.build(tg, T, cfg, go_on_tcc=False)
    res = {"has_flag": has_flag}
    for k, f in todo.items():
        res[k] = digest(f())
        print(k, res[k]["sha256"][:16], res[k]["stats"], flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
