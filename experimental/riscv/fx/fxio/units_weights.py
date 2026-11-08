"""docs/fx/units-file.md: emit a circuit through chipbuild's sparse path (no file written) and list every
weight that is not a finite bf16 value, with its tensor, index and the circuit node and term it came
from. Run from experimental/riscv:

    python fx/fxio/units_weights.py test-sha                  the test map's circuit, every unit kind
    python fx/fxio/units_weights.py fx LOG.npz ELF [FRAMES]   fxrun's map, the 13 units (all+), 4096 frames
    python fx/fxio/units_weights.py live LIVE_CHIP_ARGS..     the circuit live_chip.py --write-chip-file
                                                              would write (it boots fx, writes nothing)
"""
import json, math, os, sys, time
RV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [RV, os.path.join(RV, "tests")]
import numpy as np
from chipformer.q35 import emit as E
from chipformer.q35.ir import FFN, Head, K
from chipformer.q35.merge import pick_hidden
from chipformer.rv import chipbuild as CB, core


def bad_weights(c):
    keep = set(c.keep)
    al = E.allocate(c, None, keep, c.target.intermediate)
    h, m = pick_hidden(CB.D0, 1 + al.n_slots)
    lay = E.Layout(hidden=h, off=CB.D0, m=m, slots=al.slots)
    tc = dict(hidden_size=lay.hidden, linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads, intermediate_size=c.target.intermediate,
              linear_conv_kernel_dim=4)
    old = E.check_bf16
    E.check_bf16 = lambda n, t: None
    E.torch = CB._Shim()
    try:
        sd = E.emit_layers(c, lay, tc, 0, "model.", al, keep)
    finally:
        E.check_bf16 = old
        import torch; E.torch = torch
    dim_of = {lay.dim(d): d for d in range(1, len(c.dims)) if d in al.slots}
    bad = []
    for name, t in sd.items():
        for k, v in t.d.items():
            try:
                ok = CB._is_bf16(v) and math.isfinite(v)
            except OverflowError:
                ok = False
            if not ok:
                bad.append((name, k, v))
    out = []
    for name, k, v in bad:
        L = int(name.split(".")[2])
        info = {"tensor": name, "index": list(k), "value": v}
        if name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
            units = []  # emit_layers' order: a node's terms, its handover unit; the level's clears
            for n in c.nodes:
                if isinstance(n, FFN) and n.level == 2 * L + 2:
                    units += [(n.name, i, n.terms[i]) for i in range(len(n.terms))]
                    if n.out in al.subtract:
                        units.append((n.name, "handover", None))
            u = k[0]
            if u < len(units):
                info["node"], info["term"], t = units[u]
                if t is not None:
                    e = t[2] if name.endswith("gate_proj.weight") else t[1]
                    info["gate_expr" if name.endswith("gate_proj.weight") else "up_expr"] = {
                        c.dims[d].name: str(v) for d, v in e.terms.items()}
        out.append(info)
    return {"hidden": h, "m": m, "inv_c": 1 / lay.scale, "slots": al.n_slots, "bad": out,
            "max_abs_weight": max(abs(v) for t in sd.values() for v in t.d.values() if math.isfinite(v))}


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "live":
        sys.path.insert(0, os.path.join(RV, "fx", "fxio"))
        import live_chip as LC

        class Done(BaseException):
            pass

        def report(c, out, machine=None, extra=None):
            print(c.stats(), round(time.time() - t0, 1), flush=True)
            print(json.dumps(bad_weights(c), indent=1, default=str), flush=True)
            raise Done()

        CB.standalone = report
        a = LC.parse(sys.argv[2:])
        chip, template, backend = LC.setup(a)
        try:
            LC.session(a, chip, template, backend, a.tasks[0], 1, a.out, print)
        except Done:
            pass
        sys.exit(0)
    if sys.argv[1] == "test-sha":
        import test_rv_hot as T
        calls = [("sha", 0x11000, 0x12000, 0x10040, 0x12000)]
        c = core.build(cfg=T.machine(calls)[1])
    else:
        from chipformer.rv import fxrun
        frames = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
        c = core.build(cfg=fxrun.load(sys.argv[2], sys.argv[3], frames=frames, units=["all+"])[0])
    print(c.stats(), round(time.time() - t0, 1), flush=True)
    r = bad_weights(c)
    print(json.dumps(r, indent=1, default=str))
