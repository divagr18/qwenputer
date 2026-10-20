"""The Triton executor against the IR on the core's test programs and the bridge's calc calls:
every dim at every position (tritonexec.check).

    python -m chipformer.rv.execsuite [--store]
"""

import argparse
import json
import time
from pathlib import Path

from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref, tritonexec

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "rv"


def calc_call(T, expr):
    return T.encode(f"<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n"
                    "</parameter>\n</function>\n</tool_call>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true", help="the bridge circuit with the store")
    args = ap.parse_args()
    out = {}
    t0 = time.time()
    c = core.build()
    progs = sorted(p for p in DATA.glob("rv32u*.elf") if ".qemu." not in p.name)
    n = 0
    for p in progs:
        m = ref.Machine.from_program(elf.program(str(p)))
        core.snapshot(c, m)
        toks = ref.Machine.from_program(elf.program(str(p))).run(200000)
        n += tritonexec.check(c, toks)
    out["programs"] = {"files": len(progs), "positions": n}
    print(json.dumps(out), flush=True)
    T = BR.Text(BR.default_tokenizer())
    store = None
    if args.store:
        from chipformer.rv.store import Config

        store = Config()
    cb = core.build(text=T, store=store)
    fw = str(DATA / "calc-bridge.elf")
    m0 = ref.Machine.from_program(elf.program(fw))
    BR.park(m0)
    m = ref.Machine.from_program(elf.program(fw))
    BR.park(m)
    ses = BR.Session(T, m)
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    for e in ("12*34", "7*6", "2*(3+4)-50", "99999999999*99999999999", "(1.2 - 3) * 4.5 - 0.75",
              "-84/18", "-84//18", "2**64 % 1000003", "gcd(462, 1071)"):
        ses.qwen(calc_call(T, e))
        ses.qwen(T.encode(" ok "))
    core.snapshot(cb, m0)
    if store:
        for node in cb.store.persistent.values():
            node.init = None
    out["bridge"] = {"positions": tritonexec.check(cb, ses.toks), "runs": len(ses.runs),
                     "store": bool(store)}
    out["seconds"] = round(time.time() - t0)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
