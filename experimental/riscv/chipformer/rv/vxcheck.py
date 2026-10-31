"""The VX vector unit on a merged QwenPU checkpoint's own chip layers (the HF modules),
teacher-forced.

    python -m chipformer.rv.vxcheck MODEL [ELF ...] [--steps N]

Each program (tests/test_rv_vx.py's random programs: VPUT, VQ, VRED / VGET, VMAT, VEW, the
attention ops; then every ELF given) is snapshotted into the chip view's states after a short
chat prompt; its reference trace is then fed token by token from GO, and the circuit's
prediction at every position must be the next token.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble
from chipformer.rv.chat import RvChat


def check(chat: RvChat, prog, steps: int) -> dict:
    want = ref.Machine.from_program(prog).run(steps + 1)[: steps + 1]
    ids = chat.prompt_ids([{"role": "user", "content": "Hi"}])
    ccache = chat._chip_cache()
    h = chat.cm.embed_tokens(torch.tensor([ids], device=chat.dev))
    with torch.no_grad():
        for L in range(chat.n_chip):
            h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=ccache)
    core.snapshot(chat.c, ref.Machine.from_program(prog))
    hf.inject(chat.c, ccache)
    if chat.graph is None:
        chat._capture()
    g, bad, t0 = len(ids), None, time.time()
    for i in range(len(want) - 1):
        hc = chat._chip_step(chat.v0 + want[i], g, ccache)
        g += 1
        got = chat.circuit_token(hc)
        if got != chat.v0 + want[i + 1]:
            bad = {"at": i, "want": S.name(want[i + 1]),
                   "got": None if got is None else S.name(got - chat.v0)}
            break
    fams = {S.family(t)[0] for t in want}
    return {"steps": len(want) - 1, "mismatch": bad, "VX": sorted(f for f in fams if f[0] == "V"),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("elfs", nargs="*")
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    import test_rv_vx as T

    chat = RvChat(args.model)
    assert chat.meta.get("vx"), "not a VX (QwenPU) checkpoint"
    progs = []
    for name in ("vput", "vq", "vred_vget", "vmat", "vew", "attention"):
        for seed in range(2):
            progs.append((f"{name}{seed}", getattr(T, f"{name}_program")(seed)))
    progs += [(Path(e).name, elf.program(e)) for e in args.elfs]
    ok = True
    for name, prog in progs:
        r = check(chat, prog, args.steps)
        ok &= r["mismatch"] is None
        print(name, r, flush=True)
    print("ALL EXACT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
