"""docs/fx/units-file.md: Qwen's merged side (chipformer/rv/wide.py) built from a chip file with the
text bridge, against the same built from the Python circuit: every tensor bit for bit.

    python fx/fxio/wide_file_check.py MODEL_DIR CHIP_TEXT.safetensors OUT.json LIVE_CHIP_ARGS..

LIVE_CHIP_ARGS are the --text run's (live_chip.py): the circuit is the one that run builds in Python,
from fx's boot on rv32emu (nothing is written). The anti-sink keys are fitted once and used by both.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import live_chip as LC  # noqa: E402
from chipformer.q35 import mute as MU  # noqa: E402
from chipformer.rv import chipfile as CF  # noqa: E402
from chipformer.rv import wide  # noqa: E402


class Built(BaseException):
    pass


def main() -> None:
    model_dir, chip_path, out = sys.argv[1:4]
    a = LC.parse(sys.argv[4:])
    assert a.text and not a.chip_file and not a.write_chip_file
    chip, template, backend = LC.Chip(a.elf, a.frames, a.full_regs, print, False, a.units, a.model_dir, None,
                                      None), None, None
    from chipformer.rv import exec4

    def stop(*x, **k):  # the circuit is built; the executor is not needed
        raise Built()

    exec4.Exec4 = stop
    exec4.cached_tables = lambda *x, **k: None
    try:  # (session reports the stop and then fails on the missing CPU: either way the circuit is built)
        LC.session(a, chip, LC.D.QwenTemplate(a.model_dir), backend, a.tasks[0], 1, a.out, print)
    except (Built, AttributeError):
        pass
    c, T = chip.c, chip.T
    assert c is not None and T is not None, "the circuit was not built"
    t0 = time.time()
    cf = CF.load(chip_path)
    sinks = MU.fit_anti_sink(model_dir, "cuda")
    MU.fit_anti_sink = lambda *x, **k: sinks  # (wide_state imports it at call time)
    sd_p, tc_p, meta_p = wide.wide_state(model_dir, c, T, "cuda")
    sd_f, tc_f, meta_f = wide.wide_state(model_dir, None, T, "cuda", cf)
    diff = {}
    for k in sd_p:
        x, y = sd_p[k], sd_f.get(k)
        if y is None or x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            diff[k] = None if y is None else int((x != y).sum()) if x.shape == y.shape else "shape"
    res = {"tensors": len(sd_p), "tensors_only_in_file_build": sorted(set(sd_f) - set(sd_p)),
           "tensors_differing": diff, "config_equal": tc_p == tc_f,
           "meta_python": {k: v for k, v in meta_p.items() if k not in ("code_dims",)},
           "meta_file": {k: v for k, v in meta_f.items() if k not in ("code_dims",)},
           "code_dims_equal": meta_p["code_dims"] == meta_f["code_dims"],
           "embed_bytes": int(sd_p[wide.DST + "embed_tokens.weight"].numel() * 2), "s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
