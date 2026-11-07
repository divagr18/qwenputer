"""Generate a snapshot buffer for Qwenputer (RISC-V) in the format zig-ai expects.

Usage:
    python3.14 gen_snapshot.py <model_dir> <elf_path> <output.bin>

Output: a flat f32 binary file of size 11 * 32 * 128 * 128 (5,767,168 floats = 23 MB).
Each float is a 32-bit little-endian float. The buffer can be loaded by zig-ai
and injected via restoreGpuState().

The format:
    For each chip layer L (0..10):
      For each value head vh (0..31):
        base = (L * 32 + vh) * 128 * 128
        matrix[128][128] = 0
        For each (row, vec) in init[vh]:
            for col in 0..128:
                matrix[row][col] = vec[col]
"""

import argparse
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, '/ai/repos/2026/qwenputer/experimental/riscv')

from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S

HD = 128  # key/value head dim
N_LAYERS = 11
N_VALUE_HEADS = 32
STATE_DIM = HD  # 128
VALUE_DIM = HD  # 128
BUF_SIZE = N_LAYERS * N_VALUE_HEADS * STATE_DIM * VALUE_DIM  # 5,767,168


def build_snapshot(model_dir, elf_path):
    """Build the snapshot buffer from an ELF firmware."""
    # 1. Parse ELF → machine state
    prog = elf.program(elf_path)
    machine = ref.Machine.from_program(prog)

    # 2. Build circuit
    c = core.build()

    # 3. Snapshot: fill c.tables[...].init
    core.snapshot(c, machine)

    # 4. Map head nodes to (layer, key_head)
    slots = hf.head_slots(c)

    # 5. Build the buffer
    buf = np.zeros(BUF_SIZE, dtype=np.float32)

    r = c.target.value_heads // c.target.key_heads  # = 2

    for ni, (L, j) in slots.items():
        node = c.nodes[ni]
        init = node.init if node.init is not None else []
        n_vheads = len(node.vheads()) if hasattr(node, 'vheads') else 1

        for x in range(n_vheads):
            vh_idx = j * r + x
            base = (L * N_VALUE_HEADS + vh_idx) * STATE_DIM * VALUE_DIM
            # Zero (already zero from np.zeros)
            # Fill rows
            init_x = init[x] if x < len(init) else {}
            for row, vec in init_x.items():
                vec_arr = np.asarray(vec, dtype=np.float32).flatten()
                for col in range(min(len(vec_arr), VALUE_DIM)):
                    buf[base + row * VALUE_DIM + col] = vec_arr[col]

    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("elf_path")
    ap.add_argument("output")
    args = ap.parse_args()

    buf = build_snapshot(args.model_dir, args.elf_path)
    buf.tofile(args.output)
    print(f"Wrote {len(buf)} floats ({len(buf)*4} bytes) to {args.output}")
    # Stats
    nonzero = (buf != 0).sum()
    print(f"Non-zero floats: {nonzero} ({100*nonzero/len(buf):.2f}%)")


if __name__ == "__main__":
    main()
