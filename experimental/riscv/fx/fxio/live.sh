#!/bin/bash
# The live end-to-end run: fx's exact RV32 binaries in rv32emu, real Qwen3.5-4B on the box's GPU.
# NOT RUN YET: written while no GPU box was reachable (docs/fx/io.md, Blockers). The HF backend
# is tested on a tiny Qwen3_5 only, so expect to fix something on the first real run.
#
# Copy to $P first: the repo's experimental/riscv/fx directory as $P/fx, and the two images as
# $P/row24 and $P/row11 (fx-fastcopy.elf and imports.json of each).
# usage: live.sh [REPS]
set -eu
P=/opt/dlami/nvme/fxio
REPS=${1:-2}
cd $P
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q torch transformers accelerate pyelftools wasmtime jinja2 tokenizers
gcc -O2 -shared -fPIC fx/rv/rv32emu.c -o rv32emu.so
export HF_HOME=$P/hf
M=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B'))")
O=$P/out; mkdir -p $O
for r in $(seq 1 $REPS); do
  for t in t1_bug t2_rename t3_feature t4_config t5_parse; do
    python -I fx/fxio/e2e.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json "$M" $O/r24_${t}_$r.json \
      --task $t --native --backend hf:Qwen/Qwen3.5-4B --seed $r | tee -a $O/live.log
  done
done
# row 11, one rep, for comparison
for t in t1_bug t2_rename t3_feature t4_config t5_parse; do
  python -I fx/fxio/e2e.py ./rv32emu.so row11/fx-fastcopy.elf row11/imports.json "$M" $O/r11_${t}_1.json \
    --task $t --backend hf:Qwen/Qwen3.5-4B --seed 1 | tee -a $O/live.log
done
# the units re-measure each run from its packets, with no GPU
for f in $O/r24_*_?.json; do
  t=$(basename $f .json | sed -E 's/r24_(.*)_[0-9]+/\1/')
  python -I fx/fxio/e2e.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json "$M" ${f%.json}_units.json \
    --task $t --native --units all --backend replay:$f | tee -a $O/live.log
done
