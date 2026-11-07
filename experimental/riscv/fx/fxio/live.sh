#!/bin/bash
# The live end-to-end run: fx's exact RV32 binaries in rv32emu, real Qwen3.5-4B on the box's GPU.
#
# Copy to $P first: the repo's experimental/riscv/{fx,chipformer} directories as $P/fx and
# $P/chipformer, and the two images as $P/row24 and $P/row11 (fx-fastcopy.elf and imports.json).
# usage: live.sh [REPS] [setup|row24|row11|units]   (no second argument: all of them)
set -eu
P=/opt/dlami/nvme/fxio
REPS=${1:-2}
WHAT=${2:-all}
cd $P
export HF_HOME=$P/hf PIP_CACHE_DIR=$P/pipcache TMPDIR=$P/tmp
mkdir -p $TMPDIR
if [ ! -d venv ]; then
  python3 -m venv venv
fi
. venv/bin/activate
if [ "$WHAT" = setup ] || [ "$WHAT" = all ]; then
  pip install -q torch transformers accelerate pyelftools wasmtime jinja2 tokenizers numpy
  gcc -O2 -shared -fPIC fx/rv/rv32emu.c -o rv32emu.so
fi
M=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B'))")
O=$P/out; mkdir -p $O
TASKS="t1_bug t2_rename t3_feature t4_config t5_parse"
run() { # row rep task flags
  f=$O/r$1_$3_$2.json
  [ -f $f ] && return 0
  python -I fx/fxio/e2e.py ./rv32emu.so row$1/fx-fastcopy.elf row$1/imports.json "$M" $f \
    --task $3 --backend hf:Qwen/Qwen3.5-4B --seed $2 $4 2>&1 | tail -1 | tee -a $O/live.log
}
if [ "$WHAT" = row24 ] || [ "$WHAT" = all ]; then
  for r in $(seq 1 $REPS); do for t in $TASKS; do run 24 $r $t --native; done; done
fi
if [ "$WHAT" = row11 ] || [ "$WHAT" = all ]; then
  for r in $(seq 1 $REPS); do for t in $TASKS; do run 11 $r $t ""; done; done
fi
if [ "$WHAT" = units ] || [ "$WHAT" = all ]; then
  # the units re-measure each run from its packets, with no GPU
  for f in $O/r*_?.json; do
    b=$(basename $f .json); row=$(echo $b | sed -E 's/r([0-9]+)_.*/\1/'); t=$(echo $b | sed -E 's/r[0-9]+_(.*)_[0-9]+/\1/')
    N=""; [ $row = 24 ] && N=--native
    [ -f $O/${b}_units.json ] || python -I fx/fxio/e2e.py ./rv32emu.so row$row/fx-fastcopy.elf row$row/imports.json "$M" \
      $O/${b}_units.json --task $t $N --units all --backend replay:$f 2>&1 | tail -1 | tee -a $O/live.log
  done
fi
echo live.sh finished
