#!/bin/sh
# Builds the RV32 image of fx-core: fx-core.wasm -> wasm2c -> zig cc (riscv32).
#   build_rv.sh FX_CORE_WASM OUT_DIR OPT CPU
# e.g. build_rv.sh fx-core.wasm out -O2 generic_rv32+m+zba
# Needs ZIG and WABT (wabt-1.0.42 directory) in the environment.
set -e
WASM=$1; OUT=$2; OPT=${3:--O2}; CPU=${4:-generic_rv32+m+zba}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
mkdir -p "$OUT"
[ -f "$OUT/fxcore.c" ] || "$WABT/bin/wasm2c.exe" "$WASM" -o "$OUT/fxcore.c" -n fxcore
[ -f "$OUT/fxcore_unchecked.c" ] || $PY "$HERE/patch_unchecked.py" "$OUT/fxcore.c" "$OUT/fxcore_unchecked.c"
$PY "$HERE/gen_imports.py" "$OUT/fxcore.h" "$OUT/imports.c" "$OUT/imports.json"
INC="-I$OUT -I$WABT/include"
T="-target riscv32-linux-musl -mcpu=$CPU -fno-stack-protector -ffunction-sections -fdata-sections -DNDEBUG"
W2C="-DWASM_RT_MEMCHECK_BOUNDS_CHECK=1 -DWASM_RT_NONCONFORMING_UNCHECKED_STACK_EXHAUSTION=1"
TAG=$(echo "$OPT-$CPU" | tr '+' '_')
"$ZIG" cc $T $INC $W2C $OPT -c "$OUT/fxcore_unchecked.c" -o "$OUT/fxcore-$TAG.o"
"$ZIG" cc $T $INC $W2C -O2 -ffreestanding -fno-builtin -c "$HERE/rt.c" -o "$OUT/rt-$TAG.o"
"$ZIG" cc $T $INC $W2C -O2 -c "$OUT/imports.c" -o "$OUT/imports-$TAG.o"
"$ZIG" cc -target riscv32-freestanding-none -mcpu=$CPU -nostartfiles -Wl,--gc-sections -Wl,-T,"$HERE/link.ld" \
  "$OUT/rt-$TAG.o" "$OUT/imports-$TAG.o" "$OUT/fxcore-$TAG.o" -o "$OUT/fx-$TAG.elf"
echo "$OUT/fx-$TAG.elf"
