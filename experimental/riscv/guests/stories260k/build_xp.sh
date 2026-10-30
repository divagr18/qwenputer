#!/bin/bash
# build_xp.sh NAME CFLAGS...  -> NAME.elf, like build.sh but with the xPack riscv-none-elf-gcc and its
# newlib (only memset / memcpy and friends are linked; exit is the HALT ecall in exit_xp.c)
set -e
cd "$(dirname "$0")"
ENV=../../chipformer/rv/env
GCC=${XPACK:-/d/tools/xpack-riscv-none-elf-gcc-14.3.0-1}/bin/riscv-none-elf-gcc
N=$1; shift
$GCC -march=rv32im_zba -mabi=ilp32 -O2 "$@" -I$ENV -I. -nostartfiles -static -T${LD:-vd.ld} -o $N.elf \
  $ENV/crt0.S ${SRC:-stories6.c} ${ASM-vd2.S} ${WROM:-wrom_data2.c} exit_xp.c -Wl,--start-group -lc -lgcc -Wl,--end-group
