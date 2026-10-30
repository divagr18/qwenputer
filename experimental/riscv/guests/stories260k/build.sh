#!/bin/bash
# build.sh NAME CFLAGS...  -> NAME.elf on the VDOT memory map (vd.ld)
set -e
cd "$(dirname "$0")"
ENV=../../chipformer/rv/env
PICO=/usr/lib/picolibc/riscv64-unknown-elf
GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
LIBS="-Wl,--start-group $PICO/lib/release/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group"
N=$1; shift
$GCC -march=rv32im_zba -mabi=ilp32 -O2 "$@" -isystem $PICO/include -I$ENV -I. \
  -nostdlib -nostartfiles -static -Tvd.ld -o $N.elf $ENV/crt0.S ${SRC:-stories6.c} vd2.S wrom_data2.c $ENV/sys.c $LIBS
riscv64-unknown-elf-size -A $N.elf | grep -E "text|rodata|bss|wrom"
