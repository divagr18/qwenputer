# stories260K on the RISC-V Qwenputer

Karpathy's [stories260K](https://huggingface.co/karpathy/tinyllamas) (a 260K-parameter Llama) as a
C program on the circuit's RV32IM CPU. Its weights live in the CPU's memory and every multiply is
an instruction the circuit executes; nothing about the model is compiled into the checkpoint.

Two ISA extensions make it practical (`chipformer/rv/ref.py`, `core.py`):
- `WB rs1` sets the weight base; `VDOT rd, rs1, rs2` adds the dot product of 8 int8 activations
  (two registers) with the next 8 int4 weights of the weight memory (WROM) to `rd`.
- `SMAQA rd, rs1, rs2`: four signed byte products added to `rd` (the P extension's semantics),
  for attention.

| Build | Trace tokens per story token |
|---|---|
| Plain C, int8 | 2.43M |
| Unrolled kernels, no extensions | 1.11M |
| VDOT + SMAQA, int4 (this) | 180k at position 0, 225k from position 31 |

The int4 weights (`qat_int4.npz`) come from self-distillation against the float model on its own
samples (`qat.py`, KL 0.019, top-1 agreement 93.8%). Greedy output on the CPU matches the float
model for 60 tokens.

## Run
`stories260k.elf` is the prebuilt program (60 story tokens, about 41 s each on an L4):
```sh
cd ../.. && python guests/stories260k/storyrun.py guests/stories260k/stories260k.elf
```

## Build
```sh
curl -LO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -LO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
python export_vd2.py qat_int4.npz        # weights_vd2.h, wrom_data2.c
python gen_vd2.py                        # vd2.S (the VDOT matvec kernels)
SRC=stories6.c ./build.sh vdq -DNTOK=60  # riscv64-unknown-elf-gcc + picolibc
cd ../.. && python guests/stories260k/trace.py guests/stories260k/vdq.elf   # reference machine
python guests/stories260k/storyrun.py guests/stories260k/vdq.elf           # GPU executor, checked
```
`stories.c` is the plain int8 version (memory map `big.ld`).
