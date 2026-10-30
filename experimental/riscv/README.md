# A RISC-V computer inside Qwen

Model: [divagr1925/Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) on Hugging Face, or [Spartan1925/Qwen3.5-4B-Qwenputer](https://www.modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer) on ModelScope.
This branch (`npu`) is the code for [divagr1925/Qwenputer-npu](https://huggingface.co/divagr1925/Qwenputer-npu)
([ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwenputer-npu)); the published Qwenputer
runs from `main`.

This folder compiles a 32-bit **RV32IM** CPU into stock Qwen3.5 layers, using the same compiler as
the main project. The CPU also has Zba and a bignum coprocessor. It is a self-contained snapshot:
it carries its own copy of the `chipformer` package, because its compiler changes haven't been
merged with the main tree yet.

## The NPU instructions (this branch)

The CPU gets a neural-network instruction set, enough to run a second language model inside Qwen:

- `VDOT rd, rs1, rs2`: 8 int8 activations (two registers) times the next 8 int4 weights of a
  weight memory, added to `rd`; `WB rs1` sets where the weights start.
- `SMAQA rd, rs1, rs2`: four signed byte products added to `rd` (the P extension's semantics).
- The weight memory: 32,640 words of 8 int4 weights in DeltaNet heads, from the ELF's `.wrom`.

[`guests/stories260k/`](guests/stories260k) runs Karpathy's stories260K as a C program on it, at
225,000 trace tokens per story token (2.43M in plain C). `chipformer/rv/fusedexec.py` is a faster
executor: about 41 s per story token on an L4. `tests/test_rv_vdot.py` checks the circuit against
the reference, and `python -m chipformer.rv.npucheck MODEL` checks a merged checkpoint's layers.

## What works

- The official riscv-tests suites `rv32ui`, `rv32um` and `rv32uzba` pass. They are checked against
  the reference model and against qemu.
- gcc-compiled C runs exactly: crc32, qsort, a sieve, mini-gmp and a fuzz suite, about 18M
  instructions checked free-running. A patched MicroPython build also runs exactly.
- A `calc` firmware with bignum `+ - * / // % ** gcd` is exact on 10,000 random cases.
- `rv-tiny`, a standalone stock `Qwen3_5ForCausalLM`, runs programs from a memory snapshot under
  plain `generate()`.
- Inside Qwen3.5-4B, Qwen calls `calc` and the RISC-V machine in the weights answers. A replay of
  the evaluation records found 0 mismatches.

The machine has 256 KB of code ROM, 128 KB of read-only data and 32 KB of RAM, all held in Gated
DeltaNet state. It takes 11 chip layers and about 1M nonzero weights, and costs about 1.2-1.4
tokens per instruction. A compact executor runs about 3,700 instructions a second per stream on
an L4, and in generation it runs at about 3,200 trace tokens a second.

[`docs/rv32.md`](docs/rv32.md) has the design and every measurement, and
[`docs/rv32-p3-status.md`](docs/rv32-p3-status.md) has the in-Qwen status.

```bash
cd experimental/riscv && uv sync --extra hf && uv run pytest tests -q
```
