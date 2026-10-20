# A RISC-V computer inside Qwen

Model: [divagr1925/Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer).

This folder compiles a 32-bit **RV32IM** CPU into stock Qwen3.5 layers, using the same compiler as
the main project. The CPU also has Zba and a bignum coprocessor. It is a self-contained snapshot:
it carries its own copy of the `chipformer` package, because its compiler changes haven't been
merged with the main tree yet.

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

## Filesystem

This branch adds a filesystem to the same layers: five tools, `put`, `append`, `get`, `list` and
`delete`, that Qwen calls next to `calc` and the weights answer. Files live in Gated DeltaNet
state behind a hashed directory, and a session's store can be saved and loaded into the next one.
Model: [divagr1925/Qwen3.5-4B-Qwenputer-Filesystem](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-Filesystem).
[`docs/rv-store.md`](docs/rv-store.md) has the design, the checks and the benchmarks.

```bash
cd experimental/riscv && uv sync && uv run pytest tests -q
```
