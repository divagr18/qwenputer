<p align="center">
  <img src="docs/qwenputer-banner.svg" alt="Qwenputer: a computer, compiled into the weights of a language model." width="100%">
</p>

Qwenputer model: [Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) · [ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer)  
Filesystem model: [Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-Filesystem) · [ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer-Filesystem)  
NPU model: [Hugging Face](https://huggingface.co/divagr1925/Qwenputer-npu) · [ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwenputer-npu)  
QwenPU, the vector unit: [GitHub](https://github.com/divagr18/QwenPU) · [Hugging Face](https://huggingface.co/divagr1925/QwenPU)

**Qwenputer is a new kind of experimental model: Qwen3.5-4B with a 32-bit RISC-V CPU compiled and
inserted directly into its weights.** It isn't a plugin, a tool or a sandbox. The CPU is 11
transformer layers in front of Qwen's own, and none of their weights are learned: a compiler
computes every one.

It uses the model's own Gated DeltaNet state as RAM. Qwen3.5's DeltaNet layers carry a running
state that the delta rule can overwrite one row at a time, exactly, so each row works as a memory
cell. The CPU keeps its code, data, registers and program counter there.

The CPU runs C and MicroPython code inside the model's forward passes, and passes the official
riscv-tests suites.

<p align="center"><img src="docs/benchmarks-arith.png" alt="Qwenputer vs base Qwen3.5-4B and base with a Python tool on NUPA and DeepMind arithmetic sets." width="100%"></p>

## Benchmarks

Greedy decoding, 40 problems per set. Each row compares base Qwen3.5-4B and Qwenputer on the same
problems. Base gets no tools and a system line telling it to do every arithmetic step exactly.
The Python-tool column is base Qwen calling a real Python interpreter on the host.

| Benchmark | Base Qwen3.5-4B | **Qwenputer** | Python tool (host) |
|---|---|---|---|
| Synthetic integer arithmetic | 75.0 | **97.5** | 97.5 |
| DeepMind `mul_big` | 55.0 | **97.5** | 100 |
| DeepMind `div_big` | 72.5 | **97.5** | 95.0 |
| DeepMind `mul_div_multiple_longer` | 90.0 | **100.0** | 97.5 |
| DeepMind `add_or_sub_big` | 87.5 | **97.5** | 97.5 |

We replayed about 230,000 of the computer's output positions through a reference model of the
machine and found zero mismatches. Qwen's text behaviour matches
base: KL 6e-4, against a noise floor of 5e-4.

## Quickstart

```bash
git clone https://github.com/divagr18/qwenputer && cd qwenputer/experimental/riscv && uv sync --extra hf
hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-rv
uv run python -m chipformer.rv.chat qwenputer-rv --firmware tests/data/rv/calc-bridge.elf \
    --ask "Use calc: what is 123456789 * 987654321?"
```

Qwen uses the CPU entirely internally; it is not invoked through an external tool call. Instead,
it piggybacks on the tool-call format Qwen3.5 was trained on: Qwen writes a `<tool_call>` block for
`calc` exactly as it would for any tool, but nothing outside the model ever runs it. The computer
layers read the call, run it on the RISC-V CPU, and the model itself writes the `<tool_response>`,
digit by digit. No runtime, sandbox or tool executor is involved.

The model loads as a stock `Qwen3_5ForCausalLM`. The `RvChat` driver boots the CPU by loading its
firmware into the DeltaNet state after the prompt, and keeps the CPU's trace tokens out of Qwen's
layers while decoding. Every token still comes from the weights. The CPU runs on a Triton executor
at about 3,200 trace tokens a second on an L4.

The CPU is **RV32IM** with Zba and a bignum coprocessor. Its code lives in
[`experimental/riscv/`](experimental/riscv), and
[`experimental/riscv/docs/rv32.md`](experimental/riscv/docs/rv32.md) has the design and every
measurement.

## Gated DeltaNet as RAM

<p align="center"><img src="docs/gdn-ram.png" alt="Gated DeltaNet state used as RAM: a one-hot key overwrites one row, a one-hot query reads it back, and the CPU's registers, memory and program counter live in DeltaNet heads." width="100%"></p>

The computer's RAM is the state of Qwen3.5's own Gated DeltaNet layers. Qwen3.5 mixes full attention with these layers, and each DeltaNet head keeps a state matrix that it updates at every token with the gated delta rule (panel **a**). In the computer's heads the gate is held open (α = 1, so nothing fades) and the write strength is β = 1. Each key is a one-hot vector picking a single row, so every update replaces exactly that row with the value and leaves the rest untouched. That's a RAM write. A RAM read is the same head's query: a one-hot query returns exactly that row (panel **b**).

The rest is layout (panel **c**). The low bits of an address choose the head and the remaining
bits choose the row, and each row stores one byte as bits. The machine's memory is spread over a
few dozen heads, and its registers, program counter and call stack each get heads of their own.
Each key head drives two value heads, which doubles the capacity. The gate is useful too:
α = 0 for one token wipes a whole head (that's how the CHIP-8 screen is cleared), and α = ½ makes a head
halve every step, which the circuit uses as a counter.

Writes arrive through the token stream, so the trace is the computer's execution log. An `ADDR`
token names the row, the `VAL` token after it carries the value, and `STEP` ends the instruction.
Nothing here needed a new layer type: the computer's memory is the same Gated DeltaNet
recurrence Qwen3.5 already uses for language.

## In-weights NPU (experimental)

The CPU can run neural networks too. The model is up as
[`divagr1925/Qwenputer-npu`](https://huggingface.co/divagr1925/Qwenputer-npu)
(also on [ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwenputer-npu)), and its code
lives on the [`npu`](https://github.com/divagr18/qwenputer/tree/npu/experimental/riscv) branch.

It's the same RISC-V model with three more instructions and a weight memory:
- `VDOT` multiplies 8 int8 activations by the next 8 int4 weights and adds the sum to a register.
  `WB` sets where the weights start.
- `SMAQA` adds four signed byte products to a register, as in the RISC-V P extension.
- The weight memory holds 261,120 int4 weights in DeltaNet heads, loaded from the program.

Each is one instruction and one trace token, computed exactly by the compiled layers. With them,
Karpathy's stories260K, a 260K-parameter Llama, runs as a C program on the CPU inside Qwen and
writes its stories. Its weights are data in the CPU's memory, not compiled into the checkpoint,
so any model that fits runs the same way.

| Build of stories260K | Trace tokens per story token |
|---|---|
| Plain C, int8 weights | 2,430,000 |
| **VDOT + SMAQA, int4 weights** | **225,000** |

A new fused executor runs the CPU at about 5,500 trace tokens a second on an L4, so a story token
takes about 41 seconds. The model card has the quickstart and the checks.

**Next: [QwenPU](https://github.com/divagr18/QwenPU).** It gives the CPU a vector unit of its own,
with its own memories in DeltaNet heads. The same stories260K story takes 53,000 trace tokens per
story token instead of 225,000, about 16 seconds each on an L4. It's a work in progress: exact,
but its checkpoint is 26.7 GB until the hidden size is trimmed.

## In-weight filesystem (experimental)

The next logical step was to put a filesystem inside the weights. This is experimental and still
in progress: the model is up as
[`divagr1925/Qwen3.5-4B-Qwenputer-Filesystem`](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-Filesystem)
(also on [ModelScope](https://www.modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer-Filesystem)),
and its code lives on the [`rv-store`](https://github.com/divagr18/qwenputer/tree/rv-store/experimental/riscv)
branch for now.

It's the same RISC-V model with five more tools next to `calc`: `put`, `append`, `get`, `list` and
`delete`. When Qwen calls one, the compiled layers hash the file name to a slot in a directory kept
in DeltaNet state, so a lookup takes two or three trace tokens instead of a scan.

The file's text sits in other DeltaNet heads, eight tokens to a row, and the layers print it back into the
`<tool_response>` token for token. It holds up to 512 files, about 53,000 tokens of text. Save
those heads at the end of a conversation, load them at the start of the next one, and the files
are still there.

| Benchmark | Base Qwen3.5-4B | **Qwenputer Filesystem** | Python tool (host) |
|---|---|---|---|
| Multi-session recall, 90 questions | 7.8 | **83.3** | 78.9 |
| Facts that changed in a later session, 30 questions | 3.3 | **90.0** | 83.3 |

The model card has the quickstart and the full results.

## Version 1 (CHIP-8)

[`divagr1925/Qwen3.5-4B-Qwenputer-CHIP8`](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-CHIP8)
is the first Qwenputer. It compiles a CHIP-8 machine into Qwen3.5-4B instead: a CPU, a big-number
maths library and its memory, written as transformer layers and appended to the stock model. Its
code is at the top level of this repo (`chipformer/q35/`).

When the CHIP-8 model writes `48213 × 9917 =`, the answer isn't a guess. The computer in its
weights runs the multiplication, and the model writes `478128321` itself, digit by digit, exactly.
It has never printed a wrong answer, from 3-digit sums to 20-digit products, and it still loads
like any other Qwen checkpoint.

<details><summary>CHIP-8 benchmarks, how it works and quickstart</summary>

### Benchmarks

These are Qwen3.5-4B scores, greedy decoding. Each row compares base Qwen and the CHIP-8 model on
the same problems. The Python-tool column is base Qwen calling a real
Python interpreter on the host, which is the ceiling for this kind of help.

| Benchmark | Base Qwen3.5-4B | **Qwenputer CHIP-8** | Python tool (host) |
|---|---|---|---|
| NUPA integer add / sub / mul, 3-20 digits (576) | 48.6 | **100.0** | 100.0 |
| DeepMind arithmetic, 6 sets pooled (600) | 80.8 | **92.3** | 94.2 |
| DeepMind `div_big` (100) | 77 | **97** | 85 |
| DeepMind `mul_big` (100) | 66 | **93** | 91 |
| DeepMind clean big add / mul (200) | 69.5 | **99.0** | 98.5 |

The computer has never printed a wrong result. We replayed about 3.9 million positions of model
output through a reference simulator of the circuit and found zero mismatches. MMLU: 71.9 vs 71.7 for base (99% identical answers, KL 7e-4).

It does not help everywhere. GSM8K and GSM-Hard are ties, because word problems are limited by
reasoning rather than arithmetic. On the long DeepMind expression chains (`mixed_longer`,
`mul_div_multiple_longer`) base Qwen is still ahead. Qwen tends to re-verify exact results it
has already been given, until it runs out of its token budget, and the next release targets
this.

These numbers are the best mode per benchmark across our development checkpoints. The
released checkpoint contains all the features behind them, and its own full sweep will replace
this table.

### How it works

```
Qwen writes        "48213 × 9917 ="
computer layers    read the digits, run the BIOS, emit trace tokens
the model prints   " 478128321"
Qwen continues     as if it had written the number itself
```

**Running.** Each generated trace token is one write to the computer's memory (see
[Gated DeltaNet as RAM](#gated-deltanet-as-ram)). A few tokens make one
instruction, so the trace is the machine's execution log. The MLPs do the arithmetic as exact
lookup tables, and every value stays exact in bf16.

**The bridge.** The computer layers sit before Qwen's and read Qwen's tokens: digits, operators,
tool-call tags. When a calculation is complete they run it and print the answer as normal text.
Qwen's state is frozen while the computer runs.

**What it can do.** The computer is about 0.7M nonzero weights covering add, subtract, multiply, exact division
(reduced fractions when it doesn't terminate), signed numbers, mod, gcd, and whole expressions with
brackets, powers and decimals. Stock checkpoints give
every layer the same width, so the merged model is 14.65 GB vs 9.3 GB for base Qwen. The extra is
almost all zeros, so the download is about the same size.

More in [`docs/`](docs): the architecture ([`q35.md`](docs/q35.md)), the arithmetic library
([`q35-skills.md`](docs/q35-skills.md)), an exact in-weights object store
([`q35-store.md`](docs/q35-store.md)) and shrinking the checkpoint
([`q35-size.md`](docs/q35-size.md)).

### Is there really no tool call?

There is no tool. Qwen can ask for a calculation in two ways, and in neither case does anything
outside the model see it.

It can write the calculation inline, as it would in any answer: `48213 × 9917 =`. The model then
continues with ` 478128321`.

It can also use the tool-call syntax it was trained on, writing `<tool_call><function=mul>…`.
Nothing intercepts that. No runtime parses it and no server executes it. The `<tool_response>`
that follows, with the exact product inside, is predicted by the model's own layers, token by
token, exactly as it predicts every other token.

You can check this yourself: the checkpoint runs under stock `model.generate()` with no tools
registered, no stopping criteria and no callbacks. The computer's work is visible in the token
stream, as a burst of special "trace" tokens. These are the machine's execution log, and Qwen
then reads the printed answer like ordinary text.

### Quickstart

```bash
git clone https://github.com/divagr18/qwenputer && cd qwenputer
uv sync --extra hf
hf download divagr1925/Qwen3.5-4B-Qwenputer-CHIP8 --local-dir qwenputer-4b

uv run python -m chipformer.q35.demo qwenputer-4b                   # Qwen uses its tool-call syntax
uv run python -m chipformer.q35.demo qwenputer-4b --inline --nudge  # plain inline arithmetic
```

Under the hood every prompt starts with the computer's BIOS. That's a boot record of chip
tokens, which puts the arithmetic library into the computer's memory, followed by Qwen's normal
chat template. After that it's a plain `generate()` call:

```python
import json, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from chipformer.q35.demo import TOOLS
from chipformer.q35.lib import bcd
from chipformer.ref.emu import Chip8
from chipformer.trace import boot_tokens

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # keep matmuls exact

path = "qwenputer-4b"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
off = json.load(open(f"{path}/chipformer.json"))["chip_token_offset"]
bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]

msgs = [{"role": "user", "content": "What is 48213 * 9917?"}]
ids = bios + tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                     enable_thinking=False)["input_ids"]
out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=4096, do_sample=False,
                     cache_implementation="static")
print(tok.decode(out[0, len(ids):], skip_special_tokens=True))
```

The model needs no custom modelling code. [`docs/running.md`](docs/running.md) covers the
details:
- the tools and system prompts that work best
- the generation settings shipped in `generation_config.json`
- the optional decoder settings our benchmarks used (`fp` and `tc` in `BatchDecoder`), which make
  inline mode stronger

</details>

## Citing

If you use Qwenputer in your work, please cite it. GitHub's "Cite this repository" button reads
[CITATION.cff](CITATION.cff); in BibTeX:

```bibtex
@software{agrawal2026qwenputer,
  author  = {Agrawal, Divyansh},
  title   = {{Qwenputer}: a computer compiled into the weights of a pretrained language model},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/divagr18/qwenputer},
  license = {MIT}
}
```

## License

The code is MIT. The model weights are Apache-2.0, as a modified version of
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): the config is widened and computer
layers are added, but Qwen's own weights are unchanged.
