# Shrinking the Qwenputer checkpoint

Status 2026-11-07, branch `fx-size`. Tools: `fx/size/`. Numbers: `docs/fx/size.json`, rebuilt from the
measurements by `fx/size/report.py`. **M** = measured here. **E** = estimated: either arithmetic on a
file that was not built, or a judgement. **R** = read from another doc, not re-measured.

CPU only, no GPU box, nothing downloaded whole. Tensors were read by HTTP range requests.

## Result

Today's 18.7 GB is 45% Qwen. The fx build would be 37.25 GB (E), 23% Qwen, and it does not fit a 24 GB
L4 or A10G as a stock model. Almost all the rest is zeros: the chip is 6.48M nonzero weights.

| build | disk | VRAM | Qwen | chip | runs on |
|---|---|---|---|---|---|
| rv-q4p2 today | 18.72 GB (M) | 18.72 GB | exact | exact | stock |
| fx, as designed | 37.25 GB (E) | 37.25 GB | exact | exact | stock, if it fits |
| fx, best stock, all lossless steps | 24.9 GB (E) | 35.0 GB | exact | exact | stock |
| **fx, chip split out, exact** | **8.42 GB (E)**, 5.5 GB on the wire | **8.5 GB (E)** | **exact, base's own bytes** | **exact** | executor, or `trust_remote_code` |
| fx, split + mixed quantisation | see Quantising Qwen | | changed | exact | same |

The smallest size that stays honest and exact is **8.42 GB**: base Qwen3.5-4B's text weights, unchanged,
plus about 8 MB of chip. Below that, Qwen's weights have to change. See [Quantising Qwen](#quantising-qwen).

## What the file is made of

From the safetensors headers of `divagr1925/Qwen3.5-4B-Qwenputer` (`main`, b14ec56, 11 chip layers, 32
Qwen layers, hidden 3968, vocabulary 478,656). All M.

| part | bytes | share | what it is |
|---|---|---|---|
| Qwen's own weights | 8.412 GB | 45.0% | |
| padding on Qwen's 32 layers | 3.926 GB | 21.0% | zeros, plus one mute column |
| 11 chip layers | 3.850 GB | 20.6% | 1,044,349 nonzeros, 0.054% dense |
| embedding, chip rows | 1.828 GB | 9.8% | 230,336 rows, about 92 nonzeros each |
| embedding, chip columns on Qwen's rows | 0.699 GB | 3.7% | about 88 nonzeros a row |
| total | 18.715 GB | | 55% of it is not Qwen |

Base Qwen3.5-4B's text model by family (M): FFN 4.530 GB, GDN projections 2.013 GB, embedding (tied to
`lm_head`) 1.271 GB, full attention 0.587 GB, `a`/`b` 0.008 GB, norms and the rest 0.002 GB.

**The size law.** `model.py` gives the size of any stock build from its shapes:

    bytes = 2 h [ (n_chip + 24) * 44,096 + 8 * 41,984 + vocab ] + small terms

It reproduces both published checkpoints to the byte (18,715,212,672 and 26,728,967,552, M). Everything
is linear in hidden size. A stock layer costs 88 KB per hidden dim whatever it holds, because
`intermediate_size` and the head counts are global: a layer that is only a RAM bank still carries a
9,216-unit MLP.

**Padding, sampled** (M, range reads). Nonzero fraction, and zstd-19 ratio:

| tensor family | nonzero | zstd-19 |
|---|---|---|
| chip `in_proj_qkv` | 0.16% | 0.0005 |
| chip `gate_proj` | 0.11% | 0.0006 |
| chip `down_proj` | 0.001% | 0.00007 |
| chip embedding rows | 2.3% | 0.0013 |
| padded Qwen `in_proj_qkv`, `gate_proj` | 64.5% (2560 / 3968) | 0.51 |
| padded Qwen `out_proj`, `down_proj` | 66.7% of sampled rows | 0.52 |
| base Qwen, any family | 100% | 0.78; 0.67 with hi/lo bytes split |
| base embedding | 100% | 0.61; 0.52 split |

Chip weights use 809 distinct values on the fx circuit (M). Values reach at least 2^49, so an integer dtype
does not fit, and a dtype change would not remove a zero anyway.

**Qwen's tensors are base's bytes.** Sampled rows of every tensor family, in 3 Qwen layers and the
embedding, equal base Qwen3.5-4B exactly in Qwen's columns (M). Only two things differ: `input_layernorm`
and `post_attention_layernorm` (rescaled for the wider norm), and one mute column on `in_proj_a/b/z`,
`q_proj` and `k_proj`, at column 2776 (`mu`).

**The program is not in the checkpoint.** ROM, RAM and registers are state, injected into the cache from
a snapshot (`hf.inject`, `core.snapshot`). No ROM byte is in any weight tensor, on rv-q4p2 or on fx. A
memory layer's weights are only its addressing. fx's image is 7.3 MB of ROM plus 0.5 MB of data (R), and
a release has to ship it. The state costs 110 MB of VRAM (R, `memory.md`).

## The fx build

Measured on the real design, `fx-mem` 76a75de, `core.build(text=T, cfg=fx)`, i.e. with the text bridge a
merged checkpoint needs (M). `memory.md`'s 2,987 dims, 45,973 units and 850 heads are the same circuit
without the bridge.

| | rv-q4p2 | fx, merged |
|---|---|---|
| chip layers | 11 | 59 |
| IR dims | 2,918 | 3,226 |
| FFN units | 42,114 | 45,826 |
| heads | 103 | 854 |
| residual slots | 1,265 | 1,396 |
| hidden | 3968 (m 62) | 4096 (m 64), needs 4,021 |
| chip vocabulary | 230,324 | 302,329 |
| nonzero chip weights | 1,044,349 | 6,480,659 |
| layers with an all-zero MLP | 1 | 44 |

**37.25 GB (E)**: exact arithmetic on a file nobody has built. `memory.md` says about 35 GB, scaling 48
layers at 0.34 GB. The other 2 GB is the hidden step from 3968 to 4096 and 71,936 more embedding rows.

| where the 18.5 GB of growth goes (E) | GB |
|---|---|
| 48 more layers, at 361 MB | 17.34 |
| hidden 3968 to 4096 on the existing 18.7 GB | 0.60 |
| 71,936 more vocabulary rows | 0.59 |

Of 37.25 GB: chip layers 21.32, Qwen layers widened to 4096 11.42, embedding 4.51. It would need about
58 GB of RAM to merge (R).

## Every technique

Column **needs**: S stock `transformers`, T `trust_remote_code`, X our executor only. Column **honesty**:
**L** plain lossless storage. **F** the same weights in another form, rebuildable to the dense file. **G**
the weights are gone and a generator replaces them. **C** Qwen's weights change. Sizes are for the fx build.

| technique | disk | VRAM | load | needs | honesty | verdict |
|---|---|---|---|---|---|---|
| Chip layers as sparse tensors | -21.3 GB, to 39 MB raw, 296 KB zstd (M) | -21.3 GB, to 39 MB | faster | T or X | F | **do it** |
| Qwen layers at base width | -4.3 GB | -4.3 GB | faster | T or X | F | **do it**, needs the split |
| Chip columns out of the embedding | -3.2 GB, to about 8 MB (E) | -3.2 GB | faster | T or X | F | **do it**, needs the split |
| Executor frees chip layers and slices Qwen after load | 0 | -28.7 GB (E) | slower | X | none on the file | **do it first** |
| Base Qwen by reference, not re-uploaded | 0 to -8.4 GB download | 0 | | T or X | F | do it; see Honesty |
| Lossless compression of Qwen (byte split + entropy) | -2.9 GB on the wire, 0.67 (M, sampled) | 0 | decode cost | wire only | L | optional |
| Quantise Qwen | see below | | | T, X or S | C | ranked below |
| Drop CW and EB rows | -0.54 GB (E) | -0.54 GB | | S | L | **do it**: 0 uses in fx (M) |
| `R` as a 2-token record | -0.54 GB (E) | -0.54 GB | | S | L | stock path only, +0.08% tokens (M) |
| `J` as `V JF` | -0.54 GB (E) | -0.54 GB | | S | L | no: +6.2% tokens (M) |
| Payloads of 12 bits, not 15 | -0.5 GB (E) | -0.5 GB | | S | L | no: 1.19 to 1.44 tok/instr (R) |
| All-zero tensors left out of the file | -10.0 GB (E) | 0 | slower | S | L | stock path only |
| Four-square ballast: any hidden, m = 4 | -1.2 GB (E) | -1.2 GB | | S | L | stock path only |
| Overlay chip dims on Qwen's columns | -8 GB (E), hidden about 3072 | -8 GB | | S | L | research; needs a trace change |
| Fewer frames (768: 51 layers) | -2.9 GB (E) | same | | S | L | no: faults sooner |
| 3-word RAM slots (43 layers for 2 MB) | -5.8 GB (E) | same | | S | L | no: SW costs 5 tokens (R) |
| `full_attention` type on MLP-only layers | -26 MB (E) | same | | S | L | no |
| Layers on CPU by `device_map` | 0 | -21 GB, RAM +21 GB | | S | L | no, except for a demo |
| Sparse file on the local disk | chip layers only (E) | 0 | | S | L | no: padded `in_proj` has 0% zero pages (M) |
| Hub deduplication | download, not disk | 0 | | S | L | automatic |
| Tensors sharing bytes in safetensors | | | | | | **impossible**: alias, overlap and holes all rejected (M) |
| Pickle with shared or zero-stride storage | disk only | 0 | | S | L | no: an unsafe format to save disk |
| Low-rank Qwen | **larger** | larger | | T | C | **no**: 99% energy needs rank 2,371 of 2,560, 1.18x the size (M) |
| A smaller host model (2B, 0.8B) | 80% or more off | same | | S | C | a different product; 4B is what solved fx's tasks |
| Recipe: rebuild the chip from Python on load | -21 GB | as sparse | slow | T | **G** | no: this is what the identity forbids |
| Chip only in the executor, none in any file | all | all | | X | **G** | no as a release; it is today's fast path (see Honesty) |

Notes on the less obvious rows.

**All-zero tensors.** Transformers fills a missing tensor from `_init_weights`. `initializer_range: 0`
does not work: the code is `initializer_range or 0.02`, so 0 gives random weights (M). `1e-60` works: they
round to zero in bf16 and the output is bit-identical (M, P2). It logs a MISSING table on every load, and
fills 10 GB with noise that rounds away, so load gets slower.

**Four-square ballast.** `hidden = m * 4^k` is why the fx build uses 4096 when it needs 4,021, and spends
64 dims on ballast. The constraint is only that the sum of the ballast squares is hidden times a power
of 4, and four squares reach any integer. Four bf16 ballast values give a norm that is exactly a power of
2 at every hidden size from 2600 to 6000 (M, 3,400 sizes, stock fp32 arithmetic). Hidden 3961 then fits
the same 1,396 slots. Not built: the emitter's constants and the exit's ballast clears assume equal values.

**The old "trim QwenPU to 4608 or 4096" does not fit.** At 4608, m is 18, which leaves 2,029 slots, and
its peak is about 2,250 live dims (R). On the `m * 4^k` grid the floor is 4864, -5%. Four-square would
give about 4815, -6%, if the allocator reached the peak; it uses 2,538 slots today (M), which is 5103.

**Overlay.** On a chip token Qwen's columns hold nothing, and on a text token most chip dims mean nothing,
so they could share columns: rows that read or write a Qwen column get one more weight on `is_text`, which
forces them off on text. It works: 0 changed bits on text in Qwen's columns, chip output identical, stock
kernels (M, P4, a small circuit). At RV scale it is not free. Text rows carry the ballast, so the
datapath runs on text positions too, and a circuit-owned echo is a text token whose successor the whole
datapath predicts. Each echo would need a chip token after it, a trace change. My first analysis missed
this, so its 314 chip-only dims are a lower bound, not a result. Hidden 3072 is a guess, not a measurement.

## Prototypes

| | what | result |
|---|---|---|
| **Sparse is the published file** | rv-q4p2's circuit through the repo's own `emit_layers` with a sparse tensor swapped in, expanded chunk by chunk and compared with the Hub | __VERIFY__ |
| Sparse size | 1,044,349 terms | 6.28 MB raw, **173 KB** zstd (M) |
| Sparse size, fx | 6,480,659 terms | 39.0 MB raw, **296 KB** zstd (M) |
| P1 sparse forward | Linears replaced in a stock model, small circuit | residual and logits bit-identical; 5.4 MB to 2.1 KB (M) |
| P2 zero tensors | 11 tensors omitted, stock `from_pretrained` | bit-identical, 25% smaller, with `initializer_range = 1e-60` (M) |
| P3 quantised Qwen | Qwen's layers int8 and int4, mute column kept in bf16 | chip residual and chip-position logits bit-identical, text logits changed (M) |
| P4 overlay | 38 chip dims in Qwen's columns, 96 gate weights | chip identical, Qwen's columns unchanged on text (M) |

The 173 KB is all 11 layers of the CPU. The 296 KB is 59 layers: the RAM layers repeat, so long-window
zstd removes them. Executor v4 finds the same regularity: 702,768 terms with banks against 6,142,509
without (R).

Prototypes P1 to P4 used a small circuit (3 chip layers, hidden 320) in stock `Qwen3_5ForCausalLM`,
transformers 5.19, on the CPU. They show the mechanism, not the RV core. The chip-only stock model is
already proven at scale by rv-tiny (R).

## Quantising Qwen

The user allowed this. Chip weights stay bf16 and bit-exact in every scheme.

__QUANT__

## The recommended stack

Three steps. The first needs no new file.

**Step 1, the executor stops carrying what it does not read.** After `from_pretrained`, `RvChat` frees the
chip layers and slices Qwen's layers and the embedding to 2,560 columns, taking the two norm gains from
base (the widened gains do not invert exactly). Nothing there reads the rest: `Exec4` runs the IR, and
`_qwen_step` only ever sees text tokens, so mute columns never fire. No new file, no change to the claim.

**Step 2, ship the chip as sparse tensors beside base's weights.**
- `chip.safetensors`: index and value arrays for the chip layers, the embedding's chip columns as a
  bit table plus constants, and fx's image. About 8 MB (E). The chip layers alone are 39 MB raw, 296 KB
  compressed (M).
- Qwen: base's bytes, by reference to `Qwen/Qwen3.5-4B` or as an 8.41 GB text-only upload.
- `expand.py`: rebuilds the dense stock checkpoint byte for byte, tested against the published file.
- A load check: the executor emits its circuit and compares the terms with `chip.safetensors`, so what it
  runs is shown to be what the file holds. Emission takes 16 s on fx's circuit (M). That closes the gap
  in [Honesty](#honesty). Lifting `Exec4` to read the file directly is a larger job and is not needed.
- A `trust_remote_code` class for the HF path, from P1's Linear swap, with chip and Qwen on separate
  residuals. Not built.

**Step 3, quantise Qwen,** with the native kernel below. __STEP3__

| | fx, stock | step 1 | step 2 | step 3 |
|---|---|---|---|---|
| disk | 37.25 GB (E) | 37.25 GB | **8.42 GB** | __S3DISK__ |
| wire | not measured | same | 5.5 GB (E, byte split) | __S3WIRE__ |
| VRAM | 37.25 GB, over 24 GB | **8.5 GB** | 8.5 GB | __S3VRAM__ |
| Qwen | exact | exact | exact, base's bytes | changed |
| chip | exact | exact | exact | exact |
| RAM to build | about 58 GB (R) | | under 2 GB (M, emission on this machine) | |
| effort | | half a day | 3 to 5 days | __S3EFFORT__ |
| risk | | low | low: byte-checked | __S3RISK__ |
| speed | | Qwen matmuls 1.3-1.5x faster (M on CPU, E on GPU) | same | __S3SPEED__ |

The waterfall in GB of disk (E past the first row of each):

| from rv-q4p2 | | from fx | |
|---|---|---|---|
| today | 18.72 | fx, stock | 37.25 |
| chip layers out | -3.85 | chip layers out | -21.32 |
| Qwen padding out | -3.93 | Qwen padding out | -4.28 |
| embedding chip rows and columns out | -2.53 | embedding chip rows and columns out | -3.24 |
| sparse chip back in | +0.01 | sparse chip back in | +0.01 |
| **result** | **8.42** | **result** | **8.42** |

The same 8.42 GB, because the chip's cost stops depending on how many layers it has. Executor speed is
already flat from 31 to 91 layers (R), so fx's 59 layers cost neither bytes nor time this way.

If a stock file must stay the only release, the stock steps reach 24.9 GB of disk and 35.0 GB of VRAM (E),
which still does not load on a 24 GB card.

## Honesty

**What today's fast path really runs.** `RvChat` rebuilds the chip with `core.build(...)` from Python, and
`Exec4` runs that IR, so in the production path the chip weights in the checkpoint are loaded and never
read. Only the HF path, `verify` and `gpucheck` use them. They agree bit for bit (R), but "the computer
runs from the weights" is true of the HF path and false of the fast path today. This predates this work.
**The stack here improves it:** a sparse chip file gives the executor weights it can actually load.

**What is still in the weights after the split.** Every nonzero of every chip matrix, in a tensor file,
rebuildable byte for byte into the dense stock checkpoint. What is lost: one file that stock
`from_pretrained` loads with no code. A dense rebuild keeps that claim available, not free: 37 GB and
about 58 GB of RAM (R) on demand.

**Base by reference** makes the release a chip patch on top of someone else's weights. It is accurate to
say so, and it should be said.

**Text after a chip run.** In the executor path Qwen never sees chip tokens (`_qwen_step` runs on text
only), so mute columns are unused there and base's weights are enough. The merged HF path keeps chip
positions in Qwen's full-attention cache. Text following a chip run can therefore differ between the two,
by the fixed boundary effect `q35/mute.py` describes. The split takes the executor's behaviour, which is
closer to base.

**Not measured here.** Anything on a GPU. Any full-size build of the split or the overlay. Greedy
continuations under quantisation. Hub-side stored size.

## Sources

- [safetensors crate docs](https://docs.rs/safetensors/): the buffer must be entirely indexed, with no holes. Alias and overlap rejection tested here.
- [Xet chunking](https://huggingface.co/docs/xet/chunking) and [deduplication](https://huggingface.co/docs/hub/xet/deduplication): content-defined chunks of about 64 KiB, deduplicated on upload.
- [ZipNN, IBM Research](https://research.ibm.com/blog/Zip-NN-AI-compression), [PyPI](https://pypi.org/project/zipnn/0.3.3): byte grouping before entropy coding, reported 33% on BF16.
- [Deep Compression, Han et al.](https://arxiv.org/abs/1510.00149): sparse storage, shared weights, entropy coding.
- [AWQ, Lin et al.](https://arxiv.org/abs/2306.00978): protecting about 1% of salient weights.
- [Transformers: torchao](https://huggingface.co/docs/transformers/en/quantization/torchao): `FqnToConfig`, `include_embedding`, int4 layout tied to the device it was quantised on.
- [Transformers: bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes): Linear only, `llm_int8_skip_modules`, NF4, nested quantisation.
- [prithivMLmods/Qwen3.5-4B-MTP-GGUF](https://huggingface.co/prithivMLmods/Qwen3.5-4B-MTP-GGUF), [ModelFit comparison](https://modelfit.io/quant-compare/qwen3.5-4b/): community sizes, Q4_K_M 2.78-2.81 GB, Q8_0 4.3-4.61 GB, unverified here.
