"""Quantising Qwen3.5-4B: KL against base, measured on the CPU by streaming.

    python fx/size/qstream.py SCRATCH TOKDIR [--layers N]

The checkpoint is never on disk here: each stock Qwen3_5DecoderLayer is range-read from the Hub, run
on every scheme's activations, and dropped, so all schemes share one pass of about 8 GB of network
and 225 MB of RAM per layer. Activations are checkpointed after every layer (a restart resumes).

Metric: as textcheck.py, mean KL(base || scheme) over Qwen's vocabulary and top-1 agreement, on
teacher-forced prompt + reference continuation, against the noise floor KL(base sdpa || base eager).
Differs from textcheck.py in one way: the continuations are fixed text, not base's greedy output,
because greedy generation would need a network pass per token. So there is no greedy-identical
figure here.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hub
import numpy as np
import torch

DT = torch.bfloat16
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]

PROMPTS = [
    ("chat", "Explain why the sky is blue in two sentences.",
     "The sky appears blue because molecules in the atmosphere scatter short blue wavelengths of "
     "sunlight far more strongly than long red ones. This Rayleigh scattering sends blue light toward "
     "your eyes from every direction."),
    ("raw", 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
     "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n\n"
     "if __name__ == '__main__':\n    print(fibonacci(10))\n"),
    ("chat", "What is 37 * 43? Think step by step.",
     "First, 37 * 40 = 1480. Next, 37 * 3 = 111. Adding the two gives 1480 + 111 = 1591. So the "
     "answer is 1591."),
    ("chat", "Traduis en français : The weather is lovely today and I want to go for a walk.",
     "Il fait très beau aujourd'hui et j'ai envie d'aller me promener."),
    ("chat", "写一首关于秋天的短诗。",
     "秋风轻拂过山冈，\n落叶金黄满地香。\n一行雁阵南飞去，\n留下斜阳映水长。"),
    ("chat", "List three differences between TCP and UDP.",
     "1. TCP is connection-oriented, while UDP is connectionless.\n2. TCP guarantees ordered delivery "
     "with retransmission; UDP does not.\n3. UDP has lower overhead and latency."),
    ("raw", "The CHIP-8 is a virtual machine from the 1970s that",
     " was designed to make game programming easier on early microcomputers such as the COSMAC VIP. "
     "It has 4 KB of memory, sixteen registers and a 64 by 32 display."),
    ("raw", "Q: If a train leaves at 3pm travelling 60 km/h, how far has it gone by 5:30pm?\nA:",
     " From 3pm to 5:30pm is 2.5 hours. At 60 km/h the train covers 60 * 2.5 = 150 km."),
    ("chat", "Write a haiku about a lighthouse.",
     "Alone on the rock,\na slow beam sweeps the dark sea,\nguiding strangers home."),
    ("chat", "Give me a Python one-liner that reverses the words in a sentence.",
     "```python\nreversed_sentence = ' '.join(sentence.split()[::-1])\n```\nThis splits the sentence "
     "into words, reverses the list and joins it back."),
    ("chat", "What causes the seasons on Earth?",
     "The seasons are caused by the tilt of Earth's axis, about 23.5 degrees. As Earth orbits the "
     "Sun, each hemisphere leans toward the Sun for part of the year and away for the rest."),
    ("raw", "In 1969, Apollo 11",
     " became the first crewed mission to land on the Moon. Neil Armstrong and Buzz Aldrin walked on "
     "the surface while Michael Collins orbited above."),
    ("chat", "Summarise the plot of Romeo and Juliet in three sentences.",
     "Two young lovers from feuding Verona families marry in secret. A chain of violence and "
     "miscommunication separates them. Both die, and their deaths finally reconcile the families."),
    ("chat", "Solve for x: 3x + 7 = 22.",
     "Subtract 7 from both sides: 3x = 15. Divide both sides by 3: x = 5."),
    ("raw", "SELECT name, COUNT(*) FROM orders",
     " WHERE created_at >= '2024-01-01' GROUP BY name ORDER BY COUNT(*) DESC LIMIT 10;"),
    ("chat", "¿Cuál es la capital de Australia y por qué no es Sídney?",
     "La capital de Australia es Canberra. Se eligió como compromiso entre Sídney y Melbourne, que "
     "competían por ese título."),
]

GDN = ("in_proj_qkv", "in_proj_z", "out_proj")
ATT = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")

# scheme -> family -> method; families: ffn gdn attn emb (input) head (lm_head, tied to emb in a
# real model; the two are split only to see which side matters). 'ends': override for the first and
# last two layers' ffn. Small tensors (a, b, conv, norms, dt_bias, A_log) stay bf16 in every scheme.
SCHEMES = {
    "base": {},
    "floor_eager": {"eager": True},
    # one family at a time
    "i8_ffn": {"ffn": "i8r"}, "i8_gdn": {"gdn": "i8r"}, "i8_attn": {"attn": "i8r"},
    "i8_embin": {"emb": "i8r"}, "i8_lmhead": {"head": "i8r"},
    "i4_ffn": {"ffn": "i4g"}, "i4_gdn": {"gdn": "i4g"}, "i4_attn": {"attn": "i4g"},
    "i4_embin": {"emb": "i4g"}, "i4_lmhead": {"head": "i4g"},
    # whole-model
    "i8_all": {"ffn": "i8r", "gdn": "i8r", "attn": "i8r", "emb": "i8r", "head": "i8r"},
    "i8_noemb": {"ffn": "i8r", "gdn": "i8r", "attn": "i8r"},
    "i4_all": {"ffn": "i4g", "gdn": "i4g", "attn": "i4g", "emb": "i4g", "head": "i4g"},
    "i4_noemb": {"ffn": "i4g", "gdn": "i4g", "attn": "i4g"},
    # mixed
    "mix_a": {"ffn": "i4g", "gdn": "i8r", "attn": "i8r", "emb": "i8r", "head": "i8r"},
    "mix_b": {"ffn": "i4g", "gdn": "i8r", "attn": "i8r"},
    "mix_c": {"ffn": "i4g", "gdn": "i8r", "attn": "i8r", "emb": "i8r", "head": "i8r", "ends": "i8r"},
    # what stock bitsandbytes does: Linear only, embeddings untouched, whole-tensor or 64-block scales
    "bnb_nf4": {"ffn": "nf4", "gdn": "nf4", "attn": "nf4"},
    "bnb_i8": {"ffn": "i8t", "gdn": "i8t", "attn": "i8t"},
}


def quant(w: torch.Tensor, m: str) -> torch.Tensor:
    """The weights as the scheme reconstructs them (a Linear's weight: rows = outputs)."""
    x = w.float()
    if m == "i8r":  # symmetric int8, one scale per output row (torchao per-axis)
        s = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        return ((x / s).round().clamp(-127, 127) * s).to(DT)
    if m == "i8t":  # symmetric int8, one scale for the tensor (bnb Int8Params keeps a per-column one; see doc)
        s = x.abs().amax().clamp_min(1e-12) / 127
        return ((x / s).round().clamp(-127, 127) * s).to(DT)
    if m == "i4g":  # symmetric int4, one scale per 32 weights of a row (torchao IntxWeightOnly group 32)
        r, c = x.shape
        g = 32
        pad = (-c) % g
        y = torch.cat([x, x.new_zeros(r, pad)], 1).view(r, -1, g)
        s = y.abs().amax(2, keepdim=True).clamp_min(1e-12) / 7
        return ((y / s).round().clamp(-7, 7) * s).view(r, -1)[:, :c].to(DT)
    if m == "nf4":  # bitsandbytes NF4: blocks of 64, each scaled by its max, then the 16-level codebook
        v = x.view(-1)
        pad = (-v.numel()) % 64
        y = torch.cat([v, v.new_zeros(pad)]).view(-1, 64)
        s = y.abs().amax(1, keepdim=True).clamp_min(1e-12)
        cb = torch.tensor(NF4)
        n = y / s
        mid = (cb[1:] + cb[:-1]) / 2
        q = cb[torch.bucketize(n, mid)]
        return (q * s).view(-1)[: v.numel()].view(x.shape).to(DT)
    raise ValueError(m)


def _bytes(t: torch.Tensor) -> bytes:
    """A bf16 tensor's raw little-endian bytes."""
    return (t.float().numpy().astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def fetch(t) -> torch.Tensor:
    """A tensor by parallel range reads."""
    n = t["b"] - t["a"]
    step = 6 << 20  # the link here limits each stream, up to about 16 of them
    parts = [(i, min(i + step, n)) for i in range(0, n, step)]
    # streams write straight into the tensor: no joined copy (this machine is short of commit)
    dt, it = {"BF16": (DT, torch.uint16), "F32": (torch.float32, torch.uint32)}[t["dtype"]]
    out = torch.empty(n // (2 if t["dtype"] == "BF16" else 4), dtype=it)
    buf = memoryview(out.numpy()).cast("B")

    def get(p):
        buf[p[0]:p[1]] = hub.read(t, *p)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, parts))
    del buf
    return out.view(dt).reshape(t["shape"])


def build_layer(cfg, i, sd):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    with torch.device("meta"):
        lay = Qwen3_5DecoderLayer(cfg, i)
    miss, unexp = lay.load_state_dict(sd, strict=False, assign=True)
    assert not miss and not unexp, (miss, unexp)
    return lay.eval()


def family(name):
    leaf = name.split(".")[-2]
    if leaf in GDN:
        return "gdn"
    if leaf in ATT:
        return "attn"
    if leaf in FFN:
        return "ffn"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("tokdir")
    ap.add_argument("--layers", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoTokenizer, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    H = pickle.load(open(a.scratch + "/hdr.pkl", "rb"))["base"]
    tc = json.load(open(a.scratch + "/cfg/base/config.json"))["text_config"]
    tc = {k: v for k, v in tc.items() if k not in ("architectures", "model_type", "dtype", "torch_dtype")}
    cfg = Qwen3_5TextConfig(**tc)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(a.tokdir)
    seqs = []
    for kind, p, cont in PROMPTS:
        ids = (tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       return_tensors="pt", enable_thinking=False)["input_ids"][0].tolist()
               if kind == "chat" else tok(p).input_ids)
        seqs.append(ids + tok(cont, add_special_tokens=False).input_ids)
    print("sequences", len(seqs), "positions", sum(map(len, seqs)), flush=True)
    em = H["model.language_model.embed_tokens.weight"]
    used = sorted({i for s in seqs for i in s})
    rows = {}
    with ThreadPoolExecutor(8) as ex:
        for i, r in zip(used, ex.map(lambda i: hub.rows(em, i, i + 1), used)):
            rows[i] = np.array(r[0])
    urow = torch.frombuffer(bytearray(np.stack([rows[i] for i in used]).tobytes()),
                            dtype=DT).reshape(len(used), -1).clone()
    pos = {t: i for i, t in enumerate(used)}
    print("max |emb| on used rows", float(urow.float().abs().max()), flush=True)
    emb_var = {"": urow, "i8r": quant(urow, "i8r"), "i4g": quant(urow, "i4g")}

    ck = a.scratch + "/qstream.pkl"
    done = 0
    state = None
    if os.path.exists(ck):
        done, state = pickle.load(open(ck, "rb"))
        print("resume at layer", done, flush=True)
    if state is None:
        state = {}
        for name, s in SCHEMES.items():
            v = emb_var[s.get("emb", "")]
            state[name] = [v[[pos[t] for t in q]][None] for q in seqs]
    rot = M.Qwen3_5TextRotaryEmbedding(config=cfg)

    def ctx(x, c):
        n = x.shape[1]
        p = torch.arange(n).view(1, 1, -1).expand(4, 1, -1)
        kw = dict(config=c, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=p[0])
        return (rot(x, p[1:]), {"full_attention": M.create_causal_mask(**kw),
                                "linear_attention": M.create_recurrent_attention_mask(**kw)}, p[0])

    import copy

    ecfg = copy.deepcopy(cfg)
    ecfg._attn_implementation = "eager"

    def load_layer(i):
        pre = f"model.language_model.layers.{i}."
        return {k[len(pre):]: fetch(H[k]).to(DT) for k in H if k.startswith(pre)}

    bg = ThreadPoolExecutor(1)  # the next layer is read while this one runs
    nxt = None
    with torch.no_grad():
        for i in range(done, a.layers):
            t0 = time.time()
            sd = (nxt.result() if nxt else load_layer(i))
            t1 = time.time()
            nxt = bg.submit(load_layer, i + 1) if i + 1 < a.layers else None
            var = {}

            def get(name, m):
                if (name, m) not in var:
                    var[name, m] = quant(sd[name], m)
                return var[name, m]

            lay = build_layer(cfg, i, dict(sd))
            for sname, s in SCHEMES.items():
                sub = dict(sd)
                for name in sd:
                    fam = family(name)
                    m = s.get(fam) if fam else None
                    if fam == "ffn" and "ends" in s and (i < 2 or i >= a.layers - 2):
                        m = s["ends"]
                    if m:
                        sub[name] = get(name, m)
                for name, p in lay.named_parameters():
                    p.data = sub[name]
                c = ecfg if s.get("eager") else cfg
                if tc["layer_types"][i] == "full_attention":
                    lay.self_attn.config = c
                out = []
                for x in state[sname]:
                    pe, masks, tp = ctx(x, c)
                    out.append(lay(x, position_embeddings=pe, attention_mask=masks[tc["layer_types"][i]],
                                   position_ids=tp, past_key_values=None, use_cache=False))
                state[sname] = out
            del lay, var, sd
            pickle.dump((i + 1, state), open(ck + ".tmp", "wb"))
            os.replace(ck + ".tmp", ck)
            print(f"layer {i} wait {t1 - t0:.0f}s run {time.time() - t1:.0f}s", flush=True)

    # the output head: norm, then the embedding matrix (Qwen's vocabulary). RAM is short here, so the
    # base and quantised matrices sit in scratch files (bf16, deleted at the end), read a chunk at a
    # time, and each sequence's logits are built per chunk.
    nw = fetch(H["model.language_model.norm.weight"]).to(DT)
    V, D = em["shape"]
    CH = 16384
    methods = ("", "i8r", "i4g")
    paths = {m: f"{a.scratch}/E_{m or 'base'}.bin" for m in methods}
    if not all(os.path.exists(p) and os.path.getsize(p) == V * D * 2 for p in paths.values()):
        fs = {m: open(p, "wb") for m, p in paths.items()}
        for r in range(0, V, CH):
            rb = bytearray(hub.read(em, r * D * 2, min(V, r + CH) * D * 2))
            e = torch.frombuffer(rb, dtype=DT).reshape(-1, D)
            for m in methods:
                fs[m].write(_bytes(quant(e, m) if m else e))
            print("head chunk", r, flush=True)
        for f in fs.values():
            f.close()
    mm = {m: np.memmap(p, dtype=np.uint16, mode="r", shape=(V, D)) for m, p in paths.items()}

    def chunk(m, r):
        return torch.frombuffer(bytearray(mm[m][r:min(V, r + CH)].tobytes()), dtype=DT).reshape(-1, D)

    def norm(x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1.0 + nw.float())
        return v.to(DT)

    acc = {n: {"ks": [], "top": [], "mx": []} for n in SCHEMES}
    with torch.no_grad():
        for si in range(len(seqs)):
            xs = {n: norm(state[n][si])[0] for n in SCHEMES}
            parts = {n: [] for n in SCHEMES}
            for r in range(0, V, CH):
                ec = {m: chunk(m, r) for m in methods}
                for n, s in SCHEMES.items():
                    parts[n].append((xs[n] @ ec[s.get("head", "")].T).float())
            lb = torch.log_softmax(torch.cat(parts["base"], 1), -1)
            p = lb.exp()
            for n in SCHEMES:
                lq = torch.log_softmax(torch.cat(parts[n], 1), -1)
                k = (p * (lb - lq)).sum(-1)
                acc[n]["ks"].append(float(k.mean()))
                acc[n]["top"].append(float((lb.argmax(-1) == lq.argmax(-1)).float().mean()))
                acc[n]["mx"].append(float(k.max()))
            print("head seq", si, flush=True)
    res = {}
    for n, v in acc.items():
        res[n] = {"kl_mean": sum(v["ks"]) / len(v["ks"]), "kl_per_seq": v["ks"],
                  "top1": sum(v["top"]) / len(v["top"]), "kl_max_pos": max(v["mx"])}
        print(n, f"KL {res[n]['kl_mean']:.3e} top1 {res[n]['top1']:.4f} maxKL {res[n]['kl_max_pos']:.3f}",
              flush=True)
    del mm
    for p in paths.values():
        os.remove(p)
    json.dump({"positions": sum(map(len, seqs)), "layers": a.layers, "schemes": res},
              open(a.scratch + "/qstream.json", "w"), indent=1)


if __name__ == "__main__":
    main()
