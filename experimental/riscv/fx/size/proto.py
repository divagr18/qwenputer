"""Size prototypes on a small circuit, in stock Qwen3_5ForCausalLM (transformers 5.19, CPU, bf16).

    PYTHONPATH=. python fx/size/proto.py OUTDIR

A small machine (a register head, a lag-read key, XOR logic, an exit) merged rv1-style into a tiny
stock model: chip layers, then two random 'Qwen' layers widened by zero padding, muted on chip
tokens; text rows carry the ballast, as rv1's do. Four checks:

  P1 sparse    chip tensors as index/value arrays in a Linear replacement, against the dense model
  P2 zeroes    all-zero tensors left out of the file, stock from_pretrained, initializer_range 1e-60
               (0.0 does not work: PreTrainedModel._init_weights does `initializer_range or 0.02`)
  P3 quant     Qwen's layers int8 / int4 with the mute column kept, the chip must not move a bit
  P4 overlay   chip dims in Qwen's columns, gated off on text: chip dims and Qwen's columns exact
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstream import quant

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35 import ir
from chipformer.q35.emit import Layout, embedding_rows, emit_layers
from chipformer.q35.ir import Chan, K, Target
from chipformer.q35.prims import Circ, thead

DT = torch.bfloat16
torch.manual_seed(0)
TG = Target(2, 4, 256)
TEXT = 1024
D0 = 256  # 'Qwen' width


def circuit():
    """Tokens: 0..255 write a byte, 256..511 read and XOR, 1024+ text."""
    c = Circ(TG)
    f = c.token_feature
    isW = f("isW", lambda t: int(0 <= t < 256), full=True)
    isR = f("isR", lambda t: int(256 <= t < 512), full=True)
    is_text = f("is_text", lambda t: int(t >= TEXT), full=True)
    mu = f("mu", lambda t: int(t < TEXT), full=True)
    pay = [f(f"p{i}", lambda t, i=i: (t >> i & 1) if t < 512 else 0, full=True) for i in range(8)]
    one = c.one
    ((reg,),) = [thead(c, "reg", {0: one}, {0: one}, [{"data": {2 + i: pay[i] for i in range(8)},
                                                       "ref": one, "beta": isW}])]
    # a second table keyed by the previous token's low bit (a lag tap), as a store's key is
    key = Chan(pay[0], (0.0, 1.0, 0.0, 0.0))
    ((prev,),) = [thead(c, "prev", {0: one}, {0: one},
                        [{"data": {2 + i: key if i == 0 else pay[i] for i in range(4)}, "ref": one,
                          "beta": isW}])]
    x = [c.node(c.t_and([reg[i], one - pay[i], isR]) + c.t_and([one - reg[i], pay[i], isR]), f"x{i}")
         for i in range(8)]
    y = [c.node(c.t_and([x[i], prev[i % 4]]), f"y{i}") for i in range(8)]
    z = [c.node(c.t_and([y[i], isR]), f"exit.z{i}") for i in range(8)]
    c.keep = [next(iter(e.terms)) for e in z] + [next(iter(mu.terms))]
    c.is_text_e, c.mu_e = is_text, mu
    return c


class Overlay(Layout):
    """IR dims in cols (dim -> column), as emit.py's Layout.dim."""
    cols: dict = None

    def dim(self, d):
        return self.cols[d]


def layout(c, overlay: bool):
    """Hidden, ballast and the column of every IR dim."""
    keep = set(c.keep) | {next(iter(c.is_text_e.terms))}
    text_nz = {d for d in c.token_fns if any(float(c.feature_value(d, t)) for t in (TEXT, TEXT + 7))}
    C = keep | text_nz
    dims = list(range(1, len(c.dims)))
    O = [d for d in dims if d not in C] if overlay else []
    Cl = [d for d in dims if d not in O]
    for h, m in sorted((m * 4**k, m) for m in range(3, 65) for k in range(1, 6)):
        if h >= D0 + m + 1 + len(Cl) and h % 64 == 0:
            break
    cols = {d: i for i, d in enumerate(O)}
    cols.update({d: D0 + m + 1 + i for i, d in enumerate(Cl)})
    lay = Overlay(hidden=h, off=D0, m=m)
    lay.cols = cols
    return lay, set(O)


def gate_off(sd, c, lay, O, tc, n_chip, xmax):
    """Force every row that reads a Qwen column to its off value on text tokens: one more weight, on
    is_text. A chip token has is_text = 0, so nothing changes there."""
    it = lay.cols[next(iter(c.is_text_e.terms))]
    cols = sorted(lay.cols[d] for d in O)
    n = 0
    sc = lay.scale

    def gate(W, rows):
        nonlocal n
        s = W[rows][:, cols].abs().sum(1)
        for r, v in zip(rows, s.tolist()):
            g = 2.0 ** torch.ceil(torch.log2(torch.tensor(2 * v * xmax + 4096 / sc))).item()
            W[r, it] = W[r, it] - g
            n += 1

    for L in range(n_chip):
        p = f"model.layers.{L}."
        gt, up = sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]
        dn = sd[p + "mlp.down_proj.weight"]
        rows = ((gt[:, cols] != 0).any(1) | (up[:, cols] != 0).any(1)
                | (dn[cols, :] != 0).any(0)).nonzero()[:, 0].tolist()  # reads or writes one
        gate(gt, rows)
        for nm in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                   "linear_attn.in_proj_b"):
            W = sd[p + nm + ".weight"]
            mask = (W[:, cols] != 0).any(1)
            if nm.endswith("in_proj_z"):  # a head output into a Qwen column: its gate z goes off
                mask = mask | (sd[p + "linear_attn.out_proj.weight"][cols, :] != 0).any(0)
            gate(W, mask.nonzero()[:, 0].tolist())
    return n


def build(c, overlay=False, qwen=None):
    from transformers import Qwen3_5TextConfig

    lay, O = layout(c, overlay)
    h = lay.hidden
    n_chip = c.n_layers()
    tc = dict(hidden_size=h, vocab_size=2048, num_hidden_layers=n_chip + 2, intermediate_size=256,
              layer_types=["linear_attention"] * (n_chip + 2), linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
              linear_conv_kernel_dim=4, hidden_act="silu", num_attention_heads=2, num_key_value_heads=2,
              head_dim=64, rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=4096,
              initializer_range=1e-60, attn_output_gate=True,
              rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                               "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    sd = emit_layers(c, lay, tc, 0, "model.", None, set(c.keep))
    sd = {k: v.clone() for k, v in sd.items()}
    vocab = [t for t in range(512)]
    emb = torch.zeros(2048, h)
    emb[:512] = embedding_rows(c, lay, vocab, h)
    txt = embedding_rows(c, lay, [TEXT + i for i in range(1024)], h)
    q = torch.from_numpy(qwen["emb"].numpy().copy()).float()
    txt[:, :D0] = q
    emb[TEXT:TEXT + 1024] = txt
    n_gates = 0
    if overlay:
        n_gates = gate_off(sd, c, lay, O, tc, n_chip, float(q.abs().max()))
    sd["model.embed_tokens.weight"] = emb
    sd["model.norm.weight"] = torch.zeros(h)
    mu_col = lay.cols[next(iter(c.mu_e.terms))]
    w = -1024.0 / h**0.5
    for i, (name, wt) in enumerate(sorted(qwen["layers"].items())):
        L, leaf = name.split("/")
        p = f"model.layers.{n_chip + int(L)}.{leaf}"
        wide = wt.clone()
        if leaf.endswith(("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
                          "gate_proj.weight", "up_proj.weight")):
            wide = torch.cat([wide, torch.zeros(wide.shape[0], h - D0)], 1)
            if leaf.endswith(("in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight")):
                wide[:, mu_col] = w
        elif leaf.endswith(("out_proj.weight", "down_proj.weight")):
            wide = torch.cat([wide, torch.zeros(h - D0, wide.shape[1])], 0)
        sd[p] = wide
    for L in range(2):
        p = f"model.layers.{n_chip + L}."
        sd[p + "input_layernorm.weight"] = torch.zeros(h)
        sd[p + "post_attention_layernorm.weight"] = torch.zeros(h)
        sd[p + "linear_attn.conv1d.weight"] = qwen["conv"].clone()
        sd[p + "linear_attn.dt_bias"] = torch.ones(4)
        sd[p + "linear_attn.A_log"] = torch.zeros(4)
        sd[p + "linear_attn.norm.weight"] = torch.ones(128)
    return Qwen3_5TextConfig(**tc), sd, lay, O, n_gates


def random_qwen():
    r = lambda *s: torch.randn(*s) * 0.02
    layers = {}
    for L in range(2):
        layers[f"{L}/linear_attn.in_proj_qkv.weight"] = r(2 * 2 * 128 + 4 * 128, D0)
        layers[f"{L}/linear_attn.in_proj_z.weight"] = r(4 * 128, D0)
        layers[f"{L}/linear_attn.in_proj_a.weight"] = r(4, D0)
        layers[f"{L}/linear_attn.in_proj_b.weight"] = r(4, D0)
        layers[f"{L}/linear_attn.out_proj.weight"] = r(D0, 4 * 128)
        layers[f"{L}/mlp.gate_proj.weight"] = r(256, D0)
        layers[f"{L}/mlp.up_proj.weight"] = r(256, D0)
        layers[f"{L}/mlp.down_proj.weight"] = r(D0, 256)
    return dict(layers=layers, emb=(torch.randn(1024, D0) * 0.04).clamp(-0.14, 0.14),
                conv=torch.randn(2 * 2 * 128 + 4 * 128, 1, 4) * 0.1)


def model_from(cfg, sd):
    from transformers import Qwen3_5ForCausalLM

    with torch.device("meta"):
        m = Qwen3_5ForCausalLM(cfg)
    miss, unexp = m.load_state_dict(sd, strict=False, assign=True)
    assert not unexp, unexp[:3]
    assert set(miss) <= {"lm_head.weight"}, list(miss)[:5]
    m.tie_weights()
    m.model.rotary_emb = type(m.model.rotary_emb)(config=cfg)
    return m.to(DT).eval()


class SparseLinear(torch.nn.Module):
    """A chip Linear as its nonzeros: fp32 sum, then bf16, as a dense bf16 matmul would give on
    exact values."""

    def __init__(self, W):
        super().__init__()
        idx = W.nonzero()
        self.rows, self.cols = W.shape
        self.register_buffer("r", idx[:, 0].to(torch.int32))
        self.register_buffer("c", idx[:, 1].to(torch.int32))
        self.register_buffer("v", W[idx[:, 0], idx[:, 1]].to(DT))

    def forward(self, x):
        y = x.new_zeros(*x.shape[:-1], self.rows, dtype=torch.float32)
        y.index_add_(-1, self.r.long(), x[..., self.c.long()].float() * self.v.float())
        return y.to(x.dtype)


def run(m, ids, n_chip):
    with torch.no_grad():
        o = m(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    return o.hidden_states[n_chip][0], o.logits[0]


def same(a, b):
    return bool(torch.equal(a.float(), b.float()))


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    res = {}
    c = circuit()
    n_chip = c.n_layers()
    print(c.stats(), flush=True)
    qwen = random_qwen()
    toks = [5, 7, 300, 11, 260, 1030, 1044, 1100, 12, 301, 1500, 300, 77, 310, 1033, 256, 200, 511]
    ids = list(toks)
    cfg, sd, lay, O, _ = build(c, False, qwen)
    dense = model_from(cfg, {k: v.clone() for k, v in sd.items()})
    hd, ld = run(dense, ids, n_chip)
    want = ir.evaluate(c, toks, check=False)
    dims = [d for d in range(1, len(c.dims)) if d not in set(c.keep)]
    chip_pos = [i for i, t in enumerate(toks) if t < TEXT]
    # the IR has no exit clears; the keep dims are what survives the exit
    kept = sorted(set(c.keep))
    print("hidden", lay.hidden, "m", lay.m, flush=True)
    a = hd[:, [lay.dim(d) for d in kept]].double().numpy()
    b = want[:, kept]
    res["dense_vs_ir_keep_dims_equal"] = bool(((a == b) | (abs(a - b) < 1e-30)).all())
    print("dense = IR on kept dims:", res["dense_vs_ir_keep_dims_equal"], flush=True)

    # --- P1 sparse: swap the chip layers' Linears, stock everything else ---------------------
    sp = model_from(cfg, {k: v.clone() for k, v in sd.items()})
    dense_bytes = sparse_bytes = 0
    t0 = time.time()
    for L in range(n_chip):
        layer = sp.model.layers[L]
        for owner, nm in ((layer.mlp, "gate_proj"), (layer.mlp, "up_proj"), (layer.mlp, "down_proj"),
                          (layer.linear_attn, "in_proj_qkv"), (layer.linear_attn, "in_proj_z"),
                          (layer.linear_attn, "in_proj_a"), (layer.linear_attn, "in_proj_b"),
                          (layer.linear_attn, "out_proj")):
            lin = getattr(owner, nm)
            s = SparseLinear(lin.weight.data.float())
            dense_bytes += lin.weight.numel() * 2
            sparse_bytes += s.v.numel() * 6
            setattr(owner, nm, s)
    hs, ls = run(sp, ids, n_chip)
    res["P1_sparse"] = dict(residual_identical=same(hs, hd), logits_identical=same(ls, ld),
                            dense_bytes=dense_bytes, sparse_bytes=sparse_bytes)
    t_d = time.time(); [run(dense, ids, n_chip) for _ in range(5)]; t_d = time.time() - t_d
    t_s = time.time(); [run(sp, ids, n_chip) for _ in range(5)]; t_s = time.time() - t_s
    res["P1_sparse"].update(dense_s=round(t_d, 3), sparse_s=round(t_s, 3))
    print("P1", res["P1_sparse"], flush=True)

    # --- P2 all-zero tensors left out, stock from_pretrained ----------------------------------
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    d = outdir + "/zeroes"
    os.makedirs(d, exist_ok=True)
    full = {k: v.to(DT).contiguous() for k, v in sd.items()}
    gone = [k for k, v in full.items() if v.ndim == 2 and "layers" in k and not bool(v.float().any())]
    if not gone:
        raise SystemExit("no all-zero tensor in the tiny circuit")
    save_file({k: v for k, v in full.items() if k not in gone}, d + "/model.safetensors", metadata={"format": "pt"})
    cfg.save_pretrained(d)
    m2, info = Qwen3_5ForCausalLM.from_pretrained(d, dtype=DT, output_loading_info=True)
    m2 = m2.eval()
    st2 = m2.state_dict()
    nonzero_inited = [k for k in gone if bool((st2[k].float() != 0).any())]
    print("config init range", m2.config.initializer_range, "after load max abs",
          max(float(st2[k].float().abs().max()) for k in gone), flush=True)
    h2, l2 = run(m2, ids, n_chip)
    print("omitted", sorted({k.split(".", 2)[2] for k in gone}), flush=True)
    print("omitted layers", sorted({k.split(".")[2] for k in gone}), "n_chip", n_chip,
          "residual max diff", float((h2.float() - hd.float()).abs().max()), flush=True)
    res["P2_zeroes"] = dict(omitted=len(gone), omitted_bytes=sum(full[k].numel() * 2 for k in gone),
                            file_bytes=os.path.getsize(d + "/model.safetensors"),
                            full_bytes=sum(v.numel() * 2 for v in full.values()),
                            missing_reported=len(info["missing_keys"]), any_nonzero_after_load=len(nonzero_inited),
                            residual_identical=same(h2, hd), logits_identical=same(l2, ld))
    print("P2", res["P2_zeroes"], flush=True)

    # --- P3 quantised Qwen, the chip untouched ------------------------------------------------
    for mname in ("i8r", "i4g"):
        sq = {k: v.clone() for k, v in sd.items()}
        for k in list(sq):
            L = k.split(".")
            if k.startswith("model.layers.") and int(L[2]) >= n_chip and k.endswith(
                    ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight", "gate_proj.weight",
                     "up_proj.weight", "down_proj.weight")):
                w = sq[k]
                if w.shape[1] == lay.hidden:  # Qwen columns only; the mute column stays bf16
                    w[:, :D0] = quant(w[:, :D0], mname).float()
                else:
                    w[:D0] = quant(w[:D0], mname).float()
        mq = model_from(cfg, sq)
        hq, lq = run(mq, ids, n_chip)
        lc = [i for i in chip_pos]
        lt = [i for i, t in enumerate(toks) if t >= TEXT]
        res["P3_" + mname] = dict(chip_residual_identical=same(hq, hd),
                                  chip_position_logits_identical=same(lq[lc], ld[lc]),
                                  text_logits_changed=not same(lq[lt], ld[lt]))
        print("P3", mname, res["P3_" + mname], flush=True)

    # --- P4 overlay: chip dims in Qwen's columns ------------------------------------------------
    cfg4, sd4, lay4, O4, ng = build(c, True, qwen)
    mo = model_from(cfg4, sd4)
    ho, lo = run(mo, ids, n_chip)
    tp = torch.tensor([i for i, t in enumerate(toks) if t >= TEXT])
    cp = torch.tensor(chip_pos)
    qcols = list(range(D0))
    kept_b = hd[:, [lay.dim(d) for d in kept]]
    kept_o = ho[:, [lay4.dim(d) for d in kept]]
    emb_text = qwen["emb"][[t - TEXT for t in toks if t >= TEXT]].to(DT)
    dq = (ho[tp][:, qcols].float() - emb_text.float()).abs()
    print("P4 qwen column diff on text: max", float(dq.max()), "cols", dq.amax(0).nonzero()[:, 0][:6].tolist(), flush=True)
    res["P4_overlay"] = dict(
        hidden_plain=lay.hidden, hidden_overlay=lay4.hidden, dims_overlaid=len(O4), gate_weights_added=ng,
        chip_keep_dims_identical_on_chip_tokens=same(kept_o[cp], kept_b[cp]),
        qwen_columns_identical_on_text=same(ho[tp][:, qcols], emb_text),
        qwen_columns_clear_on_chip=same(ho[cp][:, qcols], torch.zeros_like(ho[cp][:, qcols])))
    print("P4", res["P4_overlay"], flush=True)
    json.dump(res, open(outdir + "/proto.json", "w"), indent=1)


if __name__ == "__main__":
    main()
