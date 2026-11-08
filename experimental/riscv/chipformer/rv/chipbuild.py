"""Write chip.safetensors from the circuit's Python build. A build-time tool: production reads the
file (chipfile.py) and never runs this.

    python -m chipformer.rv.chipbuild rv-q4p2 --dense dist/rv-q4p2 --base BASE --out chip.safetensors \\
        [--firmware tests/data/rv/calc-bridge.elf]
    python -m chipformer.rv.chipbuild paged --frames 1024 --out chip-paged.safetensors
    python -m chipformer.rv.chipbuild fx --log LOG.npz --elf FX.elf --out chip-fx.safetensors [--base BASE]

The chip layers are emit_layers itself, with its dense tensors swapped for dict-backed ones, so a
term here is exactly a nonzero the dense emitter writes.
"""

import argparse
import dataclasses
import json
import struct
from pathlib import Path

import numpy as np
import torch

from chipformer.q35 import emit as E
from chipformer.q35.emit import Layout, allocate, embedding_rows
from chipformer.q35.ir import A_REF, HD, Head
from chipformer.q35.merge import TAU_GAIN, pick_hidden
from chipformer.q35.tiny import code_gain
from chipformer.rv import bridge as BR
from chipformer.rv import chipfile as F
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S

D0 = 2560  # Qwen3.5-4B's hidden size: the chip's columns start here


# --- emit_layers, sparse --------------------------------------------------------------------
class _ST:
    """Enough of a tensor for emit.py: t[i][j] += x, t[i, j] = x, t[i, 0, k] = x."""

    def __init__(self, shape, fill=0.0):
        self.shape, self.d, self.fill = tuple(shape), {}, fill

    def __getitem__(self, i):
        if isinstance(i, tuple):
            return self.d.get(tuple(int(x) for x in i), self.fill)
        return _Row(self, int(i))

    def __setitem__(self, i, v):
        self.d[tuple(int(x) for x in i)] = float(v)


class _Row:
    def __init__(self, t, i):
        self.t, self.i = t, i

    def __getitem__(self, j):
        return self.t.d.get((self.i, int(j)), self.t.fill)

    def __setitem__(self, j, v):
        self.t.d[(self.i, int(j))] = float(v)


class _Shim:
    """The part of the torch namespace emit_layers touches."""

    @staticmethod
    def zeros(*shape, dtype=None):
        return _ST(shape)

    @staticmethod
    def full(shape, v, dtype=None):
        return _ST(shape, float(v))

    @staticmethod
    def ones(*shape, dtype=None):
        return _ST(shape, 1.0)

    def __getattr__(self, k):
        return getattr(torch, k)


def _vocab(c) -> int:
    return getattr(c, "vocab", S.VOCAB)


def _is_bf16(x: float) -> bool:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return (b & 0xFFFF) == 0 and struct.unpack("<f", struct.pack("<f", x))[0] == x


def emit_sparse(c, lay, tc, alloc, keep) -> dict:
    def chk(name, t):
        for k, v in t.d.items():
            assert _is_bf16(v), (name, k, v)

    old_t, old_c = E.torch, E.check_bf16
    E.torch, E.check_bf16 = _Shim(), chk
    try:
        return E.emit_layers(c, lay, tc, 0, "model.", alloc, keep)
    finally:
        E.torch, E.check_bf16 = old_t, old_c


def bf16_tensor(x) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x, np.float32))
    b = t.to(torch.bfloat16)
    assert bool((b.float() == t).all()), "a value is not bf16"
    return b


def weight_arrays(sd, n_chip: int, kk: int):
    """{weights.*}, the tensor list, in the file's order: 14 tensors per layer."""
    rows, cols, vals, off, names = [], [], [], [0], []
    for L in range(n_chip):
        for short in F.LAYER_TENSORS:
            k = f"model.layers.{L}.{short}"
            t = sd[k]
            ents = []
            for key, v in t.d.items():
                if v == t.fill:
                    continue
                r = key[0]
                c = 0 if len(key) == 1 else key[-1]
                if len(key) == 3 and c < 0:
                    c += kk
                ents.append((r, c, v))
            ents.sort()
            rows += [e[0] for e in ents]
            cols += [e[1] for e in ents]
            vals += [e[2] for e in ents]
            off.append(len(rows))
            names.append([k, list(t.shape), t.fill])
    return {"weights.rows": np.asarray(rows, np.uint16), "weights.cols": np.asarray(cols, np.uint16),
            "weights.values": bf16_tensor(vals), "weights.off": np.asarray(off, np.int64)}, names


# --- the embedding's chip part --------------------------------------------------------------
def _sparse(v, cols):
    nz = np.flatnonzero(v)
    return cols[nz], v[nz]


def chip_basis(c, lay) -> dict:
    """The chip tokens' rows without the ballast. A family stores its base row and one delta per
    payload bit when its rows are linear in the bits, else every row."""
    feats = sorted(c.token_fns)
    cols = np.array([lay.dim(d) for d in feats], np.int64)

    def vec(t):
        return np.array([float(c.feature_value(d, t)) for d in feats])

    fams = dict(S.FAMILIES)
    if _vocab(c) > S.VOCAB:
        fams.update(S.NEW_FAMILIES)
    table, basis = [], []  # (first, size, bits, first basis row, kind)
    rng = np.random.default_rng(0)
    for _, (first, size) in sorted(fams.items(), key=lambda kv: kv[1][0]):
        nb = (size - 1).bit_length()
        v0 = vec(first)
        deltas = [vec(first + (1 << i)) - v0 for i in range(nb)]
        check = range(size) if size <= 4096 else rng.choice(size, 2048, replace=False)
        linear = all((vec(first + int(p)) == v0 + sum((deltas[i] for i in range(nb) if int(p) >> i & 1),
                                                      np.zeros(len(feats)))).all() for p in check)
        if linear:
            table.append((first, size, nb, len(basis), F.LINEAR))
            basis += [v0, *deltas]
        else:
            assert size <= 4096, (first, size)
            table.append((first, size, nb, len(basis), F.EXPLICIT))
            basis += [vec(first + i) for i in range(size)]
    for t in S.CONTROLS.values():
        table.append((t, 1, 0, len(basis), F.EXPLICIT))
        basis.append(vec(t))
    table.sort()
    off, bc, bv = [0], [], []
    for v in basis:
        c_, v_ = _sparse(v, cols)
        bc.append(c_)
        bv.append(v_)
        off.append(off[-1] + len(c_))
    return {"emb.family": np.asarray(table, np.int64), "emb.basis.off": np.asarray(off, np.int64),
            "emb.basis.cols": np.concatenate(bc).astype(np.uint16),
            "emb.basis.values": bf16_tensor(np.concatenate(bv))}


def text_arrays(c, lay, T, v0: int, n_text: int | None = None) -> dict:
    """Qwen's rows' chip columns, as merge.text_rows writes them: one plain row, the byte bits,
    and the rows of the tokens the circuit treats specially, whole. ``n_text``: base's token count
    (a merged tokenizer also names the reserved rows, which merge saw as empty)."""
    h, be = lay.hidden, lay.off + lay.m
    special = sorted({T.TCO, T.TCC, *T.printable})
    q_plain = next(q for q in range(T.n) if q not in set(special))
    plain = embedding_rows(c, lay, [BR.text_token(q_plain)], h)[0].numpy().astype(np.float64)
    assert (plain[lay.off:be] == lay.ballast).all()
    byte_cols = np.array([[lay.dim(next(iter(c.tb[k][i].terms))) for i in range(8)]
                          for k in range(S.TEXT_ROW)], np.int64)
    plain[byte_cols.ravel()] = 0.0
    n = min(v0, n_text or T.n)
    by = np.zeros((n, S.TEXT_ROW), np.uint8)
    for q in range(n):
        by[q] = np.frombuffer(T.row_bytes(q), np.uint8)
    pc = np.arange(be, h)
    p_c, p_v = _sparse(plain[be:], pc)
    rows = embedding_rows(c, lay, [BR.text_token(q) for q in special], h).numpy()
    off, cs, vs = [0], [], []
    for r in rows:
        assert (r[lay.off:be] == lay.ballast).all()
        c_, v_ = _sparse(r[be:].astype(np.float64), pc)
        cs.append(c_)
        vs.append(v_)
        off.append(off[-1] + len(c_))
    return {"text.plain.cols": p_c.astype(np.uint16), "text.plain.values": bf16_tensor(p_v),
            "text.byte_cols": byte_cols.astype(np.uint16), "text.bytes": by,
            "text.special.ids": np.asarray(special, np.int32),
            "text.special.off": np.asarray(off, np.int64),
            "text.special.cols": np.concatenate(cs).astype(np.uint16),
            "text.special.values": bf16_tensor(np.concatenate(vs))}


# --- the rest -------------------------------------------------------------------------------
def head_names(c) -> tuple:
    """([[layer, key head, unique name]..], {table name: head name})."""
    heads, seen, name_of = [], set(), {}
    for L in range(c.n_layers()):
        for j, n in enumerate(n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1):
            nm = n.name if n.name not in seen else f"{n.name}#{L}.{j}"
            seen.add(nm)
            heads.append([L, j, nm])
            name_of[id(n)] = nm
    return heads, {k: name_of[id(n)] for k, n in c.tables.items()}, name_of


def state_arrays(c, index_of: dict) -> dict:
    """The heads' init (core.snapshot's result) as rows of bytes: columns 0, 1 are the ballast
    (a bit each), the rest bytes, and any value that is not one a separate entry."""
    hd, xs, rs, refs, data, xi, xv = [], [], [], [], [], [], []
    for n in c.nodes:
        if not isinstance(n, Head) or not n.init:
            continue
        for x, d in enumerate(n.init):
            for r, vec in d.items():
                v = np.asarray(vec, np.float64)
                if not v.any():
                    continue
                i = len(hd)
                hd.append(index_of[id(n)])
                xs.append(x)
                rs.append(r)
                refs.append(int(v[0] == A_REF) | int(v[1] == A_REF) << 1)
                ints = np.rint(v)
                ok = (ints == v) & (v >= 0) & (v <= 255)
                ok[:2] = (v[:2] == 0) | (v[:2] == A_REF)
                data.append(np.where(ok[2:], ints[2:], 0).astype(np.uint8))
                for col in np.flatnonzero(~ok).tolist():
                    xi.append(i * HD + col)
                    xv.append(v[col])
    out = {"state.head": np.asarray(hd, np.uint16), "state.vh": np.asarray(xs, np.uint8),
           "state.row": np.asarray(rs, np.uint8), "state.ref": np.asarray(refs, np.uint8),
           "state.data": np.stack(data)}
    if xi:
        out["state.x.index"] = np.asarray(xi, np.int64)
        out["state.x.value"] = np.asarray(xv, np.float32)
    return out


def write(path, arrays: dict, meta: dict) -> dict:
    from safetensors.torch import save_file

    js = json.dumps(meta, separators=(",", ":"), sort_keys=True)
    tensors = {k: v if isinstance(v, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(v))
               for k, v in arrays.items()}
    save_file(tensors, str(path), metadata={"format": F.FORMAT, "version": str(F.VERSION),
                                            "meta": js, "config_hash": F.config_hash(js),
                                            "content_hash": F.content_hash(tensors)})
    return {k: t.numel() * t.element_size() for k, t in tensors.items()}


def build_file(c, lay, alloc, out, T=None, v0: int = 0, machine=None, extra_meta=None,
               extra_arrays=None, n_text=None) -> dict:
    keep = set(c.keep)
    tc = dict(hidden_size=lay.hidden, linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate, linear_conv_kernel_dim=4)
    arrays, names = weight_arrays(emit_sparse(c, lay, tc, alloc, keep), c.n_layers(), 4)
    arrays.update(chip_basis(c, lay))
    if T is not None:
        arrays.update(text_arrays(c, lay, T, v0, n_text))
    heads, tables, name_of = head_names(c)
    index_of = {i: x for x, i in enumerate(name_of)}
    if machine is not None:
        core.snapshot(c, machine)
        arrays.update(state_arrays(c, index_of))
    code = [next(iter(e.terms)) for e in c.code]
    gains = {lay.dim(d): code_gain(lay) for d in code}
    if T is not None:
        gains[lay.dim(next(iter(c.tau.terms)))] = TAU_GAIN
    gc = sorted(gains)
    arrays["norm.cols"] = np.asarray(gc, np.uint16)
    arrays["norm.values"] = bf16_tensor([gains[x] for x in gc])
    banks = {}
    for n in c.nodes:
        if isinstance(n, Head) and n.bank is not None:
            banks.setdefault(n.bank, []).append((n.bank_h, name_of[id(n)]))
    snap = {"reg_cols": c.reg_cols, "pc_cols": getattr(c, "pc_cols", {i: 2 + i for i in range(16)})}
    for k in ("pch_cols", "rom_plan", "img_plan"):
        if hasattr(c, k):
            snap[k] = getattr(c, k)
    meta = {
        "layout": {"hidden": lay.hidden, "off": lay.off, "m": lay.m,
                   "log2_ballast": lay.log2_ballast},
        "arch": {"chip_layers": c.n_layers(), "key_heads": c.target.key_heads,
                 "value_heads": c.target.value_heads, "intermediate": c.target.intermediate,
                 "conv": 4},
        "v0": v0, "vocab": _vocab(c), "ncode": core.NFAM + core.NPAY, "nfam": core.NFAM,
        "keep": [lay.dim(d) for d in c.keep], "tensors": names, "heads": heads, "tables": tables,
        "banks": [{"name": g, "heads": [nm for _, nm in sorted(v)]} for g, v in banks.items()],
        "snapshot": snap, "cfg": dataclasses.asdict(c.cfg) if getattr(c, "cfg", None) else None,
        "stats": c.stats(), "state": machine is not None, **(extra_meta or {})}
    arrays.update(extra_arrays or {})
    sizes = write(out, arrays, meta)
    return {"file_bytes": Path(out).stat().st_size, "arrays": sizes,
            "terms": int(arrays["weights.off"][-1])}


def read_sinks(dense: Path, meta: dict, tc: dict) -> dict:
    """The mute column of k_proj on Qwen's full-attention layers (fitted when the checkpoint was
    merged; it cannot be rebuilt from base alone)."""
    from safetensors import safe_open

    idx = json.loads((dense / "model.safetensors.index.json").read_text())["weight_map"]
    mu, out, ids = meta["mu_dim"], [], []
    for i, kind in enumerate(tc["layer_types"]):
        if kind != "full_attention":
            continue
        k = f"model.layers.{i}.self_attn.k_proj.weight"
        with safe_open(dense / idx[k], framework="pt") as f:
            out.append(f.get_slice(k)[:, mu].contiguous())
        ids.append(i)
    return {"mute.sinks": torch.stack(out), "ids": ids}


def build_q4p2(dense, base, out, firmware=None) -> dict:
    """rv-q4p2: the circuit the published checkpoint was merged from."""
    dense = Path(dense)
    meta = json.loads((dense / "chipformer.json").read_text())
    tc = json.loads((dense / "config.json").read_text())
    T = BR.Text(dense / "tokenizer.json")
    c = core.build(core.Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                               tc["intermediate_size"]), T)
    keep = set(c.keep)
    al = allocate(c, None, keep, tc["intermediate_size"])
    lay = Layout(hidden=meta["hidden"], off=meta["chip_dims_offset"], m=meta["ballast_dims"],
                 slots=al.slots)
    cols = [lay.dim(d) for d in c.keep]
    assert cols == meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]], "not this checkpoint's layout"
    s = read_sinks(dense, meta, tc)
    m = None
    if firmware:
        m = ref.Machine.from_program(elf.program(firmware))
        BR.park(m)
    em = {"model_vocab": tc["vocab_size"], "n_qwen": meta["qwen_layers_from"] and
          tc["num_hidden_layers"] - meta["chip_layers"],
          "mute": {"col": meta["mu_dim"], "full_attention": s["ids"], "muted": meta["muted"]},
          "code_dims": meta["code_dims"], "tau_dim": meta["tau_dim"], "mu_dim": meta["mu_dim"],
          "base": "Qwen/Qwen3.5-4B", "tool_call": meta["tool_call"]}
    return build_file(c, lay, al, out, T, meta["chip_token_offset"], m, em,
                      {"mute.sinks": s["mute.sinks"]},
                      BR.Text(Path(base) / "tokenizer.json").n)


def standalone(c, out, machine=None, extra=None) -> dict:
    """A circuit with no text bridge (the paged checks, fx's session)."""
    keep = set(c.keep)
    al = allocate(c, None, keep, c.target.intermediate)
    h, m = pick_hidden(D0, 1 + al.n_slots)
    lay = Layout(hidden=h, off=D0, m=m, slots=al.slots)
    code = [lay.dim(d) for d in c.keep]
    return build_file(c, lay, al, out, None, 0, machine,
                      {"code_dims": code, **(extra or {})})


def merged_fx(c, T, out, machine, n_text: int) -> dict:
    """The fx-sized circuit with its text bridge, as a release would hold it. No dense file exists
    for it, so there is no mute column and nothing to compare with."""
    keep = set(c.keep)
    al = allocate(c, None, keep, c.target.intermediate)
    h, m = pick_hidden(D0, 1 + al.n_slots)
    lay = Layout(hidden=h, off=D0, m=m, slots=al.slots)
    v0 = 248320
    cols = [lay.dim(d) for d in c.keep]
    em = {"code_dims": cols[:core.NFAM + core.NPAY], "tau_dim": cols[-2], "mu_dim": cols[-1],
          "model_vocab": -(-(v0 + c.vocab) // 64) * 64, "base": "Qwen/Qwen3.5-4B",
          "mute": {"col": cols[-1], "full_attention": [], "muted": False}}
    return build_file(c, lay, al, out, T, v0, machine, em, None, n_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("rv-q4p2", "paged", "fx"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense", help="rv-q4p2: the dense checkpoint's directory")
    ap.add_argument("--base", help="rv-q4p2: base Qwen's directory")
    ap.add_argument("--firmware", help="rv-q4p2: an ELF, parked, as the file's state")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--log", help="fx: the session log")
    ap.add_argument("--elf", help="fx: the ELF")
    a = ap.parse_args()
    if a.what == "rv-q4p2":
        r = build_q4p2(a.dense, a.base, a.out, a.firmware)
    elif a.what == "paged":
        r = standalone(core.build(cfg=S.small_cfg(a.frames)), a.out)
    else:
        from chipformer.rv import fxrun

        cfg, m, log, _ = fxrun.load(a.log, a.elf)
        fxrun.serve(m, log[0])  # the boot snapshot stands on an ecall: the host serves it first
        m.n += 1
        if a.base:  # with the text bridge, as a release would have it
            T = BR.Text(Path(a.base) / "tokenizer.json")
            r = merged_fx(core.build(core.T4B, T, cfg=cfg, go_on_tcc=False), T, a.out, m, T.n)
        else:
            r = standalone(core.build(cfg=cfg), a.out, m)
    print(json.dumps({k: v for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    main()
