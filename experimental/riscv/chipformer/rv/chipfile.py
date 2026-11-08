"""chip.safetensors: the chip's weights as a small file, and the circuit Executor v4 lifts from them.

    c = chipfile.load("chip.safetensors")               # no circuit builder anywhere on this path
    ex = exec4.Exec4(c, text=None, v0=c.v0, tables=c.exec_tables())
    c.load_state(ex)                                    # or core.snapshot(c, machine); ex.load_init()

The file holds the nonzeros of the 14 tensors of every chip layer, exactly as the dense checkpoint
has them (bf16 values, at their row and column), the chip part of the embedding, the final norm's
chip gains, Qwen's mute column, and optionally a firmware's state (ROM, RAM, registers: state, not
weights). docs/fx/chipfile.md has the format. Nothing here imports chipformer.rv.core.

**What is derived from the weights** (see lift):
  nodes     a layer's heads and MLP units are the rows of its tensors, undone from the emitter's
            scaling: x * c on the input side, K on gates, 1 / K on the a-form, the constant on the
            ballast columns.
  dims      a residual column may serve several values in turn: a slot is handed over by a unit
            -x * relu(ONE) out of x's own column. Each handover starts a new version of the column,
            so versions are the circuit's single-writer dims and the units themselves drop out.
            The exit's clears are handovers too, and the units that remove the ballast write only
            ballast columns.
  banks     a bank's address form (gate, row, head) is solved from its query and key rows: row 0 of
            head 0, and the rows and heads at powers of 2. Every other row of every head must then
            equal the form's prediction, or the load fails. So every memory weight is read.
  decode    the output code of each token is in its embedding row, as the tied output head uses it:
            the decode table is that map inverted.
  phases    exec4.schedule, from the nodes.

**What cannot come from the weights, so the file stores it** (meta): the layout (hidden size, first
chip column, ballast dims and value), the columns of the output code in bit order, the head names
and the bank groups' membership (checked, not trusted), and the field layouts a snapshot writes
(register columns, PC columns, ROM plans, the memory map). The kernel's own semantics (relu for
silu, decay by 1, 0.5 or 0, the 2-hot read) are code.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35.ir import A_REF, FFN, HD, Chan, Dim, Head, K, Target, VHead
from chipformer.rv import spec as S

FORMAT = "qwenputer-chip"
VERSION = 1
LINEAR, EXPLICIT = 0, 1  # how a token family's embedding rows are stored
LAYER_TENSORS = ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                 "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                 "linear_attn.conv1d.weight", "linear_attn.dt_bias", "linear_attn.A_log",
                 "linear_attn.norm.weight", "linear_attn.out_proj.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


class ChipFileError(ValueError):
    """The file is not a chip this code can run, or its parts disagree."""


def config_hash(meta_json: str) -> str:
    return hashlib.sha256(meta_json.encode()).hexdigest()


def content_hash(tensors: dict) -> str:
    """Of every array, in name order: dtype, shape and bytes."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k]
        h.update(f"{k} {t.dtype} {list(t.shape)}".encode())
        h.update(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class ChipFile:
    """The raw file: metadata and arrays, nothing lifted."""

    def __init__(self, path) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        with safe_open(self.path, framework="pt") as f:
            md = f.metadata() or {}
            if md.get("format") != FORMAT:
                raise ChipFileError(f"{path}: not a {FORMAT} file")
            if int(md.get("version", -1)) != VERSION:
                raise ChipFileError(f"{path}: format version {md.get('version')}, this code reads "
                                    f"{VERSION}")
            if config_hash(md["meta"]) != md["config_hash"]:
                raise ChipFileError(f"{path}: the metadata does not match its hash")
            self.meta = json.loads(md["meta"])
            self.config_hash = md["config_hash"]
            self.t = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        if content_hash(self.t) != md["content_hash"]:
            raise ChipFileError(f"{path}: the arrays do not match their hash")
        self.index = {e[0]: i for i, e in enumerate(self.meta["tensors"])}

    def arr(self, name: str) -> np.ndarray:
        t = self.t[name]
        return t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()

    def has(self, name: str) -> bool:
        return name in self.t

    def weights(self):
        """(rows, cols, values, offsets) of every chip tensor's terms, as int64 / float64."""
        if not hasattr(self, "_w"):
            self._w = (self.arr("weights.rows").astype(np.int64),
                       self.arr("weights.cols").astype(np.int64),
                       self.arr("weights.values").astype(np.float64),
                       self.arr("weights.off").astype(np.int64))
        return self._w

    def terms(self, name: str):
        """A chip tensor's nonzeros: (rows, cols, values), sorted by row then column."""
        r, c, v, off = self.weights()
        i = self.index[name]
        return r[off[i]:off[i + 1]], c[off[i]:off[i + 1]], v[off[i]:off[i + 1]]


class _Rows:
    """A tensor's terms grouped by row."""

    def __init__(self, r, c, v) -> None:
        if len(r):
            u, first = np.unique(r, return_index=True)
            ends = np.append(first[1:], len(r))
        else:
            u, first, ends = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.c, self.v = c.tolist(), v.tolist()
        self.span = dict(zip(u.tolist(), zip(first.tolist(), ends.tolist())))

    def __contains__(self, r) -> bool:
        return r in self.span

    def __iter__(self):
        return iter(self.span)

    def get(self, r):
        a, b = self.span[r]
        return self.c[a:b], self.v[a:b]


class ChipCircuit:
    """The circuit lifted from a chip file: what Exec4, hf.inject and core.snapshot read of a
    core.build() result (dims, nodes, banks, keep, tables, the snapshot's field layouts), with the
    embedding and the decode table from the file instead of Python token functions."""

    def __init__(self, f: ChipFile) -> None:
        self.file = f
        m = f.meta
        lay = m["layout"]
        self.hidden, self.off, self.m = lay["hidden"], lay["off"], lay["m"]
        self.ball_end = self.off + self.m
        ballast = 2.0 ** lay["log2_ballast"]
        log4 = math.log(self.m * ballast * ballast / self.hidden, 4)
        if abs(log4 - round(log4)) > 1e-12:
            raise ChipFileError("m * B^2 / hidden is not a power of 4")
        self.scale = 2.0 ** -round(log4)
        self.one_w = ballast * self.scale  # a constant's weight unit on the ballast columns
        a = m["arch"]
        self.n_chip, self.nk, self.nv = a["chip_layers"], a["key_heads"], a["value_heads"]
        self.inter = a["intermediate"]
        self.target = Target(self.nk, self.nv, self.inter)
        self.v0, self.vocab = m["v0"], m["vocab"]
        self.keep_cols = m["keep"]
        self.ncode, self.nfam = m["ncode"], m["nfam"]
        self.code_fields = (self.ncode, self.nfam)  # (bits of the code, family bits among them)
        snap = m.get("snapshot") or {}
        self.reg_cols = {int(k): v for k, v in snap.get("reg_cols", {}).items()}
        self.pc_cols = {int(k): v for k, v in snap.get("pc_cols", {}).items()}
        if "pch_cols" in snap:
            self.pch_cols = {int(k): v for k, v in snap["pch_cols"].items()}
        for k in ("rom_plan", "img_plan"):
            if k in snap:
                setattr(self, k, snap[k])
        self.cfg = None
        if m.get("cfg"):
            cf = dict(m["cfg"])
            cf["rw"] = tuple(tuple(x) for x in cf["rw"])
            cf["image"] = tuple(cf["image"])
            self.cfg = S.MemCfg(**cf)
        self._lift()
        self.banks = {}
        for g in m.get("banks", []):
            self._bank(g["name"], g["heads"])
        self.tables = {nm: self.by_name[nm] for nm in m["tables"]}
        self.text = None  # (the decode table holds what a Q code means)
        self._tabs = None

    def n_layers(self) -> int:
        return self.n_chip

    def stats(self) -> dict:
        return {"layers": self.n_chip, "dims": len(self.dims),
                "heads": sum(isinstance(n, Head) for n in self.nodes),
                "ffn_units": sum(len(n.terms) for n in self.nodes if isinstance(n, FFN))}

    # --- the lift ---------------------------------------------------------------------------
    def _tensor(self, L: int, short: str):
        return self.file.terms(f"model.layers.{L}.{short}")

    def _check_constants(self) -> None:
        """Tensors the kernel holds as constants in its code (a chip layer's norms are off, decay
        is keep / halve / clear) must have those values here."""
        want = {"input_layernorm.weight": 0.0, "post_attention_layernorm.weight": 0.0,
                "linear_attn.dt_bias": -110.0, "linear_attn.A_log": 0.0,
                "linear_attn.norm.weight": 1.0}
        for name, _, fill in self.file.meta["tensors"]:
            short = name.split(".", 3)[3]
            if short in want and (fill != want[short] or len(self.file.terms(name)[0])):
                raise ChipFileError(f"{name}: not the constant the kernel assumes")

    def _handovers(self):
        """col -> [levels of the slot's handover units], and per layer the units that drop out."""
        hand, drop = defaultdict(list), []
        for L in range(self.n_chip):
            level = 2 * L + 2
            ur, uc, uv = self._tensor(L, "mlp.up_proj.weight")
            gr, gc, gv = self._tensor(L, "mlp.gate_proj.weight")
            dr, dc, dv = self._tensor(L, "mlp.down_proj.weight")
            dropped = set(dc[dr < self.ball_end].tolist())  # the exit removes the ballast
            n_up = np.bincount(ur, minlength=self.inter)
            first_up = np.searchsorted(ur, np.arange(self.inter))
            cand = np.flatnonzero((dv == -1.0) & (dr >= self.ball_end) & (n_up[dc] == 1))
            first_g = np.searchsorted(gr, np.arange(self.inter + 1))
            for i in cand.tolist():
                col, u = int(dr[i]), int(dc[i])
                j = first_up[u]
                if uc[j] != col or uv[j] * self.scale * K != 1.0:
                    continue
                a, b = first_g[u], first_g[u + 1]
                if a == b or not (gc[a:b] < self.ball_end).all():
                    continue
                if float(gv[a:b].sum()) * self.one_w / K != 1.0:  # the gate must be relu(ONE)
                    continue
                hand[col].append(level)
                dropped.add(u)
            drop.append(dropped)
        return hand, drop

    def _feature_cols(self) -> np.ndarray:
        f = self.file
        cols = [f.arr("emb.basis.cols")]
        if f.has("text.plain.cols"):
            cols += [f.arr("text.plain.cols"), f.arr("text.byte_cols").ravel(),
                     f.arr("text.special.cols")]
        return np.unique(np.concatenate(cols).astype(np.int64))

    def _assign_dims(self, hand, drop):
        """The dims: every version of a column that something reads or writes, or a token
        carries. Returns (id map [column, version] -> dim, the columns' final versions)."""
        hidden = self.hidden
        ver = np.zeros(hidden, np.int64)
        by_level = defaultdict(list)
        for col, lv in hand.items():
            for x in lv:
                by_level[x].append(col)
        used = set((self._feature_cols() << 10).tolist())

        def mark(cols):
            cols = np.unique(cols[cols >= self.ball_end])
            used.update(((cols << 10) | ver[cols]).tolist())

        for L in range(self.n_chip):
            for t in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                mark(self._tensor(L, f"linear_attn.{t}.weight")[1])
            mark(self._tensor(L, "linear_attn.out_proj.weight")[0])
            dropped = np.fromiter(drop[L], np.int64, len(drop[L]))
            for t in ("gate_proj", "up_proj"):
                r, c, _ = self._tensor(L, f"mlp.{t}.weight")
                mark(c[~np.isin(r, dropped)])
            cols = by_level.get(2 * L + 2)
            if cols:
                ver[cols] += 1
            dr, dc, _ = self._tensor(L, "mlp.down_proj.weight")
            mark(dr[~np.isin(dc, dropped)])
        mark(np.asarray(self.keep_cols, np.int64))
        idmap = np.full((hidden, int(ver.max()) + 1), -1, np.int64)
        born = {}
        for col, lv in hand.items():
            for v, x in enumerate(sorted(set(lv))):
                born[col << 10 | (v + 1)] = x
        keys = sorted(used, key=lambda k: (born.get(k, 0), k))
        self.dims = [Dim("one", "one", 0)]
        for k in keys:
            col, v = k >> 10, k & 1023
            idmap[col, v] = len(self.dims)
            self.dims.append(Dim(f"c{col}.{v}", "chip", 0))
        return idmap, ver, by_level

    def _lift(self) -> None:
        self._check_constants()
        hand, drop = self._handovers()
        idmap, final_ver, by_level = self._assign_dims(hand, drop)
        ver = np.zeros(self.hidden, np.int64)
        cols = np.arange(self.hidden)
        self.keep = [int(idmap[c, final_ver[c]]) for c in self.keep_cols]
        self.idmap = idmap  # [column, version] -> dim
        self.dim0 = idmap[:, 0].copy()  # version 0: the dims a token's embedding sets
        self.nodes = []
        heads_meta = {(L, j): nm for L, j, nm in self.file.meta["heads"]}
        for L in range(self.n_chip):
            dim_at = idmap[cols, ver].tolist()
            self._lift_heads(L, dim_at, heads_meta)
            dim_at = idmap[cols, ver].tolist()
            hc = by_level.get(2 * L + 2)
            if hc:
                ver[hc] += 1
            self._lift_units(L, dim_at, idmap[cols, ver].tolist(), drop[L])
        self.by_name = {n.name: n for n in self.nodes if isinstance(n, Head)}
        self.token_fns = {int(self.dim0[c]): None for c in self._feature_cols().tolist()}
        self.full_fns = set(self.token_fns)

    def _expr(self, cols, vals, dim_at, w: float = 1.0) -> Expr:
        """A projection row back in IR units: emit.row undone."""
        d: dict = {}
        be, sc, ow = self.ball_end, self.scale, self.one_w
        for cl, x in zip(cols, vals):
            if cl < be:
                d[ONE] = d.get(ONE, 0.0) + x * ow / w
            else:
                k = dim_at[cl]
                d[k] = d.get(k, 0.0) + x * sc / w
        return Expr(d)

    def _taps(self, conv: _Rows, r: int) -> tuple:
        cs, vs = conv.get(r)
        kk = self.file.meta["arch"]["conv"]
        t = [0.0] * 4
        for k, x in zip(cs, vs):
            t[kk - 1 - k] = x
        return tuple(t)

    def _lift_heads(self, L: int, dim_at: list, names: dict) -> None:
        nk, nv = self.nk, self.nv
        rr = nv // nk
        level = 2 * L + 1
        beta = _Rows(*self._tensor(L, "linear_attn.in_proj_b.weight"))
        if not len(beta.span):
            return
        qkv = _Rows(*self._tensor(L, "linear_attn.in_proj_qkv.weight"))
        conv = _Rows(*self._tensor(L, "linear_attn.conv1d.weight"))
        zw = _Rows(*self._tensor(L, "linear_attn.in_proj_z.weight"))
        aw = _Rows(*self._tensor(L, "linear_attn.in_proj_a.weight"))
        qs, ks, vs = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for r in qkv:
            ch = Chan(self._expr(*qkv.get(r), dim_at), self._taps(conv, r))
            if r < nk * HD:
                qs[r // HD][r % HD] = ch
            elif r < 2 * nk * HD:
                ks[(r - nk * HD) // HD][(r - nk * HD) % HD] = ch
            else:
                vs[(r - 2 * nk * HD) // HD][(r - 2 * nk * HD) % HD] = ch
        zs = defaultdict(dict)
        for r in zw:
            zs[r // HD][r % HD] = self._expr(*zw.get(r), dim_at)
        outs = defaultdict(list)
        or_, oc, ov = self._tensor(L, "linear_attn.out_proj.weight")
        for col, c, w in zip(or_.tolist(), oc.tolist(), ov.tolist()):
            outs[c // HD].append((c % HD, dim_at[col], w))
        # every term is read: nothing may sit on a value head no write gate belongs to
        extra = (set(vs) | set(zs) | set(outs) | set(aw)) - set(beta)
        extra |= {j * rr for j in set(qs) | set(ks)} - set(beta)
        if extra:
            raise ChipFileError(f"layer {L}: weights on unused value heads {sorted(extra)[:4]}")
        for j in range(nk):
            vhs = []
            for x in range(rr):
                vh = j * rr + x
                if vh not in beta:
                    continue
                b = self._expr(*beta.get(vh), dim_at)
                bt = None
                if b.terms != {ONE: 128.0}:  # always written; else sigmoid(256 g - 128)
                    t = {d: v / 256.0 for d, v in b.terms.items()}
                    t[ONE] = t.get(ONE, 0.0) + 0.5
                    bt = Expr(t)
                a = self._expr(*aw.get(vh), dim_at) if vh in aw else Expr()
                vhs.append(VHead(vs.get(vh, {}), a, zs.get(vh, {}), outs.get(vh, []), bt, True))
            if not vhs:
                continue
            nm = names.get((L, j), f"h{L}.{j}")
            v0 = vhs[0]
            self.nodes.append(Head(nm, qs.get(j, {}), ks.get(j, {}), v0.v, v0.a, v0.z, v0.out,
                                   level, HD, 0.0, v0.beta, True, vhs[1:]))

    def _lift_units(self, L: int, dim_at: list, dim_after: list, dropped: set) -> None:
        gate = _Rows(*self._tensor(L, "mlp.gate_proj.weight"))
        up = _Rows(*self._tensor(L, "mlp.up_proj.weight"))
        dr, dc, dv = self._tensor(L, "mlp.down_proj.weight")
        order = np.argsort(dc, kind="stable")
        terms = defaultdict(list)
        if (set(gate) | set(up)) - set(dc.tolist()):  # every term is read
            raise ChipFileError(f"layer {L}: MLP rows of a unit with no output")
        for i in order.tolist():
            u = int(dc[i])
            if u in dropped:
                continue
            a = self._expr(*up.get(u), dim_at, 1.0 / K) if u in up else Expr()
            b = self._expr(*gate.get(u), dim_at, K) if u in gate else Expr()
            terms[int(dr[i])].append((float(dv[i]), a, b))
        level = 2 * L + 2
        for col, ts in terms.items():
            self.nodes.append(FFN(f"u{L}.{col}", ts, dim_after[col], level, True))

    # --- banks ------------------------------------------------------------------------------
    def _bank(self, group: str, names: list) -> None:
        """A bank's regular form, solved from its rows and checked against all of them."""
        ns = [self.by_name[x] for x in names]
        n0 = ns[0]
        for n in ns:
            if not _same_vheads(n0, n):
                raise ChipFileError(f"bank {group}: {n.name} has other values than {n0.name}")
        if n0.vheads()[0].a.terms:
            raise ChipFileError(f"bank {group}: a bank head cannot decay")

        def form_of(chan):  # the 0/1 form e of a key or query K * (2 e - 1)
            t = {d: v / (2 * K) for d, v in chan.expr.terms.items()}
            t[ONE] = t.get(ONE, 0.0) + 0.5
            return {d: v for d, v in t.items() if v}

        rows = sorted(n0.q)
        nr = (rows[-2] if len(rows) > 1 else 0).bit_length()
        nb = (len(ns) - 1).bit_length()
        q00 = form_of(n0.q[0])
        drow = [_sub(form_of(n0.q[1 << i]), q00) for i in range(nr)]
        dhead = [_sub(form_of(ns[1 << j].q[0]), q00) for j in range(nb)]
        half = lambda d: _scale(_add(d, {ONE: 1.0}), 0.5)  # (2 b - ONE) -> b  # noqa: E731
        rb, hb = [half(d) for d in drow], [half(d) for d in dhead]
        bits_sum: dict = {}
        for b in rb + hb:
            bits_sum = _add(bits_sum, b)
        rowf: dict = {}
        for i, b in enumerate(rb):
            rowf = _add(rowf, _scale(b, 2.0**i))
        headf: dict = {}
        for j, b in enumerate(hb):
            headf = _add(headf, _scale(b, 2.0**j))
        form = {"gate": Expr(_add(q00, bits_sum)), "row": Expr(rowf), "head": Expr(headf),
                "nh": len(ns), "zrow": None}
        rsum = {}
        for b in rb:
            rsum = _add(rsum, b)
        nreg = 1 << nr
        self._check_rows(group, ns, "q", q00, drow, dhead, nreg, (1.0, 0.0, 0.0, 0.0), rows,
                         form, rsum, form_of)
        if n0.k:
            k00 = form_of(n0.k[0])
            form["kgate"] = Expr(_add(k00, bits_sum))
            self._check_rows(group, ns, "k", k00, drow, dhead, nreg, (0.0, 1.0, 0.0, 0.0),
                             sorted(n0.k), None, rsum, form_of)
        self.banks[group] = form
        for h, n in enumerate(ns):
            n.bank, n.bank_h = group, h

    @staticmethod
    def _check_rows(group, ns, side, e00, drow, dhead, nreg, taps, rows, form, rsum, form_of):
        """Every row of every head equals the regular form's prediction. A row beyond the regular
        pattern can only be the all-zero ballast row Z of a 2-hot read (rows 127, form["zrow"])."""
        for h, n in enumerate(ns):
            eh = e00
            for j, d in enumerate(dhead):
                if h >> j & 1:
                    eh = _add(eh, d)
            chans = getattr(n, side)
            if sorted(chans) != rows:
                raise ChipFileError(f"bank {group}: {n.name} has other {side} rows")
            pred = {0: eh}
            for r in rows:
                if r >= nreg:
                    break
                if r:
                    low = r & -r
                    pred[r] = _add(pred[r & (r - 1)], drow[low.bit_length() - 1])
                if pred[r] != form_of(chans[r]) and not (r == 127 and form is not None):
                    raise ChipFileError(f"bank {group}: {n.name} {side}[{r}] is not regular")
                if chans[r].taps != taps:
                    raise ChipFileError(f"bank {group}: {n.name} {side}[{r}] has other taps")
            if form is not None and 127 in chans:
                z = _add(eh, rsum)
                got = form_of(chans[127])
                if got == z and (nreg <= 127 or got != pred.get(127)):
                    form["zrow"] = 127
                elif nreg <= 127 or got != pred.get(127):
                    raise ChipFileError(f"bank {group}: {n.name} row 127 is neither kind")

    # --- the tables -------------------------------------------------------------------------
    def _chip_rows(self):
        """(off, column, value) of every chip token's row, ballast left out: the token's features
        in residual columns. Rows follow the token ids."""
        f = self.file
        fam = f.arr("emb.family")
        boff = f.arr("emb.basis.off").astype(np.int64)
        bcols = f.arr("emb.basis.cols").astype(np.int64)
        bvals = f.arr("emb.basis.values")

        def basis(i):
            return bcols[boff[i]:boff[i + 1]], bvals[boff[i]:boff[i + 1]]

        per = [None] * self.vocab
        for first, size, nb, b0, kind in fam.tolist():
            if kind == EXPLICIT:
                for t in range(size):
                    per[first + t] = basis(b0 + t)
                continue
            cs = [basis(b0 + i) for i in range(nb + 1)]
            u = np.unique(np.concatenate([c for c, _ in cs]))
            pos = {c: i for i, c in enumerate(u.tolist())}
            base = np.zeros(len(u), np.float64)
            base[[pos[c] for c in cs[0][0].tolist()]] = cs[0][1]
            mat = np.tile(base, (size, 1))
            p = np.arange(size)
            for i in range(nb):
                dl = np.zeros(len(u), np.float64)
                dl[[pos[c] for c in cs[i + 1][0].tolist()]] = cs[i + 1][1]
                mat[(p >> i & 1) == 1] += dl
            nz = mat != 0
            n_per = nz.sum(1)
            rr, cc = np.nonzero(nz)
            vals = mat[rr, cc].astype(np.float32)
            cols = u[cc]
            ends = np.cumsum(n_per)
            starts = ends - n_per
            for t in range(size):
                per[first + t] = (cols[starts[t]:ends[t]], vals[starts[t]:ends[t]])
        n = np.fromiter((len(x[0]) for x in per), np.int64, self.vocab)
        off = np.zeros(self.vocab + 1, np.int64)
        off[1:] = np.cumsum(n)
        cols = np.concatenate([x[0] for x in per]).astype(np.int64) if self.vocab else n
        vals = np.concatenate([x[1] for x in per]).astype(np.float32) if self.vocab else n
        return off, cols, vals

    def _text_row(self, q: int):
        """(columns, values) of Qwen token q's chip part, ballast left out."""
        f = self.file
        ids = f.arr("text.special.ids")
        i = int(np.searchsorted(ids, q))
        if i < len(ids) and ids[i] == q:
            off = f.arr("text.special.off")
            a, b = int(off[i]), int(off[i + 1])
            return f.arr("text.special.cols")[a:b].astype(np.int64), f.arr("text.special.values")[a:b]
        by = f.arr("text.bytes")
        row = by[q] if q < len(by) else np.zeros(by.shape[1], np.uint8)
        bits = (row[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1
        bc = f.arr("text.byte_cols").astype(np.int64)
        cols = np.concatenate([f.arr("text.plain.cols").astype(np.int64), bc[bits == 1]])
        vals = np.concatenate([f.arr("text.plain.values"), np.ones(int(bits.sum()), np.float32)])
        return cols, vals

    def exec_tables(self) -> tuple:
        """exec4.build_tables' tuple from the file: (lut_id, lut_row, off, dims, vals)."""
        if self._tabs is None:
            self._tabs = self._build_tables()
        return self._tabs

    def _code_index(self, bits: np.ndarray) -> np.ndarray:
        """bits [n, ncode] -> the kernel's lut index fid << 16 | p."""
        w = 1 << np.arange(self.ncode - self.nfam, dtype=np.int64)
        f = 1 << np.arange(self.nfam, dtype=np.int64)
        return (bits[:, :self.nfam] @ f) << 16 | (bits[:, self.nfam:] @ w)

    def _build_tables(self) -> tuple:
        off, cols, vals = self._chip_rows()
        code_cols = np.asarray(self.keep_cols[:self.ncode], np.int64)
        where = np.full(self.hidden, -1, np.int64)
        where[code_cols] = np.arange(self.ncode)
        rows = np.repeat(np.arange(self.vocab), np.diff(off))
        # the decode table: each token's code, from its own embedding row
        pos = where[cols]
        m = pos >= 0
        bits = np.zeros((self.vocab, self.ncode), np.int64)
        bits[rows[m], pos[m]] = vals[m] > 0
        nf = 1 << self.nfam
        lut_id = np.full(nf << 16, -2, np.int32)
        lut_row = np.full(nf << 16, -1, np.int32)
        idx = self._code_index(bits)
        lut_id[idx] = self.v0 + np.arange(self.vocab)
        lut_row[idx] = np.arange(self.vocab)
        fids = idx >> 16
        for fid in np.unique(fids).tolist():  # a control is its family id, whatever the payload
            ts = np.flatnonzero(fids == fid)
            if len(ts) == 1:
                lut_id[fid << 16:(fid + 1) << 16] = self.v0 + ts[0]
                lut_row[fid << 16:(fid + 1) << 16] = ts[0]
        echo_d, echo_v = [], []
        self.echo = []
        if self.file.has("text.special.ids"):
            for q in self.file.arr("text.special.ids").tolist():
                c_, v_ = self._text_row(q)
                p = where[c_]
                if not (p >= 0).any():
                    continue
                b = np.zeros((1, self.ncode), np.int64)
                b[0, p[p >= 0]] = v_[p >= 0] > 0
                i = int(self._code_index(b)[0])
                lut_id[i] = q
                lut_row[i] = self.vocab + len(self.echo)
                self.echo.append(q)
                echo_d.append(c_)
                echo_v.append(v_)
        # the rows in dims: ONE first (every row has the ballast), then the features
        n_rows = self.vocab + len(self.echo)
        counts = np.concatenate([np.diff(off), np.fromiter((len(x) for x in echo_d), np.int64,
                                                           len(echo_d))]) + 1
        toff = np.zeros(n_rows + 1, np.int64)
        toff[1:] = np.cumsum(counts)
        dims = np.zeros(toff[-1], np.int32)
        tv = np.ones(toff[-1], np.float32)
        allc = np.concatenate([cols, *echo_d]) if echo_d else cols
        allv = np.concatenate([vals, *echo_v]) if echo_v else vals
        slots = np.ones(toff[-1], bool)
        slots[toff[:-1]] = False
        dims[slots] = self.dim0[allc]
        tv[slots] = allv
        dims[toff[:-1]] = ONE
        assert (dims >= 0).all() and toff[-1] < 2**31
        return lut_id, lut_row, toff.astype(np.int32), dims, tv

    def embed_rows(self, toks) -> np.ndarray:
        """Exec4.embed: the rows of these tokens (chip or text), dense, in dims. Text tokens the
        table lacks come from the bit table."""
        lut_id, _, off, dims, vals = self.exec_tables()
        echo = {q: self.vocab + i for i, q in enumerate(self.echo)}
        out = np.zeros((len(toks), len(self.dims)), np.float32)
        for i, t in enumerate(toks):
            t = int(t)
            if t >= S.TEXT_TOKEN:
                q = t - S.TEXT_TOKEN
                if q in echo:
                    r = echo[q]
                    out[i, dims[off[r]:off[r + 1]]] = vals[off[r]:off[r + 1]]
                else:
                    c_, v_ = self._text_row(q)
                    out[i, ONE] = 1.0
                    out[i, self.dim0[c_]] = v_
            else:
                out[i, dims[off[t]:off[t + 1]]] = vals[off[t]:off[t + 1]]
        return out

    def feature_value(self, d: int, t: int) -> float:
        """As Graph.feature_value (for ir.evaluate on the lifted circuit: an audit, not the fast
        path), from the file's rows, a token at a time."""
        c = self._feat_cache = getattr(self, "_feat_cache", {})
        if t not in c:
            if len(c) > 200000:
                c.clear()
            c[t] = self.embed_rows([t])[0]
        return float(c[t][d])

    def decode_code(self, fid: int, p: int) -> int:
        """core.decode_code from the table: the token of an output code, -1 if it names none."""
        lut_id = self.exec_tables()[0]
        i = fid << 16 | p
        if not 0 <= i < len(lut_id) or lut_id[i] == -2:
            return -1
        q = int(lut_id[i])
        return q - self.v0 if q >= self.v0 else S.TEXT_TOKEN + q

    # --- state ------------------------------------------------------------------------------
    def has_state(self) -> bool:
        return self.file.has("state.row")

    def state_rows(self):
        """The firmware's state in the file: (head index, value head, row, rows [n, HD] fp32)."""
        f = self.file
        data = f.arr("state.data")
        n = len(data)
        out = np.zeros((n, HD), np.float32)
        out[:, 2:] = data
        ref = f.arr("state.ref")
        out[:, 0] = (ref & 1).astype(np.float32) * A_REF
        out[:, 1] = (ref >> 1 & 1).astype(np.float32) * A_REF
        if f.has("state.x.index"):
            ix = f.arr("state.x.index")
            out.reshape(-1)[ix] = f.arr("state.x.value")
        return f.arr("state.head"), f.arr("state.vh"), f.arr("state.row"), out

    def load_state(self, ex) -> None:
        """Exec4's state buffer from the file (as core.snapshot then load_init, without a machine)."""
        heads = [nm for _, _, nm in self.file.meta["heads"]]
        h, x, r, rows = self.state_rows()
        vg0 = np.fromiter((ex.head_block[nm][0] for nm in heads), np.int64, len(heads))
        at = (vg0[h.astype(np.int64)] + x.astype(np.int64)) * HD * HD + r.astype(np.int64) * HD
        buf = torch.zeros(ex.S.numel(), dtype=torch.float32)
        ix = torch.from_numpy((at[:, None] + np.arange(HD)[None, :]).ravel())
        buf[ix] = torch.from_numpy(rows.ravel())
        ex.S.copy_(buf.to(ex.S.device))

    def apply_state(self) -> None:
        """The heads' init from the file's state (hf.inject and Exec4.load_init read that)."""
        heads = [nm for _, _, nm in self.file.meta["heads"]]
        for n in self.nodes:
            if isinstance(n, Head):
                n.init = None
        h, x, r, rows = self.state_rows()
        for hi, xi, ri, row in zip(h.tolist(), x.tolist(), r.tolist(), rows):
            n = self.by_name[heads[hi]]
            if n.init is None:
                n.init = [{} for _ in n.vheads()]
            n.init[xi][ri] = row


def _add(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        x = out.get(k, 0.0) + v
        if x:
            out[k] = x
        else:
            out.pop(k, None)
    return out


def _sub(a: dict, b: dict) -> dict:
    return _add(a, {k: -v for k, v in b.items()})


def _scale(a: dict, w: float) -> dict:
    return {k: v * w for k, v in a.items()}


def _same_vheads(a: Head, b: Head) -> bool:
    va, vb = a.vheads(), b.vheads()
    if len(va) != len(vb):
        return False
    for x, y in zip(va, vb):
        if x.a.terms != y.a.terms or x.out != y.out:
            return False
        if (x.beta is None) != (y.beta is None) or (x.beta and x.beta.terms != y.beta.terms):
            return False
        if x.v.keys() != y.v.keys() or x.z.keys() != y.z.keys():
            return False
        if any(x.v[c].taps != y.v[c].taps or x.v[c].expr.terms != y.v[c].expr.terms for c in x.v):
            return False
        if any(x.z[c].terms != y.z[c].terms for c in x.z):
            return False
    return True


def load(path) -> ChipCircuit:
    """The circuit of a chip file. The production path: no circuit builder is imported."""
    return ChipCircuit(ChipFile(path))


# --- Qwen beside the chip ---------------------------------------------------------------------
SRC_TEXT = "model.language_model."


def _text_config(base: Path) -> dict:
    tc0 = json.loads((base / "config.json").read_text())["text_config"]
    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(mtp_num_hidden_layers=0, tie_word_embeddings=True,
              architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")
    return tc


def _assemble(tc: dict, state: dict, device, dtype=torch.bfloat16):
    """A text-only Qwen3_5ForCausalLM on these tensors, nothing allocated but them."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    cfg = Qwen3_5TextConfig(**tc)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        m = Qwen3_5ForCausalLM(cfg)
    missing, unexpected = m.load_state_dict(state, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    assert all(k == "lm_head.weight" for k in missing), missing[:5]
    m.lm_head.weight = m.model.embed_tokens.weight
    m.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg)  # (its inv_freq stays fp32)
    for n, p in m.named_parameters():
        assert p.device.type != "meta", n
        if p.dtype != dtype:  # as from_pretrained(dtype=...) loads base's few fp32 tensors
            p.data = p.data.to(dtype)
    return m.to(device).eval()


def load_base(base, device):
    """Base Qwen3.5's text model from its own files, its own bytes: no chip tensor, no padding."""
    from safetensors import safe_open

    base = Path(base)
    state = {}
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if k.startswith(SRC_TEXT):
                    state["model." + k[len(SRC_TEXT):]] = fh.get_tensor(k)
    return _assemble(_text_config(base), state, device)


def slim_merged(model, c: ChipCircuit, base, device):
    """The stock merged model cut down to what the executor path reads: the chip layers freed, and
    Qwen's layers and embedding back to base's width, with base's norm gains (the widened gains do
    not invert exactly). Nothing here reads the rest: Exec4 runs the chip, and Qwen's layers only
    ever see text tokens, so the mute columns never fire."""
    import re

    from safetensors import safe_open

    from chipformer.q35.merge import IN_COLS, OUT_ROWS, WIDE_NORMS

    base = Path(base)
    wm = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    d0, nc, v0 = c.off, c.n_chip, c.v0
    state = {}
    old = model.state_dict()
    del model
    for k in list(old):
        w = old.pop(k)
        mt = re.match(r"model\.layers\.(\d+)\.(.*)", k)
        if k == "lm_head.weight" or (mt and int(mt[1]) < nc):
            continue
        k2 = f"model.layers.{int(mt[1]) - nc}.{mt[2]}" if mt else k
        if k == "model.embed_tokens.weight":
            w = w[:v0, :d0].clone()
        elif WIDE_NORMS.search(k) or k == "model.norm.weight":
            bk = SRC_TEXT + k2[len("model."):]
            with safe_open(base / wm[bk], framework="pt") as fh:
                w = fh.get_tensor(bk)
        elif IN_COLS.search(k):
            w = w[..., :d0].clone()
        elif OUT_ROWS.search(k):
            w = w[:d0].clone()
        state[k2] = w
    return _assemble(_text_config(base), state, device)
