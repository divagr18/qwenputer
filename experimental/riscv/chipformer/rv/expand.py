"""The dense stock checkpoint from base Qwen's bytes plus chip.safetensors, byte for byte.

    python -m chipformer.rv.expand CHIP BASE --out DIR            # write the shards
    python -m chipformer.rv.expand CHIP BASE --hub REPO [--rev R] # hash them against the Hub's
    python -m chipformer.rv.expand CHIP BASE --against DIR        # compare with local shards

Nothing is held but one tensor (a chunk of the big ones). --hub compares each shard's SHA-256 with
the Git LFS hash the Hub publishes for that file, so no shard is downloaded. That covers every
tensor of the safetensors files: the chip layers, Qwen's layers with their padding and mute column,
the norms, and the embedding. It does not cover the tokenizer and config files, which are not
weights. Qwen's tensors are rebuilt with q35.merge's rules (zero padding, the rescaled norms) and
q35.mute's; the file adds only what base cannot give: the chip's terms and the anti-sink keys.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from chipformer.q35.merge import IN_COLS, OUT_ROWS, WIDE_NORMS
from chipformer.q35.mute import MUTE
from chipformer.rv import chipfile as F

ROW_CHUNK = 1 << 24  # bytes in one chunk of a big tensor


def bf16_bytes(x: np.ndarray) -> bytes:
    """fp32 values that are bf16 values, as bf16 bytes."""
    u = np.asarray(x, np.float32).view(np.uint32)
    assert not (u & 0xFFFF).any()
    return (u >> 16).astype("<u2").tobytes()


class Expander:
    def __init__(self, chip, base) -> None:
        self.c = F.load(chip)
        self.f = self.c.file
        self.meta = self.f.meta
        self.base = Path(base)
        cfg = json.loads((self.base / "config.json").read_text())
        self.tc0 = cfg["text_config"]
        self.wm = json.loads((self.base / "model.safetensors.index.json").read_text())["weight_map"]
        self.d0 = self.tc0["hidden_size"]
        self.h = self.c.hidden
        self.nq = self.tc0["num_hidden_layers"]
        self.vocab = self.meta["model_vocab"]
        self.layer_types = self.tc0["layer_types"]
        self.shapes = {}  # dense name -> shape
        self.src = {}  # dense name -> base name
        self.dtypes = {}  # dense name -> safetensors dtype, where it is not BF16
        for e in self.meta["tensors"]:
            self.shapes[e[0]] = tuple(e[1])
        self.sparse = {e[0]: e[2] for e in self.meta["tensors"]}
        for k in self.wm:
            if not k.startswith(F.SRC_TEXT):
                continue
            k2 = "model." + k[len(F.SRC_TEXT):]
            k2 = re.sub(r"^(model\.layers\.)(\d+)\.",
                        lambda mt: f"{mt[1]}{int(mt[2]) + self.c.n_chip}.", k2)
            self.src[k2] = k
            with safe_open(self.base / self.wm[k], framework="pt") as fh:
                sl = fh.get_slice(k)
                s = list(sl.get_shape())
                self.dtypes[k2] = sl.get_dtype().upper()
            if k == F.SRC_TEXT + "embed_tokens.weight":
                s = [self.vocab, self.h]
            elif IN_COLS.search(k):
                s[-1] = self.h
            elif OUT_ROWS.search(k):
                s[0] = self.h
            elif WIDE_NORMS.search(k):
                s = [self.h]
            self.shapes[k2] = tuple(s)
        self._emb = None

    # --- the tensors ------------------------------------------------------------------------
    def _base(self, k: str) -> torch.Tensor:
        with safe_open(self.base / self.wm[k], framework="pt") as fh:
            return fh.get_tensor(k)

    def _qwen(self, name: str) -> torch.Tensor:
        """q35.merge's widening of one tensor, then q35.mute's patch."""
        k = self.src[name]
        w = self._base(k)
        scale = (self.d0 / self.h) ** 0.5
        extra = self.h - self.d0
        if IN_COLS.search(k):
            w = torch.cat([w, w.new_zeros(*w.shape[:-1], extra)], -1)
        elif OUT_ROWS.search(k):
            w = torch.cat([w, w.new_zeros(extra, *w.shape[1:])], 0)
        elif WIDE_NORMS.search(k):
            g = (1.0 + w.float()) * scale - 1.0
            w = torch.cat([g.to(w.dtype), w.new_zeros(extra)])
        mu = self.meta["mute"]["col"]
        mt = re.match(r"model\.layers\.(\d+)\.(.*)", name)
        if mt and self.meta["mute"]["muted"]:
            mw = MUTE / self.h**0.5
            i, sub = int(mt[1]), mt[2]
            if re.fullmatch(r"linear_attn\.in_proj_[abz]\.weight", sub):
                w[:, mu] = mw
            elif sub == "self_attn.q_proj.weight":
                hd = self.tc0["head_dim"]
                for h in range(w.shape[0] // (2 * hd)):
                    w[h * 2 * hd + hd: (h + 1) * 2 * hd, mu] = mw
            elif sub == "self_attn.k_proj.weight":
                pos = self.meta["mute"]["full_attention"].index(i)
                w[:, mu] = self.f.t["mute.sinks"][pos]
        return w

    def _norm(self, name: str) -> torch.Tensor:
        norm = self._qwen(name).float()
        norm[self.d0:] = -1.0
        cols = self.f.arr("norm.cols").astype(np.int64)
        norm[torch.from_numpy(cols)] = torch.from_numpy(self.f.arr("norm.values"))
        return norm.to(torch.bfloat16)

    def _sparse_chunks(self, name: str):
        shape = self.shapes[name]
        fill = self.sparse[name]
        r, c, v = self.f.terms(name)
        n = shape[0]
        width = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        step = max(1, ROW_CHUNK // (2 * width))
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            out = np.full((r1 - r0, width), fill, np.float32)
            lo, hi = np.searchsorted(r, [r0, r1])
            out[r[lo:hi] - r0, c[lo:hi]] = v[lo:hi]
            yield bf16_bytes(out.ravel())

    def _embedding_chunks(self):
        k = F.SRC_TEXT + "embed_tokens.weight"
        d0, h, v0 = self.d0, self.h, self.c.v0
        c, f = self.c, self.f
        plain = np.zeros(h - d0, np.float32)
        plain[: c.m] = 2.0 ** self.meta["layout"]["log2_ballast"]
        pc = f.arr("text.plain.cols").astype(np.int64) - d0
        plain[pc] = f.arr("text.plain.values")
        bc = f.arr("text.byte_cols").astype(np.int64) - d0
        by = f.arr("text.bytes")
        ids = f.arr("text.special.ids").tolist()
        off, cols, vals = c._chip_rows()
        step = 1 << 14
        chip_ballast = plain[: c.m]
        with safe_open(self.base / self.wm[k], framework="pt") as fh:
            sl = fh.get_slice(k)
            for r0 in range(0, self.vocab, step):
                r1 = min(self.vocab, r0 + step)
                n = r1 - r0
                head = np.zeros((n, d0), np.uint16)
                tail = np.zeros((n, h - d0), np.float32)
                t1 = min(r1, v0)
                if r0 < v0:
                    b = sl[r0:t1].to(torch.float32).numpy()
                    u = b.view(np.uint32)
                    head[: t1 - r0] = (u >> 16).astype(np.uint16)
                    tail[: t1 - r0] = plain
                    lim = min(t1, len(by))
                    if r0 < lim:
                        bits = (by[r0:lim, :, None] >> np.arange(8, dtype=np.uint8)) & 1
                        rr, kk, ii = np.nonzero(bits)
                        tail[rr, bc[kk, ii]] = 1.0
                    for q in ids:
                        if r0 <= q < t1:
                            cc, vv = c._text_row(q)
                            tail[q - r0] = 0.0
                            tail[q - r0, : c.m] = chip_ballast
                            tail[q - r0, cc - d0] = vv
                for t in range(max(r0, v0), min(r1, v0 + c.vocab)):
                    i = t - v0
                    tail[t - r0, : c.m] = chip_ballast
                    tail[t - r0, cols[off[i]:off[i + 1]] - d0] = vals[off[i]:off[i + 1]]
                ub = tail.view(np.uint32)
                assert not (ub & 0xFFFF).any()
                row = np.concatenate([head, (ub >> 16).astype(np.uint16)], 1)
                yield row.astype("<u2").tobytes()

    def chunks(self, name: str):
        if name in self.sparse:
            yield from self._sparse_chunks(name)
        elif name == "model.embed_tokens.weight":
            yield from self._embedding_chunks()
        elif name == "model.norm.weight":
            yield _bytes(self._norm(name))
        else:
            yield _bytes(self._qwen(name))

    # --- the shards -------------------------------------------------------------------------
    def shards(self) -> list:
        """merge.build_merged's split: sorted names, a new shard past 2 GiB."""
        shard, size, out = [], 0, []
        for k in sorted(self.shapes):
            shard.append(k)
            size += int(np.prod(self.shapes[k])) * 2
            if size > 2**31:
                out.append(shard)
                shard, size = [], 0
        out.append(shard)
        return out

    def _width(self, k: str) -> int:
        return 4 if self.dtypes.get(k, "BF16") == "F32" else 2

    def order(self, names: list) -> list:
        """save_file's order: by dtype, wider first (Qwen's few F32 tensors), then by name."""
        return sorted(names, key=lambda k: (-self._width(k), k))

    def header(self, names: list) -> bytes:
        """save_file's header: metadata first, then the tensors, spaces up to a multiple of 8."""
        h = {"__metadata__": {"format": "pt"}}
        a = 0
        for k in self.order(names):
            b = a + int(np.prod(self.shapes[k])) * self._width(k)
            h[k] = {"dtype": self.dtypes.get(k, "BF16"), "shape": list(self.shapes[k]),
                    "data_offsets": [a, b]}
            a = b
        js = json.dumps(h, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 8)
        return len(js).to_bytes(8, "little") + js

    def stream(self, names: list):
        yield self.header(names)
        for k in self.order(names):
            yield from self.chunks(k)


def _bytes(t: torch.Tensor) -> bytes:
    if t.dtype == torch.float32:
        return t.contiguous().numpy().astype("<f4").tobytes()
    return bf16_bytes(t.contiguous().view(-1).to(torch.float32).numpy())


def shard_name(i: int, n: int) -> str:
    return f"model-{i + 1:05d}-of-{n:05d}.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chip")
    ap.add_argument("base")
    ap.add_argument("--out")
    ap.add_argument("--hub", help="a Hub repo: compare each shard's SHA-256 with its LFS hash")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--against", help="a directory with the shards: compare bytes")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    e = Expander(a.chip, a.base)
    sh = e.shards()
    want = {}
    if a.hub:
        from huggingface_hub import HfApi

        info = HfApi().model_info(a.hub, revision=a.rev, files_metadata=True)
        want = {s.rfilename: s.lfs.sha256 for s in info.siblings
                if s.rfilename.endswith(".safetensors")}
        assert set(want) == {shard_name(i, len(sh)) for i in range(len(sh))}, sorted(want)
    res = {"tensors": len(e.shapes), "shards": [], "bytes": 0, "same": True}
    t0 = time.time()
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, names in enumerate(sh):
        fn = shard_name(i, len(sh))
        h = hashlib.sha256()
        n = 0
        first = None
        out = open(Path(a.out) / fn, "wb") if a.out else None
        cmp = open(Path(a.against) / fn, "rb") if a.against else None
        for b in e.stream(names):
            h.update(b)
            if out:
                out.write(b)
            if cmp and first is None and cmp.read(len(b)) != b:
                first = n
            n += len(b)
        d = h.hexdigest()
        ok = (d == want[fn]) if a.hub else first is None
        r = {"file": fn, "bytes": n, "sha256": d, "same": bool(ok), "first_diff_chunk_at": first,
             "s": round(time.time() - t0)}
        print(json.dumps(r), flush=True)
        res["shards"].append(r)
        res["bytes"] += n
        res["same"] = res["same"] and bool(ok)
        if out:
            out.close()
    print(json.dumps({k: v for k, v in res.items() if k != "shards"}), flush=True)
    if a.report:
        json.dump(res, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
