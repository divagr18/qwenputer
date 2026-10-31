"""stories260K for the VX build: int4 weights in the WROM as VX row-chunks (chipformer/rv/vx.py).

python export_vx.py [qat_int4.npz]   ->  weights_vx.h, wrom_vx.c
RMSNorm gains are folded into the matrices that read them (the classifier gets the final gain).
Every matrix row is k = ceil(n / 64) row-chunks of 64 cells, each with the row scale S' = s * 2^15,
laid out by VMAT op (OPB: each op's row-pair base).
Without the npz: int4 per-row MSE-clip codes (for token counts; the QAT codes for quality).
WB_* give each matrix's first op in OPB.
"""
import os
import sys

import numpy as np

sys.argv, npz = sys.argv[:1], (sys.argv[1] if len(sys.argv) > 1 else None)
import export as E  # noqa: E402

E.MAXPOS = 512  # the RoPE table (positions past the window keep their true angles)

HERE = os.path.dirname(os.path.abspath(__file__))
nl, dim, hid, vocab = E.nl, E.dim, E.hid, E.vocab
MATS = ["cls", "wq", "wk", "wv", "wo", "w1", "w3", "w2"]


def mse_codes(w):
    best = None
    for c in np.linspace(0.5, 1.0, 26):
        s = np.abs(w).max(-1, keepdims=True) * c / 7
        s[s == 0] = 1e-12
        q = np.clip(np.round(w / s), -8, 7)
        e = ((q * s - w) ** 2).sum(-1, keepdims=True)
        best = (q, s, e) if best is None else (np.where(e < best[2], q, best[0]),
                                               np.where(e < best[2], s, best[1]), np.minimum(e, best[2]))
    return best[0].astype(np.int8), best[1][..., 0]


if npz:
    z = np.load(npz)
    Q = {k: z[f"q_{k}"] for k in MATS}
    SC = {k: z[f"s_{k}"] for k in MATS}
else:
    ga, gf = E.rms_att[:, None, :], E.rms_ffn[:, None, :]
    src = {"cls": E.emb * E.rms_fin, "wq": E.wq * ga, "wk": E.wk * ga, "wv": E.wv * ga, "wo": E.wo,
           "w1": E.w1 * gf, "w2": E.w2, "w3": E.w3 * gf}
    Q, SC = {}, {}
    for k, w in src.items():
        if w.ndim == 3:
            qs = [mse_codes(w[l]) for l in range(nl)]
            Q[k], SC[k] = np.stack([a for a, _ in qs]), np.stack([b for _, b in qs])
        else:
            Q[k], SC[k] = mse_codes(w)

cells, bases, h = bytearray(), {}, []
CHUNK = 72  # VX row-chunks: 64 int4 cells, the row scale in 4 nibble cells, 4 unused


chunks = {}  # address -> 72 cells
opb = []  # every VMAT op's row-pair base, in program order (OPB in the header)
top_chunk = [0]  # the next free row-chunk


def add_matrix(name, q, scale, op_rows):
    """The matrix's rows as VMAT ops of op_rows rows. The WROM holds 3 row-chunks a row pair:
    a k = 1 op's row r is chunk 3 base + start + r (ops packed back to back), a k = 3 op's row r
    is pair base + r with chunk c in slot c. An op's pairs stay in one block of 64. OPB gets
    each op's base | start << 19 (rs1's fields); returns the index of the matrix's first op."""
    rows, n = q.shape
    k = -(-n // 64)
    assert k in (1, 3), name
    s = np.round(np.asarray(scale, np.float64) * 2**15).astype(np.int64)
    assert s.max() < 2**16, (name, s.max())
    first = len(opb)
    for r0 in range(0, rows, op_rows):
        nr = min(op_rows, rows - r0)
        a0 = top_chunk[0] if k == 1 else -(-top_chunk[0] // 3) * 3
        last = a0 + (nr - 1 if k == 1 else 3 * nr - 1)
        if a0 // 192 != last // 192:  # past the block of 64 row pairs: start the next one
            a0 = (a0 // 192 + 1) * 192
            last = a0 + (nr - 1 if k == 1 else 3 * nr - 1)
        top_chunk[0] = last + 1
        base, st = divmod(a0, 3)
        opb.append(base | st << 19)
        for r in range(nr):
            row = np.zeros(64 * k, np.int64)
            row[:n] = q[r0 + r]
            for c in range(k):
                a = 3 * (base + r) + c if k == 3 else a0 + r
                assert a not in chunks, (name, a)
                chunks[a] = ((row[64 * c:64 * c + 64] + 8).astype(np.uint8).tobytes()
                             + bytes((int(s[r0 + r]) >> 4 * i & 15) for i in range(4)) + bytes(4))
    return first


wb = {"cls": add_matrix("cls", Q["cls"], SC["cls"], 16)}
for l in range(nl):
    for k in ("wq", "wk", "wv", "wo", "w1", "w3"):
        wb[k, l] = add_matrix(k, Q[k][l], SC[k][l], 16)
    wb["w2", l] = add_matrix("w2", Q["w2"][l], SC["w2"][l], 8)
top = max(chunks) + 1
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from chipformer.rv import spec as S  # noqa: E402
assert top <= S.VX_CHUNKS, (top, S.VX_CHUNKS)
for a in range(top):
    cells.extend(chunks.get(a, bytes(CHUNK)))


def carr(name, ctype, a):
    a = np.asarray(a).ravel()
    return f"static const {ctype} {name}[{len(a)}] = {{{','.join(str(int(v)) for v in a)}}};\n"


h.append(f"#define DIM {dim}\n#define HID {hid}\n#define HIDW {-(-hid // 8) * 8}\n#define NL {nl}\n"
         f"#define NH {E.nh}\n#define NKV {E.nkv}\n#define HS {E.hs}\n#define KVD {E.kvd}\n"
         f"#define VOCAB {vocab}\n#define MAXPOS {E.MAXPOS}\n")
h.append(f"#define WB_CLS {wb['cls']}\n")
h.append(carr("OPB", "uint32_t", opb))
for k in MATS[1:]:
    h.append(carr(f"WB_{k}", "int32_t", [wb[k, l] for l in range(nl)]))
# the embedding (lookups only): int8 rows with a row scale * 2^32
q8, M8 = E.qrows(E.emb)
h.append(carr("Q_emb", "int8_t", q8))
h.append(carr("S_emb", "int32_t", M8 * 127))
cs = []
for pos in range(E.MAXPOS):
    for i in range(0, E.hs, 2):
        fr = 1.0 / 10000 ** (i / E.hs)
        cs += [round(np.cos(pos * fr) * 16384), round(np.sin(pos * fr) * 16384)]
h.append(carr("ROPE", "int16_t", cs))
h.append(carr("EXP2", "uint16_t", [round(2 ** (-i / 256) * 32768) for i in range(256)]))
# int8 K / V caches with per-layer scales: max |k| (after RoPE) and |v| over float samples, +25%
kmax, vmax_ = np.zeros(nl), np.zeros(nl)
for seq_seed in range(4):
    rng = np.random.default_rng(seq_seed)
    kc = [[None] * 64 for _ in range(nl)]
    vc = [[None] * 64 for _ in range(nl)]
    tok = 1
    for pos in range(64):
        lg = E.forward_float(tok, pos, kc, vc)
        p = np.exp((lg - lg.max()) / 0.9)
        tok = int(rng.choice(len(p), p=p / p.sum()))
    for l in range(nl):
        kmax[l] = max(kmax[l], max(np.abs(k).max() for k in kc[l]))
        vmax_[l] = max(vmax_[l], max(np.abs(v).max() for v in vc[l]))
h.append(carr("KMAX", "int32_t", np.round(kmax * 1.25 * 65536)))
h.append(carr("VMAX", "int32_t", np.round(vmax_ * 1.25 * 65536)))
offs, blob = [], b""
for pc in E.pieces:
    offs.append(len(blob))
    blob += pc
offs.append(len(blob))
h.append(carr("VOFF", "uint16_t", offs))
h.append(carr("VSTR", "uint8_t", list(blob)))
open(f"{HERE}/weights_vx.h", "w").write("#include <stdint.h>\n" + "".join(h))
open(f"{HERE}/wrom_vx.c", "w").write(
    "#include <stdint.h>\n__attribute__((section(\".wrom\"), used)) const uint8_t WROM_DATA["
    f"{len(cells)}] = {{{','.join(map(str, cells))}}};\n")
print(f"{len(chunks)} WROM row-chunks, {len(opb)} ops, {-(-top // 3)} row pairs "
      f"({-(-top // 384)} key heads), top address {top} of {S.VX_CHUNKS}")
