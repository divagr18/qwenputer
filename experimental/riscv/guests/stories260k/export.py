"""stories260K -> int8 C tables (weights.h) + a float numpy reference (greedy tokens).

python export.py [ntok]
"""
import struct
import sys

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
MAXPOS = 64

raw = open(f"{HERE}/stories260K.bin", "rb").read()
dim, hid, nl, nh, nkv, vocab, seqlen = struct.unpack("7i", raw[:28])
hs = dim // nh
kvd = nkv * hs
f = np.frombuffer(raw[28:], dtype=np.float32)
o = 0


def take(*shape):
    global o
    n = int(np.prod(shape))
    a = f[o:o + n].reshape(shape)
    o += n
    return a


emb = take(vocab, dim)
rms_att = take(nl, dim)
wq = take(nl, dim, dim)
wk = take(nl, kvd, dim)
wv = take(nl, kvd, dim)
wo = take(nl, dim, dim)
rms_ffn = take(nl, dim)
w1 = take(nl, hid, dim)
w2 = take(nl, dim, hid)
w3 = take(nl, hid, dim)
rms_fin = take(dim)

# tokenizer
tb = open(f"{HERE}/tok512.bin", "rb").read()
p = 4
pieces = []
for _ in range(vocab):
    _score, ln = struct.unpack("fi", tb[p:p + 8])
    p += 8
    pieces.append(tb[p:p + ln])
    p += ln


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x) + 1e-5) * w


def rope(v, pos, nheads):
    v = v.copy()
    for i in range(0, nheads * hs, 2):
        fr = 1.0 / 10000 ** ((i % hs) / hs)
        c, s = np.cos(pos * fr), np.sin(pos * fr)
        a, b = v[i], v[i + 1]
        v[i], v[i + 1] = a * c - b * s, a * s + b * c
    return v


def forward_float(tok, pos, kc, vc):
    x = emb[tok].astype(np.float64)
    for l in range(nl):
        h = rmsnorm(x, rms_att[l])
        q, k, v = wq[l] @ h, wk[l] @ h, wv[l] @ h
        q, k = rope(q, pos, nh), rope(k, pos, nkv)
        kc[l][pos], vc[l][pos] = k, v
        out = np.zeros(dim)
        for hh in range(nh):
            g = hh // (nh // nkv)
            qs = q[hh * hs:(hh + 1) * hs]
            sc = np.array([qs @ kc[l][t][g * hs:(g + 1) * hs] for t in range(pos + 1)]) / np.sqrt(hs)
            sc = np.exp(sc - sc.max())
            sc /= sc.sum()
            out[hh * hs:(hh + 1) * hs] = sum(sc[t] * vc[l][t][g * hs:(g + 1) * hs] for t in range(pos + 1))
        x = x + wo[l] @ out
        h = rmsnorm(x, rms_ffn[l])
        a, b = w1[l] @ h, w3[l] @ h
        x = x + w2[l] @ (a / (1 + np.exp(-a)) * b)
    x = rmsnorm(x, rms_fin)
    return emb @ x


def generate(ntok):
    kc = [[None] * MAXPOS for _ in range(nl)]
    vc = [[None] * MAXPOS for _ in range(nl)]
    tok, out = 1, []
    for pos in range(ntok):
        tok = int(np.argmax(forward_float(tok, pos, kc, vc)))
        out.append(tok)
    return out


def detok(toks, prev=1):
    s = b""
    for t in toks:
        pc = pieces[t]
        if prev == 1 and pc.startswith(b" "):
            pc = pc[1:]
        s += pc
        prev = t
    return s.decode(errors="replace")


# ---- C export: int8 rows, per-row scale M (out_q16 = (acc * amax_q16 * M) >> 32)
def qrows(w):
    w = w.reshape(-1, w.shape[-1])
    s = np.abs(w).max(1) / 127.0
    s[s == 0] = 1e-12
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
    M = np.round(s / 127.0 * 2**32).astype(np.int64)
    assert M.max() < 2**31
    return q, M


def carr(name, ctype, a, per=24):
    a = np.asarray(a).ravel()
    body = ",".join(str(int(v)) for v in a)
    return f"static const {ctype} {name}[{len(a)}] = {{{body}}};\n"


def write_header(path):
    h = [f"#define DIM {dim}\n#define HID {hid}\n#define NL {nl}\n#define NH {nh}\n"
         f"#define NKV {nkv}\n#define HS {hs}\n#define KVD {kvd}\n#define VOCAB {vocab}\n"
         f"#define MAXPOS {MAXPOS}\n"]
    for name, w in [("emb", emb), ("wq", wq), ("wk", wk), ("wv", wv), ("wo", wo),
                    ("w1", w1), ("w2", w2), ("w3", w3)]:
        q, M = qrows(w)
        h.append(carr(f"Q_{name}", "int8_t", q))
        h.append(carr(f"M_{name}", "int32_t", M))
        h.append(carr(f"S_{name}", "int32_t", M * 127))  # row scale * 2^32
    # w1 / w3 padded with zero rows to a multiple of 8 (HIDP) for the 8-row kernels
    hidp = -(-hid // 8) * 8
    h.append(f"#define HIDP {hidp}\n")
    for name, w in [("w1", w1), ("w3", w3)]:
        q, M = qrows(w)
        q = np.concatenate([q.reshape(nl, hid, dim), np.zeros((nl, hidp - hid, dim), np.int8)], 1)
        M = np.concatenate([M.reshape(nl, hid), np.zeros((nl, hidp - hid), np.int64)], 1)
        h.append(carr(f"Q_{name}p", "int8_t", q))
        h.append(carr(f"S_{name}p", "int32_t", M * 127))
    for name, w in [("rms_att", rms_att), ("rms_ffn", rms_ffn), ("rms_fin", rms_fin)]:
        h.append(carr(f"R_{name}", "int32_t", np.round(w * 65536)))
    cs = []
    for pos in range(MAXPOS):
        for i in range(0, hs, 2):
            fr = 1.0 / 10000 ** (i / hs)
            cs += [round(np.cos(pos * fr) * 16384), round(np.sin(pos * fr) * 16384)]
    h.append(carr("ROPE", "int16_t", cs))
    h.append(carr("EXP2", "uint16_t", [round(2 ** (-i / 256) * 32768) for i in range(256)]))
    offs, blob = [], b""
    for pc in pieces:
        offs.append(len(blob))
        blob += pc
    offs.append(len(blob))
    h.append(carr("VOFF", "uint16_t", offs))
    h.append(carr("VSTR", "uint8_t", list(blob)))
    open(path, "w").write("#include <stdint.h>\n" + "".join(h))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    write_header(f"{HERE}/weights.h")
    toks = generate(n)
    print("float tokens:", toks)
    print("float text:", repr(detok(toks)))
