"""Latent CPU control, Phase 0e (CPU only): order and op for the direct-answer (D) command, leave one source out.

python -m chipformer.rv.latentorder PACKS LABEL_DIRS OUT.json
PACKS: anspack pack.pt files, comma-separated. LABEL_DIRS: Phase 0c argpos dirs (meta.jsonl) for the
question numbers' values (the D candidates are the question's numbers, in order).
"""
import json
import math
import re
import sys
from pathlib import Path

import torch

OPS = ["+", "-", "*", "/"]
torch.manual_seed(0)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load(packs, labdirs):
    vals, signed = {}, {}
    for d in labdirs.split(","):
        for x in (Path(d) / "meta.jsonl").read_text().splitlines():
            m = json.loads(x)
            vals.setdefault(m["id"], [num(r[2]) for r in m["runs"] if r[0] < m["n_prompt"]])
            signed.setdefault(m["id"], any(re.search(r"(^|[+\-*/(]\s*)-\s*\d", c.get("expr", ""))
                                           for c in m["calls"]))
    items, bad = [], 0
    for f in packs.split(","):
        for it in torch.load(f, weights_only=False)["items"]:
            if it["cond"] != "D":
                continue
            v = vals.get(it["key"][2:])
            if v is None or len(v) != len(it["start"]):
                bad += 1
                continue
            it["v"] = v
            it["signed"] = signed[it["key"][2:]]
            it["M"] = it["M"].float()
            it["h"] = it["h"].float()
            items.append(it)
    return items, bad


def pair_ok(p0, p1, it, ordered=True):
    if set(it["tg"][0]) == set(it["tg"][1]):
        return p0 in it["tg"][0] and (p1 in it["tg"][0] or p1 == p0)
    g = p0 in it["tg"][0] and p1 in it["tg"][1]
    if it["op"] in "+*" or not ordered:
        g = g or (p0 in it["tg"][1] and p1 in it["tg"][0])
    return g


def logistic(X, y, n_cls, wd=1e-3, epochs=300, bias=True):
    W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=bias)
    opt = torch.optim.Adam([W, b] if bias else [W], lr=1e-2, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(X @ W + b, y).backward()
        opt.step()
    return lambda Z: (Z @ W + b).detach()


# ---- pointer pair: one learned weight per head, mass on both operands ----
def pad(items):
    LH = items[0]["M"].shape[0]
    C = max(it["M"].shape[1] for it in items)
    X = torch.zeros(len(items), LH, C)
    mask = torch.zeros(len(items), C, dtype=torch.bool)
    tg = [torch.zeros(len(items), C, dtype=torch.bool) for _ in range(2)]
    for i, it in enumerate(items):
        c = it["M"].shape[1]
        X[i, :, :c] = it["M"]
        mask[i, :c] = True
        for s in range(2):
            tg[s][i, it["tg"][s]] = True
    return X, mask, tg


def fit_pair(tr, n_iter=400):
    X, mask, tg = pad(tr)
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(n_iter):
        sc = (torch.einsum("h,nhc->nc", w, X) * 10).masked_fill(~mask, -1e9)
        lz = torch.logsumexp(sc, 1)
        loss = sum((lz - torch.logsumexp(sc.masked_fill(~t, -1e9), 1)).mean() for t in tg) + 1e-3 * w.abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = w.detach()

    def pick(it, conf=False):
        sc = w @ it["M"] * 10
        if sc.numel() < 2:
            return (0, 0, 1.0) if conf else (0, 0)
        a, b = sc.topk(2).indices.tolist()
        pr = sc.softmax(0)
        return (a, b, float(pr[a] + pr[b])) if conf else (a, b)
    return pick


# ---- order: antisymmetric features of an ordered pair (p first) ----
def ofeat(it, p, q):
    lm = torch.log(it["M"] + 1e-3)
    vp, vq = it["v"][p], it["v"][q]
    t = math.copysign(1, it["start"][q] - it["start"][p])  # +1: p earlier in the text
    vs = 0.0 if vp == vq else math.copysign(1, vp - vq)
    lr = max(-3, min(3, math.log10(abs(vp) + 1) - math.log10(abs(vq) + 1)))
    dv = float(vq != 0 and vp % vq == 0 and vp != vq) - float(vp != 0 and vq % vp == 0 and vp != vq)
    return torch.cat([lm[:, p] - lm[:, q], torch.tensor([t, vs, lr, dv])])


def ofeat_op(it, p, q, op):
    f = ofeat(it, p, q)
    z = torch.zeros_like(f)
    return torch.cat([f, z] if op == "-" else [z, f])


def fit_order(tr, kind):
    """kind: 'attn' (head mass diffs), 'cues' (text order, value, divisibility), 'all'."""
    rows, sel = [], {"attn": slice(0, 128), "cues": slice(128, 132), "all": slice(0, 132)}[kind]
    for it in tr:
        if it["op"] not in "-/" or set(it["tg"][0]) == set(it["tg"][1]):
            continue
        a, b = it["tg"][0][0], it["tg"][1][0]
        f = ofeat(it, a, b)[sel]
        rows.append(torch.cat([f, torch.zeros_like(f)] if it["op"] == "-" else [torch.zeros_like(f), f]))
    X = torch.stack(rows)
    sd = X.std(0) + 1e-3
    X = X / sd
    Xs = torch.cat([X, -X])
    y = torch.cat([torch.ones(len(X)), torch.zeros(len(X))]).long()
    f = logistic(Xs, y, 2, wd=1e-2, bias=False)

    def order(it, p, q, op, conf=False):
        if op in "+*":
            return (p, q, 1.0) if conf else (p, q)
        g = ofeat(it, p, q)[sel]
        z = torch.zeros_like(g)
        x = torch.cat([g, z] if op == "-" else [z, g]) / sd
        s = f(x[None])[0]
        pp = float(s.softmax(0).max())
        res = (p, q) if s[1] >= s[0] else (q, p)
        return (*res, pp) if conf else res
    return order


# ---- op ----
def opfeat_vals(it, p, q):
    vp, vq = it["v"][p], it["v"][q]
    hi, lo = max(abs(vp), abs(vq)), min(abs(vp), abs(vq))
    div = float(lo != 0 and hi % lo == 0)
    return torch.tensor([div, float(vp == int(vp) and vq == int(vq)), math.log10(hi + 1), math.log10(lo + 1),
                         math.log10(hi + 1) - math.log10(lo + 1), float(lo <= 12), float(hi >= 1000)])


def opfeat(it, p, q, kind, li, center):
    parts = []
    if "H" in kind:
        parts += [it["h"][i] for i in (2, 3, 4)]
    if "h" in kind:
        parts.append(it["h"][li] - center[it["src"]])
    if "a" in kind:
        parts.append(torch.log(it["M"][:, p] + it["M"][:, q] + 1e-3))
    if "v" in kind:
        parts.append(opfeat_vals(it, p, q))
    return torch.cat(parts)


def fit_op(tr, kind, li, center, pick, wd=1e-3):
    X = torch.stack([opfeat(it, it["tg"][0][0], it["tg"][1][0], kind, li, center) for it in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-4
    y = torch.tensor([OPS.index(it["op"]) for it in tr])
    f = logistic((X - mu) / sd, y, 4, wd=wd)

    def op(it, p, q, conf=False):
        z = f(((opfeat(it, p, q, kind, li, center) - mu) / sd)[None])[0]
        o = OPS[int(z.argmax())]
        return (o, float(z.softmax(0).max())) if conf else o
    return op


def main():
    packs, labdirs, out = sys.argv[1:4]
    items, bad = load(packs, labdirs)
    srcs = sorted({it["src"] for it in items})
    print("D items", len(items), "unmatched", bad, {s: sum(it["src"] == s for it in items) for s in srcs}, flush=True)
    LI = 3  # layer 20 of 8,12,16,20,24 (fixed, not chosen on the test source)
    center0 = {s: torch.zeros(items[0]["h"].shape[1]) for s in srcs}
    centerS = {s: torch.stack([it["h"][LI] for it in items if it["src"] == s]).mean(0) for s in srcs}
    rep = {}
    for held in srcs:
        tr, te = [it for it in items if it["src"] != held], [it for it in items if it["src"] == held]
        if len(te) < 10:
            continue
        pick = fit_pair(tr)
        pairs = [pick(it) for it in te]
        r = {"n": len(te), "pair": sum(pair_ok(*p, it, False) for p, it in zip(pairs, te)) / len(te)}
        # order, given the gold pair and gold op (- and / only)
        od = [it for it in te if it["op"] in "-/" and set(it["tg"][0]) != set(it["tg"][1])]
        orders = {k: fit_order(tr, k) for k in ("attn", "cues", "all")}
        orders["text"] = lambda it, p, q, op: (p, q) if op in "+*" or it["start"][p] <= it["start"][q] else (q, p)
        r["order_n"] = len(od)
        for k, o in orders.items():
            r[f"order_{k}"] = sum(pair_ok(*o(it, it["tg"][0][0], it["tg"][1][0], it["op"]), it) for it in od) / max(1, len(od))
        # op, at the gold pair
        ops = {}
        for kind, wd in (("h", 1e-3), ("h", 1e-1), ("ha", 1e-3), ("hv", 1e-3), ("hav", 1e-3), ("H", 1e-3),
                         ("H", 1e-1), ("a", 1e-3)):
            name = f"{kind}_wd{wd:g}"
            ops[name] = fit_op(tr, kind, LI, center0, pick, wd)
            r[f"op_{name}"] = sum(ops[name](it, it["tg"][0][0], it["tg"][1][0]) == it["op"] for it in te) / len(te)
        # the full command, everything predicted
        for name, ok_, oo in (("base", "h_wd0.001", "text"), ("attn_order", "h_wd0.001", "attn"),
                              ("attn_order_wd", "h_wd0.1", "attn"), ("attn_order_H", "H_wd0.1", "attn")):
            good = 0
            for (p, q), it in zip(pairs, te):
                op = ops[ok_](it, p, q)
                a, b = orders[oo](it, p, q, op)
                good += op == it["op"] and pair_ok(a, b, it)
            r[f"command_{name}"] = good / len(te)
            if name == "attn_order_wd":
                us = [(pq, it) for pq, it in zip(pairs, te) if not it["signed"]]
                g2 = 0
                for (p, q), it in us:
                    op = ops[ok_](it, p, q)
                    a, b = orders[oo](it, p, q, op)
                    g2 += op == it["op"] and pair_ok(a, b, it)
                r["unsigned_n"] = len(us)
                r["command_attn_order_wd_unsigned"] = g2 / max(1, len(us))
                r["qwen_own_unsigned"] = sum(it["qwen_right"] for _, it in us) / max(1, len(us))
        r["qwen_own"] = sum(it["qwen_right"] for it in te) / len(te)

        # hybrid: the CPU's answer when the command is confident, else Qwen's own; threshold set on tr
        def run(its, prs):
            out = []
            for it in its:
                a0, b0, cp = pick(it, True)
                op, co = ops["h_wd0.1"](it, a0, b0, True)
                a, b, cr = orders["attn"](it, a0, b0, op, True)
                out.append((cp * co * cr, op == it["op"] and pair_ok(a, b, it), it["qwen_right"]))
            return out
        rt = run(tr, [None] * len(tr))
        best = max(sorted({c for c, _, _ in rt}) + [2.0],
                   key=lambda t: sum(g if c >= t else q for c, g, q in rt))
        rs = run(te, [None] * len(te))
        r["hybrid_tau"] = best
        r["hybrid"] = sum(g if c >= best else q for c, g, q in rs) / len(te)
        r["hybrid_cpu_share"] = sum(c >= best for c, _, _ in rs) / len(te)
        r["oracle_either"] = sum(g or q for _, g, q in rs) / len(te)
        rep[held] = r
        print(held, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
