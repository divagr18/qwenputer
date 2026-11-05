"""Phase 0 of latent CPU control: can a calc call be read from Qwen's hidden states before any text
is written?

    python -m chipformer.rv.latentprobe collect OUT [--n-synth 600] [--budget 768]
    python -m chipformer.rv.latentprobe fit OUT

A call is a command: call now, the op, and pointers to where its operands are in the context (the
CPU would copy the operands themselves, exactly). ``collect`` runs base Qwen3.5-4B in two
conditions and saves hidden states at a few layers:

  T  the calc tool and the evals' nudge; Qwen writes its calls, the host answers them in the
     firmware's format (calc.expected). Labels come from Qwen's own expressions: the op, and
     which numbers in the context are its operands. States: the position before each
     <tool_call>, every Qwen-written position (call now or not), every number in the context.
  N  no tools, "do every step exactly": synthetic questions only, labels from the generator.
     States: the last prompt position and every number in the context.

Synthetic questions put distractor numbers around the operands and phrase some ops with the roles
reversed ("subtract a from b"), so a pointer has to read the meaning, not the order.

``fit`` trains, per layer, logistic probes (call now; the op) and low-rank bilinear pointer
probes (each operand slot scores every number in the context), on 80% of the problems, and
reports the other 20% against order heuristics (first numbers of the question, last numbers
before the call). It also reports synthetic -> benchmark transfer.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch

LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
OPS = ["+", "-", "*", "/"]
NUM = re.compile(r"^[0-9]$")


# ---- synthetic questions: (question, op, a, b) with op(a, b) the answer ----
TEMPL = {
    "+": [("What is {a} plus {b}?", 0), ("Add {a} and {b}.", 0), ("What is the sum of {a} and {b}?", 0),
          ("Add {b} to {a}.", 1), ("Increase {a} by {b}. What do you get?", 0)],
    "-": [("What is {a} minus {b}?", 0), ("Subtract {b} from {a}.", 1), ("Take {b} away from {a}.", 1),
          ("What is {a} less {b}?", 0), ("How much is left when {b} is subtracted from {a}?", 1)],
    "*": [("What is {a} times {b}?", 0), ("Multiply {a} by {b}.", 0), ("What is the product of {a} and {b}?", 0),
          ("Calculate {a} multiplied by {b}.", 0), ("What do you get if you multiply {b} by {a}?", 1)],
    "/": [("What is {a} divided by {b}?", 0), ("Divide {a} by {b}.", 0), ("How many times does {b} go into {a}?", 1),
          ("What is the quotient when {a} is divided by {b}?", 0), ("Calculate {a} / {b}.", 0)],
}
DISTRACT = ["Ticket {x} was handled by agent {y}.", "In {yr}, the shop sold {x} items.",
            "Order #{x} is still pending.", "The building has {s} floors and {x} windows.",
            "Room {s} seats {y} people.", "My locker code is {x}."]


def synth(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        op = OPS[i % 4]
        d1, d2 = rng.randint(1, 9), rng.randint(1, 7)
        a, b = rng.randint(10 ** (d1 - 1), 10 ** d1 - 1), rng.randint(max(2, 10 ** (d2 - 1)), 10 ** d2 - 1)
        if op == "/":
            a = a * b if rng.random() < 0.5 else a
        if op == "-" and rng.random() < 0.3:
            a, b = b, a  # negative answers too
        t, _ = rng.choice(TEMPL[op])  # {a}, {b} are the roles; the flag marks text order reversed
        q = t.format(a=a, b=b)
        dist = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            dist.append(rng.choice(DISTRACT).format(x=rng.randint(10, 10 ** rng.randint(2, 8)),
                                                    y=rng.randint(2, 999), yr=rng.randint(1950, 2025),
                                                    s=rng.randint(2, 90)))
        k = rng.randint(0, len(dist))
        text = " ".join(dist[:k] + [q] + dist[k:])
        out.append({"id": f"synth-{seed}-{i}", "src": "synth", "question": text, "op": op,
                    "a": str(a), "b": str(b)})
    return out


# ---- numbers in a token sequence ----
def number_runs(toks: list[str]) -> list[tuple[int, int, str]]:
    """Maximal digit runs (commas and one decimal point inside allowed): (start, end, digits)."""
    runs, i, n = [], 0, len(toks)
    while i < n:
        if NUM.match(toks[i]):
            j, s = i, ""
            while j < n and (NUM.match(toks[j]) or (toks[j] in (",", ".") and j + 1 < n
                                                    and NUM.match(toks[j + 1]) and j > i)):
                s += toks[j]
                j += 1
            runs.append((i, j - 1, s.replace(",", "")))
            i = j
        else:
            i += 1
    return runs


EXPR_NUM = re.compile(r"\d+(?:\.\d+)?")
BIN = re.compile(r"^\s*\(?\s*(-?\s*\d+(?:\.\d+)?)\s*\)?\s*([+\-*/])\s*\(?\s*(-?\s*\d+(?:\.\d+)?)\s*\)?\s*$")


def parse_call(expr: str) -> dict:
    m = BIN.match(expr.replace("\n", ""))
    nums = EXPR_NUM.findall(expr)
    if m:
        return {"op": m[2], "operands": [re.sub(r"[\s-]", "", m[1]), re.sub(r"[\s-]", "", m[3])],
                "binary": True}
    return {"op": "multi", "operands": nums, "binary": False}


# ---- collect ----
def collect(args) -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35 import arith_bench as AB
    from chipformer.q35 import evals as E
    from chipformer.rv.chat import CALC_TOOL
    from chipformer.rv.evalarith import NUDGE, NUDGE_NONE, calc_response, problems, run_base

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    TC = tok.convert_tokens_to_ids("<tool_call>")
    tstr = lambda ids: [tok.decode([t]) for t in ids]  # noqa: E731

    synth_t = synth(args.n_synth, 1)
    synth_n = synth(args.n_synth_n, 2)
    bench = []
    for d in ["dmx_mul_big", "dmx_div_big", "dmx_add_or_sub_big", "dmx_mul_div_multiple_longer"]:
        bench += [{**p, "src": d} for p in AB.load(d, args.n_bench, 7)]
    bench += [{**p, "src": "arith"} for p in problems("arith", args.n_bench, 7)]
    try:
        from datasets import load_dataset
        g = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=7).select(range(args.n_gsm))
        bench += [{"id": f"gsm-{i}", "src": "gsm8k", "question": r["question"]} for i, r in enumerate(g)]
    except Exception as e:  # noqa: BLE001
        print("gsm8k skipped:", e)

    def prompt(q, tool):
        msgs = [{"role": "system", "content": NUDGE if tool else NUDGE_NONE},
                {"role": "user", "content": q + E.SUFFIX}]
        ids = tok.apply_chat_template(msgs, tools=[CALC_TOOL] if tool else None, add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        return ids if isinstance(ids, list) else ids["input_ids"]

    @torch.no_grad()
    def states(seq, pos):
        hs = model(torch.tensor([seq], device="cuda"), output_hidden_states=True).hidden_states
        idx = torch.tensor(pos, device="cuda")
        return torch.stack([hs[L][0, idx] for L in LAYERS]).to(torch.float16).cpu()  # [L, P, D]

    def user_start(seq):
        # the last "<|im_start|>user" before the generation: candidates start there
        ims = tok.convert_tokens_to_ids("<|im_start|>")
        return max(i for i, t in enumerate(seq) if t == ims and tok.decode(seq[i + 1:i + 2]).strip() == "user")

    meta, shard, k = [], [], 0

    def flush():
        nonlocal shard, k
        if shard:
            torch.save(shard, out / f"h{k:03d}.pt")
            k += 1
            shard = []

    jobs = [("T", p) for p in synth_t + bench] + [("N", p) for p in synth_n]
    done = {json.loads(x)["key"] for x in (out / "meta.jsonl").read_text().splitlines()} \
        if (out / "meta.jsonl").exists() else set()
    mf = open(out / "meta.jsonl", "a")
    for n_job, (cond, p) in enumerate(jobs):
        key = f"{cond}:{p['id']}"
        if key in done:
            continue
        ids = prompt(p["question"], cond == "T")
        if cond == "T":
            gen, n_calls, _ = run_base(model, tok, ids, True, args.budget)
            seq = ids + gen
        else:
            seq, n_calls = ids, 0
        toks = tstr(seq)
        u0 = user_start(seq)
        runs = [r for r in number_runs(toks) if r[0] > u0]
        rec = {"key": key, "cond": cond, "src": p["src"], "id": p["id"], "n_prompt": len(ids), "len": len(seq),
               "runs": runs, "calls": []}
        if p["src"] == "synth":
            rec["synth"] = {"op": p["op"], "operands": [p["a"], p["b"]]}
        dec = []
        if cond == "T":
            # Qwen's own positions (not the forced tool responses): from the prompt end, skipping
            # each response (between </tool_call> and the next Qwen token)
            text = tok.decode(seq[len(ids):], skip_special_tokens=False)
            exprs = re.findall(r"<tool_call>.*?<parameter=expression>\s*(.*?)\s*</parameter>", text, re.S)
            calls = [i for i in range(len(ids), len(seq)) if seq[i] == TC]
            for ci, pc in enumerate(calls):
                c = parse_call(exprs[ci]) if ci < len(exprs) else {"op": "none", "operands": [], "binary": False}
                c["pos"] = pc - 1
                c["expr"] = exprs[ci] if ci < len(exprs) else ""
                c["targets"] = [[ri for ri, r in enumerate(runs) if r[1] < pc and r[2] == o] for o in c["operands"]]
                rec["calls"].append(c)
            # the host's forced responses: after each </tool_call>, exactly the tokens run_base added
            tcc = tok.convert_tokens_to_ids("</tool_call>")
            forced = set()
            for e in [i for i in range(len(ids), len(seq)) if seq[i] == tcc]:
                txt = tok.decode(seq[len(ids):e + 1], skip_special_tokens=False)
                mm = re.findall(r"<parameter=[^>]*>(.*?)</parameter>", txt.split("<tool_call>")[-1], re.S)
                n_resp = len(tok(calc_response(mm[0] if mm else ""), add_special_tokens=False).input_ids)
                forced.update(range(e + 1, e + 1 + n_resp))
            own = [len(ids) - 1]
            own += [i for i in range(len(ids), len(seq) - 1) if i not in forced]
            rec["own"] = own
            rec["next_is_call"] = [int(seq[i + 1] == TC) if i + 1 < len(seq) else 0 for i in own]
            dec = own
        else:
            rec["own"], rec["next_is_call"] = [len(ids) - 1], [0]
            dec = rec["own"]
        rec["n_calls"] = n_calls
        pos = sorted(set(dec) | {r[1] for r in runs})
        rec["pos"] = pos
        shard.append({"key": key, "pos": pos, "h": states(seq, pos)})
        rec["shard"] = k
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        if len(shard) >= 64:
            flush()
        if n_job % 25 == 0:
            print(n_job, len(jobs), key, len(seq), n_calls, flush=True)
    flush()
    print("done", flush=True)


# ---- fit ----
def load(out: Path):
    meta = [json.loads(x) for x in (out / "meta.jsonl").read_text().splitlines()]
    H = {}
    for f in sorted(out.glob("h*.pt")):
        for s in torch.load(f):
            H[s["key"]] = s
    meta = [m for m in meta if m["key"] in H]
    return meta, H


def split(meta, frac=0.8, seed=0):
    ids = sorted({m["id"] for m in meta})
    random.Random(seed).shuffle(ids)
    tr = set(ids[: int(frac * len(ids))])
    return [m for m in meta if m["id"] in tr], [m for m in meta if m["id"] not in tr]


def at(H, m, li, positions):
    s = H[m["key"]]
    ix = [s["pos"].index(p) for p in positions]
    return s["h"][li, ix].float()


def logistic(X, y, Xt, n_cls, epochs=300, wd=1e-3, balance=False):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    W = torch.zeros(X.shape[1], n_cls, requires_grad=True, device=X.device)
    b = torch.zeros(n_cls, requires_grad=True, device=X.device)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        w = (len(y) / (n_cls * torch.bincount(y, minlength=n_cls).clamp(min=1)).float()) if balance else None
        torch.nn.functional.cross_entropy(X @ W + b, y, weight=w).backward()
        opt.step()
    with torch.no_grad():
        return (Xt @ W + b)


def commands(meta, cond):
    """(meta, query pos, op, [slot0 run targets], [slot1 run targets]) for binary commands."""
    out = []
    for m in meta:
        if m["cond"] != cond:
            continue
        if cond == "N":
            op, ops = m["synth"]["op"], m["synth"]["operands"]
            tg = [[ri for ri, r in enumerate(m["runs"]) if r[2] == o] for o in ops]
            out.append((m, m["own"][0], op, tg))
        else:
            saved = set(m["pos"])  # a call right after a tool response decides at a forced token (not saved)
            for c in m["calls"]:
                if c["binary"] and c["op"] in OPS and c["pos"] in saved:
                    out.append((m, c["pos"], c["op"], c["targets"]))
    return out


def pointer_fit(H, tr, te, li, rank=64, epochs=200, dev="cuda"):
    """A bilinear pointer per slot: score(q, key) = (A q) . (B k) over the context's numbers."""
    D = next(iter(H.values()))["h"].shape[-1]
    res = []
    for slot in range(2):
        def batch(items):
            for m, q, op, tg in items:
                cand = [r[1] for r in m["runs"] if r[1] < q + 1]
                if not tg[slot] or not cand:
                    continue
                Q = at(H, m, li, [q]).to(dev)
                K = at(H, m, li, cand).to(dev)
                yield Q, K, tg[slot], tg[1 - slot], op, m
        trb = list(batch(tr))
        # standardized inputs: a few of Qwen's dims are orders of magnitude larger than the rest
        qs, ks = torch.cat([b[0] for b in trb]), torch.cat([b[1] for b in trb])
        A = torch.nn.Sequential(Std(qs), torch.nn.Linear(D, rank, bias=False)).to(dev)
        B = torch.nn.Sequential(Std(ks), torch.nn.Linear(D, rank, bias=False)).to(dev)
        opt = torch.optim.Adam([*A[1].parameters(), *B[1].parameters()], lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs // 10):
            random.shuffle(trb)
            for Q, K, pos, _, _, _ in trb:
                opt.zero_grad()
                s = (A(Q) @ B(K).T)[0] / rank ** 0.5
                loss = -torch.logsumexp(s[pos], 0) + torch.logsumexp(s, 0)
                loss.backward()
                opt.step()
        res.append((A, B))
    return res


class Std(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0) + 1e-3)

    def forward(self, x):
        return (x - self.mu) / self.sd


def evaluate(H, items, li, ptr, dev="cuda"):
    """Per command: the op from the op probe is scored separately; here: slot picks vs targets.
    + and * accept the swapped pair."""
    ok = tot = 0
    with torch.no_grad():
        for m, q, op, tg in items:
            cand = [r[1] for r in m["runs"] if r[1] < q + 1]
            if not cand or not tg[0] or not tg[1]:
                continue
            Q = at(H, m, li, [q]).to(dev)
            K = at(H, m, li, cand).to(dev)
            pick = [int((A(Q) @ B(K).T)[0].argmax()) for A, B in ptr]
            good = pick[0] in tg[0] and pick[1] in tg[1]
            if op in "+*":
                good = good or (pick[0] in tg[1] and pick[1] in tg[0])
            ok += good
            tot += 1
    return ok, tot


def heuristics(items):
    """Order baselines: the first two numbers of the question; the last two numbers before q."""
    r = {"first-two": [0, 0], "last-two": [0, 0]}
    for m, q, op, tg in items:
        cand = [i for i, run in enumerate(m["runs"]) if run[1] < q + 1]
        if not tg[0] or not tg[1] or len(cand) < 2:
            continue
        for name, pick in (("first-two", cand[:2]), ("last-two", cand[-2:])):
            good = pick[0] in tg[0] and pick[1] in tg[1]
            if op in "+*":
                good = good or (pick[0] in tg[1] and pick[1] in tg[0])
            r[name][0] += good
            r[name][1] += 1
    return r


def fit(args) -> None:
    out = Path(args.out)
    meta, H = load(out)
    print(f"{len(meta)} sequences; T {sum(m['cond'] == 'T' for m in meta)}, N {sum(m['cond'] == 'N' for m in meta)}")
    T = [m for m in meta if m["cond"] == "T"]
    calls = [c for m in T for c in m["calls"]]
    bin_ = [c for c in calls if c["binary"]]
    found = [c for c in bin_ if all(c["targets"])]
    print(f"T calls {len(calls)}: binary {len(bin_)}, operands all present in the context {len(found)}")
    by_src = {}
    for m in T:
        for c in m["calls"]:
            s = by_src.setdefault(m["src"], [0, 0, 0])
            s[0] += 1
            s[1] += c["binary"]
            s[2] += c["binary"] and all(c["targets"])
    for s, v in sorted(by_src.items()):
        print(f"  {s}: calls {v[0]}, binary {v[1]}, operands in context {v[2]}")
    tr, te = split(meta)
    report = {"layers": {}}
    for li, L in enumerate(LAYERS):
        row = {}
        # call now: every Qwen-written position in T
        def cn(ms):
            X, y = [], []
            for m in ms:
                if m["cond"] != "T":
                    continue
                X.append(at(H, m, li, m["own"]))
                y += m["next_is_call"]
            return torch.cat(X).cuda(), torch.tensor(y).cuda()
        Xa, ya = cn(tr)
        Xb, yb = cn(te)
        p = logistic(Xa, ya, Xb, 2, balance=True).softmax(-1)[:, 1]
        pred = p > 0.5
        pos_s, neg_s = p[yb == 1], p[yb == 0]
        auc = float((pos_s[:, None] > neg_s[None, :]).float().mean()) if len(pos_s) and len(neg_s) else 0.0
        tp = int((pred & (yb == 1)).sum())
        row["call_now"] = {"auc": auc, "pos": int(yb.sum()), "neg": int((yb == 0).sum()), "recall": tp / max(1, int(yb.sum())),
                           "precision": tp / max(1, int(pred.sum())), "false_pos": int((pred & (yb == 0)).sum())}
        # op and pointers per condition
        for cond in ("T", "N"):
            a, b = commands(tr, cond), commands(te, cond)
            if len(a) < 20 or len(b) < 5:
                continue
            Xa = torch.cat([at(H, m, li, [q]) for m, q, _, _ in a]).cuda()
            Xb = torch.cat([at(H, m, li, [q]) for m, q, _, _ in b]).cuda()
            ya = torch.tensor([OPS.index(o) for _, _, o, _ in a]).cuda()
            yb = torch.tensor([OPS.index(o) for _, _, o, _ in b]).cuda()
            op_acc = float((logistic(Xa, ya, Xb, 4).argmax(-1) == yb).float().mean())
            ptr = pointer_fit(H, a, b, li)
            ok, tot = evaluate(H, b, li, ptr)
            row[cond] = {"n_train": len(a), "n_test": len(b), "op_acc": op_acc, "pointer_acc": ok / max(1, tot),
                         "pointer_n": tot, "heuristics": {k: v[0] / max(1, v[1]) for k, v in heuristics(b).items()}}
        report["layers"][L] = row
        print(L, json.dumps(row), flush=True)
    # transfer: train T pointers/op on synthetic, test on benchmark calls (best layer by T pointer acc)
    best = max(report["layers"], key=lambda L: report["layers"][L].get("T", {}).get("pointer_acc", 0))
    li = LAYERS.index(best)
    a = commands([m for m in meta if m["src"] == "synth"], "T")
    b = commands([m for m in meta if m["src"] != "synth"], "T")
    if len(a) > 20 and len(b) > 5:
        ptr = pointer_fit(H, a, b, li)
        ok, tot = evaluate(H, b, li, ptr)
        report["transfer_synth_to_bench"] = {"layer": best, "pointer_acc": ok / max(1, tot), "n": tot,
                                             "heuristics": {k: v[0] / max(1, v[1]) for k, v in heuristics(b).items()}}
        print("transfer", json.dumps(report["transfer_synth_to_bench"]))
    (out / "report.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "fit"))
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n-synth", type=int, default=600)
    ap.add_argument("--n-synth-n", type=int, default=1200)
    ap.add_argument("--n-bench", type=int, default=100)
    ap.add_argument("--n-gsm", type=int, default=150)
    ap.add_argument("--budget", type=int, default=768)
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
