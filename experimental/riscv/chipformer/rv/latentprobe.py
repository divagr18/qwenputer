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
PTR_BS = 1  # refit --bs: pointer minibatch size (1 = Phase 0 per-command steps)
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


def data(args):
    """(synthetic T questions, synthetic N questions, benchmark and word problems)."""
    from chipformer.q35 import arith_bench as AB
    from chipformer.rv.evalarith import problems

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
    if args.n_svamp:
        try:
            from datasets import load_dataset
            sv = load_dataset("ChilleD/SVAMP", split="test")
            sv = sv.shuffle(seed=7).select(range(min(args.n_svamp, len(sv))))
            bench += [{"id": f"svamp-{i}", "src": "svamp",
                       "question": r.get("question_concat") or f"{r['Body']} {r['Question']}"} for i, r in enumerate(sv)]
        except Exception as e:  # noqa: BLE001
            print("svamp skipped:", e)
    return synth_t, synth_n, bench


def helpers(tok, model, E, CALC_TOOL, NUDGE, NUDGE_NONE):
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

    return prompt, states, user_start


def resume(out):
    """Shards already saved; drops meta records whose shard was never written (killed mid-shard)."""
    k = len(list(out.glob("h*.pt")))
    if (out / "meta.jsonl").exists():
        keep = [x for x in (out / "meta.jsonl").read_text().splitlines() if json.loads(x)["shard"] < k]
        (out / "meta.jsonl").write_text("".join(x + "\n" for x in keep))
    return k


def select(jobs, out, args):
    done = {json.loads(x)["key"] for x in (out / "meta.jsonl").read_text().splitlines()} \
        if (out / "meta.jsonl").exists() else set()
    if args.part:  # i/n: this process takes every n-th job
        pi, pn = map(int, args.part.split("/"))
        jobs = [j for ix, j in enumerate(jobs) if ix % pn == pi]
    for d in filter(None, args.skip_from.split(",")):  # jobs another output dir already has
        if (Path(d) / "meta.jsonl").exists():
            done |= {json.loads(x)["key"] for x in (Path(d) / "meta.jsonl").read_text().splitlines()}
    return jobs, done


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
    if args.n_svamp:
        try:
            from datasets import load_dataset
            sv = load_dataset("ChilleD/SVAMP", split="test")
            sv = sv.shuffle(seed=7).select(range(min(args.n_svamp, len(sv))))
            bench += [{"id": f"svamp-{i}", "src": "svamp",
                       "question": r.get("question_concat") or f"{r['Body']} {r['Question']}"} for i, r in enumerate(sv)]
        except Exception as e:  # noqa: BLE001
            print("svamp skipped:", e)
    (out / "layers.json").write_text(json.dumps(LAYERS))
    own_rng = random.Random(3)

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

    meta, shard, k = [], [], len(list(out.glob("h*.pt")))  # resume after the shards already saved
    if (out / "meta.jsonl").exists():  # drop records whose shard was never written (killed mid-shard)
        keep = [x for x in (out / "meta.jsonl").read_text().splitlines() if json.loads(x)["shard"] < k]
        (out / "meta.jsonl").write_text("".join(x + "\n" for x in keep))

    def flush():
        nonlocal shard, k
        if shard:
            torch.save(shard, out / f"h{k:03d}.pt")
            k += 1
            shard = []

    jobs = [("T", p) for p in synth_t + bench] + [("N", p) for p in synth_n]
    done = {json.loads(x)["key"] for x in (out / "meta.jsonl").read_text().splitlines()} \
        if (out / "meta.jsonl").exists() else set()
    if args.part:  # i/n: this process takes every n-th job
        pi, pn = map(int, args.part.split("/"))
        jobs = [j for ix, j in enumerate(jobs) if ix % pn == pi]
    for d in filter(None, args.skip_from.split(",")):  # jobs another output dir already has
        if (Path(d) / "meta.jsonl").exists():
            done |= {json.loads(x)["key"] for x in (Path(d) / "meta.jsonl").read_text().splitlines()}
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
            if args.own_sample < 1:  # keep every position before a call, a sample of the rest (call-now negatives)
                own = [i for i in own if (i + 1 < len(seq) and seq[i + 1] == TC) or own_rng.random() < args.own_sample]
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
    """out: one collect directory, or several joined by commas (from collect --part)."""
    global LAYERS
    dirs = [Path(d) for d in str(out).split(",")]
    if (dirs[0] / "layers.json").exists():
        LAYERS = json.loads((dirs[0] / "layers.json").read_text())
    meta, H, seen = [], {}, set()
    for d in dirs:
        for x in (d / "meta.jsonl").read_text().splitlines():  # a key twice (parallel hand-over): keep one
            m = json.loads(x)
            if m["key"] not in seen:
                seen.add(m["key"])
                meta.append(m)
        for f in sorted(d.glob("h*.pt")):
            for s in torch.load(f):
                H.setdefault(s["key"], s)  # first dir wins, as for meta
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


class Order(torch.nn.Module):
    """A learned score bias from a candidate's place: rank from the end, rank from the start, and
    whether it is in the prompt. Lets a pointer combine content with the order priors."""
    R = 16

    def __init__(self):
        super().__init__()
        self.back = torch.nn.Parameter(torch.zeros(self.R))
        self.front = torch.nn.Parameter(torch.zeros(self.R))
        self.prompt = torch.nn.Parameter(torch.zeros(2))

    def forward(self, cand, n_prompt):
        n, dev = len(cand), self.back.device
        j = torch.arange(n, device=dev)
        inp = torch.tensor([int(c < n_prompt) for c in cand], device=dev)
        return self.back[(n - 1 - j).clamp(max=self.R - 1)] + self.front[j.clamp(max=self.R - 1)] + self.prompt[inp]


def cands(m, q):
    return [r[1] for r in m["runs"] if r[1] < q + 1]


def qslot(q, slot):
    """A command's query position for one operand slot: one position for both, or one per slot."""
    return q[slot] if isinstance(q, (tuple, list)) else q


def score(p, Q, K, cand, m, rank):
    A, B, O = p
    s = (A(Q) @ B(K).T)[0] / rank ** 0.5
    return s + O(cand, m["n_prompt"]) if O is not None else s


def pointer_fit(H, tr, te, li, rank=64, epochs=200, dev="cuda", order=False, bs=1):
    """A bilinear pointer per slot: score(q, key) = (A q) . (B k) over the context's numbers
    (plus an Order bias with order=True)."""
    D = next(iter(H.values()))["h"].shape[-1]
    res = []
    for slot in range(2):
        def batch(items):
            for m, q, op, tg in items:
                q = qslot(q, slot)
                cand = cands(m, q)
                if not tg[slot] or not cand:
                    continue
                Q = at(H, m, li, [q]).to(dev)
                K = at(H, m, li, cand).to(dev)
                yield Q, K, tg[slot], cand, m
        trb = list(batch(tr))
        # standardized inputs: a few of Qwen's dims are orders of magnitude larger than the rest
        qs, ks = torch.cat([b[0] for b in trb]), torch.cat([b[1] for b in trb])
        A = torch.nn.Sequential(Std(qs), torch.nn.Linear(D, rank, bias=False)).to(dev)
        B = torch.nn.Sequential(Std(ks), torch.nn.Linear(D, rank, bias=False)).to(dev)
        O = Order().to(dev) if order else None
        groups = [{"params": [*A[1].parameters(), *B[1].parameters()], "weight_decay": 1e-4}]
        if O is not None:
            groups.append({"params": list(O.parameters()), "weight_decay": 0.0, "lr": 1e-2})
        opt = torch.optim.Adam(groups, lr=1e-3)
        p = (A, B, O)
        if bs <= 1:  # Phase 0: one command per step
            for _ in range(epochs // 10):
                random.shuffle(trb)
                for Q, K, pos, cand, m in trb:
                    opt.zero_grad()
                    s = score(p, Q, K, cand, m, rank)
                    loss = -torch.logsumexp(s[pos], 0) + torch.logsumexp(s, 0)
                    loss.backward()
                    opt.step()
            res.append(p)
            continue
        # padded minibatches: same loss, about 100x faster
        N, C = len(trb), max(len(b[3]) for b in trb)
        Qp = torch.cat([b[0] for b in trb])
        Kp = torch.zeros(N, C, D, dtype=torch.float16, device=dev)
        mask = torch.zeros(N, C, dtype=torch.bool, device=dev)
        tgt = torch.zeros(N, C, dtype=torch.bool, device=dev)
        back = torch.zeros(N, C, dtype=torch.long, device=dev)
        front = torch.zeros(N, C, dtype=torch.long, device=dev)
        inp = torch.zeros(N, C, dtype=torch.long, device=dev)
        for i, (Q, K, pos, cand, m) in enumerate(trb):
            n = len(cand)
            Kp[i, :n] = K.half()
            mask[i, :n] = True
            tgt[i, pos] = True
            j = torch.arange(n, device=dev)
            back[i, :n] = (n - 1 - j).clamp(max=Order.R - 1)
            front[i, :n] = j.clamp(max=Order.R - 1)
            inp[i, :n] = torch.tensor([int(c < m["n_prompt"]) for c in cand], device=dev)
        del trb
        for _ in range(epochs // 2):
            perm = torch.randperm(N, device=dev)
            for b0 in range(0, N, bs):
                ix = perm[b0:b0 + bs]
                s = (B(Kp[ix].float()) @ A(Qp[ix])[:, :, None])[..., 0] / rank ** 0.5
                if O is not None:
                    s = s + O.back[back[ix]] + O.front[front[ix]] + O.prompt[inp[ix]]
                s = s.masked_fill(~mask[ix], -1e9)
                loss = (torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~tgt[ix], -1e9), 1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        del Kp
        res.append(p)
    return res


class Std(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0) + 1e-3)

    def forward(self, x):
        return (x - self.mu) / self.sd


def picks(H, items, li, ptr, dev="cuda", rank=64):
    """Per command: True/False if both slots have a target in the context (+ and * accept the
    swapped pair), else None."""
    out = []
    with torch.no_grad():
        for m, q, op, tg in items:
            cand = cands(m, qslot(q, 0))
            if not cand or not tg[0] or not tg[1]:
                out.append(None)
                continue
            pick = []
            for slot, p in enumerate(ptr):
                cand = cands(m, qslot(q, slot))
                Q = at(H, m, li, [qslot(q, slot)]).to(dev)
                K = at(H, m, li, cand).to(dev)
                pick.append(int(score(p, Q, K, cand, m, rank).argmax()))
            good = pick[0] in tg[0] and pick[1] in tg[1]
            if op in "+*":
                good = good or (pick[0] in tg[1] and pick[1] in tg[0])
            out.append(bool(good))
    return out


def evaluate(H, items, li, ptr, dev="cuda"):
    g = [x for x in picks(H, items, li, ptr, dev) if x is not None]
    return sum(g), len(g)


def heuristics(items):
    """Order baselines: the first two numbers of the question; the last two numbers before q."""
    r = {"first-two": [0, 0], "last-two": [0, 0]}
    for m, q, op, tg in items:
        cand = [i for i, run in enumerate(m["runs"]) if run[1] < qslot(q, 0) + 1]
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


def op_probe(H, a, b, li):
    Xa = torch.cat([at(H, m, li, [qslot(q, 0)]) for m, q, _, _ in a]).cuda()
    Xb = torch.cat([at(H, m, li, [qslot(q, 0)]) for m, q, _, _ in b]).cuda()
    ya = torch.tensor([OPS.index(o) for _, _, o, _ in a]).cuda()
    yb = torch.tensor([OPS.index(o) for _, _, o, _ in b]).cuda()
    return (logistic(Xa, ya, Xb, 4).argmax(-1) == yb).tolist()


def scored(a, b, H, li, order):
    """op accuracy, pointer accuracy, command accuracy (op and both pointers right) on b."""
    op_ok = op_probe(H, a, b, li)
    pk = picks(H, b, li, pointer_fit(H, a, b, li, order=order, bs=PTR_BS))
    both = [(o, p) for o, p in zip(op_ok, pk) if p is not None]
    return {"op": sum(op_ok) / len(op_ok), "pointer": sum(p for _, p in both) / max(1, len(both)),
            "command": sum(o and p for o, p in both) / max(1, len(both)), "n": len(both)}


def refit(args) -> None:
    """Phase 0b: is the synthetic -> benchmark transfer failure a data problem or a method problem?
    Leave one source out: train the op and pointer probes on every other source (synthetic,
    benchmark and word problems mixed), test on the held-out one. Each fit runs twice: content only
    (Phase 0's pointer) and content plus a learned order bias (Order). Also a mixed random split."""
    meta, H = load(args.out)
    out = Path(args.out.split(",")[0])
    T = [m for m in meta if m["cond"] == "T"]
    srcs = sorted({m["src"] for m in T})
    by = {s: commands([m for m in T if m["src"] == s], "T") for s in srcs}
    nsyn = commands(meta, "N")
    report = {"sources": {}, "layers": {}}
    for s in srcs:
        ms = [m for m in T if m["src"] == s]
        calls = [c for m in ms for c in m["calls"]]
        report["sources"][s] = {"problems": len(ms), "calls": len(calls), "binary": sum(c["binary"] for c in calls),
                                "commands": len(by[s])}
    print(json.dumps(report["sources"]), flush=True)
    layers = [int(x) for x in args.fit_layers.split(",")] if args.fit_layers else LAYERS
    for L in layers:
        li = LAYERS.index(L)
        row = {}
        for held in srcs:
            if len(by[held]) < 10:
                continue
            a = [c for s in srcs if s != held for c in by[s]] + (nsyn if held != "synth" else [])
            b = by[held]
            r = {"heuristics": {k: v[0] / max(1, v[1]) for k, v in heuristics(b).items()}}
            for name, order in (("content", False), ("content+order", True)):
                r[name] = scored(a, b, H, li, order)
            row[held] = r
            print(L, held, json.dumps(r), flush=True)
        tr, te = split(T)
        a, b = commands(tr, "T") + nsyn, commands(te, "T")
        mx = {name: scored(a, b, H, li, order) for name, order in (("content", False), ("content+order", True))}
        mx["heuristics"] = {k: v[0] / max(1, v[1]) for k, v in heuristics(b).items()}
        row["_mixed_split"] = mx
        print(L, "mixed", json.dumps(mx), flush=True)
        report["layers"][L] = row
        (out / "refit-report.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


# ---- Phase 0c: read the operands where Qwen writes them ----
def full_attn_layers(model):
    """{layer index: self_attn module} of the language model's softmax-attention layers."""
    out = {}
    for name, mod in model.named_modules():
        mm = re.search(r"language_model\.layers\.(\d+)\.self_attn$", name)
        if mm:
            out[int(mm[1])] = mod
    return out


def argpos(args) -> None:
    """Collect for Phase 0c. Condition T only (Qwen writes its calls; the host answers). Saves, per
    sequence: the token ids; hidden states at the position before each <tool_call> ("pre"), before
    each operand's first digit ("q1", "q2", the positions that predict them) and at every number's
    last digit (the pointer keys); and the full-attention layers' attention from every position that
    predicts an operand digit, over every number token in the sequence (eager attention, hooked)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35 import evals as E
    from chipformer.rv.chat import CALC_TOOL
    from chipformer.rv.evalarith import NUDGE, NUDGE_NONE, run_base

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation="eager").to("cuda").eval()
    TC, TCC = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</tool_call>")
    tstr = lambda ids: [tok.decode([t]) for t in ids]  # noqa: E731
    fl = full_attn_layers(model)
    AL = sorted(fl)
    print("full-attention layers", AL, flush=True)
    (out / "layers.json").write_text(json.dumps(LAYERS))
    (out / "attn_layers.json").write_text(json.dumps(AL))
    cap = {}

    def hook(L):
        def f(mod, inp, outp):
            if not cap.get("on"):
                return
            w = outp[1]
            assert w is not None, "no attention weights (needs eager attention)"
            wr = w[0][:, cap["rows"]].float()  # [heads, rows, T]
            mx, am = wr.max(-1)
            wp = wr[:, :, cap["P"]]
            cap[L] = (wp.half().cpu(), am.int().cpu(), mx.half().cpu(), wp.sum(-1).half().cpu())
        return f

    for L, mod in fl.items():
        mod.register_forward_hook(hook(L))
    _, _, bench = data(args)
    synth_t = synth(args.n_synth, 1)
    prompt, states, user_start = helpers(tok, model, E, CALC_TOOL, NUDGE, NUDGE_NONE)
    shard, k = [], resume(out)

    def flush():
        nonlocal shard, k
        if shard:
            torch.save(shard, out / f"h{k:03d}.pt")
            k += 1
            shard = []

    jobs, done = select([("T", p) for p in synth_t + bench], out, args)
    mf = open(out / "meta.jsonl", "a")
    for n_job, (cond, p) in enumerate(jobs):
        key = f"{cond}:{p['id']}"
        if key in done:
            continue
        ids = prompt(p["question"], True)
        gen, n_calls, _ = run_base(model, tok, ids, True, args.budget)
        seq = ids + gen
        toks = tstr(seq)
        u0 = user_start(seq)
        runs = [r for r in number_runs(toks) if r[0] > u0]
        rec = {"key": key, "cond": cond, "src": p["src"], "id": p["id"], "n_prompt": len(ids), "len": len(seq),
               "runs": runs, "calls": [], "seq": seq, "n_calls": n_calls,
               "dig": [t for r in runs for t in range(r[0], r[1] + 1) if NUM.match(toks[t])]}
        text = tok.decode(seq[len(ids):], skip_special_tokens=False)
        exprs = re.findall(r"<tool_call>.*?<parameter=expression>\s*(.*?)\s*</parameter>", text, re.S)
        calls = [i for i in range(len(ids), len(seq)) if seq[i] == TC]
        rows, qpos = [], []
        for ci, pc in enumerate(calls):
            c = parse_call(exprs[ci]) if ci < len(exprs) else {"op": "none", "operands": [], "binary": False}
            end = next((i for i in range(pc, len(seq)) if seq[i] == TCC), len(seq))
            c.update(pc=pc, pos=pc - 1, end=end, expr=exprs[ci] if ci < len(exprs) else "")
            c["targets"] = [[ri for ri, r in enumerate(runs) if r[1] < pc and r[2] == o] for o in c["operands"]]
            crun = [r for r in runs if pc < r[0] and r[1] < end]
            qpos.append(pc - 1)
            if c["binary"] and len(crun) >= 2 and [crun[0][2], crun[1][2]] == c["operands"]:
                c["q1"], c["q2"] = crun[0][0] - 1, crun[1][0] - 1
                qpos += [c["q1"], c["q2"]]
                c["rows"] = []
                for slot, r in enumerate(crun[:2]):
                    digits = [t for t in range(r[0], r[1] + 1) if NUM.match(toks[t])]
                    for kd, d in enumerate(digits):
                        c["rows"].append([slot, kd, len(rows)])
                        rows.append(d - 1)
            rec["calls"].append(c)
        pos = sorted(set(qpos) | {r[1] for r in runs})
        rec["pos"] = pos
        P = sorted({t for r in runs for t in range(r[0], r[1] + 1)})
        cap.clear()
        if rows:
            cap.update(on=True, rows=rows, P=P)
        h = states(seq, pos)
        ent = {"key": key, "pos": pos, "h": h, "rows": rows, "P": P}
        if rows:
            ent["att"] = torch.stack([cap[L][0] for L in AL])  # [attn layers, heads, rows, P]
            ent["amax"] = torch.stack([cap[L][1] for L in AL])  # argmax over the whole row
            ent["mmax"] = torch.stack([cap[L][2] for L in AL])
            ent["pmass"] = torch.stack([cap[L][3] for L in AL])  # mass on number tokens
        cap.clear()
        shard.append(ent)
        rec["shard"] = k
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        if len(shard) >= 64:
            flush()
        if n_job % 25 == 0:
            print(n_job, len(jobs), key, len(seq), n_calls, len(rows), flush=True)
    flush()
    print("done", flush=True)


def arg_items(T, where):
    """Per source, binary commands with both operands located in the call, as (meta with the runs
    before the call, query, op, targets). where: "pre" (one query, the position before <tool_call>) or
    "arg" (one query per slot: the position before that operand's first digit)."""
    by = {}
    for m in T:
        for c in m["calls"]:
            if "q1" not in c or c["op"] not in OPS:
                continue
            mc = {**m, "runs": [r for r in m["runs"] if r[1] < c["pc"]]}
            q = c["pos"] if where == "pre" else (c["q1"], c["q2"])
            by.setdefault(m["src"], []).append((mc, q, c["op"], c["targets"], c))
    return by


def head_items(T, H):
    """Per command: first-digit attention mass per candidate number [layers*heads, runs] for each slot,
    and per continuation digit whether each head's argmax (over number tokens before the call) is the
    matching digit of a target number."""
    items = []
    for m in T:
        s = H[m["key"]]
        if "att" not in s:
            continue
        A = s["att"].float()
        nL, nH = A.shape[:2]
        A = A.reshape(nL * nH, A.shape[2], A.shape[3])
        Pix = {p: i for i, p in enumerate(s["P"])}
        for c in m["calls"]:
            if "rows" not in c or c["op"] not in OPS or not all(c["targets"]):
                continue
            before = [r for r in m["runs"] if r[1] < c["pc"]]
            cols = [[Pix[t] for t in range(r[0], r[1] + 1)] for r in before]
            allc = [x for cl in cols for x in cl]
            it = {"src": m["src"], "op": c["op"], "tg": c["targets"], "M": [], "cont": [[], []]}
            for slot, kd, g in c["rows"]:
                a = A[:, g]  # [LH, P]
                if kd == 0:
                    it["M"].append(torch.stack([a[:, cl].sum(-1) for cl in cols], 1))  # [LH, runs]
                else:
                    am = a[:, allc].argmax(-1)
                    want = set()
                    for ri in c["targets"][slot]:
                        r = m["runs"][ri]
                        dg = [t for t in range(r[0], r[1] + 1) if t in m["digit"]]
                        if kd < len(dg):
                            want.add(Pix[dg[kd]])
                    it["cont"][slot].append(torch.tensor([allc[int(i)] in want for i in am]))
            if len(it["M"]) == 2:
                items.append(it)
    return items


def head_eval(tr, te, n_iter=400):
    """Training-free and lightly fit pointers from attention: the best single head and top-k sums
    (heads ranked by first-digit hit rate on tr), and a learned weight per head (softmax over the
    candidates, trained on tr). Pointer = both slots right (+ and * accept the swap)."""
    def ok(picks, it):
        g = picks[0] in it["tg"][0] and picks[1] in it["tg"][1]
        if it["op"] in "+*":
            g = g or (picks[0] in it["tg"][1] and picks[1] in it["tg"][0])
        return g

    LH = tr[0]["M"][0].shape[0]
    hit = torch.zeros(LH)
    n = 0
    for it in tr:
        for slot in range(2):
            am = it["M"][slot].argmax(1)
            hit += torch.tensor([int(a) in it["tg"][slot] for a in am], dtype=torch.float)
            n += 1
    rank = hit.argsort(descending=True)
    res = {"train_hit_top": [[int(h), float(hit[h] / n)] for h in rank[:8]]}
    for kk in (1, 4, 16):
        sel = rank[:kk]
        g = [ok([int(it["M"][s][sel].sum(0).argmax()) for s in range(2)], it) for it in te]
        res[f"top{kk}"] = sum(g) / max(1, len(g))
    # held-out first-digit hit rate of the train-best head, per slot
    h0 = rank[0]
    res["best_head_test_hit"] = sum(int(it["M"][s][h0].argmax()) in it["tg"][s] for it in te for s in range(2)) \
        / max(1, 2 * len(te))
    cont = [x[h0].item() for it in te for s in range(2) for x in it["cont"][s]]
    res["best_head_test_cont_hit"] = sum(cont) / max(1, len(cont))
    # learned per-head weights (one per slot)
    def pad(items, slot):
        C = max(it["M"][slot].shape[1] for it in items)
        X = torch.zeros(len(items), LH, C)
        mask = torch.zeros(len(items), C, dtype=torch.bool)
        tgt = torch.zeros(len(items), C, dtype=torch.bool)
        for i, it in enumerate(items):
            c = it["M"][slot].shape[1]
            X[i, :, :c] = it["M"][slot]
            mask[i, :c] = True
            tgt[i, it["tg"][slot]] = True
        return X.cuda(), mask.cuda(), tgt.cuda()

    probs = []
    for slot in range(2):
        X, mask, tgt = pad(tr, slot)
        w = torch.zeros(LH, device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(n_iter):
            sc = (torch.einsum("h,nhc->nc", w, X) * 10).masked_fill(~mask, -1e9)
            loss = (torch.logsumexp(sc, 1) - torch.logsumexp(sc.masked_fill(~tgt, -1e9), 1)).mean() + 1e-3 * w.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            def run(items):
                Xt, mt, _ = pad(items, slot)
                sc = (torch.einsum("h,nhc->nc", w, Xt) * 10).masked_fill(~mt, -1e9)
                pr = sc.softmax(1)
                return pr.argmax(1).tolist(), pr.max(1).values.tolist()
            probs.append((run(tr), run(te)))
    g_tr = [ok([probs[0][0][0][i], probs[1][0][0][i]], it) for i, it in enumerate(tr)]
    conf_tr = [min(probs[0][0][1][i], probs[1][0][1][i]) for i in range(len(tr))]
    g_te = [ok([probs[0][1][0][i], probs[1][1][0][i]], it) for i, it in enumerate(te)]
    conf_te = [min(probs[0][1][1][i], probs[1][1][1][i]) for i in range(len(te))]
    res["learned"] = sum(g_te) / max(1, len(g_te))
    # abstain below a confidence threshold set on tr for 99% train accuracy
    tau = 1.0
    for t in sorted(set(conf_tr)):
        acc = [g for g, c in zip(g_tr, conf_tr) if c >= t]
        if acc and sum(acc) / len(acc) >= 0.99:
            tau = t
            break
    acc = [g for g, c in zip(g_te, conf_te) if c >= tau]
    res["abstain"] = {"tau": tau, "coverage": len(acc) / max(1, len(g_te)), "acc": sum(acc) / max(1, len(acc))}
    res["n"] = len(te)
    return res


def posfit(args) -> None:
    """Phase 0c: leave one source out, with the probes' query at the position before <tool_call> ("pre")
    or at the position before each operand's first digit ("arg"); and pointers read from the
    full-attention heads at the first digit (training-free top-k, and one learned weight per head)."""
    meta, H = load(args.out)
    out = Path(args.out.split(",")[0])
    T = [m for m in meta if m["cond"] == "T"]
    for m in T:
        m["digit"] = set(m["dig"])
    srcs = sorted({m["src"] for m in T})
    report = {"sources": {}, "layers": {}, "heads": {}}
    pre, arg = arg_items(T, "pre"), arg_items(T, "arg")
    for s in srcs:
        report["sources"][s] = {"commands": len(pre.get(s, [])),
                                "with_targets": sum(all(x[3]) for x in pre.get(s, []))}
    print(json.dumps(report["sources"]), flush=True)
    # attention pointers
    hi = head_items(T, H)
    for held in srcs + ["_mixed_split"]:
        if held == "_mixed_split":
            ids = sorted({i for i in range(len(hi))})
            random.Random(0).shuffle(ids)
            cut = int(0.8 * len(ids))
            tr, te = [hi[i] for i in ids[:cut]], [hi[i] for i in ids[cut:]]
        else:
            tr, te = [it for it in hi if it["src"] != held], [it for it in hi if it["src"] == held]
        if len(te) < 10:
            continue
        r = head_eval(tr, te)
        report["heads"][held] = r
        print("heads", held, json.dumps(r), flush=True)
        (out / "posfit-report.json").write_text(json.dumps(report, indent=1))
    layers = [int(x) for x in args.fit_layers.split(",")] if args.fit_layers else LAYERS
    strip = lambda xs: [x[:4] for x in xs]  # noqa: E731
    for L in layers:
        li = LAYERS.index(L)
        row = {}
        for held in srcs:
            if len(pre.get(held, [])) < 10:
                continue
            r = {}
            for where, by in (("pre", pre), ("arg", arg)):
                a = strip([x for s in srcs if s != held for x in by.get(s, [])])
                b = strip(by[held])
                r[where] = scored(a, b, H, li, False)
            r["heuristics"] = {k: v[0] / max(1, v[1]) for k, v in heuristics(strip(pre[held])).items()}
            row[held] = r
            print(L, held, json.dumps(r), flush=True)
        report["layers"][L] = row
        (out / "posfit-report.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


# ---- Phase 0d: read the command at the answer, with no call ----
DIRECT_SYS = "Answer with only the final number. Do not show any working."
DIRECT_SUFFIX = "\n\nReply with only the final answer, as a single number."


def anspos(args) -> None:
    """Collect for Phase 0d. No tools. Two conditions per problem: D (direct: Qwen answers with only
    the number) and C (Qwen works step by step in text, the evals' "Do every arithmetic step exactly"
    prompt). Saves, per sequence: the token ids; at the position that predicts the first digit of
    every number Qwen writes, the hidden states and the full-attention layers' attention over every
    number token in the sequence. Labels (which numbers Qwen computes, from what) come at fit time
    from Phase 0c's tool calls on the same problems."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35 import evals as E
    from chipformer.rv.chat import CALC_TOOL
    from chipformer.rv.evalarith import NUDGE, NUDGE_NONE, run_base

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation="eager").to("cuda").eval()
    tstr = lambda ids: [tok.decode([t]) for t in ids]  # noqa: E731
    fl = full_attn_layers(model)
    AL = sorted(fl)
    (out / "layers.json").write_text(json.dumps(LAYERS))
    (out / "attn_layers.json").write_text(json.dumps(AL))
    cap = {}

    def hook(L):
        def f(mod, inp, outp):
            if not cap.get("on"):
                return
            assert outp[1] is not None, "no attention weights (needs eager attention)"
            wr = outp[1][0][:, cap["rows"]].float()  # [heads, rows, T]
            cap[L] = (wr[:, :, cap["P"]].half().cpu(), wr[:, :, cap["P"]].sum(-1).half().cpu())
        return f

    for L, mod in fl.items():
        mod.register_forward_hook(hook(L))
    _, _, bench = data(args)
    synth_t = synth(args.n_synth, 1)
    _, states, user_start = helpers(tok, model, E, CALC_TOOL, NUDGE, NUDGE_NONE)

    def prompt(q, cond):
        sysm, suf = (DIRECT_SYS, DIRECT_SUFFIX) if cond == "D" else (NUDGE_NONE, E.SUFFIX)
        ids = tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": q + suf}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return ids if isinstance(ids, list) else ids["input_ids"]

    shard, k = [], resume(out)

    def flush():
        nonlocal shard, k
        if shard:
            torch.save(shard, out / f"h{k:03d}.pt")
            k += 1
            shard = []

    probs = synth_t + bench
    jobs = [(c, p) for p in probs for c in ("D", "C")]
    random.Random(0).shuffle(jobs)  # parts get a mix of the short D and long C jobs
    jobs, done = select(jobs, out, args)
    mf = open(out / "meta.jsonl", "a")
    for n_job, (cond, p) in enumerate(jobs):
        key = f"{cond}:{p['id']}"
        if key in done:
            continue
        ids = prompt(p["question"], cond)
        model.set_attn_implementation("sdpa")  # decode fast; eager only for the attention pass
        gen, _, _ = run_base(model, tok, ids, False, 48 if cond == "D" else args.budget)
        model.set_attn_implementation("eager")
        seq = ids + gen
        toks = tstr(seq)
        u0 = user_start(seq)
        runs = [r for r in number_runs(toks) if r[0] > u0]
        gi = [i for i, r in enumerate(runs) if r[0] >= len(ids)][:64]  # numbers Qwen wrote
        rows = [runs[i][0] - 1 for i in gi]
        P = sorted({t for r in runs for t in range(r[0], r[1] + 1)})
        rec = {"key": key, "cond": cond, "src": p["src"], "id": p["id"], "n_prompt": len(ids), "len": len(seq),
               "runs": runs, "gen_runs": gi, "rows": rows, "seq": seq,
               "text": tok.decode(gen, skip_special_tokens=True),
               "dig": [t for r in runs for t in range(r[0], r[1] + 1) if NUM.match(toks[t])]}
        for g in ("gold", "answer"):
            if g in p:
                rec["gold"] = str(p[g])
        cap.clear()
        if rows:
            cap.update(on=True, rows=rows, P=P)
        ent = {"key": key, "pos": rows, "P": P}
        ent["h"] = states(seq, rows) if rows else None
        if rows:
            ent["att"] = torch.stack([cap[L][0] for L in AL])  # [attn layers, heads, rows, P]
            ent["pmass"] = torch.stack([cap[L][1] for L in AL])
        cap.clear()
        shard.append(ent)
        rec["shard"] = k
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        if len(shard) >= 64:
            flush()
        if n_job % 25 == 0:
            print(n_job, len(jobs), key, len(seq), len(rows), flush=True)
    flush()
    print("done", flush=True)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def same(v, R, vs):
    """Qwen's written number (value v, string vs) equals the exact result R, up to v's decimals."""
    if v is None or R is None:
        return False
    dec = len(vs.split(".")[1]) if "." in vs else 0
    tol = 0.5 * 10 ** -dec if (dec or R != int(R)) else 0
    return abs(v - R) <= max(1e-9 * abs(R), tol) + 1e-12


def apply_op(op, a, b):
    if a is None or b is None:
        return None
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]


def tool_labels(dirs):
    """Phase 0c meta: per problem id, its binary calls (op, a, b, exact result R)."""
    lab = {}
    for d in dirs.split(","):
        for x in (Path(d) / "meta.jsonl").read_text().splitlines():
            m = json.loads(x)
            calls = []
            for c in m["calls"]:
                if c.get("binary") and c["op"] in OPS:
                    a, b = num(c["operands"][0]), num(c["operands"][1])
                    calls.append({"op": c["op"], "ops": c["operands"], "a": a, "b": b,
                                  "R": apply_op(c["op"], a, b)})
            lab.setdefault(m["id"], {"calls": calls, "n": len(m["calls"])})
    return lab


def ans_items(meta, H, lab, tok_text):
    """Labelled rows. D: the first number Qwen writes, labelled with the problem's one tool call (problems
    whose Phase 0c run made exactly one call, binary). C: every number Qwen writes that is not a copy
    of an earlier number and equals the result of one of the problem's tool calls whose operands are
    both earlier numbers; "written" when the operands and "=" are in the text right before it."""
    items, trig = [], []
    for m in meta:
        s = H[m["key"]]
        L = lab.get(m["id"])
        if s.get("att") is None or L is None:
            continue
        A = s["att"].float()
        nL, nH = A.shape[:2]
        A = A.reshape(nL * nH, A.shape[2], A.shape[3])
        Pix = {p: i for i, p in enumerate(s["P"])}
        runs = m["runs"]
        for j, gi in enumerate(m["gen_runs"]):
            g = runs[gi]
            before = [i for i, r in enumerate(runs) if r[1] < g[0]]
            v = num(g[2])
            copied = any(num(runs[i][2]) == v for i in before)
            if m["cond"] == "C":
                trig.append({"src": m["src"], "key": m["key"], "row": j, "y": int(not copied)})
            cands = []
            if m["cond"] == "D":
                if j == 0 and L["n"] == 1 and len(L["calls"]) == 1:
                    cands = L["calls"]
            elif not copied:
                cands = [c for c in L["calls"] if same(v, c["R"], g[2])]
            for c in cands:
                tg = [[x for x, i in enumerate(before) if num(runs[i][2]) == c[k]] for k in ("a", "b")]
                if not tg[0] or not tg[1]:
                    continue
                cols = [[Pix[t] for t in range(runs[i][0], runs[i][1] + 1)] for i in before]
                M = torch.stack([A[:, j, cl].sum(-1) for cl in cols], 1)  # [LH, runs before]
                tail = re.sub(r"[,\s$]", "", tok_text(m["seq"][max(0, g[0] - 24):g[0]]))
                written = tail.endswith("=") and c["ops"][0] in tail and c["ops"][1] in tail
                items.append({"src": m["src"], "cond": m["cond"], "key": m["key"], "row": j, "op": c["op"],
                              "tg": tg, "M": M, "written": written, "R": c["R"],
                              "qwen_right": same(v, c["R"], g[2]), "start": [runs[i][0] for i in before]})
                break
    return items, trig


def pair_ok(p0, p1, it, ordered=True):
    if set(it["tg"][0]) == set(it["tg"][1]):  # a == b: one number, both slots
        return p0 in it["tg"][0] and (p1 in it["tg"][0] or p1 == p0)
    g = p0 in it["tg"][0] and p1 in it["tg"][1]
    if it["op"] in "+*" or not ordered:
        g = g or (p0 in it["tg"][1] and p1 in it["tg"][0])
    return g


def top2(sc):
    if sc.numel() < 2:
        return 0, 0
    t = sc.topk(2).indices.tolist()
    return t[0], t[1]


def in_order(p, it):
    """The pair in text order (the earlier number as the first operand)."""
    a, b = p
    return (a, b) if it["start"][a] <= it["start"][b] else (b, a)


def ans_eval(tr, te, n_iter=400):
    """Pointers from one query position: unordered pair = the two numbers with the most attention
    (one head, a top-k sum, or one learned weight per head); ordered = that pair in text order, or one
    learned weight vector per slot. Heads ranked on tr by unordered pair hits."""
    LH = tr[0]["M"].shape[0]
    hit = torch.zeros(LH)
    for it in tr:
        for h in range(LH):
            hit[h] += pair_ok(*top2(it["M"][h]), it, ordered=False)
    rank = hit.argsort(descending=True)
    res = {"n": len(te), "train_top_heads": [[int(h), float(hit[h] / len(tr))] for h in rank[:6]]}
    for kk in (1, 4, 16):
        sel = rank[:kk]
        pr = [top2(it["M"][sel].sum(0)) for it in te]
        res[f"top{kk}_pair"] = sum(pair_ok(*p, it, False) for p, it in zip(pr, te)) / max(1, len(te))
        res[f"top{kk}_ordered"] = sum(pair_ok(*in_order(p, it), it) for p, it in zip(pr, te)) / max(1, len(te))

    def pad(items):
        C = max(it["M"].shape[1] for it in items)
        X = torch.zeros(len(items), LH, C)
        mask = torch.zeros(len(items), C, dtype=torch.bool)
        tg = [torch.zeros(len(items), C, dtype=torch.bool) for _ in range(2)]
        for i, it in enumerate(items):
            X[i, :, :it["M"].shape[1]] = it["M"]
            mask[i, :it["M"].shape[1]] = True
            for s in range(2):
                tg[s][i, it["tg"][s]] = True
        return X.cuda(), mask.cuda(), [t.cuda() for t in tg]

    X, mask, tg = pad(tr)
    Xt, mt, _ = pad(te)

    def train(targets):
        w = torch.zeros(LH, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(n_iter):
            sc = (torch.einsum("h,nhc->nc", w, X) * 10).masked_fill(~mask, -1e9)
            lz = torch.logsumexp(sc, 1)
            loss = sum((lz - torch.logsumexp(sc.masked_fill(~t, -1e9), 1)).mean() for t in targets) \
                + 1e-3 * w.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (torch.einsum("h,nhc->nc", w, Xt) * 10).masked_fill(~mt, -1e9)

    both = train(tg)  # one weight vector, mass on both operands
    pr = [top2(both[i, :it["M"].shape[1]]) for i, it in enumerate(te)]
    res["learned_pair"] = sum(pair_ok(*p, it, False) for p, it in zip(pr, te)) / max(1, len(te))
    res["learned_ordered_textorder"] = sum(pair_ok(*in_order(p, it), it) for p, it in zip(pr, te)) / max(1, len(te))
    s0, s1 = train([tg[0]]), train([tg[1]])
    ok = []
    for i, it in enumerate(te):
        c = it["M"].shape[1]
        a, b = int(s0[i, :c].argmax()), int(s1[i, :c].argmax())
        if a == b and c > 1:  # one number for both slots: the runner-up of the less sure slot
            p0, p1 = s0[i, :c].softmax(0), s1[i, :c].softmax(0)
            if p0[a] < p1[b]:
                a = int(s0[i, :c].topk(2).indices[1])
            else:
                b = int(s1[i, :c].topk(2).indices[1])
        ok.append(pair_ok(a, b, it))
    res["learned_ordered_slots"] = sum(ok) / max(1, len(te))
    res["_ok_ordered"] = ok
    # order heuristics over the numbers before the row
    for name, f in (("first-two", lambda c: (0, 1)), ("last-two", lambda c: (c - 2, c - 1))):
        g = [pair_ok(*f(it["M"].shape[1]), it) for it in te if it["M"].shape[1] >= 2]
        res[name] = sum(g) / max(1, len(g))
    return res


def auc(score, y):
    pos = [s for s, t in zip(score, y) if t]
    neg = [s for s, t in zip(score, y) if not t]
    if not pos or not neg:
        return None
    p, n = torch.tensor(pos)[:, None], torch.tensor(neg)[None, :]
    return float(((p > n).float() + 0.5 * (p == n).float()).mean())


def ans_build(args):
    """Labelled rows of anspos dirs, each with its hidden states [layers, D] attached (fp16); trigger
    rows keep every computed number and a quarter of the copies."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    meta, H = load(args.out)
    items, trig = ans_items(meta, H, tool_labels(args.labels), lambda ids: tok.decode(ids))
    rng = random.Random(0)
    trig = [t for t in trig if t["y"] or rng.random() < 0.25]
    for x in items + trig:
        x["h"] = H[x["key"]]["h"][:, x["row"]].half().clone()
    d_rows = {}
    for m in meta:
        if m["cond"] == "D":
            d_rows[m["src"]] = d_rows.get(m["src"], 0) + 1
    return {"items": items, "trig": trig, "layers": LAYERS, "d_rows": d_rows}


def anspack(args) -> None:
    """Phase 0d: reduce this box's anspos dirs to the labelled rows (pack.pt in the first dir)."""
    pk = ans_build(args)
    torch.save(pk, Path(args.out.split(",")[0]) / "pack.pt")
    print("packed", len(pk["items"]), len(pk["trig"]), flush=True)


def ansfit(args) -> None:
    """Phase 0d: pointers and op at the answer position, leave one source out, per condition (D direct,
    C-bare: computed numbers in step-by-step text with no expression right before them, C-written).
    args.out: anspos dirs, or pack.pt files (anspack), comma-separated."""
    global LAYERS
    if args.out.endswith(".pt"):
        packs = [torch.load(f) for f in args.out.split(",")]
        out = Path(args.out.split(",")[0]).parent
    else:
        packs = [ans_build(args)]
        out = Path(args.out.split(",")[0])
    LAYERS = packs[0]["layers"]
    items = [x for pk in packs for x in pk["items"]]
    trig = [x for pk in packs for x in pk["trig"]]
    d_rows = {}
    for pk in packs:
        for k, v in pk["d_rows"].items():
            d_rows[k] = d_rows.get(k, 0) + v
    groups = {"D": [it for it in items if it["cond"] == "D"],
              "C-bare": [it for it in items if it["cond"] == "C" and not it["written"]],
              "C-written": [it for it in items if it["cond"] == "C" and it["written"]]}
    report = {"counts": {g: {s: sum(it["src"] == s for it in v) for s in sorted({x["src"] for x in v})}
                         for g, v in groups.items()},
              "d_rows": d_rows,
              "pointer": {}, "op": {}, "trigger": {}, "end_to_end": {}}
    print(json.dumps(report["counts"]), flush=True)
    LI = {L: i for i, L in enumerate(LAYERS)}

    def hrow(it, li):
        return it["h"][li].float()

    for g, its in groups.items():
        srcs = sorted({it["src"] for it in its})
        report["pointer"][g], report["op"][g], report["end_to_end"][g] = {}, {}, {}
        for held in srcs:
            tr, te = [it for it in its if it["src"] != held], [it for it in its if it["src"] == held]
            if len(te) < 10 or len(tr) < 20:
                continue
            r = ans_eval(tr, te)
            ok = r.pop("_ok_ordered")
            report["pointer"][g][held] = r
            ops, best = {}, None
            for L, li in LI.items():
                Xa = torch.stack([hrow(it, li) for it in tr]).cuda()
                Xb = torch.stack([hrow(it, li) for it in te]).cuda()
                ya = torch.tensor([OPS.index(it["op"]) for it in tr]).cuda()
                yb = torch.tensor([OPS.index(it["op"]) for it in te]).cuda()
                o = (logistic(Xa, ya, Xb, 4).argmax(-1) == yb).tolist()
                ops[L] = sum(o) / len(o)
                if best is None or ops[L] > ops[best[0]]:
                    best = (L, o)
            report["op"][g][held] = ops
            cmd = [a and b for a, b in zip(ok, best[1])]
            report["end_to_end"][g][held] = {
                "n": len(te), "qwen_writes_R": sum(it["qwen_right"] for it in te) / len(te),
                "latent_command_right": sum(cmd) / len(te), "op_layer": best[0]}
            print(g, held, json.dumps({"ptr": r, "op": ops, "e2e": report["end_to_end"][g][held]}), flush=True)
            (out / "ansfit-report.json").write_text(json.dumps(report, indent=1))
    # trigger: will the next number be computed (not a copy)? C rows, per layer, leave one source out
    srcs = sorted({t["src"] for t in trig})
    for L, li in LI.items():
        row = {}
        for held in srcs:
            tr, te = [t for t in trig if t["src"] != held], [t for t in trig if t["src"] == held]
            if len({t["y"] for t in te}) < 2:
                continue
            Xa = torch.stack([t["h"][li].float() for t in tr]).cuda()
            Xb = torch.stack([t["h"][li].float() for t in te]).cuda()
            ya = torch.tensor([t["y"] for t in tr]).cuda()
            sc = logistic(Xa, ya, Xb, 2, balance=True).softmax(-1)[:, 1].tolist()
            row[held] = {"auc": auc(sc, [t["y"] for t in te]), "n": len(te), "pos": sum(t["y"] for t in te)}
        report["trigger"][L] = row
        print("trigger", L, json.dumps(row), flush=True)
        (out / "ansfit-report.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "fit", "refit", "argpos", "posfit", "anspos", "anspack", "ansfit"))
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n-synth", type=int, default=600)
    ap.add_argument("--n-synth-n", type=int, default=1200)
    ap.add_argument("--n-bench", type=int, default=100)
    ap.add_argument("--n-gsm", type=int, default=150)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--n-svamp", type=int, default=0)
    ap.add_argument("--layers", default="", help="collect: layers to save, e.g. 8,12,16,20")
    ap.add_argument("--own-sample", type=float, default=1.0, help="collect: fraction of non-call positions kept")
    ap.add_argument("--fit-layers", default="", help="refit: layers to fit (default: all saved)")
    ap.add_argument("--part", default="", help="collect: i/n, take every n-th job")
    ap.add_argument("--skip-from", default="", help="collect: skip jobs these output dirs already have")
    ap.add_argument("--bs", type=int, default=1, help="refit: pointer minibatch size")
    ap.add_argument("--labels", default="", help="ansfit: Phase 0c argpos dirs for the tool-call labels")
    args = ap.parse_args()
    global LAYERS, PTR_BS
    PTR_BS = args.bs
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",")]
    {"collect": collect, "fit": fit, "refit": refit, "argpos": argpos, "posfit": posfit, "anspos": anspos, "anspack": anspack, "ansfit": ansfit}[args.cmd](args)


if __name__ == "__main__":
    main()
