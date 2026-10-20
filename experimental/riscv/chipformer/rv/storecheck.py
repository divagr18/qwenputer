"""GPU checks of a merged checkpoint with the object store (merge --store).

    python -m chipformer.rv.storecheck MODEL [--exact] [--live] [--roundtrip] [--batch B]
        [--out res.json]

--exact      teacher-forced: a reference stream of random store calls with random tokens (and
             calc calls) goes through the chip layers (the HF modules as one chunked prefill,
             and the executor token by token); the circuit's token at every position it owns
             must be the reference's, and it must own exactly those positions. The store
             heads' states after the stream must be the same on both paths.
--live       Qwen decides: prompts that ask it to save, append, read, list and delete; the HF
             chip path and the executor run side by side ("verify": their residuals compared
             bitwise), every circuit token checked in lockstep against the reference, then the
             executor alone must give the same tokens, and the stream is replayed in the IR.
--roundtrip  session 1 stores facts; its store heads are saved (save_store); session 2 starts
             from an empty context with them loaded (load_store) and reads them back (lock
             against the reference resumed with session 1's store, IR replay from the snapshot).
--batch B    the live prompts decoded B at a time (rvbatch), together and each alone in the
             B-row decoder: the same tokens either way, no lock errors.
"""

import argparse
import copy
import json
import random
import time

import numpy as np
import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S
from chipformer.rv import store as ST
from chipformer.rv.chat import CALC_TOOL, RvChat
from chipformer.rv.replay import replay_ids

FIRMWARE = "tests/data/rv/calc-bridge.elf"
SYSTEM = ("You have a memory that keeps texts across conversations, with the tools put, append, "
          "get, list and delete, and a calc tool for exact arithmetic. When the user asks you to "
          "remember, change or recall something, use the memory tools.")
TOOLS = [CALC_TOOL] + ST.TOOLS
LIVE = [
    "Remember this under the name launch: the launch code is 4471-AZ.",
    "Save my grocery list (eggs, milk, bread) under the name groceries. Then append butter to "
    "it, and read it back to me with get.",
    "Store 'alpha' under the name a and 'beta' under the name b, then list the stored names.",
    "Save the text 'temporary data' under the name tmp, then delete tmp, then try to get tmp.",
    "What is 123456789 * 987654321? Save the result under the name product.",
    "Put 'The meeting moved to Thursday 3pm, room 4B.' under the name meeting, then get it.",
]
FACTS = [("city", "I live in Lisbon."), ("pet", "My cat is called Miso and is 4 years old."),
         ("code", "The vault code is 7731-QX-90."), ("plan", "Dentist on May 3, then a train to "
                                                            "Porto at 18:40.")]


def machine(firmware):
    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    return m


def prompt(chat, user, system=SYSTEM):
    return chat.prompt_ids([{"role": "system", "content": system},
                            {"role": "user", "content": user}], tools=TOOLS)


def lock_of(chat, ids, firmware, dev=None):
    lock = ST.StoreSession(chat.T, machine(firmware), chat.store_cfg, copy.deepcopy(dev))
    lock.history(ids)
    return lock


def model_ids(chat, toks):
    return [t + chat.v0 if t < S.VOCAB else t - S.TEXT_TOKEN for t in toks]


# --- teacher-forced -------------------------------------------------------------------------


def reference_stream(chat, ids, firmware, n_ops, seed):
    """A StoreSession after the prompt (history): random store calls with random tokens, calc
    calls and Qwen text between them."""
    rng = random.Random(seed)
    T = chat.T
    ses = lock_of(chat, ids, firmware)
    base = [[rng.randrange(ST.SPECIAL) for _ in range(rng.randint(1, 3))] for _ in range(8)]
    names = base + [base[0] + [rng.randrange(ST.SPECIAL)], base[1][:1] * 7 + [5],
                    base[1][:1] * 7 + [6]]
    names += [T.encode(w) for w in ("notes", "user 7", "plan/today")]
    for i in range(n_ops):
        op = rng.choice(["put", "put", "append", "append", "get", "get", "delete", "list"])
        nm = rng.choice(names)
        k = rng.choice([0, 1, 3, 10, 40, ST.CMAX - 1, ST.CMAX, ST.CMAX + 9])
        text = [q for q in (rng.randrange(ST.SPECIAL + 200) for _ in range(k))
                if q not in (ST.Q_TCO, ST.Q_TCC)]
        if op == "list":
            ses.call("list")
        elif op in ("get", "delete"):
            ses.call(op, nm)
        else:
            ses.call(op, nm, text)
        ses.qwen(T.encode(rng.choice([" OK.", " Done, saved.", "\nNext: ", " Hmm, "])))
        if i % 9 == 4:
            ses.qwen(T.encode(f"<tool_call>\n<function=calc>\n<parameter=expression>\n"
                              f"{rng.randint(2, 99999)}*{rng.randint(2, 99999)}\n</parameter>\n"
                              f"</function>\n</tool_call>"))
    return ses


@torch.no_grad()
def exact(chat, firmware, n_ops=60, seed=1):
    ids = prompt(chat, "Hello.")
    ses = reference_stream(chat, ids, firmware, n_ops, seed)
    n0 = len(ids)
    toks = ses.toks
    rest = model_ids(chat, toks[n0:])
    want = [toks[i + 1] for i in range(n0, len(toks) - 1)]
    own = ses.own[n0:len(toks) - 1]
    out = {"positions": len(rest), "owned": int(sum(own)), "calls": len(ses.answers),
           "calc_runs": len(ses.runs)}
    # the HF chip layers: the prompt, the snapshot (an empty store), the rest as a prefill
    core.snapshot(chat.c, machine(firmware))
    chat.load_store(None)
    cc = chat._chip_cache()
    x = torch.tensor([ids], device=chat.dev)
    h = chat.cm.embed_tokens(x)
    for L in range(chat.n_chip):
        h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                              position_ids=None, past_key_values=cc)
    hf.inject(chat.c, cc)
    if chat.graph is None:
        chat._capture()
    preds_hf, g = [], len(ids)
    for t in rest:  # token by token, as in generation (the decode kernel)
        preds_hf.append(chat.circuit_token(chat._chip_step(t, g, cc)))
        g += 1
    snap_hf = chat.save_store(("hf", cc))
    # (for information: the same stream as chunked prefills, which generation never runs on the
    # circuit's state: the prompt's prefill comes before the snapshot is injected)
    core.snapshot(chat.c, machine(firmware))
    chat.load_store(None)
    cc = chat._chip_cache()
    h = chat.cm.embed_tokens(x)
    for L in range(chat.n_chip):
        h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                              position_ids=None, past_key_values=cc)
    hf.inject(chat.c, cc)
    preds_pf = []
    for a in range(0, len(rest), 1024):
        xx = torch.tensor([rest[a:a + 1024]], device=chat.dev)
        h = chat.cm.embed_tokens(xx)
        for L in range(chat.n_chip):
            h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=cc)
        for t in range(h.shape[1]):
            preds_pf.append(chat.circuit_token(h[0, t]))
    # the executor, token by token
    core.snapshot(chat.c, machine(firmware))
    chat.load_store(None)
    chat.erow.start(ids)
    preds_ex = []
    for t in rest:
        preds_ex.append(chat.keep_token(chat.erow.step(t)))
    snap_ex = chat.save_store(("exec", chat.erow))

    def score(preds):
        wrong = owner = 0
        for i in range(len(want)):
            p = preds[i]
            if (p is not None) != bool(own[i]):
                owner += 1
            elif p is not None:
                w = want[i]
                wid = w + chat.v0 if w < S.VOCAB else w - S.TEXT_TOKEN
                wrong += p != wid
        return {"wrong": wrong, "owner_mismatch": owner}

    out["hf"], out["exec"] = score(preds_hf), score(preds_ex)
    out["hf_prefill_info"] = score(preds_pf)
    same = set(snap_hf) == set(snap_ex) and all(
        len(snap_hf[k]) == len(snap_ex[k]) and all(
            set(a) == set(b) and all(np.array_equal(a[r], b[r]) for r in a)
            for a, b in zip(snap_hf[k], snap_ex[k])) for k in snap_hf)
    out["store_state_hf_eq_exec"] = bool(same)
    out["store_rows"] = int(sum(len(v) for vs in snap_ex.values() for v in vs))
    return out


# --- live ----------------------------------------------------------------------------------


def run_live(chat, user, firmware, budget, dev=None, snap=None, system=SYSTEM):
    ids = prompt(chat, user, system)
    lock = lock_of(chat, ids, firmware, dev)
    chat.store_snap = snap
    r = chat.generate(ids, machine(firmware), budget, check_head=True, lock=lock)
    r["ids"], r["lock"] = ids, lock
    return r


def summary(chat, r, firmware, dev=None, snap=None):
    rp = replay_ids(chat.c, chat.T, chat.v0, r["ids"], r["tokens"], firmware, dev, snap)
    calls = [a[0] for a in r["lock"].answers]
    return {"text": chat.text_of(r["tokens"])[-600:], "store_calls": calls,
            "calc_runs": len(r["lock"].runs), "lock_error": r["lock_error"],
            "head_mismatch": r["head_mismatch"], "verify_mismatch": r["verify_mismatch"],
            "chip_tokens": r["chip_tokens"], "text_tokens": r["text_tokens"],
            "replay": {k: rp.get(k) for k in ("ok", "owned", "wrong_predictions",
                                              "owner_mismatch", "error")},
            "printed": r["lock"].printed()}


def live(chat, firmware, budget):
    res = []
    for user in LIVE:
        chat.verify, chat.fast = True, True
        rv = run_live(chat, user, firmware, budget)
        chat.verify, chat.fast = False, False
        re_ = run_live(chat, user, firmware, budget)
        s = summary(chat, rv, firmware)
        s["exec_same_tokens"] = re_["tokens"] == rv["tokens"]
        s["exec_lock_error"] = re_["lock_error"]
        res.append({"user": user, **s})
        print(json.dumps({k: res[-1][k] for k in ("store_calls", "lock_error", "verify_mismatch",
                                                   "exec_same_tokens", "replay")}), flush=True)
    return res


def roundtrip(chat, firmware, budget):
    """Session 1 saves FACTS; session 2 (empty context, the store loaded) reads them back."""
    out = {}
    chat.verify, chat.fast = False, False
    user = "Remember these, each with put under its name:\n" + "\n".join(
        f"- {n}: {t}" for n, t in FACTS)
    r1 = run_live(chat, user, firmware, budget)
    snap = chat.save_store()
    dev1 = r1["lock"].dev
    out["session1"] = summary(chat, r1, firmware)
    out["stored"] = {chat.T.decode_bytes(list(k[1][:min(k[0], ST.NMAX)])).decode("utf-8",
                                                                                   "replace"):
                     chat.T.decode_bytes(dev1.text(k) or []).decode("utf-8", "replace")
                     for k in dev1.names()}
    out["snapshot_rows"] = int(sum(len(v) for vs in snap.values() for v in vs))
    s2 = []
    asks = ["Use the get tool to read the memory named code and tell me what it says.",
            "List what is in your memory, then get plan and tell me when my train leaves.",
            "What is my cat called? Look in your memory (get pet)."]
    for ask in asks:
        r2 = run_live(chat, ask, firmware, budget, dev=dev1, snap=snap)
        s = summary(chat, r2, firmware, dev1, snap)
        s2.append({"user": ask, **s})
        print(json.dumps({k: s[k] for k in ("store_calls", "lock_error", "replay", "printed")}),
              flush=True)
    out["session2"] = s2
    chat.store_snap = None
    return out


def batch_check(chat, firmware, budget, B):
    from chipformer.rv.rvbatch import Job, RvBatch

    chat.verify, chat.fast = False, False
    users = LIVE[:B] if len(LIVE) >= B else LIVE

    def job(u):
        ids = prompt(chat, u)
        return Job(u, ids, machine(firmware), lock_of(chat, ids, firmware))

    rb = RvBatch(chat, B, 6144)
    together, alone = {}, {}
    rb.run((job(u) for u in users), budget, 200000, lambda j, o: together.__setitem__(j.key, o))
    for u in users:
        rb.run([job(u)], budget, 200000, lambda j, o: alone.__setitem__(j.key, o))
    return [{"user": u, "same_tokens": together[u]["tokens"] == alone[u]["tokens"],
             "lock_error": [together[u]["lock_error"], alone[u]["lock_error"]],
             "head_mismatch": [together[u]["head_mismatch"], alone[u]["head_mismatch"]],
             "len": len(together[u]["tokens"])} for u in users]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default=FIRMWARE)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--ops", type=int, default=60)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--out", default="storecheck.json")
    args = ap.parse_args()
    chat = RvChat(args.model, executor="verify")
    assert chat.store_cfg is not None, "not a store checkpoint"
    res = {}
    t0 = time.time()
    if args.exact:
        res["exact"] = exact(chat, args.firmware, args.ops)
        print(json.dumps(res["exact"]), flush=True)
    if args.live:
        res["live"] = live(chat, args.firmware, args.budget)
    if args.roundtrip:
        res["roundtrip"] = roundtrip(chat, args.firmware, args.budget)
    if args.batch:
        res["batch"] = batch_check(chat, args.firmware, args.budget, args.batch)
        print(json.dumps(res["batch"]), flush=True)
    res["seconds"] = round(time.time() - t0)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
