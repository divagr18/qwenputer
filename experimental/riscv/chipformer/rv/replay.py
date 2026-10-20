"""Offline IR replay of rv1 eval records (no model needed).

    python -m chipformer.rv.replay MODEL_DIR records.jsonl [--firmware ELF] [--limit N]

Each record's prompt and generated ids (evalarith stores them) become the circuit's token
stream; the reference machine (fed the prompt as history and Qwen's own tokens) must produce the
same chip tokens and echoes, and the IR circuit must own exactly the reference's positions and
predict every token there.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from chipformer.q35 import ir
from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S


def replay_ids(c, T, v0, prompt, gen, firmware, store_dev=None, store_init=None) -> dict:
    """``store_dev`` / ``store_init``: the store's reference state and the matching head
    states (chat.save_store) the record started from; empty when None."""
    import copy

    stream = [i - v0 if i >= v0 else BR.text_token(i) for i in prompt + gen]
    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    if c.store is not None:
        from chipformer.rv import store as ST

        ses = ST.StoreSession(T, m, c.store.cfg, copy.deepcopy(store_dev))
    else:
        ses = BR.Session(T, m)
    ses.history(prompt)
    skip = {x for x in range(len(prompt)) if prompt[x] == T.TCC}
    pos = len(prompt)
    while pos < len(stream):
        t = stream[pos]
        if not S.is_text(t):
            return {"ok": False, "where": pos, "error": "chip token without a run"}
        k = len(ses.toks)
        ses.qwen([t - S.TEXT_TOKEN])
        pos += 1
        added = ses.toks[k + 1:]
        if added:
            got = stream[pos:pos + len(added)]
            if got != added[:len(got)]:
                return {"ok": False, "where": pos, "error": "stream differs from the reference"}
            pos += len(added)
    toks = ses.toks[:len(stream)]
    assert toks == stream
    m2 = ref.Machine.from_program(elf.program(firmware))
    BR.park(m2)
    core.snapshot(c, m2)
    for name, n in (c.store.persistent.items() if c.store is not None else ()):
        n.init = None if store_init is None else store_init.get(name)
    carry, preds, owns = {}, [], []  # per chunk: a long stream's whole matrix is GBs
    for a in range(0, len(stream), 2048):
        mat = ir.evaluate(c, stream[a:a + 2048], carry=carry)
        preds.append(core.predictions(c, mat))
        owns.append(np.rint(ir.ev(c.owner, mat)).astype(int))
    pred, own = np.concatenate(preds), np.concatenate(owns)
    bad_own = int(sum(own[x] != int(ses.own[x]) for x in range(len(stream))))
    bad = int(sum(1 for x in range(len(stream) - 1)
                  if ses.own[x] and x not in skip and pred[x] != stream[x + 1]))
    return {"ok": bad == 0 and bad_own == 0, "positions": len(stream),
            "owned": int(sum(ses.own)) - len(skip), "wrong_predictions": bad,
            "owner_mismatch": bad_own, "runs": len(ses.runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("records")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--limit", type=int, default=10**9)
    args = ap.parse_args()
    meta = json.loads((Path(args.model) / "chipformer.json").read_text())
    T = BR.Text(Path(args.model) / "tokenizer.json")
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    store = None
    if meta.get("store"):
        from chipformer.rv.store import Config

        store = Config(**meta["store"])
    c = core.build(Target(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
                          cfg["intermediate_size"]), T, store)
    tot = {"records": 0, "ok": 0, "positions": 0, "owned": 0}
    for line in Path(args.records).read_text().splitlines()[: args.limit]:
        r = json.loads(line)
        if "gen_ids" not in r:
            continue
        res = replay_ids(c, T, meta["chip_token_offset"], r["prompt_ids"], r["gen_ids"],
                         args.firmware)
        print(r["id"], json.dumps(res), flush=True)
        tot["records"] += 1
        tot["ok"] += res["ok"]
        tot["positions"] += res.get("positions", 0)
        tot["owned"] += res.get("owned", 0)
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
