"""Executor v4 inside RvChat against executor v2 (tritonexec), on rv1 eval records.

    python -m chipformer.rv.exec4chat MODEL REC.jsonl:ID [...] [--lock] [--out OUT.jsonl]

Per record, in one process on one GPU: RvChat.generate with v2, then with v4, both without the
lockstep reference (production); the two token streams must be identical (and the v4 one is
also checked with the lockstep reference with --lock, and by rv/replay.py offline). Prints the
chip-token rate of each (chip tokens / the time spent on chip tokens, Python included) and the
whole record's wall time.
"""

import argparse
import json

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv.chat import ExecRow, RvChat


def run(chat, ids, firmware, budget, max_chip, locked):
    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    lock = None
    if locked:
        m2 = ref.Machine.from_program(elf.program(firmware))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
    return chat.generate(ids, m, budget, max_chip=max_chip, check_head=True, lock=lock)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("records", nargs="+")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-chip", type=int, default=200000)
    ap.add_argument("--lock", action="store_true", help="also v4 with the lockstep reference")
    ap.add_argument("--no-v2", action="store_true")
    ap.add_argument("--out", default=None, help="append v4's records (prompt_ids, gen_ids)")
    args = ap.parse_args()
    recs = []
    for spec in args.records:
        path, rid = spec.rsplit(":", 1)
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]
        recs.append(next(r for r in rows if r["id"] == rid))
    chat = RvChat(args.model, executor=True, exec_kind="v4")
    e4 = chat.erow
    e2 = None if args.no_v2 else ExecRow(chat)
    tot = {"v2": [0, 0.0], "v4": [0, 0.0]}
    for r in recs:
        out = {"id": r["id"]}
        res = {}
        for kind, erow in (("v2", e2), ("v4", e4)):
            if erow is None:
                continue
            chat.erow, chat.exec_kind = erow, kind
            e = run(chat, r["prompt_ids"], args.firmware, args.budget, args.max_chip, False)
            res[kind] = e
            out[kind] = {"tokens": len(e["tokens"]), "chip_tokens": e["chip_tokens"],
                         "text_tokens": e["text_tokens"], "chip_s": e["chip_seconds"],
                         "text_s": e["text_seconds"], "seconds": round(e["seconds"], 2),
                         "bursts": e.get("bursts"),
                         "chip_tok_per_s": round(e["chip_tokens"] / max(e["chip_seconds"], 1e-9)),
                         "same_as_record": e["tokens"] == r["gen_ids"]}
            tot[kind][0] += e["chip_tokens"]
            tot[kind][1] += e["chip_seconds"]
        if "v2" in res:
            a, b = res["v2"]["tokens"], res["v4"]["tokens"]
            out["v4_same_as_v2"] = a == b
            if a != b:
                out["first_diff"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                         min(len(a), len(b)))
        if args.lock:
            chat.erow, chat.exec_kind = e4, "v4"
            e = run(chat, r["prompt_ids"], args.firmware, args.budget, args.max_chip, True)
            out["v4_lock"] = {"lock_error": e["lock_error"], "head_mismatch": e["head_mismatch"],
                              "same_as_nolock": e["tokens"] == res["v4"]["tokens"]}
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": r["id"], "prompt_ids": r["prompt_ids"],
                                    "gen_ids": res["v4"]["tokens"]}) + "\n")
        print(json.dumps(out), flush=True)
    print(json.dumps({k: {"chip_tokens": v[0], "chip_s": round(v[1], 2),
                          "chip_tok_per_s": round(v[0] / max(v[1], 1e-9))}
                      for k, v in tot.items() if v[0]}), flush=True)


if __name__ == "__main__":
    main()
