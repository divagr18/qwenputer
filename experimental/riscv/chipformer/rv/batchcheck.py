"""Batched rv1 decoding (rvbatch) against batch-1 records: the same prompts, decoded N at a time,
must give each record's tokens.

    python -m chipformer.rv.batchcheck MODEL REC.jsonl:ID [...] --batch 4 [--batch 8 ...]
"""

import argparse
import json
import time

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv.chat import RvChat
from chipformer.rv.rvbatch import Job, RvBatch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("records", nargs="+")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--batch", type=int, action="append", required=True)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-chip", type=int, default=200000)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--local-ref", action="store_true",
                    help="compare with RvChat.generate here instead of the records' tokens")
    ap.add_argument("--exec", default="v4", choices=("v2", "v4"), help="the executor")
    ap.add_argument("--dump", default=None, help="append every batched result's tokens here")
    args = ap.parse_args()
    recs = []
    for spec in args.records:
        path, rid = spec.rsplit(":", 1)
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]
        recs.append(next(r for r in rows if r["id"] == rid))
    chat = RvChat(args.model, executor=True, exec_kind=args.exec)

    def job(r):
        m = ref.Machine.from_program(elf.program(args.firmware))
        BR.park(m)
        m2 = ref.Machine.from_program(elf.program(args.firmware))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(r["prompt_ids"])
        return Job(r["id"], r["prompt_ids"], m, lock, {"r": r})

    if args.local_ref:  # the reference: RvChat.generate (batch 1) on this GPU
        for r in recs:
            j = job(r)
            t0 = time.time()
            out = chat.generate(j.ids, j.machine, args.budget, max_chip=args.max_chip,
                                check_head=True, lock=j.lock)
            same = out["tokens"] == r["gen_ids"]
            r["gen_ids"] = out["tokens"]
            print(json.dumps({"batch": "generate", "id": r["id"], "same_as_record": same,
                              "len": len(out["tokens"]), "seconds": round(time.time() - t0, 1),
                              "lock_error": out["lock_error"]}), flush=True)
    for B in args.batch:
        rb = RvBatch(chat, B, args.max_len)
        res = {}
        t0 = time.time()
        st = rb.run((job(r) for r in recs), args.budget, args.max_chip,
                    lambda j, out: res.__setitem__(j.key, out))
        wall = time.time() - t0
        if args.dump:
            with open(args.dump, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps({"exec": args.exec, "batch": B, "id": r["id"],
                                        "tokens": res[r["id"]]["tokens"]}) + "\n")
        for r in recs:
            got, want = res[r["id"]]["tokens"], r["gen_ids"]
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         None if len(got) == len(want) else min(len(got), len(want)))
            print(json.dumps({"batch": B, "id": r["id"], "same_tokens": got == want,
                              "first_diff": first, "len": len(got), "record_len": len(want),
                              "lock_error": res[r["id"]]["lock_error"],
                              "head_mismatch": res[r["id"]]["head_mismatch"],
                              "diff_ctx": chat.tok.decode([t for t in want[max(0, (first or 0) - 8):(first or 0) + 4] if t < chat.v0]) if first is not None else ""}),
                  flush=True)
        print(json.dumps({"batch": B, "wall_s": round(wall, 1), "stats": st,
                          "problems_per_hour": round(len(recs) / wall * 3600, 1)}), flush=True)
        del rb


if __name__ == "__main__":
    main()
