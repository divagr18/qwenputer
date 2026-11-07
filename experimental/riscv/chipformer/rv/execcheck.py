"""The executor path of RvChat against the HF chip path, on rv1 eval records.

    python -m chipformer.rv.execcheck MODEL REC.jsonl:ID [REC.jsonl:ID ...] [--no-verify]

Per record: (1) "verify" mode: the HF chip view drives generation (it must reproduce the
record's tokens) and the executor runs beside it, its residual after the chip layers compared
bitwise at every position (the whole hc on text tokens, the kept dims on chip tokens); (2) the
executor alone: its tokens must equal the record's. Prints both timings.
"""

import argparse
import json

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv.chat import RvChat


def run(chat, r, firmware, budget, max_chip, locked=True):
    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    lock = None
    if locked:
        m2 = ref.Machine.from_program(elf.program(firmware))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(r["prompt_ids"])
    return chat.generate(r["prompt_ids"], m, budget, max_chip=max_chip, check_head=True, lock=lock)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("records", nargs="+")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--max-chip", type=int, default=200000)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--exec", default="v4", choices=("v2", "v4"), help="the executor")
    ap.add_argument("--no-lock", action="store_true",
                    help="also decode without the lockstep reference (production speed)")
    args = ap.parse_args()
    recs = []
    for spec in args.records:
        path, rid = spec.rsplit(":", 1)
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]
        recs.append(next(r for r in rows if r["id"] == rid))
    chat = RvChat(args.model, executor="verify", exec_kind=args.exec)
    for r in recs:
        out = {"id": r["id"], "record_tokens": len(r["gen_ids"])}
        if not args.no_verify:
            chat.verify, chat.fast = True, True
            v = run(chat, r, args.firmware, args.budget, args.max_chip)
            out["hf"] = {"same_tokens": v["tokens"] == r["gen_ids"],
                         "verify_mismatch": v["verify_mismatch"], "lock_error": v["lock_error"],
                         "chip_tokens": v["chip_tokens"], "chip_s": v["chip_seconds"],
                         "text_s": v["text_seconds"]}
        chat.verify, chat.fast = False, False
        e = run(chat, r, args.firmware, args.budget, args.max_chip)
        first = next((i for i, (a, b) in enumerate(zip(e["tokens"], r["gen_ids"])) if a != b),
                     None if len(e["tokens"]) == len(r["gen_ids"]) else
                     min(len(e["tokens"]), len(r["gen_ids"])))
        out["exec"] = {"same_tokens": e["tokens"] == r["gen_ids"], "first_diff": first,
                       "lock_error": e["lock_error"], "head_mismatch": e["head_mismatch"],
                       "chip_tokens": e["chip_tokens"], "chip_s": e["chip_seconds"],
                       "text_s": e["text_seconds"], "seconds": round(e["seconds"], 1),
                       "chip_tok_per_s": round(e["chip_tokens"] / max(e["chip_seconds"], 1e-9))}
        if args.no_lock:  # production: no reference machine beside the decoder
            e = run(chat, r, args.firmware, args.budget, args.max_chip, locked=False)
            out["nolock"] = {"same_tokens": e["tokens"] == r["gen_ids"],
                             "chip_tokens": e["chip_tokens"], "chip_s": e["chip_seconds"],
                             "text_s": e["text_seconds"], "seconds": round(e["seconds"], 1),
                             "bursts": e.get("bursts"),
                             "chip_tok_per_s": round(e["chip_tokens"]
                                                     / max(e["chip_seconds"], 1e-9))}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
