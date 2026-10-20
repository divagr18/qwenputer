"""Tables of the F5 benchmarks (storebench records).

    python -m chipformer.rv.benchsum DIR [DIR ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(dirs):
    rows = defaultdict(list)
    for d in dirs:
        for f in Path(d).glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rows[f.stem].append(r)
    return rows


def pct(xs):
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def main() -> None:
    rows = load(sys.argv[1:])
    print("A. multi-session recall")
    print("| condition | all | facts | updated facts | lock errors | tool calls / question |")
    print("|---|---|---|---|---|---|")
    for cond in ("base", "store", "host", "context"):
        rs = rows.get(f"recall.{cond}", [])
        if not rs:
            continue
        lock = sum(1 for r in rs if r["info"].get("lock_error"))
        lock += sum(1 for r in rs for s in (r.get("sessions") or []) if s.get("lock_error"))
        calls = sum(len(r["info"]["store_calls"]) for r in rs) / max(len(rs), 1)
        print(f"| {cond} | {pct([r['correct'] for r in rs])} | "
              f"{pct([r['correct'] for r in rs if r['kind'] == 'fact'])} | "
              f"{pct([r['correct'] for r in rs if r['kind'] == 'update'])} | {lock} | "
              f"{calls:.1f} |")
    for cond in ("store", "host"):
        rs = rows.get(f"recall.{cond}", [])
        ss = [s for r in rs for s in (r.get("sessions") or [])]
        if ss:
            n_put = sum(c in ("put", "append") for s in ss for c in s["store_calls"])
            no_save = sum(not any(c in ("put", "append") for c in s["store_calls"]) for s in ss)
            secs = sum(s["seconds"] for s in ss) + sum(r["info"]["seconds"] for r in rs)
            print(f"  {cond}: {len(ss)} past sessions, {n_put} put/append calls, {no_save} "
                  f"sessions with no save; {secs / 60:.1f} min of generation")
    print()
    print("B. needle")
    for stem in sorted(k for k in rows if k.startswith("needle")):
        rs = rows[stem]
        stored = next((r["stored"] for r in rs if r.get("stored") is not None), None)
        by = defaultdict(list)
        for r in rs:
            by[(r["pad"], r["ask"])].append(r)
        lock = sum(1 for r in rs if r["info"].get("lock_error"))
        print(f"{stem}: stored {stored}, lock errors {lock}")
        for (pad, ask), xs in sorted(by.items()):
            calls = sum(len(r["info"]["store_calls"]) > 0 for r in xs)
            secs = sum(r["info"]["seconds"] for r in xs) / len(xs)
            print(f"  pad {pad:>6} {ask:5}: {pct([r['correct'] for r in xs])} "
                  f"(called a tool {calls}/{len(xs)}, {secs:.1f} s per question)")


if __name__ == "__main__":
    main()
