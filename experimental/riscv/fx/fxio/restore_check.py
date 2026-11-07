"""Qwen's state, saved and restored on the path the live loop uses (docs/fx/live.md, io.md blocker 5).

  python restore_check.py NAME MODEL_DIR RUN_JSON OUT_JSON

HfBackend on real weights, through DeltaLink's own feed. A real last request of a session is fed
with its message-boundary snapshots. For each snapshot: restore, and the logits must equal those at
the save, bit for bit; a sampled continuation from the same seed must equal the one taken before
the save was ever restored; and every tensor of the restored cache must equal the save's. The
copies stay on the GPU, as in the run.
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import delta as D  # noqa: E402
import protocol as P  # noqa: E402
import qwen_hf as QH  # noqa: E402


def tensors(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from tensors(v, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from tensors(v, seen)
    elif hasattr(obj, "__dict__"):
        yield from tensors(vars(obj), seen)


def same(a, b) -> bool:
    ta, tb = list(tensors(a)), list(tensors(b))
    return len(ta) == len(tb) and all(x.shape == y.shape and bool((x == y).all()) for x, y in zip(ta, tb))


@torch.no_grad()
def main():
    name, model_dir, run, out = sys.argv[1:]
    t = D.QwenTemplate(model_dir)
    pk = P.decode_stream(bytes.fromhex(json.load(open(run))["packets_hex"]))
    reqs = [p for p in pk if p.kind == P.REQUEST]
    _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
    ids = t.encode(t.render(msgs, tools))
    b = QH.HfBackend.load(name, temperature=1.0, top_k=0, top_p=1.0, seed=7)
    link = D.DeltaLink(t, b)
    link._feed(ids, 0)
    res = {"tokens": len(ids), "snapshots": len(link.snaps), "rows": [], "memory_allocated": torch.cuda.memory_allocated()}
    live = {}
    for s in sorted(link.snaps, reverse=True):
        cache0, logits0 = b.snaps[s]
        keep = {k: v for k, v in b.snaps.items() if k <= s}
        # the first continuation: from a fresh copy, so the saved state itself stays untouched
        b.gen.manual_seed(11)
        t0 = time.perf_counter()
        b.restore(s)
        restore_s = time.perf_counter() - t0
        logits_eq = bool((b.logits == logits0).all())
        cache_eq = same(b.cache, cache0)
        first = b.generate(24, t.stop_ids)
        b.gen.manual_seed(11)
        b.restore(s)
        second = b.generate(24, t.stop_ids)
        b.restore(s)
        res["rows"].append({"at": s, "bytes": QH.size_of(cache0), "restore_s": round(restore_s, 4), "logits_equal": logits_eq,
                            "cache_equal": cache_eq, "continuation_equal": first == second, "continuation_tokens": len(first)})
        b.snaps = dict(keep)
    res["all_equal"] = all(r["logits_equal"] and r["cache_equal"] and r["continuation_equal"] for r in res["rows"])
    res["seconds"] = b.seconds
    res["peak"] = torch.cuda.max_memory_allocated()
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
