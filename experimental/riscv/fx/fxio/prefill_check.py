"""RvChat's text path on real Qwen3.5 weights (docs/fx/io.md): chunked prefill, the token-at-a-time
step, and the model's own forward on the same render, with an RvChat shell around the base model.

  python prefill_check.py MODEL_NAME MODEL_DIR RUN_JSON OUT_JSON

n_chip = 0: the base model has no chip layers, so this checks _qwen_prefill, _qwen_step, _text_hc
(its non-executor branch) and FxQwen's snapshots. The executor branch needs a merged checkpoint.
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RISCV = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, RISCV]
import delta as D  # noqa: E402
import protocol as P  # noqa: E402
import qwen_hf as QH  # noqa: E402


def compare(a, b):
    a, b = a.float(), b.float()
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return {"max_abs": float((a - b).abs().max()), "top1_equal": int(a.argmax()) == int(b.argmax()),
            "top10_equal": a.topk(10).indices.tolist() == b.topk(10).indices.tolist(),
            "kl": float((pa.exp() * (pa - pb)).sum())}


@torch.no_grad()
def main():
    name, model_dir, run, out = sys.argv[1:]
    from transformers import AutoModelForCausalLM, DynamicCache

    from chipformer.rv.chat import FxQwen, RvChat

    t = D.QwenTemplate(model_dir)
    pk = P.decode_stream(bytes.fromhex(json.load(open(run))["packets_hex"]))
    reqs = [p for p in pk if p.kind == P.REQUEST]
    _, msgs, tools = D.request_messages(json.loads(P.parse_request(reqs[-1])[2]))
    ids = t.encode(t.render(msgs, tools))
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    c = RvChat.__new__(RvChat)
    c.model, c.dev, c.executor, c.n_chip = m, "cuda", False, 0
    c.lm = getattr(m.model, "language_model", m.model)
    c.v0 = m.get_output_embeddings().weight.shape[0]
    res = {"model": name, "class": type(m).__name__, "prompt_tokens": len(ids), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "memory_after_load": torch.cuda.memory_allocated()}

    def timed(f):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = f()
        torch.cuda.synchronize()
        return r, time.perf_counter() - t0

    full, res["forward_s"] = timed(lambda: m(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1])

    def prefill(chunk):
        b = FxQwen(c, chunk=chunk)
        b.feed(ids)
        return b

    b, res["prefill_s"] = timed(lambda: prefill(2048))
    res["prefill_tok_s"] = len(ids) / res["prefill_s"]
    res["prefill_vs_forward"] = compare(b.logits, full)
    b2, _ = timed(lambda: prefill(64))
    res["prefill64_vs_prefill"] = compare(b2.logits, b.logits)

    def one():
        cache, r = DynamicCache(config=c.lm.config), None
        for i, x in enumerate(ids):
            r = c._qwen_step(c._text_hc([x])[0], cache, i)
        return r

    step, res["token_by_token_s"] = timed(one)
    res["ms_per_token_step"] = 1000 * res["token_by_token_s"] / len(ids)
    res["step_vs_prefill"] = compare(step, b.logits)

    # the same greedy continuation from each path, and from a restored snapshot
    def greedy(q, n=64):
        q.forced = None
        return q.generate(n, set())

    a = prefill(2048)
    a.snapshot(a.pos)
    g1, res["generate_64_s"] = timed(lambda: greedy(a))
    res["ms_per_token_generate"] = 1000 * res["generate_64_s"] / 64
    a.restore(len(ids))
    g2 = greedy(a)
    g3 = greedy(prefill(64))
    res["greedy_after_restore_equal"] = g1 == g2
    res["greedy_chunk64_equal"] = g1 == g3
    res["greedy_chunk64_common"] = D.common_prefix(g1, g3)
    res["snapshot_bytes"] = QH.size_of(a.cache)
    res["memory_peak"] = torch.cuda.max_memory_allocated()
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
