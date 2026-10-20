"""Debug: where the executor / the HF decode path differ from the reference on a store stream.

    python -m chipformer.rv.storedbg MODEL [--ops 60] [--seed 1] [--hf] [--save stream.json]
"""

import argparse
import json

import torch

from chipformer.rv import core, hf
from chipformer.rv import spec as S
from chipformer.rv import storecheck as SC
from chipformer.rv.chat import RvChat


def name(chat, t):
    if t is None:
        return None
    return S.name(t - chat.v0) if t >= chat.v0 else repr(chat.tok.decode([t]))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ops", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    chat = RvChat(args.model, executor="verify")
    fw = SC.FIRMWARE
    ids = SC.prompt(chat, "Hello.")
    ses = SC.reference_stream(chat, ids, fw, args.ops, args.seed)
    n0 = len(ids)
    toks = ses.toks
    rest = SC.model_ids(chat, toks[n0:])
    want = SC.model_ids(chat, toks[n0 + 1:])
    own = ses.own[n0:len(toks) - 1]
    if args.save:
        json.dump({"prompt": ids, "rest": rest, "own": [int(o) for o in own]},
                  open(args.save, "w"))
    paths = ["exec"] + (["hf"] if args.hf else [])
    for path in paths:
        core.snapshot(chat.c, SC.machine(fw))
        chat.load_store(None)
        preds = []
        if path == "exec":
            chat.erow.start(ids)
            for t in rest:
                preds.append(chat.keep_token(chat.erow.step(t)))
        else:
            cc = chat._chip_cache()
            x = torch.tensor([ids], device=chat.dev)
            h = chat.cm.embed_tokens(x)
            for L in range(chat.n_chip):
                h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                      position_ids=None, past_key_values=cc)
            hf.inject(chat.c, cc)
            if chat.graph is None:
                chat._capture()
            g = len(ids)
            for t in rest:
                hc = chat._chip_step(t, g, cc)
                g += 1
                preds.append(chat.circuit_token(hc))
        bad = [i for i in range(len(want))
               if (preds[i] is not None) != bool(own[i]) or (own[i] and preds[i] != want[i])]
        print(json.dumps({"path": path, "bad": len(bad), "owned": int(sum(own))}), flush=True)
        for i in bad[:12]:
            ctx = [name(chat, t) for t in rest[max(0, i - 6):i + 1]]
            print(json.dumps({"i": i, "ctx": ctx, "want": name(chat, want[i]) if own[i] else None,
                              "got": name(chat, preds[i])}), flush=True)


if __name__ == "__main__":
    main()
