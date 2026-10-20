"""F5 benchmarks of the object store (docs/rv-store.md).

    python -m chipformer.rv.storebench recall MODEL --cond store|host|base|context --out DIR
    python -m chipformer.rv.storebench needle MODEL --cond store|host|incontext --out DIR
        [--n 50] [--pad 0,10000,100000]

A, multi-session recall (``recall``): synthetic LongMemEval-style cases. Each case has 4 past
sessions (one user message each) that mention facts about the user; the 4th session updates 2
earlier facts. Then 3 questions, each in a new session: a fact from the first sessions, an
updated fact (the new value counts, the old one must not appear), another fact.

  store    the Qwenputer with the store tools (and calc) and MEM_SYSTEM; its store heads are
           saved after each session and loaded into the next (the question sessions start from
           an empty context)
  host     base Qwen3.5 with the same tools, system prompt and sessions; the tools run in Python
           on the host (the store's reference: the same responses token for token)
  base     base Qwen3.5, a new session with the question only
  context  base Qwen3.5 with the past sessions pasted into the question's message

B, long-context needle (``needle``): N named strings (random words and codes, ~15 tokens). The
model is told to save them (store, host: with put; incontext: they are just in the
conversation), then asked for one of them:

  pad P > 0   in the same conversation, after P tokens of unrelated chat (GSM8K questions and
              answers as user / assistant turns)
  pad 0       store, host: in a fresh session (an empty context; the store loaded)

with a plain ask ("What is the text for s17?") and a tool ask ("Use the get tool to read s17").
A long shared prefix is prefilled once and each question runs only its own tokens.

Answers are scored on Qwen's own text (the tool calls and responses removed): a fact or string
counts when it appears in it (case-insensitive for facts, exact for strings).
"""

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv import spec as S
from chipformer.rv import store as ST
from chipformer.rv.chat import CALC_TOOL, cache_restore, cache_snapshot

FW = "tests/data/rv/calc-bridge.elf"
TOOLS = [CALC_TOOL] + ST.TOOLS
MEM_SYSTEM = ST.MEM_SYSTEM
PLAIN_SYSTEM = "You are a helpful assistant."
PAD_DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "rv" / "bench_padding.json"

# --- A: multi-session recall ---------------------------------------------------------------

ATTRS = [  # (key, question, statement, update statement or None, values)
    ("city", "Which city do I live in?", "I live in {v}.",
     "Big change: I just moved to {v}, so that's home now.",
     ["Lisbon", "Porto", "Osaka", "Denver", "Tallinn", "Valencia", "Halifax", "Leeds", "Perth",
      "Krakow", "Bergen", "Austin"]),
    ("job", "What do I do for work?", "I work as a {v}.",
     "Career news: I quit my old job and I'm a {v} now.",
     ["nurse", "architect", "pastry chef", "pilot", "librarian", "welder", "pharmacist",
      "translator", "geologist", "carpenter"]),
    ("pet", "What is my dog's name?", "My dog is called {v}.",
     "Sad news, my old dog passed away. We adopted a new dog and named him {v}.",
     ["Miso", "Pickle", "Biscuit", "Juno", "Waffles", "Pepper", "Nacho", "Olive", "Ziggy",
      "Mochi"]),
    ("food", "What is my favorite food?", "My favorite food is {v}.",
     "My taste changed: my favorite food these days is {v}.",
     ["ramen", "paella", "pierogi", "tacos", "lasagna", "pho", "falafel", "risotto", "dumplings",
      "biryani"]),
    ("partner", "What is my partner's name?", "My partner's name is {v}.", None,
     ["Amara", "Tobias", "Ingrid", "Rafael", "Priya", "Mateo", "Hana", "Elias", "Noor", "Felix"]),
    ("car", "What car do I drive?", "I drive a {v}.", "I sold my old car and now I drive a {v}.",
     ["Volvo XC40", "Mazda 3", "Ford Ranger", "Honda Civic", "Subaru Outback", "Kia Niro",
      "Fiat 500", "Toyota Yaris"]),
    ("hobby", "What is my main hobby?", "My main hobby is {v}.",
     "I've got a new obsession: my main hobby is now {v}.",
     ["bouldering", "pottery", "chess", "birdwatching", "sailing", "woodworking",
      "salsa dancing", "astronomy"]),
    ("allergy", "What am I allergic to?", "I'm allergic to {v}.", None,
     ["peanuts", "shellfish", "penicillin", "cats", "pollen", "sesame", "latex"]),
    ("sister", "What is my sister's name?", "My sister is called {v}.", None,
     ["Greta", "Lucia", "Yara", "Maren", "Sofia", "Wren", "Dalia", "Leona"]),
    ("gym", "Which day of the week do I go to the gym?", "I go to the gym every {v}.",
     "I switched my gym day, it's every {v} now.",
     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
]
OPENERS = ["Hi again!", "Hey, quick hello.", "Good morning!", "Hello!", "Hey there."]
FILLERS = ["It's been a busy week.", "The weather here has been strange lately.",
           "I've been reading a lot recently.", "Work was hectic today.",
           "I finally had a quiet weekend."]
CLOSERS = ["Anyway, how are you doing?", "What's a good quick dinner idea?",
           "Any tip for sleeping better?", "Can you suggest a short book to read?",
           "What's a fun fact about octopuses?"]


def recall_cases(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        stay = rng.sample(ATTRS, 8)
        updated = [a for a in stay if a[3]][:2]
        assert len(updated) == 2
        vals = {a[0]: rng.sample(a[4], 2) for a in stay}
        order = [a for a in stay if a not in updated]
        early = updated + order[:2]  # facts of sessions 1 and 2 (the updated ones among them)
        rng.shuffle(early)
        later = order[2:6]  # sessions 3 and 4
        facts = [early[0:2], early[2:4], later[0:2], later[2:4]]
        sessions = []
        for k, fs in enumerate(facts):
            parts = [rng.choice(OPENERS), rng.choice(FILLERS)]
            parts += [a[2].format(v=vals[a[0]][0]) for a in fs]
            if k == 3:
                parts += [a[3].format(v=vals[a[0]][1]) for a in updated]
            parts.append(rng.choice(CLOSERS))
            sessions.append(re.sub(r"\ba ([aeiouAEIOU])", r"an \1", " ".join(parts)))
        static = [a for a in order if a in early] + later
        q1, q3 = static[0], rng.choice(later[1:])
        qs = [{"q": q1[1], "gold": vals[q1[0]][0], "old": None, "kind": "fact"},
              {"q": updated[0][1], "gold": vals[updated[0][0]][1],
               "old": vals[updated[0][0]][0], "kind": "update"},
              {"q": q3[1], "gold": vals[q3[0]][0], "old": None, "kind": "fact"}]
        out.append({"id": f"recall-{i}", "sessions": sessions, "questions": qs})
    return out


def score_fact(text: str, q: dict) -> bool:
    t = text.lower()
    ok = q["gold"].lower() in t
    if q["old"] is not None:
        ok = ok and q["old"].lower() not in t
    return ok


# --- B: needle -----------------------------------------------------------------------------

WORDS = ("amber falcon lantern quince ledger harbor violet orchid tundra copper maple sparrow "
         "willow cobalt meadow pewter saffron canyon ember glacier hazel juniper marble nectar "
         "onyx pepper quartz raven sienna thistle umber velvet walnut yarrow zephyr basil cedar "
         "dune fennel garnet ivory jasper kelp lilac mango nutmeg olive plum rust slate topaz")


def needle_items(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    w = WORDS.split()
    out, seen = [], set()
    while len(out) < n:
        text = (f"{rng.choice(w)} {rng.choice(w)} {rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
                f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randint(1000, 9999)} "
                f"{rng.choice(w)} {rng.randint(10, 99)}.{rng.randint(0, 9)}")
        if text not in seen:
            seen.add(text)
            out.append({"name": f"s{len(out) + 1}", "text": text})
    return out


def padding_turns(tok, n_tokens: int, seed: int = 0) -> list[dict]:
    """User / assistant turns of GSM8K problems and answers, about n_tokens long."""
    pairs = json.loads(PAD_DATA.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    out, n = [], 0
    for q, a in pairs:
        if n >= n_tokens:
            break
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        n += len(tok.encode(q + a)) + 10
    assert n >= n_tokens, "not enough padding"
    return out


def score_string(text: str, item: dict) -> bool:
    return " ".join(item["text"].split()) in " ".join(text.split())


# --- answers -------------------------------------------------------------------------------


def own_text(text: str) -> str:
    """Qwen's own words of a turn: tool calls, tool responses and think blocks removed."""
    t = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.S)
    t = re.sub(r"<tool_response>.*?</tool_response>", " ", t, flags=re.S)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
    return t.replace("<|im_end|>", "").strip()


def machine():
    m = ref.Machine.from_program(elf.program(FW))
    BR.park(m)
    return m


# --- the models ----------------------------------------------------------------------------


class StoreModel:
    """The Qwenputer (RvChat on the executor). A conversation's state: the store heads'
    snapshot and the reference device (for the lockstep check)."""

    def __init__(self, path):
        from chipformer.rv.chat import RvChat

        self.chat = RvChat(path, executor=True)
        self.tok, self.T = self.chat.tok, self.chat.T
        self.cfg = self.chat.store_cfg

    def ids(self, messages, tools):
        return self.chat.prompt_ids(messages, tools=tools)

    def make_prefix(self, ids):
        return self.chat.make_prefix(ids)

    def turn(self, messages, state, budget, tools=TOOLS, prefix=None, ids=None):
        chat = self.chat
        ids = ids or self.ids(messages, tools)
        snap, dev = state or (None, None)
        lock = ST.StoreSession(self.T, machine(), self.cfg, copy.deepcopy(dev))
        lock.history(ids)
        chat.store_snap = snap
        t0 = time.time()
        r = chat.generate(ids, machine(), budget, max_chip=200000, check_head=True, lock=lock,
                          prefix=prefix)
        text = chat.text_of(r["tokens"])
        info = {"store_calls": [a[0] for a in lock.answers], "calc_runs": len(lock.runs),
                "lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"],
                "text_tokens": r["text_tokens"], "chip_tokens": r["chip_tokens"],
                "seconds": round(time.time() - t0, 1), "prompt_tokens": len(ids)}
        return text, info, (chat.save_store(), lock.dev)


class BaseModel:
    """Base Qwen3.5 (HF, greedy). With ``host`` the store's tools run in Python: the store's
    reference is fed the prompt and Qwen's tokens and its responses are forced, token for
    token what the Qwenputer prints. A conversation's state: the reference device."""

    def __init__(self, path, dev="cuda"):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
        self.model = self.model.to(dev).eval()
        self.dev = dev
        self.T = BR.Text(Path(path) / "tokenizer.json")
        self.cfg = ST.Config()
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"),
                    self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.nq = len(self.tok)

    def ids(self, messages, tools):
        out = self.tok.apply_chat_template(messages, tools=tools or None,
                                           add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)
        return [int(x) for x in (out if isinstance(out, list) else out["input_ids"])]

    @torch.no_grad()
    def _feed(self, ids, cache, chunk=8192):
        out = None
        for a in range(0, len(ids), chunk):
            x = torch.tensor([ids[a:a + chunk]], device=self.dev)
            out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        return out

    def make_prefix(self, ids):
        out = self._feed(ids, None)
        return {"ids": list(ids), "cache": out.past_key_values,
                "snap": cache_snapshot(out.past_key_values)}

    @torch.no_grad()
    def turn(self, messages, state, budget, tools=None, prefix=None, ids=None):
        ids = ids or self.ids(messages, tools)
        host = None
        if tools:
            host = ST.StoreSession(self.T, machine(), self.cfg, copy.deepcopy(state))
            host.history(ids)
        t0 = time.time()
        if prefix is not None:
            assert ids[:len(prefix["ids"])] == prefix["ids"]
            cache = prefix["cache"]
            cache_restore(cache, prefix["snap"])
            out = self._feed(ids[len(prefix["ids"]):], cache)
        else:
            out = self._feed(ids, None)
        cache = out.past_key_values
        gen, own = [], 0
        while own < budget:
            t = int(torch.argmax(out.logits[0, -1, : self.nq]))
            gen.append(t)
            own += 1
            if t in self.eos:
                break
            feed = [t]
            if host is not None:
                k = len(host.toks)
                try:
                    host.qwen([t])
                except RuntimeError:
                    break
                forced = [x - S.TEXT_TOKEN for x in host.toks[k + 1:] if S.is_text(x)]
                gen += forced
                feed += forced
            out = self._feed(feed, cache)
            cache = out.past_key_values
        text = self.tok.decode(gen, skip_special_tokens=False)
        info = {"store_calls": [a[0] for a in host.answers] if host else [],
                "calc_runs": len(host.runs) if host else 0, "text_tokens": own,
                "seconds": round(time.time() - t0, 1), "prompt_tokens": len(ids)}
        return text, info, (host.dev if host else state)


def load_model(path, cond):
    return StoreModel(path) if cond == "store" else BaseModel(path)


# --- the runs ------------------------------------------------------------------------------


def write(out, rec):
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k not in ("text", "sessions_text")})[:400],
          flush=True)


def run_recall(args):
    cases = recall_cases(args.n, args.seed)
    out = Path(args.out) / f"recall.{args.cond}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines()} \
        if out.exists() else set()
    M = load_model(args.model, args.cond)
    tools = TOOLS if args.cond in ("store", "host") else None
    system = MEM_SYSTEM if tools else PLAIN_SYSTEM
    for case in cases:
        if all(f"{case['id']}/{j}" in done for j in range(len(case["questions"]))):
            continue
        state, sess = None, []
        if tools:  # the past sessions, the store carried from one to the next
            for s_ in case["sessions"]:
                text, info, state = M.turn([{"role": "system", "content": system},
                                            {"role": "user", "content": s_}], state,
                                           args.budget, tools)
                sess.append({"text": text, **info})
        for j, q in enumerate(case["questions"]):
            key = f"{case['id']}/{j}"
            if key in done:
                continue
            user = q["q"]
            if args.cond == "context":
                past = "\n\n".join(f"[Session {k + 1}]\nUser: {s_}"
                                   for k, s_ in enumerate(case["sessions"]))
                user = (f"Here are our past conversations:\n\n{past}\n\nNow my question: "
                        f"{q['q']}")
            text, info, _ = M.turn([{"role": "system", "content": system},
                                    {"role": "user", "content": user}], state, args.budget,
                                   tools)
            ans = own_text(text)
            write(out, {"key": key, "cond": args.cond, "kind": q["kind"], "q": q["q"],
                        "gold": q["gold"], "old": q["old"], "correct": score_fact(ans, q),
                        "answer": ans[:500], "text": text, "info": info,
                        "sessions": [{k: v for k, v in x.items() if k != "text"}
                                     for x in sess] if j == 0 else None,
                        "sessions_text": [x["text"] for x in sess] if j == 0 else None})


def run_needle(args):
    items = needle_items(args.n, args.seed)
    rng = random.Random(args.seed + 1)
    ask = rng.sample(items, min(args.ask, len(items)))
    pads = [int(p) for p in args.pad.split(",")]
    out = Path(args.out) / f"needle{args.n}.{args.cond}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines()} \
        if out.exists() else set()
    M = load_model(args.model, args.cond)
    tools = TOOLS if args.cond in ("store", "host") else None
    system = MEM_SYSTEM if tools else PLAIN_SYSTEM
    sys_m = {"role": "system", "content": system}
    # the save phase: 10 items per user message
    hist, state, save_info = [sys_m], None, []
    for a in range(0, len(items), 10):
        lines = "\n".join(f"{it['name']}: {it['text']}" for it in items[a:a + 10])
        if tools:
            msg = ("Save each of these in your memory with put, one call per item, the name and "
                   f"the text exactly as given:\n{lines}")
            hist.append({"role": "user", "content": msg})
            # each batch in a session of its own (the store carried over); a single growing
            # conversation whose earlier turns show only the replies teaches Qwen to answer
            # "saved" without calling put
            text, info, state = M.turn([sys_m, hist[-1]], state, args.budget * 4, tools)
            save_info.append(info)
            hist.append({"role": "assistant", "content": own_text(text) or "Saved."})
        else:
            hist.append({"role": "user", "content": f"Please keep these in mind:\n{lines}"})
            hist.append({"role": "assistant", "content": "Got it, I'll keep them in mind."})
    stored = None
    if tools:  # items whose exact text is in the store after the save phase
        dev = state if args.cond == "host" else state[1]
        have = {M.T.decode_bytes(dev.text(k) or []).decode("utf-8", "replace").strip()
                for k in dev.names()}
        stored = sum(it["text"] in have for it in items)
    asks = ([("plain", "What is the text for {n}? Reply with the text only."),
             ("tool", "Use the get tool to read {n}, then reply with the text only.")]
            if tools else [("plain", "What is the text for {n}? Reply with the text only.")])
    for pad in pads:
        if pad == 0 and not tools:
            continue
        base = [sys_m] if pad == 0 else hist + padding_turns(M.tok, pad, args.seed)
        prefix = None
        for kind, tmpl in asks:
            todo = [it for it in ask if f"{pad}/{kind}/{it['name']}" not in done]
            if not todo:
                continue
            if pad >= args.prefix_from and prefix is None:  # (shared by both asks)
                ids0 = [M.ids(base + [{"role": "user", "content": t_.format(n=ask[0]["name"])}],
                              tools) for _, t_ in asks]
                if len(ids0) == 1:
                    ids0.append(M.ids(base + [{"role": "user", "content": tmpl.format(
                        n=ask[1]["name"])}], tools))
                n = next(i for i, (x, y) in enumerate(zip(*ids0)) if x != y)
                t0 = time.time()
                prefix = M.make_prefix(ids0[0][:n - 2])
                print(json.dumps({"prefix_tokens": n - 2, "seconds": round(time.time() - t0)}),
                      flush=True)
            for it in todo:
                msgs = base + [{"role": "user", "content": tmpl.format(n=it["name"])}]
                text, info, _ = M.turn(msgs, state, args.budget, tools, prefix=prefix)
                ans = own_text(text)
                write(out, {"key": f"{pad}/{kind}/{it['name']}", "cond": args.cond, "pad": pad,
                            "ask": kind, "name": it["name"], "gold": it["text"],
                            "correct": score_string(ans, it), "answer": ans[:300],
                            "text": text[-1500:], "info": info, "stored": stored,
                            "save_info": save_info if not done and it is todo[0] else None})
                done.add(f"{pad}/{kind}/{it['name']}")
        prefix = None
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=("recall", "needle"))
    ap.add_argument("model")
    ap.add_argument("--cond", required=True,
                    choices=("store", "host", "base", "context", "incontext"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ask", type=int, default=50, help="needle: questions per condition")
    ap.add_argument("--pad", default="0,10000,100000")
    ap.add_argument("--prefix-from", type=int, default=1, help="needle: pads that use a prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=512)
    args = ap.parse_args()
    (run_recall if args.bench == "recall" else run_needle)(args)


if __name__ == "__main__":
    main()
