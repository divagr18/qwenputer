"""The object store v2 on the RV core (rv/store.py, rv/store_rv.py): the reference against a
plain filesystem, and the circuit against the reference token by token in the IR (every bf16
rounding checked): all ops, forced hash collisions, tombstones, appends, names equal in their
first NMAX tokens, full stores, calc calls in between, a restored snapshot."""

import random
from pathlib import Path

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv import store as ST

CALC = Path(__file__).parent / "data" / "rv" / "calc-bridge.elf"
SMALL = ST.Config(slot_bits=4, n_chunks=16, n_names=8, hash_bits=2)  # collisions everywhere
TINY_FULL = ST.Config(slot_bits=4, n_chunks=8, n_names=4, hash_bits=1)


@pytest.fixture(scope="module")
def T():
    try:
        return BR.Text(BR.default_tokenizer())
    except FileNotFoundError:
        pytest.skip("no Qwen3.5 tokenizer in the HF cache")


_CIRC: dict = {}


def circuit(T, cfg):
    if cfg not in _CIRC:
        _CIRC[cfg] = core.build(text=T, store=cfg)
    return _CIRC[cfg]


def fresh_machine():
    m = ref.Machine.from_program(elf.program(str(CALC)))
    BR.park(m)
    return m


def session(T, cfg, dev=None):
    return ST.StoreSession(T, fresh_machine(), cfg, dev)


def calc_call(T, expr: str) -> list[int]:
    return T.encode(f"<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n"
                    "</parameter>\n</function>\n</tool_call>")


def tname(T, t):
    return S.name(t) if t < S.VOCAB else repr(T.tok.decode([t - S.TEXT_TOKEN]))


def run_ir(c, toks, init=None):
    """IR residuals of a stream (chunked, state carried), from the machine's snapshot."""
    core.snapshot(c, fresh_machine())
    for name, node in c.store.persistent.items():
        node.init = None if init is None else init.get(name)
    carry, mats = {}, []
    for a in range(0, len(toks), 2048):
        mats.append(ir.evaluate(c, toks[a:a + 2048], carry=carry))
    return np.concatenate(mats), carry


def check(T, c, ses, init=None, skip=frozenset()):
    """IR == reference: the owner at every position, the next token wherever the circuit owns
    it. Returns the IR's carried state (for snapshots) and the number of owned positions."""
    toks = ses.toks
    mat, carry = run_ir(c, toks, init)
    pred = core.predictions(c, mat)
    own = np.rint(ir.ev(c.owner, mat)).astype(int)
    bo = [i for i in range(len(toks)) if own[i] != int(ses.own[i])]
    assert bo == [], [(i, tname(T, toks[i]), own[i]) for i in bo[:6]]
    bad = [i for i in range(len(toks) - 1)
           if ses.own[i] and i not in skip and pred[i] != toks[i + 1]]
    assert bad == [], [(i, tname(T, toks[i]), tname(T, pred[i]) if pred[i] >= 0 else pred[i],
                        tname(T, toks[i + 1])) for i in bad[:8]]
    return carry, sum(ses.own)


def snapshot(c, carry):
    """The persistent heads' states from the IR's carry: what store_snapshot saves."""
    idx = {id(n): i for i, n in enumerate(c.nodes)}
    out = {}
    for name, node in c.store.persistent.items():
        ni = idx[id(node)]
        out[name] = [{r: row.copy() for r, row in carry.get((ni, "S", x), {}).items()}
                     for x in range(len(node.vheads()))]
    return out


# --- ids and the reference -------------------------------------------------------------------


def test_token_ids(T):
    one = {s: T.encode(s) for s in ["\n", "=", "function", "parameter", "</", "<", ">", "ok",
                                    "not", " found", "empty", "full"]}
    assert one["\n"] == [ST.Q_NL] and one["="] == [ST.Q_EQ]
    assert one["function"] == [ST.Q_FUNCTION] and one["parameter"] == [ST.Q_PARAMETER]
    assert one["</"] == [ST.Q_LTSLASH] and one["<"] == [ST.Q_LT] and one[">"] == [ST.Q_GT]
    assert one["ok"] == ST.MSG["ok"] and one["not"] + one[" found"] == ST.MSG["not_found"]
    assert one["empty"] == ST.MSG["empty"] and one["full"] == ST.MSG["full"]
    ids = T.tok.get_vocab(with_added_tokens=True)
    for s, q in (("<tool_call>", ST.Q_TCO), ("</tool_call>", ST.Q_TCC),
                 ("<tool_response>", ST.Q_TRO), ("</tool_response>", ST.Q_TRC),
                 ("<think>", ST.Q_THO), ("</think>", ST.Q_THC), ("<|endoftext|>", ST.SPECIAL)):
        assert ids[s] == q
    for op, q in ST.OP_BARE.items():
        assert T.encode(op) == [q]
    for op, q in ST.OP_JOINED.items():
        assert T.encode("=" + op) == [q]
    for p, q in ST.PARAM_BARE.items():
        assert T.encode(p) == [q]
    for p, q in ST.PARAM_JOINED.items():
        assert T.encode("=" + p) == [q]
    text = ("<tool_call>\n<function=put>\n<parameter=name>\nnotes\n</parameter>\n"
            "<parameter=text>\nhi there\n</parameter>\n</function>\n</tool_call>")
    assert ST.StoreSession.call_tokens("put", T.encode("notes"), T.encode("hi there")) == \
        T.encode(text)
    for op in ("get", "list", "append", "delete"):
        name = None if op == "list" else T.encode("a")
        txt = T.encode("b") if op == "append" else None
        s_ = f"<tool_call>\n<function={op}>\n"
        if name:
            s_ += "<parameter=name>\na\n</parameter>\n"
        if txt:
            s_ += "<parameter=text>\nb\n</parameter>\n"
        s_ += "</function>\n</tool_call>"
        assert ST.StoreSession.call_tokens(op, name, txt) == T.encode(s_), op
    # the tail is the calc tool's (bridge.TOOL_TAIL)
    assert T.decode_bytes(ST.TAIL).decode() == "</tool_response>\n<think>\n\n</think>\n\n"


class Oracle:
    """A filesystem of named texts with the store's capacity rules. Keys are the store's
    (length, first NMAX tokens); ``list`` is in order of first creation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = {}
        self.order = []  # keys in order of first creation
        self.chunks = 0

    def run(self, op, key, text):
        cfg = self.cfg
        if op in ("put", "append"):
            new = key not in self.order
            if self.chunks >= cfg.n_chunks or (new and len(self.order) >= cfg.n_names):
                return "full\n"
            self.chunks += 1
            if new:
                self.order.append(key)
            if op == "append" and key in self.files:
                self.files[key] = self.files[key] + text
            else:
                self.files[key] = text
            return "ok\n"
        if op == "get":
            return self.files.get(key)
        if op == "delete":
            return "ok\n" if self.files.pop(key, None) is not None else "not found\n"
        live = [k for k in self.order if k in self.files]
        return live if self.order else "empty\n"


def plain_tokens(rng, n):
    """n random Qwen tokens that cannot end a parameter or a call."""
    out = []
    while len(out) < n:
        q = rng.randrange(ST.SPECIAL)
        if q not in (ST.Q_LTSLASH, ST.Q_PARAMETER):
            out.append(q)
    return out


def key_of(name):
    """The store's key of a name as the template spells it: its tokens, the newline, and the
    closer's "</" (captured too: it is only known to close at the next token), the length
    without the "</"."""
    toks = name + [ST.Q_NL, ST.Q_LTSLASH]
    n = min(len(name) + 1, ST.NMAX + 1)
    t6 = [ST.stored(q) for q in toks[:ST.NMAX]]
    return n, tuple(t6 + [0] * (ST.NMAX - len(t6)))


def random_ops(rng, n_ops, names, cfg, max_text=12, long_text=False):
    for _ in range(n_ops):
        op = rng.choice(["put", "put", "append", "append", "get", "get", "delete", "list"])
        nm = rng.choice(names)
        k = rng.randint(0, max_text)
        if long_text and rng.random() < 0.15:
            k = rng.choice([ST.CMAX - 1, ST.CMAX, ST.CMAX + 5])
        yield op, nm, plain_tokens(rng, k)


def apply(ses, op, nm, text):
    if op == "list":
        return ses.call("list")
    if op in ("get", "delete"):
        return ses.call(op, nm)
    return ses.call(op, nm, text)


def reference_vs_oracle(T, cfg, n_ops, seed, names):
    rng = random.Random(seed)
    ses = session(T, cfg)
    orc = Oracle(cfg)
    for op, nm, text in random_ops(rng, n_ops, names, cfg, long_text=True):
        start = len(ses.toks)
        apply(ses, op, nm, text)
        (got,) = ses.printed(start)
        key = key_of(nm)
        stored_text = [ST.stored(q) for q in text[:ST.CMAX]] + ([ST.Q_NL] if
                                                                 len(text) < ST.CMAX else [])
        want = orc.run(op, key, stored_text)
        if op == "get":
            want = "not found\n" if want is None else T.decode_bytes(want).decode("utf-8",
                                                                                   "replace")
        elif op == "list" and isinstance(want, list):
            want = "".join(T.decode_bytes([q for q in k[1][:min(k[0], ST.NMAX)]])
                           .decode("utf-8", "replace") for k in want)
        assert got == want, (op, got, want)
    return ses, orc


def test_reference_vs_dict(T):
    """1000 random ops: the reference answers what a plain dict-backed filesystem does (with a
    real hash, a tiny hash that collides everywhere, and stores that fill up)."""
    rng = random.Random(5)
    names = [plain_tokens(rng, rng.randint(1, 3)) for _ in range(12)]
    names += [names[0] + plain_tokens(rng, 1)]  # a longer name with the same start
    names += [names[1][:1] * 7 + [5], names[1][:1] * 7 + [6]]  # equal in their first NMAX
    for cfg, n in ((ST.Config(), 1000), (SMALL, 400), (TINY_FULL, 200),
                   (ST.Config(slot_bits=4, n_chunks=64, n_names=8, hash_bits=0), 300)):
        ses, orc = reference_vs_oracle(T, cfg, n, n, names)
        # the directory agrees with the oracle's live names
        assert sorted(ses.dev.names()) == sorted(k for k in orc.order if k in orc.files)


def test_hash_spreads(T):
    """The default hash puts 512 random names into 1024 slots with short probe chains."""
    cfg = ST.Config()
    rng = random.Random(1)
    slots = [cfg.slot_of(key_of(plain_tokens(rng, rng.randint(1, 4)))) for _ in range(512)]
    assert len(set(slots)) > 360  # a uniform hash gives about 404 distinct slots


# --- the circuit against the reference ---------------------------------------------------------


def test_circuit_all_ops(T):
    c = circuit(T, SMALL)
    ses = session(T, SMALL)
    ses.qwen(T.encode("<|im_start|>assistant\nHello there."))
    nm = T.encode("notes/plan")
    ses.call("list")  # empty
    ses.call("get", nm)  # not found
    ses.call("put", nm, T.encode("Buy milk, then call Bob at 5pm."))
    ses.call("get", nm)
    ses.call("append", nm, T.encode(" Also eggs."))
    ses.call("append", nm, T.encode("and bread"))
    ses.call("get", nm)
    ses.call("put", T.encode("id"), T.encode("7f3a-99b2"))
    ses.call("list")
    ses.call("delete", nm)
    ses.call("get", nm)
    ses.call("list")
    ses.call("delete", nm)  # not found any more
    ses.call("put", nm, [])  # an empty text, the tombstone revived
    ses.call("get", nm)
    ses.call("list")
    ses.qwen(T.encode("Done."))
    out = ses.printed()
    assert out[3] == "Buy milk, then call Bob at 5pm.\n"
    assert out[6] == "Buy milk, then call Bob at 5pm.\n Also eggs.\nand bread\n"
    assert out[8] == "notes/plan\nid\n" and out[11] == "id\n" and out[12] == "not found\n"
    assert out[14] == "\n" and out[15] == "notes/plan\nid\n"
    _, n = check(T, c, ses)
    assert n > 200


def test_circuit_with_calc(T):
    """Store calls next to calc calls (the machine runs, the store answers), stored text that
    looks like arithmetic, and a calc call right after a get."""
    c = circuit(T, SMALL)
    ses = session(T, SMALL)
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    ses.qwen(calc_call(T, "12*34"))
    ses.call("put", T.encode("sum"), T.encode("12*34 = 408 and 7*6"))
    ses.qwen(calc_call(T, "7*6"))
    ses.call("get", T.encode("sum"))
    ses.qwen(calc_call(T, "99999999999*99999999999"))
    ses.call("list")
    out = [T.decode_bytes(r).decode() for r in ses.runs]
    assert [o.split("\n")[2] for o in out] == ["408", "42", str(99999999999**2)]
    assert ses.printed()[3] == "12*34 = 408 and 7*6\n"  # (calc, put, calc, get)
    check(T, c, ses)


@pytest.mark.parametrize("cfg,seed", [(SMALL, 3), (TINY_FULL, 4)])
def test_circuit_random(T, cfg, seed):
    """Random calls with random tokens: collisions (tiny hash), overwrites, append chains,
    deletes and re-puts, names equal in their first NMAX tokens, empty and over-long texts,
    full stores (TINY_FULL), any token at all."""
    c = circuit(T, cfg)
    rng = random.Random(seed)
    base = [[rng.randrange(ST.SPECIAL) for _ in range(rng.randint(1, 3))] for _ in range(5)]
    names = base + [base[0] + [rng.randrange(ST.SPECIAL)]]
    names += [base[1][:1] * 7 + [5], base[1][:1] * 7 + [6]]
    ses = session(T, cfg)
    ses.qwen(T.encode("<|im_start|>assistant\nOK."))
    for i, (op, nm, text) in enumerate(random_ops(rng, 40, names, cfg, long_text=True)):
        text = [rng.randrange(ST.SPECIAL + 300) for _ in text]  # (specials are stored as 0)
        apply(ses, op, nm, [q for q in text if q not in (ST.Q_TCO, ST.Q_TCC)])
        if i % 13 == 5:
            ses.qwen(calc_call(T, f"{i}*{i + 1}"))
    assert len(ses.answers) == 40
    check(T, c, ses)


def test_circuit_full_size(T):
    """The 4B configuration (1024 slots, 512 chunks, 512 names), with kernel-exact reads."""
    cfg = ST.Config()
    c = circuit(T, cfg)
    rng = random.Random(11)
    names = [T.encode(f"user_{i}") for i in range(6)] + [T.encode("a" * 9), T.encode("a" * 10)]
    ses = session(T, cfg)
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    for op, nm, text in random_ops(rng, 24, names, cfg, long_text=True):
        apply(ses, op, nm, text)
    toks = ses.toks
    core.snapshot(c, fresh_machine())
    for node in c.store.persistent.values():
        node.init = None
    ir.evaluate(c, toks[:1200], kernel=True)  # every head read as the HF kernels do it
    check(T, c, ses)


def test_long_text_between(T):
    """Thousands of Qwen tokens between calls (the counters run far past their range)."""
    c = circuit(T, SMALL)
    rng = random.Random(2)
    ses = session(T, SMALL)
    ses.call("put", T.encode("k"), T.encode("first"))
    ses.qwen(plain_tokens(rng, 3000))
    ses.call("get", T.encode("k"))
    ses.call("append", T.encode("k"), T.encode("second"))
    ses.qwen(plain_tokens(rng, 2000))
    ses.call("get", T.encode("k"))
    assert ses.printed()[-1] == "first\nsecond\n"
    check(T, c, ses)


def test_snapshot_restore(T):
    """A fresh session whose store heads start from session 1's state answers from it: get,
    list, append, delete, collisions (the tiny hash)."""
    c = circuit(T, SMALL)
    rng = random.Random(9)
    names = [T.encode(w) for w in ("alpha", "beta", "gamma", "delta", "eps")]
    s1 = session(T, SMALL)
    s1.qwen(T.encode("<|im_start|>assistant\n"))
    for nm in names:
        s1.call("put", nm, plain_tokens(rng, rng.randint(1, 20)))
    s1.call("append", names[0], plain_tokens(rng, 5))
    s1.call("delete", names[2])
    carry, _ = check(T, c, s1)
    snap = snapshot(c, carry)
    s2 = ST.resume(s1, fresh_machine())
    s2.qwen(T.encode("<|im_start|>user\nWhat do you remember?<|im_end|>\n<|im_start|>assistant\n"))
    s2.call("list")
    for nm in names:
        s2.call("get", nm)
    s2.call("append", names[1], T.encode("more"))
    s2.call("get", names[1])
    s2.call("put", names[2], T.encode("back"))
    s2.call("list")
    out = s2.printed()
    assert out[0] == "alpha\nbeta\ndelta\neps\n" and out[3] == "not found\n"
    check(T, c, s2, init=snap)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_store_in_hf(T, mode):
    """The 4B-size store circuit as stock Qwen3_5ForCausalLM modules (bf16, the torch kernels):
    the residual equals the IR's on every dim at every position (slow on CPU; uses CUDA if
    there is one)."""
    torch = pytest.importorskip("torch")
    from chipformer.rv import hf

    cfg = ST.Config()
    c = circuit(T, cfg)
    ses = session(T, cfg)
    ses.qwen(T.encode("<|im_start|>assistant\nHi."))
    nm = T.encode("notes")
    ses.call("put", nm, T.encode("Buy milk."))
    ses.call("append", nm, T.encode("And eggs."))
    ses.call("get", nm)
    ses.call("list")
    ses.call("delete", nm)
    ses.call("get", nm)
    toks = ses.toks
    core.snapshot(c, fresh_machine())
    for node in c.store.persistent.values():
        node.init = None
    model, lay, ids = hf.build_model(c, toks)
    if torch.cuda.is_available():
        model = model.cuda()
    assert hf.check(c, model, lay, ids, toks, mode) == len(toks)
