"""The object store (v2) on the RV32 Qwenputer: named texts kept in the chip layers' state.

Qwen gets five native tools next to ``calc``:

    put(name, text)     store text under name (replaces it)
    append(name, text)  add text at the end (creates the file if it is missing)
    get(name)           the stored text, token for token
    list()              the names, one per line
    delete(name)        forget a name

The circuit captures the name and the text *as Qwen writes the call* (no extra trace tokens),
keeps them in Gated DeltaNet heads, and answers at ``</tool_call>`` with a few chip records and
a ``<tool_response>`` it prints itself. A ``calc`` call still goes to the RV32 machine. Any Qwen
token can be stored and printed: every vocabulary row carries the 18 bits of its own id (input)
and an id code (output, see store_rv.py).

v2's directory is content-addressed: a name hashes (a fixed random GF(2) map of its bits) to a
slot of the directory, collisions probe the next slot (linear probing), a delete leaves a
tombstone (used, not alive) that probes walk past and a re-put of the same name revives, and
``list`` walks an order table (slots in the order their names were first created).

Records (spec.STORE_FAMILIES, one token each):

  SP(s)   probe directory slot s (the hash of the name, then s + 1, ...)
  SN(g)   put / append that creates or replaces the file: its text is chunk g; dir[s] =
          (name, alive, head g, tail g), len[g] = text length, NEXTG = g + 1
  SA(g)   append to an existing file: next[tail] = g, tail = g, len[g], NEXTG = g + 1
  SO(n)   the name's slot is new: order[n] = s, NORD = n + 1
  SD      delete: dir[s] becomes a tombstone
  SG(c)   get: chunk c is printed next (a chain of chunks)
  SL(i)   list: order entry i (its name is printed when it is alive)

(the slot s of SN / SA / SD / SO is the last SP's, the circuit reads it through a conv tap)

An answer: the records, ``\\n<tool_response>\\n``, the body, ``</tool_response>\\n<think>\\n\\n
</think>\\n\\n`` (as the calc tool's answer ends; each ``\\n\\n`` is printed as two ``\\n``).
Bodies: ``ok`` / ``not found`` / ``empty`` / ``full`` and a newline; a get's text (its chunks,
with an SG record before each, invisible to Qwen); a list's names (each name keeps its newline).

``StoreSession`` is the reference: bridge.Session with the store's capture and answers, token for
token.
"""

import copy
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from chipformer.rv import bridge as BR
from chipformer.rv import spec as S

ID_BITS = 18  # Qwen3.5's vocabulary (248320 ids) in 18 bits
SPECIAL = 248044  # <|endoftext|> and up: never stored or printed by the store (stored as 0)
NMAX = 6  # name tokens kept and compared (a name's trailing newline counts)
LEN_BITS = 3  # a name's length, min(length, NMAX + 1)
CMAX = 104  # text tokens per call (one chunk); a longer text keeps its first CMAX
TPR = 14  # tokens per content row (two value heads of 7)
ROWS_PER_CHUNK = -(-CMAX // TPR)  # 8
OPS = ("put", "append", "get", "list", "delete")

# Qwen3.5 token ids (tests check them against the tokenizer)
Q_NL, Q_EQ, Q_FUNCTION, Q_PARAMETER, Q_LTSLASH, Q_LT, Q_GT = 198, 28, 1628, 15704, 510, 27, 29
Q_TCO, Q_TCC, Q_TRO, Q_TRC, Q_THO, Q_THC = 248058, 248059, 248066, 248067, 248068, 248069
# "<function=put>": "<" "function" "=" "put" ">"; "=get" and "=list" are one token each
OP_BARE = {"put": 611, "append": 4928, "delete": 4394, "get": 447, "list": 1551}
OP_JOINED = {"get": 27362, "list": 39619}
PARAM_BARE = {"name": 591, "text": 1272}
PARAM_JOINED = {"name": 37943, "text": 46696}
MSG = {"ok": [547], "not_found": [1856, 1669], "empty": [3092], "full": [8610]}
TAIL = [Q_TRC, Q_NL, Q_THO, Q_NL, Q_NL, Q_THC, Q_NL, Q_NL]  # the last NL is Qwen's to follow

# the system prompt the benchmarks use with the store tools (docs/rv-store.md)
MEM_SYSTEM = ("You have a memory that keeps texts across conversations, with the tools put, "
              "append, get, list and delete, and a calc tool for exact arithmetic. When the user "
              "tells you something about themselves (facts, preferences, changes), save it in "
              "your memory with put, one short name per fact; when a fact changes, put the new "
              "value under the same name. Before you answer a question about the user, look in "
              "your memory first (list, then get).")
TOOLS = [
    {"type": "function", "function": {
        "name": "put", "description": "Store a text under a name in your memory (replaces it). "
                                      "The memory keeps it across conversations.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a short name"},
            "text": {"type": "string", "description": f"the text (up to {CMAX} tokens)"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "append", "description": "Add a text at the end of a stored text.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "text": {"type": "string"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "get", "description": "The text stored under a name, exactly.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list", "description": "The stored names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "delete", "description": "Forget a stored name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
]


def qid_of(t: int) -> int | None:
    """The Qwen id of a stream token (None for a chip token)."""
    return t - S.TEXT_TOKEN if t >= S.TEXT_TOKEN else None


def id_bits(qid: int | None) -> list[int]:
    """The 18 id bits a row carries (0 for chip tokens and for the special tokens)."""
    if qid is None or qid >= SPECIAL:
        return [0] * ID_BITS
    return [qid >> i & 1 for i in range(ID_BITS)]


def stored(qid: int) -> int:
    return qid if qid < SPECIAL else 0


@dataclass(frozen=True)
class Config:
    """Sizes. ``hash_bits`` < ``slot_bits`` makes a tiny hash (the slot's upper bits are 0), which
    forces collisions in tests. Chunks and names are powers of two (a bit of NEXTG / NORD says
    full)."""

    slot_bits: int = 10
    n_chunks: int = 512
    n_names: int = 512
    hash_bits: int | None = None
    seed: int = 2035

    def __post_init__(self):
        assert self.n_chunks & (self.n_chunks - 1) == 0 and self.n_chunks <= 512
        assert self.n_names & (self.n_names - 1) == 0 and self.n_names <= 1 << (self.slot_bits - 1)
        assert self.slot_bits <= 10 and self.slot_bits % 2 == 0
        assert (self.hash_bits or 0) <= self.slot_bits

    @property
    def n_slots(self) -> int:
        return 1 << self.slot_bits

    @property
    def hbits(self) -> int:
        return self.slot_bits if self.hash_bits is None else self.hash_bits

    @cached_property
    def hash_matrix(self) -> np.ndarray:
        """hbits x (NMAX * ID_BITS + LEN_BITS) of 0/1: hash bit j = parity of the key bits its
        row selects."""
        rng = np.random.RandomState(self.seed)
        return rng.randint(0, 2, size=(self.hbits, NMAX * ID_BITS + LEN_BITS))

    def slot_of(self, key) -> int:
        v = np.array(key_bits(key))
        return int(sum(int(b) << j for j, b in enumerate(self.hash_matrix @ v % 2)))


def key_bits(key) -> list[int]:
    """A name key (length, NMAX ids) as the hash sees it: each token's 18 bits, then the length's
    3 bits."""
    n, toks = key
    out = []
    for q in toks:
        out += id_bits(q)
    return out + [n >> i & 1 for i in range(LEN_BITS)]


@dataclass
class Entry:
    name: tuple
    length: int
    alive: int
    head: int = 0
    tail: int = 0

    @property
    def key(self):
        return self.length, self.name


@dataclass
class Device:
    """The store's state (what a snapshot keeps)."""

    dir: dict = field(default_factory=dict)  # slot -> Entry (a slot with an entry is used)
    order: list = field(default_factory=list)  # order index -> slot
    lens: dict = field(default_factory=dict)  # chunk -> text length
    nexts: dict = field(default_factory=dict)  # chunk -> the next chunk of its file
    content: dict = field(default_factory=dict)  # (chunk, index) -> qid
    nextg: int = 0
    gcur: int = 0
    lcur: int = 0

    @property
    def nord(self) -> int:
        return len(self.order)

    def text(self, key) -> list[int] | None:
        """The text of a live file (for tests)."""
        for e in self.dir.values():
            if e.alive and e.key == key:
                out, g = [], e.head
                while True:
                    out += [self.content.get((g, i), 0) for i in range(self.lens.get(g, 0))]
                    if g not in self.nexts:
                        return out
                    g = self.nexts[g]
        return None

    def names(self) -> list:
        return [self.dir[s].key for s in self.order if self.dir[s].alive]


class StoreScan:
    """What the circuit captures from Qwen's tool calls (the reference for store_rv.capture).

    Fed every token of the stream. Inside a tool call (``<tool_call>`` to ``</tool_call>``):

    - the op: at a ``>`` after ``function`` ``=`` OP (or ``function`` ``=OP``);
    - a parameter starts at its name token (``parameter`` ``=name``, or ``parameter`` ``=``
      ``name``) and ends at ``</`` ``parameter``; its value is every token after the first
      token that holds a newline (so the value keeps its last newline, and the ``</`` is
      counted too, which the lengths subtract);
    - name tokens: the first NMAX kept, the length counted; text tokens go to chunk NEXTG,
      index i < CMAX, as they are written.
    """

    def __init__(self, cfg: Config, dev: Device) -> None:
        self.cfg, self.dev = cfg, dev
        self.p = [None, None, None]  # qids of the last three tokens (None: a chip token)
        self.in_call = False
        self.op = None
        self.pk = None
        self.hdr = False  # a newline token since the parameter's name
        self.name: list[int] = []
        self.n_name = 0
        self.n_text = 0
        self.nl = set()  # qids that hold a newline (set by StoreSession)

    def feed(self, t: int) -> None:
        q = qid_of(t)
        p1, p2, p3 = self.p
        self.p = [q, p1, p2]
        if q == Q_TCO:
            self.in_call = True
            self.op, self.pk, self.name, self.n_name, self.n_text = None, None, [], 0, 0
        elif q == Q_TCC:
            self.in_call = False
        ic = self.in_call and q is not None and q != Q_TCO
        param = None
        if ic:
            for p, i in PARAM_JOINED.items():
                if q == i and p1 == Q_PARAMETER:
                    param = p
            for p, i in PARAM_BARE.items():
                if q == i and p1 == Q_EQ and p2 == Q_PARAMETER:
                    param = p
            if q == Q_GT:
                for op, i in OP_BARE.items():
                    if p1 == i and p2 == Q_EQ and p3 == Q_FUNCTION:
                        self.op = op
                for op, i in OP_JOINED.items():
                    if p1 == i and p2 == Q_FUNCTION:
                        self.op = op
        if param is not None:
            self.hdr = False
        if p1 is not None and p1 in self.nl:
            self.hdr = True
        if param is not None:
            self.pk = param
            if param == "name":
                self.name, self.n_name = [], 0
            else:
                self.n_text = 0
        elif ic and q == Q_PARAMETER and p1 == Q_LTSLASH:
            self.pk = None
        if not (ic and self.pk is not None and self.hdr):
            return
        if self.pk == "name":
            if self.n_name < NMAX:
                self.name.append(stored(q))
            self.n_name += 1
        else:
            dev = self.dev
            if self.n_text < CMAX and dev.nextg < self.cfg.n_chunks:
                dev.content[dev.nextg, self.n_text] = stored(q)
            self.n_text += 1

    def name_key(self) -> tuple:
        """(length, the first NMAX tokens). The closer's "</" was captured too (it is only
        known to close the value at the next token): it is not counted in the length, but when
        the name is shorter than NMAX it is the token after it in the key (the same for every
        call that spells the name)."""
        n = max(min(self.n_name, NMAX + 2) - 1, 0)
        toks = self.name[:NMAX]
        return n, tuple(toks + [0] * (NMAX - len(toks)))

    def text_len(self) -> int:
        return max(min(self.n_text, CMAX + 1) - 1, 0)

    def is_store_call(self) -> bool:
        return self.op is not None


class StoreSession(BR.Session):
    """bridge.Session with the object store: a store call's ``</tool_call>`` is answered by the
    store (the circuit owns its records and print), any other call runs the machine."""

    def __init__(self, text: BR.Text, machine, cfg: Config | None = None,
                 dev: Device | None = None) -> None:
        self.cfg = cfg or Config()
        self.dev = dev if dev is not None else Device()
        self.scan = StoreScan(self.cfg, self.dev)
        self.scan.nl = {q for q in range(text.n) if b"\n" in text.bytes[q]}
        self.answers: list[tuple] = []
        super().__init__(text, machine)

    def _push(self, t: int, owner: bool) -> None:
        super()._push(t, owner)
        self.scan.feed(t)

    def qwen(self, qids: list[int], max_records: int = 10**6, run: bool = True) -> None:
        for q in qids:
            if q == self.T.TCC and self.scan.is_store_call():
                self.armed = False
                self._push(BR.text_token(q), True)
                if run:
                    self.respond()
            else:
                super().qwen([q], max_records, run)

    # --- Qwen's calls (as Qwen3.5's template spells them) --------------------------------------

    @staticmethod
    def call_tokens(op: str, name: list[int] | None = None, text: list[int] | None = None):
        t = [Q_TCO, Q_NL, Q_LT, Q_FUNCTION]
        t += [OP_JOINED[op]] if op in OP_JOINED else [Q_EQ, OP_BARE[op]]
        t += [Q_GT, Q_NL]
        for p, val in (("name", name), ("text", text)):
            if val is None:
                continue
            t += [Q_LT, Q_PARAMETER, PARAM_JOINED[p], Q_GT, Q_NL, *val, Q_NL, Q_LTSLASH,
                  Q_PARAMETER, Q_GT, Q_NL]
        return t + [Q_LTSLASH, Q_FUNCTION, Q_GT, Q_NL, Q_TCC]

    def call(self, op: str, name: list[int] | None = None, text: list[int] | None = None):
        n = len(self.answers)
        self.qwen(self.call_tokens(op, name, text))
        assert len(self.answers) == n + 1, "the call did not reach the store"
        return self.answers[-1]

    # --- the circuit's answer --------------------------------------------------------------

    def _chip(self, fam: str, p: int = 0) -> None:
        self._push(S.tok(fam, p), True)

    def _print(self, q: int, own: bool = True) -> None:
        self._push(BR.text_token(q), own)

    def _header(self) -> None:
        for q in (Q_NL, Q_TRO, Q_NL):
            self._print(q)

    def _msg(self, name: str) -> None:
        for q in MSG[name]:
            self._print(q)
        self._print(Q_NL)

    def _tail(self) -> None:
        for i, q in enumerate(TAIL):
            self._print(q, i < len(TAIL) - 1)

    def respond(self) -> None:
        op, cfg, dev, scan = self.scan.op, self.cfg, self.dev, self.scan
        if op == "list":
            self._header()
            if dev.nord == 0:
                self._msg("empty")
            for i, s in enumerate(list(dev.order)):
                self._chip("SL", i)
                dev.lcur = i
                e = dev.dir[s]
                if e.alive:
                    for q in e.name[:min(e.length, NMAX)]:
                        self._print(q)
            self._tail()
            self.answers.append(("list",))
            return
        key = scan.name_key()
        s = cfg.slot_of(key)
        while True:
            self._chip("SP", s)
            e = dev.dir.get(s)
            if e is None or e.key == key:
                break
            s = (s + 1) % cfg.n_slots
        used = e is not None
        found = used and bool(e.alive)
        full = False
        if op in ("put", "append"):
            full = dev.nextg >= cfg.n_chunks or (not used and dev.nord >= cfg.n_names)
            if not full:
                g = dev.nextg
                if op == "append" and found:
                    self._chip("SA", g)
                    dev.nexts[e.tail] = g
                    e.tail = g
                else:
                    self._chip("SN", g)
                    dev.dir[s] = Entry(key[1], key[0], 1, g, g)
                dev.lens[g] = scan.text_len()
                dev.nextg = g + 1
                if not used:
                    self._chip("SO", dev.nord)
                    dev.order.append(s)
            self._header()
            self._msg("full" if full else "ok")
        elif op == "delete":
            if found:
                self._chip("SD")
                e.alive, e.head = 0, 0
            self._header()
            self._msg("ok" if found else "not_found")
        else:  # get
            self._header()
            if not found:
                self._msg("not_found")
            else:
                g = e.head
                while True:
                    self._chip("SG", g)
                    dev.gcur = g
                    for i in range(dev.lens.get(g, 0)):
                        self._print(dev.content.get((g, i), 0))
                    if g not in dev.nexts:
                        break
                    g = dev.nexts[g]
        self._tail()
        self.answers.append((op, found, full))

    # --- for tests ---------------------------------------------------------------------------

    def printed(self, start: int = 0) -> list[str]:
        """The bodies the store printed after stream position ``start``, one per response."""
        out, cur = [], None
        for t in self.toks[start:]:
            q = qid_of(t)
            if q == Q_TRO:
                cur = []
            elif q == Q_TRC and cur is not None:
                out.append(self.T.decode_bytes(cur).decode("utf-8", "replace")[1:])
                cur = None
            elif cur is not None and q is not None:
                cur.append(q)
        return out


def resume(prev: StoreSession, machine) -> StoreSession:
    """A new session (empty context) whose store is ``prev``'s: what a restored snapshot gives."""
    return StoreSession(prev.T, machine, prev.cfg, copy.deepcopy(prev.dev))
