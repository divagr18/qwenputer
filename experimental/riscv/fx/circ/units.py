"""The exact units for fx's hot code (circuits.md): where each one triggers in a given fx
RV32 image, its exits and their dead registers, and the Python reference of its semantics.

A unit is bound to the instruction words of fx's fixed ROM: `specs_for` finds each unit's code
region in the image by its instruction signature (the words of the row-11 build, with LUI /
AUIPC immediates and call offsets masked, since those move with the layout), and installs the
unit only where the signature matches exactly. Dead registers at an exit come from a static
liveness analysis of the containing function (`liveness`); at a function return, from the
calling convention (caller-saved registers other than a0).
"""
from __future__ import annotations

import os
import re
import struct
from functools import lru_cache

from elftools.elf.elffile import ELFFile

# unit kinds (circ_units.h)
KINDS = {"MEMMOVE": 1, "MEMCPY": 2, "MEMSET": 3, "EQL": 4, "TOKEST": 5, "JSCAN": 6, "JESC": 7, "ASCII": 8,
         "MASKP": 9, "MEMCHR": 10, "BCOPY": 11, "SHA": 12, "WYHASH": 13}
ABI_DEAD = sum(1 << r for r in (5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31))  # t*, a1-a7
CALLER_SAVED = ABI_DEAD | 1 << 1 | 1 << 10
CALLEE = sum(1 << r for r in [2, 3, 4, 8, 9] + list(range(18, 28)))  # sp gp tp s0-s11
ARGS = sum(1 << r for r in range(10, 18))
ALL = 0xFFFFFFFE
REG = ("zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 "
       "t3 t4 t5 t6").split()


def demangle(n):
    n = re.sub(r"^w2c_fxcore_", "", n)
    return re.sub(r"0x([0-9A-F]{2})", lambda m: chr(int(m.group(1), 16)), n)


class Image:
    def __init__(self, path):
        f = open(path, "rb")
        e = ELFFile(f)
        t = e.get_section_by_name(".text")
        self.text = t.data()
        self.base = t["sh_addr"]
        self.funcs = {}
        for s in e.get_section_by_name(".symtab").iter_symbols():
            if s["st_info"]["type"] == "STT_FUNC" and s["st_size"]:
                self.funcs[demangle(s.name)] = (s["st_value"], s["st_size"])
        self.path = path

        self.segs = [(sg["p_vaddr"], sg.data()) for sg in e.iter_segments()
                     if sg["p_type"] == "PT_LOAD" and sg["p_filesz"]]

    def data_word(self, a):
        for va, d in self.segs:
            if va <= a and a + 4 <= va + len(d):
                return struct.unpack("<I", d[a - va:a - va + 4])[0]
        raise KeyError(hex(a))

    def data_byte(self, a):
        for va, d in self.segs:
            if va <= a < va + len(d):
                return d[a - va]
        raise KeyError(hex(a))

    def find_slices(self, strings):
        """Addresses of arrays of (ptr, len) slices pointing at exactly these strings."""
        out = []
        n0 = len(strings[0])
        for va, d in self.segs:
            for k in range(0, len(d) - 8 * len(strings) + 1, 4):
                if struct.unpack("<I", d[k + 4:k + 8])[0] != n0:
                    continue
                ok = True
                for i, st in enumerate(strings):
                    p, n = struct.unpack("<II", d[k + 8 * i:k + 8 * i + 8])
                    try:
                        ok = n == len(st) and bytes(self.data_byte(p + j) for j in range(n)) == st.encode()
                    except KeyError:
                        ok = False
                    if not ok:
                        break
                if ok:
                    out.append(va + k)
        return out

    def word(self, pc):
        o = pc - self.base
        return struct.unpack("<I", self.text[o:o + 4])[0]

    def words(self, pc, n):
        return [self.word(pc + 4 * i) for i in range(n)]


@lru_cache(maxsize=4)
def image(path):
    return Image(path)


# ---- decoding for signatures and liveness ------------------------------------------------------

def sx(v, b):
    v &= (1 << b) - 1
    return v - (1 << b) if v >> (b - 1) else v


def dec(w, pc):
    """(class, rd, rs1, rs2, target) of an RV32IM+Zba(+Zbb/Zbs) word."""
    op, rd, f3, rs1, rs2 = w & 0x7F, w >> 7 & 31, w >> 12 & 7, w >> 15 & 31, w >> 20 & 31
    if op in (0x37, 0x17):
        return ("u", rd, 0, 0, None)
    if op == 0x6F:
        im = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11 | (w >> 21 & 0x3FF) << 1, 21)
        return ("jal", rd, 0, 0, pc + im)
    if op == 0x67:
        return ("jalr", rd, rs1, 0, sx(w >> 20, 12))
    if op == 0x63:
        im = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13)
        return ("br", 0, rs1, rs2, pc + im)
    if op == 0x03:
        return ("ld", rd, rs1, 0, None)
    if op == 0x23:
        return ("st", 0, rs1, rs2, None)
    if op == 0x13:
        return ("i", rd, rs1, 0, None)
    if op == 0x33:
        return ("r", rd, rs1, rs2, None)
    if op == 0x73:
        return ("ecall", 0, 0, 0, None) if w == 0x73 else ("csr", rd, rs1, 0, None)
    if op == 0x2F:
        return ("r", rd, rs1, rs2, None)
    if op == 0x0F:
        return ("fence", 0, 0, 0, None)
    return ("?", 0, 0, 0, None)


def sig_mask(w):
    """The part of an instruction word a signature compares: LUI / AUIPC immediates and JALR
    offsets after an AUIPC move with the layout."""
    op = w & 0x7F
    if op in (0x37, 0x17):
        return w & 0xFFF
    if op == 0x6F and (w >> 7 & 31) == 1:  # a call: its offset moves
        return w & 0xFFF
    if op == 0x67 and (w >> 7 & 31) == 1:  # auipc ra; jalr ra, off(ra): the call's low offset
        return w & 0xFFFFF
    return w


def find_region(img, fn, sig, nth=0, count=1):
    """Address of the nth occurrence of the instruction sequence `sig` (masked) in function `fn`
    of img, or None unless it occurs exactly `count` times."""
    if fn not in img.funcs:
        return None
    a, sz = img.funcs[fn]
    n = sz // 4
    ws = img.words(a, n)
    m = [sig_mask(w) for w in ws]
    s = [sig_mask(w) for w in sig]
    hits = [i for i in range(n - len(s) + 1) if m[i:i + len(s)] == s]
    return a + 4 * hits[nth] if len(hits) == count else None


def liveness(img, fn):
    """Live-in register masks per PC of function fn (backward dataflow over its CFG; calls
    clobber caller-saved registers and read the argument registers; an indirect jump reads
    everything)."""
    a, sz = img.funcs[fn]
    n = sz // 4
    ws = img.words(a, n)
    use, dfn, succ = [0] * n, [0] * n, [[] for _ in range(n)]
    for i, w in enumerate(ws):
        pc = a + 4 * i
        c, rd, r1, r2, tg = dec(w, pc)
        u = d = 0
        nxt = [i + 1] if i + 1 < n else []
        if c == "u":
            d = 1 << rd
        elif c == "jal":
            if rd == 0:
                if a <= tg < a + sz:
                    nxt = [(tg - a) // 4]
                else:  # tail call
                    u, nxt = ARGS | CALLEE | 2, []
            else:  # call
                u, d = ARGS | 1 << 2, CALLER_SAVED
        elif c == "jalr":
            if rd == 0 and r1 == 1 and tg == 0:  # ret
                u, nxt = 1 << 10 | 1 << 11 | CALLEE | 1 << 1, []
            elif rd == 0:  # jump table / tail call through a register
                u, nxt = ALL, []
            else:
                u, d = ARGS | 1 << r1 | 1 << 2, CALLER_SAVED
        elif c == "br":
            u = 1 << r1 | 1 << r2
            nxt = nxt + ([(tg - a) // 4] if a <= tg < a + sz else [])
        elif c in ("ld", "i", "csr"):
            u, d = 1 << r1, 1 << rd
        elif c == "st":
            u = 1 << r1 | 1 << r2
        elif c == "r":
            u, d = 1 << r1 | 1 << r2, 1 << rd
        elif c == "ecall":
            u, d = ARGS, 1 << 10
        elif c == "?":
            u, nxt = ALL, []
        use[i], dfn[i], succ[i] = u & ~1, d & ~1, nxt
    live = [0] * n
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            out = 0
            for j in succ[i]:
                out |= live[j]
            v = use[i] | (out & ~dfn[i])
            if v != live[i]:
                live[i] = v
                changed = True
    return {a + 4 * i: live[i] for i in range(n)}


# ---- the units ---------------------------------------------------------------------------------

# Each spec: name, label, kind, fn (containing function), sig_at (row-11 address of the
# signature's first word), sig_len (words), trigger (row-11 address), exits (row-11 addresses,
# in the order circ_loops.h indexes them; a function unit's exit is the return), params
# (callable (img, base, row11_base) -> list of extra parameters).

def lui_addi(img, pc):
    """The constant a `lui rd, hi; addi rd, rd, lo` pair at pc builds."""
    w1, w2 = img.word(pc), img.word(pc + 4)
    assert w1 & 0x7F == 0x37 and w2 & 0x707F == 0x13, (hex(pc), hex(w1), hex(w2))
    return ((w1 & 0xFFFFF000) + sx(w2 >> 20, 12)) & 0xFFFFFFFF


def jscan_plain(img, off):
    """json.Scanner.next's string loop (trigger 0x263ffbc in row 11): the 256-bit set of bytes
    whose iteration returns to the loop head (the unit stops before the others, which leave the
    loop). From the code: a byte b with b - 0x22 > 0xd2 (unsigned) goes to 0x264001c; the others
    through the jump table at the `lui/addi a6` pair. 0x264001c continues to 0x263fff0 when
    b >= 0x20; from 0x263fff0, b returns to the head when it is signed > 0x5c, below 0x22, or
    0x23..0x5b. Any other jump-table target leaves the loop."""
    jt = lui_addi(img, 0x263FFDC + off)
    t_ctl, t_tail = 0x264001C + off, 0x263FFF0 + off

    def tail(b):  # the code at 0x263fff0
        sb = b - 256 if b >= 128 else b
        return sb > 0x5C or b < 0x22 or ((b - 0x23) & 0xFF) < 0x39

    def via(tgt, b):
        if tgt == t_ctl:
            return b >= 0x20 and tail(b)
        return tgt == t_tail and tail(b)

    plain = []
    for b in range(256):
        if (b - 0x22) & 0xFFFFFFFF > 0xD2:
            plain.append(via(t_ctl, b))
        else:
            plain.append(via(img.data_word(jt + 4 * (b - 0x22)), b))
    return [sum(1 << i for i in range(32) if plain[32 * k + i]) for k in range(8)]


SECRET_PREFIXES = ["OPENAI_API_KEY=", "ANTHROPIC_API_KEY=", "AI_GATEWAY_API_KEY=", "VERCEL_OIDC_TOKEN=",
                   "GITHUB_TOKEN=", "AWS_SECRET_ACCESS_KEY=", "DATABASE_URL=", "SECRET_KEY=", "API_KEY=",
                   "TOKEN=", "PASSWORD=", "PRIVATE_KEY="]


def maskp_table(img, off):
    """text_utils.secret_prefixes (text_utils.zig:825): the wasm address of its 12 (ptr, len)
    slices, which maskSecrets builds in s5 (`lui s5` / `addi s5, s5` at its entry). The unit
    checks s5 against it and the prefixes' bytes are watched as read-only during the run."""
    a, sz = img.funcs["core.shared.text_utils.maskSecrets"]
    hi = lo = None
    for i in range(min(sz // 4, 64)):
        w = img.word(a + 4 * i)
        if w & 0xFFF == 0x037 | 21 << 7 and hi is None:  # lui s5
            hi = w & 0xFFFFF000
        if w & 0xFFFFF == 0x13 | 21 << 7 | 21 << 15 and lo is None:  # addi s5, s5, imm
            lo = sx(w >> 20, 12)
    assert hi is not None and lo is not None
    return [(hi + lo) & 0xFFFFFFFF]


# Each loop unit: fn (containing function), region (row-11 address, words: the signature),
# trigger (row-11 PC), exits (row-11 PCs, indexed by circ_loops.h), params (function of the
# image and the image's offset from row 11).
UNITS = [
    {"name": "move", "label": "memmove", "kind": "MEMMOVE", "fn": "memmove", "entry": True},
    {"name": "move", "label": "memcpy", "kind": "MEMCPY", "fn": "memcpy", "entry": True},
    {"name": "set", "label": "memset", "kind": "MEMSET", "fn": "memset", "entry": True},
    {"name": "eql", "label": "mem.eqlBytes", "kind": "EQL", "fn": "mem.eqlBytes", "entry": True},
    {"name": "sha", "label": "crypto.sha2 SHA-256 round (block compression)", "kind": "SHA",
     "fn": "crypto.sha2.Sha2x32(.{ 1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, "
           "528734635, 1541459225 },256).round", "entry": True},
    {"name": "wyhash", "label": "hash.wyhash.Wyhash.hash", "kind": "WYHASH", "fn": "hash.wyhash.Wyhash.hash",
     "entry": True, "dead": ABI_DEAD & ~(1 << 11)},
    {"name": "tokest", "label": "measureProviderRequest whitespace-run loop", "kind": "TOKEST",
     "fn": "core.agent.runtime.prompt_context.measureProviderRequest", "region": (0x22C4554, 39),
     "trigger": 0x22C4574, "exits": [0x22C45F0], "count": 2},
    {"name": "tokest", "label": "measureProviderRequest whitespace-run loop (2nd copy)", "kind": "TOKEST",
     "fn": "core.agent.runtime.prompt_context.measureProviderRequest", "region": (0x22C52F8, 39),
     "trigger": 0x22C52F8 + 0x20, "exits": [0x22C52F8 + 0x9C], "count": 2, "nth": 1},
    {"name": "jscan", "label": "json.Scanner.next string loop", "kind": "JSCAN", "fn": "json.Scanner.next",
     "region": (0x263FFBC, 27), "trigger": 0x263FFBC, "exits": [0x263FFBC], "params": jscan_plain},
    {"name": "jesc", "label": "json.Stringify.encodeJsonString plain-run loop", "kind": "JESC",
     "fn": "json.Stringify.encodeJsonString", "region": (0x263DEA4, 78), "trigger": 0x263DEA4,
     "exits": [0x263DEA4]},
    {"name": "ascii", "label": "json.Stringify.write ASCII-run loop", "kind": "ASCII",
     "fn": "json.Stringify.write__func_1398", "region": (0x263D664, 6), "trigger": 0x263D664,
     "exits": [0x263D7F4, 0x263D67C]},
    {"name": "maskp", "label": "text_utils.maskSecrets secret_prefixes probe", "kind": "MASKP",
     "fn": "core.shared.text_utils.maskSecrets", "region": (0x24D9E44, 51), "trigger": 0x24D9E44,
     "exits": [0x24D9F10, 0x24D9EB8], "params": maskp_table},
    {"name": "memchr", "label": "acp.jsonrpc.writeOneLine find-newline loop", "kind": "MEMCHR",
     "fn": "acp.jsonrpc.writeOneLine", "region": (0x20CE79C, 8), "trigger": 0x20CE7A8,
     "exits": [0x20CE8E0, 0x20CE7BC]},
    {"name": "memchr", "label": "acp.jsonrpc.writeOneLine find-CR loop", "kind": "MEMCHR",
     "fn": "acp.jsonrpc.writeOneLine", "region": (0x20CE7C4, 8), "trigger": 0x20CE7D0,
     "exits": [0x20CE8E8, 0x20CE7E4]},
    {"name": "bcopy", "label": "c.malloc.realloc byte-copy loop", "kind": "BCOPY", "fn": "c.malloc.realloc",
     "region": (0x21E4498, 7), "trigger": 0x21E44A0, "exits": [0x21E44B4]},
]

# the row-11 image the signatures come from (its function bytes are the reference)
REF_IMAGE = os.environ.get("FXCIRC_REF_ELF")
_SIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signatures.json")


def load_sigs():
    import json
    return json.load(open(_SIGS)) if os.path.exists(_SIGS) else {}


def specs_for(elf, which="all"):
    """Unit triggers for image `elf`: [{name, label, kind, pc, params, dead}], and the units
    whose signature is not found there."""
    if which in (None, "", "none"):
        return [], []
    img = image(elf)
    sigs = load_sigs()
    sel = None if which == "all" else set(which.split(","))
    out, skipped = [], []
    for u in UNITS:
        if sel is not None and u["name"] not in sel and u["label"] not in sel:
            continue
        sg = sigs.get(u["label"])
        if sg is None:
            skipped.append({"unit": u["label"], "why": "no signature"})
            continue
        base = find_region(img, u["fn"], sg["words"], u.get("nth", 0), u.get("count", 1))
        if base is None:
            skipped.append({"unit": u["label"], "why": "signature not found in this image"})
            continue
        if u.get("entry"):
            out.append({"name": u["name"], "label": u["label"], "kind": KINDS[u["kind"]], "pc": base,
                        "params": [], "dead": [u.get("dead", ABI_DEAD)]})
            continue
        off = base - sg["at"]
        live = liveness(img, u["fn"])
        exits = [x + off for x in sg["exits"]]
        dead = [ALL & ~live[x] & ~1 for x in exits]
        params = exits + (u["params"](img, off) if u.get("params") else [])
        out.append({"name": u["name"], "label": u["label"], "kind": KINDS[u["kind"]], "pc": sg["trigger"] + off,
                    "params": params, "dead": dead})
    return out, skipped


def make_signatures(elf):
    """Writes signatures.json from the row-11 image (the reference build)."""
    import json
    img = image(elf)
    sigs = {}
    for u in UNITS:
        if u.get("entry"):
            a, sz = img.funcs[u["fn"]]
            sigs[u["label"]] = {"at": a, "words": img.words(a, sz // 4)}
        else:
            at, n = u["region"]
            nth = u.get("nth", 0)
            first = find_region(img, u["fn"], img.words(at, n), 0, u.get("count", 1))
            assert first is not None, u["label"]
            sigs[u["label"]] = {"at": at, "words": img.words(at, n), "trigger": u["trigger"],
                                "exits": u["exits"]}
            if nth:  # a second copy of the same code: same words, its own addresses
                assert find_region(img, u["fn"], img.words(at, n), nth, u["count"]) == at
    sigs["_source"] = {"elf": os.path.basename(os.path.dirname(elf)) + "/" + os.path.basename(elf)}
    json.dump(sigs, open(_SIGS, "w"), indent=0)
    return sigs


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "sigs":
        make_signatures(sys.argv[2])
        print("wrote", _SIGS)
    elif sys.argv[1] == "specs":
        sp, sk = specs_for(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all")
        for s in sp:
            print(s["label"], hex(s["pc"]), [hex(p) for p in s["params"]],
                  [" ".join(REG[r] for r in range(32) if d >> r & 1) for d in s["dead"]])
        print("skipped", sk)
