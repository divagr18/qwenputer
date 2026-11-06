"""Profiles what remains after the speedups (speed-iter.md item 5) from an rvhost.py
--prof run: hottest mnemonic pairs/triples (superinstruction candidates), instruction
share of the copy/compare/hash/JSON loops (exact-hardware candidates), and, for the top
candidates, the instructions saved if each became one streaming op at ~1 token per 16
bytes moved.

  python -I profile_iter.py PROF_RUN_JSON OUT_JSON [--tokens-per-instr 1.45]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

# Function groups that a streaming unit could take whole. Names are wasm2c's
# (w2c_fxcore_<zig name with 0xNN escapes>), demangled first.
GROUPS = [
    ("memcpy/memmove/memset (rt.c, wasm memory.copy/fill)", r"^(memcpy|memmove|memset|copy_fwd|copy_bwd|memory_copy|memory_fill|mem\.copy|mem\.set|mem\.replace)"),
    ("compare: mem.eqlBytes / memcmp / startsWith / eqlIgnoreCase", r"^(mem\.eqlBytes|mem\.eql|memcmp|bcmp|mem\.startsWith|mem\.endsWith|ascii\.eqlIgnoreCase|mem\.order)"),
    ("scan: mem.findScalar / indexOf / memchr", r"^(mem\.findScalar|mem\.find|mem\.indexOf|mem\.lastIndexOf|mem\.count|mem\.trim|strlen)"),
    ("SHA-256", r"(sha2|Sha2)"),
    ("wyhash (+ its 128-bit multiply)", r"(wyhash|Wyhash|__multi3|mulXi3)"),
    ("JSON scanner / tokenizer", r"^json\.(Scanner|scanner)"),
    ("JSON write / string escape", r"^(json\.Stringify|json\.stringify|json\.encode|acp\.jsonrpc\.writeJson)"),
    ("UTF-8 validation / text scan", r"(unicode\.utf8|utf8Validate|Utf8View|isWordByte)"),
    ("token estimate (incl. prompt_context.measureProviderRequest, where it is inlined)", r"(token_estimate|measureProviderRequest|language_script\.count_text)"),
    ("secret masking (text_utils.maskSecrets)", r"text_utils\.(maskSecrets|findSecretSpan|matches)"),
    ("Io.Writer byte appends", r"^Io\.Writer\.(writeByte|write|writeAll|splatByte)"),
]


def demangle(n):
    n = re.sub(r"^w2c_fxcore_", "", n)
    return re.sub(r"0x([0-9A-F]{2})", lambda m: chr(int(m.group(1), 16)), n)


def main():
    src, out = sys.argv[1], sys.argv[2]
    tpi = 1.45
    if "--tokens-per-instr" in sys.argv:
        tpi = float(sys.argv[sys.argv.index("--tokens-per-instr") + 1])
    r = json.load(open(src))
    total = r["rv_instructions_total"]
    per = Counter()
    byt = Counter()
    for k, v in r["per_function"].items():
        per[demangle(k)] += v
    for k, v in r["bytes_by_function"].items():
        byt[demangle(k)] += v
    groups = []
    used = set()
    for name, rx in GROUPS:
        fs = [f for f in per if re.search(rx, f) and f not in used]
        used.update(fs)
        ins = sum(per[f] for f in fs)
        b = sum(byt[f] for f in fs)
        stream_tokens = b / 16
        saved_instr = ins - stream_tokens / tpi
        groups.append({"group": name, "instructions": ins, "share": ins / total, "bytes_moved": b,
                       "instr_per_byte": ins / b if b else None, "streaming_tokens_at_16B": stream_tokens,
                       "instructions_saved_equiv": saved_instr, "saved_share": saved_instr / total,
                       "top_functions": sorted(((f, per[f]) for f in fs), key=lambda x: -x[1])[:5]})
    groups.sort(key=lambda g: -g["instructions_saved_equiv"])
    top = sorted(per.items(), key=lambda x: -x[1])[:40]
    pairs = r["pairs"][:40]
    trips = r["triples"][:40]
    res = {"source_run": src, "total_instructions": total, "tokens_per_instruction": tpi,
           "groups": groups, "top_functions": [{"function": f, "instructions": n, "share": n / total,
                                                "bytes": byt.get(f, 0)} for f, n in top],
           "pairs": [{"pair": f"{a} {b}", "count": n, "dependent": d, "share": n / total} for a, b, n, d in pairs],
           "triples": [{"triple": f"{a} {b} {c}", "count": n, "share": n / total} for a, b, c, n in trips],
           "dynamic_ops": r["dynamic_ops"]}
    json.dump(res, open(out, "w"), indent=1)
    print(f"total {total:,}")
    print("GROUPS (instructions, share, bytes, instr/byte, saved at 1 tok/16 B)")
    for g in groups:
        ipb = f"{g['instr_per_byte']:.1f}" if g["instr_per_byte"] else "-"
        print(f"  {g['group'][:60]:60s} {g['instructions']:>10,} {100 * g['share']:5.1f}% {g['bytes_moved']:>9,} {ipb:>6} "
              f"saved {g['instructions_saved_equiv']:>10,.0f} ({100 * g['saved_share']:.1f}%)")
    print("TOP FUNCTIONS")
    for f in res["top_functions"][:25]:
        print(f"  {f['instructions']:>10,} {100 * f['share']:5.2f}% {f['bytes']:>9,}  {f['function'][:100]}")
    print("PAIRS")
    for p in res["pairs"][:20]:
        print(f"  {p['pair']:16s} {p['count']:>10,} {100 * p['share']:5.2f}% dep {p['dependent']:,}")
    print("TRIPLES")
    for t in res["triples"][:15]:
        print(f"  {t['triple']:22s} {t['count']:>10,} {100 * t['share']:5.2f}%")


if __name__ == "__main__":
    main()
