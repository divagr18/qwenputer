"""Collects the P0 runs into p0-results.json and prints the report's tables.

Usage: python -I analyze.py RUNS_DIR BUILDS_JSON OUT_JSON
BUILDS_JSON maps labels to {"wasm": path} or {"elf": path}.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wasmbin  # noqa: E402
from elftools.elf.elffile import ELFFile  # noqa: E402

RUNS, BUILDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = lambda n: json.load(open(os.path.join(RUNS, n)))  # noqa: E731

CATS = [
    ("memcpy/memset/memcmp", r"^(mem\.eql|mem\.findScalar|mem\.find|mem\.indexOf|mem\.lastIndexOf|mem\.copy|memcpy|memmove|memset|memcmp|mem\.startsWith|mem\.endsWith|mem\.order|mem\.trim|mem\.splitScalar|mem\.tokenize|mem\.count|mem\.replace|__builtin_mem|memcpy)"),
    ("hashing", r"(sha2|Sha2|wyhash|Wyhash|hash_map\.hashString|crc|siphash|xxhash|\.hash_part|hashWord|Hasher)"),
    ("JSON parse", r"^(json\.Scanner|json\.static|json\.dynamic|json\.scanner|json\.parse|json\.Value|array_list\.AlignedManaged\(json|array_hash_map\.Custom\(\[\]const u8,json)"),
    ("JSON write/escape", r"^(json\.Stringify|json\.stringify|json\.encode|acp\.jsonrpc\.writeJson|json\.write)"),
    ("allocator", r"^(heap\.|c\.malloc|c\.free|mem\.Allocator|dlmalloc|malloc|free|realloc|calloc|array_list\.AlignedManaged\(u8|array_list\.Aligned|aligned_alloc|sbrk|__libc_malloc|internal_memalign|dispose_chunk|abort)"),
    ("UTF-8 / text scan", r"(unicode\.|utf8|Utf8|display_width|language_script|isWordByte|ascii\.)"),
    ("formatting / Io.Writer", r"^(fmt\.|Io\.Writer|Io\.Reader|__multi3|__udivti3|__divti3|compiler_rt\.|__mul|__div|__udiv|__umod|__mod|__ashl|__lshr)"),
    ("SSE / model stream", r"^(gateway\.sse|gateway\.chat_completions|gateway\.host_chat_completions|gateway\.host_stream|gateway\.json_depth|gateway\.client)"),
    ("token estimate / secrets / compactor", r"^(core\.shared\.token_estimate|core\.shared\.text_utils|core\.compactor|core\.redact)"),
    ("ACP / session / journal", r"^(acp\.|core\.session|session_manager|core\.app)"),
    ("agent loop", r"^(core\.agent|core\.tooling|core\.config|core\.output|core\.context|core\.workspace)"),
]


def category(name):
    for c, rx in CATS:
        if re.search(rx, name):
            return c
    return "other"


def demangle(n):
    n = re.sub(r"^w2c_fxcore_", "", n)
    return re.sub(r"0x([0-9A-F]{2})", lambda m: chr(int(m.group(1), 16)), n)


# ---------------- RV32 decode for the static census ----------------
def mnemonic(w):
    op, f3, f7 = w & 0x7F, (w >> 12) & 7, w >> 25
    if w & 3 != 3:
        return "C?"
    if op == 0x37: return "LUI"
    if op == 0x17: return "AUIPC"
    if op == 0x6F: return "JAL"
    if op == 0x67: return "JALR"
    if op == 0x63: return {0: "BEQ", 1: "BNE", 4: "BLT", 5: "BGE", 6: "BLTU", 7: "BGEU"}.get(f3, "?")
    if op == 0x03: return {0: "LB", 1: "LH", 2: "LW", 4: "LBU", 5: "LHU"}.get(f3, "?")
    if op == 0x23: return {0: "SB", 1: "SH", 2: "SW"}.get(f3, "?")
    if op == 0x13:
        if f3 == 1: return "SLLI" if f7 == 0 else "?"
        if f3 == 5: return {0: "SRLI", 0x20: "SRAI"}.get(f7, "?")
        return {0: "ADDI", 2: "SLTI", 3: "SLTIU", 4: "XORI", 6: "ORI", 7: "ANDI"}[f3]
    if op == 0x33:
        if f7 == 0: return ["ADD", "SLL", "SLT", "SLTU", "XOR", "SRL", "OR", "AND"][f3]
        if f7 == 0x20: return {0: "SUB", 5: "SRA"}.get(f3, "?")
        if f7 == 1: return ["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"][f3]
        if f7 == 0x10: return {2: "SH1ADD", 4: "SH2ADD", 6: "SH3ADD"}.get(f3, "?")
        return "?"
    if op == 0x0F: return "FENCE"
    if op == 0x73: return {0x73: "ECALL", 0x100073: "EBREAK"}.get(w, "CSR")
    if op == 0x2F: return {2: "LR.W", 3: "SC.W"}.get(w >> 27, "AMO")
    return "?"


EXT = {"MUL": "M", "MULH": "M", "MULHSU": "M", "MULHU": "M", "DIV": "M", "DIVU": "M", "REM": "M", "REMU": "M",
       "SH1ADD": "Zba", "SH2ADD": "Zba", "SH3ADD": "Zba", "LR.W": "Zalrsc", "SC.W": "Zalrsc", "AMO": "Zaamo",
       "CSR": "Zicsr", "C?": "C", "?": "unknown"}
CLASS = {"LUI": "ALU", "AUIPC": "ALU", "JAL": "jump", "JALR": "jump", "LB": "load", "LH": "load", "LW": "load",
         "LBU": "load", "LHU": "load", "SB": "store", "SH": "store", "SW": "store", "FENCE": "system",
         "ECALL": "system", "EBREAK": "system", "LR.W": "load", "SC.W": "store"}


def iclass(m):
    if m.startswith("B"): return "branch"
    if m in CLASS: return CLASS[m]
    if EXT.get(m) == "M": return "mul/div"
    return "ALU"


def elf_info(path):
    e = ELFFile(open(path, "rb"))
    secs = {s.name: s["sh_size"] for s in e.iter_sections() if s.name in (".text", ".rodata", ".data", ".bss")}
    t = e.get_section_by_name(".text")
    data = t.data()
    hist = Counter(mnemonic(int.from_bytes(data[i:i + 4], "little")) for i in range(0, len(data), 4))
    sym = e.get_section_by_name(".symtab")
    fx_bytes = sum(s["st_size"] for s in sym.iter_symbols() if s["st_info"]["type"] == "STT_FUNC" and s.name.startswith(("w2c_fxcore", "wrap_w2c")))
    return {"sections": secs, "fx_function_bytes": fx_bytes, "static_ops": dict(hist.most_common()),
            "static_ext": dict(Counter(EXT.get(k, "I") for k in hist.elements())),
            "static_class": dict(Counter(iclass(k) for k in hist.elements()))}


def wasm_info(path):
    m = wasmbin.Module(open(path, "rb").read())
    c, d = m.section(10), m.section(11)
    return {"code_bytes": c[3] - c[2], "data_bytes": d[3] - d[2], "functions": len(m.code),
            "imports": len(m.imports), "file_bytes": len(m.data), "static_wasm_ops": sum(wasmbin.static_census(m).values())}


def segments(events, base=0):
    """Splits a run's event list into boot / requests / per-model-step work."""
    ev = [(f - base, k, d) for f, k, d in events if k != "deliver"]
    first = {}
    for f, k, d in ev:
        first.setdefault((k, json.dumps(d, sort_keys=True)), f)
    out = {"instantiate": ev[0][0] if base == 0 else None}
    t = {k: [] for k in ("stream_open", "stream_close", "tool_call", "tool_result", "request", "response")}
    for f, k, d in ev:
        if k in t:
            t[k].append((f, d))
    start = next(f for f, k, d in ev if k == "start")
    boot_end = next(f for f, k, d in ev if k == "boot_end")
    out["boot"] = boot_end - start
    reqs = t["request"]
    resps = t["response"]
    out["initialize"] = resps[0][0] - reqs[0][0]
    out["libfx/new"] = resps[1][0] - reqs[1][0]
    turns = []
    for ti in range(2):
        r0 = reqs[2 + ti][0]
        r1 = resps[2 + ti][0]
        turns.append({"prompt_sent": r0, "turn_end": r1, "total": r1 - r0})
    out["turns"] = turns
    steps = []
    opens, closes = t["stream_open"], t["stream_close"]
    calls, results = t["tool_call"], t["tool_result"]
    ci = 0
    for k, (fo, _) in enumerate(opens):
        fc = closes[k][0]
        # what started this step: the prompt, or the previous tool's result
        prev = turns[0]["prompt_sent"] if k == 0 else (turns[1]["prompt_sent"] if k == 6 else results[k - 1][0])
        step = {"step": k + 1, "intake_and_request_build": fo - prev, "stream_consume": fc - fo}
        if k < len(calls) and k < 5:
            step["tool"] = calls[k][1]["name"]
            step["after_stream_to_tool_call"] = calls[k][0] - fc
            step["tool_in_guest"] = results[k][0] - calls[k][0]
        else:
            end = turns[0]["turn_end"] if k == 5 else turns[1]["turn_end"]
            step["tool"] = None
            step["after_stream_to_turn_end"] = end - fc
        steps.append(step)
    out["steps"] = steps
    return out


def main():
    builds = json.load(open(BUILDS))
    res = {"builds": {}, "runs": {}}
    for label, b in builds.items():
        res["builds"][label] = wasm_info(b["wasm"]) if "wasm" in b else elf_info(b["elf"])
    names = R("names.json")
    instr = R("names_instr.json")
    res["runs"]["wasm_releasesmall"] = {"fuel_total": names["fuel_total"], "peak_linear_memory_bytes": names["peak_linear_memory_bytes"],
                                        "segments": segments(names["events"]), "model_requests": names["model_requests"],
                                        "import_calls": names["import_calls"], "messages_out": names["messages_out"]}
    for lab, fn in (("wasm_stock_build", "stock.json"), ("wasm_published_libfx_0.0.16", "published.json"),
                    ("wasm_releasefast", "fast.json"), ("wasm_stripped_root", "stripped.json")):
        r = R(fn)
        res["runs"][lab] = {"fuel_total": r["fuel_total"], "peak_linear_memory_bytes": r["peak_linear_memory_bytes"], "segments": segments(r["events"])}
    for lab, fn in (("rv32_os_zba_unaligned", "rv_os_ua.json"), ("rv32_os_zba_aligned_only", "rv_os.json"),
                    ("rv32_releasefast_os_zba_unaligned", "rv_fast_ua.json"), ("rv32_stripped_os_zba_unaligned", "rv_stripped_ua.json")):
        r = R(fn)
        res["runs"][lab] = {"rv_instructions_total": r["rv_instructions_total"], "misaligned_accesses": r["misaligned_accesses"],
                            "peak_linear_memory_bytes": r["peak_linear_memory_bytes"], "segments": segments(r["events"]),
                            "dynamic_ops": r["dynamic_ops"], "same_repo_result_as_wasm": r["final_repo"] == names["final_repo"]}
        if "memory_access" in r and lab == "rv32_os_zba_unaligned":
            res["memory_access"] = r["memory_access"]
            res["memory_access"]["measured_on"] = lab
    ua = R("rv_os_ua.json")
    # ---- hot functions: wasm (fuel-weighted) vs rv32 ----
    fn = instr["func_names"]
    rv = Counter()
    for k, v in ua["per_function"].items():
        rv[demangle(k)] += v
    rows = []
    tot = sum(n + c + b for _, n, c, b in instr["final_counts"])
    for i, n, c, b in instr["final_counts"]:
        name = fn.get(str(i), f"f{i}")
        rows.append({"function": name, "wasm_ops": n, "calls": c, "bulk_bytes": b, "fuel": n + c + b,
                     "share": (n + c + b) / tot, "rv32_instructions": rv.get(name), "category": category(name)})
    rows.sort(key=lambda r: -r["fuel"])
    res["hot_functions_top60"] = rows[:60]
    cat = Counter()
    rvcat = Counter()
    for r in rows:
        cat[r["category"]] += r["fuel"]
    rvtot = sum(rv.values())
    for name, v in rv.items():
        rvcat[category(name) if name not in ("memmove", "memcpy", "memset", "memcmp", "bcmp", "strlen") else "memcpy/memset/memcmp"] += v
    res["hot_categories_wasm_fuel"] = {k: {"fuel": v, "share": v / tot} for k, v in cat.most_common()}
    res["hot_categories_rv32"] = {k: {"rv32_instructions": v, "share": v / rvtot} for k, v in rvcat.most_common()}
    res["wasm_ops_total"] = sum(n for _, n, _, _ in instr["final_counts"])
    res["wasm_calls_total"] = sum(c for _, _, c, _ in instr["final_counts"])
    res["wasm_bulk_bytes_total"] = sum(b for _, _, _, b in instr["final_counts"])
    json.dump(res, open(OUT, "w"), indent=1)

    # ---- print tables ----
    print("BUILDS")
    for k, v in res["builds"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("static_ops",)})
    print("RUNS")
    for k, v in res["runs"].items():
        s = v["segments"]
        tot_ = v.get("fuel_total", v.get("rv_instructions_total"))
        print(k, tot_, "inst", s["instantiate"], "boot", s["boot"], "init", s["initialize"], "new", s["libfx/new"], "turns", [t["total"] for t in s["turns"]])
        for st in s["steps"]:
            print("   ", st)
    print("TOP30")
    for r in rows[:30]:
        ratio = r["rv32_instructions"] / r["fuel"] if r["rv32_instructions"] else None
        print(f"{r['fuel']:>9} {100 * r['share']:5.2f}% calls {r['calls']:>6} bulkB {r['bulk_bytes']:>7} rv {r['rv32_instructions']} ratio {ratio and round(ratio, 2)} [{r['category']}] {r['function'][:90]}")
    print("CATS wasm", {k: round(100 * v['share'], 1) for k, v in res["hot_categories_wasm_fuel"].items()})
    print("CATS rv", {k: round(100 * v['share'], 1) for k, v in res["hot_categories_rv32"].items()})
    print("ops", res["wasm_ops_total"], "calls", res["wasm_calls_total"], "bulk", res["wasm_bulk_bytes_total"])


main()
