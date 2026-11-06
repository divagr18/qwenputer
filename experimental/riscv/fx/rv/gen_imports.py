"""Generates imports.c (each wasm import as an `ecall` into the host) and imports.json
from the fxcore.h that wasm2c wrote.

ABI: a7 = 0x400 + import index, a0 = pointer to the u32 argument words (a u64
argument takes two words, low first), a1 = word count. The host returns in a0.
"""
import json
import re
import sys

hdr = open(sys.argv[1]).read()
out_c, out_j = sys.argv[2], sys.argv[3]
pat = re.compile(r"/\* import: '([^']+)' '([^']+)' \*/\n(\w+) (w2c_\w+)\(([^)]*)\);")
table = []
c = ['#include <stdint.h>', '#include "fxcore.h"', '',
     'static inline uint32_t hostcall(uint32_t id, uint32_t* a, uint32_t n) {',
     '  register uint32_t a0 asm("a0") = (uint32_t)a;',
     '  register uint32_t a1 asm("a1") = n;',
     '  register uint32_t a7 asm("a7") = 0x400 + id;',
     '  asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a7) : "memory");',
     '  return a0;', '}', '']
for k, (mod, name, ret, fn, params) in enumerate(pat.findall(hdr)):
    ps = [p.strip() for p in params.split(",")]
    inst, types = ps[0], ps[1:]
    args = []
    words = []
    for i, t in enumerate(types):
        args.append(f"{t} p{i}")
        if t in ("u64", "s64"):
            words += [f"(uint32_t)p{i}", f"(uint32_t)(p{i} >> 32)"]
        elif t in ("u32", "s32"):
            words.append(f"(uint32_t)p{i}")
        else:
            raise SystemExit(f"unsupported param type {t} in {fn}")
    sig = ", ".join([inst + " inst"] + args)
    arr = "{" + ", ".join(words) + "}" if words else "{0}"
    body = f"  uint32_t a[] = {arr};\n  "
    call = f"hostcall({k}, a, {len(words)})"
    if ret == "void":
        body += call + ";"
    elif ret in ("u32", "s32"):
        body += f"return ({ret}){call};"
    else:
        raise SystemExit(f"unsupported return type {ret} in {fn}")
    c.append(f"{ret} {fn}({sig}) {{\n{body}\n}}")
    table.append({"id": k, "module": mod, "name": name, "ret": ret, "params": types})
open(out_c, "w").write("\n".join(c) + "\n")
json.dump(table, open(out_j, "w"), indent=1)
print(len(table), "imports")
