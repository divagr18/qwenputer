"""Turns wasm2c's bounds-checked output into the unchecked variant: the RV32 core has
no MMU and fx-core is trusted code, so the wasm OOB checks are dropped (the same
choice as a guard-page build on a 64-bit host). Nothing in fx's code changes.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
k = s.index("DEFINE_LOAD(i32_load, u32, u32, u32, FORCE_READ_INT)")
inj = (
    "/* P0: no wasm bounds checks (trusted code, no MMU). */\n"
    "#undef RANGE_CHECK\n"
    "#define RANGE_CHECK(mem, offset, len) do {} while (0)\n"
    "#undef MEMCHECK_DEFAULT32\n"
    "#define MEMCHECK_DEFAULT32(mem, local_memory_size, a, t)\n"
)
open(dst, "w", encoding="utf-8", newline="\n").write(s[:k] + inj + s[k:])
