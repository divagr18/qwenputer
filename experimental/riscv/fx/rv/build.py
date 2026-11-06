"""Builds an RV32 image of fx-core.wasm: wasm2c, C patches, zig cc (riscv32), link.

Replaces build_rv.sh for the speed iteration (speed-iter.md); build_rv.sh's output is
`--split 1 --align none --membase 0x02000000`.

  python -I build.py --wasm fx-core.wasm --out DIR [--opt -Os] [--cpu generic_rv32+m+zba]
         [--split 16] [--align none|bytes|check] [--membase 0x02000000|0] [-j 16]

Toolchain changes only; fx's code is the wasm module, untouched:
- --split N    wasm2c --num-outputs=N. clang -O2 dies on the single 47 MB file; split files
               compile one by one (and in parallel).
- --align      none:  every access is wasm2c's memcpy (aligned only if the CPU has
                      unaligned scalar access, which P0 assumed).
               bytes: alignsplit.py tags each access with its own memarg alignment;
                      naturally aligned accesses (capped at 4 bytes) become plain LW/SW/LH,
                      under-aligned ones stay byte-safe memcpy.
               check: as bytes, but an under-aligned access tests the address and takes the
                      word path when it happens to be aligned.
- --membase    where wasm linear memory sits. 0 maps wasm address = physical address, so
               a wasm load is one LW off the wasm pointer (no add of a memory base). The
               image then sits at 0x02000000 and the C stack under 128 MB.
Needs ZIG and WABT in the environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

UNCHECKED = """
/* speed-iter: no wasm bounds checks (trusted code, no MMU), as P0. */
#undef RANGE_CHECK
#define RANGE_CHECK(mem, offset, len) do {} while (0)
#undef MEMCHECK_DEFAULT32
#define MEMCHECK_DEFAULT32(mem, local_memory_size, a, t)
"""

MEMBASE0 = """
/* speed-iter: linear memory at physical address 0, so a wasm address is the pointer. */
#undef MEM_ADDR_MEMOP
#define MEM_ADDR_MEMOP(mem, addr, n) ((uint8_t*)(uintptr_t)(uint32_t)(addr))
"""

ALIGNED = r"""
/* speed-iter: alignment-honouring accesses. alignsplit.py moved every access whose memarg
 * alignment is below min(natural, 4) to memory M1; build.py renamed those calls to the
 * name##_ua variants (on memory 0). The rest are aligned by their hint: plain LW/SW. */
#undef DEFINE_LOAD
#undef DEFINE_STORE
#define RV_AL(t) (sizeof(t) < 4 ? sizeof(t) : 4)
#if UA_CHECK
#define UA_COPY(d, s, n, a)                                                       \
  do {                                                                            \
    if ((((uintptr_t)(d) | (uintptr_t)(s)) & ((a) - 1)) == 0)                     \
      wasm_rt_memcpy(__builtin_assume_aligned(d, a), __builtin_assume_aligned(s, a), n); \
    else                                                                          \
      wasm_rt_memcpy(d, s, n);                                                    \
  } while (0)
#else
#define UA_COPY(d, s, n, a) wasm_rt_memcpy(d, s, n)
#endif
#define DEFINE_LOAD(name, t1, t2, t3, force_read)                                              \
  static inline t3 name##_unchecked(uint8_t* const wasm_rt_local_memory_base,                  \
                                    wasm_rt_memory_t* mem, u64 addr) {                         \
    t1 result;                                                                                 \
    wasm_rt_memcpy(&result, __builtin_assume_aligned(MEM_ADDR_MEMOP(mem, addr, sizeof(t1)), RV_AL(t1)), sizeof(t1)); \
    t3 ret = (t3)(t2)result;                                                                   \
    force_read(ret);                                                                           \
    return ret;                                                                                \
  }                                                                                            \
  DEF_MEM_CHECKS0(name, _, t1, return, t3)                                                     \
  static inline t3 name##_ua_unchecked(uint8_t* const wasm_rt_local_memory_base,               \
                                       wasm_rt_memory_t* mem, u64 addr) {                      \
    t1 result;                                                                                 \
    UA_COPY(&result, MEM_ADDR_MEMOP(mem, addr, sizeof(t1)), sizeof(t1), RV_AL(t1));            \
    t3 ret = (t3)(t2)result;                                                                   \
    force_read(ret);                                                                           \
    return ret;                                                                                \
  }                                                                                            \
  DEF_MEM_CHECKS0(name##_ua, _, t1, return, t3)
#define DEFINE_STORE(name, t1, t2)                                                             \
  static inline void name##_unchecked(uint8_t* const wasm_rt_local_memory_base,                \
                                      wasm_rt_memory_t* mem, u64 addr, t2 value) {             \
    t1 wrapped = (t1)value;                                                                    \
    wasm_rt_memcpy(__builtin_assume_aligned(MEM_ADDR_MEMOP(mem, addr, sizeof(t1)), RV_AL(t1)), &wrapped, sizeof(t1)); \
  }                                                                                            \
  DEF_MEM_CHECKS1(name, _, t1, , void, t2)                                                     \
  static inline void name##_ua_unchecked(uint8_t* const wasm_rt_local_memory_base,             \
                                         wasm_rt_memory_t* mem, u64 addr, t2 value) {          \
    t1 wrapped = (t1)value;                                                                    \
    UA_COPY(MEM_ADDR_MEMOP(mem, addr, sizeof(t1)), &wrapped, sizeof(t1), RV_AL(t1));           \
  }                                                                                            \
  DEF_MEM_CHECKS1(name##_ua, _, t1, , void, t2)
"""

M1_CALL = re.compile(r"(\b[a-z0-9_]+?)(_default32)?\(wasm_rt_local_memory_base, wasm_rt_local_memory_size, &instance->w2c_M1,")


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise SystemExit(f"FAILED: {' '.join(cmd)}\n{r.stdout[:3000]}\n{r.stderr[:3000]}")
    return r


def wasm2c(a, cdir):
    wasm = a.wasm
    os.makedirs(cdir, exist_ok=True)
    stamp = os.path.join(cdir, "stamp")
    key = f"{os.path.abspath(a.wasm)} {os.path.getmtime(a.wasm)} {a.split} {a.align != 'none'}"
    if os.path.exists(stamp) and open(stamp).read() == key:
        return
    if a.align != "none":
        wasm = os.path.join(cdir, "fx-core-al.wasm")
        print(sh([sys.executable, "-I", os.path.join(HERE, "alignsplit.py"), a.wasm, wasm]).stdout.strip())
    args = [os.path.join(os.environ["WABT"], "bin", "wasm2c.exe"), wasm, "-o", os.path.join(cdir, "fxcore.c"), "-n", "fxcore"]
    if a.split > 1:
        args.append(f"--num-outputs={a.split}")
    sh(args)
    if a.split > 1:
        g = isolate_giant(cdir, a.split)
        json.dump(g, open(os.path.join(cdir, "giant.json"), "w"))
        print("isolated:", g)
    if a.align != "none":
        n = 0
        for f in c_files(cdir, a.split):
            s = open(f, encoding="utf-8").read()
            s, k = M1_CALL.subn(lambda m: f"{m.group(1)}_ua{m.group(2) or ''}(wasm_rt_local_memory_base, wasm_rt_local_memory_size, &instance->w2c_memory,", s)
            n += k
            assert "&instance->w2c_M1, (u64)" not in s, f  # only M1's allocation/free may remain
            open(f, "w", encoding="utf-8", newline="\n").write(s)
        print(f"under-aligned call sites renamed to _ua: {n}")
    open(stamp, "w").write(key)


def c_files(cdir, split):
    if split > 1:
        fs = [os.path.join(cdir, f"fxcore_{i}.c") for i in range(split)]
        big = os.path.join(cdir, "fxcore_giant.c")
        return fs + ([big] if os.path.exists(big) else [])
    return [os.path.join(cdir, "fxcore.c")]


FUNC_HEAD = re.compile(r"^(?:static )?(?:u32|u64|s32|s64|f32|f64|void) (w2c_fxcore_[A-Za-z0-9_]+)\(w2c_fxcore\* instance", re.M)  # long headers wrap


def isolate_giant(cdir, split, min_lines=20000):
    """Moves wasm2c's largest function (fx's 138k-line orchestrator function in the
    ReleaseSmall build, cold in these runs) into fxcore_giant.c, so clang can compile it
    alone, at a lower level if it has to, while its neighbours keep the requested one."""
    best = None
    for f in c_files(cdir, split):
        txt = open(f, encoding="utf-8").read()
        heads = [(m.start(), m.group(1)) for m in FUNC_HEAD.finditer(txt)]
        for k, (st, name) in enumerate(heads):
            end = heads[k + 1][0] if k + 1 < len(heads) else len(txt)
            lines = txt.count("\n", st, end)
            if best is None or lines > best[0]:
                best = (lines, f, name, st, end)
    if not best or best[0] < min_lines:
        return None
    lines, f, name, st, end = best
    txt = open(f, encoding="utf-8").read()
    body = txt[st:end]
    head = txt[:txt.index("\n") + 1] + txt[txt.index("\n") + 1:].split("\n", 1)[0] + "\n"  # banner + include
    assert '#include "fxcore-impl.h"' in head
    open(os.path.join(cdir, "fxcore_giant.c"), "w", encoding="utf-8", newline="\n").write(head + "\n" + body)
    open(f, "w", encoding="utf-8", newline="\n").write(txt[:st] + txt[end:])
    return {"function": name, "lines": lines, "from": os.path.basename(f)}


def patch_header(a, cdir, pdir):
    """Writes the patched impl header (or single C file) into pdir. In split mode the
    patched header also replaces c/fxcore-impl.h: the split files `#include "fxcore-impl.h"`
    with quotes, which finds the file next to them before any -I directory."""
    src = os.path.join(cdir, "fxcore-impl.h" if a.split > 1 else "fxcore.c")
    orig = src + ".orig"
    if a.split > 1 and not os.path.exists(orig):
        os.replace(src, orig)
    s = open(orig if a.split > 1 else src, encoding="utf-8").read()
    k = s.index("DEFINE_LOAD(i32_load, u32, u32, u32, FORCE_READ_INT)")
    inj = UNCHECKED
    if int(a.membase, 0) == 0:
        inj += MEMBASE0
    if a.align != "none":
        inj += f"#define UA_CHECK {1 if a.align == 'check' else 0}\n" + ALIGNED
    out = os.path.join(pdir, os.path.basename(src))
    open(out, "w", encoding="utf-8", newline="\n").write(s[:k] + inj + s[k:])
    if a.split > 1:
        open(src, "w", encoding="utf-8", newline="\n").write(s[:k] + inj + s[k:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wasm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt", default="-Os")
    ap.add_argument("--cpu", default="generic_rv32+m+zba")
    ap.add_argument("--split", type=int, default=16)
    ap.add_argument("--align", choices=("none", "bytes", "check"), default="none")
    ap.add_argument("--membase", default="0x02000000")
    ap.add_argument("--cflags", default="")
    ap.add_argument("--lto", action="store_true",
                    help="ThinLTO across the split files: splitting alone loses clang's whole-module view (P0 compiled one 47 MB file)")
    ap.add_argument("--both-copies", action="store_true", help="also link fx-fastcopy.elf (rt.c with RT_FASTCOPY)")
    ap.add_argument("--clean", action="store_true", help="delete the C and object files after linking")
    ap.add_argument("--fastcopy", action="store_true", help="rt.c: word-at-a-time memcpy/memmove for any alignment")
    ap.add_argument("--fallback-opt", default="-O1,-O0", help="opt levels to retry the largest file at if clang runs out of memory")
    ap.add_argument("--big-opt", default=None, help="compile the largest file at this level directly")
    ap.add_argument("--native-tools", action="store_true", help="imports.c calls fxplat_native_tool first")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    a = ap.parse_args()
    zig = os.environ["ZIG"]
    wabt = os.environ["WABT"]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    cdir = os.path.join(out, "c")
    wasm2c(a, cdir)
    pdir = os.path.join(out, "p")
    os.makedirs(pdir, exist_ok=True)
    hdr = patch_header(a, cdir, pdir)
    sources = c_files(cdir, a.split) if a.split > 1 else [hdr]
    sh([sys.executable, "-I", os.path.join(HERE, "gen_imports.py"), os.path.join(cdir, "fxcore.h"),
        os.path.join(out, "imports.c"), os.path.join(out, "imports.json")] + (["--native-tools"] if a.native_tools else []))
    base = int(a.membase, 0)
    # Image after linear memory, C stack under 128 MB (rv32emu's memory). P0's layout put the
    # image at 0x10000, below linear memory, which caps it at 28 MB.
    text_base, cstack = (base + 0x02000000, 0x07FFFFF0)
    lin_end = base + 0x02000000
    t = ["-target", "riscv32-linux-musl", f"-mcpu={a.cpu}", "-fno-stack-protector", "-ffunction-sections",
         "-fdata-sections", "-DNDEBUG", "-fno-delete-null-pointer-checks"]
    inc = [f"-I{pdir}", f"-I{cdir}", f"-I{os.path.join(wabt, 'include')}"]
    w2c = ["-DWASM_RT_MEMCHECK_BOUNDS_CHECK=1", "-DWASM_RT_NONCONFORMING_UNCHECKED_STACK_EXHAUSTION=1"]
    extra = a.cflags.split()
    lay = [f"-DLINEAR_BASE_ADDR={base:#x}u", f"-DCSTACK_TOP={cstack:#x}"]
    jobs = []
    objs = []
    key = json.dumps(["v3", a.opt, a.cpu, a.cflags, a.lto, open(hdr, encoding="utf-8").read()[-20000:]])
    keyf = os.path.join(out, "objkey.json")
    reuse = os.path.exists(keyf) and open(keyf).read() == key
    for src in sources:
        o = os.path.join(out, os.path.splitext(os.path.basename(src))[0] + ".o")
        objs.append(o)
        if reuse and os.path.exists(o) and os.path.getmtime(o) > os.path.getmtime(src):
            jobs.append(None)  # up to date
        else:
            lto = ["-flto=thin"] if a.lto and not src.endswith("fxcore_giant.c") else []
            jobs.append([zig, "cc", *t, *inc, *w2c, a.opt, *lto, *extra, "-c", src, "-o", o])
    o = os.path.join(out, "rt.o")
    fc = ["-DRT_FASTCOPY=1"] + (["-DRT_UNALIGNED=1"] if "unaligned" in a.cpu else [])
    rt_args = [zig, "cc", *t, *inc, *w2c, *lay, "-O2", "-ffreestanding", "-fno-builtin", "-c", os.path.join(HERE, "rt.c")]
    jobs.append(rt_args[:-2] + (fc if a.fastcopy else []) + rt_args[-2:] + ["-o", o])
    objs.append(o)
    rt_fc = os.path.join(out, "rt-fastcopy.o")
    if a.both_copies:
        jobs.append(rt_args[:-2] + fc + rt_args[-2:] + ["-o", rt_fc])
    o_rt = o
    o = os.path.join(out, "imports.o")
    jobs.append([zig, "cc", *t, *inc, *w2c, "-O2", "-c", os.path.join(out, "imports.c"), "-o", o])
    objs.append(o)
    # Memory: clang needs several GB for wasm2c's largest function (one 138k-line
    # orchestrator function, never executed in these runs). Compile the largest file
    # alone first; if clang runs out of memory there, retry it at --fallback-opt and
    # record that (fallbacks.json).
    big = max(range(len(sources)), key=lambda i: os.path.getsize(sources[i]))
    fallbacks = []
    if jobs[big] is not None:
        tries = ([a.big_opt] if a.big_opt else [a.opt]) + [o for o in a.fallback_opt.split(",") if o]
        for k, o in enumerate(tries):
            try:
                sh([x if x != a.opt else o for x in jobs[big]])
            except SystemExit as e:
                if "out of memory" not in str(e) or k == len(tries) - 1:
                    raise
                continue
            if o != a.opt:
                fallbacks.append({"file": os.path.basename(sources[big]), "opt": o})
            break
    rest = [j for i, j in enumerate(jobs) if i != big and j is not None]
    with cf.ThreadPoolExecutor(a.j) as ex:
        list(ex.map(sh, rest))
    if jobs[big] is not None:
        json.dump(fallbacks, open(os.path.join(out, "fallbacks.json"), "w"))
    open(keyf, "w").write(key)
    ld = os.path.join(out, "link.ld")
    img_limit = cstack - 0x400000
    lds = open(os.path.join(HERE, "link.ld")).read()
    lds = lds.replace(". = 0x00010000;", f". = {text_base:#x};")
    lds = lds.replace('ASSERT(_end < 0x01C00000, "image overlaps the stack")', f'ASSERT(_end < {img_limit:#x}, "image overlaps the stack")')
    if not base:
        lds = lds.replace("ENTRY(_start)", f"ENTRY(_start)\n/* linear memory [0, {lin_end:#x}) */")
    open(ld, "w", newline="\n").write(lds)
    elf = os.path.join(out, "fx-fastcopy.elf" if a.fastcopy else "fx.elf")
    lto = ["-flto=thin"] if a.lto else []  # zig cc rejects --thinlto-jobs and --lto-O*
    link = [zig, "cc", "-target", "riscv32-freestanding-none", f"-mcpu={a.cpu}", "-O2", "-fno-sanitize=undefined",
            "-nostartfiles", "-Wl,--gc-sections",
            *lto, f"-Wl,-T,{ld}"]
    sh(link + objs + ["-o", elf])
    print(elf)
    if a.both_copies:
        elf2 = os.path.join(out, "fx-fastcopy.elf")
        sh(link + [rt_fc if x == o_rt else x for x in objs] + ["-o", elf2])
        print(elf2)
    if a.clean:  # disk: keep the images, imports.json and the build notes only
        for f in os.listdir(out):
            if f.endswith(".o"):
                os.remove(os.path.join(out, f))
        for f in os.listdir(cdir):
            if f.endswith((".c", ".h", ".orig", ".wasm")):
                os.remove(os.path.join(cdir, f))
        open(os.path.join(cdir, "stamp"), "w").write("cleaned")


if __name__ == "__main__":
    main()
