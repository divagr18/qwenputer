"""fx on the paged machine: the reference, Executor v4 free-running against it, and timing.

    python -m chipformer.rv.fxrun LOG.npz ELF ref                 # the reference alone, on the CPU
    python -m chipformer.rv.fxrun LOG.npz ELF run [--from-reset] [--frames N] [--max-tokens N]
    python -m chipformer.rv.fxrun LOG.npz ELF tf  [--n 4000]      # the residual against the IR

LOG.npz comes from fx/rv/snaplog.py (a session on rv32emu): the host's side of every import and
the machine state at boot's end. Between ecalls the CPU runs alone: the GPU generates every
token, and each must equal the reference machine's. At an ecall the circuit prints its fault
record and halts; the host then does what an OS does: it serves the import from the log and
resumes at PC + 4 (registers and the bytes it wrote are poked into the state, then GO). The real
I/O path is not here (it is the bridge's): this is what the run needs to reach the code after
each call. The registers are compared with rv32emu's at every ecall, and all of memory at the end.
"""

import argparse
import json
import pickle
import time

import numpy as np

from chipformer.q35.ir import HD
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S

ROM_BIT, WBITS = 25, 21  # the row-11 / row-24 images: code at 0x02000000
ROM_KH = 224  # 224 << 13 = 1.835M words (row 24: 1.743M)
FRAMES = 1024  # the measured session maps 704 pages (docs/fx/memory.md)
RW = ((0, 22), (0x07FC0000, 18), (0x026C0000, 18), (0x02700000, 18))  # linear, C stack, .bss


def windows(ad_words: np.ndarray, data_segs, code_end: int) -> tuple:
    """The image windows: every 32 KB window with a nonzero word outside the code, and every one
    the ELF's data covers. (.rodata follows .text: the ROM's address range may hold data, as
    fetches and loads are separate.)"""
    rom_lo = 1 << ROM_BIT
    ad = ad_words.astype(np.int64) << 2
    out = {int(x) for x in np.unique(ad[(ad < rom_lo) | (ad >= code_end)] >> S.WIN_BITS)}
    for va, blob in data_segs:
        out |= set(range(va >> S.WIN_BITS, (va + len(blob) - 1 >> S.WIN_BITS) + 1))
    return tuple(sorted(out))


def fx_cfg(image, frames=FRAMES, rom_kh=ROM_KH) -> S.MemCfg:
    return S.MemCfg("fx", ROM_BIT, WBITS, rom_kh, RW, frames, image)


def load(log_path, elf_path, from_reset=False, frames=FRAMES):
    """(cfg, a PagedMachine at the start, the log's entries left to serve, the log's end state)."""
    z = np.load(log_path)
    log = pickle.loads(z["log"].tobytes())
    entry, segs = elf.segments(elf_path)
    rom_seg = [(va, b) for va, b in segs if va == 1 << ROM_BIT]
    assert len(rom_seg) == 1
    code_end = (1 << ROM_BIT) + len(rom_seg[0][1])
    data = [(va, b) for va, b in segs if va != 1 << ROM_BIT]
    if from_reset:
        cfg = fx_cfg(windows(np.zeros(0, np.int64), data, code_end), frames)
        m = ref.PagedMachine.from_segments(cfg, entry, rom_seg + data, x=[0] * 32, code_end=code_end)
        m.code_end = code_end
        return cfg, m, log, z
    ad, val = z["boot_ad"], z["boot_val"]
    cfg = fx_cfg(windows(ad, data, code_end), frames)
    m = ref.PagedMachine.from_segments(cfg, entry, rom_seg, x=[int(v) for v in z["boot_x"]],
                                       code_end=code_end)
    m.code_end = code_end
    for a, v in zip(ad.tolist(), val.tolist()):
        if not cfg.rom_base <= a << 2 < code_end:
            m.image[a] = v
    pc = int(z["boot_pc"])
    m.anchor, m.n = (pc - cfg.rom_base) >> 2, 0  # (the host's resume: see serve)
    # the machine stands on the ecall of log entry boot_index
    return cfg, m, log[int(z["boot_index"]):], z


def serve(m, e) -> dict:
    """The host's side of one import on the reference machine; the words and pages it touched."""
    touched = set()
    for addr, b in e["w"]:
        for a in range(addr >> 2, (addr + len(b) - 1 >> 2) + 1):
            w0 = a << 2
            old = m.ram_word(w0).to_bytes(4, "little")
            nb = bytes(b[w0 + i - addr] if addr <= w0 + i < addr + len(b) else old[i] for i in range(4))
            pg = w0 >> S.PAGE_BITS
            if pg not in m.pt:
                m.nf += 1
                assert m.nf < m.cfg.frames, "out of frames"
                m.pt[pg] = m.nf
            m.wr[a] = int.from_bytes(nb, "little")
            touched.add(a)
    if "a0" in e:
        m.x[10] = e["a0"]
    return touched


def segments(m, log, max_tokens=None, boot_stands_on_ecall=True):
    """Reference segments: yields (tokens from GO, the fault / exit), serving the log in between."""
    i = 0
    total = 0
    if boot_stands_on_ecall and log and m.pc == log[0]["pc"]:
        assert m.x == log[0]["x"], "boot registers"
        yield None, serve(m, log[0])
        m.n += 1
        i = 1
    while True:
        toks = [S.tok("GO")]
        while not m.halted:
            toks += m.record()
        total += len(toks)
        yield toks, None
        if m.fault is None or S.FAULTS[m.fault[0]] != "ecall":
            return
        a7 = m.x[17]
        m.halted = False
        pc = m.fault[1]
        m.fault = None
        m.steps -= 1  # (an ecall the host serves is one instruction; counted below)
        if a7 < 0x400:
            assert a7 in (95, 96, 97), a7
            m.n += 1
            m.steps += 1
            yield None, set()
        else:
            e = log[i]
            i += 1
            assert e["pc"] == pc and e["x"] == m.x, (i, hex(pc), hex(e["pc"]))
            if e.get("exit"):
                return
            t = serve(m, e)
            m.n += 1
            m.steps += 1
            yield None, t
        if max_tokens and total >= max_tokens:
            return


def state_rows(c, m, touched, known_pages: set) -> list:
    """The rows a resume writes: the core registers, and each word and page the host touched."""
    core.snapshot_regs(c, m)
    rows = []
    for nm in ("pc", "pch", "count", "j", "vcnt", "regs.rs1", "regs.rs2", "regs.rd", "palloc"):
        for x, d in enumerate(c.tables[nm].init):
            rows += [(nm, x, r, v) for r, v in d.items()]
    cfg = c.cfg
    FB = (cfg.frames - 1).bit_length()
    for pg in {a >> (S.PAGE_BITS - 2) for a in touched} - known_pages:
        known_pages.add(pg)
        f = m.pt[pg]
        ea = pg << S.PAGE_BITS
        wi = cfg.rw_window(ea)
        p = (ea - cfg.rw[wi][0]) >> S.PAGE_BITS
        row = np.zeros(HD)
        row[0:2] = core.A_REF
        for i in range(FB):
            row[2 + i] = core.A_BIT * (f >> i & 1)
        row[2 + FB] = core.A_BIT
        nx = 2 if cfg.rw[wi][1] - S.PAGE_BITS >= 8 else 1
        rows.append((f"pt{wi}.{p >> 8}", (p >> 7 & 1) if nx == 2 else 0, p & 127, row))
    for a in touched:
        f = m.pt[a >> (S.PAGE_BITS - 2)]
        v = m.wr[a]
        row = np.zeros(HD)
        row[0:2] = core.A_REF
        for i in range(32):
            row[2 + i] = core.A_BIT * (v >> i & 1)
        row[66] = core.A_BIT
        rows.append((f"ram{f >> 1}", f & 1, a & 127, row))
    return rows


def end_check(m, z) -> dict:
    """The machine's memory and registers against rv32emu's at the end of the log."""
    ad, val = z["end_ad"], z["end_val"]
    bad = 0
    for a, v in zip(ad.tolist(), val.tolist()):
        if not m.cfg.rom_base <= a << 2 < m.code_end and m.ram_word(a << 2) != v:
            bad += 1
    # and nothing nonzero that rv32emu does not have
    have = set(ad.tolist())
    extra = sum(1 for a, v in m.wr.items() if v and a not in have)
    return {"end_words": len(ad), "end_diff": bad, "extra": extra,
            "end_x_equal": bool(m.x == [int(x) for x in z["end_x"]])}


def ir_run(c, m, log, a, z, t0, res) -> None:
    """The circuit in the IR, teacher-forced on the reference's stream, over the same ecalls: every
    next token predicted (no GPU). Between segments the host's writes go into the IR's state."""
    from chipformer.q35 import ir

    index = {n.name: i for i, n in enumerate(c.nodes) if isinstance(n, ir.Head)}
    carry: dict = {}
    n_tok = seg = wrong = 0
    known: set = set()
    limit = a.max_tokens or None
    out = open(a.out, "a") if a.out else None
    for toks, touched in segments(m, log, limit, not a.from_reset):
        if toks is None:
            rows = state_rows(c, m, touched, known)
            for k in [k for k in carry if k[1] != "S"]:  # a fresh conv window after a resume
                del carry[k]
            if n_tok:
                for nm, x, r, vec in rows:
                    carry[index[nm], "S", x][r] = np.asarray(vec, np.float64).copy()
            continue
        if limit and n_tok >= limit:
            break
        seg += 1
        prev = None
        for s in range(0, len(toks), a.chunk):
            part = toks[s:s + a.chunk]
            mat = ir.evaluate(c, part, check=n_tok < a.check, carry=carry)
            pred = core.predictions(c, mat)
            if prev is not None and prev != part[0]:
                wrong += 1
            wrong += sum(p != t for p, t in zip(pred, part[1:]))
            prev = pred[-1]
            n_tok += len(part)
            if wrong:
                print(json.dumps({"wrong": wrong, "at": n_tok, "segment": seg}), flush=True)
                return
            if n_tok % (64 * a.chunk) < a.chunk:
                line = {"tokens": n_tok, "segment": seg, "instr": m.steps, "wrong": wrong,
                        "pages": len(m.pt), "s": round(time.time() - t0)}
                print(json.dumps(line), flush=True)
                if out:
                    out.write(json.dumps(line) + "\n")
                    out.flush()
    print(json.dumps({"done": True, "tokens": n_tok, "segments": seg, "instr": m.steps, "wrong": wrong,
                      "s": round(time.time() - t0)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("elf")
    ap.add_argument("mode", choices=("ref", "run", "tf", "ir", "plan"))
    ap.add_argument("--from-reset", action="store_true")
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-banks", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="ir: exactness asserts on the first N tokens")
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    t0 = time.time()
    cfg, m, log, z = load(a.log, a.elf, a.from_reset, a.frames)
    print("cfg image windows", len(cfg.image), "rom words", len(m.rom), "image words", len(m.image), flush=True)
    res = {"frames": a.frames, "from_reset": a.from_reset, "image_windows": len(cfg.image)}
    if a.mode == "ref":
        n = seg = 0
        last = None
        for toks, t in segments(m, log, a.max_tokens or None, not a.from_reset):
            if toks is not None:
                n += len(toks)
                seg += 1
                last = toks
        fams = {}
        print("segments", seg, "tokens", n, "instr", m.steps, "pages", len(m.pt), "frames used", m.nf,
              "words", len(m.wr), "s", round(time.time() - t0, 1))
        print("last", [S.name(t) for t in last[-3:]], "end", end_check(m, z) if not a.max_tokens else "")
        return
    c = core.build(cfg=cfg)
    print("built", c.stats(), round(time.time() - t0, 1), flush=True)
    res["stats"] = c.stats()
    m0 = load(a.log, a.elf, a.from_reset, a.frames)[1]
    if a.mode == "plan":
        from chipformer.rv import exec4

        for banks in ([True] if a.no_banks is False else [False]):
            ex = exec4.Exec4(c, device="cpu", v0=0, tables=(), banks=banks, plan_only=True)
            r = {"banks": banks, "phases": ex.NPH, "slots": int(ex.n_slots), "smem": int(ex.smem),
                 "D": ex.D, "value_heads_state": len(ex.vheads),
                 "state_bytes": len(ex.vheads) * HD * HD * 4,
                 "terms": sum(p["terms"] for p in ex.phase_stats),
                 "head_records": sum(p["heads"] for p in ex.phase_stats),
                 "per_phase": [(p["terms"], p["heads"]) for p in ex.phase_stats]}
            print(json.dumps(r), flush=True)
            res.setdefault("plans", []).append(r)
        if a.out:
            json.dump(res, open(a.out, "w"))
        return
    if not a.from_reset:  # the boot snapshot stands on an ecall: the host serves it first
        serve(m0, log[0])
        m0.n += 1
    core.snapshot(c, m0)
    print("snapshot", round(time.time() - t0, 1), flush=True)
    if a.mode == "ir":
        ir_run(c, m, log, a, z, t0, res)
        return
    from chipformer.rv import exec4

    ex = exec4.Exec4(c, text=None, v0=0, tables=exec4.cached_tables(c, None, 0), banks=not a.no_banks)
    print("exec4 phases", ex.NPH, "slots", ex.n_slots, "smem", ex.smem, "D", ex.D, round(time.time() - t0, 1), flush=True)
    res.update(phases=ex.NPH, slots=int(ex.n_slots), smem=int(ex.smem), S_bytes=int(ex.S.numel() * 4))
    ex.load_init()
    print("loaded", round(time.time() - t0, 1), flush=True)
    import torch

    go = ex.embed([S.tok("GO")]).contiguous()
    if a.mode == "tf":
        from chipformer.q35 import ir

        # from the start of the replay: the first segment's tokens, teacher-forced
        stream = []
        for toks, t in segments(m, log, a.n, not a.from_reset):
            if toks is not None:
                stream = toks[:a.n]
                break
        ex.reset()
        emb = ex.embed(stream)
        _, _, _, xs = ex.run(emb, mode=0, max_steps=len(stream), dbg=True)
        carry, bad, first = {}, 0, None
        core.snapshot(c, m0)
        for s in range(0, len(stream), 512):
            want = ir.evaluate(c, stream[s:s + 512], check=False, carry=carry)
            got = xs[s:s + 512]
            diff = np.nonzero((got != want).any(1))[0]
            if len(diff) and first is None:
                d = int(np.nonzero(got[diff[0]] != want[diff[0]])[0][0])
                first = (s + int(diff[0]), c.dims[d].name, float(got[diff[0], d]), float(want[diff[0], d]))
            bad += len(diff)
        print(json.dumps({"tf_positions": len(stream), "dims": ex.D, "differing": bad, "first": first}))
        return
    known = set()
    steps = gpu = 0.0
    wrong = None
    seg = 0
    pending = None
    instr0 = m.steps
    t_start = time.time()
    t_ref = 0.0
    it = segments(m, log, a.max_tokens or None, not a.from_reset)
    while True:
        tr = time.perf_counter()
        try:
            toks, touched = next(it)
        except StopIteration:
            break
        t_ref += time.perf_counter() - tr
        if toks is None:
            ex.poke(state_rows(c, m, touched, known))
            continue
        ex.reset()
        n = len(toks) - 1
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ids = ex.run(go, mode=1, max_steps=n, keep=False)
        torch.cuda.synchronize()
        gpu += time.perf_counter() - t1
        steps += n
        seg += 1
        want = np.asarray(toks[1:], np.int64)
        if len(ids) != n or (ids != want).any():
            k = int(np.argmax(ids[:n] != want[:len(ids)])) if len(ids) else 0
            wrong = {"segment": seg, "at": k, "of": n, "got": int(ids[k]) if k < len(ids) else None,
                     "want": S.name(int(want[k]))}
            print("MISMATCH", wrong, flush=True)
            break
        if seg % 20 == 0:
            print(f"seg {seg} tokens {int(steps)} instr {m.steps - instr0} "
                  f"gpu {gpu:.1f}s {steps / max(gpu, 1e-9):.0f} tok/s", flush=True)
    res.update(tokens=int(steps), instructions=int(m.steps - instr0), segments=seg, gpu_s=round(gpu, 2),
               ref_s=round(t_ref, 2), tok_s=round(steps / max(gpu, 1e-9)), wrong=wrong,
               us_step=round(1e6 * gpu / max(steps, 1), 2), pages=len(m.pt))
    if not wrong and not a.max_tokens:
        res["end"] = end_check(m, z)
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
