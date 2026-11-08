"""The live fxputer loop: fx's exact binary on Executor v4, real Qwen3.5-4B beside it on one GPU, a
P0c task from prompt to hidden check (docs/fx/live.md).

    python fx/fxio/live_chip.py DLL ELF IMPORTS MODEL_DIR OUTDIR --tasks t1_bug --reps 1 2
        [--backend hf:Qwen/Qwen3.5-4B | scripted | replay:RUN.json] [--turns 2] [--frames N]
        [--full-regs] [--no-audit] [--max-instr N] [--units all+] [--chain t1_bug t2_rename ..]
        [--text] [--probe K --base-lp BASE.pt --probe-out OUT.json]

--units: the hot-code units (chipformer/rv/hot.py) built into the circuit, firing at fixed ROM PCs.
--chain: one long session, a turn per task's prompt on one repository; every task's hidden check
runs on the final repository.
--text: the circuit with the text bridge. Qwen is its merged form (chipformer/rv/wide.py, with hf:
backends), fx parks for every model call, and Qwen's tokens go through the circuit as model input.
--probe K: at model call K run fx/fxio/text_paged.py's checks, the machine holding fx's state.

The host is rvhost's with its memory, registers and run loop replaced by chipformer.rv.fxlive
(tokens, no pokes): the same import handlers, file store and model link. rv32emu boots fx to the
first stdin read and that state is the circuit's boot snapshot (as docs/fx/memory.md).

Checks run beside the chip. None feeds anything into it:
- rv32emu in lockstep, on the same model replies: at every ecall the PC, the registers the chip
  printed and the instruction count must match, and every request body must match byte for byte.
  A difference stops the run, as does a chip token of the wrong kind. With units on this rv32emu
  runs the units' twins (fx/rv/circ_units.h, circ_loops.h), so all 31 registers still compare.
- with units on, a second rv32emu runs the original code, no units, on the same replies: PC, every
  register a unit may not leave stale, every request body, the host I/O hash chain, the repository.
- after the run, every chip token against the reference machine, offline (fxlive.audit).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(FX))  # experimental/riscv
import e2e as E  # noqa: E402  (brings fx/, fx/rv, fx/circ, fx/p0c, fx/fxio onto the path)
import delta as D  # noqa: E402
import host as H  # noqa: E402
import run as RUN  # noqa: E402
import rvhost as RH  # noqa: E402
import tasks as TK  # noqa: E402
import wasifs  # noqa: E402

from chipformer.rv import core  # noqa: E402
from chipformer.rv import fxlive as FL  # noqa: E402

MEMSZ = 0x08000000
PA0 = FL.S.NEW_FAMILIES["PA"][0]


class BootDone(BaseException):
    """The machine stands on its first stdin read (not served yet)."""


class Divergence(RuntimeError):
    """The chip and rv32emu differ."""


class Booting:
    booted = False

    def hostcall(self, ident, argp, nwords):
        if (not self.booted and self.imports[ident]["name"] == "fd_read" and self.state == "init"
                and not self.stdin and struct.unpack("<I", self.arg_bytes(argp, 1))[0] == 0):
            raise BootDone
        return super().hostcall(ident, argp, nwords)

    def mem_array(self) -> np.ndarray:
        return np.ctypeslib.as_array((ctypes.c_uint8 * MEMSZ).from_address(self.base)).view("<u4")


class TeeLink:
    """The model link; the shadow answers from what it returned."""

    def __init__(self, link, park: bool = False, probe=None) -> None:
        self.link, self.last, self.seconds, self.calls = link, None, 0.0, 0
        self.host, self.park, self.probe = None, park, probe

    def call(self, raw: bytes):
        cpu = self.host.cpu
        if self.park:
            cpu.park()
        if self.probe and self.calls == self.probe[0]:
            self.probe[1](cpu)
        t0 = time.perf_counter()
        t_text = cpu.seconds["text"]
        chunks, rec = self.link.call(raw)
        dt = time.perf_counter() - t0
        rec["chip_text_s"] = round(cpu.seconds["text"] - t_text, 3)
        if self.park:
            cpu.unpark()
        self.seconds += dt
        self.calls += 1
        rec["link_s"] = round(dt, 3)
        rec["chip_s_at_open"] = round(sum(self.host.cpu.seconds.values()), 3)
        self.last = (bytes(raw), list(chunks), dict(rec))
        return chunks, rec


class ShadowLink:
    def __init__(self, tee: TeeLink) -> None:
        self.tee, self.same = tee, 0

    def call(self, raw: bytes):
        r0, chunks, rec = self.tee.last
        if r0 != bytes(raw):
            n = next((i for i in range(min(len(r0), len(raw))) if r0[i] != raw[i]), min(len(r0), len(raw)))
            raise Divergence(f"request {self.same}: chip and rv32emu differ at byte {n}: "
                             f"{bytes(r0[n:n + 40])!r} / {bytes(raw[n:n + 40])!r}")
        self.same += 1
        return list(chunks), dict(rec)


# caller-saved registers a unit may leave stale against the original code: t0-t2, a2-a6, t3-t6
# (dead by the ABI, docs/fx/circuits2.md). Not compared with the units-off rv32emu.
DEAD_ABI = sum(1 << r for r in (5, 6, 7, 12, 13, 14, 15, 16, 28, 29, 30, 31))
# rv32emu's twins of the units built in hot.py, as fx/rv/snaplog.py --units
TWINS = "move,set,eql,bcopy,jscan,sha,tok2,chr2a,chr2b,asc2"
TWINS_ALL = TWINS + ",jesc2,maskp2"


def install_twins(dll, elf_path, names) -> None:
    import units as CU

    names = names.split(",")
    specs, skipped = CU.specs_for(elf_path, ",".join(n for n in names if n not in CU.TWINS))
    assert not skipped, skipped
    specs += CU.twin_specs(names)
    for sp in specs:
        if sp["name"] == "sha":  # as built: message words aligned
            sp["kind"] = CU.TWIN_KIND["U_SHA2"]
    RUN.install(dll, specs)


class ShadowHost(Booting, E.IoHost):
    """rv32emu on the chip's model replies, one ecall at a time. twins: its units go on at boot's
    end, where the chip starts."""

    twins = False

    def start(self) -> None:
        try:
            RH.RvHost.run(self)
        except BootDone:
            self.booted = True
        if self.twins:
            self.e.rv_units_mode(1)
        self.r, self.pc, self.fresh, self.alive = 1, self.e.rv_get_pc(), True, True

    def step(self) -> None:
        if not self.fresh:
            self.r = self.e.rv_run(1 << 40)
            self.pc = self.e.rv_get_pc()
        self.fresh = False

    def serve(self) -> None:
        self.alive = self.dispatch(self.r, self.pc)


class ChipE:
    """What rvhost's loop needs of an emulator, on the chip."""

    def __init__(self, h, cpu, shadow, boot_ic: int, max_instr: int, orig=None) -> None:
        self.h, self.cpu, self.shadow, self.boot_ic, self.orig = h, cpu, shadow, boot_ic, orig
        self.pc, self.ret, self.started, self.exited = 0, 0, False, False
        self.max_instr = max_instr
        self.ic_slack: dict = {}
        self.dead_differ = 0  # registers that differ from the original code, all in DEAD_ABI
        self.t_shadow = 0.0
        self.ecalls = 0

    def rv_icount(self) -> int:
        return self.boot_ic + self.cpu.instr

    def rv_tokens(self) -> int:
        return sum(self.cpu.tokens.values())

    def rv_get_pc(self) -> int:
        return self.pc

    def rv_set_pc(self, pc: int) -> None:
        self.ret = pc

    def rv_misaligned(self) -> int:
        return 0

    def rv_fault_addr(self) -> int:
        return 0

    def __getattr__(self, name):  # the memory tracer's calls
        return lambda *a: None

    def _shadow(self, fn) -> None:
        t0 = time.perf_counter()
        fn()
        self.t_shadow += time.perf_counter() - t0

    def _serve_shadow(self) -> None:
        for s in (self.shadow, self.orig):
            if s is None:
                continue
            s.serve()
            if s.alive and s.regs[10] != self.h.regs[10]:
                raise Divergence(f"ecall {self.ecalls}: a0 from the host {self.h.regs[10]:#x}, "
                                 f"rv32emu's {s.regs[10]:#x}")

    def _sync(self, kind, pc, regs) -> None:
        s = self.shadow
        s.step()
        if s.r != 1:
            raise Divergence(f"ecall {self.ecalls}: rv32emu stopped with {s.r}")
        if kind == "ecall" and s.pc != pc:
            raise Divergence(f"ecall {self.ecalls}: chip pc {pc:#x}, rv32emu {s.pc:#x}")
        bad = {r: (v, s.regs[r]) for r, v in regs.items() if s.regs[r] != v}
        if bad:
            raise Divergence(f"ecall {self.ecalls} at {pc}: registers {bad}")
        d = int(s.e.rv_icount()) - self.rv_icount()
        self.ic_slack[d] = self.ic_slack.get(d, 0) + 1
        o = self.orig
        if o is None:
            return
        o.step()
        if o.r != 1:
            raise Divergence(f"ecall {self.ecalls}: the original code stopped with {o.r}")
        if kind == "ecall" and o.pc != pc:
            raise Divergence(f"ecall {self.ecalls}: chip pc {pc:#x}, the original code {o.pc:#x}")
        bad = {r: (v, o.regs[r]) for r, v in regs.items() if o.regs[r] != v and not DEAD_ABI >> r & 1}
        if bad:
            raise Divergence(f"ecall {self.ecalls} at {pc}: registers against the original code {bad}")
        self.dead_differ += sum(o.regs[r] != v for r, v in regs.items())

    def rv_run(self, _n) -> int:
        h, cpu = self.h, self.cpu
        if self.started:
            if self.shadow:
                self._shadow(self._serve_shadow)
            if self.max_instr and cpu.instr >= self.max_instr:
                raise SystemExit  # (a capped run, for tests)
            chip, stop = cpu.resume(self.ret, h.regs[FL.A7], h.regs[FL.A0])
        else:
            chip, stop = cpu.go()
            self.started = True
        kind, pc = cpu.parse_stop(chip, stop)
        regs = cpu.enter()
        for r, v in regs.items():
            h.regs[r] = v
        self.pc = pc or 0
        self.ecalls += 1
        if self.shadow:
            self._shadow(lambda: self._sync(kind, pc, regs))
        if kind == "exit":
            self.exited = True
            assert regs[FL.A7] == 93, regs
        return 1


class ChipHost(Booting, E.IoHost):
    """rvhost's host, with the machine behind tokens once it has booted."""

    cpu = None

    def attach_chip(self, chip, shadow, max_instr, orig=None) -> None:
        self.chip, self.shadow, self.max_instr, self.orig = chip, shadow, max_instr, orig

    def rd(self, p, n):
        return self.cpu.read(self.lin + p, n) if self.cpu else super().rd(p, n)

    def wr(self, p, b):
        if self.cpu:
            self.cpu.write(self.lin + p, bytes(b))
        else:
            super().wr(p, b)

    def arg_bytes(self, argp, nwords):
        return self.cpu.read(argp, 4 * nwords) if self.cpu else super().arg_bytes(argp, nwords)

    def linear_bytes(self):
        return 0 if self.cpu else super().linear_bytes()

    def hand_over(self) -> None:
        e = self.e
        pc = e.rv_get_pc()
        mem = self.mem_array()
        ad = np.nonzero(mem)[0].astype(np.uint32)
        val = mem[ad].copy()
        x = [self.regs[i] for i in range(32)]
        self.boot = {"ad": ad, "val": val, "x": x, "pc": pc, "ic": e.rv_icount()}
        self.booted = True
        self.cpu = self.chip.cpu_for(self.boot)
        self.emu_regs, self.regs = self.regs, [0] * 32
        self.e = ChipE(self, self.cpu, self.shadow, self.boot["ic"], self.max_instr, self.orig)

    def run(self):
        try:
            RH.RvHost.run(self)
        except BootDone:
            pass
        self.hand_over()
        RH.RvHost.run(self)


class Chip:
    """The circuit, its kernel and the boot snapshot, kept across runs."""

    def __init__(self, elf: str, frames: int, full_regs: bool, log=print, ref: bool = False, units=(),
                 model_dir: str | None = None) -> None:
        self.elf, self.frames, self.full, self.log, self.ref = elf, frames, full_regs, log, ref
        self.units = units
        self.model_dir = model_dir  # the text bridge: Qwen's tokenizer and config
        self.go_on_tcc = False  # fx's bridge: GO comes from the host's unpark (docs/fx/fwd.md)
        self.ex, self.build_s = None, 0.0
        self.bios = FL.Bios(FL.FX_IO_BASE)

    def _build(self, b) -> None:
        t0 = time.time()
        self.cfg, m = FL.machine_at(self.elf, b["ad"], b["val"], b["x"], b["pc"], self.frames, self.bios, self.units)
        self.boot = b
        self.boot_hash = self._hash(b)
        if self.ref:
            self.ex = True
            self.build_s = time.time() - t0
            return
        from chipformer.rv import exec4

        T, v0, target = None, 0, None
        if self.model_dir:
            from chipformer.rv import bridge as BR
            from chipformer.rv import wide

            T = self.T = BR.Text(os.path.join(self.model_dir, "tokenizer.json"))
            v0 = json.load(open(os.path.join(self.model_dir, "config.json")))["text_config"]["vocab_size"]
            target = wide.target_of(self.model_dir)
        self.c = core.build(target, T, self.cfg, go_on_tcc=self.go_on_tcc)
        self.log("built", self.c.stats(), round(time.time() - t0, 1))
        core.snapshot(self.c, m)
        self.ex = exec4.Exec4(self.c, text=T, v0=v0, tables=exec4.cached_tables(self.c, T, v0))
        self.log("exec4 phases", self.ex.NPH, "slots", self.ex.n_slots, "smem", self.ex.smem, round(time.time() - t0, 1))
        self.ex.load_init()
        FL.install_stops(self.ex)
        self.S0 = self.ex.S.clone()
        self.build_s = time.time() - t0

    @staticmethod
    def _hash(b) -> str:
        h = hashlib.sha256()
        for a in (b["ad"], b["val"], np.asarray(b["x"], np.uint32)):
            h.update(np.asarray(a).tobytes())
        h.update(struct.pack("<I", b["pc"]))
        return h.hexdigest()

    def cpu_for(self, b):
        if self.ex is None:
            self._build(b)
        assert self._hash(b) == self.boot_hash, "boot state differs between runs"
        if self.ref:
            return FL.LiveCPU(FL.RefBackend(self.machine()), self.bios, self.cfg, self.full)
        self.ex.S.copy_(self.S0)
        self.ex.reset()
        return FL.LiveCPU(FL.ExecBackend(self.ex), self.bios, self.cfg, self.full)

    def wide(self) -> tuple:
        """Qwen's merged layers for this circuit, once."""
        if getattr(self, "_wide", None) is None:
            from chipformer.rv import wide

            self._wide = wide.wide_chat(self.model_dir, self.c, self.T)
        return self._wide

    def machine(self):
        b = self.boot
        return FL.machine_at(self.elf, b["ad"], b["val"], b["x"], b["pc"], self.frames, self.bios, self.units)[1]


def stable_inodes(h, tmp) -> None:
    """A deterministic inode, as e2e / circ/run.py."""
    root = os.path.dirname(os.path.abspath(tmp))

    def _filestat(path, out, fs=h.fs):
        try:
            st = os.stat(path)
        except OSError:
            return wasifs.ENOENT
        ino = zlib.crc32(os.path.relpath(path, root).replace(os.sep, "/").encode())
        ft = wasifs.FT_DIR if os.path.isdir(path) else wasifs.FT_REG
        fs.h.wr(out, struct.pack("<QQBxxxxxxxQQQQQ", 1, ino, ft, 1, st.st_size, 1_767_000_000_000_000_000,
                                 1_767_000_000_000_000_000, 1_767_000_000_000_000_000))
        return wasifs.ESUCCESS

    h.fs._filestat = _filestat


def make_host(cls, dll_src, a, imports, workdir, name, link, twins=False):
    so = os.path.join(workdir, f"{name}.so")
    shutil.copy(dll_src, so)  # a host's emulator state lives in its own copy of the library
    dll = RUN.setup_dll(ctypes.CDLL(os.path.abspath(so)))
    repo = H.repo_dir(tempfile.mkdtemp(prefix=f"fxlive-{name}-", dir=workdir))
    E.write_repo(repo)
    h = cls(dll, a.elf, imports, repo, a.turns, log=lambda *x: None, prof=False, memtrace=False)
    stable_inodes(h, repo)
    h.attach(link)
    if twins:
        install_twins(dll, a.elf, TWINS_ALL if a.twins_all else TWINS)
        h.twins = True
    dll.rv_units_mode(0)
    rec = RUN.Recorder(h)
    return h, rec, repo


def session(a, chip: Chip, template, backend, task: str, rep: int, outdir: str, log=print) -> dict:
    probe = None
    if a.probe >= 0 and a.text:
        import text_paged as TP

        probe = (a.probe, lambda cpu: TP.probe(*chip.wide(), cpu, chip.ex, chip.c, a.base_lp, a.probe_out, log))
    H.set_opts(**RH.parse_opts(E.GW + ["native=1", "fs=repo"]))
    H.PROMPTS[:] = ([TK.TASKS[t]["prompt"] for t in a.chain] if a.chain
                    else [TK.TASKS[task]["prompt"], E.SECOND_PROMPT][:a.turns])
    work = tempfile.mkdtemp(prefix="fxlive-", dir=a.tmp)
    if backend is None:
        replay = a.backend.partition(":")[2].format(task=task, rep=rep)
        link = E.ReplayLink(replay)
    elif a.backend == "scripted":
        import scripted as SC

        link = D.DeltaLink(template, SC.ScriptedBackend(template, task, extra=a.turns - 1))
    else:
        backend.reset()
        backend.gen.manual_seed(rep)
        for k in backend.seconds:
            backend.seconds[k] = 0.0
        link = D.DeltaLink(template, backend)
    tee = TeeLink(link, park=bool(a.text), probe=probe)
    imports = json.load(open(a.imports))
    shadow, srec, srepo = None, None, None
    orig, orec, orepo = None, None, None
    slink, olink = ShadowLink(tee), ShadowLink(tee)
    if not a.no_shadow:
        shadow, srec, srepo = make_host(ShadowHost, a.dll, a, imports, work, "emu", slink, twins=bool(a.units))
        shadow.start()
        if a.units:
            orig, orec, orepo = make_host(ShadowHost, a.dll, a, imports, work, "orig", olink)
            orig.start()
    h, rec, repo = make_host(ChipHost, a.dll, a, imports, work, "chip", tee)
    h.attach_chip(chip, shadow, a.max_instr, orig)
    tee.host = h
    torch = None
    if not chip.ref:
        import torch

        torch.cuda.reset_peak_memory_stats()
    if backend is not None and a.text:
        backend.cpu_of = lambda: h.cpu
    stopped, t0, built0 = None, time.perf_counter(), chip.build_s
    try:
        h.run()
        if shadow and shadow.alive:
            h.e._shadow(shadow.serve)
        if orig and orig.alive:
            h.e._shadow(orig.serve)
    except KeyboardInterrupt:
        raise
    except BaseException as e:  # (a wrong token, a divergence, anything: reported, not hidden)
        stopped = f"{type(e).__name__}: {e}"
        log("STOPPED", stopped)
    wall = time.perf_counter() - t0
    cpu, e = h.cpu, h.e
    res = {"task": task, "rep": rep, "backend": a.backend, "elf": os.path.basename(a.elf),
           "elf_sha256": hashlib.sha256(open(a.elf, "rb").read()).hexdigest(), "stopped": stopped,
           "frames": a.frames, "turns": a.turns, "units": a.units, "chain": a.chain}
    ok, why = E.solved(repo, task) if not stopped else (False, "")
    if a.chain and not stopped:
        each = {t: E.solved(repo, t) for t in a.chain}
        ok, why = all(v[0] for v in each.values()), {t: v[1] for t, v in each.items()}
    files = {r: open(os.path.join(repo, r), encoding="utf-8").read() for r in h.repo.files() if "__pycache__" not in r}
    res.update(solved=ok, check=why, steps=len(h.calls_log), model_calls=h.calls_log, final_repo=files,
               final_repo_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
               packets_hex=b"".join(h.packets).hex(), host_calls=rec.calls, host_io_sha256=rec.sha.hexdigest(),
               import_calls=h.import_calls, unsupported_imports=h.unsupported, native_calls=h.native_calls,
               trap=[x for x in h.events if x[1] == "trap"],
               boot={"instructions": h.boot["ic"], "state_sha256": chip.boot_hash})
    b = getattr(tee.link, "b", None)
    qsec = dict(b.seconds) if b is not None and hasattr(b, "seconds") else {}
    t_model = tee.seconds
    res["chip"] = {"tokens": dict(cpu.tokens), "seconds": {k: round(v, 3) for k, v in cpu.seconds.items()},
                   "launches": dict(cpu.launches), "instructions": cpu.instr,
                   "ecalls": e.ecalls, "pages_host_touched": len(cpu.x),
                   "forced_positions_checked": cpu.ok_forced,
                   "tok_s_fx": round(cpu.tokens["fx"] / max(cpu.seconds["fx"], 1e-9))}
    res["qwen"] = {"seconds": {k: round(v, 3) for k, v in qsec.items()}, "link_s": round(t_model, 3),
                   "generated": sum(r.get("generated_tokens", 0) for r in h.calls_log),
                   "fed": sum(r.get("delta_tokens", 0) for r in h.calls_log),
                   "rollbacks": sum(r.get("kind") == "rollback" for r in h.calls_log),
                   "snapshot_bytes": getattr(b, "snapshot_bytes", 0)}
    t_cpu = sum(v for k, v in cpu.seconds.items() if k != "text")  # (the text's is in the model link's)
    res["host_s"] = round(wall - t_cpu - t_model - e.t_shadow - (chip.build_s - built0), 3)  # (not the circuit's build)
    res["shadow_s"] = round(e.t_shadow, 3)
    res["wall_s"] = round(wall, 3)
    res["gpu_peak_bytes"] = torch.cuda.max_memory_allocated() if torch else 0
    res["gpu_reserved_peak_bytes"] = torch.cuda.max_memory_reserved() if torch else 0
    if shadow:
        res["emu"] = {"requests_identical": slink.same, "instr": int(shadow.e.rv_icount()),
                      "ic_minus_chip": {str(k): v for k, v in e.ic_slack.items()},
                      "host_calls": srec.calls, "host_io_sha256": srec.sha.hexdigest(),
                      "host_io_equal": srec.sha.hexdigest() == rec.sha.hexdigest()}
        sfiles = {r: open(os.path.join(srepo, r), encoding="utf-8").read() for r in shadow.repo.files()
                  if "__pycache__" not in r}
        res["emu"]["final_repo_equal"] = sfiles == files
        if orig:
            ofiles = {r: open(os.path.join(orepo, r), encoding="utf-8").read() for r in orig.repo.files()
                      if "__pycache__" not in r}
            res["orig"] = {"requests_identical": olink.same, "instr": int(orig.e.rv_icount()),
                           "host_calls": orec.calls, "host_io_equal": orec.sha.hexdigest() == rec.sha.hexdigest(),
                           "final_repo_equal": ofiles == files, "registers_differing_dead": e.dead_differ,
                           "memory": orig_memory(shadow, orig)}
        if not stopped and not a.max_instr:
            res["end"] = end_check(cpu, shadow, chip, h.boot, a)
    if not a.no_audit:
        toks, lens, forced = cpu.fed()
        res["chip"]["pages_mapped"] = int(np.sum((toks >= PA0) & (toks < PA0 + 4096)))
        np.savez_compressed(os.path.join(outdir, f"tokens_{task}_{rep}.npz"), tokens=toks, lengths=lens, forced=forced)
        t1 = time.perf_counter()
        m = chip.machine()
        r = FL.audit(m, toks, lens, forced)
        r["seconds"] = round(time.perf_counter() - t1, 1)
        if shadow and not r["wrong"]:
            mem = shadow.mem_array()
            r["memory_words_compared"] = len(m.wr)
            r["memory_diff"] = sum(1 for k, v in m.wr.items() if int(mem[k]) != v)
        res["audit"] = r
    shutil.rmtree(work, ignore_errors=True)
    return res


def orig_memory(twin, orig) -> dict:
    """rv32emu with units against rv32emu on the original code: which words differ at the end."""
    a, b = twin.mem_array(), orig.mem_array()
    d = np.nonzero(a != b)[0].astype(np.int64) << 2
    stack = (d >= 0x07FC0000) & (d < 0x08000000)
    sp = int(orig.regs[2])
    return {"words_differing": len(d), "on_the_c_stack": int(stack.sum()),
            "on_the_c_stack_above_sp": int((d[stack] >= sp).sum()),
            "elsewhere": [hex(x) for x in d[~stack][:16].tolist()], "elsewhere_n": int((~stack).sum())}


def end_check(cpu, shadow, chip, boot, a) -> dict:
    """The end of the run: all registers and every page rv32emu changed, printed by the chip."""
    regs = cpu.enter(full=True)
    bad_regs = {r: (v, shadow.regs[r]) for r, v in regs.items() if shadow.regs[r] != v}
    mem = shadow.mem_array().copy()
    linear = np.nonzero(mem[: 0x02000000 >> 2])[0]
    top = int(linear[-1]) << 2 if len(linear) else 0
    base = np.zeros(len(mem), np.uint32)
    base[boot["ad"]] = boot["val"]
    diff = np.nonzero(mem != base)[0]
    pages = np.unique(diff >> 7).tolist()
    bad = 0
    for pg in pages:
        got = cpu.read_words(pg << 9, 128, kind="audit")
        bad += int((np.asarray(got, np.uint32) != mem[pg << 7:(pg + 1) << 7]).sum())
    return {"registers_differing": len(bad_regs), "registers": {str(k): v for k, v in bad_regs.items()},
            "pages_changed_by_rv32emu": len(pages), "words_differing": bad, "words_read": 128 * len(pages),
            "linear_top": top}


def main():
    ap = argparse.ArgumentParser()
    for x in ("dll", "elf", "imports", "model_dir", "out"):
        ap.add_argument(x)
    ap.add_argument("--tasks", nargs="+", default=["t1_bug"])
    ap.add_argument("--reps", nargs="+", type=int, default=[1])
    ap.add_argument("--backend", default="hf:Qwen/Qwen3.5-4B")
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--turns", type=int, default=1, choices=(1, 2))
    ap.add_argument("--chain", nargs="*", default=[], help="one session: a turn per task, in order")
    ap.add_argument("--units", default="", help="unit names (hot.FX_R24), comma-separated, or all+")
    ap.add_argument("--full-regs", action="store_true")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--max-instr", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--cpu-ref", action="store_true", help="the reference machine as the chip: tests the host, no GPU")
    ap.add_argument("--text", action="store_true", help="the circuit with the text bridge; Qwen in merged form")
    ap.add_argument("--probe", type=int, default=-1, help="text_paged's checks at this model call")
    ap.add_argument("--base-lp", default="")
    ap.add_argument("--probe-out", default="probe.json")
    ap.add_argument("--go-on-tcc", action="store_true",
                    help="--text: the old bridge, </tool_call> predicts GO (docs/fx/fwd.md: the before)")
    a = ap.parse_args()
    a.tmp = a.tmp or tempfile.gettempdir()
    a.units = [u for u in a.units.split(",") if u]
    a.twins_all = "all+" in a.units or "jesc" in a.units
    if a.chain:
        a.turns = len(a.chain)
    os.makedirs(a.out, exist_ok=True)
    template = None if a.backend.startswith("replay:") else D.QwenTemplate(a.model_dir)
    backend = None
    log = lambda *x: print(*x, flush=True)
    chip = Chip(a.elf, a.frames, a.full_regs, log, a.cpu_ref, a.units, a.model_dir if a.text else None)
    chip.go_on_tcc = a.go_on_tcc
    if a.backend.startswith("hf:") and a.text:
        import qwen_wide as QW

        backend = QW.WideBackend(chip, a.model_dir, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=0)
    elif a.backend.startswith("hf:"):
        import qwen_hf as QH

        t0 = time.time()
        backend = QH.HfBackend.load(a.backend[3:], temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=0)
        print("qwen loaded", round(time.time() - t0, 1), flush=True)
    for rep in a.reps:
        for task in a.tasks:
            r = session(a, chip, template, backend, task, rep, a.out, log)
            r["build_s"] = round(chip.build_s, 1)
            p = os.path.join(a.out, f"live_{task}_{rep}{'_chain' if a.chain else '_t2' if a.turns == 2 else ''}.json")
            json.dump(r, open(p, "w"))
            show = {k: r.get(k) for k in ("task", "rep", "solved", "check", "stopped", "steps", "chip", "qwen",
                                          "host_s", "shadow_s", "wall_s", "gpu_peak_bytes", "end")}
            if "emu" in r:
                show["emu"] = {k: v for k, v in r["emu"].items() if k != "host_io_sha256"}
            if "orig" in r:
                show["orig"] = r["orig"]
            if "audit" in r:
                show["audit"] = r["audit"]
            log(json.dumps(show, default=str))


if __name__ == "__main__":
    main()
