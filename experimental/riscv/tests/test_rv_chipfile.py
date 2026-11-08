"""chip.safetensors (chipfile.py): the circuit lifted from the file is the circuit, and the production
path works with the circuit builder made unavailable.

The first group runs anywhere there is RAM for core.build. The last tests need a GPU, and
CHIP_Q4P2 / QWEN_BASE naming rv-q4p2's chip file and base Qwen's directory.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from chipformer.q35 import ir
from chipformer.rv import chipbuild, chipcheck, core, exec4, ref
from chipformer.rv import chipfile as F
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

ROOT = Path(__file__).resolve().parents[1]
SRC = """
    li a0, 1234
    li a1, 5678
    mul a2, a0, a1
    sw a2, 0x100(zero)
    lw a3, 0x100(zero)
    add a4, a3, a0
    li a7, 93
    ecall
"""


@pytest.fixture(scope="module")
def legacy():
    return core.build()


@pytest.fixture(scope="module")
def paged():
    return core.build(cfg=S.small_cfg(8))


@pytest.fixture(scope="module")
def prog():
    return assemble(SRC)


@pytest.fixture(scope="module")
def chip_legacy(tmp_path_factory, legacy, prog):
    p = tmp_path_factory.mktemp("chip") / "chip.safetensors"
    chipbuild.standalone(legacy, p, ref.Machine.from_program(prog))
    return p


@pytest.fixture(scope="module")
def chip_paged(tmp_path_factory, paged):
    p = tmp_path_factory.mktemp("chip") / "chip.safetensors"
    chipbuild.standalone(paged, p)
    return p


def test_lift_is_the_circuit(legacy, chip_legacy):
    r = chipcheck.lint(F.load(chip_legacy), legacy)
    assert r["lifted_dims"] == r["ir_dims"] and r["heads"] == r["heads_lifted"]
    assert r["ir_dims_with_no_lifted_dim"] == r["lifted_dims_with_no_ir_dim"] == 0
    assert r["head_diffs"] == 0 and r["unit_group_diffs"] == 0, r["first"]


def test_paged_banks_are_solved_from_the_rows(paged, chip_paged):
    cf = F.load(chip_paged)
    r = chipcheck.lint(cf, paged)
    assert r["head_diffs"] == 0 and r["unit_group_diffs"] == 0, r["first"]
    assert set(cf.banks) == set(paged.banks) and cf.banks
    assert r["bank_form_diffs"] == 0, r


def test_plan_is_the_same_size(paged, chip_paged, legacy, chip_legacy):
    for cp, path in ((legacy, chip_legacy), (paged, chip_paged)):
        cf = F.load(path)
        a = exec4.Exec4(cp, device="cpu", v0=0, tables=(), plan_only=True)
        b = exec4.Exec4(cf, device="cpu", v0=0, tables=(), plan_only=True)
        assert (a.NPH, a.D) == (b.NPH, b.D)
        assert [p["terms"] for p in a.phase_stats] == [p["terms"] for p in b.phase_stats]
        assert [p["heads"] for p in a.phase_stats] == [p["heads"] for p in b.phase_stats]


def test_tables_equal_the_python_tables(legacy, chip_legacy):
    r = chipcheck.table_check(F.load(chip_legacy), legacy, None, 0)
    assert r["lut_id_diff"] == r["lut_row_diff"] == r["emb_row_diff"] == 0, r


def test_residual_in_the_ir_equals_the_python_circuit(legacy, chip_legacy, prog):
    """ir.evaluate on the lifted circuit against ir.evaluate on the Python one, every dim."""
    cf = F.load(chip_legacy)
    ir_d, lift_d, missing, _ = chipcheck.dim_map(cf, legacy)
    toks = ref.Machine.from_program(prog).run()
    core.snapshot(legacy, ref.Machine.from_program(prog))
    core.snapshot(cf, ref.Machine.from_program(prog))
    a = ir.evaluate(legacy, toks, check=False)
    b = ir.evaluate(cf, toks, check=False)
    assert (a[:, ir_d] == b[:, lift_d]).all()
    assert not missing


def test_state_round_trip(legacy, chip_legacy, prog):
    """The file's firmware is core.snapshot's state, row for row."""
    cf = F.load(chip_legacy)
    core.snapshot(legacy, ref.Machine.from_program(prog))
    want = {(n.name, x, r): np.asarray(v, np.float32) for n in legacy.nodes
            if isinstance(n, ir.Head) and n.init for x, d in enumerate(n.init)
            for r, v in d.items() if np.asarray(v).any()}
    heads = [nm for _, _, nm in cf.file.meta["heads"]]
    h, x, r, rows = cf.state_rows()
    got = {(heads[i], int(a), int(b)): v for i, a, b, v in zip(h.tolist(), x, r, rows)}
    assert got.keys() == want.keys()
    assert all((got[k] == want[k]).all() for k in got)


def _rewrite(src: Path, dst: Path, edit) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src, framework="pt") as f:
        md = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    edit(t, md)
    md["content_hash"] = F.content_hash(t)
    save_file(t, str(dst), metadata=md)


def test_a_bad_file_is_refused(chip_paged, tmp_path):
    def forged_meta(t, md):
        md["meta"] = md["meta"].replace('"m":', '"m": ')

    p = tmp_path / "a.safetensors"
    _rewrite(chip_paged, p, forged_meta)
    with pytest.raises(F.ChipFileError, match="hash"):
        F.load(p)

    def other_version(t, md):
        md["version"] = "99"

    _rewrite(chip_paged, p, other_version)
    with pytest.raises(F.ChipFileError, match="version"):
        F.load(p)


def test_changed_arrays_are_refused(chip_paged, tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(chip_paged, framework="pt") as f:
        md = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    t["weights.values"][0] = 3.0
    p = tmp_path / "c.safetensors"
    save_file(t, str(p), metadata=md)
    with pytest.raises(F.ChipFileError, match="arrays"):
        F.load(p)


def test_a_changed_constant_is_refused(chip_paged, tmp_path):
    def fill(t, md):
        m = json.loads(md["meta"])
        m["tensors"][7][2] = -100.0  # dt_bias of layer 0
        md["meta"] = json.dumps(m, separators=(",", ":"), sort_keys=True)
        md["config_hash"] = F.config_hash(md["meta"])

    p = tmp_path / "d.safetensors"
    _rewrite(chip_paged, p, fill)
    with pytest.raises(F.ChipFileError, match="constant"):
        F.load(p)


def test_a_changed_memory_weight_is_caught(chip_paged, tmp_path):
    """Banks run from a form solved from few rows; the other rows are checked, not skipped."""
    cf = F.load(chip_paged)
    head = next(n for n in cf.by_name.values() if n.bank == "ram" and n.bank_h == 3)
    L, j = (head.level - 1) // 2, next(j for L_, j, nm in cf.file.meta["heads"] if nm == head.name)
    i = cf.file.index[f"model.layers.{L}.linear_attn.in_proj_qkv.weight"]
    r, _, _, off = cf.file.weights()
    at = off[i] + int(np.searchsorted(r[off[i]:off[i + 1]], j * 128 + 77))

    def flip(t, md):
        t["weights.values"][at] = t["weights.values"][at] * 2

    p = tmp_path / "b.safetensors"
    _rewrite(chip_paged, p, flip)
    with pytest.raises(F.ChipFileError, match="not regular|other"):
        F.load(p)


def test_production_path_needs_no_builder(chip_legacy, legacy, prog, tmp_path):
    """In a process where chipformer.rv.core cannot even be imported: the file loads, the plan is
    made, the tables are built and the lifted circuit runs a program from the file's state. The
    run uses ir.evaluate with no GPU, or Exec4 with one."""
    toks = ref.Machine.from_program(prog).run()
    np.save(tmp_path / "toks.npy", np.asarray(toks))
    code = textwrap.dedent(f"""
        import sys
        import numpy as np
        for name in ("chipformer.rv.core", "chipformer.q35.emit", "chipformer.rv.chipbuild",
                     "chipformer.q35.prims", "chipformer.rv.ref"):
            sys.modules[name] = None          # import raises
        import torch
        from chipformer.q35 import ir
        from chipformer.rv import chipfile, exec4
        c = chipfile.load({str(chip_legacy)!r})
        toks = np.load({str(tmp_path / 'toks.npy')!r}).tolist()
        lut = c.exec_tables()[0]
        if torch.cuda.is_available():
            ex = exec4.Exec4(c, text=None, v0=c.v0, tables=c.exec_tables())
            c.load_state(ex)
            ex.reset()
            got = ex.run(ex.embed(toks[:1]).contiguous(), mode=1, max_steps=len(toks) - 1,
                         keep=False).tolist()
        else:
            exec4.Exec4(c, device="cpu", v0=c.v0, tables=c.exec_tables(), plan_only=True)
            c.apply_state()
            mat = ir.evaluate(c, toks, check=False)
            bits = (mat[:, c.keep[:c.ncode]] > 0).astype(np.int64)
            got = lut[c._code_index(bits)][:-1].tolist()
        assert got == toks[1:], [i for i, (a, b) in enumerate(zip(got, toks[1:])) if a != b][:3]
        print("OK", len(got))
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-1500:]


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("CHIP_Q4P2")
                         and os.environ.get("QWEN_BASE")), reason="needs a GPU and rv-q4p2's file")
def test_rvchat_runs_from_the_file_beside_base():
    """RvChat on base Qwen's own files and the chip file, in a process where the circuit builder
    and the emitter cannot be imported and the state comes from the file: Qwen calls calc, the
    circuit answers, and the digits are right."""
    code = textwrap.dedent("""
        import os
        import sys
        for name in ("chipformer.rv.core", "chipformer.q35.emit", "chipformer.rv.chipbuild",
                     "chipformer.q35.prims", "chipformer.rv.hf"):
            sys.modules[name] = None          # import raises
        from chipformer.rv.chat import RvChat
        chat = RvChat(os.environ["QWEN_BASE"], executor=True, chip=os.environ["CHIP_Q4P2"])
        assert sum(p.numel() for p in chat.model.parameters()) * 2 < 9e9
        ids = chat.prompt_ids([{"role": "user",
                                 "content": "Use calc: what is 123456789 * 987654321?"}])
        r = chat.generate(ids, None, 300, 60000)
        text = chat.text_of(r["tokens"]).replace(",", "")
        assert r["chip_tokens"] > 100, r
        assert "121932631112635269" in text, text[-300:]
        print("OK", r["chip_tokens"])
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-1500:]


def test_meta_is_json_with_no_python_objects(chip_paged):
    meta = F.ChipFile(chip_paged).meta
    assert json.loads(json.dumps(meta)) == meta
