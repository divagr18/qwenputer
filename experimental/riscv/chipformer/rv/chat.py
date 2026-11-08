"""Talk to a merged rv1 checkpoint: Qwen writes, the RV32 machine runs inside the same model.

    python -m chipformer.rv.chat dist/q4-rv1 --firmware tests/data/rv/c-calc.elf \\
        --ask "What is 123456789 * 987654321?"

Decoding is greedy and transparent to chip tokens (as BatchDecoder's fp + tc): a chip token
runs through the chip layers only, so Qwen's layers, caches and RoPE positions see the text
alone; a text token runs through every layer. At every position the next token is the
circuit's when its output code is set (the owner), else Qwen's argmax over its own vocabulary.
The firmware (an ELF, parked at its first exit ecall) is a state snapshot written into the chip
layers' recurrent states after the prompt's prefill (rv/firmware.py, rv/hf.py).

``--check`` replays the whole stream through the IR evaluator and the reference machine: every
circuit-owned token must equal the IR's prediction and the reference's next token.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv import spec as S

CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an arithmetic expression exactly, with numbers of up to 62 digits: "
                   "+ - * / and parentheses on integers and fractions (exact: the result is an "
                   "integer or a reduced fraction p/q), + - * on decimals, and on integers also "
                   "// (floor division), %, ** and gcd(a,b). Returns the exact value.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string",
                       "description": "e.g. 123456789*987654321-42 or 1.25*-3.5 or 84/18"}},
        "required": ["expression"]}}}


class ExecRow:
    """One conversation's circuit on the Triton executor (batch 1) with its own CUDA graph."""

    def __init__(self, chat: "RvChat") -> None:
        from chipformer.rv.tritonexec import Executor2

        self.chat = chat
        self.ex = Executor2(chat.c, 1, chat.dev)
        self.heads = [st for st in self.ex.steps if st["kind"] == "head"]
        self.xbuf = torch.zeros(1, self.ex.D, device=chat.dev)
        self.kbuf = torch.zeros(len(chat.keep_cols), device=chat.dev)
        self.graph = None

    def _body(self):
        self.ex.step(self.xbuf)
        self.kbuf.copy_(self.xbuf[0, self.chat.keep_ir])

    @torch.no_grad()
    def _capture(self):
        """Warm-up runs change the state, so it is saved and put back around them."""
        saved = [(st["S"].clone(), st["hist"].clone()) for st in self.heads]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        for st, (S_, h) in zip(self.heads, saved):
            st["S"].copy_(S_)
            st["hist"].copy_(h)

    @torch.no_grad()
    def start(self, ids: list[int]) -> None:
        """As the HF path does it: the prompt through the circuit from zero states, then every
        head's state set to its init (the snapshot now in the circuit; hf.inject), the lag
        windows kept."""
        if self.graph is None:
            self._capture()
        for st in self.heads:
            st["S"].zero_()
            st["hist"].zero_()
        emb = self.chat._exec_emb(ids)
        for t in range(len(ids)):
            self.xbuf.copy_(emb[t:t + 1])
            self.graph.replay()
        self.ex.load_init()

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        """One token through the circuit; returns the kept values (fp32, the code first)."""
        self.xbuf.copy_(self.chat._exec_emb([ids]))
        self.graph.replay()
        return self.kbuf


class ExecRow4:
    """One conversation's circuit on executor v4 (exec4.py: one persistent kernel per launch).
    ExecRow's interface plus burst(): the chip-token loop on the GPU."""

    def __init__(self, chat: "RvChat", base=None) -> None:
        from chipformer.rv import exec4

        self.chat = chat
        if base is None:
            self.ex = exec4.Exec4(chat.c, chat.dev, text=chat.T, v0=chat.v0,
                                  tables=exec4.cached_tables(chat.c, chat.T, chat.v0))
        else:
            self.ex = base.ex.fork()

    @torch.no_grad()
    def start(self, ids: list[int]) -> None:
        """As ExecRow.start: the prompt from zero states (one launch), then every head's state
        set to its init, the lag history kept."""
        self.ex.zero_state()
        self.ex.reset()
        if ids:
            self.ex.run(self.chat._exec_emb(ids), mode=self.ex.FORCED, max_steps=len(ids))
        if self.chat.state_from_file:
            self.chat.c.load_state(self.ex)
        else:
            self.ex.load_init()

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        """One token through the circuit; returns the kept values (fp32, the code first)."""
        _, _, keep = self.ex.run(self.chat._exec_emb([ids]), mode=self.ex.FORCED, max_steps=1)
        return keep

    @torch.no_grad()
    def burst(self, ids: int, limit: int) -> list[int]:
        """A chip token, then the circuit's predictions fed back while they are chip tokens, at
        most ``limit`` steps: the prediction after every step (model ids; -1: the circuit owns
        no next token; -2: its code names no token)."""
        toks = self.ex.run(self.chat._exec_emb([ids]), mode=self.ex.CHIP,
                           max_steps=max(1, limit), keep=False)
        return toks.tolist()


class RvChat:
    def __init__(self, path, device: str | None = None, fast: bool = True,
                 executor: bool = False, exec_kind: str = "v4", chip=None, base=None,
                 python_circuit: bool = False) -> None:
        """``path``: base Qwen3.5's directory (its text model loads from its own bytes), or a
        merged checkpoint. ``chip``: the chip file (default: path / chip.safetensors). The circuit
        comes from that file; core.build runs only with ``python_circuit``, an audit. With the
        executor alone a merged checkpoint is cut to Qwen's own tensors (``base``: base's
        directory, for its norm gains); the HF chip path (no executor, or "verify") keeps it."""
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        from chipformer.q35.fast import exact_matmuls
        from chipformer.rv import chipfile

        exact_matmuls()
        self.path = Path(path)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # executor "verify": both chip paths, the HF view's hc checked against the executor's
        self.executor = executor if self.dev == "cuda" else False
        self.verify = self.executor == "verify"
        self.state_from_file = False
        cfg = json.loads((self.path / "config.json").read_text())
        is_base = "text_config" in cfg
        self.slim = is_base or bool(self.executor and not self.verify and not python_circuit)
        self.T = BR.Text(self.path / "tokenizer.json")
        chip = Path(chip) if chip else self.path / "chip.safetensors"
        if python_circuit:
            from chipformer.q35.ir import Target
            from chipformer.rv import core

            self.c = core.build(Target(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
                                       cfg["intermediate_size"]), self.T)
            self.meta = json.loads((self.path / "chipformer.json").read_text())
        else:
            if not chip.exists():
                raise FileNotFoundError(f"{chip}: no chip file (python -m chipformer.rv.chipbuild "
                                        "writes one; python_circuit=True is the audit path)")
            self.c = chipfile.load(chip)
            m = self.c.file.meta
            self.meta = {"chip_layers": self.c.n_chip, "chip_token_offset": self.c.v0,
                         "code_dims": m["code_dims"], "tau_dim": m["tau_dim"], "mu_dim": m["mu_dim"],
                         "chip_dims_offset": self.c.off}
        if is_base:
            self.model = chipfile.load_base(self.path, self.dev)
        else:
            self.model = Qwen3_5ForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16)
            if self.slim:
                self.model = chipfile.slim_merged(self.model, self.c, base, self.dev)
            else:
                self.model = self.model.to(self.dev).eval()
        self.lm = self.model.model
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.chip_layers, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.n_chip = 0 if self.slim else self.chip_layers  # the first of Qwen's layers
        self.code_dims = torch.tensor(self.meta["code_dims"], device=self.dev)
        self.eos = {248046, 248044}
        self.fast = fast and self.dev == "cuda" and not self.slim
        # the executor: "v4" (exec4.py, the chip-token loop on the GPU) or "v2" (tritonexec)
        self.exec_kind = exec_kind
        if self.executor:
            self._build_exec()
        if self.fast and (not self.executor or self.verify):
            self._build_fast()
        self.fast = self.fast and (not self.executor or self.verify)

    # --- the executor path -----------------------------------------------------------------
    def _build_exec(self) -> None:
        """The chip layers as the circuit itself (tritonexec.Executor2: the IR's arithmetic in
        fp32, one CUDA graph per token) instead of their HF modules. After the chip layers the
        merged residual holds Qwen's embedding dims, the kept dims (output code, tau, mu) and
        zeros (the exit clears every other chip dim and the ballast), so a text token's hc is
        built from the embedding row and the executor's kept values; chip tokens need only the
        code."""
        self.keep_ir = torch.tensor(self.c.keep, device=self.dev)
        cols = self.meta["code_dims"] + [self.meta["tau_dim"], self.meta["mu_dim"]]
        assert len(cols) == len(self.c.keep)
        self.keep_cols = torch.tensor(cols, device=self.dev)
        self.d0 = self.meta["chip_dims_offset"]
        self.emb_cache: dict[int, torch.Tensor] = {}
        self.erow = ExecRow4(self) if self.exec_kind == "v4" else ExecRow(self)

    def new_erow(self):
        """A further conversation row's executor (RvBatch)."""
        return ExecRow4(self, self.erow) if self.exec_kind == "v4" else ExecRow(self)

    def _exec_emb(self, ids: list[int]) -> torch.Tensor:
        """Executor input rows for model ids (cached per id; shared by every ExecRow)."""
        new = sorted({i for i in ids if i not in self.emb_cache})
        if new:
            rows = self.erow.ex.embed([i - self.v0 if i >= self.v0 else BR.text_token(i)
                                       for i in new])
            for i, r in zip(new, rows):
                self.emb_cache[i] = r
        return torch.stack([self.emb_cache[i] for i in ids])

    def _exec_start(self, ids: list[int]) -> None:
        self.erow.start(ids)

    def _exec_step(self, ids: int) -> torch.Tensor:
        return self.erow.step(ids)

    def _exec_hc(self, ids: int, keep: torch.Tensor) -> torch.Tensor:
        """A text token's residual after the chip layers (slim: Qwen's columns are all of it, and
        the output code stays in keep)."""
        if self.slim:
            self._keep = keep
            return self.lm.embed_tokens.weight[ids]
        hc = torch.zeros(self.lm.embed_tokens.weight.shape[1], dtype=torch.bfloat16,
                         device=self.dev)
        hc[: self.d0] = self.lm.embed_tokens.weight[ids, : self.d0]
        hc[self.keep_cols] = keep.to(torch.bfloat16)
        return hc

    def code_token(self, code) -> int | None:
        """circuit_token from the output code values."""
        code = code.float()
        if not bool((code != 0).any()):
            return None
        bits = (code > 0).long().tolist()
        if hasattr(self.c, "decode_code"):
            nf = self.c.nfam
            dec = self.c.decode_code
        else:  # (the audit path)
            from chipformer.rv import core

            nf = core.NFAM
            dec = lambda f, p: core.decode_code(f, p, self.T)  # noqa: E731
        fid = sum(b << i for i, b in enumerate(bits[:nf]))
        p = sum(b << i for i, b in enumerate(bits[nf:]))
        t = dec(fid, p)
        if t < 0:
            raise RuntimeError(f"the circuit predicts no token (family {fid}, payload {p})")
        return t - S.TEXT_TOKEN if S.is_text(t) else self.v0 + t

    # --- the fast chip path ----------------------------------------------------------------
    def _build_fast(self) -> None:
        """A second view of the chip layers: a text model of the n chip layers (the same module
        objects) plus one zero full-attention layer, its linear maps compacted to their nonzero
        blocks (fast.compact; exact), with its own static cache, run as one CUDA graph per token
        (torch.compile reduce-overhead). Chip tokens run only here; text tokens run here and then
        through Qwen's layers with the main cache."""
        import copy

        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        from chipformer.q35 import fast as F

        cfg = copy.deepcopy(self.model.config)
        cfg.num_hidden_layers = self.n_chip + 1
        cfg.layer_types = ["linear_attention"] * self.n_chip + ["full_attention"]
        self.ccfg = cfg
        dummy = q.Qwen3_5DecoderLayer(cfg, self.n_chip).to(self.dev, torch.bfloat16)
        for p in dummy.parameters():
            p.data.zero_()
        with torch.device("meta"):
            cm = q.Qwen3_5TextModel(cfg)
        cm.embed_tokens = self.lm.embed_tokens
        cm.layers = torch.nn.ModuleList([*self.lm.layers[: self.n_chip], dummy])
        cm.norm = torch.nn.Identity()
        cm.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(self.dev)
        self.cm = cm.eval()

        class _Wrap(torch.nn.Module):  # fast.compact's view of it
            def __init__(self, inner):
                super().__init__()
                self.model = inner

        self.compact = F.compact(_Wrap(cm), qwen=False, n_chip=self.n_chip)
        self.graph = None  # captured after the first prefill (the cache buffers are static)

    def _chip_cache(self):
        """The chip view's cache, allocated once: its state buffers keep their addresses (HF
        updates them in place), so one CUDA graph serves every conversation."""
        from transformers import DynamicCache

        if getattr(self, "ccache", None) is None:
            self.ccache = DynamicCache(config=self.ccfg)
        else:
            for L in range(self.n_chip):
                lay = self.ccache.layers[L]
                lay.conv_states[0].zero_()
                lay.recurrent_states[0].zero_()
                lay.has_previous_state[0] = False
        return self.ccache

    @torch.no_grad()
    def _chip_body(self):
        h = self.cm.embed_tokens(self.g_ids)
        for L in range(self.n_chip):
            h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=self.ccache)
        self.g_out.copy_(h)

    @torch.no_grad()
    def _capture(self):
        """One CUDA graph of the chip layers' single-token decode step. Warm-up runs change the
        cache, so the state is saved and put back around them."""
        self.g_ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.g_out = torch.zeros((1, 1, self.model.config.hidden_size), dtype=torch.bfloat16,
                                 device=self.dev)
        lays = [self.ccache.layers[L] for L in range(self.n_chip)]
        saved = [(lay.conv_states[0].clone(), lay.recurrent_states[0].clone()) for lay in lays]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._chip_body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._chip_body()
        for lay, (cs, rs) in zip(lays, saved):
            lay.conv_states[0].copy_(cs)
            lay.recurrent_states[0].copy_(rs)

    @torch.no_grad()
    def _chip_step(self, ids: int, g: int, cache, compiled: bool = True):
        if not compiled or self.graph is None:
            x = torch.tensor([[ids]], device=self.dev)
            out = self.cm(input_ids=x, past_key_values=cache, use_cache=True,
                          cache_position=torch.tensor([g], device=self.dev))
            return out.last_hidden_state[0, -1].clone()
        self.g_ids.fill_(ids)
        self.graph.replay()
        return self.g_out[0, -1].clone()

    @torch.no_grad()
    def _qwen_step(self, hc, cache, pos: int):
        """Qwen's layers on a text token whose residual after the chip layers is hc."""
        h = hc.view(1, 1, -1)
        p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
        pe = self.lm.rotary_emb(h, p4[1:])
        for i in range(self.n_chip, len(self.lm.layers)):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=p4[0], past_key_values=cache)
        return self.model.lm_head(self.lm.norm(h))[0, -1]

    @torch.no_grad()
    def _qwen_prefill(self, hcs, cache, pos: int):
        """_qwen_step on a run of text tokens as one chunk (speedups.md §4): Qwen predicts nothing
        inside a run of circuit-emitted text and Qwen's layers never feed the chip layers, so its
        positions need not go one at a time. hcs: (n, hidden). Returns the last position's logits.
        Text numerics move at bf16 rounding against _qwen_step, as prefill does against decode."""
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        h = hcs.unsqueeze(0)
        ids = torch.arange(pos, pos + h.shape[1], device=self.dev).view(1, -1)
        p4 = ids.unsqueeze(0).expand(4, 1, -1)
        kw = {"config": self.lm.config, "inputs_embeds": h, "attention_mask": None,
              "past_key_values": cache, "position_ids": p4[0]}
        masks = {"full_attention": create_causal_mask(**kw),
                 "linear_attention": create_recurrent_attention_mask(**kw)}
        pe = self.lm.rotary_emb(h, p4[1:])
        for i in range(self.n_chip, len(self.lm.layers)):
            h = self.lm.layers[i](h, position_embeddings=pe, position_ids=p4[0], past_key_values=cache,
                                  attention_mask=masks[self.lm.config.layer_types[i]])
        return self.model.lm_head(self.lm.norm(h))[0, -1]

    def _text_hc(self, ids: list[int]):
        """Residuals after the chip layers for text the chip does not run: the embedding rows,
        and with the executor the kept dims as they stand at the park (fx_keep)."""
        x = self.lm.embed_tokens(torch.tensor(ids, device=self.dev))
        if not self.executor or self.slim:
            return x
        h = torch.zeros_like(x)
        h[:, : self.d0] = x[:, : self.d0]
        keep = getattr(self, "fx_keep", None)
        if keep is not None:
            h[:, self.keep_cols] = keep.to(x.dtype)
        return h

    # --- one position ----------------------------------------------------------------------
    @torch.no_grad()
    def _layers(self, ids, cache, pos: int | None):
        """Run one token through the chip layers (pos None) or the whole stack (text at RoPE
        position pos); returns (residual after the chip layers, logits or None)."""
        h = self.lm.embed_tokens(torch.tensor([[ids]], device=self.dev))
        pe = tp = None
        if pos is not None:
            p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
            tp = p4[0]
            pe = self.lm.rotary_emb(h, p4[1:])
        hc = None
        n = self.n_chip if pos is None else len(self.lm.layers)
        for i in range(n):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=tp, past_key_values=cache)
            if i == self.n_chip - 1:
                hc = h
        logits = None if pos is None else self.model.lm_head(self.lm.norm(h))[0, -1]
        return hc[0, -1], logits

    def circuit_token(self, hc) -> int | None:
        """The model id the circuit predicts at a position, None where Qwen predicts."""
        if self.slim:
            return self.code_token(self._keep[: len(self.code_dims)])
        return self.code_token(hc[self.code_dims])

    def _snapshot(self, machine) -> None:
        """The state a conversation starts from: a machine's (the host's loader), or with None
        the firmware in the chip file."""
        self.state_from_file = machine is None
        if machine is not None:
            from chipformer.rv import core

            core.snapshot(self.c, machine)
        elif not self.c.has_state():
            raise ValueError("this chip file holds no firmware: pass a machine")
        elif not self.slim:
            self.c.apply_state()

    # --- a conversation --------------------------------------------------------------------
    def prompt_ids(self, messages, tools=(CALC_TOOL,)) -> list[int]:
        out = self.tok.apply_chat_template(messages, tools=list(tools) or None,
                                           add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        return [int(x) for x in out]

    @torch.no_grad()
    def generate(self, ids: list[int], machine: "ref.Machine", max_new: int = 4000,
                 max_chip: int = 60000, check_head: bool = False, verbose: bool = False,
                 lock: "BR.Session | None" = None, on_token=None) -> dict:
        """Greedy decoding after the prompt ``ids`` with the parked ``machine`` loaded.
        ``lock`` (a Session on a copy of the machine): stop at the first token that differs from
        the reference's. ``on_token(id)`` is called with every generated model id as it comes."""
        if self.executor and not self.verify and self.exec_kind == "v4":
            return self._generate_v4(ids, machine, max_new, max_chip, check_head, verbose, lock,
                                     on_token)
        from chipformer.rv import hf

        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, use_cache=True)
        cache = out.past_key_values  # (fast: Qwen's layers only)
        self._snapshot(machine)
        if self.executor:
            self._exec_start(ids)
        if self.fast:
            ccache = self._chip_cache()
            h = self.cm.embed_tokens(x)  # the prompt through the chip layers only
            for L in range(self.n_chip):
                h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                      position_ids=None, past_key_values=ccache)
            hf.inject(self.c, ccache)
            if self.graph is None:
                self._capture()
            g = len(ids)  # the chip view's cache position (every token)
        elif not self.executor:
            hf.inject(self.c, cache)
        n_verify_bad = 0
        logits = out.logits[0, -1]
        pos = len(ids)  # the next text token's RoPE position
        gen, n_text, n_chip_tok = [], 0, 0
        # the prompt's last position: Qwen predicts (nothing captured yet)
        nxt = int(torch.argmax(logits[: self.v0]))
        t0 = time.time()
        mismatch = 0
        t_chip = t_text = 0.0
        expect, lock_error = [], None
        while n_text < max_new and n_chip_tok < max_chip:
            gen.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if lock is not None:  # check against the reference as we go
                st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else self.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else self.tok.decode([e - S.TEXT_TOKEN]),
                                      "recent": [S.name(t - self.v0) if t >= self.v0 else
                                                 self.tok.decode([t]) for t in gen[-12:]]}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    try:
                        lock.qwen([nxt])
                    except RuntimeError as e:  # the reference run did not halt
                        lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                        break
                    expect = list(lock.toks[k + 1:])
            if nxt in self.eos:
                break
            if verbose and len(gen) % 500 == 0:
                tail = [S.name(t - self.v0) if t >= self.v0 else repr(self.tok.decode([t]))
                        for t in gen[-8:]]
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) "
                      f"{time.time() - t0:.0f}s: {' '.join(tail)}", flush=True)
            ts = time.time()
            chip_in = nxt >= self.v0
            hc = keep = None
            if self.executor:
                keep = self._exec_step(nxt)
                if not chip_in:
                    hc = self._exec_hc(nxt, keep)
            if self.fast and (self.verify or not self.executor):
                hf_hc = self._chip_step(nxt, g, ccache)
                g += 1
                if self.verify:
                    same = (bool((hf_hc[self.keep_cols] == keep.to(torch.bfloat16)).all())
                            if chip_in else bool((hf_hc == hc).all()))
                    if not same:
                        n_verify_bad += 1
                hc = hf_hc
            elif not self.executor:
                hc, logits = self._layers(nxt, cache, None if chip_in else pos)
            if chip_in:  # a chip token
                n_chip_tok += 1
                nxt = (self.code_token(keep[: len(self.code_dims)]) if hc is None
                       else self.circuit_token(hc))
                t_chip += time.time() - ts
                if nxt is None:
                    raise RuntimeError("the circuit gave up a chip position")
            else:
                n_text += 1
                if self.fast or self.executor:
                    logits = self._qwen_step(hc, cache, pos)
                pos += 1
                ct = self.circuit_token(hc)
                q = int(torch.argmax(logits))
                if ct is not None and check_head and q != ct:
                    mismatch += 1
                nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
                t_text += time.time() - ts
        dt = time.time() - t0
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": dt, "head_mismatch": mismatch, "lock_error": lock_error,
                "verify_mismatch": n_verify_bad if self.verify else None,
                "chip_seconds": round(t_chip, 2), "text_seconds": round(t_text, 2)}

    @torch.no_grad()
    def _generate_v4(self, ids, machine, max_new, max_chip, check_head, verbose, lock,
                     on_token) -> dict:
        """generate on executor v4: the same tokens and result; a chip token starts a burst
        (ExecRow4.burst: the circuit runs on the GPU until it predicts a token that is not a
        chip token), whose predictions are then taken one by one as generate takes them. Slim
        has no chip output head, so no head check."""
        check_head = check_head and not self.slim
        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, use_cache=True)
        cache = out.past_key_values
        self._snapshot(machine)
        self._exec_start(ids)
        logits = out.logits[0, -1]
        pos = len(ids)
        gen, n_text, n_chip_tok = [], 0, 0
        nxt = int(torch.argmax(logits[: self.v0]))
        t0 = time.time()
        mismatch = 0
        t_chip = t_text = 0.0
        expect, lock_error = [], None
        pending: list[int] = []  # the current burst's predictions not taken yet (reversed)
        n_bursts = 0
        while n_text < max_new and n_chip_tok < max_chip:
            gen.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if lock is not None:  # check against the reference as we go
                st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else self.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else self.tok.decode([e - S.TEXT_TOKEN]),
                                      "recent": [S.name(t - self.v0) if t >= self.v0 else
                                                 self.tok.decode([t]) for t in gen[-12:]]}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    try:
                        lock.qwen([nxt])
                    except RuntimeError as e:  # the reference run did not halt
                        lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                        break
                    expect = list(lock.toks[k + 1:])
            if nxt in self.eos:
                break
            if verbose and len(gen) % 500 == 0:
                tail = [S.name(t - self.v0) if t >= self.v0 else repr(self.tok.decode([t]))
                        for t in gen[-8:]]
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) "
                      f"{time.time() - t0:.0f}s: {' '.join(tail)}", flush=True)
            ts = time.time()
            if nxt >= self.v0 and not pending and lock is None and on_token is None \
                    and not verbose:
                # a burst taken whole: the loop would append its chip predictions one by one
                # and only the last can be a text token (or -1 / -2) or reach max_chip
                toks = self.erow.burst(nxt, max_chip - n_chip_tok)
                n_bursts += 1
                gen.extend(toks[:-1])
                n_chip_tok += len(toks)
                nxt = toks[-1]
                t_chip += time.time() - ts
                if nxt == -1:
                    raise RuntimeError("the circuit gave up a chip position")
                if nxt == -2:
                    raise RuntimeError("the circuit predicts no token")
                continue
            if nxt >= self.v0:  # a chip token
                if not pending:
                    pending = self.erow.burst(nxt, max_chip - n_chip_tok)
                    pending.reverse()
                    n_bursts += 1
                n_chip_tok += 1
                nxt = pending.pop()
                t_chip += time.time() - ts
                if nxt == -1:
                    raise RuntimeError("the circuit gave up a chip position")
                if nxt == -2:
                    raise RuntimeError("the circuit predicts no token")
            else:
                assert not pending
                n_text += 1
                keep = self._exec_step(nxt)
                hc = self._exec_hc(nxt, keep)
                logits = self._qwen_step(hc, cache, pos)
                pos += 1
                ct = self.circuit_token(hc)
                q = int(torch.argmax(logits))
                if ct is not None and check_head and q != ct:
                    mismatch += 1
                nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
                t_text += time.time() - ts
        dt = time.time() - t0
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": dt, "head_mismatch": mismatch, "lock_error": lock_error,
                "verify_mismatch": None, "chip_seconds": round(t_chip, 2),
                "text_seconds": round(t_text, 2), "bursts": n_bursts}

    def to_stream(self, ids: list[int]) -> list[int]:
        """Model ids -> the circuit's token stream (chip tokens, TEXT_TOKEN | qwen id)."""
        return [i - self.v0 if i >= self.v0 else BR.text_token(i) for i in ids]

    def text_of(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.v0], skip_special_tokens=False)


class FxQwen:
    """RvChat's Qwen as a fx/fxio DeltaLink backend (docs/fx/io.md): text fx has rendered goes in
    as one chunked prefill, not a token at a time, and Qwen's state keeps what it already read.
    The prefill does not touch the chip layers: with fx parked in fx_http_stream_open there is
    no run for that text to belong to. Snapshots are host-side copies of Qwen's state.
    ``forced``: generations to replay instead of predicting (a test's teacher forcing)."""

    def __init__(self, chat: RvChat, forced=None, chunk: int = 2048) -> None:
        self.chat, self.forced, self.chunk = chat, forced, chunk
        self.reset()

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache(config=self.chat.model.config)
        self.pos, self.logits = 0, None
        self.snaps: dict[int, tuple] = {}

    def feed(self, ids: list[int]) -> None:
        for a in range(0, len(ids), self.chunk):
            part = ids[a:a + self.chunk]
            self.logits = self.chat._qwen_prefill(self.chat._text_hc(part), self.cache, self.pos)
            self.pos += len(part)

    def snapshot(self, n: int) -> None:
        assert n == self.pos
        self.snaps[n] = (deepcopy(self.cache), self.logits.clone())

    def restore(self, n: int) -> None:
        cache, logits = self.snaps[n]
        self.cache, self.logits, self.pos = deepcopy(cache), logits.clone(), n
        for k in [k for k in self.snaps if k > n]:
            del self.snaps[k]

    def _hc(self, t: int):
        c = self.chat
        if c.executor:  # the chip reads what Qwen writes, as in generate
            c.fx_keep = c._exec_step(t)
            return c._exec_hc(t, c.fx_keep)
        return c._text_hc([t])[0]

    @torch.no_grad()
    def generate(self, max_new: int, stop_ids) -> list[int]:
        c = self.chat
        forced = self.forced.pop(0) if self.forced else None
        out: list[int] = []
        while len(out) < max_new:
            t = forced[len(out)] if forced is not None else int(torch.argmax(self.logits[: c.v0]))
            out.append(t)
            if t in stop_ids:
                break
            self.logits = c._qwen_step(self._hc(t), self.cache, self.pos)
            self.pos += 1
        return out


def fx_session(chat: RvChat, chip, template, max_new: int = 8192, forced=None) -> list[dict]:
    """The packet path: fx's chip runs until it wants the model, Qwen answers, repeat.
    ``chip``: next_request() -> a REQUEST packet or None; deliver(a REPLY packet).
    No chip that runs fx on the circuit exists yet (the memory map is separate work); the tests
    use one that replays recorded packets."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fx" / "fxio"))
    import delta as D
    import protocol as P

    link = D.DeltaLink(template, FxQwen(chat, forced), max_new=max_new)
    while (pkt := chip.next_request()) is not None:
        chunks, _ = link.call(P.parse_request(pkt)[2])
        chip.deliver(P.reply(pkt.seq, chunks))
    return link.records


def replay(chat: RvChat, prompt: list[int], gen: list[int], firmware) -> dict:
    """Check a generated stream: the reference machine (fed Qwen's own tokens) must produce the
    same chip tokens and echoes, and the IR circuit must predict every circuit-owned token. An
    audit, so it may build the Python circuit: the tree's must be the chip file's."""
    from chipformer.q35 import ir
    from chipformer.q35.ir import Target
    from chipformer.rv import core

    cc = chat.c if hasattr(chat.c, "owner") else core.build(
        Target(chat.c.nk, chat.c.nv, chat.c.inter), chat.T)
    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    ses = BR.Session(chat.T, m)
    stream = chat.to_stream(prompt + gen)
    # the prompt is prefilled history (its system prompt shows the tool-call format, so it holds
    # <tool_call> .. </tool_call> tokens: the circuit owns those </tool_call> positions, but no
    # run follows); then the reference is fed Qwen's tokens and appends the runs itself
    ses.history(prompt)
    skip = {x for x in range(len(prompt)) if prompt[x] == chat.T.TCC}
    i = 0
    pos = len(prompt)
    while pos < len(stream):
        t = stream[pos]
        if not S.is_text(t):
            raise AssertionError(f"chip token {S.name(t)} at {pos} where the reference has none")
        k = len(ses.toks)
        ses.qwen([t - S.TEXT_TOKEN])
        pos += 1
        added = ses.toks[k + 1:]
        if added:  # a run: the stream must continue with exactly the reference's tokens
            got = stream[pos:pos + len(added)]
            if got != added:
                j = next((x for x in range(min(len(got), len(added))) if got[x] != added[x]),
                         min(len(got), len(added)))
                return {"ok": False, "where": pos + j, "got": got[j:j + 5], "want": added[j:j + 5]}
            pos += len(added)
        i += 1
    m2 = ref.Machine.from_program(elf.program(firmware))
    BR.park(m2)
    core.snapshot(cc, m2)
    carry, mats = {}, []
    for a in range(0, len(stream), 2048):
        mats.append(ir.evaluate(cc, stream[a:a + 2048], carry=carry))
    mat = np.concatenate(mats)
    pred = core.predictions(cc, mat)
    own = np.rint(ir.ev(cc.owner, mat)).astype(int)
    bad_own = int(sum(own[x] != int(ses.own[x]) for x in range(len(stream))))
    bad = int(sum(1 for x in range(len(stream) - 1)
                  if ses.own[x] and x not in skip and pred[x] != stream[x + 1]))
    return {"ok": bad == 0 and bad_own == 0, "positions": len(stream),
            "owned": int(sum(ses.own)) - len(skip), "wrong_predictions": bad,
            "owner_mismatch": bad_own, "runs": len(ses.runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--ask", default="Use calc: what is 123456789 * 987654321?")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--slow", action="store_true", help="every layer through the HF modules")
    ap.add_argument("--chip", help="the chip file (default: MODEL/chip.safetensors)")
    args = ap.parse_args()
    chat = RvChat(args.model, fast=not args.slow, chip=args.chip)
    m = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m)
    ids = chat.prompt_ids([{"role": "user", "content": args.ask}])
    m2 = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m2)
    lock = BR.Session(chat.T, m2)
    lock.history(ids)
    r = chat.generate(ids, m, args.max_new, check_head=True, verbose=True, lock=lock)
    print(chat.text_of(r["tokens"]))
    print(f"[{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s, "
          f"head mismatches {r['head_mismatch']}, lock error {r['lock_error']}]")
    if args.check:
        print(replay(chat, ids, r["tokens"], args.firmware))


if __name__ == "__main__":
    main()
