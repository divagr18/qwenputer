"""The fx packet path (docs/fx/io.md): protocol, template, delta, parser, SSE, HF backend.

Needs Qwen3.5's tokenizer.json and chat_template.jinja in the HF cache, no weights. The model
tests use a tiny random Qwen3_5 on the CPU.
"""
import glob
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fx" / "fxio"))
sys.path.insert(0, str(ROOT))

import delta as D  # noqa: E402
import protocol as P  # noqa: E402
import scripted as SC  # noqa: E402


def model_dir():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    hits = sorted(glob.glob(str(hub / "models--Qwen--Qwen3.5-4B" / "snapshots" / "*" / "tokenizer.json")))
    if not hits:
        pytest.skip("no Qwen3.5-4B tokenizer in the HF cache")
    return Path(hits[-1]).parent


@pytest.fixture(scope="module")
def tpl():
    return D.QwenTemplate(model_dir())


TOOLS = [{"type": "function", "function": {"name": "edit_file", "description": "Edit.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                     "new_string": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}]


def gateway_body(steps, system="Be concise.", user="Fix it."):
    """A gateway request after `steps` = (reasoning, text, name, args, result)."""
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": user}]}]
    for i, (r, text, name, args, result) in enumerate(steps):
        content = []
        if r:
            content.append({"type": "reasoning", "text": r})
        if text:
            content.append({"type": "text", "text": text})
        if name:
            content.append({"type": "tool-call", "toolCallId": f"c{i}", "toolName": name, "input": args})
        prompt.append({"role": "assistant", "content": content})
        if name:
            prompt.append({"role": "tool", "content": [{"type": "tool-result", "toolCallId": f"c{i}", "toolName": name,
                                                        "output": {"type": "text", "value": result}}]})
    tools = [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
              "inputSchema": t["function"]["parameters"]} for t in TOOLS]
    return json.dumps({"prompt": prompt, "tools": tools, "toolChoice": {"type": "auto"}}).encode()


class Recorder:
    """A backend that returns canned text and tracks exactly what it was fed."""

    def __init__(self, t, raws):
        self.t, self.raws, self.ids, self.snaps, self.events = t, list(raws), [], {}, []

    def reset(self):
        self.ids = []
        self.snaps = {}
        self.events.append("reset")

    def feed(self, ids):
        self.ids += ids

    def snapshot(self, n):
        assert n == len(self.ids)
        self.snaps[n] = list(self.ids)

    def restore(self, n):
        self.ids = list(self.snaps[n])
        self.events.append(("restore", n))

    def generate(self, max_new, stop):
        g = self.raws.pop(0)
        g = self.t.encode(g) if isinstance(g, str) else list(g)
        self.ids += g
        return g + [self.t.tok.token_to_id("<|im_end|>")]


def test_import_table_covers_the_images():
    names = """fx_attachment_put fx_attachment_size fx_attachment_take fx_host_tool_call fx_host_tool_result_read
    fx_host_tool_result_release fx_http_request fx_http_stream_close fx_http_stream_next fx_http_stream_open
    fx_http_stream_retry_after fx_http_stream_status fx_journal_flush fx_oauth_session_commit fx_oauth_session_load
    fx_oauth_session_remove fx_session_commit fx_session_list fx_session_load fx_session_remove fx_steering_close
    fx_steering_state fx_steering_take args_get args_sizes_get clock_res_get clock_time_get environ_get
    environ_sizes_get fd_close fd_fdstat_get fd_filestat_get fd_filestat_set_size fd_filestat_set_times fd_pread
    fd_prestat_dir_name fd_prestat_get fd_pwrite fd_read fd_readdir fd_seek fd_sync fd_write path_create_directory
    path_filestat_get path_filestat_set_times path_link path_open path_readlink path_remove_directory path_rename
    path_symlink path_unlink_file poll_oneoff proc_exit random_get""".split()
    assert len(names) == 56 and set(names) == set(P.IMPORTS)
    rows = P.classify([{"id": i, "module": "m", "name": n} for i, n in enumerate(names)])
    assert {r["group"] for r in rows} == {"a", "b", "c"}
    with pytest.raises(KeyError):
        P.classify([{"id": 0, "module": "m", "name": "new_import"}])


def test_packets_round_trip():
    q = P.request(7, b"https://x", b"[]", b'{"prompt":[]}')
    a = P.reply(7, [b"data: 1\n\n", b"data: 2\n\n"])
    pk = P.decode_stream(q.encode() + a.encode())
    assert [p.kind for p in pk] == [P.REQUEST, P.REPLY] and pk[0].seq == 7
    assert P.parse_request(pk[0]) == (b"https://x", b"[]", b'{"prompt":[]}')
    assert P.parse_reply(pk[1]) == (200, [b"data: 1\n\n", b"data: 2\n\n"])
    with pytest.raises(ValueError):
        P.decode_stream(q.encode()[:-1])
    with pytest.raises(ValueError):
        P.decode_stream(b"XXXX" + q.encode()[4:])


def test_template_equals_transformers(tpl):
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoTokenizer.from_pretrained(model_dir())
    msgs = D.request_messages(json.loads(gateway_body([
        ("Think.\n", "Text.", "read_file", {"path": "a.py"}, "1\tprint('é')"),
        ("", "", "edit_file", {"path": "a.py", "old_string": "x\n", "new_string": "y\n"}, "edited")])))[1]
    for thinking in (True, False):
        want = hf.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
        got = tpl.render(msgs, TOOLS, enable_thinking=thinking)
        assert got == want
        ids = hf.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
        ids = ids["input_ids"] if not isinstance(ids, list) else ids
        assert tpl.encode(got) == [int(x) for x in ids]


@pytest.mark.parametrize("text,reasoning,content,calls", [
    ("plain answer", "plain answer", "", []),
    ("why\n</think>\n\nanswer", "why\n", "\n\nanswer", []),
    ("why\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
     "why\n", "", [("read_file", {"path": "a.py"})]),
    ("r</think>Doing it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
     "r", "Doing it.\n\n", [("read_file", {"path": "a.py"})]),
    # a tool call straight from reasoning: an implicit </think>
    ("r<tool_call>\n<function=read_file>\n<parameter=path>a.py</parameter>\n</function>\n</tool_call>",
     "r", "", [("read_file", {"path": "a.py"})]),
    # one newline each side is framing, the next one is the value's
    ("r</think><tool_call><function=edit_file><parameter=new_string>\nx\n\n</parameter></function></tool_call>",
     "r", "", [("edit_file", {"new_string": "x\n"})]),
    # schema types
    ("r</think><tool_call><function=edit_file><parameter=count>\n3\n</parameter></function></tool_call>",
     "r", "", [("edit_file", {"count": 3})]),
    # vLLM's regression cases: no closing tag; a value running into the next parameter
    ("r</think><tool_call>\n<function=edit_file>\n<parameter=path>\na\n<parameter=old_string>\nb\n</parameter>\n</function>\n</tool_call>",
     "r", "", [("edit_file", {"path": "a", "old_string": "b"})]),
    # an unknown tool is dropped, a call with no </tool_call> still counts
    ("r</think><tool_call>\n<function=nope>\n</function>\n</tool_call>", "r", "", []),
    ("r</think><tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>",
     "r", "", [("read_file", {"path": "a"})]),
])
def test_parser(text, reasoning, content, calls):
    c = D.parse_completion(text, TOOLS)
    assert c.reasoning == reasoning
    assert c.content == content
    assert [(t["name"], json.loads(t["arguments"])) for t in c.tool_calls] == calls


def test_parser_keeps_parameter_order():
    t = D.parse_completion("r</think><tool_call><function=edit_file><parameter=path>p</parameter>"
                           "<parameter=old_string>o</parameter><parameter=new_string>n</parameter>"
                           "</function></tool_call>", TOOLS)
    assert list(json.loads(t.tool_calls[0]["arguments"])) == ["path", "old_string", "new_string"]


def test_parser_thinking_off():
    c = D.parse_completion("just text", TOOLS, thinking=False)
    assert (c.reasoning, c.content) == ("", "just text")


@pytest.mark.parametrize("proto", ["gateway", "chat"])
def test_sse_is_one_read(proto):
    c = D.Completion("why", "text", [{"name": "read_file", "arguments": '{"path":"é"}'}])
    chunks = D.sse_reply(proto, c, 3, 100, 20)
    assert len(chunks) == 1
    events = [e[len(b"data: "):] for e in chunks[0].split(b"\n\n") if e]
    if proto == "chat":
        assert events[-1] == b"[DONE]" and len(events) == 2
        d = json.loads(events[0])["choices"][0]
        assert d["finish_reason"] == "tool_calls" and d["delta"]["tool_calls"][0]["function"]["arguments"] == '{"path":"é"}'
    else:
        kinds = [json.loads(e)["type"] for e in events]
        assert kinds == ["reasoning-start", "reasoning-delta", "reasoning-end", "text-start", "text-delta", "text-end",
                         "tool-call", "finish"]


def test_delta_is_exact_for_canonical_generations(tpl):
    r1 = SC.raw_step("I will read it.", "Reading.", ("read_file", {"path": "a.py"}))
    r2 = SC.raw_step("Now edit.", "", ("edit_file", {"path": "a.py", "old_string": "x\n", "new_string": "y\n"}))
    r3 = SC.raw_step("Done now.", "All fixed.", None)
    b = Recorder(tpl, [r1, r2, r3])
    link = D.DeltaLink(tpl, b)
    link.call(gateway_body([]))
    s1 = ("I will read it.", "Reading.", "read_file", {"path": "a.py"}, "1\tx")
    _, rec = link.call(gateway_body([s1]))
    assert rec["kind"] == "exact" and rec["rolled_back"] == 0
    s2 = ("Now edit.", "", "edit_file", {"path": "a.py", "old_string": "x\n", "new_string": "y\n"}, "edited a.py")
    _, rec = link.call(gateway_body([s1, s2]))
    assert rec["kind"] == "exact"
    assert rec["delta_tokens"] < rec["full_tokens"] / 3
    # what the backend ran equals a fresh render of the last request, plus its generation
    proto, msgs, tools = D.request_messages(json.loads(gateway_body([s1, s2])))
    want = tpl.encode(tpl.render(msgs, tools))
    assert b.ids[:len(want)] == want


def test_delta_rolls_back_on_a_noncanonical_generation(tpl):
    """Same text, different ids: identical at the text level, and the id check still catches it."""
    raw = SC.raw_step("I will read it.", "", ("read_file", {"path": "a.py"}))
    ids = tpl.encode(raw)
    k = 2  # " read", a multi-character token
    bad = ids[:k] + [i for c in tpl.decode(ids[k:k + 1]) for i in tpl.encode(c)] + ids[k + 1:]
    assert tpl.decode(bad) == tpl.decode(ids) and bad != ids
    b = Recorder(tpl, [bad, SC.raw_step("x", "done", None)])
    link = D.DeltaLink(tpl, b)
    link.call(gateway_body([]))
    n = len(b.ids) - len(bad)
    s1 = ("I will read it.", "", "read_file", {"path": "a.py"}, "1\tx")
    _, rec = link.call(gateway_body([s1]))
    assert rec["kind"] == "rollback" and rec["kept"] == n and rec["text_extends"]
    assert ("restore", n) in b.events
    proto, msgs, tools = D.request_messages(json.loads(gateway_body([s1])))
    want = tpl.encode(tpl.render(msgs, tools))
    assert b.ids[:len(want)] == want


def test_delta_rewritten_history_goes_back_to_the_last_shared_snapshot(tpl):
    """fx's turn 2 inserts a user message before the last answer, and the template then drops
    the think blocks of everything before the newest question."""
    b = Recorder(tpl, [SC.raw_step("r", "first answer", None), SC.raw_step("s", "second", None)])
    link = D.DeltaLink(tpl, b)
    link.call(gateway_body([]))
    body = json.loads(gateway_body([("r", "first answer", None, None, None)]))
    body["prompt"].insert(2, {"role": "user", "content": [{"type": "text", "text": "Session file evidence."}]})
    body["prompt"].append({"role": "user", "content": [{"type": "text", "text": "Next."}]})
    _, rec = link.call(json.dumps(body).encode())
    assert rec["kind"] == "rollback" and rec["rolled_back"] > 0


def test_delta_rebuilds_when_the_system_prompt_changes(tpl):
    b = Recorder(tpl, [SC.raw_step("r", "a", None), SC.raw_step("s", "b", None)])
    link = D.DeltaLink(tpl, b)
    link.call(gateway_body([]))
    _, rec = link.call(gateway_body([], system="Different from the start, this one."))
    assert rec["kind"] == "rebuild" and rec["kept"] == 0 and "reset" in b.events


def tiny_qwen():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5ForCausalLM"):
        pytest.skip("transformers without Qwen3.5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=4,
                            linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    torch.manual_seed(0)
    return tr.Qwen3_5ForCausalLM(cfg).eval()


def test_hf_backend_suffix_feed_equals_a_full_prefill():
    torch = pytest.importorskip("torch")
    import qwen_hf as QH

    m = tiny_qwen()
    ids = [5, 9, 44, 201, 7, 7, 13, 90, 150, 3, 28, 6]
    full = QH.HfBackend(m)
    full.feed(ids)
    inc = QH.HfBackend(m)
    inc.feed(ids[:5])
    inc.snapshot(5)
    inc.feed(ids[5:9])
    inc.feed(ids[9:])
    assert torch.allclose(full.logits, inc.logits, atol=2e-3, rtol=2e-3)
    assert int(full.logits.argmax()) == int(inc.logits.argmax())
    # a bad continuation, then back to the snapshot and on with the right tokens
    other = QH.HfBackend(m)
    other.feed(ids[:5])
    other.snapshot(5)
    other.feed([100, 101, 102, 103])
    other.restore(5)
    assert other.pos == 5
    other.feed(ids[5:])
    assert torch.allclose(full.logits, other.logits, atol=2e-3, rtol=2e-3)


def test_hf_backend_generates_and_stops():
    import qwen_hf as QH

    b = QH.HfBackend(tiny_qwen(), temperature=0.7, top_k=10, top_p=0.9, seed=3)
    b.feed([1, 2, 3])
    g = b.generate(6, {299})
    assert 1 <= len(g) <= 6 and b.pos == 3 + len([t for t in g if t != 299])


# ---- RvChat's packet path, with a chip that replays recorded fx packets ------------------------

DATA = ROOT / "tests" / "data" / "fx"


class ReplayChip:
    """fx as seen from the model side: the REQUEST packets of a recorded session, in order."""

    def __init__(self, data: bytes) -> None:
        self.reqs = [p for p in P.decode_stream(data) if p.kind == P.REQUEST]
        self.got: list[P.Packet] = []

    def next_request(self):
        return self.reqs.pop(0) if self.reqs else None

    def deliver(self, pkt):
        assert pkt.kind == P.REPLY
        self.got.append(pkt)


def tiny_chat(n_chip=1):
    """An RvChat with a tiny real Qwen3_5 instead of a merged checkpoint; layer 0 stands in for
    the chip layers, which _qwen_step and _qwen_prefill skip."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5ForCausalLM"):
        pytest.skip("transformers without Qwen3.5")
    from chipformer.rv.chat import RvChat
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(vocab_size=248100, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=4,
                            linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    torch.manual_seed(1)
    c = RvChat.__new__(RvChat)
    c.model = tr.Qwen3_5ForCausalLM(cfg).eval()
    c.lm, c.n_chip, c.dev, c.executor, c.v0 = c.model.model, n_chip, "cpu", False, 248100
    return c


def test_prefill_equals_token_by_token():
    torch = pytest.importorskip("torch")
    c = tiny_chat()
    from transformers import DynamicCache

    ids = [5, 900, 44, 201, 7, 7, 13, 90, 150, 3, 28, 6, 1000, 2000, 31]
    a = DynamicCache(config=c.model.config)
    for i, t in enumerate(ids):
        one = c._qwen_step(c._text_hc([t])[0], a, i)
    b = DynamicCache(config=c.model.config)
    many = c._qwen_prefill(c._text_hc(ids), b, 0)
    assert torch.allclose(one, many, atol=2e-3, rtol=2e-3)
    # and it goes on from there: a chunk after a chunk against two decode steps
    c._qwen_step(c._text_hc([77])[0], a, len(ids))
    a2 = c._qwen_step(c._text_hc([78])[0], a, len(ids) + 1)
    c2 = c._qwen_prefill(c._text_hc([77, 78]), b, len(ids))
    assert torch.allclose(a2, c2, atol=2e-3, rtol=2e-3)


def test_rvchat_replays_recorded_packets_with_teacher_forcing(tpl):
    """fx's recorded requests and the recorded text: every later call is a pure extension."""
    from chipformer.rv import chat as CH

    stop = tpl.tok.token_to_id("<|im_end|>")
    sb = SC.ScriptedBackend(tpl, "t1_bug")
    forced = [sb.generate(0, None) for _ in sb.steps]
    assert all(f[-1] == stop for f in forced)
    chip = ReplayChip((DATA / "t1_bug.fxio").read_bytes())
    n = len(chip.reqs)
    recs = CH.fx_session(tiny_chat(), chip, tpl, forced=forced)
    assert [r["kind"] for r in recs] == ["first"] + ["exact"] * (n - 1)
    assert sum(r["rolled_back"] for r in recs) == 0
    assert all(r["delta_tokens"] < r["full_tokens"] for r in recs[1:])
    assert [P.parse_reply(p)[0] for p in chip.got] == [200] * n
    first = P.parse_reply(chip.got[0])[1]
    assert len(first) == 1 and b'"toolName":"read_file"' in first[0]


def test_rvchat_real_generation_falls_back_to_snapshots(tpl):
    """A tiny random model writes nonsense, so fx's recorded history never matches what it
    generated: each call goes back to where generation started. Only the work is lost."""
    from chipformer.rv import chat as CH

    chip = ReplayChip((DATA / "t1_bug.fxio").read_bytes())
    n = len(chip.reqs)
    recs = CH.fx_session(tiny_chat(), chip, tpl, max_new=3)
    assert [r["kind"] for r in recs] == ["first"] + ["rollback"] * (n - 1)
    assert all(r["kept"] == recs[i]["full_tokens"] for i, r in enumerate(recs[1:]))
    assert len(chip.got) == n
