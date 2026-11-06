"""Small non-tensor fixtures for the actual generation loop and token consumer."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import time

SOURCE = None
TCO, TCC, CHIP, EOS, ANSWER = 248058, 248059, 248321, 248046, 301
ACK = "This calculation is already complete. I will use the returned result.\n"


class Tokenizer:
    def __init__(self):
        self.text = {11: "This calculation is already complete. ",
                     12: "I will use the returned result.\n",
                     TCO: "<tool_call>", TCC: "</tool_call>", ANSWER: "The answer is 3."}

    def encode(self, text, add_special_tokens=False):
        assert text == ACK and add_special_tokens is False
        return [11, 12]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.text.get(i, "") for i in ids)


def complete_call(control, tokenizer, expression="1+2", response="3", *, json_call=False):
    tokenizer.text[100] = (
        '{"name":"calc","arguments":{"expression":' + json.dumps(expression) + "}}"
        if json_call else
        f"<function=calc><parameter=expression>{expression}</parameter></function>")
    tokenizer.text[200] = f"<tool_response>{response}</tool_response>\n"
    for current, predicted in ((TCO, None), (100, None), (TCC, CHIP), (CHIP, 200)):
        assert control(current, predicted, predicted or ANSWER) is None
    selected = []
    token = control(200, None, ANSWER)
    while token is not None:
        assert len(selected) < 64
        selected.append(token)
        token = control(token, None, ANSWER)
    return selected


class Scores:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, key):
        return self


class Hidden(dict):
    def __getitem__(self, key):
        return self if isinstance(key, slice) else super().__getitem__(key)


TORCH = SimpleNamespace(no_grad=lambda: lambda fn: fn, tensor=lambda value, **kw: value,
                        argmax=lambda value: value.token, bfloat16="bfloat16")


def generate_method():
    raw = SOURCE if SOURCE is not None else (
        Path(__file__).parents[1] / "chipformer/rv/chat.py").read_bytes()
    tree = ast.parse(raw)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    method = next(n for n in classes["RvChat"].body
                  if isinstance(n, ast.FunctionDef) and n.name == "generate")
    nodes = ([classes["NextTokenControlError"]] if "NextTokenControlError" in classes else [])
    scope = {"torch": TORCH, "time": time, "core": SimpleNamespace(snapshot=lambda *a: None),
             "hf": SimpleNamespace(inject=lambda *a: None),
             "BR": SimpleNamespace(text_token=lambda token: token + 500000),
             "S": SimpleNamespace(VOCAB=230324, TEXT_TOKEN=500000, name=str)}
    exec(compile(ast.Module(body=nodes + [method], type_ignores=[]),
                 "<generate-control-flow-test>", "exec"), scope)
    return scope["generate"]


class View:
    def __init__(self, backend="hf", stream=None):
        self.v0, self.eos, self.dev = 248320, {EOS}, "fixture"
        self.executor, self.fast, self.verify = backend == "executor", backend == "fast", False
        self.n_chip, self.c, self.code_dims = 0, object(), [1]
        self.graph, self.tok = object(), Tokenizer()
        self.cm = SimpleNamespace(embed_tokens=lambda x: x)
        self.index, self.forward_count, self.prefills = 0, 0, 0
        self.stream = stream or [(10, None, 20), (20, CHIP, CHIP),
                                 (CHIP, 30, 30), (30, None, 40)]
        self.model = self.prefill

    def prefill(self, **kwargs):
        self.prefills += 1
        return SimpleNamespace(past_key_values=object(), logits=Scores(self.stream[0][0]))

    def process(self, token):
        self.forward_count += 1
        if self.index < len(self.stream):
            current, predicted, ordinary = self.stream[self.index]
            assert token == current, (token, current)
        else:
            predicted, ordinary = None, EOS if token == ANSWER else ANSWER
        self.index += 1
        return Hidden(next=predicted, scores=Scores(ordinary))

    def _layers(self, token, cache, pos):
        hidden = self.process(token)
        return hidden, hidden["scores"]

    def circuit_token(self, hidden):
        return self.code_token(hidden)

    def code_token(self, hidden):
        return hidden["next"]

    def _exec_start(self, ids):
        pass

    def _exec_step(self, token):
        return self.process(token)

    def _exec_hc(self, token, hidden):
        return hidden

    def _chip_cache(self):
        return object()

    def _chip_step(self, token, pos, cache):
        return self.process(token)

    def _qwen_step(self, hidden, cache, pos):
        return hidden["scores"]
