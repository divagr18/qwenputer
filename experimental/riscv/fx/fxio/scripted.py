"""A stand-in for Qwen: fixed tool-call plans for the P0c tasks, written out in Qwen's raw text
format and tokenized with Qwen's real tokenizer.

THIS IS NOT QWEN. A solved task here means the packet path carried a correct plan through fx's
binary. It says nothing about whether Qwen can solve the task. It exists so that everything
around the model can be run and measured without a GPU: the template, the delta, the parser,
the SSE, fx's agent loop and tools, instruction and token counts.

Each plan copies the shape of a real P0c noshell run (the recorded tool order; shell steps left
out, since the shell is not one of the five tools here), including t1's failed first edit. P0c
did not record arguments or reasoning, so the arguments are written by hand from the recorded
diffs, and each step's reasoning is filler sized to that run's recorded output tokens per step,
so that request bodies grow the way they did.
"""
from __future__ import annotations

FILLER = ("Let me look at what the code does here and decide on the smallest change that keeps "
          "the existing behaviour for every other caller. ")

MONEY_OLD = "    for i in range(len(prices) - 1):\n"
MONEY_NEW = "    for i in range(len(prices)):\n"
PARSE_OLD = "    return name, int(qty)\n"
PARSE_NEW = "    return name.strip(), int(qty)\n"
TEXT_PY = ("import re\n\n\ndef slugify(text):\n"
           "    return re.sub(r\"[^a-z0-9]+\", \"-\", text.lower()).strip(\"-\")\n")


def _read(p):
    return ("read_file", {"path": p})


def _edit(p, old, new):
    return ("edit_file", {"path": p, "old_string": old, "new_string": new})


# task: (recorded output tokens per step of the run copied, steps as (text, tool call or None))
PLANS = {
    "t1_bug": (143, [
        ("", _read("inventory/money.py")),
        ("Found the bug: the loop skips the last element.",
         _edit("inventory/money.py", "\tfor i in range(len(prices) - 1):\n", "\tfor i in range(len(prices)):\n")),
        ("The file may use different indentation. Let me check.", _read("inventory/money.py")),
        ("", _edit("inventory/money.py", MONEY_OLD, MONEY_NEW)),
        ("Fixed. Let me verify the change.", _read("inventory/money.py")),
        ("Done. The loop skipped the last price; it now covers all of them.", None),
    ]),
    "t2_rename": (109, [
        ("I'll find every use first.", ("grep_files", {"pattern": "get_count", "path": "."})),
        ("", _read("inventory/store.py")),
        ("", _read("inventory/report.py")),
        ("", _read("tests/test_store.py")),
        ("Now the three edits.", _edit("inventory/store.py", "def get_count(self, name):", "def count(self, name):")),
        ("", _edit("inventory/report.py", "store.get_count(name)", "store.count(name)")),
        ("", _edit("tests/test_store.py", "s.get_count(\"apple\")", "s.count(\"apple\")")),
        ("Let me verify nothing is left.", ("grep_files", {"pattern": "get_count", "path": "."})),
        ("Renamed `get_count` to `count` in the method, its caller and the test.", None),
    ]),
    "t3_feature": (200, [
        ("Let me see the layout.", ("glob_files", {"pattern": "**/*.py"})),
        ("", _read("inventory/money.py")),
        ("", _read("inventory/parse.py")),
        ("", ("write_file", {"path": "inventory/text.py", "content": TEXT_PY})),
        ("Let me verify the file.", _read("inventory/text.py")),
        ("Added `slugify` in inventory/text.py.", None),
    ]),
    "t4_config": (78, [
        ("I'll find where the threshold lives.", ("grep_files", {"pattern": "LOW_STOCK", "path": "."})),
        ("", ("grep_files", {"pattern": "low_stock", "path": "."})),
        ("", ("grep_files", {"pattern": "< 5", "path": "."})),
        ("", ("glob_files", {"pattern": "**/*.py"})),
        ("", _read("inventory/config.py")),
        ("", _edit("inventory/config.py", "LOW_STOCK = 5\n", "LOW_STOCK = 10\n")),
        ("", _read("inventory/config.py")),
        ("Low stock now means fewer than 10 units.", None),
    ]),
    "t5_parse": (204, [
        ("", _read("inventory/parse.py")),
        ("The name keeps its whitespace.", _edit("inventory/parse.py", PARSE_OLD, PARSE_NEW)),
        ("Let me verify.", _read("inventory/parse.py")),
        ("`parse_line` now trims the name and keeps inner spaces.", None),
    ]),
}


def tool_text(name: str, args: dict) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def raw_step(reasoning: str, text: str, call) -> str:
    """Exactly what a generation looks like after the prompt's `<think>\\n`."""
    out = reasoning + "\n</think>\n\n" + text
    if call:
        out += ("\n\n" if text else "") + tool_text(*call)
    return out


class ScriptedBackend:
    """DeltaLink's backend interface with no model behind it."""

    def __init__(self, template, task: str, pad: bool = True, extra: int = 0) -> None:
        self.t = template
        self.per_step, steps = PLANS[task]
        self.steps = steps + [("In short: the change is made and verified.", None)] * extra
        self.pad = pad
        self.i = 0
        self.pos = 0
        self.fed = 0
        self.restores: list[int] = []

    def reset(self) -> None:
        self.pos = 0

    def feed(self, ids: list[int]) -> None:
        self.pos += len(ids)
        self.fed += len(ids)

    def snapshot(self, n: int) -> None:
        assert n == self.pos, (n, self.pos)

    def restore(self, n: int) -> None:
        self.restores.append(n)
        self.pos = n

    def _reasoning(self, text: str, call) -> str:
        r = "This step follows from what the task asks for."
        if self.pad:
            base = len(self.t.encode(raw_step(r, text, call)))
            unit = len(self.t.encode(FILLER))
            r = (FILLER * max(0, (self.per_step - base) // unit)).strip() + " " + r
        return r.strip()

    def generate(self, max_new: int, stop_ids) -> list[int]:
        if self.i >= len(self.steps):
            raise RuntimeError("the plan is finished but fx asked again")
        text, call = self.steps[self.i]
        self.i += 1
        ids = self.t.encode(raw_step(self._reasoning(text, call), text, call))
        self.pos += len(ids)
        return ids + [self.t.tok.token_to_id("<|im_end|>")]
