"""Prompts with tool-call markup from earlier turns, for plain-forward checks (docs/fx/fwd.md).

Every conversation here has one or more completed calls: the assistant's ``<tool_call>`` block,
then ``<|im_end|>`` and the response as a user turn, as Qwen3.5's chat template renders a history.
``with_tools`` adds the tools block to the system prompt, whose format example holds two more
``</tool_call>`` tokens.
"""

from __future__ import annotations

CALC = {"type": "function", "function": {
    "name": "calc", "description": "Evaluate an arithmetic expression exactly.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}
READ = {"type": "function", "function": {
    "name": "read_file", "description": "Read a text file of the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def conversations(long: bool = False) -> list[tuple[str, list, list]]:
    """(name, messages, tools of the system prompt). Each ends on a user turn."""
    calc1 = [{"role": "user", "content": "What is 123456789 * 987654321?"},
             _call("calc", expression="123456789*987654321"),
             {"role": "tool", "content": "121932631112635269"},
             {"role": "assistant", "content": "123456789 * 987654321 = 121932631112635269."},
             {"role": "user", "content": "And what is that divided by 3?"}]
    calc3 = [{"role": "user", "content": "Compute 17*23, then 2**40, then 99999*99999."},
             _call("calc", expression="17*23"), {"role": "tool", "content": "391"},
             _call("calc", expression="2**40"), {"role": "tool", "content": "1099511627776"},
             _call("calc", expression="99999*99999"), {"role": "tool", "content": "9999800001"},
             {"role": "assistant", "content": "17*23 = 391, 2**40 = 1099511627776 and 99999*99999 = 9999800001."},
             {"role": "user", "content": "Which of the three is the largest? Answer in one sentence."}]
    read1 = [{"role": "user", "content": "What does a.py print?"},
             _call("read_file", path="a.py"),
             {"role": "tool", "content": "1\tprint('hello')"},
             {"role": "assistant", "content": "It prints hello."},
             {"role": "user", "content": "Write a one-line docstring for it."}]
    many = [{"role": "user", "content": "Add up the squares of 1 to 12, one call at a time."}]
    for k in range(1, 13):
        many += [_call("calc", expression=f"{k}**2"), {"role": "tool", "content": str(k * k)}]
    many += [{"role": "assistant", "content": "The squares are 1, 4, 9, ..., 144."},
             {"role": "user", "content": "Now give the total and explain how you got it."}]
    out = [("calc1", calc1, []), ("calc3", calc3, []), ("read1", read1, []),
           ("calc1_tools", calc1, [CALC]), ("read1_tools", read1, [READ]),
           ("calc12_tools", many, [CALC])]
    if long:  # 2,145 tokens with the tools block, 42 </tool_call>
        lots = [{"role": "user", "content": "Cube 1 to 40, one call each."}]
        for k in range(1, 41):
            lots += [_call("calc", expression=f"{k}**3"), {"role": "tool", "content": str(k ** 3)}]
        lots += [{"role": "assistant", "content": "Done: the cubes of 1 to 40."},
                 {"role": "user", "content": "Which of them are also perfect squares?"}]
        out.append(("calc40_tools", lots, [CALC]))
    return out


def encode_all(tok, long: bool = False) -> list[tuple[str, list[int]]]:
    out = []
    for name, msgs, tools in conversations(long):
        ids = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        out.append((name, list(ids)))
    return out
