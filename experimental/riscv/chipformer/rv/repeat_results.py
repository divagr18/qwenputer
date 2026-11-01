"""Optional acknowledgement after a calculator successfully repeats an expression."""

from collections import deque
import json
import re

ACKNOWLEDGEMENT = "This calculation is already complete. I will use the returned result.\n"
_LIMIT = 8192
_NUMERIC = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+|/[0-9]+)?")


def _expression(body):
    if body is None:
        return None
    xml = re.fullmatch(
        r"\s*<function=calc>\s*<parameter=expression>\s*(.*?)\s*</parameter>\s*</function>\s*",
        body, re.DOTALL)
    if xml:
        return xml[1]
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or value.get("name") != "calc":
        return None
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        return None
    expression = arguments.get("expression")
    return expression if isinstance(expression, str) else None


def _append(buffer, token):
    if buffer is None or len(buffer) >= _LIMIT:
        return None
    buffer.append(token)
    return buffer


class RepeatedResultAcknowledgement:
    """An on_step callable. Create a fresh instance for each generate() call.

    Only completed, circuit-owned numeric responses count. After the second
    success for a whitespace-normalized expression, select ACKNOWLEDGEMENT once.
    Other expressions and errors do not trigger it. This is explicit decoding
    control: it chooses text tokens, without changing the calculator's result.

    Buffers are bounded; an oversized call or response receives no acknowledgement.
    After max_calls completions the controller becomes inactive, finishing any
    acknowledgement already selected at the final completion.
    """

    def __init__(self, tokenizer, text_vocab_size, tool_call_open, tool_call_close,
                 *, eos_tokens=(), max_calls=128):
        if (type(text_vocab_size) is not int or not 1 <= text_vocab_size <= 2**31
                or type(max_calls) is not int or not 1 <= max_calls <= 4096
                or not callable(getattr(tokenizer, "encode", None))
                or not callable(getattr(tokenizer, "decode", None))):
            raise ValueError("Invalid repeated-result controller settings.")
        if (type(eos_tokens) not in (tuple, list, set, frozenset)
                or len(eos_tokens) > 32
                or any(type(t) is not int or not 0 <= t < text_vocab_size
                       for t in (*eos_tokens, tool_call_open, tool_call_close))
                or tool_call_open == tool_call_close):
            raise ValueError("Invalid text control tokens.")
        prefix = tokenizer.encode(ACKNOWLEDGEMENT, add_special_tokens=False)
        forbidden = set(eos_tokens) | {tool_call_open, tool_call_close}
        if (type(prefix) is not list or not 1 <= len(prefix) <= 64
                or any(type(t) is not int or not 0 <= t < text_vocab_size or t in forbidden
                       for t in prefix)
                or tokenizer.decode(prefix, skip_special_tokens=False) != ACKNOWLEDGEMENT):
            raise ValueError("The acknowledgement must encode as ordinary text tokens.")
        self.tokenizer, self.v0 = tokenizer, text_vocab_size
        self.open_token, self.close_token = tool_call_open, tool_call_close
        self.max_calls, self.prefix = max_calls, tuple(prefix)
        self.completed_calls = self.acknowledgements = 0
        self._successes, self._acknowledged = set(), set()
        self._previous = self._chosen = None
        self._body, self._echo = None, []
        self._expression, self._in_call = None, False
        self._pending = deque()

    @classmethod
    def for_chat(cls, chat, **options):
        return cls(chat.tok, chat.v0, chat.T.TCO, chat.T.TCC,
                   eos_tokens=chat.eos, **options)

    @property
    def exhausted(self):
        return self.completed_calls >= self.max_calls

    def _decode(self, tokens):
        if tokens is None:
            return None
        value = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if not isinstance(value, str):
            raise ValueError("The tokenizer returned non-text output.")
        try:
            return value if len(value.encode("utf-8")) <= _LIMIT else None
        except UnicodeEncodeError:
            return None

    def _complete(self):
        text = self._decode(self._echo)
        bodies = [] if text is None else re.findall(
            r"<tool_response>(.*?)</tool_response>", text, re.DOTALL)
        response = bodies[0].strip() if len(bodies) == 1 else ""
        numeric = _NUMERIC.fullmatch(response) is not None
        if numeric and "/" in response:
            numeric = bool(response.rsplit("/", 1)[1].strip("0"))
        key = "" if self._expression is None else "".join(self._expression.split())
        self.completed_calls += 1
        if key and numeric:
            if key in self._successes and key not in self._acknowledged:
                self._pending.extend(self.prefix)
                self._acknowledged.add(key)
                self.acknowledgements += 1
            self._successes.add(key)
        self._expression, self._in_call, self._echo = None, False, []

    def __call__(self, current_token, circuit_next, default_next):
        if self.exhausted and not self._pending:
            return None
        if (type(current_token) is not int or not 0 <= current_token < 2**31
                or circuit_next is not None and
                (type(circuit_next) is not int or not 0 <= circuit_next < 2**31)
                or type(default_next) is not int or not 0 <= default_next < 2**31):
            raise ValueError("Invalid next-token callback arguments.")
        if self._chosen is not None and current_token != self._chosen:
            raise ValueError("The selected text token was not emitted.")
        self._chosen = None
        if self._previous is not None:
            if current_token != self._previous or not self._in_call:
                raise ValueError("The calculator token stream lost circuit ownership.")
            if current_token < self.v0:
                self._echo = _append(self._echo, current_token)
            if circuit_next is None:
                self._complete()
        else:
            if current_token >= self.v0:
                raise ValueError("A machine token appeared without circuit ownership.")
            if current_token == self.open_token:
                self._body = []
            elif current_token == self.close_token:
                if self._in_call:
                    raise ValueError("A new calculator call began before the previous return.")
                self._expression = _expression(self._decode(self._body))
                self._body, self._echo, self._in_call = None, [], True
            elif self._body is not None:
                self._body = _append(self._body, current_token)
        self._previous = circuit_next
        if self._pending:
            if circuit_next is not None:
                raise ValueError("An acknowledgement cannot replace a circuit-owned token.")
            self._chosen = self._pending.popleft()
            return self._chosen
        return None
