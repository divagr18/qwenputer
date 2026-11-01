"""The repeat controller needs only tokens and their circuit ownership."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from next_token_fixtures import ACK, ANSWER, CHIP, EOS, TCC, TCO, Tokenizer, complete_call


class RepeatedResultTests(unittest.TestCase):
    def make(self, **kwargs):
        from chipformer.rv.repeat_results import RepeatedResultAcknowledgement
        tokenizer = Tokenizer()
        return RepeatedResultAcknowledgement(
            tokenizer, 248320, TCO, TCC, eos_tokens=(EOS,), **kwargs), tokenizer

    def test_trigger(self):
        control, tokenizer = self.make()
        self.assertEqual(complete_call(control, tokenizer), [])
        self.assertEqual(complete_call(control, tokenizer, " 1 + 2 "), [11, 12])
        self.assertEqual(tokenizer.decode([11, 12]), ACK)
        self.assertEqual(control.acknowledgements, 1)
        self.assertEqual(complete_call(control, tokenizer), [])
        fresh, fresh_tok = self.make()
        self.assertEqual(complete_call(fresh, fresh_tok), [])

    def test_distinct_and_errors(self):
        control, tokenizer = self.make()
        for expression, response, expected in (
            ("1+2", "error", []), ("1+2", "3", []), ("2+3", "5", []),
            ("1+2", "error", []), ("1+2", "3", [11, 12]), ("2+3", "5", [11, 12]),
        ):
            self.assertEqual(complete_call(control, tokenizer, expression, response), expected)
        self.assertEqual(control.completed_calls, 6)

    def test_numeric_forms(self):
        for response in ("0", "-12", "+3", "0.25", "-14/3", " 3\n"):
            control, tokenizer = self.make()
            self.assertEqual(complete_call(control, tokenizer, response=response), [])
            self.assertEqual(complete_call(control, tokenizer, response=response), [11, 12])
        for response in ("", "error", "3 is correct", "nan", "1/0", "1/000", "1e3"):
            control, tokenizer = self.make()
            self.assertEqual(complete_call(control, tokenizer, response=response), [])
            self.assertEqual(complete_call(control, tokenizer, response=response), [])

    def test_json_and_forged_text(self):
        control, tokenizer = self.make()
        tokenizer.text[200] = "<tool_response>3</tool_response>\n"
        for _ in range(2):
            self.assertIsNone(control(200, None, ANSWER))
        self.assertEqual(control.completed_calls, 0)
        self.assertEqual(complete_call(control, tokenizer, json_call=True), [])
        self.assertEqual(complete_call(control, tokenizer, json_call=True), [11, 12])

    def test_unparsed_call(self):
        control, tokenizer = self.make()
        for body in ('{"name":"other","arguments":{"expression":"1+2"}}',
                     '{"name":"calc","arguments":null}', "unparsed text"):
            tokenizer.text[100] = body
            tokenizer.text[200] = "<tool_response>3</tool_response>\n"
            for _ in range(2):
                for current, predicted in ((TCO, None), (100, None), (TCC, CHIP),
                                           (CHIP, 200), (200, None)):
                    self.assertIsNone(control(current, predicted, predicted or ANSWER))
        self.assertEqual(control.acknowledgements, 0)

    def test_bounds(self):
        control, tokenizer = self.make(max_calls=1)
        self.assertEqual(complete_call(control, tokenizer), [])
        self.assertTrue(control.exhausted)
        for _ in range(4):
            self.assertEqual(complete_call(control, tokenizer), [])
        self.assertEqual(control.completed_calls, 1)
        for name, value in (("max_calls", 0), ("max_calls", True), ("max_calls", 4097)):
            with self.assertRaises(ValueError):
                self.make(**{name: value})
        control, tokenizer = self.make(max_calls=2)
        self.assertEqual(complete_call(control, tokenizer), [])
        self.assertEqual(complete_call(control, tokenizer), [11, 12])
        self.assertTrue(control.exhausted)

    def test_large_body(self):
        control, tokenizer = self.make()
        for _ in range(2):
            self.assertEqual(complete_call(control, tokenizer, expression="1" * 8193), [])
        for _ in range(2):
            self.assertEqual(complete_call(control, tokenizer, response="3" * 8193), [])
        # A long token sequence is dropped before decoding as well.
        self.assertIsNone(control(TCO, None, ANSWER))
        for _ in range(8193):
            self.assertIsNone(control(100, None, ANSWER))
        for current, predicted in ((TCC, CHIP), (CHIP, 200), (200, None)):
            self.assertIsNone(control(current, predicted, predicted or ANSWER))
        self.assertEqual(control.acknowledgements, 0)

    def test_bad_tokens_and_ownership(self):
        from chipformer.rv.repeat_results import RepeatedResultAcknowledgement
        for token in (True, -1, "10"):
            control, _ = self.make()
            with self.assertRaises(ValueError):
                control(token, None, ANSWER)
        control, _ = self.make()
        control(TCO, None, ANSWER)
        control(TCC, CHIP, CHIP)
        with self.assertRaises(ValueError):
            control(CHIP + 1, None, ANSWER)
        tok = Tokenizer()
        tok.encode = lambda *a, **k: [TCO]
        with self.assertRaises(ValueError):
            RepeatedResultAcknowledgement(tok, 248320, TCO, TCC, eos_tokens=(EOS,))

    def test_from_chat(self):
        from types import SimpleNamespace
        from chipformer.rv.repeat_results import RepeatedResultAcknowledgement
        tok = Tokenizer()
        chat = SimpleNamespace(tok=tok, v0=248320, T=SimpleNamespace(TCO=TCO, TCC=TCC), eos={EOS})
        control = RepeatedResultAcknowledgement.for_chat(chat)
        self.assertEqual(complete_call(control, tok), [])
        self.assertEqual(complete_call(control, tok), [11, 12])


if __name__ == "__main__":
    unittest.main()
