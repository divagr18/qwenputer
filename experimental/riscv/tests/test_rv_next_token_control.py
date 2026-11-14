"""Exercise actual generation control flow without loading torch or a checkpoint."""

from types import SimpleNamespace
import unittest

from next_token_fixtures import ANSWER, CHIP, EOS, View, generate_method


class NextTokenTests(unittest.TestCase):
    def setUp(self):
        self.method = generate_method()

    def exercise(self, backend="hf", **kwargs):
        view = View(backend, kwargs.pop("stream", None))
        result = self.method(view, [1], object(), max_new=kwargs.pop("max_new", 64),
                        max_chip=kwargs.pop("max_chip", 64), **kwargs)
        return result, view

    def test_callback(self):
        for backend in ("hf", "fast", "executor"):
            calls = []
            def choose(current, owner, default):
                calls.append((current, owner, default))
                return 41 if current == 30 else None
            result, view = self.exercise(backend, on_step=choose)
            self.assertEqual(result["tokens"], [10, 20, CHIP, 30, 41, ANSWER, EOS])
            self.assertEqual(calls[:4], [(10, None, 20), (20, CHIP, CHIP),
                                        (CHIP, 30, 30), (30, None, 40)])
            self.assertEqual(len(calls), view.forward_count)

    def test_none_and_noop(self):
        for backend in ("hf", "fast", "executor"):
            plain, old = self.exercise(backend)
            inactive, new = self.exercise(backend, on_step=None)
            noop, _ = self.exercise(backend, on_step=lambda *args: None)
            self.assertEqual(plain["tokens"], [10, 20, CHIP, 30, 40, ANSWER, EOS])
            for value in (plain, inactive, noop):
                for key in ("seconds", "chip_seconds", "text_seconds"):
                    value.pop(key)
            self.assertEqual(plain, inactive)
            self.assertEqual(plain, noop)
            self.assertEqual((old.prefills, old.forward_count), (new.prefills, new.forward_count))

    def test_errors(self):
        method = self.method
        error = method.__globals__["NextTokenControlError"]
        view = View()
        with self.assertRaises(error) as caught:
            method(view, [1], object(), on_step=3)
        self.assertEqual(caught.exception.code, "not_callable")
        self.assertEqual(view.prefills, 0)
        for selected in (True, -1, 248320, 1.0, "1"):
            emitted = []
            with self.assertRaises(error) as caught:
                self.exercise(on_step=lambda *args: selected, on_token=emitted.append)
            self.assertEqual(caught.exception.code, "invalid_token")
            self.assertEqual(emitted, [10])
        for backend in ("hf", "fast", "executor"):
            for current in (20, CHIP):
                with self.assertRaises(error) as caught:
                    self.exercise(backend, on_step=lambda token, owner, default:
                                  30 if token == current else None)
                self.assertEqual(caught.exception.code, "machine_owned")
        failure = RuntimeError("callback failure")
        def fail(*args):
            raise failure
        with self.assertRaises(RuntimeError) as caught:
            self.exercise(on_step=fail)
        self.assertIs(caught.exception, failure)

    def test_stops(self):
        for text, chip, expected in ((0, 64, 0), (64, 0, 0), (3, 64, 4), (64, 1, 3)):
            calls = []
            result, view = self.exercise(on_step=lambda *args: calls.append(args),
                                         max_new=text, max_chip=chip)
            self.assertEqual(len(calls), expected)
            self.assertEqual(len(result["tokens"]), expected)
            self.assertEqual(view.forward_count, expected)
        calls = []
        result, view = self.exercise(stream=[(EOS, None, 1)],
                                     on_step=lambda *args: calls.append(args))
        self.assertEqual(result["tokens"], [EOS])
        self.assertEqual(calls, [])
        self.assertEqual(view.forward_count, 0)
        def fail(_):
            raise RuntimeError("reference stop")
        result, _ = self.exercise(lock=SimpleNamespace(toks=[], qwen=fail),
                                  on_step=lambda *args: self.fail("callback after lock failure"))
        self.assertIsNotNone(result["lock_error"])


if __name__ == "__main__":
    unittest.main()
