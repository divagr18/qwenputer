"""The tiny repository the measured agent works on (10 small source files)."""
import os

FILES = {
    "README.md": "# tinycalc\n\nA small calculator library used to measure fx on Qwenputer.\n\nRun `python -m tinycalc 1+2`.\n",
    "pyproject.toml": "[project]\nname = \"tinycalc\"\nversion = \"0.1.0\"\nrequires-python = \">=3.10\"\n",
    "tinycalc/__init__.py": "from .calc import add, sub, mul, div\nfrom .parse import evaluate\n\n__all__ = [\"add\", \"sub\", \"mul\", \"div\", \"evaluate\"]\n",
    "tinycalc/__main__.py": "import sys\n\nfrom .parse import evaluate\n\n\ndef main(argv):\n    for expr in argv[1:]:\n        print(expr, \"=\", evaluate(expr))\n\n\nif __name__ == \"__main__\":\n    main(sys.argv)\n",
    "tinycalc/calc.py": (
        "\"\"\"Integer arithmetic.\"\"\"\n\n\n"
        "def add(a, b):\n    return a + b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\n"
        "def mul(a, b):\n    return a * b\n\n\n"
        "def div(a, b):\n    # TODO: reject division by zero with a clear error\n    return a // b\n"
    ),
    "tinycalc/parse.py": (
        "\"\"\"A tiny left-to-right expression evaluator.\"\"\"\n\n"
        "from .calc import add, sub, mul, div\n\n"
        "OPS = {\"+\": add, \"-\": sub, \"*\": mul, \"/\": div}\n\n\n"
        "def tokens(expr):\n    num = \"\"\n    for ch in expr:\n        if ch.isdigit():\n            num += ch\n"
        "        elif ch in OPS:\n            yield int(num)\n            yield ch\n            num = \"\"\n"
        "    if num:\n        yield int(num)\n\n\n"
        "def evaluate(expr):\n    it = tokens(expr)\n    acc = next(it)\n    for op in it:\n        acc = OPS[op](acc, next(it))\n    return acc\n"
    ),
    "tinycalc/fmt.py": "def show(value, width=8):\n    return str(value).rjust(width)\n",
    "tests/test_calc.py": (
        "from tinycalc.calc import add, sub, mul, div\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_div():\n    assert div(7, 2) == 3\n"
    ),
    "tests/test_parse.py": "from tinycalc import evaluate\n\n\ndef test_evaluate():\n    assert evaluate(\"1+2*3\") == 9\n",
    "docs/notes.md": "# Notes\n\n- evaluation is left to right, no precedence\n- division is floor division\n",
}


def make_repo(root):
    for rel, text in FILES.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="\n", encoding="utf-8") as f:
            f.write(text)
    return root
