"""P0c: can base Qwen3.5-4B drive native fx? Five tiny-repo tasks with hidden checks.

    python tasks.py make ROOT        # one git repo per task under ROOT/<task>
    python tasks.py check ROOT TASK  # exit 0 if the task is solved
"""
import json, subprocess, sys, textwrap
from pathlib import Path

COMMON = {
    "README.md": "# inventory\n\nA tiny inventory library. Run the tests with `python -m unittest`.\n",
    "inventory/__init__.py": "",
    "inventory/config.py": "MAX_ITEMS = 100\nDEFAULT_CURRENCY = 'USD'\nLOW_STOCK = 5\n",
    "inventory/money.py": textwrap.dedent('''\
        def total(prices):
            """Sum of a list of prices in cents."""
            s = 0
            for i in range(len(prices) - 1):
                s += prices[i]
            return s


        def fmt(cents, currency="USD"):
            return f"{cents // 100}.{cents % 100:02d} {currency}"
        '''),
    "inventory/store.py": textwrap.dedent('''\
        from inventory.config import MAX_ITEMS, LOW_STOCK


        class Store:
            def __init__(self):
                self.items = {}

            def add(self, name, qty):
                if len(self.items) >= MAX_ITEMS and name not in self.items:
                    raise ValueError("store full")
                self.items[name] = self.items.get(name, 0) + qty

            def get_count(self, name):
                return self.items.get(name, 0)

            def low_stock(self):
                return sorted(n for n, q in self.items.items() if q < LOW_STOCK)
        '''),
    "inventory/report.py": textwrap.dedent('''\
        from inventory.store import Store


        def summary(store: Store):
            lines = []
            for name in sorted(store.items):
                lines.append(f"{name}: {store.get_count(name)}")
            return "\\n".join(lines)
        '''),
    "inventory/parse.py": textwrap.dedent('''\
        def parse_line(line):
            """'apple, 3' -> ('apple', 3). Names may contain spaces; whitespace is trimmed."""
            name, qty = line.split(",")
            return name, int(qty)
        '''),
    "tests/__init__.py": "",
    "tests/test_store.py": textwrap.dedent('''\
        import unittest
        from inventory.store import Store


        class T(unittest.TestCase):
            def test_add(self):
                s = Store()
                s.add("apple", 2)
                s.add("apple", 3)
                self.assertEqual(s.get_count("apple"), 5)
        '''),
}

TASKS = {
    "t1_bug": {
        "prompt": "The total() function in inventory/money.py gives the wrong sum. Fix it.",
        "check": "from inventory.money import total\nassert total([100, 250, 5]) == 355\nassert total([]) == 0\nassert total([7]) == 7\n",
    },
    "t2_rename": {
        "prompt": "Rename the Store method get_count to count everywhere in the repo, including its callers.",
        "check": ("from inventory.store import Store\nfrom inventory.report import summary\n"
                  "s = Store(); s.add('a', 2)\nassert s.count('a') == 2\nassert not hasattr(s, 'get_count')\n"
                  "assert summary(s) == 'a: 2'\n"),
    },
    "t3_feature": {
        "prompt": ("Add a function slugify(text) to a new file inventory/text.py. It lowercases the text, "
                   "replaces every run of characters that are not a-z or 0-9 with a single hyphen, and "
                   "strips hyphens from both ends. Example: slugify('  Hello, World!! ') == 'hello-world'."),
        "check": ("from inventory.text import slugify\nassert slugify('  Hello, World!! ') == 'hello-world'\n"
                  "assert slugify('A--B__c 9') == 'a-b-c-9'\nassert slugify('---') == ''\n"),
    },
    "t4_config": {
        "prompt": "Items should count as low stock when they have fewer than 10 units, not 5. Make that change.",
        "check": ("from inventory.store import Store\ns = Store(); s.add('x', 7); s.add('y', 12)\n"
                  "assert s.low_stock() == ['x']\nimport inventory.config as c\nassert c.MAX_ITEMS == 100\n"),
    },
    "t5_parse": {
        "prompt": ("parse_line in inventory/parse.py breaks on input like '  green apple , 12 '. "
                   "Fix it so names keep inner spaces but surrounding whitespace is trimmed."),
        "check": ("from inventory.parse import parse_line\nassert parse_line('  green apple , 12 ') == ('green apple', 12)\n"
                  "assert parse_line('pear,1') == ('pear', 1)\n"),
    },
}


def make(root: Path):
    for t in TASKS:
        d = root / t
        for p, s in COMMON.items():
            f = d / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(s)
        g = lambda *a: subprocess.run(["git", *a], cwd=d, check=True, capture_output=True)
        g("init", "-q")
        g("add", "-A")
        g("-c", "user.name=p0c", "-c", "user.email=p0c@localhost", "commit", "-qm", "init")
    (root / "tasks.json").write_text(json.dumps({t: v["prompt"] for t, v in TASKS.items()}))


def check(root: Path, t: str) -> int:
    r = subprocess.run([sys.executable, "-c", TASKS[t]["check"]], cwd=root / t, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "fail")
    return r.returncode


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]))
    else:
        sys.exit(check(Path(sys.argv[2]), sys.argv[3]))
