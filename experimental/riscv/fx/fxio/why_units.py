"""docs/fx/pr.md: which weights of the 13-unit circuit do not fit bf16 (chipbuild's emission)? Runs live_chip's
--write-chip-file path with chipbuild's bf16 check recording instead of failing."""
import collections, json, struct, sys
sys.path.insert(0, "fx/fxio")
from chipformer.rv import chipbuild as CB
bad, ex = collections.Counter(), []
orig = CB._is_bf16
def rec(x):
    try:
        ok = orig(x)
    except OverflowError:
        ok = False
    if not ok:
        bad["overflow" if abs(x) > 3.4e38 else "not_bf16"] += 1
        if len(ex) < 10:
            ex.append(x)
    return True
CB._is_bf16 = rec
import live_chip as LC
a = LC.parse(sys.argv[1:])
chip, template, backend = LC.setup(a)
try:
    LC.session(a, chip, template, backend, a.tasks[0], 1, a.out, print)
except BaseException as e:
    print("ended:", type(e).__name__, str(e)[:200])
print("BAD", json.dumps({"count": bad, "examples": ex}))
