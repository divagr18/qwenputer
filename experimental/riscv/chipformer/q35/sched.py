"""Re-place a built graph's nodes by criticality (list scheduling).

Nodes are placed greedily as they are created, so a node takes the earliest sublayer with room
even when it has slack, and a later, longer chain finds the sublayers above its inputs full.
`reschedule` places every node again, sublayer by sublayer: of the nodes whose inputs are ready,
the ones with the longest chain of dependents to a sink go first. Semantics don't change (the
evaluator runs nodes in creation order; levels only have to respect every read), and a node
keeps its min_level.

A node depends on the writers of the dims it reads that were created before it. A writer that
adds onto a dim (into) after a reader was created must not move below that reader. Nodes named
`last` (the exit) go above every other node, and their min_level (set for greedy placement) is
dropped.
"""

from collections import defaultdict

from chipformer.q35.emit import reads
from chipformer.q35.ir import FFN


def outputs(n) -> list[int]:
    if isinstance(n, FFN):
        return [n.out]
    return [d for vh in n.vheads() for _, d, _ in vh.out]


def reschedule(g, unit_cap: int, head_cap: int, last: str = "exit.") -> None:
    nodes = g.nodes
    n_nodes = len(nodes)
    is_ffn = [isinstance(n, FFN) for n in nodes]
    writers = defaultdict(list)  # dim -> writer node indices, in creation order
    readers = defaultdict(list)
    deps = [set() for _ in range(n_nodes)]  # (pred, gap): node >= pred + gap
    for i, n in enumerate(nodes):
        for d in set(reads(n)):
            for j in writers[d]:
                deps[i].add((j, 1))
            readers[d].append(i)
        for d in outputs(n):
            for r in readers[d]:  # read before this write: stay at or above the reader
                if r != i:
                    deps[i].add((r, 0))
            writers[d].append(i)
    succ = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j, gap in deps[i]:
            succ[j].append((i, gap))
    # height: sublayers from a node to the end of its longest chain of dependents
    height = [0] * n_nodes
    for i in range(n_nodes - 1, -1, -1):
        h = 0
        for j, gap in succ[i]:
            step = 0 if gap == 0 else (2 if is_ffn[i] == is_ffn[j] else 1)
            h = max(h, height[j] + step)
        height[i] = h
    level = [None] * n_nodes
    n_left = [len(deps[i]) for i in range(n_nodes)]
    is_last = [n.name.startswith(last) for n in nodes]
    earliest = [1 if is_last[i] else getattr(n, "min_level", 0) + 1 for i, n in enumerate(nodes)]
    n_body = n_nodes - sum(is_last)
    ready = [i for i in range(n_nodes) if not n_left[i]]
    placed, placed_body, sub = 0, 0, 1
    units, heads = defaultdict(int), defaultdict(int)
    while placed < n_nodes:
        want_ffn = sub % 2 == 0
        body_done = placed_body == n_body
        floor = (max((level[i] for i in range(n_nodes) if not is_last[i]), default=0) + 1
                 if body_done else None)
        cand = sorted((i for i in ready if is_ffn[i] == want_ffn and earliest[i] <= sub
                       and (not is_last[i] or body_done and sub >= floor)),
                      key=lambda i: (-height[i], i))
        done = []
        for i in cand:
            if want_ffn:
                need = len(nodes[i].terms)
                if units[sub] + need > unit_cap:
                    continue
                units[sub] += need
            else:
                if heads[sub] + 1 > head_cap:
                    continue
                heads[sub] += 1
            level[i] = sub
            done.append(i)
        if done:
            ds = set(done)
            ready = [i for i in ready if i not in ds]
            placed += len(done)
            placed_body += sum(not is_last[i] for i in done)
            for i in done:
                for j, gap in succ[i]:
                    earliest[j] = max(earliest[j], sub + gap)
                    n_left[j] -= 1
                    if not n_left[j]:
                        ready.append(j)
            # a gap-0 dependent (a later writer into a dim i reads) may go in this sublayer
            continue
        sub += 1
        assert sub < 1000, "reschedule: no progress"
    for i, n in enumerate(nodes):
        n.level = level[i]
    for d, ws in writers.items():
        if ws:
            g.dims[d].level = max(level[i] for i in ws)
    g.units = dict(units)
    g.heads = dict(heads)


def check(g) -> None:
    """Levels respect every read: a node sees each writer of a dim it reads that was created
    before it (strictly below) and none created after it (at or above); kinds sit on their
    parity; capacities hold."""
    writers = defaultdict(list)
    order = {}
    for i, n in enumerate(g.nodes):
        order[i] = n.level
        assert (n.level % 2 == 0) == isinstance(n, FFN), (n.name, n.level)
        for d in outputs(n):
            writers[d].append(i)
    for i, n in enumerate(g.nodes):
        for d in set(reads(n)):
            for j in writers.get(d, ()):
                if j < i:
                    assert order[j] < n.level, (n.name, n.level, g.nodes[j].name, order[j])
                elif j > i:
                    assert order[j] >= n.level, (n.name, n.level, g.nodes[j].name, order[j])
    units, heads = defaultdict(int), defaultdict(int)
    for n in g.nodes:
        if isinstance(n, FFN):
            units[n.level] += len(n.terms)
        else:
            heads[n.level] += 1
    assert max(units.values()) <= g.target.intermediate, max(units.values())
    assert max(heads.values()) <= g.target.key_heads, max(heads.values())
