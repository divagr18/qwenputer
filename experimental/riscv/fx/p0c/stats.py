import json, glob, collections, sys
os_ = sys.argv[1]
for cond in ["full", "noshell"]:
    R = []
    for f in sorted(glob.glob(f"{os_}/results/{cond}/t*-*.json")):
        d = json.load(open(f)); tc = d["tool_calls"]
        R.append(dict(steps=d["steps"], calls=len(tc), tc=tc, inp=d["usage"]["input_tokens"] or 0, out=d["usage"]["output_tokens"] or 0))
    n = len(R)
    tot = collections.Counter(c["name"] for r in R for c in r["tc"])
    errs = collections.Counter(c["name"] for r in R for c in r["tc"] if c["status"] != "success")
    print(cond, "runs", n, "steps mean", round(sum(r["steps"] for r in R) / n, 1), "max", max(r["steps"] for r in R),
          "| calls mean", round(sum(r["calls"] for r in R) / n, 1), "max", max(r["calls"] for r in R),
          "| input tok mean", round(sum(r["inp"] for r in R) / n), "| output tok mean", round(sum(r["out"] for r in R) / n))
    print("  tools", dict(tot.most_common()))
    print("  non-success by tool", dict(errs))
s = [json.loads(l) for l in open(f"{os_}/results/summary.jsonl")]
for cond in ["full", "noshell"]:
    S = [x for x in s if x["cond"] == cond]
    print(cond, "solved", sum(x["solved"] for x in S), "/", len(S), "seconds mean", round(sum(x["seconds"] for x in S) / len(S), 1), "max", max(x["seconds"] for x in S))
p = [json.loads(l) for l in open(f"{os_}/proxy.jsonl")]
print("model requests", len(p), "| system msgs/req", dict(collections.Counter(x["n_system"] for x in p)), "| system chars", collections.Counter(x["system_chars"] for x in p).most_common(2), "| tools exposed", len(p[-1]["tools"]))
