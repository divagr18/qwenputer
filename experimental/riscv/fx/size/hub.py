"""Range-read access to safetensors files on the Hub: headers and tensor slices, never whole shards."""
import json, struct, time, sys, functools
import numpy as np
import requests
from huggingface_hub import hf_hub_url, HfApi, get_token

DT = {"BF16": ("<u2", 2), "F16": ("<u2", 2), "F32": ("<f4", 4), "F64": ("<f8", 8), "U8": ("u1", 1),
      "I8": ("i1", 1), "I64": ("<i8", 8), "BOOL": ("u1", 1), "U16": ("<u2", 2), "I32": ("<i4", 4)}
S = requests.Session()
_t = get_token()
if _t:
    S.headers["Authorization"] = "Bearer " + _t

def rng(url, a, b):
    """bytes a..b-1"""
    for i in range(6):
        try:
            r = S.get(url, headers={"Range": f"bytes={a}-{b-1}"}, timeout=120, allow_redirects=True)
            if r.status_code in (200, 206):
                assert r.status_code == 206, r.status_code
                assert len(r.content) == b - a, (len(r.content), b - a)
                return r.content
            print("status", r.status_code, file=sys.stderr)
        except Exception as e:
            print("retry", e, file=sys.stderr)
        time.sleep(2 ** i)
    raise RuntimeError("range read failed")

def shards(repo, rev="main"):
    return [s.rfilename for s in HfApi().model_info(repo, revision=rev, files_metadata=True).siblings
            if s.rfilename.endswith(".safetensors")]

def header(repo, fn, rev="main"):
    url = hf_hub_url(repo, fn, revision=rev)
    n = struct.unpack("<Q", rng(url, 0, 8))[0]
    h = json.loads(rng(url, 8, 8 + n))
    meta = h.pop("__metadata__", None)
    return url, 8 + n, h, meta

def all_tensors(repo, rev="main"):
    out = {}
    for fn in shards(repo, rev):
        url, base, h, _ = header(repo, fn, rev)
        for k, v in h.items():
            a, b = v["data_offsets"]
            out[k] = dict(url=url, a=base + a, b=base + b, dtype=v["dtype"], shape=v["shape"], file=fn)
    return out

def read(t, a=0, b=None):
    """Raw bytes of tensor t's byte range a..b."""
    b = (t["b"] - t["a"]) if b is None else b
    return rng(t["url"], t["a"] + a, t["a"] + b)

def rows(t, r0, r1):
    """Rows r0..r1 as a little-endian numpy array of the stored type."""
    d, w = DT[t["dtype"]]
    cols = int(np.prod(t["shape"][1:]))
    return np.frombuffer(read(t, r0 * cols * w, r1 * cols * w), dtype=d).reshape(r1 - r0, *t["shape"][1:])
