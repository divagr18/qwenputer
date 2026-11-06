"""OpenAI-compatible pass-through to vLLM that merges every system message into one at the start
(Qwen3.5's chat template rejects a system message anywhere else). Logs each request's shape."""
import json, time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

UP = "http://127.0.0.1:8000"
LOG = open("/opt/dlami/nvme/p0c/proxy.jsonl", "a")
app = FastAPI()
client = httpx.AsyncClient(timeout=None)


def text_of(c):
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict))


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    sys_parts = [text_of(m.get("content", "")) for m in msgs if m.get("role") in ("system", "developer")]
    rest = [m for m in msgs if m.get("role") not in ("system", "developer")]
    body["messages"] = ([{"role": "system", "content": "\n\n".join(sys_parts)}] if sys_parts else []) + rest
    LOG.write(json.dumps({"t": time.time(), "n_msgs": len(msgs), "n_system": len(sys_parts),
                          "roles": [m.get("role") for m in msgs],
                          "tools": [t.get("function", {}).get("name") for t in body.get("tools", [])],
                          "system_chars": sum(map(len, sys_parts)), "stream": body.get("stream"),
                          "keys": sorted(body)}) + "\n")
    LOG.flush()
    if body.get("stream"):
        r = client.build_request("POST", UP + "/v1/chat/completions", json=body)
        resp = await client.send(r, stream=True)

        async def gen():
            async for chunk in resp.aiter_raw():
                yield chunk
            await resp.aclose()
        return StreamingResponse(gen(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "text/event-stream"))
    resp = await client.post(UP + "/v1/chat/completions", json=body)
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def models():
    return (await client.get(UP + "/v1/models")).json()
