import asyncio
import httpx
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_URL = "http://192.168.86.48:8000"

async def prewarm_task():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Poll vLLM until /v1/models returns HTTP 200
        for _ in range(60):
            try:
                r = await client.get(f"{VLLM_URL}/v1/models")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(10)
        else:
            print("[pre-warm] vLLM health check timed out, skipping pre-warm.")
            return

        # Pre-warm vLLM Automatic Prefix Caching (APC) with standard system prompt
        warmup_payload = {
            "model": "minimax-m3",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI coding assistant connected to a workspace. "
                        "Follow instructions precisely, format code cleanly in markdown, "
                        "preserve documentation integrity, and analyze logs before diagnosing failures."
                    )
                },
                {"role": "user", "content": "ping"}
            ],
            "max_tokens": 1
        }
        try:
            rw = await client.post(f"{VLLM_URL}/v1/chat/completions", json=warmup_payload)
            if rw.status_code == 200:
                print("[pre-warm] System prompt KV cache pre-warmed successfully (APC hit ready).")
            else:
                print(f"[pre-warm] Warmup request returned status {rw.status_code}")
        except Exception as e:
            print(f"[pre-warm] Error sending warmup request: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(prewarm_task())

async def stream_generator(path: str, method: str, headers: dict, content: bytes):
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            method,
            f"{VLLM_URL}{path}",
            headers=headers,
            content=content
        ) as r:
            async for chunk in r.aiter_lines():
                if chunk.startswith("data: "):
                    data_str = chunk[6:].strip()
                    if data_str == "[DONE]":
                        yield f"{chunk}\n\n"
                        continue
                    try:
                        # Translate the "reasoning" key to "reasoning_content" for client parser compatibility
                        updated_data = data_str.replace('"reasoning":', '"reasoning_content":')
                        yield f"data: {updated_data}\n\n"
                    except Exception:
                        yield f"{chunk}\n\n"
                else:
                    yield f"{chunk}\n"

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
    method = request.method
    body = await request.body()
    
    is_chat_completions = path == "v1/chat/completions"
    is_streaming = False
    if is_chat_completions and body:
        try:
            req_json = json.loads(body)
            is_streaming = req_json.get("stream", False)
        except Exception:
            pass
            
    if is_chat_completions and is_streaming:
        return StreamingResponse(
            stream_generator(f"/{path}", method, headers, body),
            media_type="text/event-stream"
        )
        
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.request(
            method,
            f"{VLLM_URL}/{path}",
            headers=headers,
            content=body
        )
        
        content = r.content
        if is_chat_completions and content:
            try:
                content = content.replace(b'"reasoning":', b'"reasoning_content":')
            except Exception:
                pass
                
        resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in ["content-encoding", "content-length", "transfer-encoding"]}
        return Response(
            content=content,
            status_code=r.status_code,
            headers=resp_headers
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

