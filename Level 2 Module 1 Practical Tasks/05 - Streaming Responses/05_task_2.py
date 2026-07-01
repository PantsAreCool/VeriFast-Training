# Task 2: Implement a Streaming Proxy Server
# Build a FastAPI server that acts as a streaming proxy between a frontend and multiple LLM providers. 
# The server should: (1) accept POST requests with messages and a provider selection, (2) stream responses via SSE, 
# (3) accumulate the full response and log it, (4) handle connection drops gracefully (client disconnects mid-stream), 
# (5) support multiple concurrent connections, and (6) provide a /stats endpoint showing server-wide streaming statistics.


import asyncio
import json
import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Streaming Proxy Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from dotenv import load_dotenv

load_dotenv()
api_key=os.environ.get("OPENROUTER_API_KEY")
openai_client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

SERVER_STATS = {
    "total_requests": 0,
    "active_connections": 0,
    "total_tokens_streamed": 0,
    "completed_streams": 0,
    "disconnected_streams": 0
}

@app.post("/api/proxy/stream")
async def proxy_stream(request: Request):
    """
    Accepts JSON: {"messages": [...], "provider": "openai"}
    Streams the response back to the client via SSE.
    """
    body = await request.json()
    messages = body.get("messages", [])
    provider = body.get("provider", "openai")

    SERVER_STATS["total_requests"] += 1
    SERVER_STATS["active_connections"] += 1

    async def sse_generator():
        accumulated_text = ""
        tokens_in_this_stream = 0
        stream_completed_successfully = False
        
        try:
            if provider == "openai":
                stream = await openai_client.chat.completions.create(
                    model="mistralai/mistral-large",
                    messages=messages,
                    stream=True
                )

                async for chunk in stream:
                    if await request.is_disconnected():
                        logger.warning("disconnected prematurely mid-stream.")
                        SERVER_STATS["disconnected_streams"] += 1
                        return

                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        accumulated_text += token
                        tokens_in_this_stream += 1
                        SERVER_STATS["total_tokens_streamed"] += 1

                        yield f"data: {json.dumps({'token': token})}\n\n"

            else:
                yield f"data: {json.dumps({'error': f'Provider {provider} not implemented'})}\n\n"
                return

            yield "data: [DONE]\n\n"
            stream_completed_successfully = True
            SERVER_STATS["completed_streams"] += 1

        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            yield f"data: {json.dumps({'error': 'Internal Stream Error'})}\n\n"
        
        finally:
            SERVER_STATS["active_connections"] -= 1
            status_msg = "SUCCESS" if stream_completed_successfully else "CLIENT_DISCONNECT/ERROR"
            
            logger.info(
                f"Session Log:\n"
                f"Status: {status_msg}\n"
                f"Tokens Streamed: {tokens_in_this_stream}\n"
                f"Full Accumulated Content: {accumulated_text[:100]}...\n"
                f"-" * 40
            )

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/stats")
async def get_stats():
    """Returns server-wide streaming diagnostics metrics."""
    return SERVER_STATS