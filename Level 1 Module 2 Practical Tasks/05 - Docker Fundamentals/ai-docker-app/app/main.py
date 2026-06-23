"""
app/main.py - FastAPI AI Application designed to run inside Docker.
Demonstrates Redis caching, health checks, and container-aware configuration.
"""

import json
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Redis connection (uses Docker network hostname "redis")
redis_client = redis.Redis(
    host="redis",  # Docker Compose service name
    port=6379,
    db=0,
    decode_responses=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: test Redis connection
    try:
        redis_client.ping()
        print("Connected to Redis successfully!")
    except redis.ConnectionError:
        print("WARNING: Could not connect to Redis")
    yield
    # Shutdown
    redis_client.close()
    print("Application shutting down.")


app = FastAPI(
    title="AI Docker App",
    description="A FastAPI AI application running in Docker with Redis caching.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Request/Response Models ----

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)


class PromptResponse(BaseModel):
    id: str
    prompt: str
    response: str
    model: str
    cached: bool
    latency_ms: float


# ---- Endpoints ----

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck and load balancers."""
    redis_ok = False
    try:
        redis_ok = redis_client.ping()
    except redis.ConnectionError:
        pass
    return {
        "status": "healthy",
        "redis": "connected" if redis_ok else "disconnected",
    }


@app.post("/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    """Generate an AI response with Redis caching."""
    import hashlib
    cache_key = hashlib.md5(
        f"{request.prompt}:{request.model}:{request.temperature}".encode()
    ).hexdigest()

    # Check cache first
    cached_result = redis_client.get(f"prompt:{cache_key}")
    if cached_result:
        data = json.loads(cached_result)
        data["cached"] = True
        return PromptResponse(**data)

    # Simulate AI generation (replace with actual API call)
    start_time = time.time()
    time.sleep(0.5)  # Simulate latency
    response_text = f"[Simulated] Response to: {request.prompt[:50]}..."
    latency = (time.time() - start_time) * 1000

    result = {
        "id": cache_key[:8],
        "prompt": request.prompt,
        "response": response_text,
        "model": request.model,
        "cached": False,
        "latency_ms": round(latency, 2),
    }

    # Cache the result with a 1-hour TTL
    redis_client.setex(
        f"prompt:{cache_key}",
        3600,
        json.dumps(result),
    )

    return PromptResponse(**result)


@app.get("/cache/stats")
async def cache_stats():
    """Get Redis cache statistics."""
    info = redis_client.info("stats")
    return {
        "total_keys": redis_client.dbsize(),
        "hits": info.get("keyspace_hits", 0),
        "misses": info.get("keyspace_misses", 0),
        "hit_rate": round(
            info.get("keyspace_hits", 0)
            / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            * 100, 2
        ),
    }


@app.delete("/cache/flush")
async def flush_cache():
    """Clear the entire cache."""
    redis_client.flushdb()
    return {"message": "Cache flushed successfully"}