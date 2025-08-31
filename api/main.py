import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

from .config import settings
from .routes.iot import router as iot_router
from .routes.auth import router as auth_router
from .routes.tenants import router as tenants_router
from .middleware.correlation import correlation_middleware

# App logger
logger = logging.getLogger("pdm-v2")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDM Platform v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(correlation_middleware)

pool: Optional[asyncpg.pool.Pool] = None

async def init_db_pool() -> asyncpg.pool.Pool:
    global pool
    # asyncpg pool
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
    return pool

async def init_redis():
    # redis client
    r = redis.from_url(settings.redis_url, decode_responses=True)
    # quick check
    await r.ping()
    return r

@app.on_event("startup")
async def on_startup():
    global pool
    pool = await init_db_pool()
    app.state.pool = pool
    app.state.redis = await init_redis()
    app.logger = logger
    logger.info("API started: DB and Redis ready.")

@app.on_event("shutdown")
async def on_shutdown():
    global pool
    if pool:
        await pool.close()
    r = getattr(app.state, "redis", None)
    if r:
        await r.close()
    logger.info("API shutdown complete.")

@app.get("/health")
async def health() -> Dict[str, Any]:
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")
    # Redis ping
    await app.state.redis.ping()
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return "pdm_up 1\n"

# Routers
app.include_router(auth_router, prefix="")
app.include_router(tenants_router, prefix="")
app.include_router(iot_router, prefix="")
