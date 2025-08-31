import math
import time
from fastapi import HTTPException, Request
from ..config import settings

def _fixed_window_key(prefix: str, ident: str, now: float) -> str:
    minute = math.floor(now / 60.0)
    return f"rl:{prefix}:{ident}:{minute}"

async def rate_limit_tenant(request: Request, tenant_id: str):
    if not tenant_id:
        return
    now = time.time()
    key = _fixed_window_key("tenant", tenant_id, now)
    limit = settings.rl_tenant_per_min
    await _hit_and_check(request, key, limit)

async def rate_limit_ip(request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    now = time.time()
    key = _fixed_window_key("ip", ip, now)
    limit = settings.rl_ip_per_min
    await _hit_and_check(request, key, limit)

async def _hit_and_check(request: Request, key: str, limit: int):
    # Redis client is created at startup and stored in app.state.redis
    r = request.app.state.redis
    # increment counter with 60s TTL
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    count, _ = await pipe.execute()
    if int(count) > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

