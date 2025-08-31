import uuid
from typing import Callable
from fastapi import Request, Response

HEADER = "X-Request-ID"

async def correlation_middleware(request: Request, call_next: Callable):
    rid = request.headers.get(HEADER) or str(uuid.uuid4())
    request.state.correlation_id = rid
    response: Response = await call_next(request)
    response.headers[HEADER] = rid
    return response

