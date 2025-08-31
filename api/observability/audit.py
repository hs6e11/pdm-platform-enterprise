import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Request

async def audit_log(
    request: Request,
    event: str,
    user: Optional[Dict[str, Any]],
    details: Dict[str, Any] | None = None,
):
    """
    Writes a row to audit_logs and also logs structured JSON.
    """
    rid = getattr(request.state, "correlation_id", None)
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else None)

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user_id": user.get("id") if user else None,
        "tenant_id": user.get("tenant_id") if user else None,
        "role": user.get("role") if user else None,
        "client_ip": client_ip,
        "method": request.method,
        "path": request.url.path,
        "rid": rid,
        "details": details or {},
    }

    # DB write
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO audit_logs (event, user_id, tenant_id, role, client_ip, method, path, rid, details) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
            payload["event"],
            payload["user_id"],
            payload["tenant_id"],
            payload["role"],
            payload["client_ip"],
            payload["method"],
            payload["path"],
            payload["rid"],
            json.dumps(payload["details"]),
        )

    # App log (structured)
    request.app.logger.info(json.dumps(payload))

