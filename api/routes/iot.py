from datetime import datetime, timezone
from typing import Dict, Any
import json

from fastapi import APIRouter, HTTPException, Request, Depends
from ..dependencies.auth import require_roles
from ..dependencies.ratelimit import rate_limit_tenant, rate_limit_ip
from ..observability.audit import audit_log

router = APIRouter()

# operator, client_admin, platform_admin can ingest
RequireOperator = require_roles({"operator", "client_admin", "platform_admin"})

@router.post("/api/iot/data")
async def ingest_sensor_data(
    request: Request,
    user: dict = Depends(RequireOperator),
) -> Dict[str, Any]:
    # Rate limits (fixed window)
    await rate_limit_ip(request)
    tenant_id = user.get("tenant_id")
    if tenant_id is None:
        raise HTTPException(status_code=400, detail="platform_admin must specify a tenant")
    await rate_limit_tenant(request, tenant_id)

    # Validate payload
    body = await request.json()
    machine_id = body.get("machine_id")
    if not machine_id or not isinstance(machine_id, str):
        raise HTTPException(status_code=400, detail="machine_id required")
    ts_str = body.get("timestamp")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z","+00:00"))
        except Exception:
            raise HTTPException(status_code=400, detail="timestamp must be ISO8601")
    else:
        ts = datetime.now(timezone.utc)

    # Insert
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sensor_readings (tenant_id, machine_id, timestamp, payload) VALUES ($1,$2,$3,$4)",
            tenant_id, machine_id, ts, json.dumps(body)
        )

    # Audit
    await audit_log(
        request,
        event="iot_ingest",
        user=user,
        details={"machine_id": machine_id, "ts": ts.isoformat()}
    )

    return {"status": "stored", "tenant": tenant_id, "machine_id": machine_id, "ts": ts.isoformat()}

