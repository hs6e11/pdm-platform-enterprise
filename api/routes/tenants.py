# api/routes/tenants.py
from typing import List, Dict, Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from ..dependencies.auth import require_roles

router = APIRouter()
RequirePlatformAdmin = require_roles({"platform_admin"})

@router.post("/tenants")
async def create_tenant(
    request: Request,
    name: str = Query(..., min_length=2, max_length=200),
    _admin=Depends(RequirePlatformAdmin),
) -> Dict[str, Any]:
    pool = request.app.state.pool
    tid = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO tenants (id, name) VALUES ($1, $2)", tid, name.strip())
    return {"tenant_id": str(tid), "name": name.strip()}

@router.get("/tenants")
async def list_tenants(
    request: Request,
    _admin=Depends(RequirePlatformAdmin),
) -> List[Dict[str, Any]]:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id::text AS id, name, created_at FROM tenants ORDER BY created_at DESC")
    return [{"tenant_id": r["id"], "name": r["name"], "created_at": r["created_at"].isoformat()} for r in rows]

