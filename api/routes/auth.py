from typing import Dict, Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from ..dependencies.auth import create_access_token, verify_password, hash_password
from ..dependencies.auth import require_roles, get_current_user

router = APIRouter()

@router.post("/auth/token")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id::text, tenant_id::text, email, password_hash, role, is_active FROM users WHERE email = $1",
            form_data.username.lower().strip()
        )
    if not row or not row["is_active"] or not verify_password(form_data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    claims = {"sub": row["id"], "role": row["role"]}
    # only include tenant_id in token if user is not platform_admin
    if row["role"] != "platform_admin":
        claims["tenant_id"] = row["tenant_id"]

    token = create_access_token(claims)
    return {"access_token": token, "token_type": "bearer", "role": row["role"], "tenant_id": row["tenant_id"]}

@router.post("/auth/bootstrap")
async def bootstrap_platform_admin(request: Request, email: str, password: str) -> Dict[str, Any]:
    """
    One-time convenience endpoint: creates a platform_admin if none exist.
    Returns 409 if one already exists.
    """
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM users WHERE role='platform_admin' LIMIT 1;")
        if exists:
            raise HTTPException(status_code=409, detail="platform_admin already exists")
        uid = uuid.uuid4()
        await conn.execute(
            "INSERT INTO users (id, tenant_id, email, password_hash, role, is_active) VALUES ($1, NULL, $2, $3, 'platform_admin', TRUE)",
            uid, email.lower().strip(), hash_password(password)
        )
    return {"status": "created", "user_id": str(uid), "role": "platform_admin"}

