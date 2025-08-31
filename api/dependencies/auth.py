from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable
import uuid

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

from ..config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def create_access_token(data: dict, expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or settings.access_token_exp_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)

async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> dict:
    """
    Returns a dict: {id, tenant_id, email, role}
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        role = payload.get("role")
        if not user_id or not role:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="DB pool not ready")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id::text, tenant_id::text, email, role, is_active FROM users WHERE id = $1",
            uuid.UUID(user_id)
        )
    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="User inactive or not found")

    # tenant guard: platform_admin may have tenant_id NULL (platform-wide)
    if row["role"] != "platform_admin":
        if not tenant_id or row["tenant_id"] != tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch")

    return {"id": row["id"], "tenant_id": row["tenant_id"], "email": row["email"], "role": row["role"]}

def require_roles(allowed: Iterable[str]):
    allowed_set = set(allowed)
    async def _guard(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_set:
            raise HTTPException(status_code=403, detail="Forbidden (role)")
        return user
    return _guard

