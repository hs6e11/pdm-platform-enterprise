#!/usr/bin/env python3
import os, uuid, asyncio, sys
import asyncpg
from passlib.context import CryptContext

DB = os.getenv("DATABASE_URL", "postgres://postgres:password@localhost:5433/pdm_v2")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

USAGE = "Usage: create_user.py <email> <password> <role> [tenant_id]\nRoles: platform_admin | client_admin | operator | viewer"

async def main():
    if len(sys.argv) < 4:
        print(USAGE); sys.exit(1)
    email = sys.argv[1].lower().strip()
    password = sys.argv[2]
    role = sys.argv[3]
    tenant_id = sys.argv[4] if len(sys.argv) > 4 else None
    if role != "platform_admin" and not tenant_id:
        print("tenant_id is required for non-platform_admin users"); sys.exit(1)

    conn = await asyncpg.connect(DB)
    try:
        uid = uuid.uuid4()
        await conn.execute(
            "INSERT INTO users (id, tenant_id, email, password_hash, role, is_active) "
            "VALUES ($1,$2,$3,$4,$5,TRUE)",
            uid, tenant_id, email, pwd.hash(password), role
        )
        print("created:", uid)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
