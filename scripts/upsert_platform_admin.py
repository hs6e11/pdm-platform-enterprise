#!/usr/bin/env python3
import os, asyncio, uuid
import asyncpg
from passlib.context import CryptContext

DB = os.getenv("DATABASE_URL", "postgres://postgres:password@localhost:5433/pdm_v2")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def upsert_admin(email: str, new_password: str):
    email = email.lower().strip()
    conn = await asyncpg.connect(DB)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE role='platform_admin' AND lower(email)=lower($1)",
            email
        )
        hashed = pwd.hash(new_password)
        if row:
            await conn.execute(
                "UPDATE users SET password_hash=$1, is_active=TRUE WHERE id=$2",
                hashed, row["id"]
            )
            print("updated:", row["id"])
        else:
            # ensure there isn't another platform_admin with a different email
            exists = await conn.fetchval("SELECT 1 FROM users WHERE role='platform_admin' LIMIT 1;")
            if exists:
                raise SystemExit("A platform_admin already exists with a different email. Use that email or delete it first.")
            uid = uuid.uuid4()
            await conn.execute(
                "INSERT INTO users (id, tenant_id, email, password_hash, role, is_active) "
                "VALUES ($1, NULL, $2, $3, 'platform_admin', TRUE)",
                uid, email, hashed
            )
            print("created:", uid)
    finally:
        await conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: upsert_platform_admin.py <email> <new_password>")
        raise SystemExit(1)
    asyncio.run(upsert_admin(sys.argv[1], sys.argv[2]))

