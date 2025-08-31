#!/usr/bin/env python3
import os, asyncio
import asyncpg
from passlib.context import CryptContext

DB = os.getenv("DATABASE_URL", "postgres://postgres:password@localhost:5433/pdm_v2")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def main(email: str, new_password: str):
    conn = await asyncpg.connect(DB)
    try:
        hashed = pwd.hash(new_password)
        rows = await conn.execute(
            "UPDATE users SET password_hash=$1 WHERE role='platform_admin' AND lower(email)=lower($2)",
            hashed, email
        )
        print("Updated:", rows)
    finally:
        await conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: reset_admin_password.py <email> <new_password>")
        raise SystemExit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))

