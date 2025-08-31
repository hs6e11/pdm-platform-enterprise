#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://postgres:password@localhost:5433/pdm_v2")
SCHEMA_FILE = os.getenv("SCHEMA_FILE", "database/schemas/001_init.sql")

def load_sql(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    # Strip UTF-8 BOM if present
    text = text.lstrip("\ufeff")
    # Drop full-line comments and blank lines to avoid empty statements
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

async def main():
    sql = load_sql(SCHEMA_FILE)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Let Postgres handle multiple statements in one call
        await conn.execute(sql)
        print("Schema created/verified successfully.")
    finally:
        await conn.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[setup_database] ERROR: {e.__class__.__name__}: {e}", file=sys.stderr)
        raise

