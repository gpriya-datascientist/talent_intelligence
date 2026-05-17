"""
create_db.py — creates the talent_db database.
Password is loaded from .env — never hardcoded.
Run: python create_db.py
"""
import asyncpg
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DATABASE_URL = os.getenv("DATABASE_URL", "")
# Parse password from DATABASE_URL
# Format: postgresql+asyncpg://user:password@host:port/dbname
import re
match = re.match(r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", DATABASE_URL)
if not match:
    print("ERROR: Could not parse DATABASE_URL from .env")
    sys.exit(1)

DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME = match.groups()

async def create_db():
    conn = await asyncpg.connect(
        host=DB_HOST, port=int(DB_PORT),
        user=DB_USER, password=DB_PASS,
        database="postgres",
    )
    exists = await conn.fetchval(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
    if exists:
        print(f"{DB_NAME} already exists")
    else:
        await conn.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"{DB_NAME} created successfully")
    await conn.close()

asyncio.run(create_db())
