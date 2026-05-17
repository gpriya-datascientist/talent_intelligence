"""
test_db.py — tests PostgreSQL connection using credentials from .env
Run: python test_db.py
"""
import asyncpg
import asyncio
import sys
import os
import re
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DATABASE_URL = os.getenv("DATABASE_URL", "")
match = re.match(r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", DATABASE_URL)
if not match:
    print("ERROR: Could not parse DATABASE_URL from .env")
    sys.exit(1)

DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME = match.groups()

async def test():
    try:
        conn = await asyncpg.connect(
            host=DB_HOST, port=int(DB_PORT),
            user=DB_USER, password=DB_PASS,
            database=DB_NAME,
        )
        version = await conn.fetchval("SELECT version()")
        print("Connected OK:", version[:60])
        await conn.close()
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e))

asyncio.run(test())
