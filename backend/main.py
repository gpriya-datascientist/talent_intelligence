"""
main.py — FastAPI application entry point.
FIXED: LoggingMiddleware added, Langfuse initialised on startup.
"""
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.db.database import init_db
from backend.middleware import LoggingMiddleware
from backend.observability import get_langfuse
from backend.routers import wishes, employees, availability


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    get_langfuse()   # initialise Langfuse client once at startup
    yield
    # Shutdown — flush Langfuse traces
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

app.include_router(wishes.router)
app.include_router(employees.router)
app.include_router(availability.router)


@app.get("/health")
async def health():
    lf = get_langfuse()
    return {
        "status":    "ok",
        "env":       settings.ENV,
        "langfuse":  "connected" if lf else "not configured",
        "n8n":       "configured" if settings.N8N_WEBHOOK_URL != "http://localhost:5678" else "not configured",
    }
