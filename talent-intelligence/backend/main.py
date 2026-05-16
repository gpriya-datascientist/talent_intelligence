"""
main.py — FastAPI application entry point.
Registers all routers, startup/shutdown events, middleware.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.db.database import init_db
from backend.routers import wishes, employees, availability


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown — add cleanup here if needed


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wishes.router)
app.include_router(employees.router)
app.include_router(availability.router)


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.ENV}
