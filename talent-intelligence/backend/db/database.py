from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
import asyncio
import sys

from backend.config import settings

# Fix Windows asyncpg event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if settings.ENV == "development":
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        poolclass=NullPool,
    )
else:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables on startup."""
    from backend.models import employee, skill, availability, wish  # noqa
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all tables — tests only."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
