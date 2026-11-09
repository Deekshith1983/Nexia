"""PostgreSQL database connection management with SQLAlchemy and asyncpg."""

from collections.abc import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

import asyncio

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine_loop: asyncio.AbstractEventLoop | None = None


def get_engine() -> AsyncEngine:
    """Retrieve or lazily initialize the AsyncEngine matching the current event loop."""
    global _engine, _session_factory, _engine_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _engine is None or (_engine_loop is not None and current_loop is not None and _engine_loop is not current_loop):
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_recycle=settings.DB_POOL_RECYCLE,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_pre_ping=settings.DB_PRE_PING,
            connect_args={"timeout": settings.DB_CONNECT_TIMEOUT},
            echo=False,
        )
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        _engine_loop = current_loop
    return _engine


async def dispose_engine() -> None:
    """Dispose the database connection pool cleanly on shutdown."""
    global _engine, _session_factory, _engine_loop
    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception:
            pass
        finally:
            _engine = None
            _session_factory = None
            _engine_loop = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retrieve the session factory."""
    global _session_factory
    get_engine()
    assert _session_factory is not None
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency / generator providing an async database session."""
    session_maker = get_session_factory()
    async with session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def ping_database() -> bool:
    """Verify connectivity to PostgreSQL."""
    try:
        session_maker = get_session_factory()
        async with session_maker() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception:
        return False
