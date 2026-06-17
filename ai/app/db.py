from collections.abc import AsyncIterator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

settings = get_settings()

# Async engine — used by the FastAPI request path (aiosqlite).
engine = create_async_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Sync engine — used by the training thread (see app/runner.py + training/).
# Same SQLite file, plain driver. WAL (below) lets the trainer write while the
# API reads concurrently.
sync_database_url = settings.database_url.replace("+aiosqlite", "")
sync_engine = create_engine(sync_database_url, future=True)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


def _apply_sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


event.listen(sync_engine, "connect", _apply_sqlite_pragmas)
event.listen(engine.sync_engine, "connect", _apply_sqlite_pragmas)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one async session per request."""
    async with SessionLocal() as session:
        yield session
