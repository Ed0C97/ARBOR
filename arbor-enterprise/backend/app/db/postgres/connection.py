"""PostgreSQL dual-database connection management.

TWO separate databases:
1. magazine_h182 (Render) → READ-ONLY — brands, venues (managed by Flask)
2. arbor_db (local/cloud)  → READ-WRITE — arbor_enrichments, arbor_gold_standard, etc.

Both use async SQLAlchemy engines with independent connection pools.

TIER 2 - Point 6: Dynamic Connection Pool Tuning
- POOL_SIZE = 40 (Active connections held)
- MAX_OVERFLOW = 20 (Burst connections)
- POOL_RECYCLE = 1800 (Reset connection every 30m)
- POOL_PRE_PING = True (Check liveness before checkout)

TIER 7 - Point 30: CQRS Support
- Separate read/write session factories for future replica support
"""

import logging
import ssl
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for database connections."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE 1: magazine_h182 — READ-ONLY (brands, venues)
# ═══════════════════════════════════════════════════════════════════════════

_magazine_connect_args: dict = {}
if settings.database_ssl:
    _magazine_connect_args["ssl"] = _create_ssl_context()

# Read-only database: smaller pool as it's only for reads
magazine_engine = (
    create_async_engine(
        settings.database_url,  # magazine_h182 on Render
        echo=settings.app_debug and settings.app_env != "production",
        pool_size=settings.db_pool_size // 2,  # Half size for read-only
        max_overflow=settings.db_max_overflow // 2,
        pool_pre_ping=settings.db_pool_pre_ping,
        pool_recycle=settings.db_pool_recycle,
        pool_timeout=30,  # Wait max 30s for connection
        connect_args=_magazine_connect_args,
    )
    if settings.database_url
    else None
)

magazine_session_factory = (
    async_sessionmaker(
        magazine_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,  # Read-only, no need for autoflush
    )
    if magazine_engine
    else None
)


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE 2: arbor_db — READ-WRITE (enrichments, gold standard, feedback)
# TIER 2 - Point 6: Optimized connection pool
# ═══════════════════════════════════════════════════════════════════════════

_arbor_connect_args: dict = {}
if settings.arbor_database_ssl:
    _arbor_connect_args["ssl"] = _create_ssl_context()

# Read-write database: full pool size
arbor_engine = (
    create_async_engine(
        settings.arbor_database_url,  # Dedicated ARBOR database
        echo=settings.app_debug and settings.app_env != "production",
        pool_size=settings.db_pool_size,  # TIER 2: 40 connections
        max_overflow=settings.db_max_overflow,  # TIER 2: 20 burst
        pool_pre_ping=settings.db_pool_pre_ping,  # TIER 2: Check liveness
        pool_recycle=settings.db_pool_recycle,  # TIER 2: 30 min recycle
        pool_timeout=30,
        connect_args=_arbor_connect_args,
    )
    if settings.arbor_database_url
    else None
)

arbor_session_factory = (
    async_sessionmaker(
        arbor_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    if arbor_engine
    else None
)


# ═══════════════════════════════════════════════════════════════════════════
# CQRS Support (TIER 7 - Point 30)
# Separate read/write factories for future read replica support
# ═══════════════════════════════════════════════════════════════════════════

# For now, both point to same engine. In production with replicas:
# arbor_read_engine -> Replica (SELECT)
# arbor_write_engine -> Master (INSERT/UPDATE/DELETE)
arbor_read_session_factory = arbor_session_factory
arbor_write_session_factory = arbor_session_factory


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY — default "engine" points to magazine
# ═══════════════════════════════════════════════════════════════════════════

engine = magazine_engine
async_session_factory = magazine_session_factory


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Dependencies
# ═══════════════════════════════════════════════════════════════════════════


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session connected to magazine_h182 (read-only).

    Use this for API endpoints that read brands/venues.
    """
    if magazine_session_factory is None:
        raise RuntimeError("Magazine database not configured. Set DATABASE_URL.")

    async with magazine_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_arbor_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session connected to arbor_db (read-write).

    Use this for API endpoints that write enrichments, gold standard, feedback.
    """
    if arbor_session_factory is None:
        raise RuntimeError("ARBOR database not configured. Set ARBOR_DATABASE_URL.")

    async with arbor_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_read() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for read operations (CQRS pattern).

    In production with replicas, this would route to read replica.
    """
    if arbor_read_session_factory is None:
        raise RuntimeError("ARBOR database not configured.")

    async with arbor_read_session_factory() as session:
        try:
            yield session
        except Exception:
            raise


async def get_db_write() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for write operations (CQRS pattern).

    Routes to master database for INSERT/UPDATE/DELETE.
    """
    if arbor_write_session_factory is None:
        raise RuntimeError("ARBOR database not configured.")

    async with arbor_write_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ═══════════════════════════════════════════════════════════════════════════
# Connection Management
# ═══════════════════════════════════════════════════════════════════════════


async def create_arbor_tables():
    """Create all ARBOR-owned tables in the arbor_db.

    Call this once at startup or via a migration script.
    Only creates arbor_* tables, never touches magazine_h182.
    """
    if arbor_engine is None:
        logger.warning("ARBOR database not configured, skipping table creation")
        return

    from app.db.postgres.models import ArborBase

    async with arbor_engine.begin() as conn:
        await conn.run_sync(ArborBase.metadata.create_all)
    logger.info("ARBOR tables created/verified")


async def check_magazine_connection() -> bool:
    """Check if magazine database is reachable."""
    if magazine_engine is None:
        return False
    try:
        async with magazine_engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Magazine database connection failed: {e}")
        return False


async def check_arbor_connection() -> bool:
    """Check if ARBOR database is reachable."""
    if arbor_engine is None:
        return False
    try:
        async with arbor_engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"ARBOR database connection failed: {e}")
        return False


async def close_all_connections():
    """Close all database connections. Call during shutdown."""
    if magazine_engine:
        await magazine_engine.dispose()
        logger.info("Magazine database connections closed")
    if arbor_engine:
        await arbor_engine.dispose()
        logger.info("ARBOR database connections closed")


def get_pool_status() -> dict:
    """Get connection pool status for monitoring."""
    status = {}

    if magazine_engine and hasattr(magazine_engine.pool, "status"):
        status["magazine"] = {
            "pool_size": magazine_engine.pool.size(),
            "checked_in": magazine_engine.pool.checkedin(),
            "checked_out": magazine_engine.pool.checkedout(),
            "overflow": magazine_engine.pool.overflow(),
        }

    if arbor_engine and hasattr(arbor_engine.pool, "status"):
        status["arbor"] = {
            "pool_size": arbor_engine.pool.size(),
            "checked_in": arbor_engine.pool.checkedin(),
            "checked_out": arbor_engine.pool.checkedout(),
            "overflow": arbor_engine.pool.overflow(),
        }

    return status
