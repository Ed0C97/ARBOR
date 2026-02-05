"""FastAPI dependency injection."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.connection import get_db


async def get_session() -> AsyncSession:
    """Get database session dependency."""
    async for session in get_db():
        yield session
