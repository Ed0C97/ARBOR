"""Metadata Agent - structured queries via PostgreSQL."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.repository import UnifiedEntityRepository

logger = logging.getLogger(__name__)


class MetadataAgent:
    """Query structured metadata from PostgreSQL."""

    def __init__(self, session: AsyncSession | None = None, arbor_session: AsyncSession | None = None):
        self._session = session
        self._arbor_session = arbor_session

    async def execute(
        self,
        filters: dict,
        limit: int = 10,
    ) -> list[dict]:
        """Query entities by structured metadata filters."""
        if not self._session:
            return []

        repo = UnifiedEntityRepository(self._session, self._arbor_session)

        entities, total = await repo.list_all(
            category=filters.get("category"),
            city=filters.get("city"),
            is_active=filters.get("is_active", True),
            offset=0,
            limit=limit,
        )

        results = []
        for entity in entities:
            results.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "category": entity.category,
                    "city": entity.city,
                    "price_range": entity.price_range,
                    "is_active": entity.is_active,
                    "vibe_dna": entity.vibe_dna,
                    "description": entity.description,
                    "source": "metadata",
                }
            )

        logger.info(f"Metadata search returned {len(results)} results (total: {total})")
        return results
