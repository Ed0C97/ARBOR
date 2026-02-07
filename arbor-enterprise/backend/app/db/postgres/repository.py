"""Repository layer for A.R.B.O.R. Enterprise.

Manages arbor_enrichments, arbor_feedback (read-write on ARBOR-owned tables).

Source entity access (any schema) is handled by:
- GenericEntityRepository (single entity type)
- UnifiedRepositoryManager (all entity types, with enrichment join)
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import ArborEnrichment, ArborFeedback

# ── Re-exports for backwards compatibility ──────────────────────────────
# Callers that still do ``from app.db.postgres.repository import UnifiedEntity``
# or ``UnifiedEntityRepository`` will get the new schema-agnostic versions.
from app.db.postgres.generic_repository import UnifiedEntity  # noqa: F401
from app.db.postgres.unified_manager import (  # noqa: F401
    UnifiedRepositoryManager as UnifiedEntityRepository,
)


# ==========================================================================
# Enrichment Repository (read-write — ARBOR-owned)
# ==========================================================================


class EnrichmentRepository:
    """CRUD for arbor_enrichments table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, entity_type: str, source_id: int) -> ArborEnrichment | None:
        result = await self.session.execute(
            select(ArborEnrichment).where(
                ArborEnrichment.entity_type == entity_type,
                ArborEnrichment.source_id == source_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_batch(
        self, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], ArborEnrichment]:
        """Fetch enrichments for multiple entities at once."""
        if not keys:
            return {}

        # Group by entity_type for efficient IN queries
        by_type: dict[str, list[int]] = {}
        for entity_type, source_id in keys:
            by_type.setdefault(entity_type, []).append(source_id)

        # Build optimized query using IN operator
        conditions = [
            (ArborEnrichment.entity_type == entity_type)
            & (ArborEnrichment.source_id.in_(source_ids))
            for entity_type, source_ids in by_type.items()
        ]

        result = await self.session.execute(select(ArborEnrichment).where(or_(*conditions)))
        enrichments = result.scalars().all()
        return {(e.entity_type, e.source_id): e for e in enrichments}

    async def upsert(self, entity_type: str, source_id: int, **kwargs: Any) -> ArborEnrichment:
        existing = await self.get(entity_type, source_id)
        if existing:
            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.session.flush()
            return existing
        else:
            enrichment = ArborEnrichment(
                entity_type=entity_type,
                source_id=source_id,
                **kwargs,
            )
            self.session.add(enrichment)
            await self.session.flush()
            return enrichment

    async def delete(self, entity_type: str, source_id: int) -> bool:
        result = await self.session.execute(
            delete(ArborEnrichment).where(
                ArborEnrichment.entity_type == entity_type,
                ArborEnrichment.source_id == source_id,
            )
        )
        return result.rowcount > 0


# ==========================================================================
# Feedback Repository (read-write — ARBOR-owned)
# ==========================================================================


class FeedbackRepository:
    """CRUD operations for user feedback."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs: Any) -> ArborFeedback:
        feedback = ArborFeedback(**kwargs)
        self.session.add(feedback)
        await self.session.flush()
        return feedback

    async def get_by_user(self, user_id: uuid.UUID, limit: int = 100) -> list[ArborFeedback]:
        result = await self.session.execute(
            select(ArborFeedback)
            .where(ArborFeedback.user_id == user_id)
            .order_by(ArborFeedback.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
