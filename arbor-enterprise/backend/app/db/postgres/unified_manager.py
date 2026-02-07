"""Unified Entity Repository Manager.

This module provides a manager that coordinates access to all configured
entity types, combining the GenericEntityRepository with ArborEnrichment data.

It replaces the old UnifiedEntityRepository that was hardcoded for brands/venues.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import MetaData, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.postgres.dynamic_model import reflect_all_entity_tables
from app.db.postgres.generic_repository import GenericEntityRepository, UnifiedEntity
from app.db.postgres.models import ArborEnrichment

logger = logging.getLogger(__name__)


class UnifiedRepositoryManager:
    """Manager for accessing all configured entity types.

    Provides a unified interface for querying entities from any configured
    source table, automatically joining with ARBOR enrichment data.

    Usage:
        manager = UnifiedRepositoryManager(source_session, arbor_session)
        await manager.initialize()  # Reflects tables

        # Query any entity type
        entities, total = await manager.list_all(entity_type="product", limit=50)

        # Get by composite ID
        entity = await manager.get_by_composite_id("product_42")
    """

    def __init__(
        self,
        source_session: AsyncSession,
        arbor_session: AsyncSession,
        metadata: MetaData | None = None,
    ):
        self.source_session = source_session
        self.arbor_session = arbor_session
        self.metadata = metadata or MetaData()
        self.settings = get_settings()

        # Repositories per entity type (initialized on first use)
        self._repositories: dict[str, GenericEntityRepository] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize dynamic models for all configured entity types.

        Must be called before using the repository.
        """
        if self._initialized:
            return

        configs = self.settings.get_entity_type_configs()
        engine = self.source_session.get_bind()

        tables = await reflect_all_entity_tables(configs, engine, self.metadata)

        for config in configs:
            table = tables.get(config.entity_type)
            if table is not None:
                self._repositories[config.entity_type] = GenericEntityRepository(
                    self.source_session, config, table
                )
            else:
                logger.error(f"Failed to initialize repository for {config.entity_type}")

        self._initialized = True
        logger.info(f"Initialized {len(self._repositories)} entity repositories")

    def get_repository(self, entity_type: str) -> GenericEntityRepository | None:
        """Get the repository for a specific entity type."""
        return self._repositories.get(entity_type)

    async def get_by_composite_id(self, composite_id: str) -> UnifiedEntity | None:
        """Fetch a single entity by composite ID like 'product_42'.

        Automatically joins with ARBOR enrichment data if available.
        """
        parts = composite_id.split("_", 1)
        if len(parts) != 2:
            return None

        entity_type, raw_id = parts
        try:
            source_id = int(raw_id)
        except ValueError:
            return None

        repo = self.get_repository(entity_type)
        if repo is None:
            return None

        entity = await repo.get_by_id(source_id)
        if entity is None:
            return None

        # Fetch enrichment
        enrichment = await self._get_enrichment(entity_type, source_id)
        if enrichment:
            entity.vibe_dna = enrichment.vibe_dna
            entity.tags = enrichment.tags

        return entity

    async def list_all(
        self,
        entity_type: str | None = None,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        search: str | None = None,
        is_active: bool | None = True,
        is_featured: bool | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UnifiedEntity], int]:
        """List entities from all or specific entity types.

        If entity_type is None, queries all configured types and merges results.
        """
        if entity_type:
            # Single entity type
            repo = self.get_repository(entity_type)
            if repo is None:
                return [], 0

            entities, total = await repo.list_entities(
                category=category,
                city=city,
                country=country,
                search=search,
                is_active=is_active,
                is_featured=is_featured,
                offset=offset,
                limit=limit,
            )

            # Batch fetch enrichments
            await self._attach_enrichments(entities)
            return entities, total

        # All entity types
        all_entities: list[UnifiedEntity] = []
        total = 0

        for etype, repo in self._repositories.items():
            entities, count = await repo.list_entities(
                category=category,
                city=city,
                country=country,
                search=search,
                is_active=is_active,
                is_featured=is_featured,
                offset=0,  # Fetch all, paginate after merge
                limit=9999,
            )
            all_entities.extend(entities)
            total += count

        # Sort by priority (desc) then name
        all_entities.sort(key=lambda e: (-(e.priority or 0), e.name))

        # Apply pagination
        paginated = all_entities[offset : offset + limit]

        # Batch fetch enrichments
        await self._attach_enrichments(paginated)

        return paginated, total

    async def list_with_cursor(
        self,
        entity_type: str | None = None,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        cursor_created_at: str | None = None,
        cursor_id: int | None = None,
        limit: int = 50,
    ) -> tuple[list[UnifiedEntity], int]:
        """List entities using keyset (cursor-based) pagination.

        For a single entity_type, delegates to the repo's cursor method.
        For all types, queries each repo and merges results.
        """
        if entity_type:
            repo = self.get_repository(entity_type)
            if repo is None:
                return [], 0

            entities, total = await repo.list_with_cursor(
                category=category,
                city=city,
                country=country,
                cursor_created_at=cursor_created_at,
                cursor_id=cursor_id,
                limit=limit,
            )
            await self._attach_enrichments(entities)
            return entities, total

        # All entity types: query each, merge, sort, trim
        all_entities: list[UnifiedEntity] = []
        total = 0
        per_type_limit = max(limit, limit // max(len(self._repositories), 1))

        for etype, repo in self._repositories.items():
            entities, count = await repo.list_with_cursor(
                category=category,
                city=city,
                country=country,
                cursor_created_at=cursor_created_at,
                cursor_id=cursor_id,
                limit=per_type_limit,
            )
            all_entities.extend(entities)
            total += count

        # Sort by created_at desc, then source_id desc
        all_entities.sort(
            key=lambda e: (e.created_at or "", e.source_id),
            reverse=True,
        )
        trimmed = all_entities[:limit]
        await self._attach_enrichments(trimmed)
        return trimmed, total

    async def _get_enrichment(
        self, entity_type: str, source_id: int
    ) -> ArborEnrichment | None:
        """Fetch enrichment for a single entity."""
        result = await self.arbor_session.execute(
            select(ArborEnrichment).where(
                ArborEnrichment.entity_type == entity_type,
                ArborEnrichment.source_id == source_id,
            )
        )
        return result.scalar_one_or_none()

    async def _attach_enrichments(self, entities: list[UnifiedEntity]) -> None:
        """Batch fetch and attach enrichments to entities."""
        if not entities:
            return

        # Group by entity type for efficient queries
        by_type: dict[str, list[int]] = {}
        for entity in entities:
            by_type.setdefault(entity.entity_type, []).append(entity.source_id)

        # Fetch all enrichments in one query
        from sqlalchemy import or_

        conditions = [
            (ArborEnrichment.entity_type == etype)
            & (ArborEnrichment.source_id.in_(source_ids))
            for etype, source_ids in by_type.items()
        ]

        result = await self.arbor_session.execute(
            select(ArborEnrichment).where(or_(*conditions))
        )
        enrichments = {
            (e.entity_type, e.source_id): e for e in result.scalars().all()
        }

        # Attach to entities
        for entity in entities:
            enrichment = enrichments.get((entity.entity_type, entity.source_id))
            if enrichment:
                entity.vibe_dna = enrichment.vibe_dna
                entity.tags = enrichment.tags

    async def stats(self) -> dict[str, Any]:
        """Return aggregate statistics for all entity types."""
        stats = {
            "total_entities": 0,
            "by_type": {},
        }

        for entity_type, repo in self._repositories.items():
            count = await repo.count()
            stats["by_type"][entity_type] = count
            stats["total_entities"] += count

        # Enrichment count
        result = await self.arbor_session.execute(
            select(func.count()).select_from(ArborEnrichment)
        )
        stats["enriched_entities"] = result.scalar_one()

        return stats

    def get_available_entity_types(self) -> list[str]:
        """Return list of configured entity type names."""
        return list(self._repositories.keys())
