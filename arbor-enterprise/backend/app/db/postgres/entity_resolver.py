"""Shared helper to resolve entity fields from source tables.

Replaces all hardcoded Brand/Venue lookups with schema-agnostic resolution
using GenericEntityRepository and dynamic models.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import EntityTypeConfig, get_settings
from app.db.postgres.dynamic_model import create_dynamic_model, get_dynamic_model
from app.db.postgres.generic_repository import GenericEntityRepository

logger = logging.getLogger(__name__)


async def _get_table(
    config: EntityTypeConfig,
    engine: AsyncEngine,
    metadata: MetaData | None = None,
):
    """Get or create the SQLAlchemy Table for an entity type config."""
    table = get_dynamic_model(config.entity_type)
    if table is not None:
        return table
    _metadata = metadata or MetaData()
    return await create_dynamic_model(config, engine, _metadata)


async def resolve_entity_fields(
    session: AsyncSession,
    entity_type: str,
    source_id: int,
    fields: list[str] | None = None,
    engine: AsyncEngine | None = None,
    metadata: MetaData | None = None,
) -> dict[str, Any]:
    """Resolve field values for an entity from the source table.

    This is the schema-agnostic replacement for all hardcoded
    Brand/Venue lookup patterns.

    Args:
        session: AsyncSession connected to the source database.
        entity_type: Entity type string (e.g. "brand", "venue", "product").
        source_id: Primary key value in the source table.
        fields: ARBOR field names to resolve (default: name, category, city).
        engine: Optional engine; defaults to session's bind.
        metadata: Optional MetaData; creates new if None.

    Returns:
        Dict mapping field names to values. Empty dict if entity not found.
    """
    if fields is None:
        fields = ["name", "category", "city"]

    settings = get_settings()
    config = settings.get_entity_config(entity_type)
    if config is None:
        logger.warning("No config for entity type: %s", entity_type)
        return {}

    _engine = engine or session.get_bind()
    table = await _get_table(config, _engine, metadata)
    repo = GenericEntityRepository(session, config, table)
    entity = await repo.get_by_id(source_id)
    if entity is None:
        return {}

    return {f: getattr(entity, f, None) for f in fields}


async def resolve_entity_fields_batch(
    session: AsyncSession,
    entity_type: str,
    source_ids: list[int],
    fields: list[str] | None = None,
    engine: AsyncEngine | None = None,
    metadata: MetaData | None = None,
) -> dict[int, dict[str, Any]]:
    """Batch-resolve field values for multiple entities of the same type.

    Uses a single SQL ``SELECT ... WHERE id IN (...)`` query for efficiency.

    Args:
        session: AsyncSession connected to the source database.
        entity_type: Entity type string.
        source_ids: List of primary key values.
        fields: ARBOR field names to resolve (default: name, category, city).
        engine: Optional engine; defaults to session's bind.
        metadata: Optional MetaData; creates new if None.

    Returns:
        Dict mapping source_id → {field: value}.
    """
    if fields is None:
        fields = ["name", "category", "city"]

    if not source_ids:
        return {}

    settings = get_settings()
    config = settings.get_entity_config(entity_type)
    if config is None:
        logger.warning("No config for entity type: %s", entity_type)
        return {}

    _engine = engine or session.get_bind()
    table = await _get_table(config, _engine, metadata)

    # Build a single SELECT ... WHERE id IN (...) query
    id_col = table.c[config.id_column]
    query = select(table).where(id_col.in_(source_ids))
    result = await session.execute(query)
    rows = result.all()

    repo = GenericEntityRepository(session, config, table)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        entity = repo._row_to_unified(row)
        out[entity.source_id] = {f: getattr(entity, f, None) for f in fields}

    return out
