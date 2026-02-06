"""Dynamic ORM Model Factory for Schema-Agnostic Data Access.

This module creates SQLAlchemy models at runtime based on configuration,
enabling ARBOR to work with ANY table schema without code changes.

The factory uses SQLAlchemy reflection to discover columns or explicit
mapping from EntityTypeConfig to create type-safe models.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import registry

from app.config import EntityTypeConfig

logger = logging.getLogger(__name__)

# Registry for dynamic models (one per entity type)
_dynamic_models: dict[str, Table] = {}
_mapper_registry = registry()


def get_dynamic_model(entity_type: str) -> Table | None:
    """Get a previously created dynamic model by entity type."""
    return _dynamic_models.get(entity_type)


async def create_dynamic_model(
    config: EntityTypeConfig,
    engine: AsyncEngine,
    metadata: MetaData,
    use_reflection: bool = True,
) -> Table:
    """Create a SQLAlchemy Table model from EntityTypeConfig.

    This function creates a Table object that maps to the source database
    table as defined in the config. It can either:
    1. Use reflection to auto-discover columns from the database
    2. Use explicit mapping from config.all_mappings

    Args:
        config: EntityTypeConfig defining the entity type
        engine: AsyncEngine connected to the source database
        metadata: SQLAlchemy MetaData instance
        use_reflection: If True, discover columns via reflection

    Returns:
        SQLAlchemy Table object ready for queries
    """
    if config.entity_type in _dynamic_models:
        return _dynamic_models[config.entity_type]

    if use_reflection:
        # Use reflection to discover table structure
        try:
            async with engine.connect() as conn:
                # Reflect the table from the database
                await conn.run_sync(
                    lambda sync_conn: metadata.reflect(
                        bind=sync_conn,
                        only=[config.table_name],
                        extend_existing=True,
                    )
                )

            table = metadata.tables.get(config.table_name)
            if table is not None:
                _dynamic_models[config.entity_type] = table
                logger.info(
                    f"Reflected table '{config.table_name}' for entity type '{config.entity_type}' "
                    f"with {len(table.columns)} columns"
                )
                return table
            else:
                logger.warning(
                    f"Table '{config.table_name}' not found via reflection, "
                    "falling back to explicit mapping"
                )
        except Exception as e:
            logger.warning(f"Reflection failed for {config.table_name}: {e}")

    # Fall back to explicit column definition from config
    columns = [
        Column(config.id_column, Integer, primary_key=True),
    ]

    # Add columns from mappings
    for arbor_field, source_column in config.all_mappings.items():
        if source_column and source_column != config.id_column:
            # Infer column type from field name
            col_type = _infer_column_type(arbor_field, source_column)
            columns.append(Column(source_column, col_type))

    table = Table(
        config.table_name,
        metadata,
        *columns,
        extend_existing=True,
    )

    _dynamic_models[config.entity_type] = table
    logger.info(
        f"Created explicit model for '{config.table_name}' "
        f"with {len(columns)} columns"
    )

    return table


def _infer_column_type(arbor_field: str, source_column: str) -> Any:
    """Infer SQLAlchemy column type from field name.

    This uses naming conventions to guess appropriate types.
    """
    field_lower = arbor_field.lower()
    column_lower = source_column.lower()

    # Boolean fields
    if any(
        x in field_lower
        for x in ["is_", "has_", "active", "featured", "verified", "enabled"]
    ):
        return Boolean

    # Float/Numeric fields
    if any(x in field_lower for x in ["latitude", "longitude", "lat", "lng", "rating", "price", "score"]):
        return Float

    # Integer fields
    if any(x in field_lower for x in ["priority", "count", "id", "created_by"]):
        return Integer

    # DateTime fields
    if any(x in column_lower for x in ["_at", "date", "time", "created", "updated"]):
        return DateTime

    # Text fields (long content)
    if any(x in field_lower for x in ["description", "notes", "content", "body", "text"]):
        return Text

    # Default to String
    return String


async def reflect_all_entity_tables(
    configs: list[EntityTypeConfig],
    engine: AsyncEngine,
    metadata: MetaData,
) -> dict[str, Table]:
    """Create dynamic models for all configured entity types.

    Args:
        configs: List of EntityTypeConfig objects
        engine: AsyncEngine connected to source database
        metadata: SQLAlchemy MetaData instance

    Returns:
        Dict mapping entity_type -> Table
    """
    tables = {}
    for config in configs:
        try:
            table = await create_dynamic_model(config, engine, metadata)
            tables[config.entity_type] = table
        except Exception as e:
            logger.error(f"Failed to create model for {config.entity_type}: {e}")

    return tables


def clear_dynamic_models() -> None:
    """Clear all cached dynamic models.

    Useful for testing or when reconfiguring schemas.
    """
    _dynamic_models.clear()
    logger.info("Cleared all dynamic models")


def get_column_value(row: Any, config: EntityTypeConfig, arbor_field: str) -> Any:
    """Get a value from a row using the column mapping.

    Args:
        row: SQLAlchemy row result
        config: EntityTypeConfig with mappings
        arbor_field: The ARBOR field name to get

    Returns:
        The value from the row, or None if not mapped
    """
    source_column = config.all_mappings.get(arbor_field)
    if source_column is None:
        return None

    # Handle both dict-like and attribute access
    if hasattr(row, "_mapping"):
        return row._mapping.get(source_column)
    elif hasattr(row, source_column):
        return getattr(row, source_column)
    elif isinstance(row, dict):
        return row.get(source_column)

    return None


def build_text_for_embedding(row: Any, config: EntityTypeConfig) -> str:
    """Build text content for embedding generation.

    Concatenates the configured text fields from the row.

    Args:
        row: SQLAlchemy row result
        config: EntityTypeConfig with text_fields_for_embedding

    Returns:
        Concatenated text string for embedding
    """
    parts = []
    for field in config.text_fields_for_embedding:
        # Get column name from mappings or use field directly
        source_column = config.all_mappings.get(field, field)

        if hasattr(row, "_mapping"):
            value = row._mapping.get(source_column)
        elif hasattr(row, source_column):
            value = getattr(row, source_column)
        elif isinstance(row, dict):
            value = row.get(source_column)
        else:
            value = None

        if value:
            parts.append(str(value))

    return " ".join(parts)
