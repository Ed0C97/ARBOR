"""Generic Entity Repository for Schema-Agnostic Data Access.

This module provides a generic repository that works with any table schema
as defined by EntityTypeConfig. It replaces the hardcoded BrandRepository
and VenueRepository with a single, configurable implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Table, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EntityTypeConfig

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED ENTITY - Generic data structure for all entity types
# =============================================================================


@dataclass
class UnifiedEntity:
    """A generic entity that can represent any data source.

    This is the common format returned to the API layer, regardless of
    the underlying table schema. Fields are populated based on the
    column mappings in EntityTypeConfig.
    """

    # Identity
    id: str  # Composite: "product_42", "store_17"
    entity_type: str  # From config: "product", "store", etc.
    source_id: int  # Raw ID from source table

    # Core fields (always present if mapped)
    name: str
    slug: str | None = None
    category: str | None = None

    # Location
    city: str | None = None
    region: str | None = None
    country: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    maps_url: str | None = None

    # Contact
    website: str | None = None
    instagram: str | None = None
    email: str | None = None
    phone: str | None = None
    contact_person: str | None = None

    # Content
    description: str | None = None
    specialty: str | None = None
    notes: str | None = None

    # Attributes
    gender: str | None = None
    style: str | None = None
    rating: float | None = None
    price_range: str | None = None

    # Flags
    is_featured: bool = False
    is_active: bool = True
    priority: int | None = None
    verified: bool | None = None

    # ARBOR enrichment (populated from arbor_enrichments table)
    vibe_dna: dict | None = None
    tags: list | None = None

    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None

    # Dynamic extra fields (for columns not in standard mapping)
    extra: dict = field(default_factory=dict)


def _safe_float(val: str | float | int | None) -> float | None:
    """Convert a value to float safely."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_bool(val: str | bool | None) -> bool | None:
    """Convert a value to bool safely."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1", "si", "y")
    return bool(val)


def _safe_datetime_str(val: datetime | str | None) -> str | None:
    """Convert datetime to ISO string safely."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


# =============================================================================
# GENERIC ENTITY REPOSITORY
# =============================================================================


class GenericEntityRepository:
    """Generic repository that works with any configured entity type.

    Replaces hardcoded BrandRepository/VenueRepository with a single
    implementation that reads configuration from EntityTypeConfig.

    Usage:
        config = settings.get_entity_config("product")
        repo = GenericEntityRepository(session, config, table)
        entities = await repo.list_entities(limit=50)
    """

    def __init__(
        self,
        session: AsyncSession,
        config: EntityTypeConfig,
        table: Table,
    ):
        self.session = session
        self.config = config
        self.table = table

    async def get_by_id(self, entity_id: int) -> UnifiedEntity | None:
        """Fetch a single entity by its source ID."""
        id_col = self.table.c[self.config.id_column]
        query = select(self.table).where(id_col == entity_id)

        result = await self.session.execute(query)
        row = result.first()

        if row is None:
            return None

        return self._row_to_unified(row)

    async def get_by_slug(self, slug: str) -> UnifiedEntity | None:
        """Fetch a single entity by slug (if slug column is mapped)."""
        slug_col_name = self.config.all_mappings.get("slug")
        if not slug_col_name or slug_col_name not in self.table.c:
            return None

        slug_col = self.table.c[slug_col_name]
        query = select(self.table).where(slug_col == slug)

        result = await self.session.execute(query)
        row = result.first()

        if row is None:
            return None

        return self._row_to_unified(row)

    async def list_entities(
        self,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        search: str | None = None,
        is_active: bool | None = True,
        is_featured: bool | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UnifiedEntity], int]:
        """List entities with optional filters.

        Filters are applied dynamically based on available column mappings.
        """
        query = select(self.table)
        count_query = select(func.count()).select_from(self.table)

        filters = []

        # Apply filters only if the column is mapped
        if category:
            col_name = self.config.all_mappings.get("category")
            if col_name and col_name in self.table.c:
                filters.append(self.table.c[col_name] == category)

        if city:
            col_name = self.config.all_mappings.get("city")
            if col_name and col_name in self.table.c:
                filters.append(self.table.c[col_name].ilike(f"%{city}%"))

        if country:
            col_name = self.config.all_mappings.get("country")
            if col_name and col_name in self.table.c:
                filters.append(self.table.c[col_name] == country)

        if is_active is not None and self.config.active_filter_column:
            col_name = self.config.active_filter_column
            if col_name in self.table.c:
                filters.append(self.table.c[col_name] == is_active)

        if is_featured is not None:
            col_name = self.config.all_mappings.get("is_featured")
            if col_name and col_name in self.table.c:
                filters.append(self.table.c[col_name] == is_featured)

        if search:
            # Search across name and description
            search_conditions = []
            for field_name in ["name", "description", "specialty", "notes"]:
                col_name = self.config.all_mappings.get(field_name)
                if col_name and col_name in self.table.c:
                    search_conditions.append(
                        self.table.c[col_name].ilike(f"%{search}%")
                    )
            if search_conditions:
                filters.append(or_(*search_conditions))

        # Apply all filters
        for f in filters:
            query = query.where(f)
            count_query = count_query.where(f)

        # Ordering
        priority_col = self.config.all_mappings.get("priority")
        name_col = self.config.all_mappings.get("name", "name")
        if priority_col and priority_col in self.table.c:
            query = query.order_by(
                self.table.c[priority_col].desc().nullslast(),
                self.table.c[name_col] if name_col in self.table.c else self.table.c[self.config.id_column],
            )
        elif name_col in self.table.c:
            query = query.order_by(self.table.c[name_col])

        # Pagination
        query = query.offset(offset).limit(limit)

        # Execute
        result = await self.session.execute(query)
        rows = result.all()

        count_result = await self.session.execute(count_query)
        total = count_result.scalar_one()

        entities = [self._row_to_unified(row) for row in rows]

        return entities, total

    async def list_with_cursor(
        self,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        cursor_created_at: str | None = None,
        cursor_id: int | None = None,
        limit: int = 50,
    ) -> tuple[list[UnifiedEntity], int]:
        """List entities using keyset (cursor-based) pagination.

        Uses ``WHERE (created_at, id) < (cursor, cursor_id)``
        for O(1) performance regardless of page number.
        """
        from datetime import datetime as _dt

        query = select(self.table)
        count_query = select(func.count()).select_from(self.table)

        filters: list = []
        count_filters: list = []

        if category:
            col_name = self.config.all_mappings.get("category")
            if col_name and col_name in self.table.c:
                f = self.table.c[col_name] == category
                filters.append(f)
                count_filters.append(f)

        if city:
            col_name = self.config.all_mappings.get("city")
            if col_name and col_name in self.table.c:
                f = self.table.c[col_name].ilike(f"%{city}%")
                filters.append(f)
                count_filters.append(f)

        if country:
            col_name = self.config.all_mappings.get("country")
            if col_name and col_name in self.table.c:
                f = self.table.c[col_name] == country
                filters.append(f)
                count_filters.append(f)

        # Keyset cursor filter
        created_at_col = self.config.all_mappings.get("created_at")
        id_col_name = self.config.id_column
        if (
            cursor_created_at
            and cursor_id is not None
            and created_at_col
            and created_at_col in self.table.c
        ):
            try:
                cursor_dt = _dt.fromisoformat(cursor_created_at)
            except ValueError:
                cursor_dt = None
            if cursor_dt is not None:
                ca_col = self.table.c[created_at_col]
                pk_col = self.table.c[id_col_name]
                filters.append(
                    or_(
                        ca_col < cursor_dt,
                        (ca_col == cursor_dt) & (pk_col < cursor_id),
                    )
                )

        for f in filters:
            query = query.where(f)
        for f in count_filters:
            count_query = count_query.where(f)

        # Order by created_at DESC, id DESC for stable cursor pagination
        if created_at_col and created_at_col in self.table.c:
            query = query.order_by(
                self.table.c[created_at_col].desc(),
                self.table.c[id_col_name].desc(),
            )
        else:
            query = query.order_by(self.table.c[id_col_name].desc())

        query = query.limit(limit)

        result = await self.session.execute(query)
        rows = result.all()

        count_result = await self.session.execute(count_query)
        total = count_result.scalar_one()

        entities = [self._row_to_unified(row) for row in rows]
        return entities, total

    async def count(self) -> int:
        """Count total entities."""
        query = select(func.count()).select_from(self.table)
        result = await self.session.execute(query)
        return result.scalar_one()

    def _row_to_unified(self, row: Any) -> UnifiedEntity:
        """Convert a database row to UnifiedEntity using config mappings."""
        # Get the row data as a dict-like object
        if hasattr(row, "_mapping"):
            row_data = row._mapping
        elif hasattr(row, "_asdict"):
            row_data = row._asdict()
        else:
            row_data = row

        def get_val(arbor_field: str, default: Any = None) -> Any:
            col_name = self.config.all_mappings.get(arbor_field)
            if col_name is None:
                return default
            return row_data.get(col_name, default)

        # Get source ID
        source_id = row_data.get(self.config.id_column, 0)

        return UnifiedEntity(
            id=f"{self.config.entity_type}_{source_id}",
            entity_type=self.config.entity_type,
            source_id=source_id,
            name=get_val("name", "Unknown"),
            slug=get_val("slug"),
            category=get_val("category"),
            city=get_val("city"),
            region=get_val("region"),
            country=get_val("country"),
            address=get_val("address"),
            latitude=_safe_float(get_val("latitude")),
            longitude=_safe_float(get_val("longitude")),
            maps_url=get_val("maps_url"),
            website=get_val("website"),
            instagram=get_val("instagram"),
            email=get_val("email"),
            phone=get_val("phone"),
            contact_person=get_val("contact_person"),
            description=get_val("description"),
            specialty=get_val("specialty"),
            notes=get_val("notes"),
            gender=get_val("gender"),
            style=get_val("style"),
            rating=_safe_float(get_val("rating")),
            price_range=get_val("price_range"),
            is_featured=_safe_bool(get_val("is_featured")) or False,
            is_active=_safe_bool(get_val("is_active")) if get_val("is_active") is not None else True,
            priority=get_val("priority"),
            verified=_safe_bool(get_val("verified")),
            created_at=_safe_datetime_str(get_val("created_at")),
            updated_at=_safe_datetime_str(get_val("updated_at")),
        )

    def get_text_for_embedding(self, entity: UnifiedEntity) -> str:
        """Build text content for embedding generation.

        Uses the text_fields_for_embedding from config to determine
        which fields to concatenate.
        """
        parts = []
        for field_name in self.config.text_fields_for_embedding:
            value = getattr(entity, field_name, None) or entity.extra.get(field_name)
            if value:
                parts.append(str(value))
        return " ".join(parts)
