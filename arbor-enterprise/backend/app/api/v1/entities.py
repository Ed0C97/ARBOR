"""Entity endpoints — reads from brands/venues in magazine_h182.

All data is READ-ONLY. Writes go through the Flask backend.
Enrichment data (vibe_dna) is served from arbor_enrichments.

TIER 1 - Point 3: PostgreSQL Keyset Pagination
- Uses cursor-based pagination for O(1) performance
- Replaces OFFSET which degrades at scale
"""

import base64
import json
import logging
from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.connection import get_arbor_db, get_db
from app.db.postgres.unified_manager import UnifiedRepositoryManager

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Cursor utilities (TIER 1 - Point 3)
# ---------------------------------------------------------------------------


def encode_cursor(created_at: datetime | str | None, entity_id: int) -> str:
    """Encode pagination cursor as base64 JSON.

    Cursor format: {"created_at": "ISO datetime", "id": entity_id}
    """
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    data = {"created_at": created_at, "id": entity_id}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> tuple[str | None, int | None]:
    """Decode base64 cursor to (created_at, id) tuple.

    Returns (None, None) if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return data.get("created_at"), data.get("id")
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class EntityResponse(BaseModel):
    """Unified response for both brands and venues.

    All fields except id/entity_type/source_id/name/slug are nullable
    because the magazine_h182 data may have incomplete records.
    """

    id: str  # Composite: "{entity_type}_{source_id}"
    entity_type: str  # Configured entity type
    source_id: int
    name: str
    slug: str | None = None
    category: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    maps_url: str | None = None
    website: str | None = None
    instagram: str | None = None
    email: str | None = None
    phone: str | None = None
    contact_person: str | None = None
    description: str | None = None
    specialty: str | None = None
    notes: str | None = None
    gender: str | None = None
    style: str | None = None
    rating: float | None = None
    price_range: str | None = None
    is_featured: bool = False
    is_active: bool = True
    priority: int | None = None
    verified: bool | None = None
    # ARBOR enrichment
    vibe_dna: dict | None = None
    tags: list | None = None
    # Audit
    created_at: str | None = None
    updated_at: str | None = None


class EntityListResponse(BaseModel):
    """Response for entity list endpoints."""

    items: list[EntityResponse]
    total: int
    offset: int
    limit: int


class CursorPaginatedResponse(BaseModel):
    """Response with cursor-based pagination.

    TIER 1 - Point 3: Keyset pagination response.
    """

    items: list[EntityResponse]
    total: int
    next_cursor: str | None = None
    prev_cursor: str | None = None
    has_more: bool = False


# ---------------------------------------------------------------------------
# Endpoints — Unified (brands + venues)
# ---------------------------------------------------------------------------


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    entity_type: str | None = Query(None, description="Filter by 'brand' or 'venue'"),
    category: str | None = None,
    city: str | None = None,
    country: str | None = None,
    gender: str | None = None,
    style: str | None = None,
    is_active: bool | None = Query(None, description="Filter by active status"),
    is_featured: bool | None = None,
    search: str | None = Query(None, description="Full-text search across name, description"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """List entities (brands + venues) with optional filters."""
    manager = UnifiedRepositoryManager(session, arbor_session)
    await manager.initialize()
    entities, total = await manager.list_all(
        entity_type=entity_type,
        category=category,
        city=city,
        country=country,
        gender=gender,
        style=style,
        is_active=is_active,
        is_featured=is_featured,
        search=search,
        offset=offset,
        limit=limit,
    )
    return EntityListResponse(
        items=[EntityResponse(**asdict(e)) for e in entities],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Get a single entity by composite ID (e.g. 'brand_42' or 'venue_17')."""
    manager = UnifiedRepositoryManager(session, arbor_session)
    await manager.initialize()
    entity = await manager.get_by_composite_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return EntityResponse(**asdict(entity))


# ---------------------------------------------------------------------------
# Legacy endpoints — backward-compatible shortcuts for default schema
# ---------------------------------------------------------------------------


@router.get("/brands", response_model=EntityListResponse)
async def list_brands(
    category: str | None = None,
    country: str | None = None,
    gender: str | None = None,
    style: str | None = None,
    is_active: bool | None = Query(None),
    is_featured: bool | None = None,
    search: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """List brands only."""
    manager = UnifiedRepositoryManager(session, arbor_session)
    await manager.initialize()
    entities, total = await manager.list_all(
        entity_type="brand",
        category=category,
        country=country,
        gender=gender,
        style=style,
        is_active=is_active,
        is_featured=is_featured,
        search=search,
        offset=offset,
        limit=limit,
    )
    return EntityListResponse(
        items=[EntityResponse(**asdict(e)) for e in entities],
        total=total,
        offset=offset,
        limit=limit,
    )


# (Venues shortcut)


@router.get("/venues", response_model=EntityListResponse)
async def list_venues(
    category: str | None = None,
    city: str | None = None,
    country: str | None = None,
    gender: str | None = None,
    style: str | None = None,
    is_active: bool | None = Query(None),
    is_featured: bool | None = None,
    search: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """List venues only."""
    manager = UnifiedRepositoryManager(session, arbor_session)
    await manager.initialize()
    entities, total = await manager.list_all(
        entity_type="venue",
        category=category,
        city=city,
        country=country,
        gender=gender,
        style=style,
        is_active=is_active,
        is_featured=is_featured,
        search=search,
        offset=offset,
        limit=limit,
    )
    return EntityListResponse(
        items=[EntityResponse(**asdict(e)) for e in entities],
        total=total,
        offset=offset,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# TIER 1 - Point 3: Cursor-based pagination endpoint
# ---------------------------------------------------------------------------


@router.get("/entities/cursor", response_model=CursorPaginatedResponse)
async def list_entities_cursor(
    entity_type: str | None = Query(None, description="Filter by 'brand' or 'venue'"),
    category: str | None = None,
    city: str | None = None,
    country: str | None = None,
    cursor: str | None = Query(None, description="Base64-encoded cursor for keyset pagination"),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """List entities with cursor-based (keyset) pagination.

    TIER 1 - Point 3: PostgreSQL Keyset Pagination

    This endpoint provides O(1) pagination performance regardless of offset,
    unlike traditional OFFSET-based pagination which degrades linearly.

    Usage:
        1. First request: GET /entities/cursor?limit=20
        2. Next page: GET /entities/cursor?cursor=<next_cursor>&limit=20
        3. Repeat until has_more=false

    The cursor encodes (created_at, id) to ensure stable pagination
    even when new items are added.
    """
    manager = UnifiedRepositoryManager(session, arbor_session)
    await manager.initialize()

    # Decode cursor if provided
    cursor_created_at, cursor_id = None, None
    if cursor:
        cursor_created_at, cursor_id = decode_cursor(cursor)
        if cursor_created_at is None:
            raise HTTPException(status_code=400, detail="Invalid cursor format")

    # Get entities using keyset pagination
    entities, total = await manager.list_with_cursor(
        entity_type=entity_type,
        category=category,
        city=city,
        country=country,
        cursor_created_at=cursor_created_at,
        cursor_id=cursor_id,
        limit=limit + 1,  # Fetch one extra to check has_more
    )

    # Check if there are more results
    has_more = len(entities) > limit
    if has_more:
        entities = entities[:limit]

    # Generate next cursor from last item
    next_cursor = None
    if has_more and entities:
        last = entities[-1]
        next_cursor = encode_cursor(last.created_at, last.source_id)

    items = [EntityResponse(**asdict(e)) for e in entities]

    return CursorPaginatedResponse(
        items=items,
        total=total,
        next_cursor=next_cursor,
        has_more=has_more,
    )
