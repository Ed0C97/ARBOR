"""Repository layer for A.R.B.O.R. Enterprise.

Reads brands/venues from magazine_h182 (read-only).
Manages arbor_enrichments, arbor_users, arbor_feedback (read-write).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import (
    ArborEnrichment,
    ArborFeedback,
    Brand,
    Venue,
)

# ---------------------------------------------------------------------------
# Unified entity dict — the common shape returned to the API
# ---------------------------------------------------------------------------


@dataclass
class UnifiedEntity:
    """A brand or venue mapped to a common shape for the API."""

    id: str  # "brand_42" or "venue_17"
    entity_type: str  # "brand" | "venue"
    source_id: int
    name: str
    slug: str
    category: str | None
    city: str | None
    region: str | None
    country: str | None
    address: str | None
    latitude: float | None
    longitude: float | None
    maps_url: str | None
    website: str | None
    instagram: str | None
    email: str | None
    phone: str | None
    contact_person: str | None
    description: str | None
    specialty: str | None
    notes: str | None
    gender: str | None
    style: str | None
    rating: float | None
    price_range: str | None
    is_featured: bool
    is_active: bool
    priority: int | None
    verified: bool | None
    # ARBOR enrichment (may be None)
    vibe_dna: dict | None
    tags: list | None
    # Audit
    created_at: str | None
    updated_at: str | None


def _safe_float(val: str | float | None) -> float | None:
    """Convert a string or numeric value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_bool(val: str | bool | None) -> bool | None:
    """Convert a string or bool value to bool, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return val.lower() in ("true", "yes", "1", "si", "y") if isinstance(val, str) else None


def _brand_to_unified(brand: Brand, enrichment: ArborEnrichment | None = None) -> UnifiedEntity:
    return UnifiedEntity(
        id=f"brand_{brand.id}",
        entity_type="brand",
        source_id=brand.id,
        name=brand.name,
        slug=brand.slug,
        category=brand.category,
        city=brand.city or brand.area or brand.neighborhood,
        region=brand.region,
        country=brand.country,
        address=brand.address,
        latitude=brand.latitude,
        longitude=brand.longitude,
        maps_url=brand.maps_url,
        website=brand.website,
        instagram=brand.instagram,
        email=brand.email,
        phone=brand.phone,
        contact_person=brand.contact_person,
        description=brand.description,
        specialty=brand.specialty,
        notes=brand.notes,
        gender=brand.gender,
        style=brand.style,
        rating=_safe_float(brand.rating),  # varchar in DB → convert to float
        price_range=None,
        is_featured=brand.is_featured or False,
        is_active=brand.is_active if brand.is_active is not None else True,
        priority=brand.priority,
        verified=_safe_bool(brand.visited),  # varchar in DB → convert to bool
        vibe_dna=enrichment.vibe_dna if enrichment else None,
        tags=enrichment.tags if enrichment else None,
        created_at=brand.created_at.isoformat() if brand.created_at else None,
        updated_at=brand.updated_at.isoformat() if brand.updated_at else None,
    )


def _venue_to_unified(venue: Venue, enrichment: ArborEnrichment | None = None) -> UnifiedEntity:
    return UnifiedEntity(
        id=f"venue_{venue.id}",
        entity_type="venue",
        source_id=venue.id,
        name=venue.name,
        slug=venue.slug,
        category=venue.category,
        city=venue.city,
        region=venue.region,
        country=venue.country,
        address=venue.address,
        latitude=venue.latitude,
        longitude=venue.longitude,
        maps_url=venue.maps_url,
        website=venue.website,
        instagram=venue.instagram,
        email=venue.email,
        phone=venue.phone,
        contact_person=venue.contact_person,
        description=venue.description,
        specialty=None,
        notes=venue.notes,
        gender=venue.gender,
        style=venue.style,
        rating=_safe_float(venue.rating),  # varchar in DB → convert to float
        price_range=venue.price_range,
        is_featured=venue.is_featured or False,
        is_active=venue.is_active if venue.is_active is not None else True,
        priority=venue.priority,
        verified=_safe_bool(venue.verified),  # varchar in DB → convert to bool
        vibe_dna=enrichment.vibe_dna if enrichment else None,
        tags=enrichment.tags if enrichment else None,
        created_at=venue.created_at.isoformat() if venue.created_at else None,
        updated_at=venue.updated_at.isoformat() if venue.updated_at else None,
    )


# ==========================================================================
# Brand Repository (read-only)
# ==========================================================================


class BrandRepository:
    """Read-only access to the brands table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, brand_id: int) -> Brand | None:
        result = await self.session.execute(select(Brand).where(Brand.id == brand_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Brand | None:
        result = await self.session.execute(select(Brand).where(Brand.slug == slug))
        return result.scalar_one_or_none()

    async def list_brands(
        self,
        category: str | None = None,
        country: str | None = None,
        gender: str | None = None,
        style: str | None = None,
        is_active: bool | None = True,
        is_featured: bool | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Brand], int]:
        query = select(Brand)
        count_q = select(func.count()).select_from(Brand)

        filters = []
        if category:
            filters.append(Brand.category == category)
        if country:
            filters.append(Brand.country == country)
        if gender:
            filters.append(Brand.gender == gender)
        if style:
            filters.append(Brand.style.ilike(f"%{style}%"))
        if is_active is not None:
            filters.append(Brand.is_active == is_active)
        if is_featured is not None:
            filters.append(Brand.is_featured == is_featured)
        if search:
            filters.append(
                or_(
                    Brand.name.ilike(f"%{search}%"),
                    Brand.description.ilike(f"%{search}%"),
                    Brand.specialty.ilike(f"%{search}%"),
                )
            )

        for f in filters:
            query = query.where(f)
            count_q = count_q.where(f)

        query = (
            query.order_by(Brand.priority.desc().nullslast(), Brand.name)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(query)
        brands = list(result.scalars().all())

        count_result = await self.session.execute(count_q)
        total = count_result.scalar_one()

        return brands, total

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Brand))
        return result.scalar_one()


# ==========================================================================
# Venue Repository (read-only)
# ==========================================================================


class VenueRepository:
    """Read-only access to the venues table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, venue_id: int) -> Venue | None:
        result = await self.session.execute(select(Venue).where(Venue.id == venue_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Venue | None:
        result = await self.session.execute(select(Venue).where(Venue.slug == slug))
        return result.scalar_one_or_none()

    async def list_venues(
        self,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        gender: str | None = None,
        style: str | None = None,
        is_active: bool | None = True,
        is_featured: bool | None = None,
        verified: bool | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Venue], int]:
        query = select(Venue)
        count_q = select(func.count()).select_from(Venue)

        filters = []
        if category:
            filters.append(Venue.category == category)
        if city:
            filters.append(Venue.city.ilike(f"%{city}%"))
        if country:
            filters.append(Venue.country == country)
        if gender:
            filters.append(Venue.gender == gender)
        if style:
            filters.append(Venue.style.ilike(f"%{style}%"))
        if is_active is not None:
            filters.append(Venue.is_active == is_active)
        if is_featured is not None:
            filters.append(Venue.is_featured == is_featured)
        if verified is not None:
            filters.append(Venue.verified == verified)
        if search:
            filters.append(
                or_(
                    Venue.name.ilike(f"%{search}%"),
                    Venue.description.ilike(f"%{search}%"),
                    Venue.city.ilike(f"%{search}%"),
                )
            )

        for f in filters:
            query = query.where(f)
            count_q = count_q.where(f)

        query = (
            query.order_by(Venue.priority.desc().nullslast(), Venue.name)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(query)
        venues = list(result.scalars().all())

        count_result = await self.session.execute(count_q)
        total = count_result.scalar_one()

        return venues, total

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Venue))
        return result.scalar_one()


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
# Unified Entity Repository — merges brands + venues with enrichments
# ==========================================================================


class UnifiedEntityRepository:
    """Queries across both brands and venues, joining with enrichment data."""

    def __init__(self, session: AsyncSession, arbor_session: AsyncSession | None = None):
        self.session = session
        self.arbor_session = arbor_session or session
        self.brands = BrandRepository(session)
        self.venues = VenueRepository(session)
        self.enrichments = EnrichmentRepository(self.arbor_session)

    async def get_by_composite_id(self, composite_id: str) -> UnifiedEntity | None:
        """Fetch a single entity by composite ID like 'brand_42' or 'venue_17'."""
        parts = composite_id.split("_", 1)
        if len(parts) != 2:
            return None

        entity_type, raw_id = parts
        try:
            source_id = int(raw_id)
        except ValueError:
            return None

        enrichment = await self.enrichments.get(entity_type, source_id)

        if entity_type == "brand":
            brand = await self.brands.get_by_id(source_id)
            return _brand_to_unified(brand, enrichment) if brand else None
        elif entity_type == "venue":
            venue = await self.venues.get_by_id(source_id)
            return _venue_to_unified(venue, enrichment) if venue else None

        return None

    async def list_all(
        self,
        entity_type: str | None = None,
        category: str | None = None,
        city: str | None = None,
        country: str | None = None,
        gender: str | None = None,
        style: str | None = None,
        is_active: bool | None = True,
        is_featured: bool | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UnifiedEntity], int]:
        """List entities from both brands and venues, merged and paginated."""
        results: list[UnifiedEntity] = []
        total = 0

        # --- Brands ---
        if entity_type is None or entity_type == "brand":
            brands, brands_total = await self.brands.list_brands(
                category=category,
                country=country,
                gender=gender,
                style=style,
                is_active=is_active,
                is_featured=is_featured,
                search=search,
                offset=0,
                limit=9999,  # We paginate after merge
            )
            total += brands_total

            # Batch-fetch enrichments
            brand_keys = [("brand", b.id) for b in brands]
            brand_enrichments = await self.enrichments.get_batch(brand_keys)

            for b in brands:
                enr = brand_enrichments.get(("brand", b.id))
                results.append(_brand_to_unified(b, enr))

        # --- Venues ---
        if entity_type is None or entity_type == "venue":
            venues, venues_total = await self.venues.list_venues(
                category=category,
                city=city,
                country=country,
                gender=gender,
                style=style,
                is_active=is_active,
                is_featured=is_featured,
                search=search,
                offset=0,
                limit=9999,
            )
            total += venues_total

            venue_keys = [("venue", v.id) for v in venues]
            venue_enrichments = await self.enrichments.get_batch(venue_keys)

            for v in venues:
                enr = venue_enrichments.get(("venue", v.id))
                results.append(_venue_to_unified(v, enr))

        # Sort by priority (desc, nulls last), then name
        results.sort(key=lambda e: (-(e.priority or 0), e.name))

        # Apply pagination
        paginated = results[offset : offset + limit]

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

        TIER 1 - Point 3: PostgreSQL Keyset Pagination

        Instead of OFFSET N (which scans N rows), uses:
        WHERE (created_at, id) < (cursor_created_at, cursor_id)
        ORDER BY created_at DESC, id DESC
        LIMIT N

        This provides O(1) performance regardless of page number.

        Args:
            entity_type: Filter by 'brand' or 'venue'
            category: Category filter
            city: City filter
            country: Country filter
            cursor_created_at: ISO datetime from previous page's last item
            cursor_id: ID from previous page's last item
            limit: Number of items to fetch

        Returns:
            Tuple of (entities, total_count)
        """
        from datetime import datetime

        results: list[UnifiedEntity] = []
        total = 0

        # Parse cursor timestamp
        cursor_dt = None
        if cursor_created_at:
            try:
                cursor_dt = datetime.fromisoformat(cursor_created_at)
            except ValueError:
                pass

        # --- Brands with keyset pagination ---
        if entity_type is None or entity_type == "brand":
            query = select(Brand)
            count_q = select(func.count()).select_from(Brand)

            filters = []
            if category:
                filters.append(Brand.category == category)
            if country:
                filters.append(Brand.country == country)

            # Keyset pagination filter
            if cursor_dt is not None and cursor_id is not None:
                # Seek method: (created_at, id) < (cursor_created_at, cursor_id)
                filters.append(
                    or_(
                        Brand.created_at < cursor_dt,
                        (Brand.created_at == cursor_dt) & (Brand.id < cursor_id),
                    )
                )

            for f in filters:
                query = query.where(f)
                if f != filters[-1] or cursor_dt is None:  # Don't count cursor filter
                    count_q = count_q.where(f)

            # Order by created_at DESC, id DESC for stable pagination
            query = query.order_by(Brand.created_at.desc(), Brand.id.desc()).limit(
                limit if entity_type == "brand" else limit // 2
            )

            result = await self.session.execute(query)
            brands = list(result.scalars().all())

            count_result = await self.session.execute(count_q)
            total += count_result.scalar_one()

            brand_keys = [("brand", b.id) for b in brands]
            brand_enrichments = await self.enrichments.get_batch(brand_keys)

            for b in brands:
                enr = brand_enrichments.get(("brand", b.id))
                results.append(_brand_to_unified(b, enr))

        # --- Venues with keyset pagination ---
        if entity_type is None or entity_type == "venue":
            query = select(Venue)
            count_q = select(func.count()).select_from(Venue)

            filters = []
            if category:
                filters.append(Venue.category == category)
            if city:
                filters.append(Venue.city.ilike(f"%{city}%"))
            if country:
                filters.append(Venue.country == country)

            # Keyset pagination filter
            if cursor_dt is not None and cursor_id is not None:
                filters.append(
                    or_(
                        Venue.created_at < cursor_dt,
                        (Venue.created_at == cursor_dt) & (Venue.id < cursor_id),
                    )
                )

            for f in filters:
                query = query.where(f)
                if f != filters[-1] or cursor_dt is None:
                    count_q = count_q.where(f)

            query = query.order_by(Venue.created_at.desc(), Venue.id.desc()).limit(
                limit if entity_type == "venue" else limit // 2
            )

            result = await self.session.execute(query)
            venues = list(result.scalars().all())

            count_result = await self.session.execute(count_q)
            total += count_result.scalar_one()

            venue_keys = [("venue", v.id) for v in venues]
            venue_enrichments = await self.enrichments.get_batch(venue_keys)

            for v in venues:
                enr = venue_enrichments.get(("venue", v.id))
                results.append(_venue_to_unified(v, enr))

        # Sort combined results by created_at desc
        results.sort(
            key=lambda e: (e.created_at or "", -e.source_id),
            reverse=True,
        )

        # Trim to limit
        results = results[:limit]

        return results, total

    async def stats(self) -> dict:
        """Return aggregate statistics."""
        brands_count = await self.brands.count()
        venues_count = await self.venues.count()

        enrichment_count_result = await self.enrichments.session.execute(
            select(func.count()).select_from(ArborEnrichment)
        )
        enriched = enrichment_count_result.scalar_one()

        return {
            "total_entities": brands_count + venues_count,
            "total_brands": brands_count,
            "total_venues": venues_count,
            "enriched_entities": enriched,
        }


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
