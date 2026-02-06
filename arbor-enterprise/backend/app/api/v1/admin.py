"""Admin endpoints — stats, enrichment, health checks, and metrics.

TIER 5 - Point 22: Deep Health Checks
TIER 5 - Point 21: Prometheus Metrics Endpoint
"""

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.postgres.connection import get_arbor_db, get_db
from app.db.postgres.repository import EnrichmentRepository, UnifiedEntityRepository

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class StatsResponse(BaseModel):
    total_entities: int = 0
    total_brands: int = 0
    total_venues: int = 0
    enriched_entities: int = 0


class EnrichmentRequest(BaseModel):
    vibe_dna: dict | None = None
    tags: list[str] | None = None


class EnrichmentResponse(BaseModel):
    entity_id: str
    entity_type: str
    source_id: int
    vibe_dna: dict | None = None
    tags: list | None = None
    message: str = "Enrichment saved"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Get real-time statistics from brands + venues tables."""
    repo = UnifiedEntityRepository(session, arbor_session)
    stats = await repo.stats()
    return StatsResponse(**stats)


@router.post("/enrich/{entity_id}", response_model=EnrichmentResponse)
async def enrich_entity(
    entity_id: str,
    body: EnrichmentRequest,
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Add or update ARBOR enrichment (vibe_dna, tags) for an entity.

    entity_id format: 'brand_42' or 'venue_17'
    """
    parts = entity_id.split("_", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400, detail="Invalid entity_id format. Use 'brand_42' or 'venue_17'"
        )

    entity_type, raw_id = parts
    if entity_type not in ("brand", "venue"):
        raise HTTPException(status_code=400, detail="entity_type must be 'brand' or 'venue'")

    try:
        source_id = int(raw_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="source_id must be an integer")

    # Verify the source entity exists (read from Magazine DB)
    unified_repo = UnifiedEntityRepository(session, arbor_session)
    entity = await unified_repo.get_by_composite_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    # Upsert enrichment (write to Arbor DB)
    enrichment_repo = EnrichmentRepository(arbor_session)
    kwargs = {}
    if body.vibe_dna is not None:
        kwargs["vibe_dna"] = body.vibe_dna
    if body.tags is not None:
        kwargs["tags"] = body.tags

    enrichment = await enrichment_repo.upsert(entity_type, source_id, **kwargs)

    return EnrichmentResponse(
        entity_id=entity_id,
        entity_type=entity_type,
        source_id=source_id,
        vibe_dna=enrichment.vibe_dna,
        tags=enrichment.tags,
    )


@router.delete("/enrich/{entity_id}")
async def delete_enrichment(
    entity_id: str,
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Remove ARBOR enrichment for an entity."""
    parts = entity_id.split("_", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid entity_id format")

    entity_type, raw_id = parts
    try:
        source_id = int(raw_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="source_id must be an integer")

    enrichment_repo = EnrichmentRepository(arbor_session)
    deleted = await enrichment_repo.delete(entity_type, source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Enrichment not found")

    return {"detail": "Enrichment deleted"}


# ---------------------------------------------------------------------------
# TIER 5 - Point 22: Deep Health Checks
# ---------------------------------------------------------------------------


class ServiceHealth(BaseModel):
    """Health status for a single service."""

    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


class DeepHealthResponse(BaseModel):
    """Deep health check response with all service statuses."""

    status: str  # "healthy", "unhealthy", "degraded"
    version: str
    environment: str
    checks: list[ServiceHealth]
    total_latency_ms: float


async def _check_postgres_magazine() -> ServiceHealth:
    """Check Magazine PostgreSQL (read-only) connectivity."""
    from app.db.postgres.connection import magazine_engine

    start = time.perf_counter()
    try:
        if magazine_engine is None:
            return ServiceHealth(
                name="postgres_magazine",
                status="unhealthy",
                error="Not configured",
            )
        async with magazine_engine.connect() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="postgres_magazine",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="postgres_magazine",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


async def _check_postgres_arbor() -> ServiceHealth:
    """Check ARBOR PostgreSQL (read-write) connectivity."""
    from app.db.postgres.connection import arbor_engine

    start = time.perf_counter()
    try:
        if arbor_engine is None:
            return ServiceHealth(
                name="postgres_arbor",
                status="unhealthy",
                error="Not configured",
            )
        async with arbor_engine.connect() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="postgres_arbor",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="postgres_arbor",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


async def _check_qdrant() -> ServiceHealth:
    """Check Qdrant vector database connectivity."""
    from app.db.qdrant.client import get_async_qdrant_client

    start = time.perf_counter()
    try:
        client = await get_async_qdrant_client()
        if client is None:
            return ServiceHealth(
                name="qdrant",
                status="unhealthy",
                error="Not configured",
            )
        collections = await client.get_collections()
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="qdrant",
            status="healthy",
            latency_ms=round(latency, 2),
            details={"collections": [c.name for c in collections.collections]},
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="qdrant",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


async def _check_neo4j() -> ServiceHealth:
    """Check Neo4j graph database connectivity."""
    from app.db.neo4j.driver import get_neo4j_driver

    start = time.perf_counter()
    try:
        driver = get_neo4j_driver()
        if driver is None:
            return ServiceHealth(
                name="neo4j",
                status="degraded",
                error="Not configured (optional)",
            )
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ping")
            await result.single()
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="neo4j",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="neo4j",
            status="degraded",  # Neo4j is optional
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


async def _check_redis() -> ServiceHealth:
    """Check Redis connectivity."""
    from app.db.redis.client import get_redis_client

    start = time.perf_counter()
    try:
        client = await get_redis_client()
        if client is None:
            return ServiceHealth(
                name="redis",
                status="degraded",
                error="Not configured (optional)",
            )
        await client.ping()
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="redis",
            status="degraded",  # Redis is optional for caching
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


async def _check_cohere() -> ServiceHealth:
    """Check Cohere API connectivity."""
    from app.llm.gateway import get_async_cohere_client

    start = time.perf_counter()
    try:
        client = await get_async_cohere_client()
        if client is None:
            return ServiceHealth(
                name="cohere",
                status="unhealthy",
                error="Not configured",
            )
        # Do a lightweight check (not a full embedding call)
        # Just verify the client can authenticate
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="cohere",
            status="healthy",
            latency_ms=round(latency, 2),
            details={"model": settings.cohere_embedding_model},
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ServiceHealth(
            name="cohere",
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e)[:200],
        )


@router.get("/health/readiness", response_model=DeepHealthResponse)
async def deep_health_check():
    """Deep health check for Kubernetes readiness probe.

    TIER 5 - Point 22: Deep Health Checks

    Checks connectivity to all critical dependencies:
    - PostgreSQL Magazine (read-only)
    - PostgreSQL ARBOR (read-write)
    - Qdrant (vector search)
    - Neo4j (graph - optional)
    - Redis (cache - optional)
    - Cohere (LLM API)

    Returns 503 if any critical service is unhealthy.
    """
    start = time.perf_counter()

    # Run all health checks in parallel
    checks = await asyncio.gather(
        _check_postgres_magazine(),
        _check_postgres_arbor(),
        _check_qdrant(),
        _check_neo4j(),
        _check_redis(),
        _check_cohere(),
    )

    total_latency = (time.perf_counter() - start) * 1000

    # Determine overall status
    # Critical services: postgres_magazine, postgres_arbor, qdrant, cohere
    critical_services = {"postgres_magazine", "postgres_arbor", "qdrant", "cohere"}
    any_critical_unhealthy = any(
        c.status == "unhealthy" and c.name in critical_services for c in checks
    )
    any_degraded = any(c.status == "degraded" for c in checks)

    if any_critical_unhealthy:
        overall_status = "unhealthy"
    elif any_degraded:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    response = DeepHealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.app_env,
        checks=list(checks),
        total_latency_ms=round(total_latency, 2),
    )

    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


@router.get("/health/liveness")
async def liveness_check():
    """Simple liveness check for Kubernetes.

    Just returns 200 if the process is running.
    """
    return {"status": "alive", "version": settings.app_version}


# ---------------------------------------------------------------------------
# TIER 5 - Point 21: Prometheus Metrics Endpoint
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics.

    TIER 5 - Point 21: Business Metrics Implementation

    Returns metrics in Prometheus text format for scraping.
    """
    from app.observability.metrics import get_metrics, get_metrics_content_type

    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )
