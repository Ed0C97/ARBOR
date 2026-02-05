"""Curator API — review queue, gold standard, and enrichment pipeline triggers.

Provides endpoints for:
- Reviewing and approving enrichments flagged by the confidence engine
- Managing the Gold Standard reference set
- Triggering enrichment for individual entities or batches
- Monitoring feedback loop and drift detection
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.connection import get_db, get_arbor_db
from app.db.postgres.models import (
    ArborGoldStandard,
    ArborReviewQueue,
    ArborEnrichment,
    Brand,
    Venue,
)
from app.db.postgres.repository import UnifiedEntityRepository

router = APIRouter(prefix="/curator")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ReviewQueueItemResponse(BaseModel):
    id: str
    entity_id: str
    entity_type: str
    source_id: int
    reasons: list[str] | None = None
    priority: float = 0.0
    scored_vibe_snapshot: dict | None = None
    fact_sheet_snapshot: dict | None = None
    status: str = "needs_review"
    reviewer: str | None = None
    reviewed_at: str | None = None
    reviewer_notes: str | None = None
    overridden_scores: dict | None = None
    created_at: str | None = None


class ReviewQueueListResponse(BaseModel):
    items: list[ReviewQueueItemResponse]
    total: int
    pending: int


class ReviewDecisionRequest(BaseModel):
    action: str  # "approve" | "reject" | "override"
    reviewer: str
    reviewer_notes: str = ""
    overridden_scores: dict[str, int] | None = None
    overridden_tags: list[str] | None = None
    promote_to_gold: bool = False


class GoldStandardRequest(BaseModel):
    entity_type: str
    source_id: int
    scores: dict[str, int]
    tags: list[str] = []
    curator_notes: str = ""
    curated_by: str = ""


class GoldStandardResponse(BaseModel):
    entity_id: str
    entity_type: str
    source_id: int
    ground_truth_scores: dict | None = None
    ground_truth_tags: list | None = None
    curator_notes: str | None = None
    curated_by: str | None = None
    created_at: str | None = None


class GoldStandardListResponse(BaseModel):
    items: list[GoldStandardResponse]
    total: int


class EnrichRequest(BaseModel):
    entity_type: str
    source_id: int


class BatchEnrichRequest(BaseModel):
    max_entities: int = 50
    entity_type: str | None = None


class EnrichmentStatusResponse(BaseModel):
    total_entities: int = 0
    enriched_entities: int = 0
    pending_review: int = 0
    gold_standard_count: int = 0
    enrichment_coverage: float = 0.0


class DriftReportResponse(BaseModel):
    gold_standard_count: int = 0
    matched_entities: int = 0
    avg_mae: float = 0.0
    per_dimension_mae: dict[str, float] = {}
    drifted_dimensions: list[str] = []
    needs_recalibration: bool = False


# ---------------------------------------------------------------------------
# Review Queue Endpoints
# ---------------------------------------------------------------------------

@router.get("/review-queue", response_model=ReviewQueueListResponse)
async def list_review_queue(
    status: str | None = Query(None, description="Filter by status: needs_review, approved, rejected"),
    sort_by: str = Query("priority", description="Sort by: priority, created_at"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_arbor_db),
):
    """List items in the curator review queue."""
    # ... (body unchanged)

@router.post("/review-queue/{item_id}/decide")
async def decide_review(
    item_id: str,
    body: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_arbor_db),
):
    """Approve, reject, or override an enrichment in the review queue."""
    # ... (body unchanged)

@router.get("/gold-standard", response_model=GoldStandardListResponse)
async def list_gold_standard(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_arbor_db),
):
    """List all gold standard reference entities."""
    # ... (body unchanged)

@router.post("/gold-standard", response_model=GoldStandardResponse)
async def add_gold_standard(
    body: GoldStandardRequest,
    session: AsyncSession = Depends(get_arbor_db),
):
    """Add or update a gold standard reference entity."""
    # ... (body unchanged)

@router.delete("/gold-standard/{entity_id}")
async def remove_gold_standard(
    entity_id: str,
    session: AsyncSession = Depends(get_arbor_db),
):
    """Remove an entity from the gold standard set."""
    # ... (body unchanged)

@router.post("/enrich/single")
async def trigger_single_enrichment(
    body: EnrichRequest,
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Trigger the 5-layer enrichment pipeline for a single entity.

    Runs synchronously (no Temporal) for immediate feedback.
    """
    from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator

    # Verify entity exists (Magazine DB)
    if body.entity_type == "brand":
        entity = await session.get(Brand, body.source_id)
    elif body.entity_type == "venue":
        entity = await session.get(Venue, body.source_id)
    else:
        raise HTTPException(status_code=400, detail="entity_type must be 'brand' or 'venue'")

    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    # Build kwargs
    kwargs = _entity_to_enrichment_kwargs(entity, body.entity_type, body.source_id)

    # Run enrichment (Arbor DB)
    orchestrator = EnrichmentOrchestrator(arbor_session)
    result = await orchestrator.enrich_entity(**kwargs)

    return {
        "entity_id": result.entity_id,
        "success": result.success,
        "error": result.error,
        "sources_collected": result.sources_collected,
        "was_calibrated": result.was_calibrated,
        "confidence": result.scored_vibe.overall_confidence if result.scored_vibe else 0.0,
        "needs_review": result.scored_vibe.needs_review if result.scored_vibe else False,
    }


@router.post("/enrich/batch")
async def trigger_batch_enrichment(
    body: BatchEnrichRequest,
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Trigger enrichment for all unenriched entities.

    For large batches, prefer the Temporal workflow approach.
    This endpoint processes entities sequentially.
    """
    from sqlalchemy import and_
    from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator

    entities_to_enrich = []

    # Find unenriched brands (Need check against Arbor DB for exclusion)
    # This logic is tricky: "not in arbor_enrichments".
    # Since they are different DBs, we can't do a direct SQL subquery join unless fdw used.
    # We must fetch IDs from Arbor first, then query Magazine.
    
    # 1. Get existing enriched IDs from Arbor
    existing_enrichments_result = await arbor_session.execute(
        select(ArborEnrichment.entity_type, ArborEnrichment.source_id)
    )
    existing_map = {(row.entity_type, row.source_id) for row in existing_enrichments_result.all()}
    
    # 2. Query Magazine for entities (fetch slightly more to handle client-side filtering)
    # Warning: this is inefficient for large datasets, but acceptable for this batch endpoint logic
    # Better approach: Iterate Magazine entities and skip if in set.
    
    # Simpler implementation for prototype:
    # Just fetch candidates from Magazine and check against local set.
    
    candidates = []
    
    if body.entity_type is None or body.entity_type == "brand":
        res = await session.execute(select(Brand).where(Brand.is_active == True).limit(body.max_entities * 2))
        for b in res.scalars().all():
            if ("brand", b.id) not in existing_map:
                candidates.append(("brand", b))
                if len(candidates) >= body.max_entities:
                    break

    remaining = body.max_entities - len(candidates)
    if remaining > 0 and (body.entity_type is None or body.entity_type == "venue"):
        res = await session.execute(select(Venue).where(Venue.is_active == True).limit(remaining * 5))
        for v in res.scalars().all():
            if ("venue", v.id) not in existing_map:
                candidates.append(("venue", v))
                if len(candidates) >= body.max_entities:
                     break
                     
    if not candidates:
        return {"message": "No unenriched entities found", "total": 0, "enriched": 0, "failed": 0}

    # Process each entity (Arbor DB)
    orchestrator = EnrichmentOrchestrator(arbor_session)
    enriched = 0
    failed = 0
    results = []

    for entity_type, entity in candidates:
        source_id = entity.id
        kwargs = _entity_to_enrichment_kwargs(entity, entity_type, source_id)
        result = await orchestrator.enrich_entity(**kwargs)
        if result.success:
            enriched += 1
        else:
            failed += 1
        results.append({
            "entity_id": result.entity_id,
            "success": result.success,
            "error": result.error,
        })

    return {
        "total": len(candidates),
        "enriched": enriched,
        "failed": failed,
        "results": results[:20],
    }


@router.get("/status", response_model=EnrichmentStatusResponse)
async def get_enrichment_status(
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Get overall enrichment pipeline status and statistics."""
    repo = UnifiedEntityRepository(session, arbor_session)
    stats = await repo.stats()

    pending_result = await arbor_session.execute(
        select(func.count()).select_from(ArborReviewQueue)
        .where(ArborReviewQueue.status == "needs_review")
    )
    pending_review = pending_result.scalar_one()

    gold_result = await arbor_session.execute(
        select(func.count()).select_from(ArborGoldStandard)
    )
    gold_count = gold_result.scalar_one()

    total = stats["total_entities"]
    enriched = stats["enriched_entities"]
    coverage = (enriched / total * 100) if total > 0 else 0.0

    return EnrichmentStatusResponse(
        total_entities=total,
        enriched_entities=enriched,
        pending_review=pending_review,
        gold_standard_count=gold_count,
        enrichment_coverage=round(coverage, 1),
    )


@router.get("/drift-report", response_model=DriftReportResponse)
async def get_drift_report(
    session: AsyncSession = Depends(get_arbor_db),
):
    """Compare current enrichment scores against gold standard to detect drift."""
    from app.ingestion.pipeline.gold_standard import GoldStandardManager

    # Collect current scores
    result = await session.execute(select(ArborEnrichment))
    enrichments = list(result.scalars().all())

    current_scores = {}
    for enr in enrichments:
        entity_id = f"{enr.entity_type}_{enr.source_id}"
        if enr.vibe_dna and "dimensions" in enr.vibe_dna:
            current_scores[entity_id] = enr.vibe_dna["dimensions"]

    manager = GoldStandardManager(session)
    drift = await manager.compute_drift(current_scores)

    return DriftReportResponse(**drift)


@router.get("/feedback-analysis")
async def analyze_feedback(
    since_hours: int = Query(24, ge=1, le=720),
    session: AsyncSession = Depends(get_arbor_db),
):
    """Analyze implicit user feedback to identify scoring anomalies."""
    from app.ingestion.pipeline.feedback_loop import ContinuousLearner

    learner = ContinuousLearner(session)
    return await learner.analyze_implicit_feedback(since_hours=since_hours)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_to_enrichment_kwargs(entity, entity_type: str, source_id: int) -> dict:
    """Convert a Brand or Venue ORM object to enrichment kwargs."""
    base = {
        "entity_type": entity_type,
        "source_id": source_id,
        "name": entity.name,
        "category": entity.category,
        "description": getattr(entity, "description", None),
        "specialty": getattr(entity, "specialty", None),
        "notes": getattr(entity, "notes", None),
        "website": getattr(entity, "website", None),
        "instagram": getattr(entity, "instagram", None),
        "style": getattr(entity, "style", None),
        "gender": getattr(entity, "gender", None),
        "rating": getattr(entity, "rating", None),
        "is_featured": getattr(entity, "is_featured", False),
        "country": getattr(entity, "country", None),
    }

    if entity_type == "venue":
        base.update({
            "city": getattr(entity, "city", None),
            "address": getattr(entity, "address", None),
            "latitude": getattr(entity, "latitude", None),
            "longitude": getattr(entity, "longitude", None),
            "neighborhood": getattr(entity, "region", None),
            "maps_url": getattr(entity, "maps_url", None),
            "price_range": getattr(entity, "price_range", None),
        })
    else:
        base.update({
            "city": getattr(entity, "area", None),
            "neighborhood": getattr(entity, "neighborhood", None),
        })

    return base
