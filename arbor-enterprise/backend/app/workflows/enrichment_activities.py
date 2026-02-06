"""Temporal.io activity definitions for the 5-layer enrichment pipeline.

Each activity imports its dependencies inside the function body
(Temporal requirement for proper serialization).
"""

import logging

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def run_full_enrichment(entity_type: str, source_id: int) -> dict:
    """Activity: Run the complete 5-layer enrichment pipeline for one entity.

    This is the recommended single-activity approach — it runs all 5 layers
    within one activity, keeping database session management simple.
    """
    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import Brand, Venue
    from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator

    async with async_session_factory() as session:
        # Fetch the source entity
        if entity_type == "brand":
            entity = await session.get(Brand, source_id)
        elif entity_type == "venue":
            entity = await session.get(Venue, source_id)
        else:
            return {"success": False, "error": f"Unknown entity_type: {entity_type}"}

        if not entity:
            return {"success": False, "error": f"{entity_type}_{source_id} not found"}

        # Build kwargs from entity fields
        kwargs = _entity_to_kwargs(entity, entity_type, source_id)

        # Run the orchestrator
        orchestrator = EnrichmentOrchestrator(session)
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


@activity.defn
async def collect_entity_data(entity_type: str, source_id: int) -> dict:
    """Activity: Collect data from all sources for an entity (Layer 1)."""
    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import Brand, Venue
    from app.ingestion.pipeline.collector import MultiSourceCollector

    async with async_session_factory() as session:
        if entity_type == "brand":
            entity = await session.get(Brand, source_id)
        elif entity_type == "venue":
            entity = await session.get(Venue, source_id)
        else:
            return {"error": f"Unknown entity_type: {entity_type}"}

        if not entity:
            return {"error": f"{entity_type}_{source_id} not found"}

        kwargs = _entity_to_kwargs(entity, entity_type, source_id)
        collector = MultiSourceCollector()
        collected = await collector.collect(**kwargs)

        # Serialize collected sources for passing between activities
        return {
            "entity_id": collected.entity_id,
            "entity_type": collected.entity_type,
            "source_id": collected.source_id,
            "name": collected.name,
            "category": collected.category,
            "city": collected.city,
            "sources_count": len(collected.sources),
            "sources": [
                {
                    "source_type": s.source_type.value,
                    "raw_text": s.raw_text[:5000],  # Truncate for serialization
                    "images": s.images[:10],
                    "structured_data": s.structured_data,
                }
                for s in collected.sources
            ],
        }


@activity.defn
async def run_fact_analyzers(collected_data: dict) -> dict:
    """Activity: Run fact-based analyzers on collected data (Layer 2)."""
    import asyncio

    from app.ingestion.pipeline.context_analyzer import ContextAnalyzer
    from app.ingestion.pipeline.price_analyzer import PriceAnalyzer
    from app.ingestion.pipeline.schemas import SourceType
    from app.ingestion.pipeline.text_fact_analyzer import TextFactAnalyzer
    from app.ingestion.pipeline.vision_fact_analyzer import VisionFactAnalyzer

    text_analyzer = TextFactAnalyzer()
    vision_analyzer = VisionFactAnalyzer()
    context_analyzer = ContextAnalyzer()
    price_analyzer = PriceAnalyzer()

    # Gather reviews, images, and metadata from collected sources
    all_reviews = []
    all_images = []
    db_structured = {}

    for source in collected_data.get("sources", []):
        st = source["source_type"]
        if st in ("google_reviews", "instagram"):
            reviews = source["structured_data"].get("reviews") or source["structured_data"].get(
                "captions", []
            )
            all_reviews.extend(reviews)
        all_images.extend(source.get("images", []))
        if st == "database":
            db_structured = source.get("structured_data", {})

    # Run analyzers in parallel
    text_result, vision_result, context_result = await asyncio.gather(
        text_analyzer.analyze(
            reviews=all_reviews,
            description=db_structured.get("description", ""),
            entity_name=collected_data["name"],
            notes=db_structured.get("notes", ""),
        ),
        vision_analyzer.analyze(image_urls=all_images[:8]),
        context_analyzer.analyze(
            name=collected_data["name"],
            category=collected_data["category"],
            city=collected_data.get("city"),
            neighborhood=db_structured.get("neighborhood"),
            country=db_structured.get("country"),
            address=db_structured.get("address"),
            style=db_structured.get("style"),
            gender=db_structured.get("gender"),
            specialty=db_structured.get("specialty"),
            rating=db_structured.get("rating"),
            is_featured=db_structured.get("is_featured", False),
        ),
        return_exceptions=True,
    )

    # Handle exceptions gracefully
    if isinstance(text_result, Exception):
        logger.error(f"Text analysis failed: {text_result}")
        text_result = text_analyzer._empty_result()
    if isinstance(vision_result, Exception):
        logger.error(f"Vision analysis failed: {vision_result}")
        vision_result = vision_analyzer._empty_result()
    if isinstance(context_result, Exception):
        logger.error(f"Context analysis failed: {context_result}")
        context_result = {"location_context": [], "brand_signals": [], "audience_indicators": []}

    # Run price analyzer
    try:
        price_result = await price_analyzer.analyze(
            price_range_db=db_structured.get("price_range"),
            review_price_facts=text_result.get("price_points", []),
            description=db_structured.get("description", ""),
            entity_name=collected_data["name"],
            category=collected_data["category"],
        )
    except Exception as e:
        logger.error(f"Price analysis failed: {e}")
        price_result = {
            "price_tier": "$$",
            "avg_price": None,
            "price_confidence": 0.2,
            "price_facts": [],
        }

    # Serialize for inter-activity transfer
    def facts_to_dicts(facts):
        return (
            [
                {
                    "fact_type": f.fact_type,
                    "value": f.value,
                    "source": f.source.value,
                    "confidence": f.confidence,
                }
                for f in facts
            ]
            if facts and hasattr(facts[0] if facts else None, "fact_type")
            else facts
        )

    return {
        "entity_id": collected_data["entity_id"],
        "name": collected_data["name"],
        "category": collected_data["category"],
        "text_result": {
            k: (
                facts_to_dicts(v)
                if isinstance(v, list) and v and hasattr(v[0] if v else None, "fact_type")
                else v
            )
            for k, v in text_result.items()
        },
        "vision_result": {
            k: (
                facts_to_dicts(v)
                if isinstance(v, list) and v and hasattr(v[0] if v else None, "fact_type")
                else v
            )
            for k, v in vision_result.items()
        },
        "context_result": {
            k: (
                facts_to_dicts(v)
                if isinstance(v, list) and v and hasattr(v[0] if v else None, "fact_type")
                else v
            )
            for k, v in context_result.items()
        },
        "price_result": {
            "price_tier": price_result.get("price_tier", "$$"),
            "avg_price": price_result.get("avg_price"),
            "price_confidence": price_result.get("price_confidence", 0.2),
        },
        "sources_used": [s["source_type"] for s in collected_data.get("sources", [])],
        "review_count": len(all_reviews),
        "image_count": len(all_images),
        "common_themes": text_result.get("common_themes", []),
        "signature_items": text_result.get("signature_items", []),
        "visual_tags": vision_result.get("visual_tags", []),
        "visual_summary": vision_result.get("visual_summary", ""),
        "visual_style": vision_result.get("visual_style", ""),
    }


@activity.defn
async def score_with_calibration(
    fact_sheet_data: dict,
    entity_type: str,
    source_id: int,
) -> dict:
    """Activity: Score the fact sheet using the calibrated engine (Layer 3)."""
    from app.db.postgres.connection import async_session_factory
    from app.ingestion.pipeline.gold_standard import GoldStandardManager
    from app.ingestion.pipeline.schemas import FactSheet
    from app.ingestion.pipeline.scoring_engine import CalibratedScoringEngine

    # Reconstruct a minimal FactSheet for scoring
    fact_sheet = FactSheet(
        entity_id=fact_sheet_data["entity_id"],
        name=fact_sheet_data["name"],
        category=fact_sheet_data["category"],
    )
    fact_sheet.visual_tags = fact_sheet_data.get("visual_tags", [])
    fact_sheet.visual_summary = fact_sheet_data.get("visual_summary", "")
    fact_sheet.visual_style = fact_sheet_data.get("visual_style", "")
    fact_sheet.price_range_label = fact_sheet_data.get("price_result", {}).get("price_tier", "")
    fact_sheet.avg_price = fact_sheet_data.get("price_result", {}).get("avg_price")
    fact_sheet.signature_items = fact_sheet_data.get("signature_items", [])
    fact_sheet.review_count = fact_sheet_data.get("review_count", 0)
    fact_sheet.common_themes = fact_sheet_data.get("common_themes", [])

    # Get gold standard examples
    async with async_session_factory() as session:
        gold_mgr = GoldStandardManager(session)
        gold_examples = await gold_mgr.get_few_shot_examples(
            category=fact_sheet_data["category"],
            max_examples=5,
        )

    # Score
    scoring_engine = CalibratedScoringEngine()
    scored = await scoring_engine.score(fact_sheet, gold_examples or None)

    # Serialize for inter-activity transfer
    return {
        "entity_id": scored.entity_id,
        "dimensions": [
            {
                "dimension": d.dimension,
                "score": d.score,
                "confidence": d.confidence,
                "source_scores": d.source_scores,
            }
            for d in scored.dimensions
        ],
        "tags": scored.tags,
        "target_audience": scored.target_audience,
        "summary": scored.summary,
        "overall_confidence": scored.overall_confidence,
        "needs_review": scored.needs_review,
        "review_reasons": scored.review_reasons,
        "calibrated": scored.calibrated,
        "sources_count": scored.sources_count,
    }


@activity.defn
async def validate_and_persist(
    scored_data: dict,
    fact_sheet_data: dict,
    entity_type: str,
    source_id: int,
) -> dict:
    """Activity: Validate confidence, create review items, and persist (Layer 4+5)."""
    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import ArborReviewQueue
    from app.db.postgres.repository import EnrichmentRepository

    # Build vibe_dna dict for storage
    vibe_dna = {
        "dimensions": {d["dimension"]: d["score"] for d in scored_data["dimensions"]},
        "confidence": {
            d["dimension"]: round(d["confidence"], 3) for d in scored_data["dimensions"]
        },
        "tags": scored_data.get("tags", []),
        "target_audience": scored_data.get("target_audience", "General"),
        "summary": scored_data.get("summary", ""),
        "overall_confidence": scored_data.get("overall_confidence", 0.0),
        "needs_review": scored_data.get("needs_review", False),
        "review_reasons": scored_data.get("review_reasons", []),
        "calibrated": scored_data.get("calibrated", False),
        "sources_count": scored_data.get("sources_count", 0),
    }

    async with async_session_factory() as session:
        # Persist enrichment
        enr_repo = EnrichmentRepository(session)
        await enr_repo.upsert(
            entity_type=entity_type,
            source_id=source_id,
            vibe_dna=vibe_dna,
            tags=scored_data.get("tags", []),
            neo4j_synced=False,
        )

        # Add to review queue if needed
        if scored_data.get("needs_review"):
            queue_item = ArborReviewQueue(
                entity_type=entity_type,
                source_id=source_id,
                reasons=scored_data.get("review_reasons", []),
                priority=_compute_priority(scored_data),
                scored_vibe_snapshot=vibe_dna,
                fact_sheet_snapshot={
                    "name": fact_sheet_data.get("name"),
                    "category": fact_sheet_data.get("category"),
                    "sources_used": fact_sheet_data.get("sources_used", []),
                    "review_count": fact_sheet_data.get("review_count", 0),
                },
                status="needs_review",
            )
            session.add(queue_item)

        await session.commit()

    entity_id = f"{entity_type}_{source_id}"
    return {
        "entity_id": entity_id,
        "success": True,
        "confidence": scored_data.get("overall_confidence", 0.0),
        "needs_review": scored_data.get("needs_review", False),
        "calibrated": scored_data.get("calibrated", False),
    }


@activity.defn
async def get_unenriched_entities(
    max_entities: int = 100,
    entity_type: str | None = None,
) -> list[dict]:
    """Activity: Get list of entities that don't have enrichment data yet."""
    from sqlalchemy import and_, select

    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import ArborEnrichment, Brand, Venue

    async with async_session_factory() as session:
        entities = []

        if entity_type is None or entity_type == "brand":
            # Find brands without enrichment
            subq = select(ArborEnrichment.source_id).where(ArborEnrichment.entity_type == "brand")
            result = await session.execute(
                select(Brand.id, Brand.name, Brand.category)
                .where(
                    and_(
                        Brand.is_active == True,  # noqa: E712
                        Brand.id.not_in(subq),
                    )
                )
                .limit(max_entities)
            )
            for row in result:
                entities.append(
                    {
                        "entity_type": "brand",
                        "source_id": row.id,
                        "name": row.name,
                        "category": row.category,
                    }
                )

        if entity_type is None or entity_type == "venue":
            remaining = max_entities - len(entities)
            if remaining > 0:
                subq = select(ArborEnrichment.source_id).where(
                    ArborEnrichment.entity_type == "venue"
                )
                result = await session.execute(
                    select(Venue.id, Venue.name, Venue.category)
                    .where(
                        and_(
                            Venue.is_active == True,  # noqa: E712
                            Venue.id.not_in(subq),
                        )
                    )
                    .limit(remaining)
                )
                for row in result:
                    entities.append(
                        {
                            "entity_type": "venue",
                            "source_id": row.id,
                            "name": row.name,
                            "category": row.category,
                        }
                    )

        return entities


@activity.defn
async def generate_and_sync_embedding(entity_type: str, source_id: int) -> dict:
    """Activity: Generate embedding from enrichment and sync to Qdrant + Neo4j."""
    from app.db.neo4j.queries import Neo4jQueries
    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import ArborEnrichment, Brand, Venue
    from app.db.qdrant.collections import QdrantCollections
    from app.ingestion.analyzers.embedding import EmbeddingGenerator

    entity_id = f"{entity_type}_{source_id}"

    async with async_session_factory() as session:
        # Get enrichment
        from sqlalchemy import select

        result = await session.execute(
            select(ArborEnrichment).where(
                ArborEnrichment.entity_type == entity_type,
                ArborEnrichment.source_id == source_id,
            )
        )
        enrichment = result.scalar_one_or_none()
        if not enrichment or not enrichment.vibe_dna:
            return {"synced": False, "reason": "No enrichment found"}

        # Get entity name and category
        if entity_type == "brand":
            entity = await session.get(Brand, source_id)
        else:
            entity = await session.get(Venue, source_id)

        if not entity:
            return {"synced": False, "reason": "Entity not found"}

        vibe = enrichment.vibe_dna
        tags = enrichment.tags or []

        # Generate embedding
        embedding_text = (
            f"{entity.name} | {entity.category} | "
            f"{' '.join(tags[:10])} | "
            f"{vibe.get('summary', '')} | "
            f"{vibe.get('target_audience', '')}"
        )
        gen = EmbeddingGenerator()
        embedding = await gen.generate(embedding_text)

        # Save to Qdrant
        payload = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "name": entity.name,
            "category": entity.category,
            "city": getattr(entity, "city", None),
            "dimensions": vibe.get("dimensions", {}),
            "tags": tags,
            "confidence": vibe.get("overall_confidence", 0.0),
            "status": "approved" if not vibe.get("needs_review") else "pending",
        }
        qdrant = QdrantCollections()
        qdrant.upsert_vector(entity_id, embedding, payload)

        # Save to Neo4j
        neo4j = Neo4jQueries()
        await neo4j.create_entity_node(
            entity_id=entity_id,
            name=entity.name,
            category=entity.category,
            city=getattr(entity, "city", None),
        )

        # Mark as synced
        enrichment.embedding_id = entity_id
        enrichment.neo4j_synced = True
        await session.commit()

    return {"synced": True, "entity_id": entity_id}


# ── Helper functions ─────────────────────────────────────────────────


def _entity_to_kwargs(entity, entity_type: str, source_id: int) -> dict:
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

    # Venue-specific fields
    if entity_type == "venue":
        base.update(
            {
                "city": getattr(entity, "city", None),
                "address": getattr(entity, "address", None),
                "latitude": getattr(entity, "latitude", None),
                "longitude": getattr(entity, "longitude", None),
                "neighborhood": getattr(entity, "region", None),  # region ≈ neighborhood
                "maps_url": getattr(entity, "maps_url", None),
                "price_range": getattr(entity, "price_range", None),
            }
        )
    else:
        # Brand-specific
        base.update(
            {
                "city": getattr(entity, "area", None),  # area ≈ city for brands
                "neighborhood": getattr(entity, "neighborhood", None),
            }
        )

    return base


def _compute_priority(scored_data: dict) -> float:
    """Compute review priority from scored data."""
    priority = 0.0

    # Low confidence = higher priority
    confidence = scored_data.get("overall_confidence", 0.5)
    if confidence < 0.3:
        priority += 3.0
    elif confidence < 0.5:
        priority += 1.5

    # More review reasons = higher priority
    reasons = scored_data.get("review_reasons", [])
    priority += len(reasons) * 0.5

    return round(priority, 2)
