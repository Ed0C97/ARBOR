"""Enrichment Orchestrator — coordinates all 5 layers of the pipeline.

This is the main entry point for enriching an entity. It:
1. Collects data from all sources (Layer 1)
2. Runs fact-based analyzers in parallel (Layer 2)
3. Scores using the calibrated engine with gold standard (Layer 3)
4. Validates with confidence scoring and creates review items (Layer 4)
5. Persists results and triggers sync (Layer 5)
"""

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import ArborReviewQueue
from app.db.postgres.repository import EnrichmentRepository
from app.ingestion.pipeline.collector import MultiSourceCollector
from app.ingestion.pipeline.confidence import ConfidenceAnalyzer
from app.ingestion.pipeline.context_analyzer import ContextAnalyzer
from app.ingestion.pipeline.gold_standard import GoldStandardManager
from app.ingestion.pipeline.price_analyzer import PriceAnalyzer
from app.ingestion.pipeline.schemas import (
    CollectedSources,
    FactSheet,
    ReviewQueueItem,
    ScoredVibeDNA,
    SourceType,
)
from app.ingestion.pipeline.scoring_engine import CalibratedScoringEngine
from app.ingestion.pipeline.text_fact_analyzer import TextFactAnalyzer
from app.ingestion.pipeline.vision_fact_analyzer import VisionFactAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Complete result from the enrichment pipeline."""

    entity_id: str
    entity_type: str
    source_id: int
    name: str

    fact_sheet: FactSheet | None = None
    scored_vibe: ScoredVibeDNA | None = None
    review_item: ReviewQueueItem | None = None

    # Status
    success: bool = False
    error: str | None = None
    sources_collected: int = 0
    was_calibrated: bool = False


class EnrichmentOrchestrator:
    """Main orchestrator for the 5-layer enrichment pipeline."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.collector = MultiSourceCollector()
        self.text_analyzer = TextFactAnalyzer()
        self.vision_analyzer = VisionFactAnalyzer()
        self.price_analyzer = PriceAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.scoring_engine = CalibratedScoringEngine()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.gold_standard = GoldStandardManager(session)
        self.enrichment_repo = EnrichmentRepository(session)

    async def enrich_entity(
        self,
        entity_type: str,
        source_id: int,
        name: str,
        category: str,
        city: str | None = None,
        description: str | None = None,
        specialty: str | None = None,
        notes: str | None = None,
        website: str | None = None,
        instagram: str | None = None,
        style: str | None = None,
        gender: str | None = None,
        rating: float | None = None,
        price_range: str | None = None,
        address: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        neighborhood: str | None = None,
        country: str | None = None,
        maps_url: str | None = None,
        is_featured: bool = False,
    ) -> EnrichmentResult:
        """Run the full 5-layer enrichment pipeline for a single entity."""
        entity_id = f"{entity_type}_{source_id}"
        result = EnrichmentResult(
            entity_id=entity_id,
            entity_type=entity_type,
            source_id=source_id,
            name=name,
        )

        try:
            # ── Layer 1: Multi-source data collection ─────────────────
            logger.info(f"[L1] Collecting sources for {entity_id}")
            collected = await self.collector.collect(
                entity_type=entity_type,
                source_id=source_id,
                name=name,
                category=category,
                city=city,
                description=description,
                specialty=specialty,
                notes=notes,
                website=website,
                instagram=instagram,
                style=style,
                gender=gender,
                rating=rating,
                price_range=price_range,
                address=address,
                latitude=latitude,
                longitude=longitude,
                neighborhood=neighborhood,
                country=country,
                maps_url=maps_url,
                is_featured=is_featured,
            )
            result.sources_collected = len(collected.sources)

            # ── Layer 2: Parallel fact-based analysis ─────────────────
            logger.info(f"[L2] Analyzing facts for {entity_id} ({len(collected.sources)} sources)")
            fact_sheet = await self._run_analyzers(collected, entity_type, source_id, is_featured)
            result.fact_sheet = fact_sheet

            # ── Layer 3: Calibrated scoring ───────────────────────────
            logger.info(f"[L3] Scoring {entity_id} with calibrated engine")
            gold_examples = await self.gold_standard.get_few_shot_examples(
                category=category, max_examples=5
            )
            scored = await self.scoring_engine.score(fact_sheet, gold_examples)
            result.was_calibrated = bool(gold_examples)

            # ── Layer 4: Confidence + disagreement detection ──────────
            logger.info(f"[L4] Validating confidence for {entity_id}")
            scored = self.confidence_analyzer.analyze(scored, fact_sheet, is_featured)
            result.scored_vibe = scored

            # Create review item if needed
            review_item = self.confidence_analyzer.create_review_item(
                scored=scored,
                fact_sheet=fact_sheet,
                entity_type=entity_type,
                source_id=source_id,
                name=name,
                category=category,
                is_featured=is_featured,
            )
            result.review_item = review_item

            # ── Layer 5: Persist results ──────────────────────────────
            logger.info(f"[L5] Persisting enrichment for {entity_id}")
            await self._persist_results(entity_type, source_id, scored, fact_sheet, review_item)

            result.success = True
            logger.info(
                f"Enrichment complete for {entity_id}: "
                f"confidence={scored.overall_confidence:.2f}, "
                f"review={'YES' if scored.needs_review else 'NO'}, "
                f"calibrated={result.was_calibrated}"
            )

        except Exception as e:
            logger.error(f"Enrichment failed for {entity_id}: {e}", exc_info=True)
            result.error = str(e)

        return result

    async def enrich_batch(
        self,
        entities: list[dict],
    ) -> list[EnrichmentResult]:
        """Enrich a batch of entities sequentially.

        Each entity dict should have the same keys as enrich_entity's parameters.
        """
        results = []
        for entity_data in entities:
            result = await self.enrich_entity(**entity_data)
            results.append(result)
        return results

    # ── Internal methods ──────────────────────────────────────────────

    async def _run_analyzers(
        self,
        collected: CollectedSources,
        entity_type: str,
        source_id: int,
        is_featured: bool,
    ) -> FactSheet:
        """Run all fact-based analyzers in parallel and merge into a FactSheet."""
        fact_sheet = FactSheet(
            entity_id=collected.entity_id,
            name=collected.name,
            category=collected.category,
        )

        # Gather all available text (reviews + descriptions)
        all_reviews: list[str] = []
        all_images: list[str] = []
        db_data = collected.get_source(SourceType.DATABASE)
        description = ""
        notes_text = ""

        if db_data:
            description = db_data.structured_data.get("description", "") or ""
            notes_text = db_data.structured_data.get("notes", "") or ""

        for source in collected.sources:
            if source.source_type in (SourceType.GOOGLE_REVIEWS, SourceType.INSTAGRAM):
                reviews_from_source = source.structured_data.get(
                    "reviews"
                ) or source.structured_data.get("captions", [])
                all_reviews.extend(reviews_from_source)
            all_images.extend(source.images)

        # Run analyzers in parallel
        text_task = self.text_analyzer.analyze(
            reviews=all_reviews,
            description=description,
            entity_name=collected.name,
            notes=notes_text,
        )

        vision_task = self.vision_analyzer.analyze(
            image_urls=all_images[:8],
            source=(
                SourceType.GOOGLE_PHOTOS
                if any(s.source_type == SourceType.GOOGLE_REVIEWS for s in collected.sources)
                else SourceType.INSTAGRAM
            ),
        )

        context_task = self.context_analyzer.analyze(
            name=collected.name,
            category=collected.category,
            city=collected.city,
            neighborhood=db_data.structured_data.get("neighborhood") if db_data else None,
            country=db_data.structured_data.get("country") if db_data else None,
            address=db_data.structured_data.get("address") if db_data else None,
            description=description,
            style=db_data.structured_data.get("style") if db_data else None,
            gender=db_data.structured_data.get("gender") if db_data else None,
            specialty=db_data.structured_data.get("specialty") if db_data else None,
            rating=db_data.structured_data.get("rating") if db_data else None,
            is_featured=is_featured,
        )

        text_result, vision_result, context_result = await asyncio.gather(
            text_task,
            vision_task,
            context_task,
            return_exceptions=True,
        )

        # Handle exceptions gracefully
        if isinstance(text_result, Exception):
            logger.error(f"Text analysis failed: {text_result}")
            text_result = self.text_analyzer._empty_result()
        if isinstance(vision_result, Exception):
            logger.error(f"Vision analysis failed: {vision_result}")
            vision_result = self.vision_analyzer._empty_result()
        if isinstance(context_result, Exception):
            logger.error(f"Context analysis failed: {context_result}")
            context_result = {
                "location_context": [],
                "brand_signals": [],
                "audience_indicators": [],
            }

        # Now run price analyzer with text facts as input
        price_result = await self.price_analyzer.analyze(
            price_range_db=db_data.structured_data.get("price_range") if db_data else None,
            review_price_facts=text_result.get("price_points", []),
            description=description,
            entity_name=collected.name,
            category=collected.category,
        )

        # ── Merge everything into the FactSheet ──
        fact_sheet.materials = text_result.get("materials", []) + vision_result.get("materials", [])
        fact_sheet.price_points = text_result.get("price_points", []) + price_result.get(
            "price_facts", []
        )
        fact_sheet.interior_elements = text_result.get("interior_elements", []) + vision_result.get(
            "interior_elements", []
        )
        fact_sheet.service_indicators = text_result.get("service_indicators", [])
        fact_sheet.audience_indicators = (
            text_result.get("audience_indicators", [])
            + vision_result.get("audience_indicators", [])
            + context_result.get("audience_indicators", [])
        )
        fact_sheet.brand_signals = (
            text_result.get("brand_signals", [])
            + vision_result.get("brand_signals", [])
            + context_result.get("brand_signals", [])
        )
        fact_sheet.location_context = context_result.get("location_context", [])

        # Visual facts
        fact_sheet.visual_style = vision_result.get("visual_style", "")
        fact_sheet.visual_tags = vision_result.get("visual_tags", [])
        fact_sheet.visual_summary = vision_result.get("visual_summary", "")

        # Price
        fact_sheet.avg_price = price_result.get("avg_price")
        fact_sheet.price_range_label = price_result.get("price_tier", "")
        fact_sheet.signature_items = text_result.get("signature_items", [])

        # Text summary
        fact_sheet.review_sentiment = 0.0  # Could be refined
        fact_sheet.review_count = len(all_reviews)
        fact_sheet.common_themes = text_result.get("common_themes", [])

        # Track sources used
        fact_sheet.sources_used = [s.source_type for s in collected.sources]

        return fact_sheet

    async def _persist_results(
        self,
        entity_type: str,
        source_id: int,
        scored: ScoredVibeDNA,
        fact_sheet: FactSheet,
        review_item: ReviewQueueItem | None,
    ) -> None:
        """Persist enrichment results to PostgreSQL."""
        # Upsert enrichment
        vibe_dna = scored.to_vibe_dna_dict()
        await self.enrichment_repo.upsert(
            entity_type=entity_type,
            source_id=source_id,
            vibe_dna=vibe_dna,
            tags=scored.tags,
            neo4j_synced=False,  # Will be synced by SyncWorkflow
        )

        # Add to review queue if needed
        if review_item:
            queue_item = ArborReviewQueue(
                entity_type=entity_type,
                source_id=source_id,
                reasons=review_item.reasons,
                priority=review_item.priority,
                scored_vibe_snapshot=vibe_dna,
                fact_sheet_snapshot={
                    "name": fact_sheet.name,
                    "category": fact_sheet.category,
                    "sources_used": [s.value for s in fact_sheet.sources_used],
                    "fact_text": fact_sheet.to_scoring_text(),
                },
                status="needs_review",
            )
            self.session.add(queue_item)

        await self.session.commit()
