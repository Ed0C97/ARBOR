"""Full enrichment of all unenriched entities."""

import asyncio
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.repository import EnrichmentRepository
from app.db.postgres.models import Brand, Venue, ArborEnrichment
from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Enrich ALL unenriched entities."""

    async with magazine_session_factory() as main_session, arbor_session_factory() as arbor_session:
        # Get existing enrichments
        existing_result = await arbor_session.execute(
            select(ArborEnrichment.entity_type, ArborEnrichment.source_id)
        )
        existing_map = {(row.entity_type, row.source_id) for row in existing_result.all()}

        logger.info(f"Found {len(existing_map)} existing enrichments")

        # Get all active brands
        brands_result = await main_session.execute(
            select(Brand).where(Brand.is_active == True)
        )
        all_brands = list(brands_result.scalars().all())

        # Get all active venues
        venues_result = await main_session.execute(
            select(Venue).where(Venue.is_active == True)
        )
        all_venues = list(venues_result.scalars().all())

        logger.info(f"Found {len(all_brands)} active brands, {len(all_venues)} active venues")

        # Filter unenriched
        unenriched_brands = [b for b in all_brands if ("brand", b.id) not in existing_map]
        unenriched_venues = [v for v in all_venues if ("venue", v.id) not in existing_map]

        logger.info(f"Need to enrich: {len(unenriched_brands)} brands, {len(unenriched_venues)} venues")

        if not unenriched_brands and not unenriched_venues:
            logger.info("All entities already enriched!")
            return

        # Enrich all
        orchestrator = EnrichmentOrchestrator(arbor_session)

        total = len(unenriched_brands) + len(unenriched_venues)
        enriched = 0
        failed = 0

        # Enrich brands
        for i, brand in enumerate(unenriched_brands, 1):
            try:
                logger.info(f"[{i}/{len(unenriched_brands)}] Enriching brand: {brand.name} (id={brand.id})")
                result = await orchestrator.enrich_entity(
                    entity_type="brand",
                    source_id=brand.id,
                    name=brand.name,
                    category=brand.category,
                    description=brand.description,
                    website=brand.website,
                    instagram=brand.instagram,
                    city=brand.city,
                    country=brand.country
                )

                if result.success:
                    enriched += 1
                    logger.info(f"  ✓ Success (confidence: {result.overall_confidence:.2f})")
                else:
                    failed += 1
                    logger.warning(f"  ✗ Failed: {result.error}")

            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Exception: {e}")

        # Enrich venues
        for i, venue in enumerate(unenriched_venues, 1):
            try:
                logger.info(f"[{i}/{len(unenriched_venues)}] Enriching venue: {venue.name} (id={venue.id})")
                result = await orchestrator.enrich_entity(
                    entity_type="venue",
                    source_id=venue.id,
                    name=venue.name,
                    category=venue.category,
                    description=venue.description,
                    website=venue.website,
                    instagram=venue.instagram,
                    city=venue.city,
                    country=venue.country,
                    address=venue.address,
                    latitude=venue.latitude,
                    longitude=venue.longitude
                )

                if result.success:
                    enriched += 1
                    logger.info(f"  ✓ Success (confidence: {result.overall_confidence:.2f})")
                else:
                    failed += 1
                    logger.warning(f"  ✗ Failed: {result.error}")

            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Exception: {e}")

        logger.info(f"\n=== ENRICHMENT COMPLETE ===")
        logger.info(f"Total processed: {total}")
        logger.info(f"Enriched: {enriched}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {enriched/total*100 if total else 0:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
