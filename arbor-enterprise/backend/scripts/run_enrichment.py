"""Script to run batch enrichment on unenriched entities."""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.repository import EnrichmentRepository, UnifiedEntityRepository
from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator
from sqlalchemy import select, func
from app.db.postgres.models import ArborEnrichment, Brand, Venue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run batch enrichment for unenriched entities."""
    max_entities = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    entity_type = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info(f"Starting batch enrichment (max={max_entities}, type={entity_type})...")

    async with magazine_session_factory() as main_session, arbor_session_factory() as arbor_session:
        # Get unified repository
        unified_repo = UnifiedEntityRepository(main_session, arbor_session)

        # Find entities without enrichment
        logger.info("Finding entities without enrichment...")

        # Get all entity IDs from main DB
        all_entities = []

        # Get brands
        if entity_type is None or entity_type == "brand":
            result = await main_session.execute(
                select(Brand.id, Brand.name)
                .where(Brand.is_active == True)
                .limit(max_entities * 2)
            )
            brands = result.all()
            for brand_id, name in brands:
                all_entities.append({
                    "entity_type": "brand",
                    "source_id": brand_id,
                    "name": name
                })

        # Get venues
        if entity_type is None or entity_type == "venue":
            result = await main_session.execute(
                select(Venue.id, Venue.name)
                .where(Venue.is_active == True)
                .limit(max_entities * 2)
            )
            venues = result.all()
            for venue_id, name in venues:
                all_entities.append({
                    "entity_type": "venue",
                    "source_id": venue_id,
                    "name": name
                })

        # Filter out already enriched
        enrichment_repo = EnrichmentRepository(arbor_session)
        unenriched = []

        for entity in all_entities:
            enr = await enrichment_repo.get(
                entity["entity_type"],
                entity["source_id"]
            )
            if enr is None:
                unenriched.append(entity)
                if len(unenriched) >= max_entities:
                    break

        logger.info(f"Found {len(unenriched)} unenriched entities")

        if not unenriched:
            logger.info("No entities to enrich.")
            return

        # Enrich each entity
        orchestrator = EnrichmentOrchestrator(main_session, arbor_session)

        enriched_count = 0
        failed_count = 0

        for entity in unenriched:
            try:
                logger.info(f"Enriching {entity['entity_type']}_{entity['source_id']}: {entity['name']}...")

                result = await orchestrator.enrich_entity(
                    entity_type=entity["entity_type"],
                    source_id=entity["source_id"]
                )

                if result and result.get("success"):
                    enriched_count += 1
                    logger.info(f"  ✓ Success (confidence: {result.get('overall_confidence', 0):.2f})")
                else:
                    failed_count += 1
                    logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                failed_count += 1
                logger.error(f"  ✗ Exception: {e}")

        logger.info(f"\nBatch enrichment complete:")
        logger.info(f"  - Processed: {len(unenriched)}")
        logger.info(f"  - Succeeded: {enriched_count}")
        logger.info(f"  - Failed: {failed_count}")

if __name__ == "__main__":
    asyncio.run(main())
