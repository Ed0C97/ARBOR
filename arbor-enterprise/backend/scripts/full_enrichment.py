"""Full enrichment of all unenriched entities."""

import asyncio
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.models import ArborEnrichment
from app.config import get_settings
from app.db.postgres.dynamic_model import create_dynamic_model
from app.db.postgres.generic_repository import GenericEntityRepository
from app.db.postgres.connection import magazine_engine
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

        from sqlalchemy import MetaData
        settings = get_settings()
        configs = settings.get_entity_type_configs()
        _metadata = MetaData()

        # Enrich all
        orchestrator = EnrichmentOrchestrator(arbor_session)

        total = 0
        enriched = 0
        failed = 0

        for config in configs:
            table = await create_dynamic_model(config, magazine_engine, _metadata)
            repo = GenericEntityRepository(main_session, config, table)
            entities, count = await repo.list_entities(is_active=True, offset=0, limit=9999)

            unenriched = [
                e for e in entities
                if (config.entity_type, e.source_id) not in existing_map
            ]
            logger.info(f"Need to enrich: {len(unenriched)} {config.entity_type} entities")
            total += len(unenriched)

            for i, entity in enumerate(unenriched, 1):
                try:
                    from dataclasses import asdict
                    logger.info(
                        f"[{i}/{len(unenriched)}] Enriching {config.entity_type}: "
                        f"{entity.name} (id={entity.source_id})"
                    )
                    entity_dict = asdict(entity)
                    valid_args = {
                        "name", "category", "city", "description", "specialty", "notes",
                        "website", "instagram", "style", "gender", "rating", "price_range",
                        "address", "latitude", "longitude", "country",
                    }
                    kwargs = {k: v for k, v in entity_dict.items() if k in valid_args and v is not None}
                    kwargs["entity_type"] = config.entity_type
                    kwargs["source_id"] = entity.source_id

                    result = await orchestrator.enrich_entity(**kwargs)

                    if result.success:
                        enriched += 1
                        logger.info(f"  -> Success (confidence: {result.overall_confidence:.2f})")
                    else:
                        failed += 1
                        logger.warning(f"  -> Failed: {result.error}")

                except Exception as e:
                    failed += 1
                    logger.error(f"  -> Exception: {e}")

        logger.info(f"\n=== ENRICHMENT COMPLETE ===")
        logger.info(f"Total processed: {total}")
        logger.info(f"Enriched: {enriched}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {enriched/total*100 if total else 0:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
