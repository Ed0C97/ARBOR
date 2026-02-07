"""Script to run batch enrichment on unenriched entities."""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.repository import EnrichmentRepository
from app.ingestion.pipeline.enrichment_orchestrator import EnrichmentOrchestrator
from sqlalchemy import select, func
from app.db.postgres.models import ArborEnrichment
from app.config import get_settings
from app.db.postgres.dynamic_model import create_dynamic_model
from app.db.postgres.connection import magazine_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run batch enrichment for unenriched entities."""
    max_entities = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    entity_type = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info(f"Starting batch enrichment (max={max_entities}, type={entity_type})...")

    async with magazine_session_factory() as main_session, arbor_session_factory() as arbor_session:
        # Find entities without enrichment
        logger.info("Finding entities without enrichment...")

        # Get all entity IDs from main DB using dynamic models
        from sqlalchemy import MetaData
        settings = get_settings()
        configs = settings.get_entity_type_configs()
        target_configs = configs
        if entity_type:
            target_configs = [c for c in configs if c.entity_type == entity_type]

        _metadata = MetaData()
        all_entities = []

        for config in target_configs:
            table = await create_dynamic_model(config, magazine_engine, _metadata)
            id_col = table.c[config.id_column]
            # Build select with optional name column
            cols = [id_col]
            name_col_name = config.required_mappings.get("name") or config.all_mappings.get("name")
            if name_col_name and name_col_name in table.c:
                cols.append(table.c[name_col_name])

            stmt = select(*cols).limit(max_entities * 2)
            # Apply active filter if configured
            if config.active_filter_column and config.active_filter_column in table.c:
                stmt = stmt.where(
                    table.c[config.active_filter_column] == config.active_filter_value
                )

            result = await main_session.execute(stmt)
            for row in result:
                all_entities.append({
                    "entity_type": config.entity_type,
                    "source_id": row[0],
                    "name": row[1] if len(cols) > 1 else f"{config.entity_type} #{row[0]}",
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
