"""Force sync ALL enrichments to Qdrant (ignore neo4j_synced flag)."""

import asyncio
import logging
import sys
import os
import hashlib
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from app.db.postgres.connection import arbor_session_factory, magazine_session_factory
from app.db.postgres.models import ArborEnrichment
from app.db.postgres.entity_resolver import resolve_entity_fields_batch
from app.db.postgres.connection import magazine_engine
from app.db.qdrant.collections import QdrantCollections
from app.ingestion.analyzers.embedding import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def composite_id_to_uuid(composite_id: str) -> str:
    """Convert composite_id like 'brand_12' to a stable UUID."""
    hash_bytes = hashlib.md5(composite_id.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


async def main():
    logger.info("Force syncing ALL enrichments to Qdrant...")

    # 1. Fetch ALL enrichments (ignore neo4j_synced flag)
    logger.info("Fetching all enrichments...")
    async with arbor_session_factory() as arbor_session:
        result = await arbor_session.execute(select(ArborEnrichment))
        enrichments = list(result.scalars().all())

    logger.info(f"Found {len(enrichments)} total enrichments")

    if not enrichments:
        logger.info("Nothing to sync!")
        return

    # 2. Prepare entity data using schema-agnostic resolver
    entities = []
    async with magazine_session_factory() as mag_session:
        # Group enrichments by entity_type for batch resolution
        by_type: dict[str, list] = {}
        for enr in enrichments:
            by_type.setdefault(enr.entity_type, []).append(enr)

        for etype, enrs in by_type.items():
            source_ids = [e.source_id for e in enrs]
            fields_map = await resolve_entity_fields_batch(
                mag_session, etype, source_ids,
                ["name", "category", "city"],
                engine=magazine_engine,
            )
            for enr in enrs:
                f = fields_map.get(enr.source_id, {})
                name = f.get("name")
                if name:
                    entities.append({
                        "composite_id": f"{enr.entity_type}_{enr.source_id}",
                        "name": name,
                        "category": f.get("category") or "",
                        "city": f.get("city"),
                        "vibe_dna": enr.vibe_dna or {},
                        "tags": enr.tags or [],
                    })

    logger.info(f"Prepared {len(entities)} entities for Qdrant sync")

    # 3. Initialize clients
    generator = EmbeddingGenerator()
    qdrant = QdrantCollections()

    synced = 0
    failed = 0

    # 4. Sync each entity to Qdrant only
    for i, entity in enumerate(entities, 1):
        try:
            logger.info(f"[{i}/{len(entities)}] Syncing {entity['name']} ({entity['composite_id']}) to Qdrant...")

            # Generate embedding
            vibe_dna = entity.get("vibe_dna", {})
            tags = entity.get("tags", [])

            embedding_text = (
                f"{entity['name']} | {entity['category']} | "
                f"{' '.join(tags)} | "
                f"{vibe_dna.get('summary', '')}"
            )
            embedding = await generator.generate(embedding_text)

            # Upsert to Qdrant (use UUID for point ID)
            point_id = composite_id_to_uuid(entity["composite_id"])
            payload = {
                "entity_id": entity["composite_id"],
                "name": entity["name"],
                "category": entity["category"],
                "city": entity.get("city"),
                "dimensions": vibe_dna.get("dimensions", {}),
                "tags": tags,
                "status": "synced",
            }
            qdrant.upsert_vector(point_id, embedding, payload)
            logger.info(f"  -> Qdrant OK")

            synced += 1

        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            failed += 1

    logger.info(f"\n=== QDRANT SYNC COMPLETE ===")
    logger.info(f"Synced: {synced}/{len(entities)}")
    logger.info(f"Failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
