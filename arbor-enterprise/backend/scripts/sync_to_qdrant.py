"""Sync enriched entities to Qdrant and Neo4j (without Temporal)."""

import asyncio
import logging
import sys
import os
import hashlib
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from app.db.postgres.connection import arbor_session_factory, magazine_session_factory
from app.db.postgres.models import ArborEnrichment, Brand, Venue
from app.db.qdrant.collections import QdrantCollections
from app.ingestion.analyzers.embedding import EmbeddingGenerator
from app.db.neo4j.queries import Neo4jQueries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def composite_id_to_uuid(composite_id: str) -> str:
    """Convert composite_id like 'brand_12' to a stable UUID."""
    # Create a stable UUID from the composite_id
    hash_bytes = hashlib.md5(composite_id.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


async def main():
    logger.info("Starting manual sync to Qdrant and Neo4j...")

    # 1. Fetch unsynced enrichments
    logger.info("Fetching unsynced enrichments...")
    async with arbor_session_factory() as arbor_session:
        result = await arbor_session.execute(
            select(ArborEnrichment).where(ArborEnrichment.neo4j_synced == False)
        )
        enrichments = list(result.scalars().all())

    logger.info(f"Found {len(enrichments)} unsynced enrichments")

    if not enrichments:
        logger.info("Nothing to sync!")
        return

    # 2. Prepare entity data
    entities = []
    async with magazine_session_factory() as mag_session:
        for enr in enrichments:
            name, category, city = None, None, None

            if enr.entity_type == "brand":
                brand = await mag_session.get(Brand, enr.source_id)
                if brand:
                    name = brand.name
                    category = brand.category
                    city = brand.area or brand.neighborhood
            elif enr.entity_type == "venue":
                venue = await mag_session.get(Venue, enr.source_id)
                if venue:
                    name = venue.name
                    category = venue.category
                    city = venue.city

            if name:
                entities.append({
                    "enrichment_id": str(enr.id),
                    "entity_type": enr.entity_type,
                    "source_id": enr.source_id,
                    "composite_id": f"{enr.entity_type}_{enr.source_id}",
                    "name": name,
                    "category": category or "",
                    "city": city,
                    "vibe_dna": enr.vibe_dna or {},
                    "tags": enr.tags or [],
                })

    logger.info(f"Prepared {len(entities)} entities for sync")

    # 3. Initialize clients
    generator = EmbeddingGenerator()
    qdrant = QdrantCollections()
    neo4j = Neo4jQueries()

    synced_ids = []

    # 4. Sync each entity
    for i, entity in enumerate(entities, 1):
        try:
            logger.info(f"[{i}/{len(entities)}] Syncing {entity['name']} ({entity['composite_id']})...")

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

            # Sync to Neo4j
            await neo4j.create_entity_node(
                entity_id=entity["composite_id"],
                name=entity["name"],
                category=entity["category"],
                city=entity.get("city"),
            )

            # Create style relationships
            for tag in vibe_dna.get("tags", [])[:5]:
                await neo4j.create_style_node(tag)
                await neo4j.create_style_relationship(entity["composite_id"], tag)

            logger.info(f"  -> Neo4j OK")

            synced_ids.append(entity["enrichment_id"])

        except Exception as e:
            logger.error(f"  -> Failed: {e}")

    # 5. Mark as synced
    if synced_ids:
        logger.info(f"Marking {len(synced_ids)} enrichments as synced...")
        async with arbor_session_factory() as session:
            from sqlalchemy import update
            from uuid import UUID

            uuids = [UUID(eid) for eid in synced_ids]
            await session.execute(
                update(ArborEnrichment)
                .where(ArborEnrichment.id.in_(uuids))
                .values(neo4j_synced=True)
            )
            await session.commit()
        logger.info(f"Marked {len(synced_ids)} enrichments as synced")

    logger.info(f"\n=== SYNC COMPLETE ===")
    logger.info(f"Synced: {len(synced_ids)}/{len(entities)}")


if __name__ == "__main__":
    asyncio.run(main())
