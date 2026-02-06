"""Temporal.io workflow for synchronizing data across the Knowledge Trinity.

Ensures PostgreSQL, Qdrant, and Neo4j stay consistent when enrichment data
changes (e.g. new vibe_dna, tags, or manual curator edits).
"""

import logging
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_unsynced_enrichments(batch_size: int = 100) -> list[dict]:
    """Fetch enrichments that have been updated but not yet synced to Qdrant/Neo4j."""
    from sqlalchemy import or_, select

    from app.db.postgres.connection import (
        arbor_session_factory,
        magazine_session_factory,
    )
    from app.db.postgres.models import ArborEnrichment, Brand, Venue

    # 1. Fetch from Arbor DB
    async with arbor_session_factory() as arbor_session:
        # Enrichments not yet synced to Neo4j, or recently updated
        result = await arbor_session.execute(
            select(ArborEnrichment)
            .where(ArborEnrichment.neo4j_synced == False)  # noqa: E712
            .limit(batch_size)
        )
        enrichments = list(result.scalars().all())

    if not enrichments:
        return []

    items = []

    # 2. Fetch details from Magazine DB
    async with magazine_session_factory() as mag_session:
        for enr in enrichments:
            # Resolve entity name and category from source table
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

            if name is None:
                continue

            items.append(
                {
                    "enrichment_id": str(enr.id),
                    "entity_type": enr.entity_type,
                    "source_id": enr.source_id,
                    "composite_id": f"{enr.entity_type}_{enr.source_id}",
                    "name": name,
                    "category": category or "",
                    "city": city,
                    "vibe_dna": enr.vibe_dna or {},
                    "tags": enr.tags or [],
                }
            )

    return items


@activity.defn
async def sync_entity_to_qdrant(entity: dict) -> bool:
    """Generate embedding and upsert vector to Qdrant."""
    from app.db.qdrant.collections import QdrantCollections
    from app.ingestion.analyzers.embedding import EmbeddingGenerator

    generator = EmbeddingGenerator()
    qdrant = QdrantCollections()

    vibe_dna = entity.get("vibe_dna", {})
    tags = entity.get("tags", [])

    embedding_text = (
        f"{entity['name']} | {entity['category']} | "
        f"{' '.join(tags)} | "
        f"{vibe_dna.get('summary', '')}"
    )
    embedding = await generator.generate(embedding_text)

    payload = {
        "entity_id": entity["composite_id"],
        "name": entity["name"],
        "category": entity["category"],
        "city": entity.get("city"),
        "dimensions": vibe_dna.get("dimensions", {}),
        "tags": tags,
        "status": "synced",
    }
    qdrant.upsert_vector(entity["composite_id"], embedding, payload)
    return True


@activity.defn
async def sync_entity_to_neo4j(entity: dict) -> bool:
    """Create or update entity node and style relationships in Neo4j."""
    from app.db.neo4j.queries import Neo4jQueries

    neo4j = Neo4jQueries()

    await neo4j.create_entity_node(
        entity_id=entity["composite_id"],
        name=entity["name"],
        category=entity["category"],
        city=entity.get("city"),
    )

    # Sync style relationships from tags
    vibe_dna = entity.get("vibe_dna", {})
    for tag in vibe_dna.get("tags", [])[:5]:
        await neo4j.create_style_node(tag)
        await neo4j.create_style_relationship(entity["composite_id"], tag)

    return True


@activity.defn
async def mark_enrichments_synced(enrichment_ids: list[str]) -> int:
    """Mark enrichments as synced in PostgreSQL."""
    from uuid import UUID

    from sqlalchemy import update

    from app.db.postgres.connection import arbor_session_factory
    from app.db.postgres.models import ArborEnrichment

    if not enrichment_ids:
        return 0

    async with arbor_session_factory() as session:
        uuids = [UUID(eid) for eid in enrichment_ids]
        result = await session.execute(
            update(ArborEnrichment).where(ArborEnrichment.id.in_(uuids)).values(neo4j_synced=True)
        )
        await session.commit()
        return result.rowcount


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


@workflow.defn
class SyncWorkflow:
    """Synchronize enrichment data from PostgreSQL to Qdrant and Neo4j.

    Designed to run on a schedule (e.g. every 5 minutes) or be triggered
    after a batch of enrichments is created/updated.

    Steps:
    1. Fetch unsynced enrichments from PostgreSQL
    2. For each entity: generate embedding + upsert to Qdrant
    3. For each entity: create/update node + relationships in Neo4j
    4. Mark enrichments as synced
    """

    @workflow.run
    async def run(self, batch_size: int = 100) -> dict:
        retry = RetryPolicy(maximum_attempts=3)

        # Step 1: Fetch unsynced enrichments
        entities = await workflow.execute_activity(
            fetch_unsynced_enrichments,
            args=[batch_size],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry,
        )

        if not entities:
            return {"synced": 0, "errors": 0}

        synced_ids = []
        errors = 0

        for entity in entities:
            try:
                # Step 2: Sync to Qdrant
                await workflow.execute_activity(
                    sync_entity_to_qdrant,
                    args=[entity],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=retry,
                )

                # Step 3: Sync to Neo4j
                await workflow.execute_activity(
                    sync_entity_to_neo4j,
                    args=[entity],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=retry,
                )

                synced_ids.append(entity["enrichment_id"])

            except Exception as e:
                workflow.logger.error(f"Failed to sync {entity['composite_id']}: {e}")
                errors += 1

        # Step 4: Mark as synced
        if synced_ids:
            marked = await workflow.execute_activity(
                mark_enrichments_synced,
                args=[synced_ids],
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry,
            )
        else:
            marked = 0

        return {"synced": marked, "errors": errors}


@workflow.defn
class FullResyncWorkflow:
    """Re-synchronize ALL enrichments to Qdrant and Neo4j.

    Useful after schema changes, embedding model upgrades,
    or data recovery scenarios.
    """

    @workflow.run
    async def run(self, batch_size: int = 50) -> dict:
        total_synced = 0
        total_errors = 0
        batch_count = 0

        while True:
            result = await workflow.execute_child_workflow(
                SyncWorkflow.run,
                args=[batch_size],
                id=f"resync-batch-{batch_count}",
            )

            total_synced += result["synced"]
            total_errors += result["errors"]
            batch_count += 1

            # Stop when no more unsynced items
            if result["synced"] == 0 and result["errors"] == 0:
                break

            # Safety: cap at 100 batches
            if batch_count >= 100:
                workflow.logger.warning("Hit max batch count, stopping resync")
                break

        return {
            "total_synced": total_synced,
            "total_errors": total_errors,
            "batches": batch_count,
        }
