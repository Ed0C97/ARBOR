"""Temporal.io durable workflow for entity ingestion."""

import uuid
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.workflows.activities import (
        analyze_with_vision,
        extract_vibe,
        generate_embedding,
        save_to_neo4j,
        save_to_qdrant,
        scrape_entity,
    )


@workflow.defn
class EntityIngestionWorkflow:
    """Durable workflow for ingesting a single entity.

    Benefits:
    - Resumes from last step on failure
    - Configurable retry per step
    - Full visibility into state
    - Timeout per step
    """

    @workflow.run
    async def run(self, source_url: str, category: str = "") -> str:
        entity_id = str(uuid.uuid4())
        retry = RetryPolicy(maximum_attempts=3)

        # Step 1: Scrape raw data
        raw_data = await workflow.execute_activity(
            scrape_entity,
            args=[source_url, "google_maps"],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )

        # Step 2: Parallel analysis - Vision + Reviews
        vision_task = workflow.execute_activity(
            analyze_with_vision,
            args=[raw_data.get("images", [])],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry,
        )
        vibe_task = workflow.execute_activity(
            extract_vibe,
            args=[raw_data.get("reviews", []), raw_data["name"]],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )

        vision_result, vibe_result = await vision_task, await vibe_task

        # Step 3: Generate embedding
        embedding_text = (
            f"{raw_data['name']} | {category} | "
            f"{' '.join(vibe_result.get('tags', []))} | "
            f"{vibe_result.get('summary', '')}"
        )
        embedding = await workflow.execute_activity(
            generate_embedding,
            args=[embedding_text],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry,
        )

        # Step 4: Save to Qdrant
        payload = {
            "entity_id": entity_id,
            "name": raw_data["name"],
            "category": category,
            "dimensions": vibe_result.get("dimensions", {}),
            "tags": vibe_result.get("tags", []),
            "status": "pending",
        }
        await workflow.execute_activity(
            save_to_qdrant,
            args=[entity_id, embedding, payload],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry,
        )

        # Step 5: Save to Neo4j
        await workflow.execute_activity(
            save_to_neo4j,
            args=[entity_id, raw_data["name"], category, None],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry,
        )

        return entity_id


@workflow.defn
class BatchIngestionWorkflow:
    """Workflow for batch ingestion of multiple entities."""

    @workflow.run
    async def run(self, source_urls: list[str], category: str = "") -> list[str]:
        entity_ids = []

        for url in source_urls:
            try:
                entity_id = await workflow.execute_child_workflow(
                    EntityIngestionWorkflow.run,
                    args=[url, category],
                    id=f"ingest-{hash(url)}",
                )
                entity_ids.append(entity_id)
            except Exception as e:
                workflow.logger.error(f"Failed to ingest {url}: {e}")

        return entity_ids
