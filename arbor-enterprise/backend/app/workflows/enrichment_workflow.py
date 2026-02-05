"""Temporal.io durable workflow for the 5-layer enrichment pipeline.

Orchestrates the complete enrichment of entities from the database
using the enterprise pipeline: collect → analyze → score → validate → persist.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.workflows.enrichment_activities import (
        collect_entity_data,
        run_fact_analyzers,
        score_with_calibration,
        validate_and_persist,
        run_full_enrichment,
        get_unenriched_entities,
        generate_and_sync_embedding,
    )


@workflow.defn
class EntityEnrichmentWorkflow:
    """Durable workflow for enriching a single entity through the 5-layer pipeline.

    Each step is an activity with independent retry and timeout.
    On failure, the workflow resumes from the last completed step.
    """

    @workflow.run
    async def run(self, entity_type: str, source_id: int) -> dict:
        """Run the full enrichment pipeline for a single entity.

        Returns a summary dict with success status, confidence, and review flag.
        """
        entity_id = f"{entity_type}_{source_id}"
        retry = RetryPolicy(maximum_attempts=3)

        # Use the single-activity approach for simplicity and atomicity
        result = await workflow.execute_activity(
            run_full_enrichment,
            args=[entity_type, source_id],
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # If enrichment succeeded and produced an embedding-worthy result,
        # generate embedding and sync to vector DB + graph
        if result.get("success") and not result.get("error"):
            try:
                await workflow.execute_activity(
                    generate_and_sync_embedding,
                    args=[entity_type, source_id],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=retry,
                )
                result["embedding_synced"] = True
            except Exception as e:
                workflow.logger.warning(
                    f"Embedding sync failed for {entity_id}: {e}"
                )
                result["embedding_synced"] = False

        return result


@workflow.defn
class SteppedEnrichmentWorkflow:
    """Alternative workflow with explicit per-step activities.

    Use this when you need fine-grained control over each pipeline stage,
    e.g. for debugging or when running on resource-constrained workers.
    """

    @workflow.run
    async def run(self, entity_type: str, source_id: int) -> dict:
        entity_id = f"{entity_type}_{source_id}"
        retry = RetryPolicy(maximum_attempts=3)

        # Step 1: Collect data from all sources
        collected_data = await workflow.execute_activity(
            collect_entity_data,
            args=[entity_type, source_id],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )

        if not collected_data or collected_data.get("error"):
            return {
                "entity_id": entity_id,
                "success": False,
                "error": collected_data.get("error", "Collection failed"),
            }

        # Step 2: Run fact analyzers in parallel (within the activity)
        fact_sheet_data = await workflow.execute_activity(
            run_fact_analyzers,
            args=[collected_data],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry,
        )

        # Step 3: Score with calibration
        scored_data = await workflow.execute_activity(
            score_with_calibration,
            args=[fact_sheet_data, entity_type, source_id],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )

        # Step 4: Validate, persist, and create review items
        persist_result = await workflow.execute_activity(
            validate_and_persist,
            args=[scored_data, fact_sheet_data, entity_type, source_id],
            start_to_close_timeout=timedelta(minutes=3),
            retry_policy=retry,
        )

        # Step 5: Generate embedding and sync
        try:
            await workflow.execute_activity(
                generate_and_sync_embedding,
                args=[entity_type, source_id],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry,
            )
            persist_result["embedding_synced"] = True
        except Exception:
            persist_result["embedding_synced"] = False

        return persist_result


@workflow.defn
class BatchEnrichmentWorkflow:
    """Workflow for enriching all unenriched entities in the database.

    Fetches entities missing enrichment data and processes them sequentially
    (or in configurable-concurrency batches).
    """

    @workflow.run
    async def run(self, max_entities: int = 100, entity_type: str | None = None) -> dict:
        retry = RetryPolicy(maximum_attempts=2)

        # Get list of unenriched entities
        entities = await workflow.execute_activity(
            get_unenriched_entities,
            args=[max_entities, entity_type],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry,
        )

        if not entities:
            return {"total": 0, "enriched": 0, "failed": 0, "message": "No entities to enrich"}

        enriched = 0
        failed = 0
        results = []

        for entity in entities:
            try:
                result = await workflow.execute_child_workflow(
                    EntityEnrichmentWorkflow.run,
                    args=[entity["entity_type"], entity["source_id"]],
                    id=f"enrich-{entity['entity_type']}-{entity['source_id']}",
                )
                if result.get("success"):
                    enriched += 1
                else:
                    failed += 1
                results.append(result)
            except Exception as e:
                failed += 1
                workflow.logger.error(
                    f"Failed to enrich {entity['entity_type']}_{entity['source_id']}: {e}"
                )

        return {
            "total": len(entities),
            "enriched": enriched,
            "failed": failed,
            "results": results[:20],  # Limit result payload
        }
