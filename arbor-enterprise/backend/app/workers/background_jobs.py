"""Pre-computation Background Jobs using Celery.

TIER 7 - Point 34: Pre-computation Background Jobs

Handles:
- Daily embedding updates for stale entities
- Batch vector indexing
- Scheduled enrichment pipelines
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Celery app configuration
celery_app = Celery(
    "arbor_workers",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # Soft limit at 55 min
    worker_prefetch_multiplier=1,  # Fairness
    task_acks_late=True,  # Ensure task completion
    task_reject_on_worker_lost=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # TIER 7 - Point 34: Daily embedding refresh
    "refresh-stale-embeddings": {
        "task": "app.workers.background_jobs.refresh_stale_embeddings",
        "schedule": crontab(hour=3, minute=0),  # 3 AM daily
        "options": {"queue": "batch"},
    },
    # Weekly full reindex check
    "weekly-reindex-check": {
        "task": "app.workers.background_jobs.check_index_health",
        "schedule": crontab(day_of_week=0, hour=2, minute=0),  # Sunday 2 AM
        "options": {"queue": "batch"},
    },
    # Hourly cache warmup
    "cache-warmup": {
        "task": "app.workers.background_jobs.warmup_popular_queries",
        "schedule": crontab(minute=0),  # Every hour
        "options": {"queue": "default"},
    },
    # Clean up old audit logs
    "cleanup-audit-logs": {
        "task": "app.workers.background_jobs.cleanup_old_data",
        "schedule": crontab(hour=4, minute=0),  # 4 AM daily
        "options": {"queue": "maintenance"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Refresh Tasks
# ═══════════════════════════════════════════════════════════════════════════


@celery_app.task(bind=True, max_retries=3)
def refresh_stale_embeddings(self, max_age_hours: int = 24) -> dict:
    """Refresh embeddings for entities older than max_age_hours.

    TIER 7 - Point 34: No entity should have embeddings older than 24 hours.

    Args:
        max_age_hours: Maximum age for embeddings before refresh

    Returns:
        Summary of refreshed entities
    """
    try:
        # Run async code in sync context
        result = asyncio.get_event_loop().run_until_complete(
            _refresh_stale_embeddings_async(max_age_hours)
        )
        return result
    except Exception as exc:
        logger.error(f"Embedding refresh failed: {exc}")
        self.retry(exc=exc, countdown=300)  # Retry in 5 min


async def _refresh_stale_embeddings_async(max_age_hours: int) -> dict:
    """Async implementation of embedding refresh."""
    from sqlalchemy import select, update
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.postgres.connection import get_arbor_session
    from app.db.postgres.models import ArborEnrichment
    from app.db.qdrant.client import get_async_qdrant_client
    from app.llm.gateway import get_llm_gateway

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    refreshed_count = 0
    failed_count = 0

    async with get_arbor_session() as session:
        # Find stale entities
        query = (
            select(ArborEnrichment).where(ArborEnrichment.last_indexed < cutoff).limit(500)
        )  # Process in batches

        result = await session.execute(query)
        stale_entities = result.scalars().all()

        if not stale_entities:
            logger.info("No stale entities found")
            return {"refreshed": 0, "failed": 0}

        logger.info(f"Found {len(stale_entities)} stale entities")

        # Get LLM gateway for embeddings
        llm = get_llm_gateway()
        qdrant = await get_async_qdrant_client()

        # Process in batches of 96 (Cohere limit)
        batch_size = 96
        for i in range(0, len(stale_entities), batch_size):
            batch = stale_entities[i : i + batch_size]

            try:
                # Generate embeddings
                texts = [e.embedding_text or e.name for e in batch]
                embeddings = await llm.get_embeddings(texts)

                # Update Qdrant
                points = []
                for entity, embedding in zip(batch, embeddings):
                    points.append(
                        {
                            "id": entity.id,
                            "vector": embedding,
                            "payload": {
                                "entity_id": str(entity.entity_id),
                                "name": entity.name,
                                "category": entity.category,
                            },
                        }
                    )

                await qdrant.upsert(collection_name="entities_vectors", points=points)

                # Update last_indexed timestamp
                entity_ids = [e.id for e in batch]
                await session.execute(
                    update(ArborEnrichment)
                    .where(ArborEnrichment.id.in_(entity_ids))
                    .values(last_indexed=datetime.utcnow())
                )
                await session.commit()

                refreshed_count += len(batch)
                logger.info(f"Refreshed batch {i // batch_size + 1}")

            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                failed_count += len(batch)

    logger.info(f"Embedding refresh complete: {refreshed_count} refreshed, {failed_count} failed")
    return {"refreshed": refreshed_count, "failed": failed_count}


# ═══════════════════════════════════════════════════════════════════════════
# Cache Warmup Tasks
# ═══════════════════════════════════════════════════════════════════════════


@celery_app.task
def warmup_popular_queries() -> dict:
    """Pre-cache results for popular search queries."""
    return asyncio.get_event_loop().run_until_complete(_warmup_popular_queries_async())


async def _warmup_popular_queries_async() -> dict:
    """Async implementation of cache warmup."""
    # Popular queries to pre-cache
    popular_queries = [
        "aperitivo a Milano",
        "ristorante romantico",
        "brunch domenica",
        "cocktail bar centro",
        "pizza gourmet",
        "locali con musica live",
        "ristorante vista",
        "bar nascosti speakeasy",
    ]

    from app.agents.graph import get_discovery_graph

    graph = get_discovery_graph()
    warmed = 0

    for query in popular_queries:
        try:
            # Run discovery to populate cache
            await graph.ainvoke(
                {
                    "query": query,
                    "user_id": "cache_warmup",
                    "skip_cache_read": False,
                }
            )
            warmed += 1
        except Exception as e:
            logger.warning(f"Failed to warm cache for '{query}': {e}")

    logger.info(f"Cache warmup complete: {warmed}/{len(popular_queries)} queries")
    return {"warmed": warmed, "total": len(popular_queries)}


# ═══════════════════════════════════════════════════════════════════════════
# Maintenance Tasks
# ═══════════════════════════════════════════════════════════════════════════


@celery_app.task
def check_index_health() -> dict:
    """Check health of vector indexes."""
    return asyncio.get_event_loop().run_until_complete(_check_index_health_async())


async def _check_index_health_async() -> dict:
    """Async implementation of index health check."""
    from app.db.qdrant.client import get_async_qdrant_client

    client = await get_async_qdrant_client()

    collections = ["entities_vectors", "semantic_cache"]
    health = {}

    for collection in collections:
        try:
            info = await client.get_collection(collection)
            health[collection] = {
                "status": info.status.name,
                "points_count": info.points_count,
                "indexed_vectors": info.indexed_vectors_count,
                "segments": len(info.segments) if info.segments else 0,
            }
        except Exception as e:
            health[collection] = {"status": "ERROR", "error": str(e)}

    logger.info(f"Index health: {health}")
    return health


@celery_app.task
def cleanup_old_data() -> dict:
    """Run data retention cleanup."""
    return asyncio.get_event_loop().run_until_complete(_cleanup_old_data_async())


async def _cleanup_old_data_async() -> dict:
    """Async implementation of cleanup."""
    from app.compliance.data_retention import get_retention_enforcer

    enforcer = get_retention_enforcer()
    result = await enforcer.enforce_all_policies()

    logger.info(f"Cleanup complete: {result}")
    return result
