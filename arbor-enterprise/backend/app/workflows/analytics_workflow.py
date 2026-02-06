"""Temporal.io workflow for periodic analytics aggregation.

Runs on a schedule (e.g. hourly / daily) to compute aggregate metrics
from search logs and feedback, storing results for the admin dashboard.
"""

import logging
from datetime import datetime, timedelta, timezone

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@activity.defn
async def compute_search_analytics(since_hours: int = 24) -> dict:
    """Aggregate search query analytics from feedback and search logs."""
    from sqlalchemy import func, select, text

    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import ArborFeedback

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    async with async_session_factory() as session:
        # Total searches (feedback entries as proxy)
        total_result = await session.execute(
            select(func.count())
            .select_from(ArborFeedback)
            .where(ArborFeedback.created_at >= cutoff)
        )
        total_interactions = total_result.scalar_one()

        # Top clicked entities
        top_entities_result = await session.execute(
            select(
                ArborFeedback.entity_type,
                ArborFeedback.source_id,
                func.count().label("click_count"),
            )
            .where(
                ArborFeedback.created_at >= cutoff,
                ArborFeedback.action == "click",
            )
            .group_by(ArborFeedback.entity_type, ArborFeedback.source_id)
            .order_by(func.count().desc())
            .limit(20)
        )
        top_entities = [
            {
                "entity_id": f"{row.entity_type}_{row.source_id}",
                "click_count": row.click_count,
            }
            for row in top_entities_result
        ]

        # Top queries
        top_queries_result = await session.execute(
            select(
                ArborFeedback.query,
                func.count().label("query_count"),
            )
            .where(
                ArborFeedback.created_at >= cutoff,
                ArborFeedback.query.isnot(None),
            )
            .group_by(ArborFeedback.query)
            .order_by(func.count().desc())
            .limit(20)
        )
        top_queries = [{"query": row.query, "count": row.query_count} for row in top_queries_result]

        # Conversion rate
        conversions_result = await session.execute(
            select(func.count())
            .select_from(ArborFeedback)
            .where(
                ArborFeedback.created_at >= cutoff,
                ArborFeedback.action == "convert",
            )
        )
        total_conversions = conversions_result.scalar_one()

        clicks_result = await session.execute(
            select(func.count())
            .select_from(ArborFeedback)
            .where(
                ArborFeedback.created_at >= cutoff,
                ArborFeedback.action == "click",
            )
        )
        total_clicks = clicks_result.scalar_one()

        conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0.0

        # Average reward (proxy for result quality)
        avg_reward_result = await session.execute(
            select(func.avg(ArborFeedback.reward)).where(
                ArborFeedback.created_at >= cutoff,
                ArborFeedback.reward.isnot(None),
            )
        )
        avg_reward = avg_reward_result.scalar_one() or 0.0

        return {
            "period_hours": since_hours,
            "total_interactions": total_interactions,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "conversion_rate": round(conversion_rate, 4),
            "avg_reward": round(float(avg_reward), 4),
            "top_entities": top_entities,
            "top_queries": top_queries,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }


@activity.defn
async def compute_entity_stats() -> dict:
    """Compute entity coverage and enrichment statistics."""
    from sqlalchemy import func, select

    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import ArborEnrichment, Brand, Venue

    async with async_session_factory() as session:
        brands_count = (await session.execute(select(func.count()).select_from(Brand))).scalar_one()

        venues_count = (await session.execute(select(func.count()).select_from(Venue))).scalar_one()

        enriched_count = (
            await session.execute(select(func.count()).select_from(ArborEnrichment))
        ).scalar_one()

        synced_count = (
            await session.execute(
                select(func.count())
                .select_from(ArborEnrichment)
                .where(ArborEnrichment.neo4j_synced == True)  # noqa: E712
            )
        ).scalar_one()

        total_entities = brands_count + venues_count
        enrichment_coverage = enriched_count / total_entities if total_entities > 0 else 0.0

        # Enrichments by entity type
        by_type_result = await session.execute(
            select(
                ArborEnrichment.entity_type,
                func.count().label("count"),
            ).group_by(ArborEnrichment.entity_type)
        )
        by_type = {row.entity_type: row.count for row in by_type_result}

        return {
            "total_entities": total_entities,
            "total_brands": brands_count,
            "total_venues": venues_count,
            "enriched_entities": enriched_count,
            "synced_entities": synced_count,
            "enrichment_coverage": round(enrichment_coverage, 4),
            "enriched_by_type": by_type,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }


@activity.defn
async def store_analytics_snapshot(snapshot: dict) -> str:
    """Store analytics snapshot to Redis for dashboard consumption."""
    import json

    from app.db.redis.client import get_redis_client

    redis = get_redis_client()
    key = f"arbor:analytics:{snapshot.get('snapshot_type', 'general')}"

    await redis.client.set(key, json.dumps(snapshot), ex=86400)  # TTL 24h
    return key


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


@workflow.defn
class AnalyticsWorkflow:
    """Periodic analytics aggregation workflow.

    Computes search analytics and entity stats, then stores results
    in Redis for fast retrieval by the admin dashboard.

    Designed to run hourly via Temporal schedule.
    """

    @workflow.run
    async def run(self, since_hours: int = 24) -> dict:
        retry = RetryPolicy(maximum_attempts=2)

        # Step 1: Compute search analytics
        search_analytics = await workflow.execute_activity(
            compute_search_analytics,
            args=[since_hours],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry,
        )

        # Step 2: Compute entity stats
        entity_stats = await workflow.execute_activity(
            compute_entity_stats,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry,
        )

        # Step 3: Store snapshots to Redis
        search_snapshot = {
            "snapshot_type": "search_analytics",
            **search_analytics,
        }
        entity_snapshot = {
            "snapshot_type": "entity_stats",
            **entity_stats,
        }

        search_key = await workflow.execute_activity(
            store_analytics_snapshot,
            args=[search_snapshot],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry,
        )

        entity_key = await workflow.execute_activity(
            store_analytics_snapshot,
            args=[entity_snapshot],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry,
        )

        return {
            "search_analytics": search_analytics,
            "entity_stats": entity_stats,
            "stored_keys": [search_key, entity_key],
        }


@workflow.defn
class DailyReportWorkflow:
    """Daily analytics report — computes 24h and 7d snapshots."""

    @workflow.run
    async def run(self) -> dict:
        # 24h snapshot
        daily = await workflow.execute_child_workflow(
            AnalyticsWorkflow.run,
            args=[24],
            id="analytics-daily-24h",
        )

        # 7d snapshot
        weekly = await workflow.execute_child_workflow(
            AnalyticsWorkflow.run,
            args=[168],
            id="analytics-daily-7d",
        )

        return {
            "daily": daily,
            "weekly": weekly,
        }
