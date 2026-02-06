"""Data Retention Policy Enforcer.

TIER 13 - Point 71: Data Retention Policy Enforcer

Automated cleanup of old data based on configurable retention policies.
Keeps storage costs stable over time.

Features:
- Configurable retention periods per data type
- Safe deletion with audit logging
- Metrics for monitoring cleanup
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class RetentionPeriod(Enum):
    """Standard retention periods."""

    DAYS_7 = 7
    DAYS_30 = 30
    DAYS_90 = 90
    DAYS_180 = 180
    DAYS_365 = 365
    DAYS_730 = 730  # 2 years


@dataclass
class RetentionPolicy:
    """Definition of a retention policy for a data type."""

    name: str
    table_name: str
    timestamp_column: str
    retention_days: int
    description: str
    soft_delete: bool = False  # If True, update is_deleted instead of DELETE


@dataclass
class CleanupResult:
    """Result of a cleanup job."""

    policy_name: str
    records_deleted: int
    space_freed_estimate_mb: float
    duration_seconds: float
    errors: list[str]


# Default retention policies
DEFAULT_POLICIES = [
    RetentionPolicy(
        name="search_logs",
        table_name="search_logs",
        timestamp_column="created_at",
        retention_days=90,
        description="Search query logs older than 90 days",
    ),
    RetentionPolicy(
        name="api_access_logs",
        table_name="api_access_logs",
        timestamp_column="timestamp",
        retention_days=30,
        description="API access logs older than 30 days",
    ),
    RetentionPolicy(
        name="session_data",
        table_name="user_sessions",
        timestamp_column="last_activity",
        retention_days=7,
        description="Inactive sessions older than 7 days",
    ),
    RetentionPolicy(
        name="temporary_uploads",
        table_name="temporary_files",
        timestamp_column="created_at",
        retention_days=1,
        description="Temporary files older than 24 hours",
    ),
    RetentionPolicy(
        name="semantic_cache",
        table_name="semantic_cache_entries",
        timestamp_column="cached_at",
        retention_days=7,
        description="Cached LLM responses older than 7 days",
    ),
    RetentionPolicy(
        name="dlq_messages",
        table_name="dlq_archive",
        timestamp_column="failed_at",
        retention_days=30,
        description="Dead letter queue messages older than 30 days",
    ),
]


class DataRetentionEnforcer:
    """Enforces data retention policies.

    TIER 13 - Point 71: Automated data cleanup.

    Usage:
        enforcer = DataRetentionEnforcer(session)
        results = await enforcer.run_cleanup()

    Schedule this as a daily cron job.
    """

    def __init__(
        self,
        session: AsyncSession,
        policies: list[RetentionPolicy] | None = None,
    ):
        self.session = session
        self.policies = policies or DEFAULT_POLICIES

    async def run_cleanup(
        self,
        dry_run: bool = False,
    ) -> list[CleanupResult]:
        """Run cleanup for all policies.

        Args:
            dry_run: If True, only report what would be deleted without deleting.

        Returns:
            List of cleanup results for each policy.
        """
        results = []

        for policy in self.policies:
            result = await self._enforce_policy(policy, dry_run=dry_run)
            results.append(result)

            if result.records_deleted > 0:
                logger.info(
                    f"Retention: {policy.name} - "
                    f"{'would delete' if dry_run else 'deleted'} "
                    f"{result.records_deleted} records"
                )

        return results

    async def _enforce_policy(
        self,
        policy: RetentionPolicy,
        dry_run: bool = False,
    ) -> CleanupResult:
        """Enforce a single retention policy."""
        import time

        start_time = time.time()
        errors = []
        records_deleted = 0

        cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)

        try:
            # Count records to delete
            count_stmt = text(
                f"SELECT COUNT(*) FROM {policy.table_name} "
                f"WHERE {policy.timestamp_column} < :cutoff"
            )
            result = await self.session.execute(count_stmt, {"cutoff": cutoff_date})
            records_to_delete = result.scalar() or 0

            if not dry_run and records_to_delete > 0:
                # Delete in batches to avoid long locks
                batch_size = 1000
                total_deleted = 0

                while total_deleted < records_to_delete:
                    if policy.soft_delete:
                        delete_stmt = text(
                            f"UPDATE {policy.table_name} "
                            f"SET is_deleted = TRUE "
                            f"WHERE {policy.timestamp_column} < :cutoff "
                            f"AND is_deleted = FALSE "
                            f"LIMIT {batch_size}"
                        )
                    else:
                        delete_stmt = text(
                            f"DELETE FROM {policy.table_name} "
                            f"WHERE ctid IN ("
                            f"  SELECT ctid FROM {policy.table_name} "
                            f"  WHERE {policy.timestamp_column} < :cutoff "
                            f"  LIMIT {batch_size}"
                            f")"
                        )

                    result = await self.session.execute(delete_stmt, {"cutoff": cutoff_date})
                    batch_deleted = result.rowcount
                    total_deleted += batch_deleted

                    if batch_deleted < batch_size:
                        break

                await self.session.commit()
                records_deleted = total_deleted
            else:
                records_deleted = records_to_delete

        except Exception as e:
            errors.append(f"{policy.table_name}: {str(e)}")
            logger.warning(f"Retention policy {policy.name} failed: {e}")

        duration = time.time() - start_time

        # Estimate space freed (rough: 1KB per record average)
        space_freed = (records_deleted * 1) / 1024  # MB

        return CleanupResult(
            policy_name=policy.name,
            records_deleted=records_deleted,
            space_freed_estimate_mb=space_freed,
            duration_seconds=round(duration, 2),
            errors=errors,
        )

    async def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a new retention policy."""
        self.policies.append(policy)

    async def get_retention_stats(self) -> dict[str, Any]:
        """Get statistics about data eligible for deletion."""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "policies": [],
        }

        for policy in self.policies:
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)

            try:
                count_stmt = text(
                    f"SELECT COUNT(*) FROM {policy.table_name} "
                    f"WHERE {policy.timestamp_column} < :cutoff"
                )
                result = await self.session.execute(count_stmt, {"cutoff": cutoff_date})
                pending_count = result.scalar() or 0

                stats["policies"].append(
                    {
                        "name": policy.name,
                        "table": policy.table_name,
                        "retention_days": policy.retention_days,
                        "pending_deletion": pending_count,
                    }
                )
            except Exception as e:
                stats["policies"].append(
                    {
                        "name": policy.name,
                        "error": str(e),
                    }
                )

        return stats


# Redis cache cleanup (separate from SQL)
async def cleanup_redis_cache(
    max_age_seconds: int = 86400 * 7,  # 7 days
    pattern: str = "llm_cache:*",
) -> int:
    """Clean up old Redis cache entries.

    TIER 13 - Point 71: LRU cache cleanup.
    """
    from app.db.redis.client import get_redis_client

    client = await get_redis_client()
    if not client:
        return 0

    deleted = 0
    cursor = 0

    try:
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)

            for key in keys:
                # Check TTL - delete if no TTL set (stale)
                ttl = await client.ttl(key)
                if ttl == -1:  # No expiry set
                    await client.delete(key)
                    deleted += 1

            if cursor == 0:
                break

    except Exception as e:
        logger.warning(f"Redis cleanup error: {e}")

    return deleted


# Qdrant vector cleanup
async def cleanup_qdrant_cache(collection: str = "semantic_cache") -> int:
    """Clean up old Qdrant cache entries.

    TIER 13 - Point 71: Vector cache cleanup.
    """
    import time

    from qdrant_client.models import FieldCondition, Filter, Range

    from app.db.qdrant.client import get_async_qdrant_client

    client = await get_async_qdrant_client()
    if not client:
        return 0

    try:
        # Delete entries older than 7 days
        cutoff = time.time() - (7 * 24 * 60 * 60)

        _result = await client.delete(  # noqa: F841
            collection_name=collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="timestamp",
                        range=Range(lt=cutoff),
                    )
                ]
            ),
        )

        logger.info(f"Qdrant cache cleanup completed for {collection}")
        return 1  # Success

    except Exception as e:
        logger.warning(f"Qdrant cleanup error: {e}")
        return 0
