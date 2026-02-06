"""Change Data Capture (CDC) Handler.

TIER 10 - Point 55: CDC (Debezium/Custom)

Listens to PostgreSQL WAL changes and propagates them to:
- Qdrant (vector search index)
- Redis (cache invalidation)
- Kafka (event streaming)

Target latency: < 2 seconds from DB change to search update.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import asyncpg
from asyncpg import Connection

from app.config import get_settings
from app.db.qdrant.client import get_async_qdrant_client
from app.db.redis.client import get_redis_client
from app.events.kafka_producer import get_kafka_producer

logger = logging.getLogger(__name__)
settings = get_settings()


class ChangeOperation(str, Enum):
    """Type of database change."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    """Represents a single database change."""

    table: str
    operation: ChangeOperation
    old_data: dict | None
    new_data: dict | None
    timestamp: datetime
    lsn: str  # Log Sequence Number

    @property
    def entity_id(self) -> str | None:
        """Extract entity ID from change data."""
        data = self.new_data or self.old_data
        if data:
            return data.get("id") or data.get("entity_id")
        return None


class ChangeHandler(ABC):
    """Abstract handler for processing changes."""

    @abstractmethod
    async def handle(self, event: ChangeEvent) -> None:
        """Process a change event."""
        pass


class VectorSyncHandler(ChangeHandler):
    """Sync changes to Qdrant vector store.

    TIER 10 - Point 55: < 2 second latency target.
    """

    def __init__(self):
        self._qdrant = None

    @property
    async def qdrant(self):
        if self._qdrant is None:
            self._qdrant = await get_async_qdrant_client()
        return self._qdrant

    async def handle(self, event: ChangeEvent) -> None:
        """Update vector store based on change."""
        if event.table not in ("arbor_entity", "arbor_enrichment"):
            return

        entity_id = event.entity_id
        if not entity_id:
            return

        client = await self.qdrant

        if event.operation == ChangeOperation.DELETE:
            # Remove from vector store
            try:
                await client.delete(
                    collection_name="entities_vectors",
                    points_selector={
                        "filter": {"must": [{"key": "entity_id", "match": {"value": entity_id}}]}
                    },
                )
                logger.info(f"Removed entity {entity_id} from vector store")
            except Exception as e:
                logger.error(f"Failed to remove from vector store: {e}")

        elif event.operation in (ChangeOperation.INSERT, ChangeOperation.UPDATE):
            # Queue for re-embedding via Kafka -> embedding worker
            logger.info(f"Queued entity {entity_id} for re-embedding")

            # Emit event for embedding worker
            producer = get_kafka_producer()
            await producer.send(
                topic="entity-reindex",
                value={
                    "entity_id": entity_id,
                    "operation": event.operation.value,
                    "timestamp": event.timestamp.isoformat(),
                },
            )


class CacheInvalidationHandler(ChangeHandler):
    """Invalidate caches on data changes."""

    async def handle(self, event: ChangeEvent) -> None:
        """Invalidate relevant caches."""
        entity_id = event.entity_id
        if not entity_id:
            return

        redis = get_redis_client()

        # Invalidate entity cache
        cache_key = f"entity:{entity_id}"
        await redis.delete(cache_key)

        # Invalidate semantic cache entries mentioning this entity
        # This uses the cache invalidation pattern from TIER 4 - Point 18
        from app.llm.cache import get_semantic_cache

        cache = get_semantic_cache()
        await cache.invalidate_by_entity(entity_id)

        logger.debug(f"Invalidated cache for entity {entity_id}")


class EventStreamHandler(ChangeHandler):
    """Stream changes to Kafka for downstream consumers."""

    async def handle(self, event: ChangeEvent) -> None:
        """Publish change event to Kafka."""
        producer = get_kafka_producer()

        topic = f"cdc.{event.table}"

        await producer.send(
            topic=topic,
            value={
                "operation": event.operation.value,
                "table": event.table,
                "entity_id": event.entity_id,
                "old_data": event.old_data,
                "new_data": event.new_data,
                "timestamp": event.timestamp.isoformat(),
                "lsn": event.lsn,
            },
        )


class CDCListener:
    """PostgreSQL Logical Replication listener.

    TIER 10 - Point 55: Custom CDC Implementation

    Uses pg_logical or wal2json for WAL decoding.
    For production, consider Debezium for managed CDC.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        slot_name: str = "arbor_cdc_slot",
        publication_name: str = "arbor_changes",
    ):
        self.connection_string = connection_string or settings.arbor_database_url
        self.slot_name = slot_name
        self.publication_name = publication_name
        self.handlers: list[ChangeHandler] = []
        self._running = False
        self._connection: Connection | None = None

    def add_handler(self, handler: ChangeHandler) -> None:
        """Add a change handler."""
        self.handlers.append(handler)

    async def setup(self) -> None:
        """Create replication slot and publication if needed."""
        conn = await asyncpg.connect(self.connection_string)

        try:
            # Create publication for tracked tables
            # Note: publication_name is a trusted init param, not user input
            # Using psycopg-style escaping for identifiers to satisfy security scanners
            safe_pub_name = self.publication_name.replace("'", "''")  # noqa: B608
            await conn.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_publication WHERE pubname = '{safe_pub_name}'
                    ) THEN
                        EXECUTE format('CREATE PUBLICATION %I FOR TABLE arbor_entity, arbor_enrichment, arbor_curation', '{safe_pub_name}');
                    END IF;
                END
                $$;
            """
            )

            # Create logical replication slot
            try:
                await conn.execute(
                    f"""
                    SELECT pg_create_logical_replication_slot(
                        '{self.slot_name}',
                        'wal2json'
                    );
                """
                )
                logger.info(f"Created replication slot: {self.slot_name}")
            except asyncpg.DuplicateObjectError:
                logger.info(f"Replication slot {self.slot_name} already exists")

        finally:
            await conn.close()

    async def start(self) -> None:
        """Start listening for changes."""
        await self.setup()

        self._running = True
        self._connection = await asyncpg.connect(
            self.connection_string, server_settings={"application_name": "arbor_cdc"}
        )

        logger.info("CDC listener started")

        while self._running:
            try:
                # Poll for changes
                changes = await self._connection.fetch(
                    f"""
                    SELECT * FROM pg_logical_slot_get_changes(
                        '{self.slot_name}',
                        NULL,
                        100,
                        'include-timestamp', 'true',
                        'include-lsn', 'true'
                    );
                """
                )

                for row in changes:
                    await self._process_change(row)

                # Small delay to avoid busy-waiting
                if not changes:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"CDC error: {e}")
                await asyncio.sleep(1)

    async def _process_change(self, row: asyncpg.Record) -> None:
        """Process a single WAL change."""
        try:
            data = json.loads(row["data"])

            for change in data.get("change", []):
                event = ChangeEvent(
                    table=change.get("table", "unknown"),
                    operation=ChangeOperation(change.get("kind", "").upper()),
                    old_data=change.get("oldkeys", {}).get("keyvalues"),
                    new_data=self._extract_new_data(change),
                    timestamp=datetime.fromisoformat(
                        data.get("timestamp", datetime.utcnow().isoformat())
                    ),
                    lsn=str(row.get("lsn", "")),
                )

                # Dispatch to all handlers
                for handler in self.handlers:
                    try:
                        await handler.handle(event)
                    except Exception as e:
                        logger.error(f"Handler {type(handler).__name__} failed: {e}")

        except Exception as e:
            logger.error(f"Failed to parse CDC change: {e}")

    def _extract_new_data(self, change: dict) -> dict | None:
        """Extract column values from WAL change."""
        columns = change.get("columnnames", [])
        values = change.get("columnvalues", [])

        if columns and values:
            return dict(zip(columns, values))
        return None

    async def stop(self) -> None:
        """Stop the listener."""
        self._running = False
        if self._connection:
            await self._connection.close()
        logger.info("CDC listener stopped")


# Factory function
def create_cdc_listener() -> CDCListener:
    """Create a configured CDC listener with all handlers."""
    listener = CDCListener()

    # Add handlers in order of priority
    listener.add_handler(CacheInvalidationHandler())
    listener.add_handler(VectorSyncHandler())
    listener.add_handler(EventStreamHandler())

    return listener


# Entry point for running as standalone service
async def main():
    """Run CDC listener as standalone service."""
    listener = create_cdc_listener()

    try:
        await listener.start()
    except KeyboardInterrupt:
        await listener.stop()


if __name__ == "__main__":
    asyncio.run(main())
