"""Analytics consumer - processes search events for usage reporting."""

import json
import logging

from aiokafka import AIOKafkaConsumer

from app.config import get_settings
from app.events.schemas import EventType

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalyticsConsumer:
    """Kafka consumer that aggregates search analytics.

    Listens to ``arbor.search.performed`` events and maintains running
    counters for popular queries, average latency, zero-result rates, etc.
    In production these would be flushed to a time-series store or warehouse.
    """

    TOPIC: str = f"arbor.{EventType.SEARCH_PERFORMED.value}"
    GROUP_ID: str = "arbor-analytics"

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._consumer: AIOKafkaConsumer | None = None
        self._running: bool = False

        # In-memory running aggregates (swap for a real store in prod)
        self.total_searches: int = 0
        self.zero_result_count: int = 0
        self.latency_sum_ms: float = 0.0
        self.popular_queries: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the Kafka consumer and begin processing."""
        self._consumer = AIOKafkaConsumer(
            self.TOPIC,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self.GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self._consumer.start()
        self._running = True
        logger.info("AnalyticsConsumer started on topic %s", self.TOPIC)

    async def stop(self) -> None:
        """Stop consuming and close the Kafka connection."""
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        logger.info("AnalyticsConsumer stopped")

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume search events in an infinite loop.

        Call :meth:`stop` from another coroutine to break the loop.
        """
        if self._consumer is None:
            await self.start()

        assert self._consumer is not None  # for type checker

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                await self._handle(message.value)
        except Exception as exc:
            logger.error("AnalyticsConsumer error: %s", exc)
            raise
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle(self, event: dict) -> None:
        """Process a single search.performed event.

        Args:
            event: Deserialized Kafka message value.
        """
        payload = event.get("payload", {})

        query: str = payload.get("query", "")
        result_count: int = payload.get("result_count", 0)
        latency_ms: float = payload.get("latency_ms", 0.0)

        self.total_searches += 1
        self.latency_sum_ms += latency_ms

        if result_count == 0:
            self.zero_result_count += 1
            logger.info("Zero-result query detected: '%s'", query)

        # Track popular queries
        normalised = query.strip().lower()
        if normalised:
            self.popular_queries[normalised] = self.popular_queries.get(normalised, 0) + 1

        logger.debug(
            "Analytics event processed: query='%s' results=%d latency=%.1fms (total=%d)",
            query,
            result_count,
            latency_ms,
            self.total_searches,
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a summary snapshot of current analytics aggregates."""
        avg_latency = self.latency_sum_ms / self.total_searches if self.total_searches else 0.0
        top_queries = sorted(self.popular_queries.items(), key=lambda kv: kv[1], reverse=True)[:20]

        return {
            "total_searches": self.total_searches,
            "zero_result_count": self.zero_result_count,
            "zero_result_rate": (
                self.zero_result_count / self.total_searches if self.total_searches else 0.0
            ),
            "avg_latency_ms": round(avg_latency, 2),
            "top_queries": top_queries,
        }
