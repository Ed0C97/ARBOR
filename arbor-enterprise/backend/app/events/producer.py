"""Kafka event producer for the A.R.B.O.R. Enterprise event bus."""

import json
import logging
from typing import Optional

from aiokafka import AIOKafkaProducer

from app.config import get_settings
from app.events.schemas import BaseEvent, EventType

logger = logging.getLogger(__name__)
settings = get_settings()


class EventBus:
    """Async Kafka producer that publishes domain events.

    Usage::

        bus = EventBus()
        await bus.start()
        await bus.emit(EntityCreatedEvent(payload={...}))
        await bus.stop()

    The bus serialises each event to JSON and publishes it to a Kafka topic
    derived from the event type (e.g., ``entity.created`` -> ``arbor.entity.created``).
    """

    TOPIC_PREFIX: str = "arbor"

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._producer: Optional[AIOKafkaProducer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create and start the underlying Kafka producer."""
        if self._producer is not None:
            logger.debug("EventBus producer already running")
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            enable_idempotence=True,
            max_batch_size=32768,
            linger_ms=10,
        )
        await self._producer.start()
        logger.info("EventBus Kafka producer started: %s", self._bootstrap_servers)

    async def stop(self) -> None:
        """Flush pending messages and stop the producer."""
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
            logger.info("EventBus Kafka producer stopped")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def emit(
        self,
        event: BaseEvent,
        key: str | None = None,
        topic_override: str | None = None,
    ) -> None:
        """Publish an event to Kafka.

        Args:
            event: The domain event to publish.
            key: Optional partition key (defaults to event_id).
            topic_override: Force a specific topic instead of deriving from event type.
        """
        if self._producer is None:
            logger.warning("EventBus not started; dropping event %s", event.event_type)
            return

        topic = topic_override or self._topic_for(event.event_type)
        partition_key = key or str(event.event_id)
        payload = event.model_dump(mode="json")

        try:
            record_metadata = await self._producer.send_and_wait(
                topic=topic,
                value=payload,
                key=partition_key,
            )
            logger.debug(
                "Event published: type=%s topic=%s partition=%d offset=%d",
                event.event_type.value,
                topic,
                record_metadata.partition,
                record_metadata.offset,
            )
        except Exception as exc:
            logger.error(
                "Failed to publish event %s to %s: %s",
                event.event_type.value,
                topic,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _topic_for(self, event_type: EventType) -> str:
        """Derive the Kafka topic name from the event type.

        ``entity.created`` -> ``arbor.entity.created``
        """
        return f"{self.TOPIC_PREFIX}.{event_type.value}"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return the singleton EventBus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
