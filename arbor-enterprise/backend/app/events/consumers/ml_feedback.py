"""ML feedback consumer - collects user interactions for ranking model updates."""

import json
import logging
from typing import Optional

from aiokafka import AIOKafkaConsumer

from app.config import get_settings
from app.events.schemas import EventType

logger = logging.getLogger(__name__)
settings = get_settings()


class MLFeedbackConsumer:
    """Kafka consumer that collects click/conversion signals for the reranker.

    Listens to ``user.clicked`` and ``user.converted`` events and batches
    them into training samples that can be periodically flushed to the
    :class:`~app.ml.feedback_loop.FeedbackCollector`.
    """

    TOPICS: list[str] = [
        f"arbor.{EventType.USER_CLICKED.value}",
        f"arbor.{EventType.USER_CONVERTED.value}",
    ]
    GROUP_ID: str = "arbor-ml-feedback"

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running: bool = False

        # Buffered training pairs: list of (query, entity_id, reward)
        self._buffer: list[dict] = []
        self._buffer_flush_size: int = 100

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the Kafka consumer and begin processing."""
        self._consumer = AIOKafkaConsumer(
            *self.TOPICS,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self.GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self._consumer.start()
        self._running = True
        logger.info("MLFeedbackConsumer started on topics %s", self.TOPICS)

    async def stop(self) -> None:
        """Flush remaining buffer and stop consuming."""
        self._running = False
        if self._buffer:
            await self._flush_buffer()
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        logger.info("MLFeedbackConsumer stopped")

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume interaction events in an infinite loop."""
        if self._consumer is None:
            await self.start()

        assert self._consumer is not None

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                await self._handle(message.value)
        except Exception as exc:
            logger.error("MLFeedbackConsumer error: %s", exc)
            raise
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle(self, event: dict) -> None:
        """Process a single click or conversion event.

        Args:
            event: Deserialized Kafka message value.
        """
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})

        query = payload.get("query", "")
        entity_id = payload.get("entity_id", "")
        position = payload.get("position", 0)

        # Compute reward signal
        reward = self._compute_reward(event_type, position, payload)

        sample = {
            "query": query,
            "entity_id": entity_id,
            "position": position,
            "reward": reward,
            "event_type": event_type,
            "user_id": payload.get("user_id", ""),
        }
        self._buffer.append(sample)

        logger.debug(
            "ML feedback sample buffered: entity=%s reward=%.2f (buffer=%d)",
            entity_id,
            reward,
            len(self._buffer),
        )

        if len(self._buffer) >= self._buffer_flush_size:
            await self._flush_buffer()

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(event_type: str, position: int, payload: dict) -> float:
        """Derive a numeric reward from the interaction signal.

        Higher rewards for conversions, with a position-decay bonus so that
        clicks on lower-ranked items signal stronger relevance.

        Args:
            event_type: The event type string.
            position: 0-based result position the user interacted with.
            payload: Full event payload for extra signal extraction.

        Returns:
            Reward float in [0, 1].
        """
        base_reward = 1.0 if event_type == EventType.USER_CONVERTED.value else 0.4

        # Position-based decay: clicking a result at position 5 is a stronger
        # signal than clicking position 0 (the user scrolled past others).
        position_bonus = min(position * 0.05, 0.3) if position > 0 else 0.0

        # Time-to-click: faster clicks on top results are weaker signal
        time_to_click_ms = payload.get("time_to_click_ms", 0)
        time_bonus = 0.1 if time_to_click_ms > 3000 else 0.0

        return min(base_reward + position_bonus + time_bonus, 1.0)

    # ------------------------------------------------------------------
    # Buffer flush
    # ------------------------------------------------------------------

    async def _flush_buffer(self) -> None:
        """Flush buffered samples to the feedback collector.

        In production this would call into
        :class:`~app.ml.feedback_loop.FeedbackCollector` to persist the
        training batch and optionally trigger an online model update.
        """
        batch_size = len(self._buffer)
        if batch_size == 0:
            return

        logger.info(
            "Flushing %d ML feedback samples to feedback collector", batch_size
        )

        from app.ml.feedback_loop import get_feedback_collector

        collector = get_feedback_collector()
        await collector.record_batch(self._buffer)

        self._buffer.clear()
