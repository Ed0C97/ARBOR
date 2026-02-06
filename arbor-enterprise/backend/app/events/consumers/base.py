"""Base Kafka consumer with Dead Letter Queue support.

TIER 3 - Point 13: Kafka Dead Letter Queue (DLQ)

Provides a base consumer class that:
- Retries failed messages up to 3 times
- Sends persistently failing messages to a DLQ topic
- Continues processing after handling poison messages

Architecture:
    Consumer (MainTopic) -> Exception? -> Retry 3x -> Exception? -> Publish to DLQTopic
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DLQMessage:
    """Message structure for Dead Letter Queue."""

    original_topic: str
    original_partition: int
    original_offset: int
    original_key: str | None
    original_value: Any
    error_message: str
    error_type: str
    retry_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    consumer_group: str = ""


class BaseConsumerWithDLQ(ABC):
    """Base Kafka consumer with automatic DLQ handling.

    TIER 3 - Point 13: Kafka Dead Letter Queue

    Subclasses must implement:
    - TOPIC: str - The topic to consume
    - GROUP_ID: str - Consumer group ID
    - _process_message(self, message: dict) -> None - Message handler

    Features:
    - Automatic retry with exponential backoff (up to 3 times)
    - DLQ publishing for persistently failing messages
    - Graceful shutdown
    - Monitoring metrics

    Usage:
        class MyConsumer(BaseConsumerWithDLQ):
            TOPIC = "arbor.my.topic"
            GROUP_ID = "my-consumer-group"

            async def _process_message(self, message: dict) -> None:
                # Process the message
                ...

        consumer = MyConsumer()
        await consumer.run()
    """

    TOPIC: str = ""
    GROUP_ID: str = ""
    MAX_RETRIES: int = 3
    RETRY_DELAYS: list[float] = [1.0, 5.0, 30.0]  # Exponential backoff

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._dlq_topic = settings.kafka_dlq_topic
        self._consumer: AIOKafkaConsumer | None = None
        self._dlq_producer: AIOKafkaProducer | None = None
        self._running: bool = False

        # Metrics
        self._processed_count: int = 0
        self._error_count: int = 0
        self._dlq_count: int = 0
        self._retry_count: int = 0

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def _process_message(self, message: dict) -> None:
        """Process a single message. Must be implemented by subclass.

        Args:
            message: Deserialized message value

        Raises:
            Any exception will trigger retry logic
        """
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize consumer and DLQ producer."""
        # Main consumer
        self._consumer = AIOKafkaConsumer(
            self.TOPIC,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self.GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # Manual commit for reliability
        )

        # DLQ producer
        self._dlq_producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks="all",
        )

        await self._consumer.start()
        await self._dlq_producer.start()
        self._running = True

        logger.info(
            "%s started: topic=%s group=%s dlq=%s",
            self.__class__.__name__,
            self.TOPIC,
            self.GROUP_ID,
            self._dlq_topic,
        )

    async def stop(self) -> None:
        """Stop consumer and producer."""
        self._running = False

        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

        if self._dlq_producer is not None:
            await self._dlq_producer.stop()
            self._dlq_producer = None

        logger.info(
            "%s stopped: processed=%d errors=%d dlq=%d",
            self.__class__.__name__,
            self._processed_count,
            self._error_count,
            self._dlq_count,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main consumer loop with DLQ handling."""
        if self._consumer is None:
            await self.start()

        assert self._consumer is not None
        assert self._dlq_producer is not None

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                await self._handle_with_retry(message)

        except Exception as exc:
            logger.error("%s fatal error: %s", self.__class__.__name__, exc)
            raise
        finally:
            await self.stop()

    async def _handle_with_retry(self, message) -> None:
        """Handle a message with retry logic and DLQ fallback.

        TIER 3 - Point 13: Retry 3x, then send to DLQ.
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                await self._process_message(message.value)

                # Success - commit offset
                await self._consumer.commit()
                self._processed_count += 1
                return

            except Exception as exc:
                last_error = exc
                self._error_count += 1

                if attempt < self.MAX_RETRIES - 1:
                    # Retry with backoff
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    self._retry_count += 1

                    logger.warning(
                        "%s retry %d/%d for message at offset %d: %s (waiting %.1fs)",
                        self.__class__.__name__,
                        attempt + 1,
                        self.MAX_RETRIES,
                        message.offset,
                        exc,
                        delay,
                    )

                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted - send to DLQ
                    logger.error(
                        "%s sending to DLQ after %d retries: offset=%d error=%s",
                        self.__class__.__name__,
                        self.MAX_RETRIES,
                        message.offset,
                        exc,
                    )

        # Send to DLQ
        await self._send_to_dlq(message, last_error)

        # Commit to move past the poison message
        await self._consumer.commit()

    async def _send_to_dlq(self, message, error: Exception | None) -> None:
        """Send a failed message to the Dead Letter Queue."""
        if self._dlq_producer is None:
            logger.error("DLQ producer not initialized, message lost!")
            return

        dlq_message = DLQMessage(
            original_topic=message.topic,
            original_partition=message.partition,
            original_offset=message.offset,
            original_key=message.key,
            original_value=message.value,
            error_message=str(error) if error else "Unknown error",
            error_type=type(error).__name__ if error else "Unknown",
            retry_count=self.MAX_RETRIES,
            consumer_group=self.GROUP_ID,
        )

        try:
            await self._dlq_producer.send_and_wait(
                topic=self._dlq_topic,
                value=dlq_message.__dict__,
                key=message.key,
            )
            self._dlq_count += 1
            logger.info(
                "Message sent to DLQ: topic=%s partition=%d offset=%d",
                message.topic,
                message.partition,
                message.offset,
            )

        except Exception as exc:
            logger.error("Failed to send to DLQ: %s", exc)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Get consumer metrics for monitoring."""
        return {
            "consumer": self.__class__.__name__,
            "topic": self.TOPIC,
            "group_id": self.GROUP_ID,
            "running": self._running,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "retry_count": self._retry_count,
            "dlq_count": self._dlq_count,
            "dlq_rate": (
                self._dlq_count / self._processed_count if self._processed_count > 0 else 0.0
            ),
        }


class DLQProcessor:
    """Processor for manually handling DLQ messages.

    Used for inspecting, replaying, or archiving failed messages.

    Usage:
        processor = DLQProcessor()
        await processor.start()

        # Inspect messages
        async for msg in processor.read_messages(limit=10):
            print(msg)

        # Replay a message (re-publish to original topic)
        await processor.replay_message(msg)
    """

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._dlq_topic = settings.kafka_dlq_topic
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Initialize consumer and producer."""
        self._consumer = AIOKafkaConsumer(
            self._dlq_topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id="arbor-dlq-processor",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks="all",
        )

        await self._consumer.start()
        await self._producer.start()

    async def stop(self) -> None:
        """Stop consumer and producer."""
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()

    async def read_messages(self, limit: int = 100) -> list[dict]:
        """Read messages from DLQ without committing."""
        if self._consumer is None:
            await self.start()

        messages = []
        count = 0

        async for message in self._consumer:
            messages.append(
                {
                    "partition": message.partition,
                    "offset": message.offset,
                    "value": message.value,
                }
            )
            count += 1
            if count >= limit:
                break

        return messages

    async def replay_message(self, dlq_message: dict) -> bool:
        """Replay a DLQ message to its original topic.

        Args:
            dlq_message: The DLQ message payload

        Returns:
            True if successfully replayed
        """
        if self._producer is None:
            await self.start()

        try:
            original_topic = dlq_message.get("original_topic")
            original_value = dlq_message.get("original_value")
            original_key = dlq_message.get("original_key")

            if not original_topic or not original_value:
                logger.error("Invalid DLQ message format")
                return False

            await self._producer.send_and_wait(
                topic=original_topic,
                value=original_value,
                key=original_key.encode("utf-8") if original_key else None,
            )

            logger.info("Replayed message to topic=%s", original_topic)
            return True

        except Exception as exc:
            logger.error("Failed to replay message: %s", exc)
            return False
