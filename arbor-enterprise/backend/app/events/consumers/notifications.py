"""Notification consumer - dispatches user-facing and internal notifications."""

import json
import logging
from typing import Any, Optional

from aiokafka import AIOKafkaConsumer

from app.config import get_settings
from app.events.schemas import EventType

logger = logging.getLogger(__name__)
settings = get_settings()


class NotificationConsumer:
    """Kafka consumer that triggers notifications based on domain events.

    Listens to entity and user events and dispatches notifications to the
    appropriate channels (email, push, webhook). This is a placeholder
    implementation; integrate a real notification service in production.
    """

    TOPICS: list[str] = [
        f"arbor.{EventType.ENTITY_CREATED.value}",
        f"arbor.{EventType.ENTITY_UPDATED.value}",
        f"arbor.{EventType.USER_CONVERTED.value}",
    ]
    GROUP_ID: str = "arbor-notifications"

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running: bool = False

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
        logger.info("NotificationConsumer started on topics %s", self.TOPICS)

    async def stop(self) -> None:
        """Stop consuming and close the Kafka connection."""
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        logger.info("NotificationConsumer stopped")

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume events in an infinite loop."""
        if self._consumer is None:
            await self.start()

        assert self._consumer is not None

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                await self._handle(message.value)
        except Exception as exc:
            logger.error("NotificationConsumer error: %s", exc)
            raise
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _handle(self, event: dict) -> None:
        """Route an event to the appropriate notification channel.

        Args:
            event: Deserialized Kafka message value.
        """
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})

        if event_type == EventType.ENTITY_CREATED.value:
            await self._notify_entity_created(payload)
        elif event_type == EventType.ENTITY_UPDATED.value:
            await self._notify_entity_updated(payload)
        elif event_type == EventType.USER_CONVERTED.value:
            await self._notify_conversion(payload)
        else:
            logger.debug("NotificationConsumer ignoring event type: %s", event_type)

    # ------------------------------------------------------------------
    # Notification stubs
    # ------------------------------------------------------------------

    async def _notify_entity_created(self, payload: dict[str, Any]) -> None:
        """Placeholder: notify curators when a new entity is ingested."""
        entity_name = payload.get("name", "unknown")
        category = payload.get("category", "unknown")
        logger.info(
            "NOTIFICATION [entity.created]: New %s entity '%s' ready for review",
            category,
            entity_name,
        )
        # TODO: send email / push to curator dashboard

    async def _notify_entity_updated(self, payload: dict[str, Any]) -> None:
        """Placeholder: notify watchers when entity data changes."""
        entity_id = payload.get("entity_id", "unknown")
        changed = payload.get("changed_fields", [])
        logger.info(
            "NOTIFICATION [entity.updated]: Entity %s fields changed: %s",
            entity_id,
            changed,
        )
        # TODO: webhook / in-app notification for subscribed users

    async def _notify_conversion(self, payload: dict[str, Any]) -> None:
        """Placeholder: internal alert on user conversions for business metrics."""
        user_id = payload.get("user_id", "unknown")
        conversion_type = payload.get("conversion_type", "unknown")
        logger.info(
            "NOTIFICATION [user.converted]: User %s conversion type '%s'",
            user_id,
            conversion_type,
        )
        # TODO: Slack webhook / analytics push
