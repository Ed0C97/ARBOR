"""Notification consumer - dispatches user-facing and internal notifications."""

import json
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer

from app.config import get_settings
from app.events.schemas import EventType

logger = logging.getLogger(__name__)
settings = get_settings()


class NotificationConsumer:
    """Kafka consumer that triggers notifications based on domain events.

    Listens to entity and user events and dispatches notifications via
    Redis Pub/Sub (real-time dashboards) and the WebhookManager
    (external integrations like Slack, email services, etc.).
    """

    TOPICS: list[str] = [
        f"arbor.{EventType.ENTITY_CREATED.value}",
        f"arbor.{EventType.ENTITY_UPDATED.value}",
        f"arbor.{EventType.USER_CONVERTED.value}",
    ]
    GROUP_ID: str = "arbor-notifications"

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._consumer: AIOKafkaConsumer | None = None
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
    # Notification dispatch
    # ------------------------------------------------------------------

    async def _notify_entity_created(self, payload: dict[str, Any]) -> None:
        """Notify curators when a new entity is ingested.

        Publishes to Redis Pub/Sub for real-time dashboards and dispatches
        registered webhooks.
        """
        entity_name = payload.get("name", "unknown")
        category = payload.get("category", "unknown")
        logger.info(
            "NOTIFICATION [entity.created]: New %s entity '%s' ready for review",
            category,
            entity_name,
        )

        notification = {
            "type": "entity.created",
            "title": f"New {category} entity: {entity_name}",
            "body": f"'{entity_name}' has been ingested and is ready for review.",
            "payload": payload,
        }
        await self._publish_redis(notification)
        await self._dispatch_webhooks("entity.created", payload)

    async def _notify_entity_updated(self, payload: dict[str, Any]) -> None:
        """Notify watchers when entity data changes.

        Publishes to Redis Pub/Sub and dispatches webhooks for subscribed
        consumers.
        """
        entity_id = payload.get("entity_id", "unknown")
        changed = payload.get("changed_fields", [])
        logger.info(
            "NOTIFICATION [entity.updated]: Entity %s fields changed: %s",
            entity_id,
            changed,
        )

        notification = {
            "type": "entity.updated",
            "title": f"Entity {entity_id} updated",
            "body": f"Changed fields: {', '.join(changed) if changed else 'N/A'}",
            "payload": payload,
        }
        await self._publish_redis(notification)
        await self._dispatch_webhooks("entity.updated", payload)

    async def _notify_conversion(self, payload: dict[str, Any]) -> None:
        """Alert on user conversions for business metrics.

        Publishes to Redis Pub/Sub and dispatches webhooks (e.g. Slack
        incoming webhook for the #conversions channel).
        """
        user_id = payload.get("user_id", "unknown")
        conversion_type = payload.get("conversion_type", "unknown")
        logger.info(
            "NOTIFICATION [user.converted]: User %s conversion type '%s'",
            user_id,
            conversion_type,
        )

        notification = {
            "type": "user.converted",
            "title": f"Conversion: {conversion_type}",
            "body": f"User {user_id} triggered a '{conversion_type}' conversion.",
            "payload": payload,
        }
        await self._publish_redis(notification)
        await self._dispatch_webhooks("user.converted", payload)

    # ------------------------------------------------------------------
    # Delivery channels
    # ------------------------------------------------------------------

    async def _publish_redis(self, notification: dict) -> None:
        """Publish a notification to Redis Pub/Sub for real-time dashboards."""
        try:
            from app.db.redis.client import get_redis_client

            client = await get_redis_client()
            if client:
                await client.publish(
                    "arbor:notifications",
                    json.dumps(notification),
                )
                logger.debug("Redis notification published: %s", notification.get("type"))
        except Exception as exc:
            logger.warning("Redis notification publish failed: %s", exc)

    async def _dispatch_webhooks(self, event_type: str, payload: dict) -> None:
        """Dispatch registered webhooks for the given event type."""
        try:
            from app.events.webhooks import get_webhook_manager

            manager = get_webhook_manager()
            await manager.dispatch(event_type, payload)
            logger.debug("Webhooks dispatched for event: %s", event_type)
        except Exception as exc:
            logger.warning("Webhook dispatch failed for %s: %s", event_type, exc)
