"""Webhook event system for the A.R.B.O.R. Enterprise platform.

Allows external systems to subscribe to domain events and receive HTTP
callbacks with HMAC-SHA256 signed payloads.  Delivery is fire-and-forget
with configurable retry logic and exponential backoff.
"""

import asyncio
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Webhook event types
# ---------------------------------------------------------------------------


class WebhookEvent(str, Enum):
    """Events that external consumers can subscribe to."""

    ENTITY_CREATED = "entity.created"
    ENTITY_UPDATED = "entity.updated"
    ENTITY_ENRICHED = "entity.enriched"
    SEARCH_PERFORMED = "search.performed"
    ANOMALY_DETECTED = "anomaly.detected"
    ENRICHMENT_COMPLETED = "enrichment.completed"


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


@dataclass
class WebhookSubscription:
    """Represents a single webhook registration for a tenant."""

    id: str
    tenant_id: str
    url: str
    events: list[WebhookEvent]
    secret: str

    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    max_retries: int = 3
    retry_delay_seconds: int = 30

    failure_count: int = 0
    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None


# ---------------------------------------------------------------------------
# Payload schema
# ---------------------------------------------------------------------------


class WebhookPayload(BaseModel):
    """Pydantic model for the JSON body sent to webhook endpoints."""

    event_type: WebhookEvent
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    subscription_id: str
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HMAC utilities
# ---------------------------------------------------------------------------


def _compute_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute the HMAC-SHA256 hex digest for *payload_bytes* using *secret*."""
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_bytes,
        digestmod=hashlib.sha256,
    ).hexdigest()


def verify_webhook_signature(
    payload_bytes: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify an incoming webhook signature.

    Args:
        payload_bytes: The raw request body bytes.
        signature: The value of the ``X-Webhook-Signature`` header.
        secret: The shared secret associated with the subscription.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    expected = _compute_signature(payload_bytes, secret)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Delivery engine
# ---------------------------------------------------------------------------


class WebhookDelivery:
    """Handles the actual HTTP delivery of webhook payloads with retry logic."""

    TIMEOUT_SECONDS: float = 10.0

    async def deliver(
        self,
        subscription: WebhookSubscription,
        payload: WebhookPayload,
    ) -> bool:
        """Deliver *payload* to the subscriber's URL.

        Retries up to ``subscription.max_retries`` times with exponential
        backoff starting at ``subscription.retry_delay_seconds``.

        Returns:
            ``True`` on successful delivery, ``False`` if all attempts failed.
        """
        payload_bytes = payload.model_dump_json().encode("utf-8")
        signature = _compute_signature(payload_bytes, subscription.secret)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": payload.event_type.value,
            "X-Webhook-Subscription-Id": subscription.id,
        }

        last_exc: Exception | None = None

        async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
            for attempt in range(1, subscription.max_retries + 1):
                try:
                    response = await client.post(
                        subscription.url,
                        content=payload_bytes,
                        headers=headers,
                    )
                    response.raise_for_status()

                    subscription.last_success_at = datetime.now(UTC)
                    subscription.failure_count = 0
                    logger.info(
                        "Webhook delivered: subscription=%s event=%s url=%s "
                        "status=%d attempt=%d",
                        subscription.id,
                        payload.event_type.value,
                        subscription.url,
                        response.status_code,
                        attempt,
                    )
                    return True

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    last_exc = exc
                    subscription.failure_count += 1
                    subscription.last_failure_at = datetime.now(UTC)

                    logger.warning(
                        "Webhook delivery failed: subscription=%s event=%s "
                        "url=%s attempt=%d/%d error=%s",
                        subscription.id,
                        payload.event_type.value,
                        subscription.url,
                        attempt,
                        subscription.max_retries,
                        exc,
                    )

                    if attempt < subscription.max_retries:
                        delay = subscription.retry_delay_seconds * (2 ** (attempt - 1))
                        logger.debug(
                            "Retrying webhook in %ds (attempt %d/%d)",
                            delay,
                            attempt + 1,
                            subscription.max_retries,
                        )
                        await asyncio.sleep(delay)

        logger.error(
            "Webhook delivery exhausted retries: subscription=%s event=%s " "url=%s last_error=%s",
            subscription.id,
            payload.event_type.value,
            subscription.url,
            last_exc,
        )
        return False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class WebhookManager:
    """Central registry for webhook subscriptions and dispatch coordinator.

    Subscriptions are stored in-memory, keyed by subscription id.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, WebhookSubscription] = {}
        self._delivery = WebhookDelivery()

    # -- Registration -------------------------------------------------------

    def register(
        self,
        tenant_id: str,
        url: str,
        events: list[WebhookEvent],
        secret: str,
    ) -> WebhookSubscription:
        """Create and store a new webhook subscription.

        Args:
            tenant_id: The tenant that owns this subscription.
            url: The HTTPS endpoint to receive POST callbacks.
            events: The list of events the subscriber is interested in.
            secret: A shared secret used for HMAC-SHA256 signing.

        Returns:
            The newly created :class:`WebhookSubscription`.
        """
        subscription = WebhookSubscription(
            id=str(uuid4()),
            tenant_id=tenant_id,
            url=url,
            events=events,
            secret=secret,
        )
        self._subscriptions[subscription.id] = subscription
        logger.info(
            "Webhook registered: id=%s tenant=%s url=%s events=%s",
            subscription.id,
            tenant_id,
            url,
            [e.value for e in events],
        )
        return subscription

    def unregister(self, subscription_id: str) -> bool:
        """Remove a subscription by id.

        Returns:
            ``True`` if the subscription existed and was removed, ``False``
            otherwise.
        """
        removed = self._subscriptions.pop(subscription_id, None)
        if removed is not None:
            logger.info("Webhook unregistered: id=%s", subscription_id)
            return True
        logger.warning("Webhook unregister failed (not found): id=%s", subscription_id)
        return False

    def list_subscriptions(self, tenant_id: str) -> list[WebhookSubscription]:
        """Return all subscriptions belonging to *tenant_id*."""
        return [sub for sub in self._subscriptions.values() if sub.tenant_id == tenant_id]

    # -- Dispatch -----------------------------------------------------------

    def dispatch(
        self,
        event_type: WebhookEvent,
        data: dict[str, Any],
    ) -> None:
        """Fan-out *event_type* to every matching, active subscription.

        Each delivery runs as a fire-and-forget ``asyncio.Task`` so the
        caller is never blocked by slow or failing endpoints.
        """
        matching = [
            sub
            for sub in self._subscriptions.values()
            if sub.is_active and event_type in sub.events
        ]

        if not matching:
            logger.debug("No active subscriptions for event %s", event_type.value)
            return

        for sub in matching:
            payload = WebhookPayload(
                event_type=event_type,
                subscription_id=sub.id,
                data=data,
            )
            asyncio.create_task(
                self._delivery.deliver(sub, payload),
                name=f"webhook-{sub.id}-{event_type.value}",
            )

        logger.info(
            "Webhook dispatch: event=%s matched=%d subscriptions",
            event_type.value,
            len(matching),
        )

    # -- Stats --------------------------------------------------------------

    def get_delivery_stats(self, subscription_id: str) -> dict[str, Any]:
        """Return delivery health statistics for a subscription.

        Returns:
            A dict with ``subscription_id``, ``is_active``,
            ``failure_count``, ``last_success_at``, ``last_failure_at``,
            and ``registered_events``.  Returns an empty dict when the
            subscription is not found.
        """
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            return {}

        return {
            "subscription_id": sub.id,
            "tenant_id": sub.tenant_id,
            "url": sub.url,
            "is_active": sub.is_active,
            "failure_count": sub.failure_count,
            "last_success_at": (sub.last_success_at.isoformat() if sub.last_success_at else None),
            "last_failure_at": (sub.last_failure_at.isoformat() if sub.last_failure_at else None),
            "registered_events": [e.value for e in sub.events],
            "created_at": sub.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_webhook_manager: WebhookManager | None = None


def get_webhook_manager() -> WebhookManager:
    """Return the singleton :class:`WebhookManager` instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager
