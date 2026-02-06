"""Transactional Outbox Pattern implementation.

TIER 10 - Point 50: Transactional Outbox Pattern

Solves the dual-write problem: ensures DB commit and event publish
are atomic. Events are stored in an outbox table within the same
transaction, then relayed to Kafka by a background process.

Benefits:
- Zero event loss even if Kafka is down
- Guaranteed at-least-once delivery
- Events are ordered by transaction commit time
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class OutboxStatus(str, Enum):
    """Status of outbox events."""

    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"


class OutboxEvent(Base):
    """Outbox table for transactional event publishing.

    TIER 10 - Point 50: Transactional Outbox table.

    Events are written in the same transaction as business data,
    then relayed to Kafka by a separate process.
    """

    __tablename__ = "outbox_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False)  # UUID

    # Event metadata
    topic = Column(String(255), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    aggregate_type = Column(String(100), nullable=False)  # e.g., "entity", "user"
    aggregate_id = Column(String(255), nullable=False, index=True)

    # Payload
    payload = Column(JSONB, nullable=False)
    headers = Column(JSONB, nullable=True)

    # Processing status
    status = Column(
        SQLEnum(OutboxStatus),
        default=OutboxStatus.PENDING,
        nullable=False,
        index=True,
    )
    retry_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)


@dataclass
class Event:
    """Domain event to be published."""

    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: dict
    topic: str | None = None
    headers: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))


class OutboxWriter:
    """Writes events to the outbox table within transactions.

    TIER 10 - Point 50: Transactional Outbox Pattern.

    Usage:
        async with session.begin():
            # Business logic
            entity = Entity(...)
            session.add(entity)

            # Publish event (same transaction)
            outbox = OutboxWriter(session)
            await outbox.publish(Event(
                event_type="EntityCreated",
                aggregate_type="entity",
                aggregate_id=entity.id,
                payload={"name": entity.name},
            ))

        # Both entity and event are committed atomically
    """

    # Default topic mapping
    TOPIC_MAP = {
        "entity": "arbor.entities",
        "user": "arbor.users",
        "search": "arbor.searches",
        "feedback": "arbor.feedback",
    }

    def __init__(self, session: AsyncSession):
        self.session = session

    async def publish(self, event: Event) -> str:
        """Publish an event to the outbox.

        The event will be sent to Kafka by the relay process.

        Args:
            event: The event to publish

        Returns:
            The event ID
        """
        # Determine topic
        topic = event.topic or self.TOPIC_MAP.get(event.aggregate_type, "arbor.events")

        outbox_event = OutboxEvent(
            event_id=event.event_id,
            topic=topic,
            event_type=event.event_type,
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            payload=event.payload,
            headers=event.headers,
            status=OutboxStatus.PENDING,
        )

        self.session.add(outbox_event)

        logger.debug(f"Outbox: Queued {event.event_type} for {event.aggregate_id}")

        return event.event_id

    async def publish_batch(self, events: list[Event]) -> list[str]:
        """Publish multiple events to the outbox."""
        event_ids = []
        for event in events:
            event_id = await self.publish(event)
            event_ids.append(event_id)
        return event_ids


class OutboxRelay:
    """Relays events from outbox to Kafka.

    TIER 10 - Point 50: Outbox relay process.

    Runs as a background process/worker that:
    1. Polls outbox for pending events
    2. Sends to Kafka
    3. Marks as sent (or failed after retries)
    """

    MAX_RETRIES = 3
    BATCH_SIZE = 100
    POLL_INTERVAL = 1.0  # seconds

    def __init__(self, session_factory, kafka_producer=None):
        self.session_factory = session_factory
        self.kafka_producer = kafka_producer
        self._running = False

    async def start(self):
        """Start the relay loop."""
        self._running = True
        logger.info("Outbox relay started")

        while self._running:
            try:
                processed = await self._process_batch()
                if processed == 0:
                    await asyncio.sleep(self.POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Outbox relay error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL * 5)

    async def stop(self):
        """Stop the relay loop."""
        self._running = False
        logger.info("Outbox relay stopped")

    async def _process_batch(self) -> int:
        """Process a batch of pending events."""
        from sqlalchemy import select, update

        async with self.session_factory() as session:
            # Fetch pending events
            stmt = (
                select(OutboxEvent)
                .where(OutboxEvent.status == OutboxStatus.PENDING)
                .order_by(OutboxEvent.created_at)
                .limit(self.BATCH_SIZE)
                .with_for_update(skip_locked=True)  # Concurrent safe
            )

            result = await session.execute(stmt)
            events = result.scalars().all()

            if not events:
                return 0

            # Mark as processing
            event_ids = [e.id for e in events]
            await session.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id.in_(event_ids))
                .values(status=OutboxStatus.PROCESSING)
            )
            await session.commit()

            # Send to Kafka
            for event in events:
                success = await self._send_to_kafka(event)

                if success:
                    await self._mark_sent(session, event)
                else:
                    await self._handle_failure(session, event)

            await session.commit()
            return len(events)

    async def _send_to_kafka(self, event: OutboxEvent) -> bool:
        """Send event to Kafka."""
        if not self.kafka_producer:
            # Mock success if no producer configured
            logger.debug(f"Outbox: Would send {event.event_type} to {event.topic}")
            return True

        try:
            message = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "aggregate_type": event.aggregate_type,
                "aggregate_id": event.aggregate_id,
                "payload": event.payload,
                "timestamp": event.created_at.isoformat(),
            }

            await self.kafka_producer.send_and_wait(
                event.topic,
                key=event.aggregate_id.encode(),
                value=json.dumps(message).encode(),
                headers=[(k, v.encode()) for k, v in (event.headers or {}).items()],
            )

            return True

        except Exception as e:
            logger.warning(f"Kafka send failed for {event.event_id}: {e}")
            return False

    async def _mark_sent(self, session: AsyncSession, event: OutboxEvent):
        """Mark event as successfully sent."""
        from sqlalchemy import update

        await session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == event.id)
            .values(
                status=OutboxStatus.SENT,
                processed_at=datetime.utcnow(),
            )
        )

    async def _handle_failure(self, session: AsyncSession, event: OutboxEvent):
        """Handle send failure with retry logic."""
        from sqlalchemy import update

        new_retry_count = event.retry_count + 1

        if new_retry_count >= self.MAX_RETRIES:
            status = OutboxStatus.FAILED
            logger.error(f"Outbox: Event {event.event_id} failed after {self.MAX_RETRIES} retries")
        else:
            status = OutboxStatus.PENDING  # Will be retried

        await session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == event.id)
            .values(
                status=status,
                retry_count=new_retry_count,
                last_error=f"Retry {new_retry_count}/{self.MAX_RETRIES}",
            )
        )


async def get_outbox_stats(session: AsyncSession) -> dict[str, Any]:
    """Get outbox statistics for monitoring."""
    from sqlalchemy import func, select

    stats = {}

    for status in OutboxStatus:
        stmt = select(func.count()).select_from(OutboxEvent).where(OutboxEvent.status == status)
        result = await session.execute(stmt)
        stats[status.value] = result.scalar() or 0

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "counts": stats,
        "total": sum(stats.values()),
    }
