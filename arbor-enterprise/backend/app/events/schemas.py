"""Pydantic event schemas for the A.R.B.O.R. Enterprise event bus."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Canonical event types flowing through the Kafka bus."""

    ENTITY_CREATED = "entity.created"
    ENTITY_UPDATED = "entity.updated"
    SEARCH_PERFORMED = "search.performed"
    USER_CLICKED = "user.clicked"
    USER_CONVERTED = "user.converted"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Common envelope for every event published to Kafka."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "arbor-enterprise"
    version: str = "1.0"
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entity events
# ---------------------------------------------------------------------------


class EntityCreatedEvent(BaseEvent):
    """Emitted when a new entity is ingested and persisted."""

    event_type: EventType = EventType.ENTITY_CREATED
    payload: dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "entity_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Cafe Otonom",
                "category": "cafe",
                "city": "Mexico City",
                "source_type": "google_maps",
            }
        },
    )


class EntityUpdatedEvent(BaseEvent):
    """Emitted when an existing entity's data changes."""

    event_type: EventType = EventType.ENTITY_UPDATED
    payload: dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "entity_id": "550e8400-e29b-41d4-a716-446655440000",
                "changed_fields": ["vibe_dna", "price_tier"],
                "updated_by": "curator",
            }
        },
    )


# ---------------------------------------------------------------------------
# Search events
# ---------------------------------------------------------------------------


class SearchPerformedEvent(BaseEvent):
    """Emitted after every user search query is executed."""

    event_type: EventType = EventType.SEARCH_PERFORMED
    payload: dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "query": "cozy coffee shop with fast wifi",
                "intent": "DISCOVERY",
                "result_count": 8,
                "latency_ms": 342.5,
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "filters": {"city": "Mexico City", "category": "cafe"},
                "result_ids": [
                    "550e8400-e29b-41d4-a716-446655440010",
                    "550e8400-e29b-41d4-a716-446655440011",
                ],
            }
        },
    )


# ---------------------------------------------------------------------------
# User interaction events
# ---------------------------------------------------------------------------


class UserClickedEvent(BaseEvent):
    """Emitted when a user clicks on a search result."""

    event_type: EventType = EventType.USER_CLICKED
    payload: dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "entity_id": "550e8400-e29b-41d4-a716-446655440010",
                "query": "cozy coffee shop with fast wifi",
                "position": 2,
                "time_to_click_ms": 1500,
            }
        },
    )


class UserConvertedEvent(BaseEvent):
    """Emitted when a click converts to a meaningful action (save, visit, book)."""

    event_type: EventType = EventType.USER_CONVERTED
    payload: dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "entity_id": "550e8400-e29b-41d4-a716-446655440010",
                "conversion_type": "save",
                "query": "cozy coffee shop with fast wifi",
                "position": 2,
            }
        },
    )


# ---------------------------------------------------------------------------
# Mapping helper
# ---------------------------------------------------------------------------

EVENT_CLASS_MAP: dict[EventType, type[BaseEvent]] = {
    EventType.ENTITY_CREATED: EntityCreatedEvent,
    EventType.ENTITY_UPDATED: EntityUpdatedEvent,
    EventType.SEARCH_PERFORMED: SearchPerformedEvent,
    EventType.USER_CLICKED: UserClickedEvent,
    EventType.USER_CONVERTED: UserConvertedEvent,
}
