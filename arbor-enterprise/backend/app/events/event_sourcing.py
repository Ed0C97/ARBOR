"""Full Event Sourcing + CQRS implementation for A.R.B.O.R. Enterprise.

Provides a complete event-sourced architecture with Command Query Responsibility
Segregation (CQRS) for the entity domain. Events are the single source of truth;
current state is derived by replaying the event stream.

Architecture:
    Command → CommandBus → Handler → Aggregate → DomainEvent → EventStore
                                                      ↓
                                          ReadModelProjection → Query

Components:
- DomainEvent:              Immutable fact representing something that happened
- EventStore:               Append-only store with optimistic concurrency + snapshots
- Aggregate:                DDD aggregate root that emits events
- EntityAggregate:          Concrete aggregate for ARBOR discovery entities
- ReadModelProjection:      Base class for materialised read-side views
- EntitySearchProjection:   Searchable index rebuilt from the event stream
- Command / CommandBus:     Write-side dispatch for CQRS separation

Integrates with:
- app.events.producer       (EventBus / Kafka publishing)
- app.events.outbox         (Transactional Outbox pattern)
- app.events.schemas        (Canonical Kafka event types)
"""

import copy
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Domain Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvent:
    """Immutable domain event representing a state change.

    Every mutation in the system is captured as a ``DomainEvent`` and persisted
    to the :class:`EventStore`.  Aggregates replay these events to rebuild
    their current state.

    Attributes:
        event_id:        Globally unique identifier for this event.
        aggregate_id:    ID of the aggregate this event belongs to.
        aggregate_type:  Type discriminator (e.g. ``"Entity"``).
        event_type:      Qualified event name (e.g. ``"EntityDiscovered"``).
        version:         Monotonically increasing version within the aggregate stream.
        timestamp:       UTC time the event was created.
        payload:         Arbitrary data specific to the event type.
        metadata:        Operational metadata (user, source, trace IDs, ...).
        causation_id:    ID of the command / event that caused this event.
        correlation_id:  End-to-end correlation ID for distributed tracing.
    """

    event_id: UUID
    aggregate_id: str
    aggregate_type: str
    event_type: str
    version: int
    timestamp: datetime
    payload: dict[str, Any]
    metadata: dict[str, Any]
    causation_id: str | None = None
    correlation_id: str | None = None

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        aggregate_id: str,
        aggregate_type: str,
        event_type: str,
        version: int,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> "DomainEvent":
        """Factory that auto-generates ``event_id`` and ``timestamp``."""
        return cls(
            event_id=uuid4(),
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            event_type=event_type,
            version=version,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
            metadata=metadata or {},
            causation_id=causation_id,
            correlation_id=correlation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (e.g. for JSON/Kafka publishing)."""
        return {
            "event_id": str(self.event_id),
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_type": self.event_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
        }


# ---------------------------------------------------------------------------
# Event Store
# ---------------------------------------------------------------------------


class ConcurrencyError(Exception):
    """Raised when an optimistic concurrency check fails on append."""


class EventStore:
    """In-memory append-only event store with optimistic concurrency control.

    Each aggregate stream is identified by ``aggregate_id`` and contains an
    ordered list of :class:`DomainEvent` instances.  The store enforces a
    strict version sequence so that concurrent writers are detected.

    Thread-safety is provided via a :class:`threading.Lock`.

    Snapshots can be saved for large aggregates to speed up rehydration:
    load the snapshot, then replay only the events *after* the snapshot
    version.

    Usage::

        store = get_event_store()
        store.append(event)
        events = store.get_events("entity-123")
    """

    def __init__(self) -> None:
        self._streams: dict[str, list[DomainEvent]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(self, event: DomainEvent) -> None:
        """Append an event to the aggregate stream.

        Optimistic concurrency: the event's ``version`` must be exactly
        ``current_version + 1``.  If another writer slipped in between,
        a :class:`ConcurrencyError` is raised.

        Args:
            event: The domain event to persist.

        Raises:
            ConcurrencyError: If the version does not match expectations.
        """
        with self._lock:
            stream = self._streams.setdefault(event.aggregate_id, [])
            expected_version = len(stream) + 1

            if event.version != expected_version:
                raise ConcurrencyError(
                    f"Concurrency conflict for aggregate {event.aggregate_id!r}: "
                    f"expected version {expected_version}, got {event.version}"
                )

            stream.append(event)

        logger.debug(
            "EventStore: appended %s v%d to aggregate %s",
            event.event_type,
            event.version,
            event.aggregate_id,
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> list[DomainEvent]:
        """Return events for an aggregate, optionally starting from a version.

        Args:
            aggregate_id: The aggregate whose stream to read.
            from_version: Return only events with ``version > from_version``.
                          Defaults to ``0`` (all events).

        Returns:
            Ordered list of :class:`DomainEvent` instances.
        """
        with self._lock:
            stream = self._streams.get(aggregate_id, [])
            return [e for e in stream if e.version > from_version]

    def get_all_events(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> list[DomainEvent]:
        """Return events across *all* aggregates for global replay.

        Useful for rebuilding read-model projections from scratch.

        Args:
            event_type: If given, filter to only this event type.
            since: If given, filter to events after this timestamp.

        Returns:
            Events sorted by timestamp (oldest first).
        """
        with self._lock:
            all_events: list[DomainEvent] = []
            for stream in self._streams.values():
                for event in stream:
                    if event_type and event.event_type != event_type:
                        continue
                    if since and event.timestamp <= since:
                        continue
                    all_events.append(event)

        all_events.sort(key=lambda e: (e.timestamp, e.version))
        return all_events

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_snapshot(self, aggregate_id: str) -> dict[str, Any] | None:
        """Return the latest snapshot for an aggregate, or ``None``.

        A snapshot contains:
        - ``state``: The serialised aggregate state at the snapshot point.
        - ``version``: The aggregate version at snapshot time.
        - ``created_at``: When the snapshot was taken.
        """
        with self._lock:
            return copy.deepcopy(self._snapshots.get(aggregate_id))

    def save_snapshot(
        self,
        aggregate_id: str,
        state: dict[str, Any],
        version: int,
    ) -> None:
        """Persist a point-in-time snapshot for faster aggregate rehydration.

        Args:
            aggregate_id: The aggregate to snapshot.
            state: Serialised aggregate state.
            version: The aggregate version this snapshot represents.
        """
        with self._lock:
            self._snapshots[aggregate_id] = {
                "state": copy.deepcopy(state),
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        logger.debug(
            "EventStore: snapshot saved for aggregate %s at v%d",
            aggregate_id,
            version,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_aggregate_version(self, aggregate_id: str) -> int:
        """Return the current version (event count) for an aggregate.

        Returns ``0`` if the aggregate has no events.
        """
        with self._lock:
            return len(self._streams.get(aggregate_id, []))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_event_store: EventStore | None = None
_event_store_lock = threading.Lock()


def get_event_store() -> EventStore:
    """Return the singleton :class:`EventStore` instance.

    Thread-safe via double-checked locking.
    """
    global _event_store
    if _event_store is None:
        with _event_store_lock:
            if _event_store is None:
                _event_store = EventStore()
                logger.info("EventStore singleton initialised")
    return _event_store


# ---------------------------------------------------------------------------
# Aggregate base class
# ---------------------------------------------------------------------------


class Aggregate(ABC):
    """DDD Aggregate Root that derives state from an event stream.

    Subclasses implement :meth:`_apply` to mutate internal state for each
    event type.  New state changes are recorded via :meth:`_record`, which
    creates a :class:`DomainEvent`, applies it locally, and queues it as
    uncommitted.

    Typical lifecycle::

        # Create
        entity = EntityAggregate(aggregate_id="ent-1")
        entity.discover("Cafe Otonom", "cafe", "Mexico City")

        # Persist
        store = get_event_store()
        for event in entity.uncommitted_events:
            store.append(event)

        # Reload later
        entity2 = EntityAggregate(aggregate_id="ent-1")
        entity2.load_from_history(store.get_events("ent-1"))
    """

    AGGREGATE_TYPE: str = "Aggregate"

    def __init__(self, aggregate_id: str) -> None:
        self._aggregate_id: str = aggregate_id
        self._version: int = 0
        self._uncommitted_events: list[DomainEvent] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def aggregate_id(self) -> str:
        return self._aggregate_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def uncommitted_events(self) -> list[DomainEvent]:
        """Events recorded since last persist / load."""
        return list(self._uncommitted_events)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    @abstractmethod
    def _apply(self, event: DomainEvent) -> None:
        """Apply an event to mutate internal state.

        Must be implemented by concrete aggregates.  This method should be
        a *pure* state transition -- no side effects, no I/O.

        Args:
            event: The domain event to apply.
        """

    def _record(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> DomainEvent:
        """Record a new event: create it, apply it, and queue it as uncommitted.

        Args:
            event_type: Qualified event name (e.g. ``"EntityDiscovered"``).
            payload: Event-specific data.
            metadata: Operational metadata.
            causation_id: Optional causation chain link.
            correlation_id: Optional distributed correlation ID.

        Returns:
            The newly created :class:`DomainEvent`.
        """
        next_version = self._version + 1

        event = DomainEvent.create(
            aggregate_id=self._aggregate_id,
            aggregate_type=self.AGGREGATE_TYPE,
            event_type=event_type,
            version=next_version,
            payload=payload,
            metadata=metadata,
            causation_id=causation_id,
            correlation_id=correlation_id,
        )

        self._apply(event)
        self._version = next_version
        self._uncommitted_events.append(event)

        logger.debug(
            "Aggregate %s: recorded %s v%d",
            self._aggregate_id,
            event_type,
            next_version,
        )
        return event

    # ------------------------------------------------------------------
    # History replay
    # ------------------------------------------------------------------

    def load_from_history(self, events: list[DomainEvent]) -> None:
        """Rebuild aggregate state by replaying historical events.

        Events are applied in order and the version counter is advanced.
        The uncommitted list is *not* populated -- these are already
        persisted events.

        Args:
            events: Ordered list of events from the :class:`EventStore`.
        """
        for event in events:
            self._apply(event)
            self._version = event.version

        logger.debug(
            "Aggregate %s: loaded from history, now at v%d",
            self._aggregate_id,
            self._version,
        )

    def clear_uncommitted(self) -> None:
        """Clear the uncommitted event queue after successful persistence."""
        self._uncommitted_events.clear()


# ---------------------------------------------------------------------------
# Entity Aggregate (concrete)
# ---------------------------------------------------------------------------


class EntityAggregate(Aggregate):
    """Concrete aggregate for A.R.B.O.R. discovery entities.

    Models the full lifecycle of a place / venue / experience entity:

    - **EntityDiscovered** -- initial creation from a scraper or manual entry
    - **EntityEnriched**   -- vibe DNA and tags added by the enrichment pipeline
    - **EntityVibeUpdated** -- individual vibe dimensions recalculated
    - **EntityDeactivated** -- soft-delete / removal from active index
    - **EntityMerged**     -- two duplicate entities merged into one

    State attributes:
        name, category, city, vibe_dna, tags, status, enrichment_count
    """

    AGGREGATE_TYPE: str = "Entity"

    # Known event types
    ENTITY_DISCOVERED = "EntityDiscovered"
    ENTITY_ENRICHED = "EntityEnriched"
    ENTITY_VIBE_UPDATED = "EntityVibeUpdated"
    ENTITY_DEACTIVATED = "EntityDeactivated"
    ENTITY_MERGED = "EntityMerged"

    def __init__(self, aggregate_id: str) -> None:
        super().__init__(aggregate_id)

        # Entity state
        self.name: str = ""
        self.category: str = ""
        self.city: str = ""
        self.vibe_dna: dict[str, float] = {}
        self.tags: list[str] = []
        self.status: str = "pending"
        self.enrichment_count: int = 0

    # ------------------------------------------------------------------
    # Commands (write methods)
    # ------------------------------------------------------------------

    def discover(
        self,
        name: str,
        category: str,
        city: str,
        metadata: dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Record the initial discovery of an entity.

        Args:
            name: Human-readable entity name.
            category: Type category (e.g. ``"cafe"``, ``"gallery"``).
            city: City where the entity is located.
            metadata: Optional operational metadata.

        Returns:
            The ``EntityDiscovered`` event.

        Raises:
            ValueError: If the entity has already been discovered.
        """
        if self.status != "pending":
            raise ValueError(
                f"Entity {self._aggregate_id} already discovered (status={self.status})"
            )

        return self._record(
            event_type=self.ENTITY_DISCOVERED,
            payload={"name": name, "category": category, "city": city},
            metadata=metadata,
        )

    def enrich(
        self,
        vibe_dna: dict[str, float],
        tags: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Enrich the entity with vibe DNA and tags.

        Args:
            vibe_dna: Dimensional vibe profile (e.g. ``{"cozy": 0.8, "lively": 0.3}``).
            tags: Descriptive tags (e.g. ``["specialty-coffee", "wifi"]``).
            metadata: Optional operational metadata.

        Returns:
            The ``EntityEnriched`` event.

        Raises:
            ValueError: If the entity is not active.
        """
        if self.status not in ("active", "pending"):
            raise ValueError(
                f"Cannot enrich entity {self._aggregate_id} with status {self.status}"
            )

        return self._record(
            event_type=self.ENTITY_ENRICHED,
            payload={"vibe_dna": vibe_dna, "tags": tags},
            metadata=metadata,
        )

    def update_vibe(
        self,
        dimensions: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Update individual vibe dimensions (partial update).

        Args:
            dimensions: Dimension overrides (merged into existing ``vibe_dna``).
            metadata: Optional operational metadata.

        Returns:
            The ``EntityVibeUpdated`` event.

        Raises:
            ValueError: If the entity is not active.
        """
        if self.status != "active":
            raise ValueError(
                f"Cannot update vibe for entity {self._aggregate_id} with status {self.status}"
            )

        return self._record(
            event_type=self.ENTITY_VIBE_UPDATED,
            payload={"dimensions": dimensions},
            metadata=metadata,
        )

    def deactivate(
        self,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Soft-delete: remove entity from the active index.

        Args:
            reason: Human-readable reason for deactivation.
            metadata: Optional operational metadata.

        Returns:
            The ``EntityDeactivated`` event.

        Raises:
            ValueError: If the entity is already deactivated.
        """
        if self.status == "deactivated":
            raise ValueError(
                f"Entity {self._aggregate_id} is already deactivated"
            )

        return self._record(
            event_type=self.ENTITY_DEACTIVATED,
            payload={"reason": reason},
            metadata=metadata,
        )

    def merge_with(
        self,
        other_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Merge another entity into this one (de-duplication).

        The *other* entity should be deactivated separately.

        Args:
            other_id: Aggregate ID of the entity being merged in.
            metadata: Optional operational metadata.

        Returns:
            The ``EntityMerged`` event.

        Raises:
            ValueError: If the entity is not active.
        """
        if self.status != "active":
            raise ValueError(
                f"Cannot merge into entity {self._aggregate_id} with status {self.status}"
            )

        return self._record(
            event_type=self.ENTITY_MERGED,
            payload={"merged_entity_id": other_id},
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Event application (state transitions)
    # ------------------------------------------------------------------

    def _apply(self, event: DomainEvent) -> None:
        """Apply event to mutate internal state.

        Pure state transition -- no side effects.
        """
        handler = self._EVENT_HANDLERS.get(event.event_type)
        if handler is not None:
            handler(self, event)
        else:
            logger.warning(
                "EntityAggregate: no handler for event type %s",
                event.event_type,
            )

    def _apply_discovered(self, event: DomainEvent) -> None:
        self.name = event.payload["name"]
        self.category = event.payload["category"]
        self.city = event.payload["city"]
        self.status = "active"

    def _apply_enriched(self, event: DomainEvent) -> None:
        self.vibe_dna = copy.deepcopy(event.payload["vibe_dna"])
        self.tags = list(event.payload["tags"])
        self.enrichment_count += 1

    def _apply_vibe_updated(self, event: DomainEvent) -> None:
        for dim, value in event.payload["dimensions"].items():
            self.vibe_dna[dim] = value

    def _apply_deactivated(self, event: DomainEvent) -> None:
        self.status = "deactivated"

    def _apply_merged(self, event: DomainEvent) -> None:
        # The merge payload is informational; state change is minimal.
        # Detailed merge logic (combining vibe_dna, tags) is handled
        # by the command handler that orchestrates both aggregates.
        pass

    # Dispatch table (avoids long if/elif chains)
    _EVENT_HANDLERS: dict[str, Callable[["EntityAggregate", DomainEvent], None]] = {
        ENTITY_DISCOVERED: _apply_discovered,
        ENTITY_ENRICHED: _apply_enriched,
        ENTITY_VIBE_UPDATED: _apply_vibe_updated,
        ENTITY_DEACTIVATED: _apply_deactivated,
        ENTITY_MERGED: _apply_merged,
    }

    # ------------------------------------------------------------------
    # Snapshot support
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        """Serialise current state for snapshotting."""
        return {
            "name": self.name,
            "category": self.category,
            "city": self.city,
            "vibe_dna": copy.deepcopy(self.vibe_dna),
            "tags": list(self.tags),
            "status": self.status,
            "enrichment_count": self.enrichment_count,
        }

    def restore_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot dict."""
        self.name = snapshot["name"]
        self.category = snapshot["category"]
        self.city = snapshot["city"]
        self.vibe_dna = copy.deepcopy(snapshot["vibe_dna"])
        self.tags = list(snapshot["tags"])
        self.status = snapshot["status"]
        self.enrichment_count = snapshot["enrichment_count"]


# ---------------------------------------------------------------------------
# Read-Model Projection (base)
# ---------------------------------------------------------------------------


class ReadModelProjection(ABC):
    """Base class for CQRS read-side projections.

    A projection subscribes to domain events and maintains a denormalised
    view optimised for a specific query pattern.  Projections are
    *disposable* -- they can be rebuilt at any time by replaying the full
    event stream.

    Subclasses implement :meth:`handle` to react to individual events.
    """

    @abstractmethod
    def handle(self, event: DomainEvent) -> None:
        """Update the read model in response to a domain event.

        Args:
            event: The event to project.
        """

    def rebuild(self, events: list[DomainEvent]) -> None:
        """Replay a full event stream to rebuild the projection from scratch.

        Args:
            events: Ordered list of all events to replay.
        """
        for event in events:
            self.handle(event)

        logger.info(
            "%s: rebuilt from %d events",
            self.__class__.__name__,
            len(events),
        )


# ---------------------------------------------------------------------------
# Entity Search Projection (concrete)
# ---------------------------------------------------------------------------


class EntitySearchProjection(ReadModelProjection):
    """Materialised read model for entity search and discovery.

    Maintains an in-memory index of entity summaries optimised for
    filtering and search.  Kept in sync by projecting domain events
    from the :class:`EventStore`.

    Projected events:
    - ``EntityDiscovered`` -- add entity to the index
    - ``EntityEnriched``   -- update vibe_dna and tags
    - ``EntityVibeUpdated`` -- merge updated dimensions
    - ``EntityDeactivated`` -- remove entity from the index

    Query example::

        projection = EntitySearchProjection()
        projection.rebuild(store.get_all_events())
        results = projection.query({"city": "Mexico City", "category": "cafe"})
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Projection handlers
    # ------------------------------------------------------------------

    def handle(self, event: DomainEvent) -> None:
        """Route event to the appropriate projection handler."""
        handler = self._HANDLERS.get(event.event_type)
        if handler is not None:
            handler(self, event)

    def _on_discovered(self, event: DomainEvent) -> None:
        self._index[event.aggregate_id] = {
            "entity_id": event.aggregate_id,
            "name": event.payload["name"],
            "category": event.payload["category"],
            "city": event.payload["city"],
            "vibe_dna": {},
            "tags": [],
            "status": "active",
            "discovered_at": event.timestamp.isoformat(),
            "last_updated": event.timestamp.isoformat(),
        }

    def _on_enriched(self, event: DomainEvent) -> None:
        entry = self._index.get(event.aggregate_id)
        if entry is None:
            return
        entry["vibe_dna"] = copy.deepcopy(event.payload["vibe_dna"])
        entry["tags"] = list(event.payload["tags"])
        entry["last_updated"] = event.timestamp.isoformat()

    def _on_vibe_updated(self, event: DomainEvent) -> None:
        entry = self._index.get(event.aggregate_id)
        if entry is None:
            return
        for dim, value in event.payload["dimensions"].items():
            entry["vibe_dna"][dim] = value
        entry["last_updated"] = event.timestamp.isoformat()

    def _on_deactivated(self, event: DomainEvent) -> None:
        self._index.pop(event.aggregate_id, None)

    _HANDLERS: dict[str, Callable[["EntitySearchProjection", DomainEvent], None]] = {
        EntityAggregate.ENTITY_DISCOVERED: _on_discovered,
        EntityAggregate.ENTITY_ENRICHED: _on_enriched,
        EntityAggregate.ENTITY_VIBE_UPDATED: _on_vibe_updated,
        EntityAggregate.ENTITY_DEACTIVATED: _on_deactivated,
    }

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query the materialised index with optional filters.

        Supported filter keys:
        - ``city`` (str):     Exact city match.
        - ``category`` (str): Exact category match.
        - ``tag`` (str):      Entity must contain this tag.
        - ``name`` (str):     Case-insensitive substring match on name.
        - ``min_vibe`` (dict[str, float]): Each specified dimension must be >= value.

        Args:
            filters: Dictionary of filter criteria.  ``None`` returns all entities.

        Returns:
            List of matching entity summary dicts.
        """
        if not filters:
            return list(self._index.values())

        results: list[dict[str, Any]] = []

        for entry in self._index.values():
            if not self._matches(entry, filters):
                continue
            results.append(entry)

        return results

    @staticmethod
    def _matches(entry: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check whether an index entry matches all supplied filters."""
        if "city" in filters and entry.get("city") != filters["city"]:
            return False

        if "category" in filters and entry.get("category") != filters["category"]:
            return False

        if "tag" in filters and filters["tag"] not in entry.get("tags", []):
            return False

        if "name" in filters:
            if filters["name"].lower() not in entry.get("name", "").lower():
                return False

        if "min_vibe" in filters:
            vibe_dna = entry.get("vibe_dna", {})
            for dim, threshold in filters["min_vibe"].items():
                if vibe_dna.get(dim, 0.0) < threshold:
                    return False

        return True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of entities in the index."""
        return len(self._index)

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """Retrieve a single entity summary by ID."""
        return self._index.get(entity_id)


# ---------------------------------------------------------------------------
# Command (CQRS write-side)
# ---------------------------------------------------------------------------


@dataclass
class Command:
    """A request to perform a write operation.

    Commands are dispatched through the :class:`CommandBus` and handled
    by registered command handlers.

    Attributes:
        command_id:   Unique identifier for idempotency / tracing.
        command_type: Discriminator string (e.g. ``"DiscoverEntity"``).
        payload:      Data required to execute the command.
        metadata:     Operational metadata (auth context, trace IDs, ...).
        issued_at:    UTC timestamp when the command was created.
        issued_by:    Identity of the issuer (user ID, service name, ...).
    """

    command_id: UUID = field(default_factory=uuid4)
    command_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issued_by: str = ""


# ---------------------------------------------------------------------------
# Command Bus
# ---------------------------------------------------------------------------


class CommandBus:
    """Dispatches commands to their registered handlers.

    Provides the *write* side of the CQRS split.  Each command type is
    mapped to exactly one handler function.  Handlers are responsible for
    loading the aggregate, calling domain methods, and persisting events.

    Usage::

        bus = CommandBus()

        def handle_discover(cmd: Command) -> dict:
            entity = EntityAggregate(aggregate_id=cmd.payload["entity_id"])
            entity.discover(cmd.payload["name"], cmd.payload["category"], cmd.payload["city"])
            store = get_event_store()
            for event in entity.uncommitted_events:
                store.append(event)
            entity.clear_uncommitted()
            return {"status": "ok", "entity_id": entity.aggregate_id}

        bus.register_handler("DiscoverEntity", handle_discover)

        result = bus.dispatch(Command(
            command_type="DiscoverEntity",
            payload={"entity_id": "ent-1", "name": "Cafe Otonom", "category": "cafe", "city": "CDMX"},
            issued_by="ingestion-pipeline",
        ))
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[Command], Any]] = {}

    def register_handler(
        self,
        command_type: str,
        handler: Callable[[Command], Any],
    ) -> None:
        """Register a handler function for a command type.

        Args:
            command_type: The command type string this handler responds to.
            handler: Callable that receives a :class:`Command` and returns a result.

        Raises:
            ValueError: If a handler is already registered for this command type.
        """
        if command_type in self._handlers:
            raise ValueError(
                f"Handler already registered for command type {command_type!r}"
            )
        self._handlers[command_type] = handler

        logger.debug("CommandBus: registered handler for %s", command_type)

    def dispatch(self, command: Command) -> Any:
        """Dispatch a command to its registered handler.

        Args:
            command: The command to execute.

        Returns:
            Whatever the handler returns.

        Raises:
            ValueError: If no handler is registered for the command type.
        """
        handler = self._handlers.get(command.command_type)
        if handler is None:
            raise ValueError(
                f"No handler registered for command type {command.command_type!r}"
            )

        logger.info(
            "CommandBus: dispatching %s (id=%s, by=%s)",
            command.command_type,
            command.command_id,
            command.issued_by,
        )

        try:
            result = handler(command)
            logger.debug(
                "CommandBus: %s completed successfully",
                command.command_type,
            )
            return result
        except Exception as exc:
            logger.error(
                "CommandBus: %s failed: %s",
                command.command_type,
                exc,
            )
            raise
