"""Conflict-free Replicated Data Types (CRDTs) for distributed state in ARBOR Enterprise.

CRDTs guarantee eventual consistency across distributed nodes without
requiring coordination or consensus protocols.  Each replica can accept
writes independently; periodic state merges converge deterministically.

Implemented data structures:

    GCounter     – Grow-only counter (one slot per node)
    PNCounter    – Positive-Negative counter (increment + decrement)
    GSet         – Grow-only set (add-only, union merge)
    ORSet        – Observed-Remove set (add/remove with unique tags)
    LWWRegister  – Last-Writer-Wins register (timestamp + node-id tiebreak)
    LWWMap       – LWW Element Map (LWWRegister per key)
    CRDTStore    – Manages named CRDT instances with sync helpers

Usage::

    store = get_crdt_store()
    votes = store.get_counter("entity_votes")
    votes.increment(1)

    # Sync with remote node
    payload = store.get_sync_payload()
    remote_store.apply_sync_payload(payload)
"""

import copy
import logging
import time
import uuid
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# GCounter – Grow-only Counter
# ═══════════════════════════════════════════════════════════════════════════


class GCounter:
    """Grow-only counter backed by a node-id → count vector.

    Each node may only increment its own slot.  The global value is the
    sum across all slots.  Merge takes the element-wise maximum.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._counts: dict[str, int] = {node_id: 0}

    # -- Mutators ----------------------------------------------------------

    def increment(self, amount: int = 1) -> None:
        """Increment this node's counter by *amount* (must be >= 0)."""
        if amount < 0:
            raise ValueError("GCounter does not support negative increments")
        self._counts[self.node_id] = self._counts.get(self.node_id, 0) + amount

    # -- Queries -----------------------------------------------------------

    @property
    def value(self) -> int:
        """Total count across all nodes."""
        return sum(self._counts.values())

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "GCounter") -> None:
        """Merge *other* into this counter (element-wise max)."""
        for node_id, count in other._counts.items():
            self._counts[node_id] = max(self._counts.get(node_id, 0), count)

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "GCounter",
            "node_id": self.node_id,
            "counts": copy.deepcopy(self._counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GCounter":
        instance = cls(node_id=data["node_id"])
        instance._counts = dict(data["counts"])
        return instance

    def __repr__(self) -> str:
        return f"GCounter(node_id={self.node_id!r}, value={self.value})"


# ═══════════════════════════════════════════════════════════════════════════
# PNCounter – Positive-Negative Counter
# ═══════════════════════════════════════════════════════════════════════════


class PNCounter:
    """Counter supporting both increment and decrement.

    Internally composed of two :class:`GCounter` instances – one for
    positive increments and one for negative (decrements).  The observable
    value is ``pos.value - neg.value``.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._pos = GCounter(node_id)
        self._neg = GCounter(node_id)

    # -- Mutators ----------------------------------------------------------

    def increment(self, amount: int = 1) -> None:
        """Increase the counter by *amount*."""
        if amount < 0:
            raise ValueError("Use decrement() for negative adjustments")
        self._pos.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrease the counter by *amount*."""
        if amount < 0:
            raise ValueError("Decrement amount must be >= 0")
        self._neg.increment(amount)

    # -- Queries -----------------------------------------------------------

    @property
    def value(self) -> int:
        """Net value (increments minus decrements)."""
        return self._pos.value - self._neg.value

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "PNCounter") -> None:
        """Merge *other* into this counter."""
        self._pos.merge(other._pos)
        self._neg.merge(other._neg)

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "PNCounter",
            "node_id": self.node_id,
            "pos": self._pos.to_dict(),
            "neg": self._neg.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PNCounter":
        instance = cls(node_id=data["node_id"])
        instance._pos = GCounter.from_dict(data["pos"])
        instance._neg = GCounter.from_dict(data["neg"])
        return instance

    def __repr__(self) -> str:
        return f"PNCounter(node_id={self.node_id!r}, value={self.value})"


# ═══════════════════════════════════════════════════════════════════════════
# GSet – Grow-only Set
# ═══════════════════════════════════════════════════════════════════════════


class GSet:
    """Immutable-add set.  Elements can be added but never removed.

    Merge is a simple set union, which is idempotent, commutative, and
    associative – satisfying the CRDT merge requirements.
    """

    def __init__(self) -> None:
        self._elements: set[Any] = set()

    # -- Mutators ----------------------------------------------------------

    def add(self, element: Any) -> None:
        """Add *element* to the set."""
        self._elements.add(element)

    # -- Queries -----------------------------------------------------------

    def contains(self, element: Any) -> bool:
        """Return ``True`` if *element* is in the set."""
        return element in self._elements

    @property
    def elements(self) -> frozenset[Any]:
        """Return a frozen copy of the current elements."""
        return frozenset(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "GSet") -> None:
        """Merge *other* into this set (union)."""
        self._elements |= other._elements

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "GSet",
            "elements": sorted(str(e) for e in self._elements),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GSet":
        instance = cls()
        instance._elements = set(data["elements"])
        return instance

    def __repr__(self) -> str:
        return f"GSet(size={len(self._elements)})"


# ═══════════════════════════════════════════════════════════════════════════
# ORSet – Observed-Remove Set
# ═══════════════════════════════════════════════════════════════════════════


class ORSet:
    """Observed-Remove set allowing both add and remove.

    Each ``add`` operation attaches a globally unique *tag* to the
    element.  A ``remove`` deletes all currently known tags for that
    element.  Concurrent adds on different replicas produce distinct
    tags and therefore survive a concurrent remove (add-wins semantics).
    """

    def __init__(self) -> None:
        # element → {tag, ...}
        self._entries: dict[str, set[str]] = {}
        # Tags that have been explicitly removed
        self._tombstones: set[str] = set()

    # -- Mutators ----------------------------------------------------------

    def add(self, element: str) -> str:
        """Add *element* with a fresh unique tag.  Returns the tag."""
        tag = uuid.uuid4().hex
        if element not in self._entries:
            self._entries[element] = set()
        self._entries[element].add(tag)
        return tag

    def remove(self, element: str) -> None:
        """Remove *element* by tombstoning all of its current tags.

        After removal, the element is no longer ``contained`` unless a
        concurrent add on another replica re-introduces it.
        """
        if element in self._entries:
            self._tombstones |= self._entries[element]
            del self._entries[element]

    # -- Queries -----------------------------------------------------------

    def contains(self, element: str) -> bool:
        """Return ``True`` if *element* has at least one live tag."""
        return element in self._entries and len(self._entries[element]) > 0

    @property
    def elements(self) -> frozenset[str]:
        """Return the set of elements with at least one live tag."""
        return frozenset(
            elem for elem, tags in self._entries.items() if len(tags) > 0
        )

    def __len__(self) -> int:
        return len(self.elements)

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "ORSet") -> None:
        """Merge *other* into this set (add-wins).

        1. Union all (element, tag) pairs from both replicas.
        2. Remove any tag present in either tombstone set.
        """
        # Combine tombstones first
        combined_tombstones = self._tombstones | other._tombstones

        # Union entries
        merged: dict[str, set[str]] = {}
        all_elements = set(self._entries.keys()) | set(other._entries.keys())

        for elem in all_elements:
            tags_self = self._entries.get(elem, set())
            tags_other = other._entries.get(elem, set())
            live_tags = (tags_self | tags_other) - combined_tombstones
            if live_tags:
                merged[elem] = live_tags

        self._entries = merged
        self._tombstones = combined_tombstones

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "ORSet",
            "entries": {
                elem: sorted(tags) for elem, tags in self._entries.items()
            },
            "tombstones": sorted(self._tombstones),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ORSet":
        instance = cls()
        instance._entries = {
            elem: set(tags) for elem, tags in data["entries"].items()
        }
        instance._tombstones = set(data["tombstones"])
        return instance

    def __repr__(self) -> str:
        return f"ORSet(size={len(self)})"


# ═══════════════════════════════════════════════════════════════════════════
# LWWRegister – Last-Writer-Wins Register
# ═══════════════════════════════════════════════════════════════════════════


class LWWRegister:
    """Single-value register with last-writer-wins conflict resolution.

    Ties on timestamp are broken by lexicographic comparison of node IDs
    so that the outcome is deterministic across replicas.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._value: Any = None
        self._timestamp: float = 0.0
        self._writer: str = node_id

    # -- Mutators ----------------------------------------------------------

    def set(self, value: Any) -> None:
        """Write *value* with the current wall-clock time."""
        self._value = value
        self._timestamp = time.time()
        self._writer = self.node_id

    # -- Queries -----------------------------------------------------------

    def get(self) -> Any:
        """Return the current value."""
        return self._value

    @property
    def timestamp(self) -> float:
        """Timestamp of the most recent write."""
        return self._timestamp

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "LWWRegister") -> None:
        """Keep the value with the latest timestamp.

        If timestamps are equal, the higher ``node_id`` (lexicographic)
        wins, ensuring a deterministic total order.
        """
        if other._timestamp > self._timestamp:
            self._value = other._value
            self._timestamp = other._timestamp
            self._writer = other._writer
        elif other._timestamp == self._timestamp and other._writer > self._writer:
            self._value = other._value
            self._timestamp = other._timestamp
            self._writer = other._writer

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "LWWRegister",
            "node_id": self.node_id,
            "value": self._value,
            "timestamp": self._timestamp,
            "writer": self._writer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LWWRegister":
        instance = cls(node_id=data["node_id"])
        instance._value = data["value"]
        instance._timestamp = data["timestamp"]
        instance._writer = data["writer"]
        return instance

    def __repr__(self) -> str:
        return (
            f"LWWRegister(node_id={self.node_id!r}, "
            f"value={self._value!r}, ts={self._timestamp:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# LWWMap – Last-Writer-Wins Element Map
# ═══════════════════════════════════════════════════════════════════════════


class LWWMap:
    """Key-value map where each key is backed by a :class:`LWWRegister`.

    Deletion is modelled as setting a key's value to ``None``; queries
    filter out ``None``-valued entries so deleted keys do not appear.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._registers: dict[str, LWWRegister] = {}

    # -- Mutators ----------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set *key* to *value* with an LWW timestamp."""
        if key not in self._registers:
            self._registers[key] = LWWRegister(self.node_id)
        self._registers[key].set(value)

    def delete(self, key: str) -> None:
        """Soft-delete *key* by setting its value to ``None``."""
        self.set(key, None)

    # -- Queries -----------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent / deleted."""
        if key in self._registers:
            val = self._registers[key].get()
            if val is not None:
                return val
        return default

    def keys(self) -> list[str]:
        """Return keys whose current value is not ``None``."""
        return [
            k for k, reg in self._registers.items() if reg.get() is not None
        ]

    def values(self) -> list[Any]:
        """Return non-``None`` values."""
        return [
            reg.get()
            for reg in self._registers.values()
            if reg.get() is not None
        ]

    def items(self) -> list[tuple[str, Any]]:
        """Return ``(key, value)`` pairs for non-``None`` entries."""
        return [
            (k, reg.get())
            for k, reg in self._registers.items()
            if reg.get() is not None
        ]

    def __len__(self) -> int:
        return len(self.keys())

    # -- Merge -------------------------------------------------------------

    def merge(self, other: "LWWMap") -> None:
        """Merge *other* into this map (per-key register merge)."""
        for key, other_reg in other._registers.items():
            if key in self._registers:
                self._registers[key].merge(other_reg)
            else:
                self._registers[key] = LWWRegister.from_dict(other_reg.to_dict())

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "LWWMap",
            "node_id": self.node_id,
            "registers": {
                k: reg.to_dict() for k, reg in self._registers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LWWMap":
        instance = cls(node_id=data["node_id"])
        instance._registers = {
            k: LWWRegister.from_dict(reg_data)
            for k, reg_data in data["registers"].items()
        }
        return instance

    def __repr__(self) -> str:
        return f"LWWMap(node_id={self.node_id!r}, size={len(self)})"


# ═══════════════════════════════════════════════════════════════════════════
# CRDTStore – Central CRDT Instance Manager
# ═══════════════════════════════════════════════════════════════════════════

# Type tag → deserialiser mapping
_CRDT_DESERIALIZERS: dict[str, type] = {
    "PNCounter": PNCounter,
    "ORSet": ORSet,
    "LWWMap": LWWMap,
    "GCounter": GCounter,
    "GSet": GSet,
    "LWWRegister": LWWRegister,
}


class CRDTStore:
    """Manages named CRDT instances and provides sync helpers.

    Each CRDT is identified by a string *name*.  The store lazily creates
    instances on first access and exposes ``get_sync_payload`` /
    ``apply_sync_payload`` for full-state replication between nodes.
    """

    def __init__(self, node_id: str | None = None) -> None:
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self._counters: dict[str, PNCounter] = {}
        self._sets: dict[str, ORSet] = {}
        self._maps: dict[str, LWWMap] = {}
        logger.info("CRDTStore initialised (node_id=%s)", self.node_id)

    # -- Typed Accessors ---------------------------------------------------

    def get_counter(self, name: str) -> PNCounter:
        """Return (or create) the :class:`PNCounter` named *name*."""
        if name not in self._counters:
            self._counters[name] = PNCounter(self.node_id)
            logger.debug("Created PNCounter %r", name)
        return self._counters[name]

    def get_set(self, name: str) -> ORSet:
        """Return (or create) the :class:`ORSet` named *name*."""
        if name not in self._sets:
            self._sets[name] = ORSet()
            logger.debug("Created ORSet %r", name)
        return self._sets[name]

    def get_map(self, name: str) -> LWWMap:
        """Return (or create) the :class:`LWWMap` named *name*."""
        if name not in self._maps:
            self._maps[name] = LWWMap(self.node_id)
            logger.debug("Created LWWMap %r", name)
        return self._maps[name]

    # -- Remote Merge (single CRDT) ----------------------------------------

    def merge_remote(self, name: str, crdt_type: str, remote_data: dict) -> None:
        """Merge a single incoming remote CRDT state into the local store.

        Parameters
        ----------
        name:
            Logical name of the CRDT (e.g. ``"entity_votes"``).
        crdt_type:
            One of ``"PNCounter"``, ``"ORSet"``, ``"LWWMap"``.
        remote_data:
            Serialised state produced by the remote replica's ``to_dict()``.
        """
        logger.debug("merge_remote: name=%r type=%r", name, crdt_type)

        if crdt_type == "PNCounter":
            remote = PNCounter.from_dict(remote_data)
            local = self.get_counter(name)
            local.merge(remote)
        elif crdt_type == "ORSet":
            remote = ORSet.from_dict(remote_data)
            local = self.get_set(name)
            local.merge(remote)
        elif crdt_type == "LWWMap":
            remote = LWWMap.from_dict(remote_data)
            local = self.get_map(name)
            local.merge(remote)
        else:
            logger.warning("Unknown CRDT type %r – skipping merge", crdt_type)

    # -- Full-State Sync ---------------------------------------------------

    def get_sync_payload(self) -> dict:
        """Serialise every managed CRDT for full-state replication.

        Returns a dict keyed by ``(crdt_type, name)`` tuples (encoded as
        strings) mapping to serialised state.
        """
        payload: dict[str, dict] = {}

        for name, counter in self._counters.items():
            payload[f"PNCounter::{name}"] = counter.to_dict()

        for name, orset in self._sets.items():
            payload[f"ORSet::{name}"] = orset.to_dict()

        for name, lww_map in self._maps.items():
            payload[f"LWWMap::{name}"] = lww_map.to_dict()

        logger.debug(
            "get_sync_payload: %d counters, %d sets, %d maps",
            len(self._counters),
            len(self._sets),
            len(self._maps),
        )
        return payload

    def apply_sync_payload(self, payload: dict) -> None:
        """Merge an incoming full-state payload produced by :meth:`get_sync_payload`.

        Each entry in *payload* is parsed and merged into the local store.
        """
        merged = 0
        for composite_key, data in payload.items():
            if "::" not in composite_key:
                logger.warning("Malformed sync key %r – skipping", composite_key)
                continue

            crdt_type, name = composite_key.split("::", 1)
            self.merge_remote(name, crdt_type, data)
            merged += 1

        logger.info("apply_sync_payload: merged %d CRDTs", merged)

    # -- Introspection -----------------------------------------------------

    def list_crdts(self) -> dict[str, list[str]]:
        """Return a mapping of CRDT type → list of names."""
        return {
            "PNCounter": list(self._counters.keys()),
            "ORSet": list(self._sets.keys()),
            "LWWMap": list(self._maps.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"CRDTStore(node_id={self.node_id!r}, "
            f"counters={len(self._counters)}, "
            f"sets={len(self._sets)}, "
            f"maps={len(self._maps)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level Singleton
# ═══════════════════════════════════════════════════════════════════════════

_crdt_store: CRDTStore | None = None


def get_crdt_store() -> CRDTStore:
    """Return the module-level :class:`CRDTStore` singleton.

    The singleton is lazily initialised on first call with a random
    node ID derived from the application instance.
    """
    global _crdt_store
    if _crdt_store is None:
        node_id = f"{settings.app_name}-{uuid.uuid4().hex[:8]}"
        _crdt_store = CRDTStore(node_id=node_id)
        logger.info("Global CRDTStore created (node_id=%s)", _crdt_store.node_id)
    return _crdt_store
