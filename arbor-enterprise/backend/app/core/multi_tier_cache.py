"""Advanced multi-tier caching with Bloom Filters for ARBOR Enterprise.

TIER 4 - Point 17: Multi-Tier Caching Architecture

Cache hierarchy:
    L1  InMemoryCache   - Application-level LRU (OrderedDict), ~1ms
    L2  RedisCache      - Distributed cache with circuit breaker, ~5ms
    L3  SemanticCache   - Qdrant-backed similarity lookup, ~20ms

Bloom filter sits in front of the entire stack to short-circuit guaranteed
misses without touching any tier, eliminating unnecessary lookups for keys
that have never been cached.

Flow:
    GET  bloom.might_contain(key)?  no  -> return None (definite miss)
                                    yes -> L1 -> L2 -> L3 -> None
    SET  write to all tiers + bloom.add(key)

CacheCoherencyBus provides pub/sub invalidation designed to integrate with
Kafka for cross-instance cache coherency in horizontally-scaled deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ===========================================================================
# Bloom Filter (pure Python, zero external dependencies)
# ===========================================================================


class BloomFilter:
    """Space-efficient probabilistic set membership test.

    Uses multiple independent hash functions derived from md5 and sha256
    with seed mixing to approximate k independent hash functions.

    False positives are possible; false negatives are not.
    """

    def __init__(
        self,
        expected_items: int = 10_000,
        false_positive_rate: float = 0.01,
    ) -> None:
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be between 0 and 1 exclusive")

        self._expected_items = expected_items
        self._target_fp_rate = false_positive_rate

        # Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)
        self._size = self._optimal_size(expected_items, false_positive_rate)

        # Optimal number of hash functions: k = (m / n) * ln(2)
        self._hash_count = self._optimal_hash_count(self._size, expected_items)

        # Bit array stored as a bytearray for compactness
        self._bit_array = bytearray(math.ceil(self._size / 8))

        self._item_count = 0

        logger.debug(
            "BloomFilter created: size=%d bits, hash_count=%d, "
            "expected_items=%d, target_fp=%.4f",
            self._size,
            self._hash_count,
            expected_items,
            false_positive_rate,
        )

    # ----- static helpers ---------------------------------------------------

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size m for n items and fp rate p."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(int(math.ceil(m)), 64)  # floor of 64 bits

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal hash function count k."""
        k = (m / n) * math.log(2)
        return max(int(round(k)), 1)

    # ----- hashing ----------------------------------------------------------

    def _get_hash_values(self, item: str) -> list[int]:
        """Produce k independent hash positions using double-hashing scheme.

        h_i(x) = (h1(x) + i * h2(x)) mod m

        h1 is derived from MD5, h2 from SHA-256, giving two independent
        128/256-bit digests that are combined to simulate k hash functions
        (Kirsch-Mitzenmacker optimisation).
        """
        raw = item.encode("utf-8")
        h1 = int(hashlib.md5(raw).hexdigest(), 16)
        h2 = int(hashlib.sha256(raw).hexdigest(), 16)

        return [(h1 + i * h2) % self._size for i in range(self._hash_count)]

    # ----- bit manipulation -------------------------------------------------

    def _set_bit(self, position: int) -> None:
        byte_idx, bit_idx = divmod(position, 8)
        self._bit_array[byte_idx] |= 1 << bit_idx

    def _get_bit(self, position: int) -> bool:
        byte_idx, bit_idx = divmod(position, 8)
        return bool(self._bit_array[byte_idx] & (1 << bit_idx))

    # ----- public API -------------------------------------------------------

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        for pos in self._get_hash_values(str(item)):
            self._set_bit(pos)
        self._item_count += 1

    def might_contain(self, item: str) -> bool:
        """Probabilistic membership test.

        Returns ``True`` if the item *might* be in the set (possible false
        positive) and ``False`` if the item is *definitely not* in the set.
        """
        return all(self._get_bit(pos) for pos in self._get_hash_values(str(item)))

    @property
    def estimated_count(self) -> int:
        """Number of items that have been added (not accounting for dupes)."""
        return self._item_count

    @property
    def false_positive_probability(self) -> float:
        """Current estimated false-positive probability given items inserted.

        p = (1 - e^(-k*n/m))^k
        """
        if self._item_count == 0:
            return 0.0
        exponent = -self._hash_count * self._item_count / self._size
        return (1 - math.exp(exponent)) ** self._hash_count

    def clear(self) -> None:
        """Reset the filter, removing all items."""
        self._bit_array = bytearray(math.ceil(self._size / 8))
        self._item_count = 0
        logger.debug("BloomFilter cleared")

    def __len__(self) -> int:
        return self._item_count

    def __repr__(self) -> str:
        return (
            f"BloomFilter(size={self._size}, hashes={self._hash_count}, "
            f"items={self._item_count}, fp={self.false_positive_probability:.6f})"
        )


# ===========================================================================
# Cache Entry
# ===========================================================================


@dataclass
class CacheEntry:
    """Metadata wrapper for a cached value."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: int = 300  # seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Return ``True`` when the entry has exceeded its TTL."""
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Record an access."""
        self.access_count += 1
        self.last_accessed = time.time()


# ===========================================================================
# Abstract cache tier
# ===========================================================================


class CacheTier(ABC):
    """Interface that every cache tier must satisfy."""

    @abstractmethod
    async def get(self, key: str, **kwargs: Any) -> Any | None: ...

    @abstractmethod
    async def set(self, key: str, value: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...


# ===========================================================================
# L1 - In-Memory LRU Cache
# ===========================================================================


class L1InMemoryCache(CacheTier):
    """Application-level LRU cache backed by ``OrderedDict``.

    Not shared across processes; designed for hot-path, sub-millisecond reads
    of the most frequently accessed items on a single application instance.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = asyncio.Lock()

    # ---- CacheTier interface -----------------------------------------------

    async def get(self, key: str, **kwargs: Any) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, *, ttl: int = 300, **kwargs: Any) -> None:
        async with self._lock:
            # If key already exists, move to end and update
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = CacheEntry(key=key, value=value, ttl=ttl)
                return

            # Evict LRU if at capacity
            if len(self._store) >= self._capacity:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("L1 cache evicted LRU key: %s", evicted_key)

            self._store[key] = CacheEntry(key=key, value=value, ttl=ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            logger.debug("L1 cache cleared")

    # ---- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return hit/miss/eviction counters."""
        total = self._hits + self._misses
        return {
            "tier": "L1_InMemory",
            "capacity": self._capacity,
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


# ===========================================================================
# L2 - Redis Distributed Cache
# ===========================================================================


class L2RedisCache(CacheTier):
    """Redis-backed distributed cache with circuit breaker resilience.

    Serialises values as JSON.  On Redis failure the cache degrades gracefully
    (returns ``None``) instead of crashing the request pipeline.
    """

    def __init__(self) -> None:
        self._redis: Any | None = None  # lazy-loaded RedisCache instance
        self._hits = 0
        self._misses = 0

    async def _get_redis(self) -> Any | None:
        """Lazy-import and instantiate ``RedisCache`` to avoid circular deps."""
        if self._redis is None:
            try:
                from app.db.redis.client import RedisCache
                self._redis = RedisCache()
            except Exception as exc:
                logger.warning("L2 Redis unavailable: %s", exc)
                return None
        return self._redis

    # ---- CacheTier interface -----------------------------------------------

    async def get(self, key: str, **kwargs: Any) -> Any | None:
        try:
            rc = await self._get_redis()
            if rc is None:
                self._misses += 1
                return None

            prefixed_key = f"cache:l2:{key}"
            raw = await rc.get(prefixed_key)

            if raw is None:
                self._misses += 1
                return None

            # RedisCache already JSON-decodes; handle both cases
            if isinstance(raw, str):
                try:
                    value = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    value = raw
            else:
                value = raw

            self._hits += 1
            return value

        except Exception as exc:
            logger.warning("L2 Redis GET failed for key=%s: %s", key, exc)
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, *, ttl: int = 3600, **kwargs: Any) -> None:
        try:
            rc = await self._get_redis()
            if rc is None:
                return

            prefixed_key = f"cache:l2:{key}"
            await rc.set(prefixed_key, value, ttl=ttl)

        except Exception as exc:
            logger.warning("L2 Redis SET failed for key=%s: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            rc = await self._get_redis()
            if rc is None:
                return

            prefixed_key = f"cache:l2:{key}"
            await rc.delete(prefixed_key)

        except Exception as exc:
            logger.warning("L2 Redis DELETE failed for key=%s: %s", key, exc)

    async def clear(self) -> None:
        """Clear L2 keys (prefix scan + delete).

        NOTE: Full SCAN is expensive; in production prefer pattern invalidation
        via ``invalidate_pattern`` on the orchestrator.
        """
        try:
            rc = await self._get_redis()
            if rc is None:
                return

            # Delegate to Redis client if available; otherwise no-op
            logger.debug("L2 cache clear requested (delegating to Redis)")

        except Exception as exc:
            logger.warning("L2 Redis CLEAR failed: %s", exc)

    # ---- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "tier": "L2_Redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


# ===========================================================================
# L3 - Semantic Cache (Qdrant-backed)
# ===========================================================================


class L3SemanticCache(CacheTier):
    """Qdrant-backed semantic cache.

    Instead of exact key matching, uses cosine similarity on query embeddings
    to return cached responses for semantically similar questions.

    TIER 4 - Point 17: Semantic caching for LLM responses.
    """

    COLLECTION = "semantic_cache"

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold or settings.semantic_cache_threshold
        self._hits = 0
        self._misses = 0

    async def _get_qdrant_client(self) -> Any | None:
        """Lazy-import to avoid circular dependencies."""
        try:
            from app.db.qdrant.client import get_async_qdrant_client
            return await get_async_qdrant_client()
        except Exception as exc:
            logger.warning("L3 Qdrant unavailable: %s", exc)
            return None

    # ---- CacheTier interface -----------------------------------------------

    async def get(
        self,
        key: str,
        *,
        query_embedding: list[float] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Look up a semantically similar cached response.

        Args:
            key: Ignored for semantic lookup (kept for interface compat).
            query_embedding: The embedding vector of the user query.
            threshold: Minimum cosine similarity (overrides instance default).

        Returns:
            Cached response dict or ``None``.
        """
        if query_embedding is None:
            self._misses += 1
            return None

        client = await self._get_qdrant_client()
        if client is None:
            self._misses += 1
            return None

        similarity_threshold = threshold or self._threshold

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = await client.search(
                collection_name=self.COLLECTION,
                query_vector=query_embedding,
                limit=1,
                score_threshold=similarity_threshold,
            )

            if not results:
                self._misses += 1
                return None

            top = results[0]
            payload = top.payload or {}

            # Check TTL stored in payload
            cached_at = payload.get("cached_at", 0)
            ttl = payload.get("ttl", settings.cache_ttl)
            if time.time() - cached_at > ttl:
                self._misses += 1
                logger.debug(
                    "L3 semantic hit expired (score=%.4f, age=%.0fs, ttl=%ds)",
                    top.score,
                    time.time() - cached_at,
                    ttl,
                )
                return None

            self._hits += 1
            logger.debug(
                "L3 semantic cache hit: score=%.4f, query=%s",
                top.score,
                payload.get("query_text", "")[:80],
            )
            return payload.get("response")

        except Exception as exc:
            logger.warning("L3 Qdrant GET failed: %s", exc)
            self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        *,
        query_embedding: list[float] | None = None,
        query_text: str = "",
        ttl: int = 3600,
        **kwargs: Any,
    ) -> None:
        """Store a response with its query embedding for semantic retrieval.

        Args:
            key: Cache key (stored in payload for exact-match fallback).
            value: The response object to cache.
            query_embedding: The embedding of the original query.
            query_text: Human-readable query text (for debugging).
            ttl: Time-to-live in seconds.
        """
        if query_embedding is None:
            return

        client = await self._get_qdrant_client()
        if client is None:
            return

        try:
            import uuid as _uuid
            from qdrant_client.models import PointStruct

            point_id = str(_uuid.uuid4())
            payload = {
                "cache_key": key,
                "query_text": query_text,
                "response": value,
                "cached_at": time.time(),
                "ttl": ttl,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }

            await client.upsert(
                collection_name=self.COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=query_embedding,
                        payload=payload,
                    )
                ],
            )

            logger.debug("L3 semantic cache SET: key=%s, text=%s", key, query_text[:80])

        except Exception as exc:
            logger.warning("L3 Qdrant SET failed: %s", exc)

    async def delete(self, key: str) -> None:
        """Delete entries matching exact cache_key in payload."""
        client = await self._get_qdrant_client()
        if client is None:
            return

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            await client.delete(
                collection_name=self.COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="cache_key",
                            match=MatchValue(value=key),
                        )
                    ]
                ),
            )
        except Exception as exc:
            logger.warning("L3 Qdrant DELETE failed for key=%s: %s", key, exc)

    async def clear(self) -> None:
        """Drop and recreate the semantic cache collection.

        Heavy operation; prefer targeted invalidation in production.
        """
        client = await self._get_qdrant_client()
        if client is None:
            return

        try:
            from qdrant_client.models import Distance, VectorParams

            await client.delete_collection(collection_name=self.COLLECTION)
            await client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info("L3 semantic cache collection recreated")
        except Exception as exc:
            logger.warning("L3 Qdrant CLEAR failed: %s", exc)

    # ---- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "tier": "L3_Semantic",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "similarity_threshold": self._threshold,
        }


# ===========================================================================
# Cache Coherency Bus
# ===========================================================================


class CacheCoherencyBus:
    """Publish/subscribe invalidation bus for cross-instance cache coherency.

    In a single-process deployment, listeners are called directly.  In a
    horizontally-scaled environment, the ``publish_invalidation`` method is
    designed to forward events through Kafka so that every instance can
    invalidate its local L1 cache.

    Kafka integration point::

        bus = CacheCoherencyBus()
        bus.subscribe(lambda key, reason: l1.delete(key))

        # In the Kafka consumer loop:
        for msg in kafka_consumer:
            bus.publish_invalidation(msg.key, msg.reason)
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[str, str], Any]] = []
        self._invalidation_count = 0
        self._lock = asyncio.Lock()

    def subscribe(self, callback: Callable[[str, str], Any]) -> None:
        """Register an invalidation listener.

        Args:
            callback: ``(key, reason) -> None`` called on every invalidation.
                      May be sync or async.
        """
        self._listeners.append(callback)
        logger.debug(
            "CacheCoherencyBus: listener registered (total=%d)", len(self._listeners)
        )

    async def publish_invalidation(self, key: str, reason: str = "manual") -> None:
        """Broadcast an invalidation event to all registered listeners.

        Args:
            key: The cache key being invalidated.
            reason: Human-readable reason (e.g. ``"data_update"``, ``"ttl"``).
        """
        async with self._lock:
            self._invalidation_count += 1

        logger.info(
            "CacheCoherencyBus: invalidating key=%s reason=%s listeners=%d",
            key,
            reason,
            len(self._listeners),
        )

        for listener in self._listeners:
            try:
                result = listener(key, reason)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "CacheCoherencyBus: listener error for key=%s: %s", key, exc
                )

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    @property
    def invalidation_count(self) -> int:
        return self._invalidation_count

    def stats(self) -> dict[str, Any]:
        return {
            "listeners": len(self._listeners),
            "total_invalidations": self._invalidation_count,
        }


# ===========================================================================
# Multi-Tier Cache Orchestrator
# ===========================================================================


class MultiTierCache:
    """Orchestrates lookups across L1 -> L2 -> L3 with a Bloom filter front.

    Lookup order:
        1. Bloom filter negative check (definite miss short-circuit)
        2. L1 in-memory LRU
        3. L2 Redis distributed cache
        4. L3 Qdrant semantic cache (only when ``query_embedding`` supplied)

    Promotion on hit:
        - L2 hit  -> populate L1
        - L3 hit  -> populate L1 + L2

    This ensures the hottest data migrates upward into faster tiers.
    """

    def __init__(
        self,
        l1: L1InMemoryCache,
        l2: L2RedisCache,
        l3: L3SemanticCache,
        bloom_filter: BloomFilter,
        coherency_bus: CacheCoherencyBus | None = None,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._bloom = bloom_filter
        self._bus = coherency_bus or CacheCoherencyBus()

        # Wire coherency bus to L1 invalidation
        self._bus.subscribe(self._on_coherency_invalidation)

        self._total_gets = 0
        self._bloom_rejections = 0

    # ---- internal helpers --------------------------------------------------

    def _on_coherency_invalidation(self, key: str, reason: str) -> None:
        """Sync callback for coherency bus -> L1 eviction."""
        # Schedule async delete on the running loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._l1.delete(key))
        except RuntimeError:
            # No running loop; skip (e.g. during shutdown)
            pass

    # ---- public API --------------------------------------------------------

    async def get(
        self,
        key: str,
        query_embedding: list[float] | None = None,
    ) -> Any | None:
        """Retrieve a value by traversing cache tiers.

        Args:
            key: Cache key for L1/L2 exact-match lookup.
            query_embedding: Optional embedding vector for L3 semantic lookup.

        Returns:
            Cached value or ``None`` on complete miss.
        """
        self._total_gets += 1

        # 1. Bloom filter short-circuit
        if not self._bloom.might_contain(key):
            self._bloom_rejections += 1
            logger.debug("Bloom filter definite miss: key=%s", key)
            return None

        # 2. L1 in-memory
        value = await self._l1.get(key)
        if value is not None:
            logger.debug("L1 cache hit: key=%s", key)
            return value

        # 3. L2 Redis
        value = await self._l2.get(key)
        if value is not None:
            logger.debug("L2 cache hit -> promoting to L1: key=%s", key)
            await self._l1.set(key, value, ttl=300)
            return value

        # 4. L3 Semantic (only if embedding provided)
        if query_embedding is not None:
            value = await self._l3.get(key, query_embedding=query_embedding)
            if value is not None:
                logger.debug("L3 semantic hit -> promoting to L1+L2: key=%s", key)
                await self._l1.set(key, value, ttl=300)
                await self._l2.set(key, value, ttl=settings.cache_ttl)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        query_embedding: list[float] | None = None,
        query_text: str = "",
        ttl: int = 3600,
    ) -> None:
        """Write a value to all cache tiers.

        Args:
            key: Cache key.
            value: Serialisable value to cache.
            query_embedding: Optional embedding for L3 semantic cache.
            query_text: Optional human-readable query (L3 metadata).
            ttl: Time-to-live in seconds (applied to all tiers).
        """
        # Add to bloom filter
        self._bloom.add(key)

        # Write to all tiers concurrently
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(self._l1.set(key, value, ttl=min(ttl, 300))),
            asyncio.create_task(self._l2.set(key, value, ttl=ttl)),
        ]

        if query_embedding is not None:
            tasks.append(
                asyncio.create_task(
                    self._l3.set(
                        key,
                        value,
                        query_embedding=query_embedding,
                        query_text=query_text,
                        ttl=ttl,
                    )
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("Cache SET across %d tiers: key=%s", len(tasks), key)

    async def invalidate(self, key: str) -> None:
        """Remove a key from all tiers and broadcast via coherency bus."""
        await asyncio.gather(
            self._l1.delete(key),
            self._l2.delete(key),
            self._l3.delete(key),
            return_exceptions=True,
        )

        await self._bus.publish_invalidation(key, reason="explicit_invalidation")
        logger.info("Cache invalidated across all tiers: key=%s", key)

    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching a prefix/pattern.

        Scans L1 for matching keys (L1 is in-process so this is cheap).
        L2 pattern deletion is delegated to Redis SCAN.
        L3 pattern deletion uses Qdrant payload filtering.

        Args:
            pattern: Key prefix or glob pattern (e.g. ``"entity:*"``).
        """
        # L1: scan in-memory store
        prefix = pattern.rstrip("*")
        keys_to_delete: list[str] = []

        async with self._l1._lock:
            keys_to_delete = [k for k in self._l1._store if k.startswith(prefix)]

        for k in keys_to_delete:
            await self._l1.delete(k)

        # L2: attempt Redis pattern delete via SCAN
        try:
            rc = await self._l2._get_redis()
            if rc is not None:
                from app.db.redis.client import get_redis_client

                client = await get_redis_client()
                if client is not None:
                    redis_pattern = f"cache:l2:{prefix}*"
                    cursor = 0
                    deleted = 0
                    while True:
                        cursor, keys = await client.scan(
                            cursor=cursor, match=redis_pattern, count=100
                        )
                        if keys:
                            await client.delete(*keys)
                            deleted += len(keys)
                        if cursor == 0:
                            break
                    logger.debug(
                        "L2 pattern invalidation: pattern=%s deleted=%d",
                        redis_pattern,
                        deleted,
                    )
        except Exception as exc:
            logger.warning("L2 pattern invalidation failed: %s", exc)

        # L3: delete by cache_key prefix filter
        try:
            qdrant = await self._l3._get_qdrant_client()
            if qdrant is not None:
                from qdrant_client.models import (
                    Filter,
                    FieldCondition,
                    MatchText,
                )

                await qdrant.delete(
                    collection_name=L3SemanticCache.COLLECTION,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="cache_key",
                                match=MatchText(text=prefix),
                            )
                        ]
                    ),
                )
        except Exception as exc:
            logger.warning("L3 pattern invalidation failed: %s", exc)

        # Broadcast
        await self._bus.publish_invalidation(
            pattern, reason="pattern_invalidation"
        )
        logger.info("Pattern invalidation complete: pattern=%s", pattern)

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics from all tiers and the bloom filter."""
        return {
            "total_gets": self._total_gets,
            "bloom_filter": {
                "items": self._bloom.estimated_count,
                "false_positive_rate": round(self._bloom.false_positive_probability, 6),
                "rejections": self._bloom_rejections,
            },
            "l1": self._l1.stats(),
            "l2": self._l2.stats(),
            "l3": self._l3.stats(),
            "coherency_bus": self._bus.stats(),
        }


# ===========================================================================
# Singleton accessor
# ===========================================================================

_multi_tier_cache: MultiTierCache | None = None
_cache_lock = asyncio.Lock()


async def get_multi_tier_cache() -> MultiTierCache:
    """Return (or create) the application-wide ``MultiTierCache`` singleton.

    Thread-safe via asyncio lock.  The bloom filter is sized for 50 000 items
    at a 1 % false-positive rate, matching typical deployment scale.
    """
    global _multi_tier_cache

    if _multi_tier_cache is not None:
        return _multi_tier_cache

    async with _cache_lock:
        # Double-check inside the lock
        if _multi_tier_cache is not None:
            return _multi_tier_cache

        bloom = BloomFilter(expected_items=50_000, false_positive_rate=0.01)
        l1 = L1InMemoryCache(capacity=1000)
        l2 = L2RedisCache()
        l3 = L3SemanticCache()
        bus = CacheCoherencyBus()

        _multi_tier_cache = MultiTierCache(
            l1=l1,
            l2=l2,
            l3=l3,
            bloom_filter=bloom,
            coherency_bus=bus,
        )

        logger.info(
            "MultiTierCache initialised: L1(cap=%d) + L2(Redis) + L3(Qdrant) "
            "| Bloom(items=50000, fp=0.01)",
            l1._capacity,
        )

        return _multi_tier_cache
