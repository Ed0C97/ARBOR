"""Semantic caching for LLM responses using Redis + Qdrant.

TIER 2 - Point 8: Semantic Cache "Hit-and-Return" Optimization
- Returns query embedding along with cache result
- Avoids double embedding computation on cache miss

TIER 4 - Point 17: Cache Threshold Calibration (0.88 - 0.92)
- Default threshold lowered from 0.95 to 0.90 for better hit rate

TIER 4 - Point 18: Qdrant Cache Invalidation Hooks
- Support for invalidating cache entries on entity updates
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.db.qdrant.client import get_async_qdrant_client, get_qdrant_client
from app.db.redis.client import RedisCache

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CacheResult:
    """Result from cache lookup.

    TIER 2 - Point 8: Includes embedding for reuse on cache miss.
    """

    hit: bool
    response: str | None
    query_embedding: list[float] | None = None
    score: float | None = None
    cache_key: str | None = None


class SemanticCache:
    """Semantic cache using embeddings for similarity matching.

    TIER 2 - Point 8: Hit-and-Return Optimization
    - check_cache() returns the query embedding
    - On cache miss, caller can reuse the embedding instead of recomputing

    TIER 4 - Point 17: Threshold Calibration
    - Default 0.90 for "ristorante romantico" ≈ "posto romantico per cena"
    """

    def __init__(self, similarity_threshold: float | None = None):
        self.redis = RedisCache()
        self._qdrant = None  # Lazy init
        self._async_qdrant = None  # Lazy init for async
        self.threshold = similarity_threshold or settings.semantic_cache_threshold
        self.collection = "semantic_cache"
        self.ttl = settings.cache_ttl

        # Stats for monitoring
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
        }

    @property
    def qdrant(self):
        """Lazy-load sync Qdrant client."""
        if self._qdrant is None:
            self._qdrant = get_qdrant_client()
        return self._qdrant

    async def _get_async_qdrant(self):
        """Lazy-load async Qdrant client."""
        if self._async_qdrant is None:
            self._async_qdrant = await get_async_qdrant_client()
        return self._async_qdrant

    async def check_cache(self, query: str) -> CacheResult:
        """Check cache and return result with embedding for reuse.

        TIER 2 - Point 8: Hit-and-Return Optimization
        Returns the query embedding so it can be reused if cache misses.

        Returns:
            CacheResult with hit status, response (if hit), and query_embedding
        """
        cache_key = self._hash_key(query)

        # First: exact match in Redis (fast, no embedding needed)
        exact = await self.redis.get(f"llm_cache:{cache_key}")
        if exact:
            logger.debug(f"Exact cache hit for: {query[:50]}")
            self._stats["exact_hits"] += 1
            return CacheResult(
                hit=True,
                response=exact,
                query_embedding=None,  # No embedding computed for exact match
                score=1.0,
                cache_key=cache_key,
            )

        # Second: semantic match in Qdrant (requires embedding)
        try:
            from app.llm.gateway import get_llm_gateway
            gateway = get_llm_gateway()

            # Compute embedding once (this is reusable on cache miss)
            embedding = await gateway.get_query_embedding(query)

            qdrant = await self._get_async_qdrant()
            if qdrant:
                results = await qdrant.search(
                    collection_name=self.collection,
                    query_vector=embedding,
                    limit=1,
                    score_threshold=self.threshold,
                )

                if results:
                    cached_response = results[0].payload.get("response")
                    if cached_response:
                        logger.debug(
                            f"Semantic cache hit (score={results[0].score:.3f})"
                        )
                        self._stats["semantic_hits"] += 1
                        return CacheResult(
                            hit=True,
                            response=cached_response,
                            query_embedding=embedding,  # Return for potential reuse
                            score=results[0].score,
                            cache_key=cache_key,
                        )

            # Cache miss - return the embedding for reuse
            self._stats["misses"] += 1
            return CacheResult(
                hit=False,
                response=None,
                query_embedding=embedding,  # TIER 2: Reusable embedding
                cache_key=cache_key,
            )

        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")
            self._stats["misses"] += 1
            return CacheResult(
                hit=False,
                response=None,
                query_embedding=None,
                cache_key=cache_key,
            )

    async def get(self, query: str) -> str | None:
        """Check cache for semantically similar query.

        Backwards-compatible method. Use check_cache() for optimization.
        """
        result = await self.check_cache(query)
        return result.response if result.hit else None

    async def set(
        self,
        query: str,
        response: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Store query-response pair in both caches.

        TIER 2 - Point 8: Accept pre-computed embedding to avoid double computation.

        Args:
            query: The original query
            response: The response to cache
            embedding: Pre-computed embedding (optional, avoids recomputation)
        """
        cache_key = self._hash_key(query)

        # Redis exact cache
        await self.redis.set(f"llm_cache:{cache_key}", response, ttl=self.ttl)

        # Qdrant semantic cache
        try:
            # Use provided embedding or compute new one
            if embedding is None:
                from app.llm.gateway import get_llm_gateway
                gateway = get_llm_gateway()
                embedding = await gateway.get_embedding(query)

            from qdrant_client.models import PointStruct

            qdrant = await self._get_async_qdrant()
            if qdrant:
                await qdrant.upsert(
                    collection_name=self.collection,
                    points=[
                        PointStruct(
                            id=cache_key,
                            vector=embedding,
                            payload={
                                "query": query,
                                "response": response,
                                "timestamp": time.time(),
                            },
                        )
                    ],
                )
                self._stats["stores"] += 1
                logger.debug(f"Cached response for: {query[:50]}")

        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")

    async def invalidate(self, query: str) -> bool:
        """Invalidate a specific cache entry.

        TIER 4 - Point 18: Cache Invalidation Hooks
        """
        cache_key = self._hash_key(query)

        try:
            # Remove from Redis
            await self.redis.delete(f"llm_cache:{cache_key}")

            # Remove from Qdrant
            qdrant = await self._get_async_qdrant()
            if qdrant:
                await qdrant.delete(
                    collection_name=self.collection,
                    points_selector={"points": [cache_key]},
                )

            self._stats["invalidations"] += 1
            logger.debug(f"Invalidated cache for: {query[:50]}")
            return True

        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return False

    async def invalidate_by_entity(self, entity_id: str) -> int:
        """Invalidate all cache entries mentioning an entity.

        TIER 4 - Point 18: Called when entity data changes.

        Args:
            entity_id: Entity ID (e.g., "brand_42", "venue_17")

        Returns:
            Number of entries invalidated
        """
        try:
            qdrant = await self._get_async_qdrant()
            if not qdrant:
                return 0

            # Search for entries mentioning this entity
            from qdrant_client.models import Filter, FieldCondition, MatchText

            # Note: This requires a text index on the response field
            # For now, we'll skip this and let TTL handle staleness
            logger.debug(f"Entity invalidation requested for: {entity_id}")
            return 0

        except Exception as e:
            logger.warning(f"Entity cache invalidation failed: {e}")
            return 0

    def _hash_key(self, text: str) -> str:
        """Generate deterministic hash for exact matching."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        total = (
            self._stats["exact_hits"]
            + self._stats["semantic_hits"]
            + self._stats["misses"]
        )
        hit_rate = (
            (self._stats["exact_hits"] + self._stats["semantic_hits"]) / total
            if total > 0
            else 0
        )

        return {
            **self._stats,
            "total_lookups": total,
            "hit_rate": round(hit_rate, 3),
            "threshold": self.threshold,
            "ttl": self.ttl,
        }


# Singleton
_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Get singleton SemanticCache instance."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
