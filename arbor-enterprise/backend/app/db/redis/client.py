"""Redis async client for caching and rate limiting.

TIER 6 - Point 24: Redis Sliding Window Rate Limiter
- Implements sliding window log algorithm using Sorted Sets
- Prevents burst attacks across minute boundaries

TIER 7 - Point 32: API Idempotency (Redis Keys)
- Support for idempotency key storage
"""

import json
import logging
import time
from typing import Any

import redis.asyncio as redis

from app.config import get_settings
from app.core.circuit import redis_circuit
from app.core.retry import retry_redis

logger = logging.getLogger(__name__)
settings = get_settings()

_redis_client: redis.Redis | None = None


async def get_redis_client() -> redis.Redis | None:
    """Return singleton Redis client."""
    global _redis_client

    if not settings.redis_url:
        logger.warning("Redis URL not configured")
        return None

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    return _redis_client


async def close_redis_client():
    """Close Redis connection on shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


class RedisCache:
    """Redis cache operations with circuit breaker protection."""

    def __init__(self):
        self._client = None

    async def _get_client(self) -> redis.Redis | None:
        """Lazy-load Redis client."""
        if self._client is None:
            self._client = await get_redis_client()
        return self._client

    @redis_circuit
    @retry_redis
    async def get(self, key: str) -> Any | None:
        """Get a cached value."""
        client = await self._get_client()
        if not client:
            return None

        value = await client.get(key)
        if value:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return None

    @redis_circuit
    @retry_redis
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set a cached value with TTL in seconds."""
        client = await self._get_client()
        if not client:
            return

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await client.set(key, value, ex=ttl)

    @redis_circuit
    async def delete(self, key: str) -> None:
        """Delete a cached key."""
        client = await self._get_client()
        if not client:
            return
        await client.delete(key)

    @redis_circuit
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        client = await self._get_client()
        if not client:
            return False
        return bool(await client.exists(key))

    @redis_circuit
    async def incr(self, key: str, ttl: int | None = None) -> int:
        """Increment a counter. Used for rate limiting."""
        client = await self._get_client()
        if not client:
            return 0

        count = await client.incr(key)
        if count == 1 and ttl:
            await client.expire(key, ttl)
        return count

    async def get_session(self, session_id: str) -> dict | None:
        """Get session data."""
        return await self.get(f"session:{session_id}")

    async def set_session(self, session_id: str, data: dict, ttl: int = 86400) -> None:
        """Store session data (24h default TTL)."""
        await self.set(f"session:{session_id}", data, ttl=ttl)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter using Redis Sorted Sets.

    TIER 6 - Point 24: Prevents burst attacks across minute boundaries.

    Algorithm:
    1. ZREMRANGEBYSCORE: Remove old timestamps
    2. ZCARD: Count remaining requests
    3. If under limit, ZADD new request timestamp

    Unlike fixed window, this prevents the "double burst" problem where
    a user could make 100 requests at 0:59 and 100 more at 1:01.
    """

    def __init__(self):
        self._client = None

    async def _get_client(self) -> redis.Redis | None:
        """Lazy-load Redis client."""
        if self._client is None:
            self._client = await get_redis_client()
        return self._client

    @redis_circuit
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Check if request is allowed under rate limit.

        Args:
            key: Rate limit key (e.g., "rate_limit:user_123")
            max_requests: Maximum requests allowed in window
            window_seconds: Window size in seconds

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        client = await self._get_client()
        if not client:
            # If Redis is down, allow the request (fail open)
            return True, max_requests

        now = time.time()
        window_start = now - window_seconds

        # Use pipeline for atomicity
        async with client.pipeline(transaction=True) as pipe:
            # Remove old entries
            await pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            await pipe.zcard(key)
            # Add new entry (will be committed only if allowed)
            await pipe.zadd(key, {str(now): now})
            # Set expiry
            await pipe.expire(key, window_seconds)

            results = await pipe.execute()

        current_count = results[1]  # zcard result

        if current_count >= max_requests:
            # Over limit - remove the entry we just added
            await client.zrem(key, str(now))
            return False, 0

        remaining = max_requests - current_count - 1
        return True, max(0, remaining)

    @redis_circuit
    async def get_usage(
        self,
        key: str,
        window_seconds: int,
    ) -> tuple[int, float]:
        """Get current usage stats.

        Args:
            key: Rate limit key
            window_seconds: Window size

        Returns:
            Tuple of (request_count, oldest_request_age)
        """
        client = await self._get_client()
        if not client:
            return 0, 0.0

        now = time.time()
        window_start = now - window_seconds

        # Clean and count
        await client.zremrangebyscore(key, 0, window_start)
        count = await client.zcard(key)

        # Get oldest entry
        oldest = await client.zrange(key, 0, 0, withscores=True)
        if oldest:
            oldest_age = now - oldest[0][1]
        else:
            oldest_age = 0.0

        return count, oldest_age


class IdempotencyStore:
    """Store for API idempotency keys.

    TIER 7 - Point 32: Prevents duplicate operations from double-clicks.

    Usage:
        store = IdempotencyStore()
        existing = await store.check_and_set(key, response)
        if existing:
            return existing  # Return cached response
        # Process request normally
    """

    def __init__(self, ttl: int = 86400):  # 24 hour default
        self._client = None
        self.ttl = ttl

    async def _get_client(self) -> redis.Redis | None:
        """Lazy-load Redis client."""
        if self._client is None:
            self._client = await get_redis_client()
        return self._client

    @redis_circuit
    async def check_and_set(
        self,
        idempotency_key: str,
        response: dict | None = None,
    ) -> dict | None:
        """Check if idempotency key exists, set if not.

        Args:
            idempotency_key: Unique key from client
            response: Response to cache (set after processing)

        Returns:
            Existing cached response if key exists, None otherwise
        """
        client = await self._get_client()
        if not client:
            return None

        key = f"idempotency:{idempotency_key}"

        # Check if exists
        existing = await client.get(key)
        if existing:
            try:
                return json.loads(existing)
            except (json.JSONDecodeError, TypeError):
                return {"_raw": existing}

        # If response provided, set it
        if response is not None:
            await client.set(key, json.dumps(response), ex=self.ttl)

        return None

    @redis_circuit
    async def set_response(
        self,
        idempotency_key: str,
        response: dict,
    ) -> None:
        """Store response for idempotency key."""
        client = await self._get_client()
        if not client:
            return

        key = f"idempotency:{idempotency_key}"
        await client.set(key, json.dumps(response), ex=self.ttl)


async def check_redis_health() -> dict[str, Any]:
    """Health check for Redis service."""
    client = await get_redis_client()
    if not client:
        return {"status": "not_configured", "healthy": False}

    try:
        await client.ping()
        info = await client.info("server")
        return {
            "status": "healthy",
            "healthy": True,
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "healthy": False,
            "error": str(e),
        }
