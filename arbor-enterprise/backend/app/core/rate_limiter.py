"""Tiered rate limiting using Redis."""

import logging

from fastapi import HTTPException, Request

from app.db.redis.client import RedisCache

logger = logging.getLogger(__name__)


class TieredRateLimiter:
    """Rate limiting based on user tier."""

    TIERS = {
        "free": {"rpm": 10, "daily": 100},
        "pro": {"rpm": 60, "daily": 1000},
        "enterprise": {"rpm": 1000, "daily": 50000},
        "admin": {"rpm": 10000, "daily": 1000000},
    }

    def __init__(self):
        self.cache = RedisCache()

    async def check_limit(self, user_id: str, tier: str = "free") -> None:
        """Check rate limits. Raises HTTPException if exceeded."""
        limits = self.TIERS.get(tier, self.TIERS["free"])

        # Check RPM (requests per minute)
        rpm_key = f"ratelimit:rpm:{user_id}"
        current_rpm = await self.cache.incr(rpm_key, ttl=60)
        if current_rpm > limits["rpm"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limits['rpm']} requests per minute",
            )

        # Check daily limit
        daily_key = f"ratelimit:daily:{user_id}"
        current_daily = await self.cache.incr(daily_key, ttl=86400)
        if current_daily > limits["daily"]:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit exceeded: {limits['daily']} requests per day",
            )


rate_limiter = TieredRateLimiter()


async def check_rate_limit(request: Request, user: dict | None = None) -> None:
    """Check rate limit for a request."""
    if user:
        user_id = user.get("sub", "anonymous")
        tier = user.get("tier", "free")
    else:
        user_id = request.client.host if request.client else "unknown"
        tier = "free"

    await rate_limiter.check_limit(user_id, tier)
