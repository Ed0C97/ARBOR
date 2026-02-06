"""Bulkhead Isolation + Adaptive Load Shedding for A.R.B.O.R. Enterprise.

TIER 3 - Resilience: Bulkhead Pattern

Prevents resource exhaustion by isolating concurrent access to shared resources
(LLM calls, database queries, vector search, etc.) into separate pools. Each
pool has its own concurrency limit and queue depth, so a burst of slow LLM calls
cannot starve the database pool.

Components:
- Bulkhead: Async semaphore-based isolation per resource
- BulkheadRegistry: Pre-configured bulkheads for every ARBOR subsystem
- LoadShedder: Adaptive load shedding based on observed error rate / latency
- AdaptiveConcurrencyLimiter: TCP Vegas-inspired dynamic concurrency control

Usage:
    registry = get_bulkhead_registry()
    bulkhead = registry.get("llm")
    result = await bulkhead.execute(call_gemini, prompt)
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Exception
# ═══════════════════════════════════════════════════════════════════════════════


class BulkheadRejectedException(Exception):
    """Raised when a bulkhead cannot accept more work.

    This occurs when both the active semaphore and the waiting queue are full.
    Callers should treat this as a 503 / backpressure signal.
    """

    def __init__(self, bulkhead_name: str, message: str | None = None):
        self.bulkhead_name = bulkhead_name
        self.message = message or (
            f"Bulkhead [{bulkhead_name}] rejected request: "
            f"concurrency and queue limits exhausted"
        )
        super().__init__(self.message)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BulkheadConfig:
    """Configuration for a single bulkhead partition."""

    name: str
    max_concurrent: int
    max_queue: int
    timeout_seconds: float
    priority: int = 0  # Higher value = more important

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.max_queue < 0:
            raise ValueError("max_queue must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass
class BulkheadMetrics:
    """Runtime metrics snapshot for a bulkhead."""

    name: str
    active_count: int = 0
    queued_count: int = 0
    rejected_count: int = 0
    completed_count: int = 0
    timeout_count: int = 0
    avg_execution_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Bulkhead
# ═══════════════════════════════════════════════════════════════════════════════


class Bulkhead:
    """Async semaphore-based bulkhead for resource isolation.

    Each bulkhead enforces:
    - A hard concurrency ceiling via ``asyncio.Semaphore``
    - A bounded overflow queue so callers block briefly rather than reject
    - A per-call timeout to prevent indefinite waits
    - Live metrics for monitoring dashboards

    Example::

        bh = Bulkhead(BulkheadConfig("llm", max_concurrent=10, max_queue=20, timeout_seconds=30))
        result = await bh.execute(some_async_fn, arg1, key=val)
    """

    def __init__(self, config: BulkheadConfig) -> None:
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active_count: int = 0
        self._queued_count: int = 0
        self._rejected_count: int = 0
        self._completed_count: int = 0
        self._timeout_count: int = 0
        self._execution_times: deque[float] = deque(maxlen=500)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* within this bulkhead's concurrency limits.

        Raises:
            BulkheadRejectedException: When both concurrency and queue are full.
            asyncio.TimeoutError: When the configured timeout expires while
                waiting for a semaphore slot.
        """
        # Fast-reject: if we cannot even queue the request, fail immediately.
        async with self._lock:
            total_in_system = self._active_count + self._queued_count
            if total_in_system >= self._config.max_concurrent + self._config.max_queue:
                self._rejected_count += 1
                logger.warning(
                    "Bulkhead [%s] rejected: active=%d queued=%d (limits %d/%d)",
                    self._config.name,
                    self._active_count,
                    self._queued_count,
                    self._config.max_concurrent,
                    self._config.max_queue,
                )
                raise BulkheadRejectedException(self._config.name)
            self._queued_count += 1

        try:
            # Wait for a semaphore slot, respecting the timeout.
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                async with self._lock:
                    self._queued_count -= 1
                    self._timeout_count += 1
                logger.warning(
                    "Bulkhead [%s] timeout after %.1fs waiting for slot",
                    self._config.name,
                    self._config.timeout_seconds,
                )
                raise

            # Transition from queued to active.
            async with self._lock:
                self._queued_count -= 1
                self._active_count += 1

            start = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                self._semaphore.release()
                async with self._lock:
                    self._active_count -= 1
                    self._completed_count += 1
                    self._execution_times.append(elapsed_ms)
        except asyncio.TimeoutError:
            # Re-raise; already handled counters above.
            raise
        except BulkheadRejectedException:
            raise

    @property
    def metrics(self) -> BulkheadMetrics:
        """Return a point-in-time snapshot of bulkhead metrics."""
        avg_ms = 0.0
        if self._execution_times:
            avg_ms = sum(self._execution_times) / len(self._execution_times)
        return BulkheadMetrics(
            name=self._config.name,
            active_count=self._active_count,
            queued_count=self._queued_count,
            rejected_count=self._rejected_count,
            completed_count=self._completed_count,
            timeout_count=self._timeout_count,
            avg_execution_ms=round(avg_ms, 2),
        )

    def is_accepting(self) -> bool:
        """Return True if this bulkhead can accept at least one more request."""
        total = self._active_count + self._queued_count
        return total < self._config.max_concurrent + self._config.max_queue

    def utilization(self) -> float:
        """Return active-slot utilization as a fraction in [0, 1]."""
        if self._config.max_concurrent == 0:
            return 1.0
        return min(self._active_count / self._config.max_concurrent, 1.0)

    def __repr__(self) -> str:
        return (
            f"<Bulkhead {self._config.name!r} "
            f"active={self._active_count}/{self._config.max_concurrent} "
            f"queued={self._queued_count}/{self._config.max_queue}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bulkhead Registry
# ═══════════════════════════════════════════════════════════════════════════════

# Default configurations per ARBOR subsystem.
_DEFAULT_BULKHEAD_CONFIGS: list[BulkheadConfig] = [
    BulkheadConfig(
        name="llm",
        max_concurrent=10,
        max_queue=20,
        timeout_seconds=30.0,
        priority=5,
    ),
    BulkheadConfig(
        name="database",
        max_concurrent=30,
        max_queue=50,
        timeout_seconds=15.0,
        priority=8,
    ),
    BulkheadConfig(
        name="vector_search",
        max_concurrent=15,
        max_queue=25,
        timeout_seconds=10.0,
        priority=6,
    ),
    BulkheadConfig(
        name="graph_search",
        max_concurrent=10,
        max_queue=15,
        timeout_seconds=10.0,
        priority=6,
    ),
    BulkheadConfig(
        name="external_api",
        max_concurrent=5,
        max_queue=10,
        timeout_seconds=20.0,
        priority=3,
    ),
]


class BulkheadRegistry:
    """Central registry of named bulkheads.

    Each ARBOR subsystem (LLM, database, vector search, etc.) gets its own
    bulkhead so that a stall in one subsystem does not cascade to the others.
    """

    def __init__(self, configs: list[BulkheadConfig] | None = None) -> None:
        self._bulkheads: dict[str, Bulkhead] = {}
        for cfg in configs or _DEFAULT_BULKHEAD_CONFIGS:
            self._bulkheads[cfg.name] = Bulkhead(cfg)
        logger.info(
            "BulkheadRegistry initialised with partitions: %s",
            list(self._bulkheads.keys()),
        )

    def get(self, name: str) -> Bulkhead:
        """Return the bulkhead for *name*, or raise ``KeyError``."""
        try:
            return self._bulkheads[name]
        except KeyError:
            available = ", ".join(sorted(self._bulkheads.keys()))
            raise KeyError(
                f"No bulkhead named {name!r}. Available: {available}"
            ) from None

    def all_metrics(self) -> dict[str, BulkheadMetrics]:
        """Return metrics for every registered bulkhead."""
        return {name: bh.metrics for name, bh in self._bulkheads.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._bulkheads

    def __repr__(self) -> str:
        return f"<BulkheadRegistry partitions={list(self._bulkheads.keys())}>"


# Singleton -----------------------------------------------------------------

_registry_instance: BulkheadRegistry | None = None
_registry_lock = asyncio.Lock()


async def get_bulkhead_registry() -> BulkheadRegistry:
    """Return the singleton ``BulkheadRegistry``, creating it on first call."""
    global _registry_instance
    if _registry_instance is None:
        async with _registry_lock:
            # Double-checked locking.
            if _registry_instance is None:
                _registry_instance = BulkheadRegistry()
    return _registry_instance


# ═══════════════════════════════════════════════════════════════════════════════
# Load Shedder
# ═══════════════════════════════════════════════════════════════════════════════

# Priority levels (convention shared with BulkheadConfig.priority):
#   0 = background / best-effort
#   1-3 = low
#   4-6 = medium
#   7-9 = high / critical

_DEFAULT_SHEDDING_POLICY: dict[str, Any] = {
    "error_rate_threshold": 0.10,          # 10 % errors -> shed low-priority
    "latency_multiplier_threshold": 2.0,   # 2x target -> shed medium-priority
    "queue_depth_threshold": 100,          # Absolute queue depth -> shed non-critical
    "target_latency_ms": 500.0,            # Baseline target latency
    "health_window_seconds": 30.0,         # Sliding window for health calculation
}


class LoadShedder:
    """Adaptive load shedder that drops requests to preserve system stability.

    The shedder maintains a sliding window of recent request observations and
    computes an aggregate health score.  When health degrades, it progressively
    sheds lower-priority traffic first, preserving headroom for critical work.
    """

    def __init__(self, check_interval: float = 5.0) -> None:
        self._check_interval = check_interval
        self._policy: dict[str, Any] = dict(_DEFAULT_SHEDDING_POLICY)

        # Sliding observation window.
        self._window_seconds: float = self._policy["health_window_seconds"]
        self._observations: deque[tuple[float, float, bool, int]] = deque()
        # Each entry: (timestamp, latency_ms, is_success, priority)

        # Cached derived metrics (refreshed lazily).
        self._last_refresh: float = 0.0
        self._request_queue_depth: int = 0
        self._avg_latency_ms: float = 0.0
        self._error_rate: float = 0.0
        self._cpu_estimate: float = 0.0  # Approximation via queue depth heuristic.
        self._shedding_active: bool = False

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def record_request(
        self, latency_ms: float, is_success: bool, priority: int = 0
    ) -> None:
        """Record the outcome of a completed request."""
        now = time.monotonic()
        self._observations.append((now, latency_ms, is_success, priority))
        self._evict_old(now)

    def set_queue_depth(self, depth: int) -> None:
        """Update the current request-queue depth (set externally by middleware)."""
        self._request_queue_depth = depth

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def should_shed(self, priority: int = 0) -> bool:
        """Return ``True`` if a request at *priority* should be shed."""
        self._refresh_metrics()

        error_threshold = self._policy["error_rate_threshold"]
        latency_mult = self._policy["latency_multiplier_threshold"]
        target_latency = self._policy["target_latency_ms"]
        queue_threshold = self._policy["queue_depth_threshold"]

        # Level 1: High error rate -> shed low-priority (priority < 4).
        if self._error_rate > error_threshold and priority < 4:
            self._shedding_active = True
            logger.info(
                "LoadShedder: shedding priority=%d (error_rate=%.2f > %.2f)",
                priority,
                self._error_rate,
                error_threshold,
            )
            return True

        # Level 2: Latency blowup -> shed medium-priority (priority < 7).
        if (
            self._avg_latency_ms > target_latency * latency_mult
            and priority < 7
        ):
            self._shedding_active = True
            logger.info(
                "LoadShedder: shedding priority=%d (avg_latency=%.0fms > %.0fms)",
                priority,
                self._avg_latency_ms,
                target_latency * latency_mult,
            )
            return True

        # Level 3: Queue depth overload -> shed all non-critical (priority < 9).
        if self._request_queue_depth > queue_threshold and priority < 9:
            self._shedding_active = True
            logger.info(
                "LoadShedder: shedding priority=%d (queue_depth=%d > %d)",
                priority,
                self._request_queue_depth,
                queue_threshold,
            )
            return True

        self._shedding_active = False
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_load_status(self) -> dict[str, Any]:
        """Return a health summary suitable for ``/health`` endpoints."""
        self._refresh_metrics()
        health_score = self._compute_health_score()
        return {
            "health_score": health_score,
            "shedding_active": self._shedding_active,
            "error_rate": round(self._error_rate, 4),
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "request_queue_depth": self._request_queue_depth,
            "cpu_estimate": round(self._cpu_estimate, 2),
            "observations_in_window": len(self._observations),
            "thresholds": {
                "error_rate": self._policy["error_rate_threshold"],
                "latency_multiplier": self._policy["latency_multiplier_threshold"],
                "target_latency_ms": self._policy["target_latency_ms"],
                "queue_depth": self._policy["queue_depth_threshold"],
            },
        }

    def set_shedding_policy(self, policy: dict[str, Any]) -> None:
        """Override one or more shedding thresholds at runtime."""
        for key, value in policy.items():
            if key in self._policy:
                self._policy[key] = value
                logger.info("LoadShedder: policy %s -> %s", key, value)
            else:
                logger.warning("LoadShedder: unknown policy key %r ignored", key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_old(self, now: float) -> None:
        """Drop observations older than the sliding window."""
        cutoff = now - self._window_seconds
        while self._observations and self._observations[0][0] < cutoff:
            self._observations.popleft()

    def _refresh_metrics(self) -> None:
        """Recompute derived metrics from the observation window."""
        now = time.monotonic()
        if now - self._last_refresh < self._check_interval:
            return
        self._last_refresh = now
        self._evict_old(now)

        if not self._observations:
            self._avg_latency_ms = 0.0
            self._error_rate = 0.0
            self._cpu_estimate = 0.0
            return

        total = len(self._observations)
        latency_sum = 0.0
        error_count = 0
        for _, lat, ok, _ in self._observations:
            latency_sum += lat
            if not ok:
                error_count += 1

        self._avg_latency_ms = latency_sum / total
        self._error_rate = error_count / total

        # CPU estimate heuristic: combine queue depth and latency deviation.
        target = self._policy["target_latency_ms"]
        latency_pressure = min(self._avg_latency_ms / max(target, 1.0), 2.0) / 2.0
        queue_pressure = min(
            self._request_queue_depth / max(self._policy["queue_depth_threshold"], 1),
            1.0,
        )
        self._cpu_estimate = min((latency_pressure + queue_pressure) / 2.0, 1.0)

    def _compute_health_score(self) -> int:
        """Compute a 0-100 health score (100 = perfectly healthy)."""
        if not self._observations:
            return 100

        # Three weighted components.
        error_penalty = self._error_rate * 40  # max 40 points lost
        target = self._policy["target_latency_ms"]
        latency_ratio = self._avg_latency_ms / max(target, 1.0)
        latency_penalty = min(latency_ratio - 1.0, 1.0) * 30 if latency_ratio > 1.0 else 0.0
        queue_ratio = self._request_queue_depth / max(
            self._policy["queue_depth_threshold"], 1
        )
        queue_penalty = min(queue_ratio, 1.0) * 30

        score = 100.0 - error_penalty - latency_penalty - queue_penalty
        return max(0, min(100, int(round(score))))


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Concurrency Limiter (TCP Vegas-inspired)
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveConcurrencyLimiter:
    """Dynamically adjusts max concurrency based on observed latency.

    Inspired by TCP Vegas congestion control:
    - Maintain a running estimate of *base* (minimum) latency.
    - On each ``release()``, compare observed latency to *base*.
      - If latency is close to base (queue is empty) -> probe higher limit.
      - If latency is significantly above base (queue building) -> back off.

    This avoids hard-coded limits that are either too conservative (under-utilise
    resources) or too aggressive (cause latency spikes).
    """

    # Vegas alpha/beta thresholds (in number of "queued" requests inferred
    # from latency delta).
    _ALPHA = 3   # Below alpha queued -> increase limit
    _BETA = 6    # Above beta queued -> decrease limit

    def __init__(
        self,
        initial_limit: int = 20,
        min_limit: int = 5,
        max_limit: int = 200,
    ) -> None:
        if not (1 <= min_limit <= initial_limit <= max_limit):
            raise ValueError(
                f"Require 1 <= min_limit({min_limit}) <= initial_limit({initial_limit}) "
                f"<= max_limit({max_limit})"
            )
        self._limit = initial_limit
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._in_flight: int = 0
        self._lock = asyncio.Lock()

        # Latency tracking.
        self._base_latency_ms: float = 0.0  # Estimated minimum (no-queue) latency.
        self._latency_samples: deque[float] = deque(maxlen=200)
        self._smoothing_factor: float = 0.05  # EWMA for base latency updates.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> bool:
        """Attempt to acquire a concurrency slot.

        Returns ``True`` if a slot was granted, ``False`` otherwise (caller
        should back off or reject the request).
        """
        async with self._lock:
            if self._in_flight < self._limit:
                self._in_flight += 1
                return True
            return False

    async def release(self, latency_ms: float) -> None:
        """Release a concurrency slot and adjust the limit based on *latency_ms*."""
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._latency_samples.append(latency_ms)
            self._update_limit(latency_ms)

    @property
    def current_limit(self) -> int:
        """The current dynamic concurrency limit."""
        return self._limit

    @property
    def current_in_flight(self) -> int:
        """Number of requests currently in flight."""
        return self._in_flight

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_limit(self, latency_ms: float) -> None:
        """Adjust the concurrency limit using the Vegas algorithm."""
        # Bootstrap: use first sample as base latency.
        if self._base_latency_ms <= 0:
            self._base_latency_ms = latency_ms
            return

        # Update base latency via EWMA toward the minimum observed.
        if latency_ms < self._base_latency_ms:
            self._base_latency_ms = (
                self._smoothing_factor * latency_ms
                + (1 - self._smoothing_factor) * self._base_latency_ms
            )

        # Estimate number of requests "queued" behind us.
        if self._base_latency_ms > 0:
            expected = self._limit * (self._base_latency_ms / max(latency_ms, 0.01))
            diff = self._limit - expected  # Estimated queue length.
        else:
            diff = 0

        old_limit = self._limit

        if diff < self._ALPHA:
            # Latency is close to baseline -> headroom available, probe up.
            self._limit = min(self._limit + 1, self._max_limit)
        elif diff > self._BETA:
            # Latency elevated -> back off.
            self._limit = max(self._limit - 1, self._min_limit)
        # else: in the stable band, hold steady.

        if self._limit != old_limit:
            logger.debug(
                "AdaptiveConcurrencyLimiter: limit %d -> %d "
                "(latency=%.1fms base=%.1fms diff=%.1f)",
                old_limit,
                self._limit,
                latency_ms,
                self._base_latency_ms,
                diff,
            )
