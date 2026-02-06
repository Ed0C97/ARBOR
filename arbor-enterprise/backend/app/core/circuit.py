"""Circuit Breaker implementation for external service resilience.

TIER 3 - Point 10: External Service Circuit Breakers

When an external service (Cohere, Qdrant, Neo4j) is down, the circuit breaker
prevents thread accumulation by failing fast instead of waiting for timeouts.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail immediately with fallback
- HALF_OPEN: Testing if service has recovered

Configuration:
- fail_max: Number of consecutive failures before opening circuit (default: 5)
- reset_timeout: Seconds to wait before attempting recovery (default: 60)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for monitoring circuit breaker health."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected while circuit is open
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changes: list[tuple[float, str]] = field(default_factory=list)


class CircuitBreakerError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, service_name: str, message: str = "Service temporarily unavailable"):
        self.service_name = service_name
        self.message = message
        super().__init__(f"[{service_name}] {message}")


class CircuitBreaker:
    """Circuit breaker for external service calls.

    Usage:
        breaker = CircuitBreaker("cohere", fail_max=5, reset_timeout=60)

        @breaker
        async def call_cohere():
            ...

        # Or manually
        async with breaker:
            await call_cohere()
    """

    def __init__(
        self,
        name: str,
        fail_max: int | None = None,
        reset_timeout: int | None = None,
        fallback: Callable[[], Any] | None = None,
        excluded_exceptions: tuple[type[Exception], ...] = (),
    ):
        """Initialize circuit breaker.

        Args:
            name: Service name for logging and monitoring
            fail_max: Failures before opening circuit (default from config)
            reset_timeout: Seconds before attempting recovery (default from config)
            fallback: Optional fallback function when circuit is open
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.fail_max = fail_max or settings.circuit_breaker_fail_max
        self.reset_timeout = reset_timeout or settings.circuit_breaker_reset_timeout
        self.fallback = fallback
        self.excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Circuit breaker statistics."""
        return self._stats

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if self._last_failure_time is None:
                return True
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(
                    f"Circuit breaker [{self.name}]: Transitioning to HALF_OPEN "
                    f"(reset_timeout={self.reset_timeout}s elapsed)"
                )
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes.append((time.time(), new_state.value))
        logger.info(f"Circuit breaker [{self.name}]: {old_state.value} -> {new_state.value}")

    def _on_success(self) -> None:
        """Handle successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Success in half-open means service recovered
            self._failure_count = 0
            self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._last_failure_time = time.time()

        # Don't count excluded exceptions
        if isinstance(error, self.excluded_exceptions):
            logger.debug(
                f"Circuit breaker [{self.name}]: Excluded exception {type(error).__name__}"
            )
            return

        self._failure_count += 1
        logger.warning(
            f"Circuit breaker [{self.name}]: Failure {self._failure_count}/{self.fail_max} - {error}"
        )

        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open means service still down
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.fail_max:
            # Too many failures, open the circuit
            self._transition_to(CircuitState.OPEN)

    async def _execute_with_fallback(self) -> Any:
        """Execute fallback when circuit is open."""
        self._stats.rejected_calls += 1
        if self.fallback:
            logger.debug(f"Circuit breaker [{self.name}]: Executing fallback")
            result = self.fallback()
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise CircuitBreakerError(
            self.name,
            f"Circuit is OPEN. Service unavailable. "
            f"Retry in {self.reset_timeout - (time.time() - (self._last_failure_time or 0)):.0f}s",
        )

    async def __aenter__(self):
        """Async context manager entry."""
        async with self._lock:
            if not self._should_allow_request():
                await self._execute_with_fallback()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            self._on_success()
        else:
            self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for circuit breaker protection."""

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with self._lock:
                if not self._should_allow_request():
                    return await self._execute_with_fallback()

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker [{self.name}]: Manually reset")

    def get_status(self) -> dict:
        """Get current circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "fail_max": self.fail_max,
            "reset_timeout": self.reset_timeout,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Pre-configured circuit breakers for external services
# ═══════════════════════════════════════════════════════════════════════════

# Cohere API (embedding + reranking)
cohere_circuit = CircuitBreaker(
    name="cohere",
    fail_max=5,
    reset_timeout=60,
    fallback=lambda: {"status": "degraded", "message": "Embedding service temporarily unavailable"},
)

# Qdrant vector search
qdrant_circuit = CircuitBreaker(
    name="qdrant",
    fail_max=3,
    reset_timeout=30,
    fallback=lambda: [],  # Return empty results
)

# Neo4j knowledge graph
neo4j_circuit = CircuitBreaker(
    name="neo4j",
    fail_max=3,
    reset_timeout=30,
    fallback=lambda: [],  # Return empty results
)

# Redis cache (non-critical, short timeout)
redis_circuit = CircuitBreaker(
    name="redis",
    fail_max=5,
    reset_timeout=15,
    fallback=lambda: None,  # Cache miss
)

# LLM provider (Gemini/OpenAI)
llm_circuit = CircuitBreaker(
    name="llm",
    fail_max=3,
    reset_timeout=60,
    fallback=lambda: "I apologize, but I'm temporarily unable to process your request. Please try again shortly.",
)


def get_all_circuit_statuses() -> dict[str, dict]:
    """Get status of all circuit breakers for monitoring."""
    return {
        "cohere": cohere_circuit.get_status(),
        "qdrant": qdrant_circuit.get_status(),
        "neo4j": neo4j_circuit.get_status(),
        "redis": redis_circuit.get_status(),
        "llm": llm_circuit.get_status(),
    }
