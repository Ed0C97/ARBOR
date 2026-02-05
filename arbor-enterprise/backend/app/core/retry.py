"""Retry policies using Tenacity.

TIER 3 - Point 12: Tenacity Retry Policies

Provides exponential backoff with jitter for transient failures.
Integrates with circuit breakers for comprehensive resilience.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
# Retryable Exceptions
# ═══════════════════════════════════════════════════════════════════════════

# Network-related exceptions that are worth retrying
NETWORK_EXCEPTIONS = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadTimeout,
    ConnectionError,
    asyncio.TimeoutError,
    OSError,
)

# Database-related transient exceptions
DATABASE_EXCEPTIONS = (
    # SQLAlchemy transient errors
    Exception,  # Will be filtered by message content
)


def is_retryable_database_error(exception: Exception) -> bool:
    """Check if a database exception is retryable."""
    error_msg = str(exception).lower()
    retryable_patterns = [
        "connection refused",
        "connection reset",
        "connection timed out",
        "temporary failure",
        "server closed the connection",
        "ssl connection has been closed",
        "too many connections",
        "connection pool exhausted",
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)


# ═══════════════════════════════════════════════════════════════════════════
# Retry Decorators
# ═══════════════════════════════════════════════════════════════════════════


def retry_on_network_error(
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying on network errors.

    Uses jittered exponential backoff to prevent thundering herd.

    Args:
        max_attempts: Maximum retry attempts (default from config)
        initial_wait: Initial wait time in seconds (default from config)
        max_wait: Maximum wait time in seconds (default from config)

    Example:
        @retry_on_network_error(max_attempts=3)
        async def call_external_api():
            ...
    """
    _max_attempts = max_attempts or settings.retry_max_attempts
    _initial_wait = initial_wait or settings.retry_initial_wait
    _max_wait = max_wait or settings.retry_max_wait

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_max_attempts),
                wait=wait_exponential_jitter(initial=_initial_wait, max=_max_wait),
                retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_any_error(
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
    excluded_exceptions: tuple[type[Exception], ...] = (),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying on any error except excluded ones.

    Args:
        max_attempts: Maximum retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        excluded_exceptions: Exceptions that should not be retried

    Example:
        @retry_on_any_error(excluded_exceptions=(ValueError, KeyError))
        async def risky_operation():
            ...
    """
    _max_attempts = max_attempts or settings.retry_max_attempts
    _initial_wait = initial_wait or settings.retry_initial_wait
    _max_wait = max_wait or settings.retry_max_wait

    def should_retry(exception: Exception) -> bool:
        if isinstance(exception, excluded_exceptions):
            return False
        return True

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_max_attempts),
                wait=wait_exponential_jitter(initial=_initial_wait, max=_max_wait),
                retry=retry_if_exception_type(Exception),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Async Retry Helper
# ═══════════════════════════════════════════════════════════════════════════


async def with_retry(
    func: Callable[..., T],
    *args,
    max_attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
    retry_exceptions: tuple[type[Exception], ...] = NETWORK_EXCEPTIONS,
    **kwargs,
) -> T:
    """Execute a function with retry logic.

    Use this for one-off retries without decorators.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_attempts: Maximum retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        retry_exceptions: Tuple of exceptions to retry on
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function call

    Raises:
        RetryError: If all retries exhausted
        Exception: The last exception if retries exhausted

    Example:
        result = await with_retry(
            fetch_data,
            url="https://api.example.com",
            max_attempts=3
        )
    """
    _max_attempts = max_attempts or settings.retry_max_attempts
    _initial_wait = initial_wait or settings.retry_initial_wait
    _max_wait = max_wait or settings.retry_max_wait

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(_max_attempts),
        wait=wait_exponential_jitter(initial=_initial_wait, max=_max_wait),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return await func(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Service-Specific Retry Policies
# ═══════════════════════════════════════════════════════════════════════════


# Cohere API retry (embedding, reranking)
retry_cohere = retry_on_network_error(
    max_attempts=3,
    initial_wait=0.5,
    max_wait=5.0,
)

# Qdrant retry (vector search)
retry_qdrant = retry_on_network_error(
    max_attempts=3,
    initial_wait=0.3,
    max_wait=3.0,
)

# Neo4j retry (graph queries)
retry_neo4j = retry_on_network_error(
    max_attempts=3,
    initial_wait=0.5,
    max_wait=5.0,
)

# Redis retry (cache operations)
retry_redis = retry_on_network_error(
    max_attempts=2,
    initial_wait=0.1,
    max_wait=1.0,
)

# LLM API retry
retry_llm = retry_on_network_error(
    max_attempts=3,
    initial_wait=1.0,
    max_wait=10.0,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def create_retry_decorator(
    service_name: str,
    max_attempts: int = 3,
    initial_wait: float = 0.5,
    max_wait: float = 5.0,
    retry_exceptions: tuple[type[Exception], ...] = NETWORK_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Factory function to create a retry decorator for a specific service.

    Args:
        service_name: Name of the service (for logging)
        max_attempts: Maximum retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        retry_exceptions: Tuple of exceptions to retry on

    Returns:
        A retry decorator configured for the service

    Example:
        my_service_retry = create_retry_decorator("my_service", max_attempts=5)

        @my_service_retry
        async def call_my_service():
            ...
    """
    service_logger = logging.getLogger(f"{__name__}.{service_name}")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(service_logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)

        return wrapper

    return decorator
