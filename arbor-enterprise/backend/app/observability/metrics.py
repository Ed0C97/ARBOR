"""Prometheus metrics for A.R.B.O.R. Enterprise runtime observability.

TIER 5 - Point 21: Business Metrics Implementation

Exposes business-relevant metrics for monitoring system health and usage.
"""

import functools
import logging
import time
from typing import Any, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TIER 5 - Point 21: Business Metrics
# ---------------------------------------------------------------------------

# Request counters by status
arbor_discover_requests_total = Counter(
    name="arbor_discover_requests_total",
    documentation="Total discovery requests by status and intent",
    labelnames=["status", "intent", "cache_hit"],
)

arbor_search_requests_total = Counter(
    name="arbor_search_requests_total",
    documentation="Total search requests by type and status",
    labelnames=["search_type", "status"],
)

# LLM latency histogram with fine-grained buckets
arbor_llm_latency_seconds = Histogram(
    name="arbor_llm_latency_seconds",
    documentation="LLM API call latency in seconds",
    labelnames=["provider", "model", "task_type"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
)

# Embedding latency
arbor_embedding_latency_seconds = Histogram(
    name="arbor_embedding_latency_seconds",
    documentation="Embedding generation latency in seconds",
    labelnames=["provider", "batch_size"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)

# Rerank latency
arbor_rerank_latency_seconds = Histogram(
    name="arbor_rerank_latency_seconds",
    documentation="Reranking latency in seconds",
    labelnames=["provider", "doc_count"],
    buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
)

# Active curators gauge
arbor_active_curators = Gauge(
    name="arbor_active_curators",
    documentation="Number of currently active curator sessions",
)

# Entity counts
arbor_entities_total = Gauge(
    name="arbor_entities_total",
    documentation="Total entities in the system by category",
    labelnames=["category", "city"],
)

# Circuit breaker status
arbor_circuit_breaker_state = Gauge(
    name="arbor_circuit_breaker_state",
    documentation="Circuit breaker state (0=closed, 1=open, 0.5=half-open)",
    labelnames=["service"],
)

# Database connection pool metrics
arbor_db_pool_connections = Gauge(
    name="arbor_db_pool_connections",
    documentation="Database connection pool status",
    labelnames=["database", "state"],  # state: active, idle, overflow
)

# Rate limiting metrics
arbor_rate_limit_hits_total = Counter(
    name="arbor_rate_limit_hits_total",
    documentation="Rate limit events by outcome",
    labelnames=["endpoint", "result"],  # result: allowed, blocked
)

# Guardrail metrics
arbor_guardrail_blocks_total = Counter(
    name="arbor_guardrail_blocks_total",
    documentation="Content blocked by guardrails",
    labelnames=["type", "reason"],  # type: input, output
)

# Service info
arbor_build_info = Info(
    name="arbor_build",
    documentation="A.R.B.O.R. build information",
)


# ---------------------------------------------------------------------------
# Original Metric definitions (kept for compatibility)
# ---------------------------------------------------------------------------

arbor_query_latency_seconds = Histogram(
    name="arbor_query_latency_seconds",
    documentation="End-to-end latency for search and discovery queries",
    labelnames=["endpoint", "intent", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

arbor_cache_hits_total = Counter(
    name="arbor_cache_hits_total",
    documentation="Total cache hit/miss events across Redis and LLM caches",
    labelnames=["cache_layer", "result"],
)

arbor_llm_tokens_used = Counter(
    name="arbor_llm_tokens_used",
    documentation="Cumulative LLM token consumption by model and direction",
    labelnames=["model", "direction"],
)

arbor_active_users = Gauge(
    name="arbor_active_users",
    documentation="Currently active user sessions",
    labelnames=["tier"],
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def record_cache_hit(cache_layer: str) -> None:
    """Increment cache hit counter.

    Args:
        cache_layer: Which cache was consulted (e.g., "redis", "llm_semantic").
    """
    arbor_cache_hits_total.labels(cache_layer=cache_layer, result="hit").inc()


def record_cache_miss(cache_layer: str) -> None:
    """Increment cache miss counter.

    Args:
        cache_layer: Which cache was consulted (e.g., "redis", "llm_semantic").
    """
    arbor_cache_hits_total.labels(cache_layer=cache_layer, result="miss").inc()


def record_llm_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage for a single LLM call.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
    """
    arbor_llm_tokens_used.labels(model=model, direction="input").inc(prompt_tokens)
    arbor_llm_tokens_used.labels(model=model, direction="output").inc(completion_tokens)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def track_latency(
    endpoint: str,
    intent: str = "unknown",
) -> Callable:
    """Decorator that records execution latency into the query histogram.

    Works with both sync and async callables.

    Args:
        endpoint: Logical name of the measured endpoint (e.g., "/api/v1/search").
        intent: Query intent classification label.

    Returns:
        Decorated callable with automatic latency recording.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = "ok"
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                arbor_query_latency_seconds.labels(
                    endpoint=endpoint,
                    intent=intent,
                    status=status,
                ).observe(elapsed)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = "ok"
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                arbor_query_latency_seconds.labels(
                    endpoint=endpoint,
                    intent=intent,
                    status=status,
                ).observe(elapsed)

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_llm_latency(
    provider: str,
    model: str,
    task_type: str = "completion",
) -> Callable:
    """Decorator that records LLM API call latency.

    TIER 5 - Point 21: Business Metrics

    Args:
        provider: LLM provider name (e.g., "gemini", "cohere")
        model: Model name
        task_type: Type of task (completion, embedding, rerank)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                arbor_llm_latency_seconds.labels(
                    provider=provider,
                    model=model,
                    task_type=task_type,
                ).observe(elapsed)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                arbor_llm_latency_seconds.labels(
                    provider=provider,
                    model=model,
                    task_type=task_type,
                ).observe(elapsed)

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# TIER 5 - Point 21: Business Metric Helpers
# ---------------------------------------------------------------------------


def record_discover_request(
    status: str,
    intent: str = "unknown",
    cache_hit: bool = False,
) -> None:
    """Record a discovery request.

    Args:
        status: Request status ("success", "error", "timeout")
        intent: User intent classification
        cache_hit: Whether the response came from cache
    """
    arbor_discover_requests_total.labels(
        status=status,
        intent=intent,
        cache_hit=str(cache_hit).lower(),
    ).inc()


def record_search_request(search_type: str, status: str) -> None:
    """Record a search request.

    Args:
        search_type: Type of search ("vector", "hybrid", "keyword")
        status: Request status
    """
    arbor_search_requests_total.labels(
        search_type=search_type,
        status=status,
    ).inc()


def record_embedding_latency(
    provider: str,
    batch_size: int,
    latency_seconds: float,
) -> None:
    """Record embedding generation latency."""
    arbor_embedding_latency_seconds.labels(
        provider=provider,
        batch_size=str(batch_size),
    ).observe(latency_seconds)


def record_rerank_latency(
    provider: str,
    doc_count: int,
    latency_seconds: float,
) -> None:
    """Record reranking latency."""
    arbor_rerank_latency_seconds.labels(
        provider=provider,
        doc_count=str(doc_count),
    ).observe(latency_seconds)


def set_circuit_breaker_state(service: str, state: str) -> None:
    """Update circuit breaker state gauge.

    Args:
        service: Service name (cohere, qdrant, neo4j, etc.)
        state: State (closed, open, half_open)
    """
    state_values = {"closed": 0.0, "open": 1.0, "half_open": 0.5}
    arbor_circuit_breaker_state.labels(service=service).set(state_values.get(state, 0.0))


def update_db_pool_metrics(database: str, active: int, idle: int, overflow: int) -> None:
    """Update database connection pool metrics."""
    arbor_db_pool_connections.labels(database=database, state="active").set(active)
    arbor_db_pool_connections.labels(database=database, state="idle").set(idle)
    arbor_db_pool_connections.labels(database=database, state="overflow").set(overflow)


def record_rate_limit(endpoint: str, allowed: bool) -> None:
    """Record rate limit event."""
    arbor_rate_limit_hits_total.labels(
        endpoint=endpoint,
        result="allowed" if allowed else "blocked",
    ).inc()


def record_guardrail_block(block_type: str, reason: str) -> None:
    """Record content blocked by guardrails."""
    arbor_guardrail_blocks_total.labels(
        type=block_type,
        reason=reason,
    ).inc()


def set_build_info(version: str, environment: str, git_sha: str = "unknown") -> None:
    """Set build information."""
    arbor_build_info.info(
        {
            "version": version,
            "environment": environment,
            "git_sha": git_sha,
        }
    )


def get_metrics() -> bytes:
    """Generate Prometheus metrics output.

    TIER 5 - Point 21: Returns metrics in Prometheus format.
    """
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST
