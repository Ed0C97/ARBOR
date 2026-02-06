"""Request-scoped context propagation for A.R.B.O.R. Enterprise.

Propagates trace IDs, user IDs, tenant IDs, and feature flags through all
async operations using ``contextvars``.  Provides structured logging that
auto-injects context fields, per-request cost tracking, and Starlette
middleware that wires everything together.

Typical usage::

    # In middleware (automatic via RequestContextMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # In application code
    from app.core.request_context import get_structured_logger, get_request_context

    logger = get_structured_logger(__name__)
    logger.info("Processing entity", entity_id="e-123")

    ctx = get_request_context()
    if ctx and ctx.cost_tracker.is_within_budget():
        result = await expensive_llm_call()
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import get_settings

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class CostTracker:
    """Tracks compute cost within a single request.

    Each recorded operation contributes to a running total that is checked
    against the request's ``cost_budget``.  Useful for capping LLM spend
    per user request.

    Usage::

        tracker = CostTracker(budget=1.0)
        tracker.record_cost("llm_call", 0.003)
        tracker.record_cost("embedding", 0.0001)
        assert tracker.is_within_budget()
    """

    def __init__(self, budget: float = 1.0) -> None:
        self._budget = budget
        self._costs: dict[str, float] = {}
        self._total: float = 0.0

    # -- recording ----------------------------------------------------------

    def record_cost(self, operation: str, amount: float) -> None:
        """Record a cost contribution for *operation*.

        Args:
            operation: Descriptive name (e.g. ``"llm_call"``, ``"embedding"``).
            amount: Dollar amount (or abstract cost units).
        """
        self._costs[operation] = self._costs.get(operation, 0.0) + amount
        self._total += amount

    # -- queries ------------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return the cumulative cost recorded so far."""
        return self._total

    def get_cost_breakdown(self) -> dict[str, float]:
        """Return a copy of the per-operation cost breakdown."""
        return dict(self._costs)

    def is_within_budget(self) -> bool:
        """Return ``True`` if total cost has not exceeded the budget."""
        return self._total <= self._budget

    @property
    def budget(self) -> float:
        """The budget ceiling for this tracker."""
        return self._budget

    @property
    def remaining(self) -> float:
        """Budget remaining (may be negative if already exceeded)."""
        return self._budget - self._total


# ---------------------------------------------------------------------------
# Request context dataclass
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    """Immutable-ish bag of per-request metadata.

    Automatically created by :class:`RequestContextMiddleware` and made
    available to all downstream handlers via :func:`get_request_context`.
    """

    # -- identifiers --------------------------------------------------------
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    span_id: str = ""
    parent_request_id: str | None = None

    # -- caller info --------------------------------------------------------
    user_id: str | None = None
    tenant_id: str | None = None
    user_role: str | None = None

    # -- feature flags & baggage -------------------------------------------
    feature_flags: dict[str, bool] = field(default_factory=dict)
    baggage: dict[str, str] = field(default_factory=dict)

    # -- timing & budget ----------------------------------------------------
    start_time: float = field(default_factory=time.time)
    deadline: float | None = None
    cost_budget: float = 1.0

    # -- runtime objects (not serialised) -----------------------------------
    cost_tracker: CostTracker = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:16]
        if self.cost_tracker is None:
            self.cost_tracker = CostTracker(budget=self.cost_budget)

    # -- helpers ------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the context was created."""
        return time.time() - self.start_time

    @property
    def is_expired(self) -> bool:
        """``True`` if the request has exceeded its deadline (if set)."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def as_log_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for structured log emission."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "user_role": self.user_role,
            "elapsed_ms": round(self.elapsed * 1000, 2),
        }

    def child_context(self, **overrides: Any) -> "RequestContext":
        """Create a child context for sub-operations.

        The child inherits the parent's trace_id and caller info but gets
        a fresh ``request_id``, ``span_id``, and ``start_time``.
        """
        defaults = {
            "trace_id": self.trace_id,
            "parent_request_id": self.request_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "user_role": self.user_role,
            "feature_flags": dict(self.feature_flags),
            "baggage": dict(self.baggage),
            "cost_budget": self.cost_tracker.remaining,
            "deadline": self.deadline,
        }
        defaults.update(overrides)
        return RequestContext(**defaults)


# ---------------------------------------------------------------------------
# ContextVar storage
# ---------------------------------------------------------------------------

_request_ctx: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def set_request_context(ctx: RequestContext) -> Any:
    """Bind *ctx* to the current async context.

    Returns a ``contextvars.Token`` that can be passed to
    ``_request_ctx.reset()`` to restore the previous value.
    """
    return _request_ctx.set(ctx)


def get_request_context() -> RequestContext | None:
    """Return the :class:`RequestContext` for the current task, or ``None``."""
    return _request_ctx.get()


def get_or_create_context() -> RequestContext:
    """Return the current context or lazily create a minimal one.

    Useful in code paths that may run both inside and outside a request
    (e.g. background workers or CLI scripts).
    """
    ctx = _request_ctx.get()
    if ctx is not None:
        return ctx
    ctx = RequestContext()
    _request_ctx.set(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Thin wrapper around :mod:`logging` that auto-injects request context.

    Every log call emits a JSON-serialisable ``extra`` dict containing the
    current ``request_id``, ``trace_id``, ``user_id``, and ``tenant_id``
    plus any additional keyword arguments supplied by the caller.

    Usage::

        logger = get_structured_logger(__name__)
        logger.info("Entity enriched", entity_id="e-42", score=0.97)
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    # -- internal -----------------------------------------------------------

    def _build_extra(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Merge caller-provided *extra* with context fields."""
        ctx = get_request_context()
        context_fields: dict[str, Any] = {}
        if ctx is not None:
            context_fields = ctx.as_log_dict()
        context_fields.update(extra)
        return context_fields

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        merged = self._build_extra(extra)
        # Emit as a structured JSON string appended to the human-readable msg
        # so that log aggregators can parse it while keeping readability.
        self._logger.log(
            level,
            "%s | %s",
            msg,
            json.dumps(merged, default=str),
            extra=merged,
        )

    # -- public API ---------------------------------------------------------

    def debug(self, msg: str, **extra: Any) -> None:
        """Log at DEBUG level with context fields."""
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        """Log at INFO level with context fields."""
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """Log at WARNING level with context fields."""
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        """Log at ERROR level with context fields."""
        self._log(logging.ERROR, msg, **extra)

    def exception(self, msg: str, **extra: Any) -> None:
        """Log at ERROR level with traceback and context fields."""
        merged = self._build_extra(extra)
        self._logger.exception(
            "%s | %s",
            msg,
            json.dumps(merged, default=str),
            extra=merged,
        )


def get_structured_logger(name: str) -> StructuredLogger:
    """Factory for :class:`StructuredLogger` instances.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A :class:`StructuredLogger` bound to the given name.
    """
    return StructuredLogger(name)


# ---------------------------------------------------------------------------
# Module-level logger (used by middleware below)
# ---------------------------------------------------------------------------

_logger = get_structured_logger(__name__)


# ---------------------------------------------------------------------------
# OpenTelemetry traceparent helpers
# ---------------------------------------------------------------------------


def _parse_traceparent(header: str) -> tuple[str, str]:
    """Extract trace_id and span_id from a W3C ``traceparent`` header.

    Format: ``{version}-{trace_id}-{parent_id}-{flags}``

    Returns:
        Tuple of ``(trace_id, span_id)``.  Falls back to empty strings if
        the header is malformed.
    """
    parts = header.strip().split("-")
    if len(parts) >= 4:
        return parts[1], parts[2]
    return "", ""


# ---------------------------------------------------------------------------
# Starlette middleware
# ---------------------------------------------------------------------------

# Paths that bypass context creation (health checks, docs, etc.).
_SKIP_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/healthz",
        "/ready",
        "/readyz",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that creates a :class:`RequestContext` per request.

    Extracts identifying headers, sets the context for the lifetime of the
    request, and enriches the response with ``X-Request-ID``.

    Recognised inbound headers:

    * ``X-Request-ID``   -- reused as ``request_id`` (generated if absent).
    * ``X-Trace-ID``     -- explicit trace id.
    * ``X-Span-ID``      -- explicit span id.
    * ``X-User-ID``      -- authenticated user identifier.
    * ``X-Tenant-ID``    -- tenant identifier.
    * ``X-User-Role``    -- caller role (``viewer``, ``curator``, ``admin``).
    * ``X-Feature-Flags``-- JSON-encoded ``dict[str, bool]``.
    * ``X-Cost-Budget``  -- maximum compute budget for this request.
    * ``traceparent``    -- W3C trace-context propagation.
    """

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        """Create context, dispatch, then clean up."""
        # Allow health/infra endpoints through without context overhead.
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        settings = get_settings()
        headers = request.headers

        # -- Extract identifiers from headers --------------------------------
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        trace_id = headers.get("x-trace-id", "")
        span_id = headers.get("x-span-id", "")

        # Fall back to OpenTelemetry traceparent if explicit headers absent.
        if not trace_id:
            traceparent = headers.get("traceparent", "")
            if traceparent:
                trace_id, span_id = _parse_traceparent(traceparent)

        user_id = headers.get("x-user-id")
        tenant_id = headers.get("x-tenant-id")
        user_role = headers.get("x-user-role")

        # Feature flags (JSON dict)
        feature_flags: dict[str, bool] = {}
        raw_flags = headers.get("x-feature-flags")
        if raw_flags:
            try:
                feature_flags = json.loads(raw_flags)
            except (json.JSONDecodeError, TypeError):
                _logger.warning(
                    "Malformed X-Feature-Flags header, ignoring",
                    raw_value=raw_flags,
                )

        # Cost budget
        cost_budget = 1.0
        raw_budget = headers.get("x-cost-budget")
        if raw_budget:
            try:
                cost_budget = float(raw_budget)
            except (ValueError, TypeError):
                pass

        # Deadline from total request timeout setting
        deadline = time.time() + settings.timeout_total_request

        # -- Build context ---------------------------------------------------
        ctx = RequestContext(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            tenant_id=tenant_id,
            user_role=user_role,
            feature_flags=feature_flags,
            cost_budget=cost_budget,
            deadline=deadline,
        )

        token = set_request_context(ctx)

        _logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)

            # Measure duration and record a baseline request cost.
            duration_ms = round(ctx.elapsed * 1000, 2)
            ctx.cost_tracker.record_cost("request_overhead", 0.0)

            _logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                total_cost=ctx.cost_tracker.get_total_cost(),
            )

            # Propagate request id back to the caller.
            response.headers["X-Request-ID"] = ctx.request_id
            if ctx.trace_id:
                response.headers["X-Trace-ID"] = ctx.trace_id

            return response

        except Exception:
            duration_ms = round(ctx.elapsed * 1000, 2)
            _logger.exception(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
            )
            raise

        finally:
            _request_ctx.reset(token)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def with_context(fn: F) -> F:
    """Decorator that ensures *fn* runs inside a :class:`RequestContext`.

    If no context is set when *fn* is called, a minimal context is created
    automatically.  Works with both sync and async callables.

    Usage::

        @with_context
        async def process_entity(entity_id: str):
            ctx = get_request_context()
            ...
    """

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = get_request_context()
            if ctx is not None:
                return await fn(*args, **kwargs)
            new_ctx = RequestContext()
            token = set_request_context(new_ctx)
            try:
                return await fn(*args, **kwargs)
            finally:
                _request_ctx.reset(token)

        return _async_wrapper  # type: ignore[return-value]

    else:

        @functools.wraps(fn)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = get_request_context()
            if ctx is not None:
                return fn(*args, **kwargs)
            new_ctx = RequestContext()
            token = set_request_context(new_ctx)
            try:
                return fn(*args, **kwargs)
            finally:
                _request_ctx.reset(token)

        return _sync_wrapper  # type: ignore[return-value]


def track_cost(operation_name: str, estimated_cost: float = 0.0) -> Callable[[F], F]:
    """Decorator that records compute cost for a function invocation.

    If the function completes successfully the *estimated_cost* is debited
    against the current request's :class:`CostTracker`.  Works with both
    sync and async callables.

    Args:
        operation_name: Human-readable label (e.g. ``"llm_call"``).
        estimated_cost: Cost units to record on each invocation.

    Usage::

        @track_cost("embedding_generation", 0.0001)
        async def generate_embedding(text: str) -> list[float]:
            ...
    """

    def decorator(fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_request_context()
                result = await fn(*args, **kwargs)
                if ctx is not None:
                    ctx.cost_tracker.record_cost(operation_name, estimated_cost)
                return result

            return _async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_request_context()
                result = fn(*args, **kwargs)
                if ctx is not None:
                    ctx.cost_tracker.record_cost(operation_name, estimated_cost)
                return result

            return _sync_wrapper  # type: ignore[return-value]

    return decorator
