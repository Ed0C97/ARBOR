"""Granular timeout management for LangGraph nodes.

TIER 3 - Point 11: Granular Node Timeouts

Provides timeout decorators and wrappers for agent nodes to prevent
zombie agents from blocking the entire chain indefinitely.

Budget:
- IntentNode: 2.0s
- SearchNode: 5.0s
- SynthesisNode: 15.0s
- Total Request: 25.0s (load balancer timeout)
"""

import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


class NodeTimeoutError(Exception):
    """Raised when a LangGraph node exceeds its timeout."""

    def __init__(self, node_name: str, timeout: float, message: str = ""):
        self.node_name = node_name
        self.timeout = timeout
        self.message = message or f"Node '{node_name}' exceeded timeout of {timeout}s"
        super().__init__(self.message)


async def with_timeout(
    coro,
    timeout: float,
    node_name: str = "unknown",
) -> Any:
    """Execute a coroutine with timeout.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        node_name: Name of the node (for logging)

    Returns:
        Result of the coroutine

    Raises:
        NodeTimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        logger.error(f"Node '{node_name}' timed out after {timeout}s")
        raise NodeTimeoutError(node_name, timeout)


def timeout(seconds: float, node_name: str | None = None) -> Callable:
    """Decorator to add timeout to async functions.

    Usage:
        @timeout(2.0, "intent_router")
        async def classify_intent(state):
            ...

    Args:
        seconds: Timeout in seconds
        node_name: Optional name for logging (defaults to function name)

    Returns:
        Decorated function with timeout protection
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = node_name or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except TimeoutError:
                logger.error(f"Node '{name}' timed out after {seconds}s")
                raise NodeTimeoutError(name, seconds)

        return wrapper

    return decorator


# Pre-configured timeout decorators for common node types
intent_timeout = timeout(settings.timeout_intent_node, "intent_router")
search_timeout = timeout(settings.timeout_search_node, "search_node")
synthesis_timeout = timeout(settings.timeout_synthesis_node, "synthesis_node")


class TimeoutBudget:
    """Manage timeout budget across multiple operations.

    Useful when a chain of operations must complete within a total budget,
    and you want to allocate remaining time to later operations.

    Usage:
        budget = TimeoutBudget(total=25.0)

        # First operation uses max 2s
        await budget.execute(classify_intent(), max_time=2.0, name="intent")

        # Second operation gets remaining time, max 10s
        await budget.execute(search(), max_time=10.0, name="search")

        # Final operation gets whatever's left
        await budget.execute(synthesize(), name="synthesis")
    """

    def __init__(self, total: float | None = None):
        """Initialize timeout budget.

        Args:
            total: Total budget in seconds (default from config)
        """
        self.total = total or settings.timeout_total_request
        self.remaining = self.total
        self.start_time = asyncio.get_event_loop().time()
        self.operations: list[dict] = []

    async def execute(
        self,
        coro,
        max_time: float | None = None,
        name: str = "operation",
    ) -> Any:
        """Execute operation within budget.

        Args:
            coro: Coroutine to execute
            max_time: Maximum time for this operation (None = use remaining)
            name: Operation name for logging

        Returns:
            Result of the operation

        Raises:
            NodeTimeoutError: If timeout exceeded
        """
        # Calculate actual timeout
        if max_time is None:
            timeout_seconds = self.remaining
        else:
            timeout_seconds = min(max_time, self.remaining)

        if timeout_seconds <= 0:
            raise NodeTimeoutError(name, 0, "Budget exhausted")

        operation_start = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)

            # Update remaining budget
            elapsed = asyncio.get_event_loop().time() - operation_start
            self.remaining -= elapsed

            self.operations.append(
                {
                    "name": name,
                    "elapsed": elapsed,
                    "timeout": timeout_seconds,
                    "success": True,
                }
            )

            return result

        except TimeoutError:
            elapsed = asyncio.get_event_loop().time() - operation_start
            self.remaining -= elapsed

            self.operations.append(
                {
                    "name": name,
                    "elapsed": elapsed,
                    "timeout": timeout_seconds,
                    "success": False,
                }
            )

            raise NodeTimeoutError(name, timeout_seconds)

    def get_remaining(self) -> float:
        """Get remaining budget in seconds."""
        return max(0, self.remaining)

    def get_elapsed(self) -> float:
        """Get total elapsed time."""
        return asyncio.get_event_loop().time() - self.start_time

    def get_summary(self) -> dict:
        """Get summary of all operations."""
        return {
            "total_budget": self.total,
            "elapsed": self.get_elapsed(),
            "remaining": self.get_remaining(),
            "operations": self.operations,
        }


def create_graceful_timeout_handler(
    default_response: Any = None,
    log_level: str = "warning",
) -> Callable:
    """Create a timeout handler that returns a default value instead of raising.

    Useful for non-critical operations where you want graceful degradation.

    Usage:
        handle_timeout = create_graceful_timeout_handler(default_response=[])

        @handle_timeout(5.0, "optional_search")
        async def search_optional_source():
            ...  # Returns [] on timeout instead of raising
    """

    def timeout_decorator(seconds: float, node_name: str = "unknown") -> Callable:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=seconds,
                    )
                except TimeoutError:
                    log_func = getattr(logger, log_level)
                    log_func(
                        f"Node '{node_name}' timed out after {seconds}s, "
                        f"returning default: {default_response}"
                    )
                    return default_response

            return wrapper

        return decorator

    return timeout_decorator


# Graceful handlers for non-critical nodes
graceful_search_timeout = create_graceful_timeout_handler(default_response=[])
graceful_enrichment_timeout = create_graceful_timeout_handler(default_response=None)
