"""Langfuse integration for LLM call tracing and evaluation."""

import functools
import logging
from collections.abc import Callable
from typing import Any

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_langfuse_client: Langfuse | None = None


def get_langfuse_client() -> Langfuse:
    """Return singleton Langfuse client.

    Returns:
        Configured Langfuse client instance.
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            enabled=bool(settings.langfuse_public_key and settings.langfuse_secret_key),
        )
        logger.info(
            "Langfuse client initialized: host=%s, enabled=%s",
            settings.langfuse_host,
            _langfuse_client.enabled,
        )
    return _langfuse_client


def trace_llm(
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Decorator that wraps an async function with Langfuse @observe tracing.

    Applies Langfuse's @observe decorator and injects custom metadata
    (model, task type, etc.) into the trace for downstream evaluation.

    Args:
        name: Override name for the trace span. Defaults to function name.
        metadata: Additional key-value pairs attached to the Langfuse trace.

    Returns:
        Decorated async function with Langfuse observation.
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @observe(name=span_name)
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Attach metadata to the current observation
            if metadata:
                langfuse_context.update_current_observation(metadata=metadata)

            langfuse_context.update_current_observation(
                metadata={
                    "service": settings.app_name,
                    "environment": settings.app_env,
                    **(metadata or {}),
                }
            )

            result = await fn(*args, **kwargs)

            # If the result contains token usage info, log it
            if isinstance(result, dict):
                usage = result.get("usage")
                if usage:
                    langfuse_context.update_current_observation(
                        usage={
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        }
                    )

            return result

        return wrapper

    return decorator


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Submit a score to Langfuse for a given trace.

    Used by the feedback loop to score LLM outputs after user interaction.

    Args:
        trace_id: The Langfuse trace ID to score.
        name: Score dimension name (e.g., "relevance", "user_satisfaction").
        value: Numeric score value.
        comment: Optional human-readable note.
    """
    client = get_langfuse_client()
    if not client.enabled:
        logger.debug("Langfuse disabled; skipping score for trace %s", trace_id)
        return

    client.score(
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment,
    )
    logger.debug("Langfuse score submitted: trace=%s, %s=%.2f", trace_id, name, value)


async def flush_langfuse() -> None:
    """Flush pending Langfuse events. Call during application shutdown."""
    global _langfuse_client
    if _langfuse_client is not None:
        _langfuse_client.flush()
        _langfuse_client.shutdown()
        _langfuse_client = None
        logger.info("Langfuse client flushed and shut down")
