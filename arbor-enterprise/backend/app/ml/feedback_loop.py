"""Feedback loop for collecting user signals and improving search ranking."""

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UserFeedback:
    """A single user feedback signal tied to a query-result pair."""

    user_id: str
    query: str
    entity_id: str
    action: str  # "click", "save", "convert", "dismiss"
    position: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    time_to_action_ms: float = 0.0
    session_id: str = ""
    reward: float = 0.0

    def __post_init__(self) -> None:
        """Compute the reward if it was not explicitly provided."""
        if self.reward == 0.0:
            self.reward = compute_reward(
                action=self.action,
                position=self.position,
                time_to_action_ms=self.time_to_action_ms,
            )


# ---------------------------------------------------------------------------
# Reward calculation
# ---------------------------------------------------------------------------


def compute_reward(
    action: str,
    position: int,
    time_to_action_ms: float = 0.0,
) -> float:
    """Calculate a scalar reward for a user interaction.

    The reward combines three components:

    1. **Action weight** - conversions are worth more than clicks.
    2. **Position discount** - interactions on lower-ranked results imply the
       user ignored higher results, so they carry a stronger relevance signal.
    3. **Dwell-time bonus** - slower, deliberate actions suggest genuine
       interest rather than accidental clicks.

    Args:
        action: Interaction type (click, save, convert, dismiss).
        position: 0-based rank position of the result.
        time_to_action_ms: Milliseconds between result display and interaction.

    Returns:
        A float reward in the range [0.0, 1.0].
    """
    action_weights: dict[str, float] = {
        "convert": 1.0,
        "save": 0.7,
        "click": 0.3,
        "dismiss": 0.0,
    }
    base = action_weights.get(action, 0.1)

    # Position discount using a gentle logarithmic decay
    # Position 0 -> 1.0, position 5 -> ~0.55, position 10 -> ~0.42
    position_factor = 1.0 / (1.0 + math.log1p(position))

    # Dwell-time bonus: actions taking > 2s score slightly higher
    dwell_bonus = 0.0
    if time_to_action_ms > 2000:
        dwell_bonus = min((time_to_action_ms - 2000) / 10000, 0.15)

    reward = base * position_factor + dwell_bonus
    return round(min(max(reward, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Feedback collector
# ---------------------------------------------------------------------------


class FeedbackCollector:
    """Collects, buffers, and persists user feedback for ranking improvement.

    Feedback signals are buffered in memory and flushed to the
    ``arbor_feedback`` table (via ``FeedbackRepository``) once the buffer
    reaches capacity. Aggregate statistics are logged on each flush.
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._buffer: list[UserFeedback] = []
        self._buffer_size = buffer_size
        self._total_collected: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record(self, feedback: UserFeedback) -> None:
        """Record a single feedback signal.

        Args:
            feedback: The user feedback dataclass to record.
        """
        self._buffer.append(feedback)
        self._total_collected += 1

        logger.debug(
            "Feedback recorded: user=%s entity=%s action=%s reward=%.4f (buffer=%d)",
            feedback.user_id,
            feedback.entity_id,
            feedback.action,
            feedback.reward,
            len(self._buffer),
        )

        if len(self._buffer) >= self._buffer_size:
            await self.flush()

    async def record_batch(self, samples: list[dict]) -> None:
        """Record a batch of feedback samples from the Kafka consumer.

        Args:
            samples: List of dicts with keys matching UserFeedback fields.
        """
        for sample in samples:
            fb = UserFeedback(
                user_id=sample.get("user_id", ""),
                query=sample.get("query", ""),
                entity_id=sample.get("entity_id", ""),
                action=sample.get("event_type", "click").split(".")[-1],
                position=sample.get("position", 0),
                reward=sample.get("reward", 0.0),
            )
            await self.record(fb)

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Flush the buffer, persist to the database, and log statistics.

        Writes each feedback record to the ``arbor_feedback`` table via
        ``FeedbackRepository``, then publishes the batch to Redis so the
        reranker can pick up the latest signals.
        """
        if not self._buffer:
            return

        batch = self._buffer.copy()
        self._buffer.clear()

        # Aggregate stats
        total_reward = sum(fb.reward for fb in batch)
        avg_reward = total_reward / len(batch)
        actions: dict[str, int] = {}
        for fb in batch:
            actions[fb.action] = actions.get(fb.action, 0) + 1

        logger.info(
            "Feedback flush: batch_size=%d avg_reward=%.4f actions=%s total_collected=%d",
            len(batch),
            avg_reward,
            actions,
            self._total_collected,
        )

        # Persist to arbor_feedback table
        try:
            from app.db.postgres.connection import arbor_session_factory
            from app.db.postgres.repository import FeedbackRepository

            if arbor_session_factory:
                async with arbor_session_factory() as session:
                    repo = FeedbackRepository(session)
                    for fb in batch:
                        await repo.create(
                            user_id=fb.user_id,
                            entity_type=(
                                fb.entity_id.split("_")[0] if "_" in fb.entity_id else "unknown"
                            ),
                            source_id=fb.entity_id,
                            query=fb.query,
                            action=fb.action,
                            position=fb.position,
                            reward=fb.reward,
                        )
                    await session.commit()
                    logger.info("Persisted %d feedback records to arbor_feedback", len(batch))
        except Exception as exc:
            logger.warning("Failed to persist feedback batch: %s", exc)

        # Publish to Redis for real-time reranker updates
        try:
            import json

            from app.db.redis.client import get_redis_client

            client = await get_redis_client()
            if client:
                summary = {
                    "batch_size": len(batch),
                    "avg_reward": round(avg_reward, 4),
                    "actions": actions,
                }
                await client.publish("arbor:feedback_flush", json.dumps(summary))
        except Exception as exc:
            logger.warning("Redis feedback publish failed: %s", exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return current collector statistics.

        Returns:
            Dict with buffer size, total collected, and average reward.
        """
        avg_reward = 0.0
        if self._buffer:
            avg_reward = sum(fb.reward for fb in self._buffer) / len(self._buffer)

        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer_size,
            "total_collected": self._total_collected,
            "current_avg_reward": round(avg_reward, 4),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_collector: FeedbackCollector | None = None


def get_feedback_collector() -> FeedbackCollector:
    """Return the singleton FeedbackCollector instance."""
    global _collector
    if _collector is None:
        _collector = FeedbackCollector()
    return _collector
