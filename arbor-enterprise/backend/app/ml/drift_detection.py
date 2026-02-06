"""Drift Detection Pipeline for ML models.

TIER 10 - Point 51: Drift Detection Pipeline

Monitors for distribution drift in:
- Embedding vectors (semantic drift)
- User queries (behavioral drift)
- Model predictions (performance drift)

Uses statistical tests to detect significant shifts.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift to monitor."""

    EMBEDDING = "embedding"
    QUERY = "query"
    PREDICTION = "prediction"
    BEHAVIORAL = "behavioral"


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    drift_type: DriftType
    severity: DriftSeverity
    score: float  # 0-1, higher = more drift
    p_value: float
    baseline_stats: dict
    current_stats: dict
    detected_at: datetime
    message: str


class EmbeddingDriftDetector:
    """Detects drift in embedding distributions.

    TIER 10 - Point 51: Vector space drift detection.

    Uses cosine similarity distribution to detect when new embeddings
    are significantly different from baseline.
    """

    # Threshold for flagging drift
    DRIFT_THRESHOLDS = {
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.35,
        DriftSeverity.CRITICAL: 0.5,
    }

    def __init__(self, baseline_embeddings: np.ndarray | None = None):
        """Initialize with baseline embeddings.

        Args:
            baseline_embeddings: Array of shape (n_samples, embedding_dim)
        """
        self.baseline = baseline_embeddings
        self.baseline_stats: dict[str, float] = {}

        if baseline_embeddings is not None:
            self._compute_baseline_stats()

    def _compute_baseline_stats(self) -> None:
        """Compute statistics for baseline embeddings."""
        if self.baseline is None:
            return

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(self.baseline, axis=1, keepdims=True)
        normalized = self.baseline / (norms + 1e-8)

        # Sample if too large
        if len(normalized) > 1000:
            idx = np.random.choice(len(normalized), 1000, replace=False)
            normalized = normalized[idx]

        similarities = normalized @ normalized.T
        upper_tri = similarities[np.triu_indices(len(similarities), k=1)]

        self.baseline_stats = {
            "mean_similarity": float(np.mean(upper_tri)),
            "std_similarity": float(np.std(upper_tri)),
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "n_samples": len(self.baseline),
        }

    def set_baseline(self, embeddings: np.ndarray) -> None:
        """Set new baseline embeddings."""
        self.baseline = embeddings
        self._compute_baseline_stats()
        logger.info(f"Baseline set with {len(embeddings)} samples")

    def detect(self, current_embeddings: np.ndarray) -> DriftResult:
        """Detect embedding drift against baseline.

        Uses two-sample statistical test on similarity distributions.

        Args:
            current_embeddings: New embeddings to check

        Returns:
            DriftResult with severity and details
        """
        if self.baseline is None or len(self.baseline) == 0:
            return DriftResult(
                drift_type=DriftType.EMBEDDING,
                severity=DriftSeverity.NONE,
                score=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
                detected_at=datetime.utcnow(),
                message="No baseline set",
            )

        # Compute current stats
        norms = np.linalg.norm(current_embeddings, axis=1, keepdims=True)
        normalized = current_embeddings / (norms + 1e-8)

        if len(normalized) > 100:
            idx = np.random.choice(len(normalized), 100, replace=False)
            normalized = normalized[idx]

        similarities = normalized @ normalized.T
        upper_tri = similarities[np.triu_indices(len(similarities), k=1)]

        current_stats = {
            "mean_similarity": float(np.mean(upper_tri)),
            "std_similarity": float(np.std(upper_tri)),
            "mean_norm": float(np.mean(norms)),
            "n_samples": len(current_embeddings),
        }

        # Compute drift score using normalized difference
        baseline_mean = self.baseline_stats["mean_similarity"]
        current_mean = current_stats["mean_similarity"]
        baseline_std = self.baseline_stats["std_similarity"]

        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
            drift_score = min(1.0, z_score / 3)  # Normalize to 0-1
        else:
            drift_score = abs(current_mean - baseline_mean)

        # Approximate p-value
        p_value = max(0.001, 1 - drift_score)

        # Determine severity
        severity = DriftSeverity.NONE
        for sev, threshold in sorted(
            self.DRIFT_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if drift_score >= threshold:
                severity = sev
                break

        message = (
            f"Embedding drift: similarity shifted from "
            f"{baseline_mean:.3f} to {current_mean:.3f}"
        )

        return DriftResult(
            drift_type=DriftType.EMBEDDING,
            severity=severity,
            score=drift_score,
            p_value=p_value,
            baseline_stats=self.baseline_stats,
            current_stats=current_stats,
            detected_at=datetime.utcnow(),
            message=message,
        )


class QueryDriftDetector:
    """Detects drift in user query patterns.

    TIER 10 - Point 51: Behavioral drift detection.
    """

    def __init__(self):
        self.baseline_distribution: dict[str, float] = {}
        self.category_counts: dict[str, int] = {}

    def set_baseline(self, query_categories: list[str]) -> None:
        """Set baseline query distribution."""
        total = len(query_categories)
        counts: dict[str, int] = {}

        for cat in query_categories:
            counts[cat] = counts.get(cat, 0) + 1

        self.baseline_distribution = {k: v / total for k, v in counts.items()}
        self.category_counts = counts

    def detect(self, recent_categories: list[str]) -> DriftResult:
        """Detect drift in query category distribution."""
        if not self.baseline_distribution:
            return DriftResult(
                drift_type=DriftType.QUERY,
                severity=DriftSeverity.NONE,
                score=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
                detected_at=datetime.utcnow(),
                message="No baseline set",
            )

        # Compute current distribution
        total = len(recent_categories)
        current_counts: dict[str, int] = {}

        for cat in recent_categories:
            current_counts[cat] = current_counts.get(cat, 0) + 1

        current_dist = {k: v / total for k, v in current_counts.items()}

        # Compute KL divergence (approximate)
        all_categories = set(self.baseline_distribution.keys()) | set(current_dist.keys())
        kl_div = 0.0

        for cat in all_categories:
            p = current_dist.get(cat, 0.001)
            q = self.baseline_distribution.get(cat, 0.001)
            kl_div += p * np.log(p / q + 1e-10)

        drift_score = min(1.0, kl_div / 2)  # Normalize

        severity = DriftSeverity.NONE
        if drift_score > 0.3:
            severity = DriftSeverity.HIGH
        elif drift_score > 0.15:
            severity = DriftSeverity.MEDIUM
        elif drift_score > 0.05:
            severity = DriftSeverity.LOW

        return DriftResult(
            drift_type=DriftType.QUERY,
            severity=severity,
            score=drift_score,
            p_value=max(0.001, 1 - drift_score),
            baseline_stats={"distribution": self.baseline_distribution},
            current_stats={"distribution": current_dist},
            detected_at=datetime.utcnow(),
            message=f"Query distribution KL divergence: {kl_div:.3f}",
        )


class DriftMonitor:
    """Unified drift monitoring service.

    TIER 10 - Point 51: Central drift detection.

    Usage:
        monitor = DriftMonitor()
        await monitor.check_all()
    """

    def __init__(self):
        self.embedding_detector = EmbeddingDriftDetector()
        self.query_detector = QueryDriftDetector()
        self.last_check: datetime | None = None
        self.alerts: list[DriftResult] = []

    async def check_all(self) -> list[DriftResult]:
        """Run all drift checks against current data.

        Fetches recent embeddings from Qdrant and query logs from Redis,
        then runs both embedding drift and query drift detection.
        """
        results = []

        try:
            from app.db.qdrant.client import get_async_qdrant_client

            qdrant = await get_async_qdrant_client()
            if qdrant:
                collection_info = await qdrant.get_collection("entities_vectors")
                points_count = collection_info.points_count or 0

                if points_count > 0:
                    sample = await qdrant.scroll(
                        collection_name="entities_vectors",
                        limit=min(200, points_count),
                        with_vectors=True,
                    )

                    if sample and sample[0]:
                        vectors = [p.vector for p in sample[0] if p.vector]
                        if vectors:
                            embedding_result = self.embedding_detector.detect(vectors)
                            results.append(embedding_result)
        except Exception as exc:
            logger.warning("Embedding drift check failed: %s", exc)

        try:
            import json

            from app.db.redis.client import get_redis_client

            client = await get_redis_client()
            if client:
                raw = await client.lrange("arbor:query_log", 0, 500)
                if raw:
                    queries = [json.loads(q) if isinstance(q, str) else q for q in raw]
                    query_result = self.query_detector.detect(queries)
                    results.append(query_result)
        except Exception as exc:
            logger.warning("Query drift check failed: %s", exc)

        self.last_check = datetime.utcnow()

        # Filter for significant drift
        self.alerts = [r for r in results if r.severity != DriftSeverity.NONE]

        return results

    def get_status(self) -> dict[str, Any]:
        """Get current drift monitoring status."""
        return {
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "active_alerts": len(self.alerts),
            "alerts": [
                {
                    "type": a.drift_type.value,
                    "severity": a.severity.value,
                    "score": a.score,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
