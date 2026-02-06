"""Embedding Space Monitoring for ARBOR Enterprise.

Monitors the geometry and health of vector embeddings in real-time,
detecting issues such as centroid drift, anisotropy, density anomalies,
query-document misalignment, and dimension collapse.

Designed to run alongside the main embedding pipeline so that
degradation in vector quality is surfaced before it impacts retrieval.
"""

import logging
import math
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert types and severity constants
# ---------------------------------------------------------------------------

ALERT_TYPES = frozenset({
    "centroid_drift",
    "density_anomaly",
    "anisotropy",
    "query_doc_misalignment",
    "dimension_collapse",
})

SEVERITY_LEVELS = ("info", "warning", "critical")

# Severity weights used when computing the aggregate health score.
_SEVERITY_WEIGHT: dict[str, float] = {
    "info": 1.0,
    "warning": 5.0,
    "critical": 15.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingAlert:
    """A single alert raised by an embedding health detector."""

    alert_id: str
    alert_type: str          # one of ALERT_TYPES
    severity: str            # one of SEVERITY_LEVELS
    metric_value: float
    threshold: float
    message: str
    detected_at: datetime
    entity_ids_affected: list[str] = field(default_factory=list)


@dataclass
class EmbeddingSnapshot:
    """Point-in-time summary of the embedding space geometry."""

    snapshot_id: str
    timestamp: datetime
    centroid: list[float]
    avg_pairwise_similarity: float
    avg_norm: float
    std_norm: float
    dimensionality: int
    sample_size: int
    anisotropy_score: float


# ---------------------------------------------------------------------------
# Helper functions (pure, stateless)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_centroid(vectors: list[list[float]]) -> list[float]:
    """Return the element-wise mean of *vectors*."""
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    centroid = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            centroid[i] += vec[i]
    return [c / n for c in centroid]


def _average_pairwise_similarity(
    vectors: list[list[float]],
    sample_size: int = 100,
) -> float:
    """Estimate average cosine similarity by sampling random pairs.

    If fewer pairs exist than *sample_size*, all pairs are evaluated.
    """
    n = len(vectors)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) // 2
    if total_pairs <= sample_size:
        # Evaluate all pairs.
        total_sim = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += _cosine_similarity(vectors[i], vectors[j])
                count += 1
        return total_sim / count if count > 0 else 0.0

    # Sample random pairs.
    total_sim = 0.0
    for _ in range(sample_size):
        i, j = random.sample(range(n), 2)
        total_sim += _cosine_similarity(vectors[i], vectors[j])
    return total_sim / sample_size


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

class CentroidDriftDetector:
    """Detect when the embedding centroid drifts from a stored baseline.

    Drift is measured as the cosine *distance* (1 - cosine_similarity)
    between the current centroid and the baseline centroid.
    """

    def __init__(self, drift_threshold: float = 0.05) -> None:
        self.drift_threshold = drift_threshold
        self._baseline_centroid: list[float] | None = None
        self._current_centroid: list[float] | None = None

    # -- public API ----------------------------------------------------------

    def set_baseline(self, centroid: list[float]) -> None:
        """Explicitly set the reference centroid."""
        self._baseline_centroid = list(centroid)
        logger.info(
            "CentroidDriftDetector baseline set (%d dimensions)",
            len(centroid),
        )

    def update(self, embeddings: list[list[float]]) -> None:
        """Recompute the current centroid from a batch of embeddings.

        If no baseline exists yet the first centroid becomes the baseline.
        """
        if not embeddings:
            return
        self._current_centroid = _compute_centroid(embeddings)
        if self._baseline_centroid is None:
            self._baseline_centroid = list(self._current_centroid)
            logger.info(
                "CentroidDriftDetector auto-initialised baseline from "
                "first batch (%d vectors)",
                len(embeddings),
            )

    def check_drift(self) -> Optional[EmbeddingAlert]:
        """Return an alert if centroid drift exceeds the threshold."""
        if self._baseline_centroid is None or self._current_centroid is None:
            return None

        similarity = _cosine_similarity(
            self._baseline_centroid, self._current_centroid,
        )
        distance = 1.0 - similarity

        if distance <= self.drift_threshold:
            return None

        severity = "critical" if distance > self.drift_threshold * 3 else "warning"

        return EmbeddingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="centroid_drift",
            severity=severity,
            metric_value=distance,
            threshold=self.drift_threshold,
            message=(
                f"Centroid drift detected: cosine distance {distance:.4f} "
                f"exceeds threshold {self.drift_threshold:.4f}"
            ),
            detected_at=datetime.now(timezone.utc),
        )


class AnisotropyDetector:
    """Detect when the embedding space becomes anisotropic.

    Anisotropy is the tendency of all vectors to point in the same
    direction.  A high average pairwise cosine similarity indicates
    that the space is collapsing — vectors are becoming less
    distinguishable.
    """

    # Thresholds for severity levels.
    CRITICAL_THRESHOLD: float = 0.9
    WARNING_THRESHOLD: float = 0.8

    def check(self, embeddings: list[list[float]]) -> Optional[EmbeddingAlert]:
        """Return an alert if the space is excessively anisotropic."""
        if len(embeddings) < 2:
            return None

        avg_sim = _average_pairwise_similarity(embeddings)

        if avg_sim > self.CRITICAL_THRESHOLD:
            severity = "critical"
        elif avg_sim > self.WARNING_THRESHOLD:
            severity = "warning"
        else:
            return None

        return EmbeddingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="anisotropy",
            severity=severity,
            metric_value=avg_sim,
            threshold=self.WARNING_THRESHOLD,
            message=(
                f"Anisotropy detected: average pairwise similarity "
                f"{avg_sim:.4f} indicates the embedding space is collapsing"
            ),
            detected_at=datetime.now(timezone.utc),
        )


class DensityAnomalyDetector:
    """Flag embeddings whose k-nearest-neighbour distance is anomalous.

    An embedding whose average kNN distance is more than 2 standard
    deviations above the mean is considered an outlier.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    # -- internal ------------------------------------------------------------

    def _compute_knn_distances(
        self,
        embeddings: list[list[float]],
        k: int = 5,
    ) -> list[float]:
        """Return the average distance to *k* nearest neighbours per vector.

        Distance is measured as cosine distance (1 - cosine_similarity).
        """
        n = len(embeddings)
        effective_k = min(k, n - 1)
        if effective_k <= 0:
            return [0.0] * n

        avg_distances: list[float] = []
        for i in range(n):
            distances: list[float] = []
            for j in range(n):
                if i == j:
                    continue
                dist = 1.0 - _cosine_similarity(embeddings[i], embeddings[j])
                distances.append(dist)
            distances.sort()
            knn_avg = sum(distances[:effective_k]) / effective_k
            avg_distances.append(knn_avg)

        return avg_distances

    # -- public API ----------------------------------------------------------

    def check(
        self,
        embeddings: list[list[float]],
        labels: list[str] | None = None,
    ) -> Optional[EmbeddingAlert]:
        """Return an alert if density anomalies are found."""
        if len(embeddings) < self.k + 1:
            return None

        knn_distances = self._compute_knn_distances(embeddings, k=self.k)

        mean_dist = sum(knn_distances) / len(knn_distances)
        variance = sum((d - mean_dist) ** 2 for d in knn_distances) / len(knn_distances)
        std_dist = math.sqrt(variance) if variance > 0 else 0.0

        if std_dist == 0.0:
            return None

        threshold = mean_dist + 2.0 * std_dist

        anomaly_indices = [
            i for i, d in enumerate(knn_distances) if d > threshold
        ]

        if not anomaly_indices:
            return None

        affected_ids: list[str] = []
        if labels:
            affected_ids = [
                labels[i] for i in anomaly_indices if i < len(labels)
            ]

        return EmbeddingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="density_anomaly",
            severity="warning" if len(anomaly_indices) < 5 else "critical",
            metric_value=float(len(anomaly_indices)),
            threshold=threshold,
            message=(
                f"{len(anomaly_indices)} embeddings have kNN distance > "
                f"2 std deviations (threshold={threshold:.4f})"
            ),
            detected_at=datetime.now(timezone.utc),
            entity_ids_affected=affected_ids,
        )


class QueryDocAlignmentDetector:
    """Detect divergence between query and document embedding distributions.

    When the centroid of query embeddings drifts far from the centroid
    of document embeddings the retrieval system will return less
    relevant results.
    """

    DISTANCE_THRESHOLD: float = 0.15

    def check(
        self,
        query_embeddings: list[list[float]],
        doc_embeddings: list[list[float]],
    ) -> Optional[EmbeddingAlert]:
        """Return an alert if query/doc distributions are misaligned."""
        if not query_embeddings or not doc_embeddings:
            return None

        query_centroid = _compute_centroid(query_embeddings)
        doc_centroid = _compute_centroid(doc_embeddings)

        similarity = _cosine_similarity(query_centroid, doc_centroid)
        distance = 1.0 - similarity

        if distance <= self.DISTANCE_THRESHOLD:
            return None

        severity = "critical" if distance > self.DISTANCE_THRESHOLD * 2 else "warning"

        return EmbeddingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="query_doc_misalignment",
            severity=severity,
            metric_value=distance,
            threshold=self.DISTANCE_THRESHOLD,
            message=(
                f"Query-document alignment degrading: centroid distance "
                f"{distance:.4f} exceeds threshold {self.DISTANCE_THRESHOLD:.4f}"
            ),
            detected_at=datetime.now(timezone.utc),
        )


class DimensionCollapseDetector:
    """Detect dimensions that carry no useful information.

    A dimension is considered collapsed when its variance across all
    embeddings falls below a minimum threshold (default 0.01).
    """

    VARIANCE_THRESHOLD: float = 0.01

    def check(self, embeddings: list[list[float]]) -> Optional[EmbeddingAlert]:
        """Return an alert listing collapsed dimensions, if any."""
        if not embeddings or len(embeddings) < 2:
            return None

        dim = len(embeddings[0])
        n = len(embeddings)

        collapsed_dims: list[int] = []

        for d in range(dim):
            mean = sum(vec[d] for vec in embeddings) / n
            variance = sum((vec[d] - mean) ** 2 for vec in embeddings) / n
            if variance < self.VARIANCE_THRESHOLD:
                collapsed_dims.append(d)

        if not collapsed_dims:
            return None

        ratio = len(collapsed_dims) / dim
        if ratio > 0.25:
            severity = "critical"
        elif ratio > 0.1:
            severity = "warning"
        else:
            severity = "info"

        return EmbeddingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="dimension_collapse",
            severity=severity,
            metric_value=float(len(collapsed_dims)),
            threshold=self.VARIANCE_THRESHOLD,
            message=(
                f"{len(collapsed_dims)}/{dim} dimensions collapsed "
                f"(variance < {self.VARIANCE_THRESHOLD}): "
                f"{collapsed_dims[:20]}{'...' if len(collapsed_dims) > 20 else ''}"
            ),
            detected_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EmbeddingSpaceMonitor:
    """Central orchestrator for all embedding health checks.

    Buffers incoming embeddings and runs all registered detectors
    on demand via :meth:`run_health_check`.
    """

    # Maximum number of embeddings retained in each buffer.
    _BUFFER_LIMIT: int = 2000

    def __init__(self) -> None:
        self._settings = get_settings()

        # Embedding buffers (bounded to prevent unbounded memory growth).
        self._doc_embeddings: deque[list[float]] = deque(maxlen=self._BUFFER_LIMIT)
        self._query_embeddings: deque[list[float]] = deque(maxlen=self._BUFFER_LIMIT)
        self._labels: deque[str] = deque(maxlen=self._BUFFER_LIMIT)

        # Detectors.
        self._centroid_detector = CentroidDriftDetector()
        self._anisotropy_detector = AnisotropyDetector()
        self._density_detector = DensityAnomalyDetector()
        self._alignment_detector = QueryDocAlignmentDetector()
        self._collapse_detector = DimensionCollapseDetector()

        # Alert history.
        self._alert_history: deque[EmbeddingAlert] = deque(maxlen=500)

        logger.info("EmbeddingSpaceMonitor initialised")

    # -- Recording -----------------------------------------------------------

    def record_embeddings(
        self,
        embeddings: list[list[float]],
        embedding_type: str = "document",
        labels: list[str] | None = None,
    ) -> None:
        """Buffer a batch of embeddings for future health checks.

        Args:
            embeddings: Vectors to record.
            embedding_type: Either ``"document"`` or ``"query"``.
            labels: Optional entity/document identifiers aligned with
                *embeddings* (used in density anomaly reports).
        """
        if embedding_type == "query":
            self._query_embeddings.extend(embeddings)
        else:
            self._doc_embeddings.extend(embeddings)
            if labels:
                self._labels.extend(labels)

        logger.debug(
            "Recorded %d %s embeddings (buffer: doc=%d, query=%d)",
            len(embeddings),
            embedding_type,
            len(self._doc_embeddings),
            len(self._query_embeddings),
        )

    def record_query_embeddings(self, embeddings: list[list[float]]) -> None:
        """Convenience wrapper for recording query embeddings."""
        self.record_embeddings(embeddings, embedding_type="query")

    # -- Health checks -------------------------------------------------------

    def run_health_check(self) -> list[EmbeddingAlert]:
        """Execute all detectors and return any alerts raised."""
        alerts: list[EmbeddingAlert] = []

        doc_list = list(self._doc_embeddings)
        query_list = list(self._query_embeddings)
        label_list = list(self._labels)

        if doc_list:
            # Centroid drift.
            self._centroid_detector.update(doc_list)
            drift_alert = self._centroid_detector.check_drift()
            if drift_alert is not None:
                alerts.append(drift_alert)

            # Anisotropy.
            aniso_alert = self._anisotropy_detector.check(doc_list)
            if aniso_alert is not None:
                alerts.append(aniso_alert)

            # Density anomaly.
            density_alert = self._density_detector.check(
                doc_list,
                labels=label_list if label_list else None,
            )
            if density_alert is not None:
                alerts.append(density_alert)

            # Dimension collapse.
            collapse_alert = self._collapse_detector.check(doc_list)
            if collapse_alert is not None:
                alerts.append(collapse_alert)

        # Query-document alignment (requires both buffers).
        if query_list and doc_list:
            align_alert = self._alignment_detector.check(query_list, doc_list)
            if align_alert is not None:
                alerts.append(align_alert)

        # Persist to history.
        self._alert_history.extend(alerts)

        if alerts:
            logger.warning(
                "Embedding health check raised %d alert(s): %s",
                len(alerts),
                ", ".join(a.alert_type for a in alerts),
            )
        else:
            logger.info("Embedding health check passed — no alerts")

        return alerts

    # -- Snapshot & reporting ------------------------------------------------

    def get_snapshot(self) -> EmbeddingSnapshot:
        """Capture a point-in-time snapshot of the embedding space."""
        doc_list = list(self._doc_embeddings)

        if not doc_list:
            return EmbeddingSnapshot(
                snapshot_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                centroid=[],
                avg_pairwise_similarity=0.0,
                avg_norm=0.0,
                std_norm=0.0,
                dimensionality=0,
                sample_size=0,
                anisotropy_score=0.0,
            )

        centroid = _compute_centroid(doc_list)
        avg_sim = _average_pairwise_similarity(doc_list)

        norms = [math.sqrt(sum(v * v for v in vec)) for vec in doc_list]
        avg_norm = sum(norms) / len(norms)
        variance = sum((n - avg_norm) ** 2 for n in norms) / len(norms)
        std_norm = math.sqrt(variance) if variance > 0 else 0.0

        return EmbeddingSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            centroid=centroid,
            avg_pairwise_similarity=avg_sim,
            avg_norm=avg_norm,
            std_norm=std_norm,
            dimensionality=len(doc_list[0]) if doc_list else 0,
            sample_size=len(doc_list),
            anisotropy_score=avg_sim,  # anisotropy approximated by avg similarity
        )

    def get_alert_history(self, limit: int = 100) -> list[EmbeddingAlert]:
        """Return the most recent alerts, newest first."""
        history = list(self._alert_history)
        history.reverse()
        return history[:limit]

    def get_health_score(self) -> float:
        """Compute a 0-100 health score based on recent alerts.

        Starts at 100 and subtracts a weighted penalty per alert.
        """
        score = 100.0
        for alert in self._alert_history:
            penalty = _SEVERITY_WEIGHT.get(alert.severity, 1.0)
            score -= penalty

        return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_monitor_instance: EmbeddingSpaceMonitor | None = None


def get_embedding_monitor() -> EmbeddingSpaceMonitor:
    """Return the singleton :class:`EmbeddingSpaceMonitor` instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = EmbeddingSpaceMonitor()
    return _monitor_instance
