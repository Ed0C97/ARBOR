"""SLO (Service Level Objectives) and Error Budget framework for A.R.B.O.R. Enterprise.

Provides in-memory tracking of request outcomes against predefined service
level objectives and calculates error budgets with burn-rate alerting.

Pre-defined SLOs:
    - api_availability    : 99.9% of all requests succeed
    - discover_latency    : P95 latency < 2000ms for /discover
    - search_latency      : P95 latency < 500ms for /search
    - error_rate          : < 1% 5xx errors globally
    - discovery_quality   : 95% of discovery responses have confidence > 0.5

Usage:
    monitor = get_slo_monitor()
    monitor.record_request("/discover", latency_ms=320.5, is_success=True)
    status = monitor.get_slo_status("api_availability")
    budget = monitor.get_error_budget("api_availability")
"""

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SLOType(Enum):
    """Classification of service level objective types."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SLODefinition:
    """Immutable definition of a service level objective.

    Attributes:
        slo_id: Unique identifier for the SLO.
        name: Human-readable name.
        description: Explanation of what the SLO measures.
        slo_type: Category of the SLO.
        target: Numeric target value.  Interpretation depends on *slo_type*:
            - AVAILABILITY : fraction, e.g. 0.999 for 99.9%
            - LATENCY      : milliseconds (P95 threshold)
            - ERROR_RATE   : fraction, e.g. 0.01 for 1%
            - THROUGHPUT   : requests per second
            - QUALITY      : fraction, e.g. 0.95 for 95%
        window_seconds: Measurement window in seconds (default 86400 = 1 day).
        endpoint: Specific endpoint path this SLO applies to, or ``None``
            for a global (all-endpoint) SLO.
        tier: Importance tier - one of "critical", "high", "medium", "low".
    """

    slo_id: str
    name: str
    description: str
    slo_type: SLOType
    target: float
    window_seconds: int = 86_400
    endpoint: Optional[str] = None
    tier: str = "medium"


@dataclass
class SLOMetric:
    """Point-in-time measurement of an SLO.

    Attributes:
        slo_id: Which SLO this metric belongs to.
        timestamp: When the measurement was taken (UTC).
        total_requests: Total requests observed in the window.
        good_requests: Requests that satisfy the SLO criterion.
        current_value: Computed ratio or percentile value.
        is_meeting_target: Whether *current_value* meets the SLO target.
    """

    slo_id: str
    timestamp: datetime
    total_requests: int
    good_requests: int
    current_value: float
    is_meeting_target: bool


@dataclass
class ErrorBudget:
    """Error budget status for a given SLO within its measurement window.

    Attributes:
        slo_id: Which SLO this budget belongs to.
        window_start: Start of the measurement window (UTC).
        window_end: End of the measurement window (UTC).
        total_budget: Allowed fraction of bad requests (1 - target).
        consumed: Fraction of the budget consumed so far.
        remaining: Fraction of the budget still available.
        burn_rate: How fast the budget is being consumed relative to the
            sustainable rate.  1.0 means the budget will be exactly exhausted
            at the end of the window; >1.0 means faster than sustainable.
        is_exhausted: ``True`` when *remaining* <= 0.
        projected_exhaustion_time: Estimated UTC datetime when the budget
            will be fully consumed at the current burn rate, or ``None``
            if the budget is not being consumed.
    """

    slo_id: str
    window_start: datetime
    window_end: datetime
    total_budget: float
    consumed: float
    remaining: float
    burn_rate: float
    is_exhausted: bool
    projected_exhaustion_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class _RequestRecord:
    """A single recorded request outcome."""

    timestamp: float  # time.monotonic()
    endpoint: str
    latency_ms: float
    is_success: bool
    is_quality_pass: bool


@dataclass
class _EndpointBucket:
    """Aggregated counters for a single endpoint within a time window."""

    total: int = 0
    successes: int = 0
    failures: int = 0
    quality_passes: int = 0
    latencies: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# BurnRateCalculator
# ---------------------------------------------------------------------------

class BurnRateCalculator:
    """Utility methods for error-budget burn-rate analysis."""

    @staticmethod
    def calculate_burn_rate(consumed: float, elapsed_fraction: float) -> float:
        """Return the burn rate given budget consumed and time elapsed.

        A burn rate of 1.0 means the budget is being consumed exactly in
        line with the window duration.  Values > 1.0 indicate that the
        budget will be exhausted before the window ends.

        Args:
            consumed: Fraction of total error budget consumed (0.0 - 1.0+).
            elapsed_fraction: Fraction of the window that has elapsed
                (0.0 - 1.0).

        Returns:
            The burn-rate multiplier.  Returns 0.0 when *elapsed_fraction*
            is essentially zero.
        """
        if elapsed_fraction < 1e-9:
            return 0.0
        return consumed / elapsed_fraction

    @staticmethod
    def project_exhaustion(
        remaining: float,
        burn_rate: float,
        window_remaining_seconds: float,
    ) -> Optional[datetime]:
        """Project when the error budget will be fully consumed.

        Args:
            remaining: Fraction of total budget remaining (0.0 - 1.0).
            burn_rate: Current burn-rate multiplier.
            window_remaining_seconds: Seconds remaining in the window.

        Returns:
            Projected UTC datetime of budget exhaustion, or ``None`` if the
            burn rate is zero or the budget will not be exhausted within the
            remaining window.
        """
        if burn_rate <= 0.0 or remaining <= 0.0:
            return None

        # At the current burn rate, how many seconds until exhaustion?
        # burn_rate = consumed_fraction / elapsed_fraction
        # If burn_rate = 1.0, budget lasts exactly the full window.
        # Remaining budget is consumed at rate (burn_rate / window_total).
        # But we only have window_remaining_seconds left.
        if burn_rate < 1e-9:
            return None

        # Fraction consumed per second at current rate, normalised over the
        # remaining window.  We use the proportion: if burn_rate is how fast
        # we consume budget per unit-window-fraction, then seconds until
        # remaining is gone = remaining / burn_rate * window_remaining / 1.0
        # Simplification: seconds_to_exhaust = (remaining / burn_rate) * window_remaining_seconds
        # but that only works when elapsed_fraction is the normaliser.
        # Cleaner: consumption_per_second = burn_rate / window_remaining_seconds (if we
        # treat the remaining window as the reference).  Actually, the
        # cleanest formulation:
        #   seconds_to_exhaust = remaining * window_remaining_seconds / burn_rate
        # This gives us: if burn_rate == 1.0 and remaining == 1.0, we
        # exhaust exactly at the window end.
        seconds_to_exhaust = (remaining / burn_rate) * window_remaining_seconds
        if seconds_to_exhaust > window_remaining_seconds:
            # Budget will NOT be exhausted within the window.
            return None

        return datetime.now(timezone.utc) + timedelta(seconds=seconds_to_exhaust)

    @staticmethod
    def classify_severity(burn_rate: float) -> str:
        """Classify the severity of a burn rate.

        Args:
            burn_rate: Current burn-rate multiplier.

        Returns:
            One of ``"healthy"``, ``"warning"``, ``"critical"``, or
            ``"exhausted"``.
        """
        if burn_rate <= 0.0:
            return "healthy"
        if burn_rate < 1.0:
            return "healthy"
        if burn_rate < 2.0:
            return "warning"
        if burn_rate < 10.0:
            return "critical"
        return "exhausted"


# ---------------------------------------------------------------------------
# SLOMonitor
# ---------------------------------------------------------------------------

class SLOMonitor:
    """In-memory SLO tracker with error-budget and burn-rate analysis.

    Thread-safe: all mutable state is protected by a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._definitions: dict[str, SLODefinition] = {}
        self._records: list[_RequestRecord] = []
        self._calculator = BurnRateCalculator()
        self._start_time = time.monotonic()
        self._start_utc = datetime.now(timezone.utc)

        # Register built-in SLOs
        self._register_default_slos()
        logger.info("SLOMonitor initialised with %d default SLOs", len(self._definitions))

    # -- Default SLOs -------------------------------------------------------

    def _register_default_slos(self) -> None:
        """Register the pre-defined A.R.B.O.R. SLO set."""
        defaults = [
            SLODefinition(
                slo_id="api_availability",
                name="API Availability",
                description="99.9% of all API requests return a successful response",
                slo_type=SLOType.AVAILABILITY,
                target=0.999,
                window_seconds=86_400,
                endpoint=None,
                tier="critical",
            ),
            SLODefinition(
                slo_id="discover_latency",
                name="Discovery Latency (P95)",
                description="95th percentile latency for /discover stays below 2000ms",
                slo_type=SLOType.LATENCY,
                target=2000.0,
                window_seconds=86_400,
                endpoint="/discover",
                tier="high",
            ),
            SLODefinition(
                slo_id="search_latency",
                name="Search Latency (P95)",
                description="95th percentile latency for /search stays below 500ms",
                slo_type=SLOType.LATENCY,
                target=500.0,
                window_seconds=86_400,
                endpoint="/search",
                tier="high",
            ),
            SLODefinition(
                slo_id="error_rate",
                name="Error Rate",
                description="Less than 1% of requests result in 5xx errors",
                slo_type=SLOType.ERROR_RATE,
                target=0.01,
                window_seconds=86_400,
                endpoint=None,
                tier="critical",
            ),
            SLODefinition(
                slo_id="discovery_quality",
                name="Discovery Quality",
                description="95% of discovery responses have confidence > 0.5",
                slo_type=SLOType.QUALITY,
                target=0.95,
                window_seconds=86_400,
                endpoint="/discover",
                tier="medium",
            ),
        ]
        for defn in defaults:
            self._definitions[defn.slo_id] = defn

    # -- Public API ---------------------------------------------------------

    def define_slo(self, definition: SLODefinition) -> None:
        """Register or update an SLO definition.

        Args:
            definition: The SLO to register.
        """
        with self._lock:
            self._definitions[definition.slo_id] = definition
        logger.info("SLO defined: %s (%s)", definition.slo_id, definition.name)

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        is_success: bool,
        is_quality_pass: bool = True,
    ) -> None:
        """Record a single request outcome.

        Args:
            endpoint: The API endpoint path (e.g. ``"/discover"``).
            latency_ms: Request latency in milliseconds.
            is_success: ``True`` if the request was successful (non-5xx).
            is_quality_pass: ``True`` if the response met quality criteria
                (e.g. confidence > threshold).  Defaults to ``True``.
        """
        record = _RequestRecord(
            timestamp=time.monotonic(),
            endpoint=endpoint,
            latency_ms=latency_ms,
            is_success=is_success,
            is_quality_pass=is_quality_pass,
        )
        with self._lock:
            self._records.append(record)

    def get_slo_status(self, slo_id: str) -> SLOMetric:
        """Compute the current status of a specific SLO.

        Args:
            slo_id: The identifier of the SLO to check.

        Returns:
            An :class:`SLOMetric` snapshot.

        Raises:
            KeyError: If *slo_id* is not registered.
        """
        with self._lock:
            defn = self._definitions[slo_id]
            bucket = self._aggregate(defn)
        return self._compute_metric(defn, bucket)

    def get_error_budget(self, slo_id: str) -> ErrorBudget:
        """Compute the error budget for a specific SLO.

        Args:
            slo_id: The identifier of the SLO to check.

        Returns:
            An :class:`ErrorBudget` snapshot.

        Raises:
            KeyError: If *slo_id* is not registered.
        """
        with self._lock:
            defn = self._definitions[slo_id]
            bucket = self._aggregate(defn)
        return self._compute_budget(defn, bucket)

    def get_all_slo_statuses(self) -> list[SLOMetric]:
        """Return current status for every registered SLO.

        Returns:
            List of :class:`SLOMetric` snapshots, one per SLO.
        """
        with self._lock:
            definitions = list(self._definitions.values())
            buckets = {defn.slo_id: self._aggregate(defn) for defn in definitions}
        return [
            self._compute_metric(defn, buckets[defn.slo_id])
            for defn in definitions
        ]

    def get_burn_rate_alerts(self) -> list[dict]:
        """Return SLOs whose burn rate exceeds sustainable levels.

        Returns:
            List of dicts containing ``slo_id``, ``name``, ``burn_rate``,
            ``severity``, and ``remaining_budget`` for each SLO where the
            burn rate exceeds 1.0.
        """
        alerts: list[dict] = []
        with self._lock:
            definitions = list(self._definitions.values())
            buckets = {defn.slo_id: self._aggregate(defn) for defn in definitions}

        for defn in definitions:
            budget = self._compute_budget(defn, buckets[defn.slo_id])
            if budget.burn_rate > 1.0:
                alerts.append({
                    "slo_id": defn.slo_id,
                    "name": defn.name,
                    "burn_rate": round(budget.burn_rate, 3),
                    "severity": self._calculator.classify_severity(budget.burn_rate),
                    "remaining_budget": round(budget.remaining, 6),
                    "tier": defn.tier,
                    "projected_exhaustion": (
                        budget.projected_exhaustion_time.isoformat()
                        if budget.projected_exhaustion_time
                        else None
                    ),
                })
        return alerts

    def should_allow_deployment(
        self,
        slo_ids: Optional[list[str]] = None,
    ) -> bool:
        """Determine whether a deployment should proceed.

        A deployment is allowed when all checked SLOs have remaining error
        budget (not exhausted) and their burn rate is below the critical
        threshold of 2.0.

        Args:
            slo_ids: Specific SLOs to check.  Defaults to all critical and
                high-tier SLOs.

        Returns:
            ``True`` if it is safe to deploy.
        """
        with self._lock:
            if slo_ids is None:
                definitions = [
                    d for d in self._definitions.values()
                    if d.tier in ("critical", "high")
                ]
            else:
                definitions = [
                    self._definitions[sid]
                    for sid in slo_ids
                    if sid in self._definitions
                ]
            buckets = {defn.slo_id: self._aggregate(defn) for defn in definitions}

        for defn in definitions:
            budget = self._compute_budget(defn, buckets[defn.slo_id])
            if budget.is_exhausted:
                logger.warning(
                    "Deployment blocked: SLO '%s' error budget exhausted",
                    defn.slo_id,
                )
                return False
            if budget.burn_rate >= 2.0:
                logger.warning(
                    "Deployment blocked: SLO '%s' burn rate %.2f (critical)",
                    defn.slo_id,
                    budget.burn_rate,
                )
                return False
        return True

    def get_report(self) -> dict:
        """Generate a comprehensive SLO dashboard report.

        Returns:
            Dictionary containing ``generated_at``, ``window_seconds``,
            ``slos`` (list of status + budget per SLO), ``alerts``,
            ``deployment_allowed``, and ``summary`` counts.
        """
        with self._lock:
            definitions = list(self._definitions.values())
            buckets = {defn.slo_id: self._aggregate(defn) for defn in definitions}

        slo_details: list[dict] = []
        meeting_count = 0

        for defn in definitions:
            bucket = buckets[defn.slo_id]
            metric = self._compute_metric(defn, bucket)
            budget = self._compute_budget(defn, bucket)

            if metric.is_meeting_target:
                meeting_count += 1

            slo_details.append({
                "slo_id": defn.slo_id,
                "name": defn.name,
                "type": defn.slo_type.value,
                "tier": defn.tier,
                "target": defn.target,
                "endpoint": defn.endpoint,
                "metric": {
                    "total_requests": metric.total_requests,
                    "good_requests": metric.good_requests,
                    "current_value": round(metric.current_value, 6),
                    "is_meeting_target": metric.is_meeting_target,
                },
                "budget": {
                    "total_budget": round(budget.total_budget, 6),
                    "consumed": round(budget.consumed, 6),
                    "remaining": round(budget.remaining, 6),
                    "burn_rate": round(budget.burn_rate, 3),
                    "is_exhausted": budget.is_exhausted,
                    "projected_exhaustion": (
                        budget.projected_exhaustion_time.isoformat()
                        if budget.projected_exhaustion_time
                        else None
                    ),
                },
            })

        alerts = self.get_burn_rate_alerts()
        total = len(definitions)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_seconds": 86_400,
            "slos": slo_details,
            "alerts": alerts,
            "deployment_allowed": self.should_allow_deployment(),
            "summary": {
                "total_slos": total,
                "meeting_target": meeting_count,
                "violating": total - meeting_count,
                "alert_count": len(alerts),
            },
        }

    # -- Internal helpers ---------------------------------------------------

    def _aggregate(self, defn: SLODefinition) -> _EndpointBucket:
        """Aggregate request records for a given SLO definition.

        Must be called while ``self._lock`` is held.

        Returns:
            An :class:`_EndpointBucket` containing the aggregated counts
            and latency values within the SLO window.
        """
        now = time.monotonic()
        cutoff = now - defn.window_seconds
        bucket = _EndpointBucket()

        for record in self._records:
            # Prune records older than the window
            if record.timestamp < cutoff:
                continue
            # Filter by endpoint if the SLO is endpoint-specific
            if defn.endpoint is not None and record.endpoint != defn.endpoint:
                continue

            bucket.total += 1
            if record.is_success:
                bucket.successes += 1
            else:
                bucket.failures += 1
            if record.is_quality_pass:
                bucket.quality_passes += 1
            bucket.latencies.append(record.latency_ms)

        return bucket

    def _compute_metric(
        self,
        defn: SLODefinition,
        bucket: _EndpointBucket,
    ) -> SLOMetric:
        """Compute an SLOMetric from a definition and aggregated bucket."""
        now_utc = datetime.now(timezone.utc)

        if bucket.total == 0:
            # No data: treat SLO as met (no violations observed).
            return SLOMetric(
                slo_id=defn.slo_id,
                timestamp=now_utc,
                total_requests=0,
                good_requests=0,
                current_value=1.0 if defn.slo_type != SLOType.ERROR_RATE else 0.0,
                is_meeting_target=True,
            )

        current_value: float
        good: int
        is_meeting: bool

        if defn.slo_type == SLOType.AVAILABILITY:
            good = bucket.successes
            current_value = good / bucket.total
            is_meeting = current_value >= defn.target

        elif defn.slo_type == SLOType.LATENCY:
            # P95 latency
            sorted_latencies = sorted(bucket.latencies)
            p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
            p95 = sorted_latencies[p95_index]
            current_value = p95
            # "good" requests are those under the target latency
            good = sum(1 for lat in bucket.latencies if lat <= defn.target)
            is_meeting = p95 <= defn.target

        elif defn.slo_type == SLOType.ERROR_RATE:
            error_rate = bucket.failures / bucket.total
            current_value = error_rate
            good = bucket.successes  # non-error requests
            is_meeting = error_rate <= defn.target

        elif defn.slo_type == SLOType.THROUGHPUT:
            # Throughput measured as requests per second over the window.
            elapsed = max(1.0, time.monotonic() - self._start_time)
            window_elapsed = min(elapsed, defn.window_seconds)
            current_value = bucket.total / window_elapsed
            good = bucket.total
            is_meeting = current_value >= defn.target

        elif defn.slo_type == SLOType.QUALITY:
            good = bucket.quality_passes
            current_value = good / bucket.total
            is_meeting = current_value >= defn.target

        else:
            # Unreachable under normal operation, but defensive.
            good = bucket.total
            current_value = 1.0
            is_meeting = True

        return SLOMetric(
            slo_id=defn.slo_id,
            timestamp=now_utc,
            total_requests=bucket.total,
            good_requests=good,
            current_value=current_value,
            is_meeting_target=is_meeting,
        )

    def _compute_budget(
        self,
        defn: SLODefinition,
        bucket: _EndpointBucket,
    ) -> ErrorBudget:
        """Compute an ErrorBudget from a definition and aggregated bucket."""
        now_mono = time.monotonic()
        now_utc = datetime.now(timezone.utc)

        elapsed_seconds = min(
            now_mono - self._start_time,
            defn.window_seconds,
        )
        elapsed_fraction = elapsed_seconds / defn.window_seconds if defn.window_seconds > 0 else 1.0
        window_remaining_seconds = max(0.0, defn.window_seconds - elapsed_seconds)

        window_start = now_utc - timedelta(seconds=elapsed_seconds)
        window_end = window_start + timedelta(seconds=defn.window_seconds)

        # Total budget: the allowable fraction of "bad" events.
        if defn.slo_type == SLOType.AVAILABILITY:
            total_budget = 1.0 - defn.target  # e.g. 0.001 for 99.9%
        elif defn.slo_type == SLOType.LATENCY:
            # For P95 latency SLOs, the budget is the allowed fraction of
            # requests that can exceed the target (5%).
            total_budget = 0.05
        elif defn.slo_type == SLOType.ERROR_RATE:
            total_budget = defn.target  # e.g. 0.01 for 1%
        elif defn.slo_type == SLOType.QUALITY:
            total_budget = 1.0 - defn.target  # e.g. 0.05 for 95%
        elif defn.slo_type == SLOType.THROUGHPUT:
            # Throughput SLOs don't map cleanly to error budgets.
            # Use a nominal budget based on target miss fraction.
            total_budget = 0.05
        else:
            total_budget = 0.01

        # Consumed fraction of the budget.
        if bucket.total == 0:
            consumed = 0.0
        else:
            if defn.slo_type == SLOType.AVAILABILITY:
                bad_fraction = bucket.failures / bucket.total
                consumed = bad_fraction / total_budget if total_budget > 0 else 0.0
            elif defn.slo_type == SLOType.LATENCY:
                bad_count = sum(1 for lat in bucket.latencies if lat > defn.target)
                bad_fraction = bad_count / bucket.total
                consumed = bad_fraction / total_budget if total_budget > 0 else 0.0
            elif defn.slo_type == SLOType.ERROR_RATE:
                error_fraction = bucket.failures / bucket.total
                consumed = error_fraction / total_budget if total_budget > 0 else 0.0
            elif defn.slo_type == SLOType.QUALITY:
                bad_quality = (bucket.total - bucket.quality_passes) / bucket.total
                consumed = bad_quality / total_budget if total_budget > 0 else 0.0
            elif defn.slo_type == SLOType.THROUGHPUT:
                elapsed = max(1.0, time.monotonic() - self._start_time)
                window_elapsed = min(elapsed, defn.window_seconds)
                actual_throughput = bucket.total / window_elapsed
                if actual_throughput >= defn.target:
                    consumed = 0.0
                else:
                    shortfall = (defn.target - actual_throughput) / defn.target
                    consumed = shortfall / total_budget if total_budget > 0 else 0.0
            else:
                consumed = 0.0

        remaining = max(0.0, 1.0 - consumed)
        burn_rate = self._calculator.calculate_burn_rate(consumed, elapsed_fraction)
        is_exhausted = remaining <= 0.0

        projected = self._calculator.project_exhaustion(
            remaining=remaining,
            burn_rate=burn_rate,
            window_remaining_seconds=window_remaining_seconds,
        )

        return ErrorBudget(
            slo_id=defn.slo_id,
            window_start=window_start,
            window_end=window_end,
            total_budget=total_budget,
            consumed=consumed,
            remaining=remaining,
            burn_rate=burn_rate,
            is_exhausted=is_exhausted,
            projected_exhaustion_time=projected,
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_slo_monitor_instance: Optional[SLOMonitor] = None
_slo_monitor_lock = threading.Lock()


def get_slo_monitor() -> SLOMonitor:
    """Return the global :class:`SLOMonitor` singleton.

    Thread-safe: uses double-checked locking to ensure only one instance
    is created.

    Returns:
        The singleton SLOMonitor instance.
    """
    global _slo_monitor_instance
    if _slo_monitor_instance is None:
        with _slo_monitor_lock:
            if _slo_monitor_instance is None:
                _slo_monitor_instance = SLOMonitor()
    return _slo_monitor_instance
