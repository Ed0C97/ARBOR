"""Self-Healing System with Chaos Engineering Automation for A.R.B.O.R. Enterprise.

TIER 3 - Resilience: Self-Healing Orchestration

Provides autonomous detection, diagnosis, and remediation of system failures.
When external services degrade (database latency, LLM timeouts, vector DB issues),
the self-healing orchestrator correlates symptoms to known failure signatures and
executes pre-defined remediation playbooks without human intervention.

Components:
- AnomalyCorrelator: Matches observed symptoms to known failure signatures via
  fuzzy matching (70%+ symptom overlap). Learns new signatures from resolved incidents.
- RemediationEngine: Maps failure signatures to ordered remediation playbooks and
  executes actions (circuit open, cache clear, reroute, scale up, etc.).
- SelfHealingOrchestrator: Top-level controller that detects anomalies, triggers
  remediation, records incidents, and runs chaos experiments to validate healing.
- VaccinationEngine: Generates reproducible chaos test specs from resolved incidents,
  building a regression suite that proves the system can heal from every past failure.

Usage:
    orchestrator = get_self_healing_orchestrator()
    incident = await orchestrator.detect_and_heal(["high_latency", "query_timeout"])
    health = orchestrator.get_system_health()
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class HealthStatus(Enum):
    """Overall system health classification."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FailureSignature:
    """A known failure pattern identified by its characteristic symptoms.

    When the system encounters a set of symptoms that match a registered
    signature (70%+ overlap), the corresponding remediation playbook
    is triggered automatically.

    Attributes:
        signature_id: Unique identifier for this failure pattern.
        failure_type: Human-readable name (e.g. "database_slow").
        symptoms: Canonical list of symptoms that define this failure.
        first_seen: When this pattern was first observed.
        last_seen: When this pattern was most recently observed.
        occurrence_count: Total number of times this pattern has been matched.
        remediation_playbook: Name of the playbook to execute, or None if manual.
        auto_remediated_count: How many times auto-remediation succeeded.
    """

    signature_id: str
    failure_type: str
    symptoms: list[str]
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    occurrence_count: int = 0
    remediation_playbook: str | None = None
    auto_remediated_count: int = 0


@dataclass
class RemediationAction:
    """A single remediation step to be executed as part of a playbook.

    Attributes:
        action_id: Unique identifier for this action instance.
        action_type: One of the supported action types.
        target: The subsystem or resource this action targets.
        parameters: Action-specific configuration.
        executed_at: When this action was executed, or None if pending.
        success: Whether the action succeeded, or None if not yet executed.
        duration_ms: Execution time in milliseconds, or None if not yet executed.
    """

    action_id: str
    action_type: str  # "restart", "rollback", "scale_up", "reroute", "circuit_open", "cache_clear", "retry_backoff"
    target: str
    parameters: dict = field(default_factory=dict)
    executed_at: datetime | None = None
    success: bool | None = None
    duration_ms: float | None = None


@dataclass
class IncidentRecord:
    """Full record of a detected incident from detection through resolution.

    Captures the entire lifecycle: symptom detection, signature correlation,
    remediation attempts, and final resolution or escalation.

    Attributes:
        incident_id: Unique identifier for this incident.
        severity: Impact classification.
        failure_signature: Matched known failure, or None if unknown.
        symptoms_observed: Raw symptoms that triggered detection.
        root_cause: Determined root cause, or None if still investigating.
        remediations_attempted: Ordered list of remediation actions tried.
        status: Current lifecycle stage of this incident.
        started_at: When the incident was first detected.
        resolved_at: When the incident was resolved, or None if still active.
    """

    incident_id: str
    severity: str  # "low", "medium", "high", "critical"
    failure_signature: FailureSignature | None = None
    symptoms_observed: list[str] = field(default_factory=list)
    root_cause: str | None = None
    remediations_attempted: list[RemediationAction] = field(default_factory=list)
    status: str = "detecting"  # "detecting", "diagnosing", "remediating", "resolved", "escalated"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly Correlator
# ═══════════════════════════════════════════════════════════════════════════════


class AnomalyCorrelator:
    """Correlates observed symptoms against a library of known failure signatures.

    Uses fuzzy matching: if 70%+ of a signature's symptoms are present in the
    observed symptom set, the signature is considered a match. When multiple
    signatures match, the one with the highest overlap ratio is returned.

    The correlator also learns from resolved incidents by extracting new failure
    signatures from symptom/root-cause pairs it has not seen before.
    """

    MATCH_THRESHOLD = 0.7  # 70% symptom overlap required for a match

    def __init__(self) -> None:
        self._signatures: dict[str, FailureSignature] = {}
        self._register_default_signatures()

    def _register_default_signatures(self) -> None:
        """Pre-register the well-known ARBOR failure signatures."""
        defaults = [
            FailureSignature(
                signature_id="sig_database_slow",
                failure_type="database_slow",
                symptoms=["high_latency", "connection_pool_exhausted", "query_timeout"],
                remediation_playbook="database_slow",
            ),
            FailureSignature(
                signature_id="sig_llm_unavailable",
                failure_type="llm_unavailable",
                symptoms=["llm_timeout", "llm_error_rate_high", "circuit_open_llm"],
                remediation_playbook="llm_unavailable",
            ),
            FailureSignature(
                signature_id="sig_vector_db_degraded",
                failure_type="vector_db_degraded",
                symptoms=["qdrant_slow", "search_latency_high", "vector_timeout"],
                remediation_playbook="vector_db_degraded",
            ),
            FailureSignature(
                signature_id="sig_memory_pressure",
                failure_type="memory_pressure",
                symptoms=["high_memory", "gc_pause", "oom_risk"],
                remediation_playbook="memory_pressure",
            ),
            FailureSignature(
                signature_id="sig_cascade_failure",
                failure_type="cascade_failure",
                symptoms=["multiple_circuits_open", "error_rate_spike", "latency_spike"],
                remediation_playbook="cascade_failure",
            ),
        ]
        for sig in defaults:
            self._signatures[sig.signature_id] = sig

    def correlate(self, symptoms: list[str]) -> FailureSignature | None:
        """Match observed symptoms against known failure signatures.

        Args:
            symptoms: List of currently observed symptom identifiers.

        Returns:
            The best-matching FailureSignature if the overlap is >= 70%,
            or None if no signature matches.
        """
        if not symptoms:
            return None

        observed = set(symptoms)
        best_match: FailureSignature | None = None
        best_ratio: float = 0.0

        for signature in self._signatures.values():
            expected = set(signature.symptoms)
            if not expected:
                continue

            overlap = len(observed & expected)
            ratio = overlap / len(expected)

            if ratio >= self.MATCH_THRESHOLD and ratio > best_ratio:
                best_ratio = ratio
                best_match = signature

        if best_match is not None:
            now = datetime.now(UTC)
            best_match.last_seen = now
            best_match.occurrence_count += 1
            logger.info(
                f"Self-healing correlator: Matched symptoms {symptoms} "
                f"to signature '{best_match.failure_type}' "
                f"(overlap={best_ratio:.0%}, occurrences={best_match.occurrence_count})"
            )

        return best_match

    def register_signature(self, signature: FailureSignature) -> None:
        """Register a new failure signature in the library.

        Args:
            signature: The failure signature to add.
        """
        self._signatures[signature.signature_id] = signature
        logger.info(
            f"Self-healing correlator: Registered new signature "
            f"'{signature.failure_type}' with {len(signature.symptoms)} symptoms"
        )

    def learn_from_incident(self, incident: IncidentRecord) -> None:
        """Extract and register a new signature from a resolved incident.

        If the incident has a root cause and symptoms that don't match any
        existing signature, a new signature is created and registered so
        future occurrences are handled automatically.

        Args:
            incident: A resolved incident with root_cause and symptoms.
        """
        if incident.status != "resolved" or not incident.root_cause:
            return
        if not incident.symptoms_observed:
            return

        # Check if these symptoms already match a known signature
        existing = self.correlate(incident.symptoms_observed)
        if existing is not None:
            logger.debug(
                f"Self-healing correlator: Incident {incident.incident_id} "
                f"already matches signature '{existing.failure_type}', skipping learning"
            )
            return

        # Create a new signature from this incident
        new_sig = FailureSignature(
            signature_id=f"sig_learned_{uuid.uuid4().hex[:8]}",
            failure_type=incident.root_cause,
            symptoms=list(incident.symptoms_observed),
            first_seen=incident.started_at,
            last_seen=datetime.now(UTC),
            occurrence_count=1,
            remediation_playbook=None,  # Manual remediation until a playbook is defined
        )
        self.register_signature(new_sig)
        logger.info(
            f"Self-healing correlator: Learned new signature '{new_sig.failure_type}' "
            f"from incident {incident.incident_id} "
            f"with symptoms {new_sig.symptoms}"
        )

    def get_all_signatures(self) -> list[FailureSignature]:
        """Return all registered failure signatures.

        Returns:
            List of all known failure signatures.
        """
        return list(self._signatures.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Remediation Engine
# ═══════════════════════════════════════════════════════════════════════════════


class RemediationEngine:
    """Maps failure signatures to remediation playbooks and executes them.

    Each playbook is an ordered sequence of RemediationActions. The engine
    executes them sequentially, recording timing and success/failure for
    each step. In this implementation, execution is logging-based (simulated)
    to allow safe testing and integration without side effects.
    """

    def __init__(self) -> None:
        self._playbooks: dict[str, list[dict[str, Any]]] = {}
        self._register_default_playbooks()

    def _register_default_playbooks(self) -> None:
        """Pre-register remediation playbooks for known failure types."""
        self._playbooks["database_slow"] = [
            {"action_type": "circuit_open", "target": "database", "parameters": {"timeout": 30}},
            {
                "action_type": "cache_clear",
                "target": "database_query_cache",
                "parameters": {"scope": "stale"},
            },
            {
                "action_type": "retry_backoff",
                "target": "database",
                "parameters": {"initial_wait": 1.0, "max_wait": 10.0, "max_attempts": 3},
            },
        ]

        self._playbooks["llm_unavailable"] = [
            {"action_type": "circuit_open", "target": "llm", "parameters": {"timeout": 60}},
            {
                "action_type": "reroute",
                "target": "llm_fallback",
                "parameters": {"fallback_provider": "cache_or_degraded"},
            },
            {
                "action_type": "cache_clear",
                "target": "llm_response_cache",
                "parameters": {"scope": "expired"},
            },
        ]

        self._playbooks["vector_db_degraded"] = [
            {"action_type": "circuit_open", "target": "qdrant", "parameters": {"timeout": 30}},
            {
                "action_type": "cache_clear",
                "target": "vector_search_cache",
                "parameters": {"scope": "all"},
            },
            {
                "action_type": "retry_backoff",
                "target": "qdrant",
                "parameters": {"initial_wait": 0.5, "max_wait": 5.0, "max_attempts": 3},
            },
        ]

        self._playbooks["memory_pressure"] = [
            {
                "action_type": "cache_clear",
                "target": "all_caches",
                "parameters": {"scope": "lru_evict", "evict_percent": 50},
            },
            {
                "action_type": "circuit_open",
                "target": "non_critical_services",
                "parameters": {"timeout": 120},
            },
            {
                "action_type": "scale_up",
                "target": "application",
                "parameters": {"strategy": "vertical", "increment": "256Mi"},
            },
        ]

        self._playbooks["cascade_failure"] = [
            {
                "action_type": "circuit_open",
                "target": "all_external_services",
                "parameters": {"timeout": 120},
            },
            {
                "action_type": "scale_up",
                "target": "application",
                "parameters": {"strategy": "horizontal", "replicas_add": 2},
            },
            {
                "action_type": "reroute",
                "target": "gradual_restore",
                "parameters": {
                    "restore_order": ["redis", "database", "qdrant", "llm"],
                    "step_delay_seconds": 15,
                },
            },
        ]

    def get_playbook(self, signature: FailureSignature) -> list[RemediationAction]:
        """Get the ordered remediation steps for a failure signature.

        Args:
            signature: The matched failure signature.

        Returns:
            List of RemediationActions to execute in order, or an empty list
            if no playbook is registered for this signature type.
        """
        playbook_name = signature.remediation_playbook
        if not playbook_name or playbook_name not in self._playbooks:
            logger.warning(
                f"Self-healing remediation: No playbook found for "
                f"signature '{signature.failure_type}' "
                f"(playbook_name={playbook_name})"
            )
            return []

        steps = self._playbooks[playbook_name]
        actions: list[RemediationAction] = []

        for step in steps:
            action = RemediationAction(
                action_id=f"act_{uuid.uuid4().hex[:8]}",
                action_type=step["action_type"],
                target=step["target"],
                parameters=dict(step.get("parameters", {})),
            )
            actions.append(action)

        logger.info(
            f"Self-healing remediation: Generated {len(actions)}-step playbook "
            f"for '{signature.failure_type}'"
        )
        return actions

    async def execute(self, action: RemediationAction) -> RemediationAction:
        """Execute a single remediation action (logging-based simulation).

        In a production deployment, each action_type would map to real
        infrastructure operations (restarting pods, adjusting circuit breakers,
        scaling container groups, etc.). This implementation logs the action
        and simulates a short execution delay.

        Args:
            action: The remediation action to execute.

        Returns:
            The same action with executed_at, success, and duration_ms populated.
        """
        start = time.monotonic()
        action.executed_at = datetime.now(UTC)

        logger.info(
            f"Self-healing remediation: Executing {action.action_type} "
            f"on target '{action.target}' "
            f"(params={action.parameters})"
        )

        try:
            # Simulate execution delay proportional to action complexity
            delay = self._get_simulated_delay(action.action_type)
            await asyncio.sleep(delay)

            # Log the specific remediation performed
            self._log_action_detail(action)

            action.success = True
            logger.info(
                f"Self-healing remediation: {action.action_type} on " f"'{action.target}' succeeded"
            )
        except Exception as e:
            action.success = False
            logger.error(
                f"Self-healing remediation: {action.action_type} on "
                f"'{action.target}' failed: {e}"
            )

        elapsed = time.monotonic() - start
        action.duration_ms = round(elapsed * 1000, 2)
        return action

    @staticmethod
    def _get_simulated_delay(action_type: str) -> float:
        """Return a simulated execution delay in seconds for each action type."""
        delays = {
            "restart": 0.05,
            "rollback": 0.04,
            "scale_up": 0.06,
            "reroute": 0.02,
            "circuit_open": 0.01,
            "cache_clear": 0.02,
            "retry_backoff": 0.01,
        }
        return delays.get(action_type, 0.03)

    @staticmethod
    def _log_action_detail(action: RemediationAction) -> None:
        """Log human-readable detail for each action type."""
        detail_map = {
            "restart": f"Restarted service '{action.target}'",
            "rollback": f"Rolled back '{action.target}' to previous stable version",
            "scale_up": f"Scaled up '{action.target}' with strategy={action.parameters.get('strategy', 'unknown')}",
            "reroute": f"Rerouted traffic for '{action.target}' (fallback={action.parameters.get('fallback_provider', 'n/a')})",
            "circuit_open": f"Opened circuit breaker for '{action.target}' (timeout={action.parameters.get('timeout', 'default')}s)",
            "cache_clear": f"Cleared cache '{action.target}' (scope={action.parameters.get('scope', 'all')})",
            "retry_backoff": f"Configured retry backoff for '{action.target}' (max_attempts={action.parameters.get('max_attempts', 3)})",
        }
        detail = detail_map.get(
            action.action_type, f"Executed {action.action_type} on '{action.target}'"
        )
        logger.debug(f"Self-healing remediation detail: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Healing Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class SelfHealingOrchestrator:
    """Top-level controller that detects, diagnoses, and remediates system failures.

    Ties together the AnomalyCorrelator and RemediationEngine into an automated
    healing loop. When symptoms are reported (via detect_and_heal or
    report_symptom), the orchestrator:

    1. Correlates symptoms to known failure signatures (fuzzy match).
    2. If a known failure is matched, executes the remediation playbook.
    3. If the failure is unknown, escalates the incident for human review.
    4. Records the full incident timeline for post-mortem analysis.

    Also supports chaos engineering via run_chaos_experiment(), which injects
    controlled failures and verifies that self-healing responds correctly.
    """

    def __init__(self) -> None:
        self.correlator = AnomalyCorrelator()
        self.remediation_engine = RemediationEngine()
        self._incident_log: list[IncidentRecord] = []
        self._active_incidents: dict[str, IncidentRecord] = {}
        self._symptom_buffer: list[str] = []
        self._symptom_counts: dict[str, int] = defaultdict(int)

    # ───────────────────────────────────────────────────────────────────────
    # Core Healing Loop
    # ───────────────────────────────────────────────────────────────────────

    async def detect_and_heal(self, symptoms: list[str]) -> IncidentRecord:
        """Run the full detection-diagnosis-remediation loop for a set of symptoms.

        Args:
            symptoms: List of observed symptom identifiers.

        Returns:
            The IncidentRecord documenting the full incident lifecycle.
        """
        incident = IncidentRecord(
            incident_id=f"inc_{uuid.uuid4().hex[:12]}",
            severity=self._assess_severity(symptoms),
            symptoms_observed=list(symptoms),
            status="detecting",
        )
        self._active_incidents[incident.incident_id] = incident

        logger.info(
            f"Self-healing: Incident {incident.incident_id} opened "
            f"(severity={incident.severity}, symptoms={symptoms})"
        )

        # Phase 1: Diagnose - correlate symptoms to known failure
        incident.status = "diagnosing"
        signature = self.correlator.correlate(symptoms)
        incident.failure_signature = signature

        if signature is not None:
            incident.root_cause = signature.failure_type
            logger.info(
                f"Self-healing: Incident {incident.incident_id} diagnosed as "
                f"'{signature.failure_type}'"
            )

            # Phase 2: Remediate - execute the playbook
            incident.status = "remediating"
            playbook = self.remediation_engine.get_playbook(signature)

            if playbook:
                all_succeeded = True
                for action in playbook:
                    executed = await self.remediation_engine.execute(action)
                    incident.remediations_attempted.append(executed)
                    if not executed.success:
                        all_succeeded = False
                        logger.warning(
                            f"Self-healing: Action {executed.action_type} failed "
                            f"for incident {incident.incident_id}"
                        )

                if all_succeeded:
                    incident.status = "resolved"
                    incident.resolved_at = datetime.now(UTC)
                    signature.auto_remediated_count += 1
                    logger.info(
                        f"Self-healing: Incident {incident.incident_id} RESOLVED "
                        f"via automated playbook '{signature.remediation_playbook}' "
                        f"({len(playbook)} actions executed)"
                    )
                else:
                    incident.status = "escalated"
                    logger.warning(
                        f"Self-healing: Incident {incident.incident_id} ESCALATED "
                        f"- some remediation actions failed"
                    )
            else:
                # Signature found but no playbook available
                incident.status = "escalated"
                logger.warning(
                    f"Self-healing: Incident {incident.incident_id} ESCALATED "
                    f"- no playbook for signature '{signature.failure_type}'"
                )
        else:
            # Unknown failure - escalate for human review
            incident.status = "escalated"
            incident.root_cause = None
            logger.warning(
                f"Self-healing: Incident {incident.incident_id} ESCALATED "
                f"- unknown failure pattern, symptoms={symptoms}"
            )

        # Record the incident
        self._incident_log.append(incident)
        self._active_incidents.pop(incident.incident_id, None)

        # Attempt to learn from resolved incidents
        if incident.status == "resolved":
            self.correlator.learn_from_incident(incident)

        return incident

    # ───────────────────────────────────────────────────────────────────────
    # Symptom Accumulation
    # ───────────────────────────────────────────────────────────────────────

    def report_symptom(self, symptom: str) -> None:
        """Accumulate a single symptom into the buffer.

        Symptoms are buffered until detect_and_heal is called with the
        accumulated set, or until an external trigger processes them.

        Args:
            symptom: A symptom identifier string.
        """
        self._symptom_buffer.append(symptom)
        self._symptom_counts[symptom] += 1
        logger.debug(
            f"Self-healing: Symptom reported: '{symptom}' "
            f"(buffer_size={len(self._symptom_buffer)}, "
            f"total_count={self._symptom_counts[symptom]})"
        )

    def drain_symptom_buffer(self) -> list[str]:
        """Drain and return all buffered symptoms, clearing the buffer.

        Returns:
            List of unique symptoms from the buffer.
        """
        unique_symptoms = list(dict.fromkeys(self._symptom_buffer))
        self._symptom_buffer.clear()
        return unique_symptoms

    # ───────────────────────────────────────────────────────────────────────
    # System Health Assessment
    # ───────────────────────────────────────────────────────────────────────

    def get_system_health(self) -> HealthStatus:
        """Assess overall system health based on active and recent incidents.

        Health levels:
        - HEALTHY: No active incidents, no recent critical/high incidents.
        - DEGRADED: Active low/medium incidents or recent escalations.
        - UNHEALTHY: Active high-severity incidents.
        - CRITICAL: Active critical-severity incidents or multiple escalated.

        Returns:
            The current HealthStatus.
        """
        active = list(self._active_incidents.values())

        # Check active incidents first
        active_severities = [inc.severity for inc in active]
        if "critical" in active_severities:
            return HealthStatus.CRITICAL
        if "high" in active_severities:
            return HealthStatus.UNHEALTHY

        # Check recent incidents (last 10 in the log)
        recent = self._incident_log[-10:] if self._incident_log else []
        escalated_count = sum(1 for inc in recent if inc.status == "escalated")
        recent_critical = sum(1 for inc in recent if inc.severity == "critical")

        if escalated_count >= 3 or recent_critical >= 2:
            return HealthStatus.CRITICAL
        if escalated_count >= 1 or active:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    # ───────────────────────────────────────────────────────────────────────
    # History and Introspection
    # ───────────────────────────────────────────────────────────────────────

    def get_incident_history(self, limit: int = 50) -> list[IncidentRecord]:
        """Return the most recent incidents from the log.

        Args:
            limit: Maximum number of incidents to return.

        Returns:
            List of IncidentRecords, most recent first.
        """
        return list(reversed(self._incident_log[-limit:]))

    def get_failure_library(self) -> list[FailureSignature]:
        """Return all known failure signatures.

        Returns:
            List of all registered FailureSignatures.
        """
        return self.correlator.get_all_signatures()

    # ───────────────────────────────────────────────────────────────────────
    # Chaos Engineering
    # ───────────────────────────────────────────────────────────────────────

    async def run_chaos_experiment(
        self,
        experiment_type: str,
        duration_seconds: float = 30,
    ) -> dict[str, Any]:
        """Inject a controlled failure and verify that self-healing responds correctly.

        Supported experiment types:
        - "database_latency": Simulates slow database responses.
        - "llm_timeout": Simulates LLM provider being unavailable.
        - "cache_failure": Simulates cache layer failure.

        Args:
            experiment_type: The type of chaos to inject.
            duration_seconds: How long the simulated failure lasts (used for logging).

        Returns:
            Dict with keys: experiment_type, success (bool), healing_time_ms (float),
            symptoms_detected (list[str]), incident_id (str), duration_seconds (float).
        """
        experiment_symptoms = {
            "database_latency": ["high_latency", "connection_pool_exhausted", "query_timeout"],
            "llm_timeout": ["llm_timeout", "llm_error_rate_high", "circuit_open_llm"],
            "cache_failure": ["cache_miss_rate_high", "redis_timeout", "cache_unavailable"],
        }

        symptoms = experiment_symptoms.get(experiment_type)
        if symptoms is None:
            logger.error(
                f"Self-healing chaos: Unknown experiment type '{experiment_type}'. "
                f"Supported: {list(experiment_symptoms.keys())}"
            )
            return {
                "experiment_type": experiment_type,
                "success": False,
                "healing_time_ms": 0.0,
                "symptoms_detected": [],
                "incident_id": None,
                "duration_seconds": duration_seconds,
                "error": f"Unknown experiment type: {experiment_type}",
            }

        logger.info(
            f"Self-healing chaos: Starting experiment '{experiment_type}' "
            f"(duration={duration_seconds}s, injected_symptoms={symptoms})"
        )

        start = time.monotonic()
        incident = await self.detect_and_heal(symptoms)
        healing_time_ms = round((time.monotonic() - start) * 1000, 2)

        healed = incident.status == "resolved"

        result = {
            "experiment_type": experiment_type,
            "success": healed,
            "healing_time_ms": healing_time_ms,
            "symptoms_detected": symptoms,
            "incident_id": incident.incident_id,
            "duration_seconds": duration_seconds,
            "incident_status": incident.status,
            "actions_taken": len(incident.remediations_attempted),
        }

        if healed:
            logger.info(
                f"Self-healing chaos: Experiment '{experiment_type}' PASSED - "
                f"system self-healed in {healing_time_ms}ms "
                f"({len(incident.remediations_attempted)} actions)"
            )
        else:
            logger.warning(
                f"Self-healing chaos: Experiment '{experiment_type}' FAILED - "
                f"incident status={incident.status}, "
                f"healing_time_ms={healing_time_ms}"
            )

        return result

    # ───────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assess_severity(symptoms: list[str]) -> str:
        """Determine incident severity from the set of observed symptoms.

        Heuristic:
        - critical: cascade failure indicators or 5+ simultaneous symptoms.
        - high: circuit open or OOM indicators present.
        - medium: 2+ symptoms present.
        - low: single symptom or informational.

        Args:
            symptoms: List of symptom identifiers.

        Returns:
            Severity string: "low", "medium", "high", or "critical".
        """
        critical_indicators = {
            "multiple_circuits_open",
            "oom_risk",
            "cascade_failure",
            "error_rate_spike",
        }
        high_indicators = {"circuit_open_llm", "connection_pool_exhausted", "high_memory"}

        symptom_set = set(symptoms)

        if symptom_set & critical_indicators or len(symptoms) >= 5:
            return "critical"
        if symptom_set & high_indicators:
            return "high"
        if len(symptoms) >= 2:
            return "medium"
        return "low"


# ═══════════════════════════════════════════════════════════════════════════════
# Vaccination Engine
# ═══════════════════════════════════════════════════════════════════════════════


class VaccinationEngine:
    """Generates reproducible chaos test specifications from resolved incidents.

    After the system successfully self-heals from a failure, the VaccinationEngine
    creates a chaos test spec that can reproduce the same failure. This builds a
    regression suite proving the system can handle every failure it has ever seen.
    """

    def __init__(self) -> None:
        self._vaccination_suite: list[dict[str, Any]] = []

    def generate_chaos_test(self, incident: IncidentRecord) -> dict[str, Any]:
        """From a resolved incident, generate a reproducible chaos test specification.

        Args:
            incident: A resolved IncidentRecord with symptoms and remediations.

        Returns:
            A dict describing a chaos test: test_id, incident_id, symptoms_to_inject,
            expected_actions, expected_outcome, generated_at.
        """
        test_spec: dict[str, Any] = {
            "test_id": f"vac_{uuid.uuid4().hex[:8]}",
            "incident_id": incident.incident_id,
            "failure_type": (
                incident.failure_signature.failure_type if incident.failure_signature else "unknown"
            ),
            "symptoms_to_inject": list(incident.symptoms_observed),
            "expected_actions": [
                {
                    "action_type": action.action_type,
                    "target": action.target,
                    "expected_success": action.success,
                }
                for action in incident.remediations_attempted
            ],
            "expected_outcome": incident.status,
            "severity": incident.severity,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        self._vaccination_suite.append(test_spec)
        logger.info(
            f"Self-healing vaccination: Generated test '{test_spec['test_id']}' "
            f"from incident {incident.incident_id} "
            f"(failure_type='{test_spec['failure_type']}', "
            f"symptoms={len(test_spec['symptoms_to_inject'])}, "
            f"actions={len(test_spec['expected_actions'])})"
        )

        return test_spec

    def get_vaccination_suite(self) -> list[dict[str, Any]]:
        """Return all generated chaos test specifications.

        Returns:
            List of chaos test spec dicts.
        """
        return list(self._vaccination_suite)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Access
# ═══════════════════════════════════════════════════════════════════════════════

_orchestrator_instance: SelfHealingOrchestrator | None = None
_vaccination_engine_instance: VaccinationEngine | None = None


def get_self_healing_orchestrator() -> SelfHealingOrchestrator:
    """Get the singleton SelfHealingOrchestrator instance.

    Returns:
        The global SelfHealingOrchestrator.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SelfHealingOrchestrator()
        logger.info("Self-healing: Orchestrator initialized")
    return _orchestrator_instance


def get_vaccination_engine() -> VaccinationEngine:
    """Get the singleton VaccinationEngine instance.

    Returns:
        The global VaccinationEngine.
    """
    global _vaccination_engine_instance
    if _vaccination_engine_instance is None:
        _vaccination_engine_instance = VaccinationEngine()
        logger.info("Self-healing: VaccinationEngine initialized")
    return _vaccination_engine_instance
