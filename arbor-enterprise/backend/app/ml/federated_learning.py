"""Federated learning module for privacy-preserving model improvement.

Enables learning from user feedback across tenants without sharing raw data.
Each tenant trains locally and submits gradient updates; the coordinator
aggregates them into a shared global model using configurable strategies
(FedAvg, FedProx, Weighted Average).

Differential privacy utilities (Gaussian noise injection and gradient
clipping) are provided so that individual tenant contributions cannot be
reverse-engineered from the published global weights.

Usage::

    coordinator = get_federated_coordinator()
    coordinator.register_model(
        model_id="rec_v1",
        model_type="recommendation",
        initial_weights={"layer1": [0.0, 0.0, 0.0]},
    )
    coordinator.submit_update(
        model_id="rec_v1",
        tenant_id="tenant_A",
        gradient_updates={"layer1": [0.1, -0.2, 0.05]},
        sample_count=500,
    )
    updated = coordinator.run_aggregation_round("rec_v1")
"""

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelUpdate:
    """A single gradient update submitted by a tenant after local training.

    Attributes:
        tenant_id: Identifier of the tenant that produced this update.
        round_number: The federated round this update belongs to.
        gradient_updates: Mapping of parameter name to gradient vector.
        sample_count: Number of local samples used to compute the gradients.
        timestamp: UTC timestamp when the update was created.
        metadata: Arbitrary extra information (e.g. local loss, epochs).
    """

    tenant_id: str
    round_number: int
    gradient_updates: dict[str, list[float]]
    sample_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


@dataclass
class FederatedModel:
    """Global model state maintained by the coordinator.

    Attributes:
        model_id: Unique identifier for this federated model.
        model_type: Descriptive model category (e.g. ``recommendation``).
        global_weights: Current aggregated parameter vectors.
        version: Monotonically increasing version bumped after each round.
        last_aggregated_at: UTC timestamp of the most recent aggregation.
        contributing_tenants: List of tenant IDs that have contributed so far.
        performance_metrics: Latest evaluation metrics (e.g. accuracy, loss).
    """

    model_id: str
    model_type: str
    global_weights: dict[str, list[float]]
    version: int = 0
    last_aggregated_at: Optional[datetime] = None
    contributing_tenants: list[str] = field(default_factory=list)
    performance_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation strategy enum
# ---------------------------------------------------------------------------


class AggregationStrategy(Enum):
    """Strategy used to combine tenant gradient updates.

    Members:
        FED_AVG: Federated Averaging - simple averaging weighted by sample
            count.
        FED_PROX: Federated Proximal - adds a proximal term to penalise
            divergence from the current global weights.
        WEIGHTED_AVG: Weighted average proportional to each tenant's sample
            count (equivalent to FED_AVG when every tenant has the same
            count, but makes the weighting explicit).
    """

    FED_AVG = "fed_avg"
    FED_PROX = "fed_prox"
    WEIGHTED_AVG = "weighted_avg"


# ---------------------------------------------------------------------------
# Gradient aggregator
# ---------------------------------------------------------------------------


class GradientAggregator:
    """Stateless helper that combines gradient updates from multiple tenants.

    Each public method accepts a list of :class:`ModelUpdate` objects and
    returns a single dict mapping parameter names to aggregated gradient
    vectors.  Dimension mismatches between updates for the same parameter
    are handled gracefully by truncating to the shortest vector.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        updates: list[ModelUpdate],
        strategy: AggregationStrategy,
        global_weights: Optional[dict[str, list[float]]] = None,
    ) -> dict[str, list[float]]:
        """Aggregate *updates* using the specified *strategy*.

        Args:
            updates: Gradient updates from participating tenants.
            strategy: The aggregation algorithm to apply.
            global_weights: Current global weights (required for
                :attr:`AggregationStrategy.FED_PROX`).

        Returns:
            Aggregated gradient vectors keyed by parameter name.

        Raises:
            ValueError: If *updates* is empty or an unknown strategy is
                provided.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates")

        if strategy == AggregationStrategy.FED_AVG:
            return self._fed_avg(updates)
        elif strategy == AggregationStrategy.FED_PROX:
            return self._fed_prox(updates, global_weights or {})
        elif strategy == AggregationStrategy.WEIGHTED_AVG:
            return self._weighted_avg(updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fed_avg(self, updates: list[ModelUpdate]) -> dict[str, list[float]]:
        """Federated Averaging: weighted mean of gradients by sample count.

        Each gradient vector is scaled by ``sample_count / total_samples``
        before summing, so tenants with more data have proportionally more
        influence on the result.

        Args:
            updates: Non-empty list of tenant gradient updates.

        Returns:
            Aggregated gradient dict.
        """
        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            total_samples = len(updates)  # fall back to uniform weighting

        all_keys: set[str] = set()
        for u in updates:
            all_keys.update(u.gradient_updates.keys())

        aggregated: dict[str, list[float]] = {}
        for key in all_keys:
            # Determine the minimum dimension for this parameter across updates
            vectors = [
                u.gradient_updates[key]
                for u in updates
                if key in u.gradient_updates
            ]
            if not vectors:
                continue

            min_dim = min(len(v) for v in vectors)
            result = [0.0] * min_dim

            for update in updates:
                if key not in update.gradient_updates:
                    continue
                weight = update.sample_count / total_samples
                grad = update.gradient_updates[key]
                for i in range(min_dim):
                    result[i] += grad[i] * weight

            aggregated[key] = result

        return aggregated

    def _fed_prox(
        self,
        updates: list[ModelUpdate],
        global_weights: dict[str, list[float]],
        mu: float = 0.01,
    ) -> dict[str, list[float]]:
        """Federated Proximal: FedAvg plus a proximal penalty term.

        The proximal term ``mu / 2 * ||w - w_global||^2`` discourages
        individual updates from diverging too far from the current global
        weights, improving convergence in heterogeneous data settings.

        After computing the standard FedAvg result the method subtracts
        ``mu * (avg_gradient - global_weight)`` per dimension, which is the
        gradient of the proximal penalty with respect to the model weights.

        Args:
            updates: Non-empty list of tenant gradient updates.
            global_weights: Current global model weights.
            mu: Strength of the proximal term (higher = more conservative).

        Returns:
            Aggregated gradient dict with proximal correction applied.
        """
        avg_gradients = self._fed_avg(updates)

        for key, grad_vec in avg_gradients.items():
            if key in global_weights:
                gw = global_weights[key]
                min_dim = min(len(grad_vec), len(gw))
                for i in range(min_dim):
                    # Proximal correction: penalise deviation from global
                    grad_vec[i] -= mu * (grad_vec[i] - gw[i])

        return avg_gradients

    def _weighted_avg(self, updates: list[ModelUpdate]) -> dict[str, list[float]]:
        """Weighted average of gradients proportional to sample count.

        Functionally equivalent to :meth:`_fed_avg` but implemented
        independently for clarity and to allow future divergence (e.g.
        adding quality-based weighting).

        Args:
            updates: Non-empty list of tenant gradient updates.

        Returns:
            Aggregated gradient dict.
        """
        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            total_samples = len(updates)

        all_keys: set[str] = set()
        for u in updates:
            all_keys.update(u.gradient_updates.keys())

        aggregated: dict[str, list[float]] = {}
        for key in all_keys:
            vectors = [
                u.gradient_updates[key]
                for u in updates
                if key in u.gradient_updates
            ]
            if not vectors:
                continue

            min_dim = min(len(v) for v in vectors)
            result = [0.0] * min_dim

            for update in updates:
                if key not in update.gradient_updates:
                    continue
                weight = update.sample_count / total_samples
                grad = update.gradient_updates[key]
                for i in range(min_dim):
                    result[i] += grad[i] * weight

            aggregated[key] = result

        return aggregated


# ---------------------------------------------------------------------------
# Federated learning coordinator
# ---------------------------------------------------------------------------


class FederatedLearningCoordinator:
    """Central coordinator for federated model training.

    Manages model registration, collection of tenant gradient updates, and
    periodic aggregation rounds.  All state is held in-memory; in a
    production deployment this would be backed by a persistent store and a
    message queue for update submission.

    Usage::

        coordinator = get_federated_coordinator()
        coordinator.register_model("rec_v1", "recommendation", {"layer1": [0.0]})
        coordinator.submit_update("rec_v1", "t1", {"layer1": [0.1]}, 100)
        coordinator.submit_update("rec_v1", "t2", {"layer1": [0.2]}, 200)
        model = coordinator.run_aggregation_round("rec_v1")
    """

    def __init__(self) -> None:
        self._models: dict[str, FederatedModel] = {}
        self._pending_updates: dict[str, list[ModelUpdate]] = {}
        self._round_counter: dict[str, int] = {}
        self._aggregator = GradientAggregator()
        logger.info("FederatedLearningCoordinator initialised (in-memory store)")

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        model_type: str,
        initial_weights: dict[str, list[float]],
    ) -> FederatedModel:
        """Register a new federated model with initial global weights.

        Args:
            model_id: Unique identifier for the model.
            model_type: Descriptive category (e.g. ``recommendation``).
            initial_weights: Starting parameter vectors.

        Returns:
            The newly created :class:`FederatedModel`.

        Raises:
            ValueError: If a model with the same *model_id* already exists.
        """
        if model_id in self._models:
            raise ValueError(f"Model '{model_id}' is already registered")

        model = FederatedModel(
            model_id=model_id,
            model_type=model_type,
            global_weights=copy.deepcopy(initial_weights),
            version=0,
            last_aggregated_at=None,
            contributing_tenants=[],
            performance_metrics={},
        )
        self._models[model_id] = model
        self._pending_updates[model_id] = []
        self._round_counter[model_id] = 0

        logger.info(
            "Registered federated model: id=%s type=%s params=%d",
            model_id,
            model_type,
            len(initial_weights),
        )
        return model

    # ------------------------------------------------------------------
    # Update submission
    # ------------------------------------------------------------------

    def submit_update(
        self,
        model_id: str,
        tenant_id: str,
        gradient_updates: dict[str, list[float]],
        sample_count: int,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Submit a gradient update from a tenant for a registered model.

        Args:
            model_id: Target model identifier.
            tenant_id: Identifier of the submitting tenant.
            gradient_updates: Gradient vectors keyed by parameter name.
            sample_count: Number of local training samples used.
            metadata: Optional extra information to attach to the update.

        Returns:
            ``True`` if the update was accepted, ``False`` if the model does
            not exist or the update was invalid.
        """
        if model_id not in self._models:
            logger.warning(
                "Update rejected: model '%s' not found (tenant=%s)",
                model_id,
                tenant_id,
            )
            return False

        if sample_count <= 0:
            logger.warning(
                "Update rejected: sample_count must be positive (tenant=%s model=%s)",
                tenant_id,
                model_id,
            )
            return False

        current_round = self._round_counter.get(model_id, 0)
        update = ModelUpdate(
            tenant_id=tenant_id,
            round_number=current_round,
            gradient_updates=copy.deepcopy(gradient_updates),
            sample_count=sample_count,
            metadata=metadata or {},
        )
        self._pending_updates[model_id].append(update)

        logger.debug(
            "Update accepted: model=%s tenant=%s round=%d samples=%d pending=%d",
            model_id,
            tenant_id,
            current_round,
            sample_count,
            len(self._pending_updates[model_id]),
        )
        return True

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def run_aggregation_round(
        self,
        model_id: str,
        strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        min_updates: int = 2,
    ) -> Optional[FederatedModel]:
        """Run a single aggregation round for the specified model.

        Collects all pending updates, aggregates them using *strategy*, and
        applies the result to the global weights.  The model version is
        incremented and the pending update queue is cleared.

        Args:
            model_id: Target model identifier.
            strategy: Aggregation algorithm to use.
            min_updates: Minimum number of pending updates required to
                proceed.  If fewer updates are available the round is
                skipped and ``None`` is returned.

        Returns:
            The updated :class:`FederatedModel`, or ``None`` if the round
            was skipped (model not found or insufficient updates).
        """
        if model_id not in self._models:
            logger.warning("Aggregation skipped: model '%s' not found", model_id)
            return None

        pending = self._pending_updates.get(model_id, [])
        if len(pending) < min_updates:
            logger.info(
                "Aggregation skipped: model=%s pending=%d min_required=%d",
                model_id,
                len(pending),
                min_updates,
            )
            return None

        model = self._models[model_id]

        # Aggregate gradients
        aggregated = self._aggregator.aggregate(
            updates=pending,
            strategy=strategy,
            global_weights=model.global_weights,
        )

        # Apply aggregated gradients to global weights
        for key, grad_vec in aggregated.items():
            if key in model.global_weights:
                gw = model.global_weights[key]
                min_dim = min(len(gw), len(grad_vec))
                for i in range(min_dim):
                    gw[i] += grad_vec[i]
            else:
                model.global_weights[key] = list(grad_vec)

        # Track contributing tenants
        round_tenants = {u.tenant_id for u in pending}
        for tid in round_tenants:
            if tid not in model.contributing_tenants:
                model.contributing_tenants.append(tid)

        # Update bookkeeping
        model.version += 1
        model.last_aggregated_at = datetime.now(timezone.utc)

        self._round_counter[model_id] = self._round_counter.get(model_id, 0) + 1
        self._pending_updates[model_id] = []

        logger.info(
            "Aggregation complete: model=%s version=%d strategy=%s tenants=%d updates=%d",
            model_id,
            model.version,
            strategy.value,
            len(round_tenants),
            len(pending),
        )
        return model

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> Optional[FederatedModel]:
        """Return the federated model with the given *model_id*, or ``None``.

        Args:
            model_id: Model identifier to look up.

        Returns:
            The :class:`FederatedModel` if found, otherwise ``None``.
        """
        return self._models.get(model_id)

    def get_model_for_tenant(
        self, model_id: str, tenant_id: str
    ) -> dict[str, list[float]]:
        """Return global weights for a specific tenant.

        In a production system this would layer tenant-specific fine-tuning
        on top of the global weights.  The current implementation returns a
        deep copy of the global weights to prevent mutation.

        Args:
            model_id: Target model identifier.
            tenant_id: Requesting tenant (reserved for future per-tenant
                personalisation).

        Returns:
            A deep copy of the global weight dict.

        Raises:
            ValueError: If the model does not exist.
        """
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")

        logger.debug(
            "Serving global weights to tenant=%s model=%s version=%d",
            tenant_id,
            model_id,
            model.version,
        )
        return copy.deepcopy(model.global_weights)

    def list_models(self) -> list[dict]:
        """Return summary information for all registered models.

        Returns:
            A list of dicts, each containing ``model_id``, ``model_type``,
            ``version``, ``contributing_tenants`` count, and
            ``last_aggregated_at``.
        """
        summaries: list[dict] = []
        for model in self._models.values():
            summaries.append(
                {
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "version": model.version,
                    "contributing_tenants": len(model.contributing_tenants),
                    "last_aggregated_at": (
                        model.last_aggregated_at.isoformat()
                        if model.last_aggregated_at
                        else None
                    ),
                }
            )
        return summaries

    def get_round_status(self, model_id: str) -> dict:
        """Return the current aggregation round status for a model.

        Args:
            model_id: Target model identifier.

        Returns:
            A dict with ``model_id``, ``current_round``, ``pending_updates``,
            ``model_version``, and ``contributing_tenants``.  Returns a dict
            with ``error`` key if the model is not found.
        """
        if model_id not in self._models:
            return {"error": f"Model '{model_id}' not found"}

        model = self._models[model_id]
        pending = self._pending_updates.get(model_id, [])
        return {
            "model_id": model_id,
            "current_round": self._round_counter.get(model_id, 0),
            "pending_updates": len(pending),
            "model_version": model.version,
            "contributing_tenants": list(model.contributing_tenants),
        }


# ---------------------------------------------------------------------------
# Differential privacy utilities
# ---------------------------------------------------------------------------


class DifferentialPrivacy:
    """Utilities for applying differential privacy to gradient updates.

    Provides Gaussian noise injection calibrated to the desired
    (epsilon, delta)-differential privacy guarantee, and L2 gradient
    clipping to bound the sensitivity of individual contributions.

    Usage::

        dp = DifferentialPrivacy()
        clipped = dp.clip_gradients(raw_gradients, max_norm=1.0)
        private = dp.add_noise(clipped, epsilon=1.0, delta=1e-5)
    """

    def add_noise(
        self,
        gradients: dict[str, list[float]],
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> dict[str, list[float]]:
        """Add calibrated Gaussian noise to gradients for privacy.

        The noise standard deviation is set to ``sensitivity * sqrt(2 *
        ln(1.25 / delta)) / epsilon`` following the Gaussian mechanism of
        (epsilon, delta)-differential privacy.  Sensitivity is assumed to
        be 1.0 (callers should clip gradients beforehand).

        Args:
            gradients: Gradient vectors keyed by parameter name.
            epsilon: Privacy budget (lower = more private, more noisy).
            delta: Probability of privacy breach.

        Returns:
            A new dict with noisy gradient vectors.
        """
        sensitivity = 1.0
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

        noisy: dict[str, list[float]] = {}
        for key, values in gradients.items():
            noisy[key] = [v + random.gauss(0.0, sigma) for v in values]

        logger.debug(
            "Differential privacy noise applied: epsilon=%.2f delta=%.1e sigma=%.4f params=%d",
            epsilon,
            delta,
            sigma,
            len(gradients),
        )
        return noisy

    def clip_gradients(
        self,
        gradients: dict[str, list[float]],
        max_norm: float = 1.0,
    ) -> dict[str, list[float]]:
        """Clip gradient vectors so their L2 norm does not exceed *max_norm*.

        Each parameter vector is independently checked.  If its L2 norm
        exceeds *max_norm* the vector is scaled down proportionally.

        Args:
            gradients: Gradient vectors keyed by parameter name.
            max_norm: Maximum allowed L2 norm per parameter vector.

        Returns:
            A new dict with clipped gradient vectors.
        """
        clipped: dict[str, list[float]] = {}
        for key, values in gradients.items():
            l2 = math.sqrt(sum(v * v for v in values)) if values else 0.0
            if l2 > max_norm and l2 > 0.0:
                scale = max_norm / l2
                clipped[key] = [v * scale for v in values]
            else:
                clipped[key] = list(values)

        logger.debug(
            "Gradient clipping applied: max_norm=%.4f params=%d",
            max_norm,
            len(gradients),
        )
        return clipped


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_coordinator: Optional[FederatedLearningCoordinator] = None


def get_federated_coordinator() -> FederatedLearningCoordinator:
    """Return the singleton FederatedLearningCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = FederatedLearningCoordinator()
    return _coordinator
