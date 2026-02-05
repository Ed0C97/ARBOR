"""A/B Testing Framework for experiments.

TIER 10 - Point 52: A/B Testing Framework

Framework for running controlled experiments on:
- Prompt versions
- Model parameters
- UI variants
- Ranking algorithms

Features:
- Variant assignment by user ID (consistent)
- Traffic allocation
- Metric collection
- Statistical significance testing
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_OUT = "rolled_out"


@dataclass
class Variant:
    """A variant in an experiment."""
    name: str
    weight: float  # Traffic percentage (0-1)
    config: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class Experiment:
    """Definition of an A/B experiment.
    
    TIER 10 - Point 52: Experiment configuration.
    """
    id: str
    name: str
    description: str
    variants: list[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    # Targeting
    traffic_percentage: float = 1.0  # % of users in experiment
    user_filter: dict = field(default_factory=dict)  # Optional user filtering
    
    # Metrics
    primary_metric: str = "conversion_rate"
    secondary_metrics: list[str] = field(default_factory=list)
    
    # Timing
    start_date: datetime | None = None
    end_date: datetime | None = None
    
    # Results
    results: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize variant weights
        total_weight = sum(v.weight for v in self.variants)
        if total_weight > 0:
            for v in self.variants:
                v.weight = v.weight / total_weight


class ExperimentAssigner:
    """Assigns users to experiment variants.
    
    TIER 10 - Point 52: Consistent variant assignment.
    
    Uses deterministic hashing for consistent assignment:
    - Same user always gets same variant
    - Assignment persists across sessions
    """
    
    def __init__(self, salt: str = "arbor_ab_salt"):
        self.salt = salt
    
    def assign(
        self,
        user_id: str,
        experiment: Experiment,
    ) -> Variant | None:
        """Assign a user to an experiment variant.
        
        Args:
            user_id: Unique user identifier
            experiment: The experiment to assign to
            
        Returns:
            Assigned variant, or None if user not in experiment
        """
        if experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # Check if user is in experiment (traffic percentage)
        if not self._in_experiment(user_id, experiment):
            return None
        
        # Determine variant
        variant = self._select_variant(user_id, experiment)
        
        logger.debug(
            f"Assigned user {user_id[:8]}... to variant "
            f"'{variant.name}' in experiment '{experiment.name}'"
        )
        
        return variant
    
    def _in_experiment(self, user_id: str, experiment: Experiment) -> bool:
        """Check if user is included in experiment based on traffic %."""
        hash_input = f"{self.salt}:experiment:{experiment.id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to 0-1 range
        normalized = (hash_value % 10000) / 10000
        
        return normalized < experiment.traffic_percentage
    
    def _select_variant(self, user_id: str, experiment: Experiment) -> Variant:
        """Select variant based on weighted random (deterministic)."""
        hash_input = f"{self.salt}:variant:{experiment.id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to 0-1 range
        normalized = (hash_value % 10000) / 10000
        
        # Select variant based on cumulative weights
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant
        
        # Fallback to last variant
        return experiment.variants[-1]


class ABTestingService:
    """Service for managing A/B experiments.
    
    TIER 10 - Point 52: A/B Testing Framework.
    
    Usage:
        ab = ABTestingService()
        
        # Create experiment
        experiment = Experiment(
            id="exp_001",
            name="New Discovery Prompt",
            description="Testing improved discovery prompt",
            variants=[
                Variant(name="control", weight=0.5, config={"prompt_version": "1.0"}),
                Variant(name="treatment", weight=0.5, config={"prompt_version": "2.0"}),
            ],
        )
        ab.register(experiment)
        
        # Assign user
        variant = ab.get_variant("user_123", "exp_001")
        if variant:
            prompt_version = variant.config["prompt_version"]
    """
    
    def __init__(self):
        self.experiments: dict[str, Experiment] = {}
        self.assigner = ExperimentAssigner()
        
        # Metrics storage (in production, use database)
        self.metrics: dict[str, dict[str, list[float]]] = {}
    
    def register(self, experiment: Experiment) -> None:
        """Register an experiment."""
        self.experiments[experiment.id] = experiment
        self.metrics[experiment.id] = {v.name: [] for v in experiment.variants}
        logger.info(f"Registered experiment: {experiment.name}")
    
    def start(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.start_date = datetime.utcnow()
        logger.info(f"Started experiment: {exp.name}")
        return True
    
    def stop(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.end_date = datetime.utcnow()
        logger.info(f"Stopped experiment: {exp.name}")
        return True
    
    def get_variant(
        self,
        user_id: str,
        experiment_id: str,
    ) -> Variant | None:
        """Get variant assignment for a user."""
        if experiment_id not in self.experiments:
            return None
        
        return self.assigner.assign(user_id, self.experiments[experiment_id])
    
    def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        value: float,
    ) -> None:
        """Record a metric observation for an experiment variant."""
        if experiment_id not in self.metrics:
            return
        
        if variant_name in self.metrics[experiment_id]:
            self.metrics[experiment_id][variant_name].append(value)
    
    def get_results(self, experiment_id: str) -> dict[str, Any]:
        """Get current experiment results with statistics."""
        if experiment_id not in self.experiments:
            return {}
        
        exp = self.experiments[experiment_id]
        results = {
            "experiment_id": experiment_id,
            "name": exp.name,
            "status": exp.status.value,
            "variants": {},
        }
        
        if experiment_id in self.metrics:
            for variant_name, values in self.metrics[experiment_id].items():
                if values:
                    import numpy as np
                    results["variants"][variant_name] = {
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                else:
                    results["variants"][variant_name] = {"n": 0}
        
        # Check for statistical significance
        if len(results["variants"]) >= 2:
            results["significance"] = self._check_significance(experiment_id)
        
        return results
    
    def _check_significance(self, experiment_id: str) -> dict[str, Any]:
        """Check statistical significance between variants."""
        metrics = self.metrics.get(experiment_id, {})
        variant_names = list(metrics.keys())
        
        if len(variant_names) < 2:
            return {"significant": False, "reason": "Not enough variants"}
        
        control_values = metrics.get(variant_names[0], [])
        treatment_values = metrics.get(variant_names[1], [])
        
        if len(control_values) < 30 or len(treatment_values) < 30:
            return {"significant": False, "reason": "Not enough samples (need 30+)"}
        
        # Simple t-test approximation
        import numpy as np
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values)
        treatment_std = np.std(treatment_values)
        
        pooled_std = np.sqrt(
            control_std**2 / len(control_values) +
            treatment_std**2 / len(treatment_values)
        )
        
        if pooled_std > 0:
            t_stat = abs(treatment_mean - control_mean) / pooled_std
            # Approximate p-value (simplified)
            p_value = max(0.001, 2 * (1 - min(0.9999, 0.5 + t_stat / 10)))
        else:
            p_value = 1.0
        
        return {
            "significant": p_value < 0.05,
            "p_value": round(p_value, 4),
            "lift": round((treatment_mean - control_mean) / control_mean * 100, 2)
            if control_mean > 0 else 0,
        }
    
    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments with status."""
        return [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status.value,
                "variants": len(exp.variants),
                "traffic": exp.traffic_percentage,
            }
            for exp in self.experiments.values()
        ]


# Singleton service
_ab_service: ABTestingService | None = None


def get_ab_service() -> ABTestingService:
    """Get singleton ABTestingService instance."""
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service
