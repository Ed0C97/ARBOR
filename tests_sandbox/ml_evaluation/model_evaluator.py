"""ML Model Evaluator for ARBOR test sandbox.

Evaluates ML model performance, scoring accuracy, and drift detection.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generators import EntityFactory, ScenarioBuilder


@dataclass
class EvaluationMetrics:
    """Evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    correlation: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "correlation": round(self.correlation, 4),
        }


@dataclass
class DriftReport:
    """Model drift detection report."""
    drift_detected: bool = False
    drift_score: float = 0.0
    feature_drifts: dict[str, float] = field(default_factory=dict)
    output_drift: float = 0.0
    recommendation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "drift_score": round(self.drift_score, 4),
            "feature_drifts": {k: round(v, 4) for k, v in self.feature_drifts.items()},
            "output_drift": round(self.output_drift, 4),
            "recommendation": self.recommendation,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    drift: DriftReport = field(default_factory=DriftReport)
    samples_evaluated: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "samples_evaluated": self.samples_evaluated,
            "metrics": self.metrics.to_dict(),
            "drift": self.drift.to_dict(),
            "details": self.details,
        }
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        print("\n🤖 ML MODEL EVALUATION")
        print("═" * 50)
        
        print("\n📊 Performance Metrics:")
        print(f"  • Accuracy:    {self.metrics.accuracy:.2%}")
        print(f"  • Precision:   {self.metrics.precision:.2%}")
        print(f"  • Recall:      {self.metrics.recall:.2%}")
        print(f"  • F1 Score:    {self.metrics.f1_score:.2%}")
        print(f"  • MAE:         {self.metrics.mae:.2f}")
        print(f"  • RMSE:        {self.metrics.rmse:.2f}")
        
        print("\n📈 Drift Detection:")
        status = "⚠️ DRIFT DETECTED" if self.drift.drift_detected else "✅ No drift"
        print(f"  • Status:      {status}")
        print(f"  • Drift Score: {self.drift.drift_score:.4f}")
        if self.drift.recommendation:
            print(f"  • Action:      {self.drift.recommendation}")
        
        print(f"\n📋 Samples Evaluated: {self.samples_evaluated}")


class ModelEvaluator:
    """Evaluator for ML model performance.
    
    Tests:
    - Scoring accuracy against ground truth
    - Score distribution analysis
    - Feature importance validation
    - Drift detection
    """
    
    DRIFT_THRESHOLD = 0.15
    
    def __init__(self, scoring_fn: Callable | None = None):
        """Initialize evaluator.
        
        Args:
            scoring_fn: Scoring function to evaluate. If None, uses mock.
        """
        self.entity_factory = EntityFactory()
        self.scenario_builder = ScenarioBuilder()
        self.scoring_fn = scoring_fn or self._mock_scoring_fn
        
        # Baseline distributions (simulated)
        self.baseline_means = {
            "fine_dining": 85,
            "trattoria": 65,
            "casual": 50,
            "luxury": 90,
            "boutique": 75,
            "business": 60,
        }
        self.baseline_stds = {k: 10 for k in self.baseline_means}
    
    def _mock_scoring_fn(self, entity: dict) -> dict:
        """Mock scoring function."""
        import random
        
        metadata = entity.get("metadata", {})
        profile = metadata.get("profile", "casual")
        
        base = self.baseline_means.get(profile, 60)
        score = random.gauss(base, self.baseline_stds.get(profile, 10))
        
        return {
            "entity_id": entity.get("id"),
            "overall_score": max(0, min(100, score)),
            "confidence": random.uniform(0.7, 0.95),
        }
    
    def evaluate(self, samples: int = 100) -> EvaluationReport:
        """Run full evaluation.
        
        Args:
            samples: Number of entities to evaluate
        """
        report = EvaluationReport()
        
        # Generate test entities with known profiles
        entities = []
        expected_scores = []
        
        profiles = list(self.baseline_means.keys())
        per_profile = samples // len(profiles)
        
        for profile in profiles:
            entity_type = "restaurant" if profile in ["fine_dining", "trattoria", "casual"] else "hotel"
            for _ in range(per_profile):
                entity = self.entity_factory.restaurant(profile=profile) if entity_type == "restaurant" else self.entity_factory.hotel(profile=profile)
                entities.append(entity)
                expected_scores.append(self.baseline_means[profile])
        
        # Score entities
        actual_scores = []
        for entity in entities:
            result = self.scoring_fn(entity)
            actual_scores.append(result.get("overall_score", 0))
        
        report.samples_evaluated = len(entities)
        
        # Calculate metrics
        report.metrics = self._calculate_metrics(expected_scores, actual_scores)
        
        # Detect drift
        report.drift = self._detect_drift(entities, actual_scores)
        
        # Store details
        report.details = {
            "score_distribution": {
                "mean": round(statistics.mean(actual_scores), 2),
                "std": round(statistics.stdev(actual_scores) if len(actual_scores) > 1 else 0, 2),
                "min": round(min(actual_scores), 2),
                "max": round(max(actual_scores), 2),
            },
            "profiles_tested": list(set(e.get("metadata", {}).get("profile", "unknown") for e in entities)),
        }
        
        return report
    
    def _calculate_metrics(
        self,
        expected: list[float],
        actual: list[float],
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics."""
        if len(expected) != len(actual) or len(expected) == 0:
            return EvaluationMetrics()
        
        # Mean Absolute Error
        mae = statistics.mean(abs(e - a) for e, a in zip(expected, actual))
        
        # Root Mean Square Error
        mse = statistics.mean((e - a) ** 2 for e, a in zip(expected, actual))
        rmse = mse ** 0.5
        
        # Correlation
        try:
            mean_e = statistics.mean(expected)
            mean_a = statistics.mean(actual)
            
            numerator = sum((e - mean_e) * (a - mean_a) for e, a in zip(expected, actual))
            
            std_e = statistics.stdev(expected) if len(expected) > 1 else 1
            std_a = statistics.stdev(actual) if len(actual) > 1 else 1
            
            denominator = (len(expected) - 1) * std_e * std_a
            
            correlation = numerator / denominator if denominator != 0 else 0
        except Exception:
            correlation = 0
        
        # Binary classification metrics (threshold at 70)
        threshold = 70
        expected_binary = [1 if e >= threshold else 0 for e in expected]
        actual_binary = [1 if a >= threshold else 0 for a in actual]
        
        tp = sum(1 for e, a in zip(expected_binary, actual_binary) if e == 1 and a == 1)
        fp = sum(1 for e, a in zip(expected_binary, actual_binary) if e == 0 and a == 1)
        fn = sum(1 for e, a in zip(expected_binary, actual_binary) if e == 1 and a == 0)
        tn = sum(1 for e, a in zip(expected_binary, actual_binary) if e == 0 and a == 0)
        
        accuracy = (tp + tn) / len(expected) if expected else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mae=mae,
            rmse=rmse,
            correlation=correlation,
        )
    
    def _detect_drift(
        self,
        entities: list[dict],
        scores: list[float],
    ) -> DriftReport:
        """Detect model drift."""
        report = DriftReport()
        
        # Group scores by profile
        scores_by_profile = {}
        for entity, score in zip(entities, scores):
            profile = entity.get("metadata", {}).get("profile", "unknown")
            if profile not in scores_by_profile:
                scores_by_profile[profile] = []
            scores_by_profile[profile].append(score)
        
        # Calculate drift for each profile
        max_drift = 0
        for profile, profile_scores in scores_by_profile.items():
            if profile not in self.baseline_means:
                continue
            
            current_mean = statistics.mean(profile_scores) if profile_scores else 0
            baseline_mean = self.baseline_means[profile]
            
            # Normalized drift
            drift = abs(current_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0
            report.feature_drifts[profile] = drift
            
            if drift > max_drift:
                max_drift = drift
        
        # Overall drift score
        report.drift_score = max_drift
        report.drift_detected = max_drift > self.DRIFT_THRESHOLD
        
        if report.drift_detected:
            worst_profile = max(report.feature_drifts, key=report.feature_drifts.get) if report.feature_drifts else "unknown"
            report.recommendation = f"Retrain model - significant drift in '{worst_profile}' profile scoring"
        
        return report


async def run(samples: int = 100, save_report: bool = True) -> EvaluationReport:
    """Run ML evaluation.
    
    Args:
        samples: Number of samples to evaluate
        save_report: Save report to file
    """
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(samples=samples)
    
    if save_report:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / "ml_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\n📁 Report saved to: {report_path}")
    
    report.print_summary()
    
    return report


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
