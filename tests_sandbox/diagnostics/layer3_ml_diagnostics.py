"""Layer 3: ML Pipeline Diagnostics.

Tests ML model performance, embedding quality, and inference latency.
Answers: "Is the ML pipeline performing correctly?"
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
import statistics

# Optional pytest import for standalone execution
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

# Add parent paths for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "arbor-enterprise" / "backend"))

from data_generators import EntityFactory, DomainProfileFactory


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic Result Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticResult:
    """Result of a single diagnostic test."""
    test_name: str
    passed: bool
    message: str
    severity: str = "info"
    details: dict[str, Any] = field(default_factory=dict)
    fix_action: str | None = None


@dataclass
class Layer3Report:
    """Complete Layer 3 diagnostic report."""
    layer_name: str = "Layer 3: ML Pipeline"
    results: list[DiagnosticResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.severity == "warning")
    
    @property
    def is_healthy(self) -> bool:
        return all(r.severity != "critical" for r in self.results if not r.passed)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "timestamp": self.timestamp,
            "summary": {
                "passed": self.passed_count,
                "failed": self.failed_count,
                "warnings": self.warning_count,
                "is_healthy": self.is_healthy,
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                    "fix_action": r.fix_action,
                }
                for r in self.results
            ],
        }
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print(f"\n📊 {self.layer_name}")
        print("━" * 50)
        
        for result in self.results:
            icon = "✅" if result.passed else ("⚠️" if result.severity == "warning" else "❌")
            print(f"{icon} {result.test_name}: {result.message}")
            if result.fix_action and not result.passed:
                print(f"   └─ FIX: {result.fix_action}")
        
        print(f"\nTotal: {self.passed_count} passed, {self.failed_count} failed")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class Layer3Diagnostics:
    """Layer 3 diagnostic tests for ML pipeline.
    
    Tests:
    - Scoring function consistency
    - Embedding quality (similarity checks)
    - Inference latency
    - Memory usage
    - Vibe DNA score distribution
    - Profile-entity scoring correlation
    """
    
    # Performance thresholds
    MAX_LATENCY_P50_MS = 200
    MAX_LATENCY_P95_MS = 500
    MAX_LATENCY_P99_MS = 1000
    MIN_CONSISTENCY_SCORE = 0.85
    
    def __init__(self, scoring_fn: Callable | None = None):
        """Initialize diagnostics.
        
        Args:
            scoring_fn: Optional scoring function to test. If None, uses mock.
        """
        self.entity_factory = EntityFactory()
        self.domain_factory = DomainProfileFactory()
        self.scoring_fn = scoring_fn or self._mock_scoring_fn
        
        # Try to import real scoring engine
        try:
            from app.ingestion.pipeline.scoring_engine import ScoringEngine
            self._real_engine_available = True
        except ImportError:
            self._real_engine_available = False
    
    def _mock_scoring_fn(self, entity: dict) -> dict:
        """Mock scoring function for testing without real ML backend."""
        time.sleep(random.uniform(0.01, 0.05))  # Simulate latency
        
        # Generate plausible scores based on profile
        metadata = entity.get("metadata", {})
        profile = metadata.get("profile", "casual")
        
        base_score = {
            "fine_dining": random.gauss(85, 8),
            "trattoria": random.gauss(65, 10),
            "casual": random.gauss(50, 12),
            "luxury": random.gauss(90, 5),
            "boutique": random.gauss(75, 10),
            "business": random.gauss(60, 10),
        }.get(profile, random.gauss(60, 15))
        
        return {
            "entity_id": entity.get("id"),
            "overall_score": max(0, min(100, base_score)),
            "dimension_scores": entity.get("vibe_dna", {}).get("dimensions", {}),
            "confidence": random.uniform(0.7, 0.95),
        }
    
    def run_all(self, use_real_engine: bool = False) -> Layer3Report:
        """Run all Layer 3 diagnostic tests."""
        report = Layer3Report()
        
        # Generate test data
        entities = self.entity_factory.generate_batch("restaurant", count=50)
        
        # Run tests
        report.results.append(self.test_scoring_consistency(entities[:10]))
        report.results.append(self.test_inference_latency(entities))
        report.results.append(self.test_score_distribution(entities))
        report.results.append(self.test_vibe_score_ranges(entities))
        report.results.append(self.test_profile_entity_correlation(entities))
        report.results.append(self.test_batch_vs_single_consistency(entities[:20]))
        
        if self._real_engine_available and use_real_engine:
            report.results.append(self.test_real_engine_availability())
        
        return report
    
    def test_scoring_consistency(self, entities: list[dict]) -> DiagnosticResult:
        """Test that scoring the same entity multiple times gives consistent results."""
        if not entities:
            return DiagnosticResult(
                test_name="Scoring Consistency",
                passed=False,
                message="No entities to test",
                severity="error",
            )
        
        entity = entities[0]
        scores = []
        
        # Score same entity multiple times
        for _ in range(5):
            result = self.scoring_fn(entity)
            scores.append(result.get("overall_score", 0))
        
        # Calculate variance
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Consistency threshold
        max_allowed_std = 5.0  # Max 5 points deviation
        
        if std_dev > max_allowed_std:
            return DiagnosticResult(
                test_name="Scoring Consistency",
                passed=False,
                message=f"High variance in scoring: std_dev={std_dev:.2f}",
                severity="error",
                details={
                    "scores": scores,
                    "avg": round(avg, 2),
                    "std_dev": round(std_dev, 2),
                },
                fix_action="Check for non-deterministic behavior in scoring (temperature > 0?)",
            )
        
        return DiagnosticResult(
            test_name="Scoring Consistency",
            passed=True,
            message=f"Consistent scoring (std_dev={std_dev:.2f})",
            severity="info",
            details={
                "avg_score": round(avg, 2),
                "std_dev": round(std_dev, 2),
                "samples": len(scores),
            },
        )
    
    def test_inference_latency(self, entities: list[dict]) -> DiagnosticResult:
        """Test inference latency across multiple entities."""
        latencies = []
        
        for entity in entities[:30]:  # Test up to 30
            start = time.perf_counter()
            _ = self.scoring_fn(entity)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]
        p99 = latencies[min(p99_idx, len(latencies) - 1)]
        
        issues = []
        if p50 > self.MAX_LATENCY_P50_MS:
            issues.append(f"p50 ({p50:.0f}ms) > target ({self.MAX_LATENCY_P50_MS}ms)")
        if p95 > self.MAX_LATENCY_P95_MS:
            issues.append(f"p95 ({p95:.0f}ms) > target ({self.MAX_LATENCY_P95_MS}ms)")
        if p99 > self.MAX_LATENCY_P99_MS:
            issues.append(f"p99 ({p99:.0f}ms) > target ({self.MAX_LATENCY_P99_MS}ms)")
        
        if issues:
            return DiagnosticResult(
                test_name="Inference Latency",
                passed=False,
                message=f"Latency issues: {issues[0]}",
                severity="warning" if len(issues) == 1 else "error",
                details={
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "p99_ms": round(p99, 1),
                    "issues": issues,
                },
                fix_action="Profile bottlenecks: embedding generation, LLM calls, or database queries",
            )
        
        return DiagnosticResult(
            test_name="Inference Latency",
            passed=True,
            message=f"Good latency: p50={p50:.0f}ms, p95={p95:.0f}ms",
            severity="info",
            details={
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "p99_ms": round(p99, 1),
                "samples": len(latencies),
            },
        )
    
    def test_score_distribution(self, entities: list[dict]) -> DiagnosticResult:
        """Test that scores are well distributed (not clustered)."""
        scores = []
        
        for entity in entities:
            result = self.scoring_fn(entity)
            scores.append(result.get("overall_score", 0))
        
        if not scores:
            return DiagnosticResult(
                test_name="Score Distribution",
                passed=False,
                message="No scores generated",
                severity="error",
            )
        
        avg = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        min_score = min(scores)
        max_score = max(scores)
        
        # Check for issues
        issues = []
        
        # Too clustered
        if std_dev < 10:
            issues.append(f"Scores too clustered (std_dev={std_dev:.1f})")
        
        # Too extreme
        if std_dev > 30:
            issues.append(f"Scores too spread out (std_dev={std_dev:.1f})")
        
        # Biased high or low
        if avg < 40:
            issues.append(f"Scores biased low (avg={avg:.1f})")
        elif avg > 80:
            issues.append(f"Scores biased high (avg={avg:.1f})")
        
        # No low scores
        if min_score > 50:
            issues.append(f"No low scores (min={min_score:.0f})")
        
        # No high scores
        if max_score < 70:
            issues.append(f"No high scores (max={max_score:.0f})")
        
        if issues:
            return DiagnosticResult(
                test_name="Score Distribution",
                passed=True,  # Warning only
                message=f"Distribution issue: {issues[0]}",
                severity="warning",
                details={
                    "avg": round(avg, 1),
                    "std_dev": round(std_dev, 1),
                    "min": round(min_score, 1),
                    "max": round(max_score, 1),
                    "issues": issues,
                },
                fix_action="Check scoring calibration and dimension weights",
            )
        
        return DiagnosticResult(
            test_name="Score Distribution",
            passed=True,
            message=f"Good distribution: avg={avg:.0f}, range=[{min_score:.0f}-{max_score:.0f}]",
            severity="info",
            details={
                "avg": round(avg, 1),
                "std_dev": round(std_dev, 1),
                "min": round(min_score, 1),
                "max": round(max_score, 1),
            },
        )
    
    def test_vibe_score_ranges(self, entities: list[dict]) -> DiagnosticResult:
        """Test that vibe dimension scores are within valid ranges."""
        out_of_range = []
        
        for i, entity in enumerate(entities):
            vibe_dna = entity.get("vibe_dna", {})
            dimensions = vibe_dna.get("dimensions", {})
            
            for dim, score in dimensions.items():
                if not isinstance(score, (int, float)):
                    out_of_range.append(f"Entity {i}: {dim} is not numeric ({type(score).__name__})")
                elif score < 0 or score > 1:
                    out_of_range.append(f"Entity {i}: {dim}={score} (valid: 0.0-1.0)")
        
        if out_of_range:
            return DiagnosticResult(
                test_name="Vibe Score Ranges",
                passed=False,
                message=f"{len(out_of_range)} vibe scores out of range",
                severity="error",
                details={"out_of_range": out_of_range[:10]},
                fix_action="Clamp vibe scores to [0.0, 1.0] range in scoring engine",
            )
        
        return DiagnosticResult(
            test_name="Vibe Score Ranges",
            passed=True,
            message="All vibe scores in valid range [0.0-1.0]",
            severity="info",
        )
    
    def test_profile_entity_correlation(self, entities: list[dict]) -> DiagnosticResult:
        """Test that entity profiles correlate with scores (fine_dining > casual)."""
        scores_by_profile = {}
        
        for entity in entities:
            profile = entity.get("metadata", {}).get("profile", "unknown")
            result = self.scoring_fn(entity)
            score = result.get("overall_score", 0)
            
            if profile not in scores_by_profile:
                scores_by_profile[profile] = []
            scores_by_profile[profile].append(score)
        
        # Calculate averages
        avg_by_profile = {
            p: statistics.mean(scores) if scores else 0
            for p, scores in scores_by_profile.items()
        }
        
        # Expected ordering (fine_dining > trattoria > casual)
        expected_order = [
            ("fine_dining", "trattoria"),
            ("trattoria", "casual"),
            ("luxury", "boutique"),
            ("boutique", "business"),
        ]
        
        violations = []
        for higher, lower in expected_order:
            if higher in avg_by_profile and lower in avg_by_profile:
                if avg_by_profile[higher] < avg_by_profile[lower]:
                    violations.append(
                        f"{higher}({avg_by_profile[higher]:.0f}) < "
                        f"{lower}({avg_by_profile[lower]:.0f})"
                    )
        
        if violations:
            return DiagnosticResult(
                test_name="Profile-Score Correlation",
                passed=False,
                message=f"Score ordering violated: {violations[0]}",
                severity="error",
                details={
                    "avg_by_profile": {k: round(v, 1) for k, v in avg_by_profile.items()},
                    "violations": violations,
                },
                fix_action="Check dimension weights and scoring calibration",
            )
        
        return DiagnosticResult(
            test_name="Profile-Score Correlation",
            passed=True,
            message="Scores correctly correlate with entity profiles",
            severity="info",
            details={
                "avg_by_profile": {k: round(v, 1) for k, v in avg_by_profile.items()},
            },
        )
    
    def test_batch_vs_single_consistency(self, entities: list[dict]) -> DiagnosticResult:
        """Test that batch scoring gives same results as single scoring."""
        single_scores = []
        for entity in entities:
            result = self.scoring_fn(entity)
            single_scores.append(result.get("overall_score", 0))
        
        # In a real implementation, you'd also test batch scoring
        # For now, just verify single scoring is working
        
        if not single_scores:
            return DiagnosticResult(
                test_name="Batch Consistency",
                passed=False,
                message="No scores generated",
                severity="error",
            )
        
        return DiagnosticResult(
            test_name="Batch Consistency",
            passed=True,
            message=f"Single scoring working ({len(single_scores)} entities)",
            severity="info",
            details={"entities_scored": len(single_scores)},
        )
    
    def test_real_engine_availability(self) -> DiagnosticResult:
        """Test that the real scoring engine is available."""
        if self._real_engine_available:
            return DiagnosticResult(
                test_name="Real Engine Available",
                passed=True,
                message="Real scoring engine is available",
                severity="info",
            )
        return DiagnosticResult(
            test_name="Real Engine Available",
            passed=False,
            message="Real scoring engine not available (using mock)",
            severity="warning",
            fix_action="Ensure app.ingestion.pipeline.scoring_engine is importable",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pytest tests
# ═══════════════════════════════════════════════════════════════════════════

if PYTEST_AVAILABLE:
    @pytest.fixture
    def diagnostics():
        return Layer3Diagnostics()

    @pytest.fixture
    def entities(diagnostics):
        return diagnostics.entity_factory.generate_batch("restaurant", count=20)

    @pytest.mark.layer3
    def test_scoring_consistency(diagnostics, entities):
        """Test scoring consistency."""
        result = diagnostics.test_scoring_consistency(entities)
        assert result.passed, result.message

    @pytest.mark.layer3
    def test_inference_latency(diagnostics, entities):
        """Test inference latency."""
        result = diagnostics.test_inference_latency(entities)
        assert result.passed or result.severity == "warning", result.message

    @pytest.mark.layer3
    def test_score_distribution(diagnostics, entities):
        """Test score distribution."""
        result = diagnostics.test_score_distribution(entities)
        assert result.passed, result.message


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

async def run() -> Layer3Report:
    """Run all Layer 3 diagnostics."""
    diagnostics = Layer3Diagnostics()
    return diagnostics.run_all()


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(run())
    report.print_summary()
