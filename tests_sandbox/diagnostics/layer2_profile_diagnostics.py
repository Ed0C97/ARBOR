"""Layer 2: Domain Profile Diagnostics.

Tests domain profile quality, dimension configuration, and calibration.
Answers: "Is the domain profile correctly configured?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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

from data_generators import DomainProfileFactory, EntityFactory


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
class Layer2Report:
    """Complete Layer 2 diagnostic report."""
    layer_name: str = "Layer 2: Domain Profile"
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
# Layer 2 Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class Layer2Diagnostics:
    """Layer 2 diagnostic tests for domain profile validation.
    
    Tests:
    - Required profile fields
    - Dimension configuration quality
    - Weight distribution balance
    - Language consistency
    - Category coverage
    - Calibration entity scoring
    """
    
    # Minimum requirements
    MIN_DIMENSIONS = 4
    MAX_DIMENSIONS = 15
    MIN_CATEGORIES = 3
    MAX_CATEGORIES = 30
    
    def __init__(self):
        self.domain_factory = DomainProfileFactory()
        self.entity_factory = EntityFactory()
    
    def run_all(self, domain_intake: Any = None) -> Layer2Report:
        """Run all Layer 2 diagnostic tests.
        
        Args:
            domain_intake: Optional DomainIntake to test. If None, generates one.
        """
        report = Layer2Report()
        
        if domain_intake is None:
            domain_intake = self.domain_factory.generate()
        
        # Run tests
        report.results.append(self.test_required_fields(domain_intake))
        report.results.append(self.test_dimension_count(domain_intake))
        report.results.append(self.test_dimension_distinctness(domain_intake))
        report.results.append(self.test_weight_distribution(domain_intake))
        report.results.append(self.test_language_consistency(domain_intake))
        report.results.append(self.test_entity_type_coverage(domain_intake))
        report.results.append(self.test_quality_aspect_coverage(domain_intake))
        report.results.append(self.test_calibration_samples(domain_intake))
        
        return report
    
    def test_required_fields(self, intake: Any) -> DiagnosticResult:
        """Test that required profile fields are present."""
        required = ["domain_name", "vertical", "language"]
        missing = []
        
        for field in required:
            value = getattr(intake, field, None)
            if value is None or (isinstance(value, str) and value == ""):
                missing.append(field)
        
        if missing:
            return DiagnosticResult(
                test_name="Required Fields",
                passed=False,
                message=f"Missing required fields: {missing}",
                severity="critical",
                details={"missing": missing},
                fix_action=f"Provide values for: {', '.join(missing)}",
            )
        
        return DiagnosticResult(
            test_name="Required Fields",
            passed=True,
            message="All required fields present",
            severity="info",
            details={
                "domain_name": intake.domain_name,
                "vertical": str(intake.vertical),
                "language": intake.language,
            },
        )
    
    def test_dimension_count(self, intake: Any) -> DiagnosticResult:
        """Test that dimension count is reasonable."""
        aspects = getattr(intake, "quality_aspects", [])
        count = len(aspects)
        
        if count < self.MIN_DIMENSIONS:
            return DiagnosticResult(
                test_name="Dimension Count",
                passed=False,
                message=f"Too few dimensions: {count} (min: {self.MIN_DIMENSIONS})",
                severity="error",
                details={"count": count, "min": self.MIN_DIMENSIONS},
                fix_action=f"Add at least {self.MIN_DIMENSIONS - count} more quality aspects",
            )
        
        if count > self.MAX_DIMENSIONS:
            return DiagnosticResult(
                test_name="Dimension Count",
                passed=False,
                message=f"Too many dimensions: {count} (max: {self.MAX_DIMENSIONS})",
                severity="warning",
                details={"count": count, "max": self.MAX_DIMENSIONS},
                fix_action="Consolidate related dimensions to reduce complexity",
            )
        
        return DiagnosticResult(
            test_name="Dimension Count",
            passed=True,
            message=f"{count} dimensions configured",
            severity="info",
            details={
                "count": count,
                "aspect_ids": [a.aspect_id for a in aspects],
            },
        )
    
    def test_dimension_distinctness(self, intake: Any) -> DiagnosticResult:
        """Test that dimensions are semantically distinct (no duplicates)."""
        aspects = getattr(intake, "quality_aspects", [])
        
        # Check for duplicate IDs
        aspect_ids = [a.aspect_id for a in aspects]
        duplicates = [id for id in set(aspect_ids) if aspect_ids.count(id) > 1]
        
        if duplicates:
            return DiagnosticResult(
                test_name="Dimension Distinctness",
                passed=False,
                message=f"Duplicate dimension IDs: {duplicates}",
                severity="error",
                details={"duplicates": duplicates},
                fix_action="Remove or rename duplicate dimensions",
            )
        
        # Check for similar aspect names (basic heuristic)
        similar_pairs = []
        for i, a1 in enumerate(aspects):
            for a2 in aspects[i+1:]:
                # Simple similarity check
                words1 = set(a1.aspect_id.lower().replace("_", " ").split())
                words2 = set(a2.aspect_id.lower().replace("_", " ").split())
                if words1 & words2:  # Overlapping words
                    similar_pairs.append((a1.aspect_id, a2.aspect_id))
        
        if similar_pairs:
            return DiagnosticResult(
                test_name="Dimension Distinctness",
                passed=True,  # Just a warning
                message=f"{len(similar_pairs)} potentially similar dimensions",
                severity="warning" if len(similar_pairs) > 2 else "info",
                details={"similar_pairs": similar_pairs[:5]},
                fix_action="Review similar dimensions to ensure they measure different qualities",
            )
        
        return DiagnosticResult(
            test_name="Dimension Distinctness",
            passed=True,
            message="All dimensions are distinct",
            severity="info",
        )
    
    def test_weight_distribution(self, intake: Any) -> DiagnosticResult:
        """Test that dimension weights/importance are well distributed."""
        aspects = getattr(intake, "quality_aspects", [])
        
        if not aspects:
            return DiagnosticResult(
                test_name="Weight Distribution",
                passed=False,
                message="No quality aspects to analyze",
                severity="error",
            )
        
        importances = [a.importance for a in aspects]
        
        # Check distribution
        avg_importance = sum(importances) / len(importances)
        min_imp = min(importances)
        max_imp = max(importances)
        
        # Calculate variance
        variance = sum((x - avg_importance) ** 2 for x in importances) / len(importances)
        std_dev = math.sqrt(variance)
        
        # Analysis
        issues = []
        
        # All same importance (no differentiation)
        if std_dev < 0.5:
            issues.append("All dimensions have similar importance - consider differentiating")
        
        # Too many low-importance dimensions
        low_count = sum(1 for i in importances if i <= 2)
        if low_count > len(importances) * 0.5:
            issues.append(f"{low_count} dimensions have low importance (≤2)")
        
        # No critical dimensions
        high_count = sum(1 for i in importances if i >= 4)
        if high_count == 0:
            issues.append("No high-importance dimensions (≥4)")
        
        if issues:
            return DiagnosticResult(
                test_name="Weight Distribution",
                passed=True,
                message="Weight distribution could be improved",
                severity="warning",
                details={
                    "avg_importance": round(avg_importance, 2),
                    "min": min_imp,
                    "max": max_imp,
                    "std_dev": round(std_dev, 2),
                    "issues": issues,
                },
                fix_action="Review and adjust dimension importances to reflect true priorities",
            )
        
        return DiagnosticResult(
            test_name="Weight Distribution",
            passed=True,
            message=f"Good weight distribution (avg: {avg_importance:.1f}, std: {std_dev:.1f})",
            severity="info",
            details={
                "avg_importance": round(avg_importance, 2),
                "min": min_imp,
                "max": max_imp,
                "std_dev": round(std_dev, 2),
            },
        )
    
    def test_language_consistency(self, intake: Any) -> DiagnosticResult:
        """Test that text content matches the specified language."""
        language = getattr(intake, "language", "en")
        domain_name = getattr(intake, "domain_name", "")
        audience = getattr(intake, "target_audience_description", "")
        
        # Simple language detection heuristic
        # In reality, you'd use a proper language detection library
        
        italian_indicators = ["ristorante", "hotel", "della", "trattoria", "italiano", "italiana"]
        english_indicators = ["restaurant", "hotel", "the", "dining", "luxury"]
        
        text = f"{domain_name} {audience}".lower()
        
        italian_score = sum(1 for ind in italian_indicators if ind in text)
        english_score = sum(1 for ind in english_indicators if ind in text)
        
        detected = "it" if italian_score > english_score else "en"
        
        if language == "it" and detected == "en" and english_score > 2:
            return DiagnosticResult(
                test_name="Language Consistency",
                passed=True,
                message="Text appears to be in English but language set to Italian",
                severity="warning",
                details={
                    "expected_language": language,
                    "detected": detected,
                    "sample_text": domain_name[:50],
                },
                fix_action="Ensure all user-facing text matches the specified language",
            )
        
        return DiagnosticResult(
            test_name="Language Consistency",
            passed=True,
            message=f"Language appears consistent ({language})",
            severity="info",
        )
    
    def test_entity_type_coverage(self, intake: Any) -> DiagnosticResult:
        """Test that entity types are properly defined."""
        entity_types = getattr(intake, "entity_types", [])
        
        if not entity_types:
            return DiagnosticResult(
                test_name="Entity Type Coverage",
                passed=False,
                message="No entity types defined",
                severity="warning",
                fix_action="Define at least one entity type for your domain",
            )
        
        # Check for required attributes
        issues = []
        for et in entity_types:
            if not getattr(et, "name", ""):
                issues.append("Entity type missing name")
            if not getattr(et, "description", ""):
                issues.append(f"Entity type '{getattr(et, 'name', '?')}' missing description")
        
        if issues:
            return DiagnosticResult(
                test_name="Entity Type Coverage",
                passed=True,
                message=f"{len(issues)} entity type issues",
                severity="warning",
                details={"issues": issues},
                fix_action="Add descriptions for all entity types",
            )
        
        return DiagnosticResult(
            test_name="Entity Type Coverage",
            passed=True,
            message=f"{len(entity_types)} entity types defined",
            severity="info",
            details={
                "types": [getattr(et, "name", "?") for et in entity_types],
            },
        )
    
    def test_quality_aspect_coverage(self, intake: Any) -> DiagnosticResult:
        """Test that quality aspects have sufficient detail."""
        aspects = getattr(intake, "quality_aspects", [])
        
        if not aspects:
            return DiagnosticResult(
                test_name="Quality Aspect Coverage",
                passed=False,
                message="No quality aspects defined",
                severity="error",
                fix_action="Define quality aspects to guide scoring",
            )
        
        # Check for high-importance aspects without descriptions
        incomplete = []
        for a in aspects:
            if a.importance >= 4:
                if not a.what_makes_it_great and not a.what_makes_it_poor:
                    incomplete.append(a.aspect_id)
        
        if incomplete:
            return DiagnosticResult(
                test_name="Quality Aspect Coverage",
                passed=True,
                message=f"{len(incomplete)} high-importance aspects lack descriptions",
                severity="warning",
                details={"incomplete": incomplete},
                fix_action="Add 'what makes it great/poor' for high-importance aspects",
            )
        
        return DiagnosticResult(
            test_name="Quality Aspect Coverage",
            passed=True,
            message=f"{len(aspects)} quality aspects with good coverage",
            severity="info",
        )
    
    def test_calibration_samples(self, intake: Any) -> DiagnosticResult:
        """Test that calibration samples are provided."""
        best = getattr(intake, "sample_best_entities", [])
        average = getattr(intake, "sample_average_entities", [])
        
        if not best and not average:
            return DiagnosticResult(
                test_name="Calibration Samples",
                passed=True,
                message="No calibration entities provided",
                severity="warning",
                fix_action="Add sample best and average entities for better scoring calibration",
            )
        
        if best and not average:
            return DiagnosticResult(
                test_name="Calibration Samples",
                passed=True,
                message="Only best entities provided, missing average",
                severity="warning",
                details={"best": best},
                fix_action="Add sample average entities for full calibration range",
            )
        
        return DiagnosticResult(
            test_name="Calibration Samples",
            passed=True,
            message=f"{len(best)} best + {len(average)} average calibration entities",
            severity="info",
            details={
                "best_entities": best,
                "average_entities": average,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pytest tests
# ═══════════════════════════════════════════════════════════════════════════

if PYTEST_AVAILABLE:
    @pytest.fixture
    def diagnostics():
        return Layer2Diagnostics()

    @pytest.fixture
    def domain_intake(diagnostics):
        return diagnostics.domain_factory.generate()

    @pytest.mark.layer2
    def test_required_fields(diagnostics, domain_intake):
        """Test required fields present."""
        result = diagnostics.test_required_fields(domain_intake)
        assert result.passed, result.message

    @pytest.mark.layer2
    def test_dimension_count(diagnostics, domain_intake):
        """Test dimension count is reasonable."""
        result = diagnostics.test_dimension_count(domain_intake)
        assert result.passed, result.message

    @pytest.mark.layer2
    def test_dimension_distinctness(diagnostics, domain_intake):
        """Test dimensions are distinct."""
        result = diagnostics.test_dimension_distinctness(domain_intake)
        assert result.passed, result.message


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

async def run(domain_intake: Any = None) -> Layer2Report:
    """Run all Layer 2 diagnostics."""
    diagnostics = Layer2Diagnostics()
    return diagnostics.run_all(domain_intake)


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(run())
    report.print_summary()
