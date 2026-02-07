"""Layer 1: Input Validation Diagnostics.

Tests data input quality, JSON schema validation, and field correctness.
Answers: "Are the input data correct?"
"""

from __future__ import annotations

import json
import re
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
    severity: str = "info"  # "info", "warning", "error", "critical"
    details: dict[str, Any] = field(default_factory=dict)
    fix_action: str | None = None


@dataclass
class Layer1Report:
    """Complete Layer 1 diagnostic report."""
    layer_name: str = "Layer 1: Input Validation"
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
# Layer 1 Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class Layer1Diagnostics:
    """Layer 1 diagnostic tests for input validation.
    
    Tests:
    - JSON schema validation
    - Required fields presence
    - Field type correctness
    - Value range validation
    - Unicode/special character handling
    - Large payload handling
    """
    
    # Expected schema for entities
    ENTITY_SCHEMA = {
        "required_fields": ["id", "name", "entity_type", "category", "city"],
        "optional_fields": ["description", "address", "price_tier", "vibe_dna", "phone", "website"],
        "field_types": {
            "id": str,
            "name": str,
            "entity_type": str,
            "category": str,
            "city": str,
            "price_tier": int,
            "description": str,
        },
        "value_constraints": {
            "price_tier": {"min": 1, "max": 5},
            "name": {"min_length": 1, "max_length": 200},
        },
    }
    
    # Expected schema for domain profiles
    DOMAIN_SCHEMA = {
        "required_fields": ["domain_name", "vertical", "language"],
        "field_types": {
            "domain_name": str,
            "vertical": str,
            "language": str,
            "geographic_focus": str,
        },
    }
    
    def __init__(self):
        self.entity_factory = EntityFactory()
        self.domain_factory = DomainProfileFactory()
    
    def run_all(self) -> Layer1Report:
        """Run all Layer 1 diagnostic tests."""
        report = Layer1Report()
        
        # Generate test data
        clean_entities = self.entity_factory.generate_batch("restaurant", count=20)
        problem_entities = [
            self.entity_factory.restaurant(with_issues=["missing_description"]),
            self.entity_factory.restaurant(with_issues=["invalid_price_tier"]),
            self.entity_factory.restaurant(with_issues=["missing_address"]),
        ]
        all_entities = clean_entities + problem_entities
        
        # Run tests
        report.results.append(self.test_required_fields(clean_entities))
        report.results.append(self.test_field_types(clean_entities))
        report.results.append(self.test_value_ranges(all_entities))
        report.results.append(self.test_unicode_handling())
        report.results.append(self.test_special_characters())
        report.results.append(self.test_empty_values(problem_entities))
        report.results.append(self.test_json_serialization(clean_entities))
        report.results.append(self.test_large_payload())
        
        return report
    
    def test_required_fields(self, entities: list[dict]) -> DiagnosticResult:
        """Test that required fields are present in all entities."""
        missing = []
        
        for i, entity in enumerate(entities):
            for field in self.ENTITY_SCHEMA["required_fields"]:
                if field not in entity or entity[field] is None:
                    missing.append(f"Entity {i}: missing '{field}'")
        
        if missing:
            return DiagnosticResult(
                test_name="Required Fields",
                passed=False,
                message=f"{len(missing)} missing required fields",
                severity="error",
                details={"missing": missing[:10]},  # First 10
                fix_action="Ensure all entities have: " + ", ".join(self.ENTITY_SCHEMA["required_fields"]),
            )
        
        return DiagnosticResult(
            test_name="Required Fields",
            passed=True,
            message=f"All {len(entities)} entities have required fields",
            severity="info",
        )
    
    def test_field_types(self, entities: list[dict]) -> DiagnosticResult:
        """Test that field types are correct."""
        type_errors = []
        
        for i, entity in enumerate(entities):
            for field, expected_type in self.ENTITY_SCHEMA["field_types"].items():
                if field in entity and entity[field] is not None:
                    if not isinstance(entity[field], expected_type):
                        type_errors.append(
                            f"Entity {i}: '{field}' is {type(entity[field]).__name__}, "
                            f"expected {expected_type.__name__}"
                        )
        
        if type_errors:
            return DiagnosticResult(
                test_name="Field Types",
                passed=False,
                message=f"{len(type_errors)} type mismatches",
                severity="warning",
                details={"errors": type_errors[:10]},
                fix_action="Add type coercion in data parser or fix source data",
            )
        
        return DiagnosticResult(
            test_name="Field Types",
            passed=True,
            message="All field types are correct",
            severity="info",
        )
    
    def test_value_ranges(self, entities: list[dict]) -> DiagnosticResult:
        """Test that values are within valid ranges."""
        range_errors = []
        
        for i, entity in enumerate(entities):
            # Check price_tier
            if "price_tier" in entity:
                pt = entity["price_tier"]
                constraints = self.ENTITY_SCHEMA["value_constraints"]["price_tier"]
                if not isinstance(pt, int) or pt < constraints["min"] or pt > constraints["max"]:
                    range_errors.append(
                        f"Entity {i}: price_tier={pt} (valid: {constraints['min']}-{constraints['max']})"
                    )
            
            # Check name length
            if "name" in entity:
                name = entity["name"]
                constraints = self.ENTITY_SCHEMA["value_constraints"]["name"]
                if len(name) < constraints["min_length"]:
                    range_errors.append(f"Entity {i}: name too short ('{name}')")
                elif len(name) > constraints["max_length"]:
                    range_errors.append(f"Entity {i}: name too long ({len(name)} chars)")
            
            # Check vibe scores
            if "vibe_dna" in entity and "dimensions" in entity.get("vibe_dna", {}):
                for dim, score in entity["vibe_dna"]["dimensions"].items():
                    if not isinstance(score, (int, float)) or score < 0 or score > 1:
                        range_errors.append(
                            f"Entity {i}: vibe dimension '{dim}'={score} (valid: 0.0-1.0)"
                        )
        
        if range_errors:
            return DiagnosticResult(
                test_name="Value Ranges",
                passed=False,
                message=f"{len(range_errors)} values out of range",
                severity="error",
                details={"errors": range_errors[:10]},
                fix_action="Validate and clamp values to valid ranges in input pipeline",
            )
        
        return DiagnosticResult(
            test_name="Value Ranges",
            passed=True,
            message="All values within valid ranges",
            severity="info",
        )
    
    def test_unicode_handling(self) -> DiagnosticResult:
        """Test handling of unicode characters."""
        test_cases = [
            {"name": "Ristorante 日本", "expected": "japanese"},
            {"name": "Café Müller", "expected": "german"},
            {"name": "Trattoria l'Étoile", "expected": "french"},
            {"name": "Restaurant 🍕🍝", "expected": "emoji"},
            {"name": "مطعم عربي", "expected": "arabic_rtl"},
        ]
        
        passed = 0
        failed = []
        
        for case in test_cases:
            try:
                # Test JSON serialization
                json.dumps(case)
                # Test string operations
                _ = case["name"].lower()
                _ = len(case["name"])
                passed += 1
            except Exception as e:
                failed.append(f"{case['expected']}: {str(e)}")
        
        if failed:
            return DiagnosticResult(
                test_name="Unicode Handling",
                passed=False,
                message=f"{len(failed)}/{len(test_cases)} unicode tests failed",
                severity="warning",
                details={"failures": failed},
                fix_action="Ensure UTF-8 encoding throughout the pipeline",
            )
        
        return DiagnosticResult(
            test_name="Unicode Handling",
            passed=True,
            message=f"All {len(test_cases)} unicode variants handled correctly",
            severity="info",
        )
    
    def test_special_characters(self) -> DiagnosticResult:
        """Test handling of special characters that could cause issues."""
        test_cases = [
            "L'Osteria dell'Arco",  # Apostrophes
            "Restaurant \"Il Bello\"",  # Quotes
            "Café & Bar",  # Ampersand
            "Trattoria <Casa>",  # Angle brackets
            "Hotel (Centro)",  # Parentheses
            "Ristorante N°1",  # Special symbols
            "Bar - Wine & Dine",  # Dashes
        ]
        
        issues = []
        
        for name in test_cases:
            try:
                # Test JSON serialization
                json.dumps({"name": name})
                
                # Test common string operations
                _ = name.lower()
                _ = re.sub(r'[^\w\s]', '', name)
            except Exception as e:
                issues.append(f"'{name}': {str(e)}")
        
        if issues:
            return DiagnosticResult(
                test_name="Special Characters",
                passed=False,
                message=f"Issues with {len(issues)} special character patterns",
                severity="warning",
                details={"issues": issues},
                fix_action="Escape or sanitize special characters in input processing",
            )
        
        return DiagnosticResult(
            test_name="Special Characters",
            passed=True,
            message="All special characters handled correctly",
            severity="info",
        )
    
    def test_empty_values(self, entities: list[dict]) -> DiagnosticResult:
        """Test detection of empty/null values in critical fields."""
        empty_issues = []
        
        for i, entity in enumerate(entities):
            # Check for empty strings in required fields
            for field in self.ENTITY_SCHEMA["required_fields"]:
                value = entity.get(field)
                if value == "" or value is None:
                    empty_issues.append(f"Entity {i}: '{field}' is empty/null")
            
            # Check for empty description (should be warning, not error)
            if entity.get("description") == "":
                empty_issues.append(f"Entity {i}: 'description' is empty")
        
        if empty_issues:
            return DiagnosticResult(
                test_name="Empty Values Detection",
                passed=True,  # Detection is working
                message=f"Detected {len(empty_issues)} empty values (detection working)",
                severity="info",
                details={"empty_values": empty_issues[:10]},
            )
        
        return DiagnosticResult(
            test_name="Empty Values Detection",
            passed=True,
            message="No empty values in critical fields",
            severity="info",
        )
    
    def test_json_serialization(self, entities: list[dict]) -> DiagnosticResult:
        """Test that all entities can be serialized to JSON."""
        failures = []
        
        for i, entity in enumerate(entities):
            try:
                json_str = json.dumps(entity, ensure_ascii=False)
                # Verify roundtrip
                parsed = json.loads(json_str)
                if parsed.get("name") != entity.get("name"):
                    failures.append(f"Entity {i}: JSON roundtrip mismatch")
            except Exception as e:
                failures.append(f"Entity {i}: {str(e)}")
        
        if failures:
            return DiagnosticResult(
                test_name="JSON Serialization",
                passed=False,
                message=f"{len(failures)} entities failed serialization",
                severity="critical",
                details={"failures": failures[:10]},
                fix_action="Fix non-serializable data types in entity structure",
            )
        
        return DiagnosticResult(
            test_name="JSON Serialization",
            passed=True,
            message=f"All {len(entities)} entities serialize correctly",
            severity="info",
        )
    
    def test_large_payload(self) -> DiagnosticResult:
        """Test handling of large payloads."""
        import time
        
        # Generate large batch
        start = time.time()
        entities = self.entity_factory.generate_batch("restaurant", count=500)
        generation_time = time.time() - start
        
        # Serialize
        start = time.time()
        json_str = json.dumps(entities, ensure_ascii=False)
        serialize_time = time.time() - start
        
        # Deserialize
        start = time.time()
        _ = json.loads(json_str)
        deserialize_time = time.time() - start
        
        size_mb = len(json_str.encode()) / (1024 * 1024)
        
        # Performance thresholds
        if generation_time > 10 or serialize_time > 5:
            return DiagnosticResult(
                test_name="Large Payload",
                passed=False,
                message=f"Performance issue with 500 entities",
                severity="warning",
                details={
                    "entities": 500,
                    "size_mb": round(size_mb, 2),
                    "generation_time_s": round(generation_time, 2),
                    "serialize_time_s": round(serialize_time, 2),
                    "deserialize_time_s": round(deserialize_time, 2),
                },
                fix_action="Optimize data generation or use streaming for large batches",
            )
        
        return DiagnosticResult(
            test_name="Large Payload",
            passed=True,
            message=f"500 entities ({size_mb:.2f}MB) processed in {generation_time + serialize_time:.2f}s",
            severity="info",
            details={
                "entities": 500,
                "size_mb": round(size_mb, 2),
                "total_time_s": round(generation_time + serialize_time + deserialize_time, 2),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pytest tests (for pytest runner)
# ═══════════════════════════════════════════════════════════════════════════

if PYTEST_AVAILABLE:
    @pytest.fixture
    def diagnostics():
        return Layer1Diagnostics()

    @pytest.fixture
    def clean_entities(diagnostics):
        return diagnostics.entity_factory.generate_batch("restaurant", count=10)

    @pytest.mark.layer1
    def test_required_fields_present(diagnostics, clean_entities):
        """Test required fields are present."""
        result = diagnostics.test_required_fields(clean_entities)
        assert result.passed, result.message

    @pytest.mark.layer1
    def test_field_types_correct(diagnostics, clean_entities):
        """Test field types are correct."""
        result = diagnostics.test_field_types(clean_entities)
        assert result.passed, result.message

    @pytest.mark.layer1
    def test_unicode_handling(diagnostics):
        """Test unicode handling."""
        result = diagnostics.test_unicode_handling()
        assert result.passed, result.message

    @pytest.mark.layer1
    def test_json_serialization(diagnostics, clean_entities):
        """Test JSON serialization."""
        result = diagnostics.test_json_serialization(clean_entities)
        assert result.passed, result.message


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

async def run() -> Layer1Report:
    """Run all Layer 1 diagnostics."""
    diagnostics = Layer1Diagnostics()
    return diagnostics.run_all()


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(run())
    report.print_summary()
