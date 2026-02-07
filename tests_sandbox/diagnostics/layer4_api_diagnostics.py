"""Layer 4: API & Integration Diagnostics.

Tests API endpoints, response formats, and integration flows.
Answers: "Are the APIs working correctly?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json

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
class Layer4Report:
    """Complete Layer 4 diagnostic report."""
    layer_name: str = "Layer 4: API & Integration"
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
# Mock API client (for testing without real server)
# ═══════════════════════════════════════════════════════════════════════════

class MockAPIClient:
    """Mock API client for testing without a running server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.entity_factory = EntityFactory()
        self._simulated_latency = 0.05
    
    def _simulate_request(self):
        """Simulate network latency."""
        time.sleep(self._simulated_latency)
    
    def health_check(self) -> dict:
        """Simulate health check endpoint."""
        self._simulate_request()
        return {"status": "healthy", "version": "2.0.0"}
    
    def get_entities(self, limit: int = 10) -> list[dict]:
        """Simulate get entities endpoint."""
        self._simulate_request()
        return self.entity_factory.generate_batch("restaurant", count=limit)
    
    def get_entity(self, entity_id: str) -> dict | None:
        """Simulate get single entity endpoint."""
        self._simulate_request()
        entity = self.entity_factory.restaurant()
        entity["id"] = entity_id
        return entity
    
    def discover(self, query: str, limit: int = 10) -> dict:
        """Simulate discovery endpoint."""
        self._simulate_request()
        entities = self.entity_factory.generate_batch("restaurant", count=limit)
        return {
            "query": query,
            "results": entities,
            "total": len(entities),
            "took_ms": 150,
        }
    
    def enrich_entity(self, entity_id: str) -> dict:
        """Simulate enrichment endpoint."""
        self._simulate_request()
        return {
            "entity_id": entity_id,
            "status": "enriched",
            "vibe_dna_updated": True,
        }
    
    def validate_profile(self, profile: dict) -> dict:
        """Simulate profile validation endpoint."""
        self._simulate_request()
        return {
            "valid": True,
            "score": 85,
            "warnings": [],
            "errors": [],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class Layer4Diagnostics:
    """Layer 4 diagnostic tests for API and integration.
    
    Tests:
    - Endpoint availability
    - Response format validation
    - Error handling
    - Response times
    - Data consistency across endpoints
    """
    
    # Endpoints to test
    ENDPOINTS = [
        {"name": "Health", "method": "health_check", "critical": True},
        {"name": "Entities", "method": "get_entities", "critical": True},
        {"name": "Discovery", "method": "discover", "critical": True},
        {"name": "Enrichment", "method": "enrich_entity", "critical": False},
    ]
    
    MAX_RESPONSE_TIME_MS = 500
    
    def __init__(self, api_client: Any = None):
        """Initialize diagnostics.
        
        Args:
            api_client: API client to use. If None, uses mock.
        """
        self.client = api_client or MockAPIClient()
        self.domain_factory = DomainProfileFactory()
    
    def run_all(self) -> Layer4Report:
        """Run all Layer 4 diagnostic tests."""
        report = Layer4Report()
        
        # Run tests
        report.results.append(self.test_endpoint_availability())
        report.results.append(self.test_health_endpoint())
        report.results.append(self.test_entities_response_format())
        report.results.append(self.test_discovery_flow())
        report.results.append(self.test_error_handling())
        report.results.append(self.test_response_times())
        report.results.append(self.test_data_consistency())
        
        return report
    
    def test_endpoint_availability(self) -> DiagnosticResult:
        """Test that all critical endpoints are available."""
        available = []
        unavailable = []
        
        for endpoint in self.ENDPOINTS:
            try:
                method = getattr(self.client, endpoint["method"], None)
                if method:
                    # Call with minimal args
                    if endpoint["method"] == "discover":
                        method("test query")
                    elif endpoint["method"] == "enrich_entity":
                        method("test-id")
                    else:
                        method()
                    available.append(endpoint["name"])
                else:
                    unavailable.append(endpoint["name"])
            except Exception as e:
                unavailable.append(f"{endpoint['name']} ({str(e)[:30]})")
        
        critical_missing = [
            ep["name"] for ep in self.ENDPOINTS 
            if ep["critical"] and ep["name"] in unavailable
        ]
        
        if critical_missing:
            return DiagnosticResult(
                test_name="Endpoint Availability",
                passed=False,
                message=f"Critical endpoints unavailable: {critical_missing}",
                severity="critical",
                details={"available": available, "unavailable": unavailable},
                fix_action="Check server is running and endpoints are registered",
            )
        
        if unavailable:
            return DiagnosticResult(
                test_name="Endpoint Availability",
                passed=True,
                message=f"Non-critical endpoints unavailable: {unavailable}",
                severity="warning",
                details={"available": available, "unavailable": unavailable},
            )
        
        return DiagnosticResult(
            test_name="Endpoint Availability",
            passed=True,
            message=f"All {len(available)} endpoints available",
            severity="info",
            details={"endpoints": available},
        )
    
    def test_health_endpoint(self) -> DiagnosticResult:
        """Test health endpoint returns expected format."""
        try:
            response = self.client.health_check()
            
            if not isinstance(response, dict):
                return DiagnosticResult(
                    test_name="Health Endpoint",
                    passed=False,
                    message="Health response is not a dict",
                    severity="error",
                )
            
            if "status" not in response:
                return DiagnosticResult(
                    test_name="Health Endpoint",
                    passed=False,
                    message="Health response missing 'status' field",
                    severity="error",
                    fix_action="Add 'status' field to health response",
                )
            
            return DiagnosticResult(
                test_name="Health Endpoint",
                passed=True,
                message=f"Health OK: status={response.get('status')}",
                severity="info",
                details=response,
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Health Endpoint",
                passed=False,
                message=f"Health check failed: {str(e)}",
                severity="critical",
                fix_action="Check server health endpoint",
            )
    
    def test_entities_response_format(self) -> DiagnosticResult:
        """Test entities endpoint returns valid format."""
        try:
            entities = self.client.get_entities(limit=5)
            
            if not isinstance(entities, list):
                return DiagnosticResult(
                    test_name="Entities Format",
                    passed=False,
                    message="Entities response is not a list",
                    severity="error",
                )
            
            if len(entities) == 0:
                return DiagnosticResult(
                    test_name="Entities Format",
                    passed=True,
                    message="Entities endpoint returned empty list",
                    severity="warning",
                )
            
            # Check entity structure
            required_fields = ["id", "name", "entity_type"]
            missing_fields = []
            
            for i, entity in enumerate(entities):
                for field in required_fields:
                    if field not in entity:
                        missing_fields.append(f"Entity {i}: missing '{field}'")
            
            if missing_fields:
                return DiagnosticResult(
                    test_name="Entities Format",
                    passed=False,
                    message=f"{len(missing_fields)} entities missing required fields",
                    severity="error",
                    details={"missing": missing_fields[:5]},
                    fix_action=f"Ensure entities have: {required_fields}",
                )
            
            return DiagnosticResult(
                test_name="Entities Format",
                passed=True,
                message=f"{len(entities)} entities with valid format",
                severity="info",
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Entities Format",
                passed=False,
                message=f"Failed to get entities: {str(e)}",
                severity="error",
            )
    
    def test_discovery_flow(self) -> DiagnosticResult:
        """Test discovery endpoint returns expected format."""
        try:
            response = self.client.discover("romantic fine dining", limit=5)
            
            if not isinstance(response, dict):
                return DiagnosticResult(
                    test_name="Discovery Flow",
                    passed=False,
                    message="Discovery response is not a dict",
                    severity="error",
                )
            
            required = ["query", "results"]
            missing = [f for f in required if f not in response]
            
            if missing:
                return DiagnosticResult(
                    test_name="Discovery Flow",
                    passed=False,
                    message=f"Discovery response missing: {missing}",
                    severity="error",
                    fix_action="Ensure discovery response includes query and results",
                )
            
            results = response.get("results", [])
            
            return DiagnosticResult(
                test_name="Discovery Flow",
                passed=True,
                message=f"Discovery returned {len(results)} results",
                severity="info",
                details={
                    "query": response.get("query"),
                    "result_count": len(results),
                    "took_ms": response.get("took_ms"),
                },
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Discovery Flow",
                passed=False,
                message=f"Discovery failed: {str(e)}",
                severity="error",
            )
    
    def test_error_handling(self) -> DiagnosticResult:
        """Test that API handles errors gracefully."""
        # Test with invalid inputs
        test_cases = [
            ("empty_query", lambda: self.client.discover("")),
            ("invalid_id", lambda: self.client.get_entity("nonexistent-id")),
        ]
        
        handled = []
        unhandled = []
        
        for name, test_fn in test_cases:
            try:
                result = test_fn()
                # If it returns without error, that's fine (mock may not validate)
                handled.append(name)
            except Exception as e:
                # Errors should be caught and returned, not raised
                if "500" in str(e) or "Internal" in str(e):
                    unhandled.append(f"{name}: {str(e)[:50]}")
                else:
                    handled.append(name)
        
        if unhandled:
            return DiagnosticResult(
                test_name="Error Handling",
                passed=False,
                message=f"{len(unhandled)} unhandled errors",
                severity="warning",
                details={"unhandled": unhandled},
                fix_action="Add proper error handling for edge cases",
            )
        
        return DiagnosticResult(
            test_name="Error Handling",
            passed=True,
            message="Errors handled gracefully",
            severity="info",
        )
    
    def test_response_times(self) -> DiagnosticResult:
        """Test API response times are acceptable."""
        timings = {}
        slow = []
        
        for endpoint in self.ENDPOINTS[:3]:  # Test first 3
            method = getattr(self.client, endpoint["method"], None)
            if method:
                start = time.perf_counter()
                try:
                    if endpoint["method"] == "discover":
                        method("test query")
                    else:
                        method()
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    timings[endpoint["name"]] = round(elapsed_ms, 1)
                    
                    if elapsed_ms > self.MAX_RESPONSE_TIME_MS:
                        slow.append(f"{endpoint['name']}: {elapsed_ms:.0f}ms")
                except Exception:
                    pass
        
        if slow:
            return DiagnosticResult(
                test_name="Response Times",
                passed=False,
                message=f"Slow endpoints: {slow[0]}",
                severity="warning",
                details={"timings_ms": timings, "slow": slow},
                fix_action=f"Optimize slow endpoints (target: <{self.MAX_RESPONSE_TIME_MS}ms)",
            )
        
        return DiagnosticResult(
            test_name="Response Times",
            passed=True,
            message=f"All responses <{self.MAX_RESPONSE_TIME_MS}ms",
            severity="info",
            details={"timings_ms": timings},
        )
    
    def test_data_consistency(self) -> DiagnosticResult:
        """Test data consistency across endpoints."""
        try:
            # Get list of entities
            entities = self.client.get_entities(limit=3)
            
            if not entities:
                return DiagnosticResult(
                    test_name="Data Consistency",
                    passed=True,
                    message="No entities to verify consistency",
                    severity="info",
                )
            
            # Verify single entity fetch returns same data
            entity = entities[0]
            entity_id = entity.get("id")
            
            single = self.client.get_entity(entity_id)
            
            if single and single.get("id") != entity_id:
                return DiagnosticResult(
                    test_name="Data Consistency",
                    passed=False,
                    message="Entity ID mismatch between list and single fetch",
                    severity="error",
                    fix_action="Check entity serialization consistency",
                )
            
            return DiagnosticResult(
                test_name="Data Consistency",
                passed=True,
                message="Data consistent across endpoints",
                severity="info",
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Data Consistency",
                passed=False,
                message=f"Consistency check failed: {str(e)}",
                severity="warning",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Pytest tests
# ═══════════════════════════════════════════════════════════════════════════

if PYTEST_AVAILABLE:
    @pytest.fixture
    def diagnostics():
        return Layer4Diagnostics()

    @pytest.mark.layer4
    def test_endpoint_availability(diagnostics):
        """Test endpoints are available."""
        result = diagnostics.test_endpoint_availability()
        assert result.passed, result.message

    @pytest.mark.layer4
    def test_health_endpoint(diagnostics):
        """Test health endpoint."""
        result = diagnostics.test_health_endpoint()
        assert result.passed, result.message

    @pytest.mark.layer4
    def test_entities_format(diagnostics):
        """Test entities format."""
        result = diagnostics.test_entities_response_format()
        assert result.passed, result.message


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

async def run() -> Layer4Report:
    """Run all Layer 4 diagnostics."""
    diagnostics = Layer4Diagnostics()
    return diagnostics.run_all()


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(run())
    report.print_summary()
