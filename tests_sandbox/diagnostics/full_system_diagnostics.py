"""Full System Diagnostics - End-to-end system testing.

Runs complete diagnostic cycle across all layers and generates
comprehensive report with actionable insights.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .layer1_input_diagnostics import Layer1Diagnostics, Layer1Report
from .layer2_profile_diagnostics import Layer2Diagnostics, Layer2Report
from .layer3_ml_diagnostics import Layer3Diagnostics, Layer3Report
from .layer4_api_diagnostics import Layer4Diagnostics, Layer4Report

from data_generators import ScenarioBuilder


@dataclass
class FullSystemReport:
    """Complete system diagnostic report across all layers."""
    
    layer1: Layer1Report | None = None
    layer2: Layer2Report | None = None
    layer3: Layer3Report | None = None
    layer4: Layer4Report | None = None
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0
    
    @property
    def total_tests(self) -> int:
        return sum(
            len(getattr(layer, "results", []))
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
            if layer
        )
    
    @property
    def passed_tests(self) -> int:
        return sum(
            sum(1 for r in getattr(layer, "results", []) if r.passed)
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
            if layer
        )
    
    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests
    
    @property
    def is_healthy(self) -> bool:
        """Check if all layers are healthy (no critical failures)."""
        return all(
            getattr(layer, "is_healthy", True)
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
            if layer
        )
    
    def get_top_issues(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get top issues sorted by severity."""
        issues = []
        
        severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer:
                for result in layer.results:
                    if not result.passed:
                        issues.append({
                            "layer": layer.layer_name,
                            "test": result.test_name,
                            "message": result.message,
                            "severity": result.severity,
                            "fix_action": result.fix_action,
                            "severity_order": severity_order.get(result.severity, 3),
                        })
        
        # Sort by severity
        issues.sort(key=lambda x: x["severity_order"])
        
        return issues[:limit]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "pass_rate": round(self.passed_tests / max(self.total_tests, 1) * 100, 1),
                "is_healthy": self.is_healthy,
            },
            "top_issues": self.get_top_issues(),
            "layers": {
                "layer1": self.layer1.to_dict() if self.layer1 else None,
                "layer2": self.layer2.to_dict() if self.layer2 else None,
                "layer3": self.layer3.to_dict() if self.layer3 else None,
                "layer4": self.layer4.to_dict() if self.layer4 else None,
            },
        }
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "═" * 60)
        print("🔬 ARBOR FULL SYSTEM DIAGNOSTICS")
        print("═" * 60)
        
        # Print each layer
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer:
                layer.print_summary()
        
        # Print overall summary
        print("\n" + "═" * 60)
        print("📋 OVERALL SUMMARY")
        print("═" * 60)
        
        pass_rate = self.passed_tests / max(self.total_tests, 1) * 100
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests} ({pass_rate:.0f}%)")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"Duration: {self.duration_seconds:.1f}s")
        
        # Print top issues
        issues = self.get_top_issues(3)
        if issues:
            print("\n🎯 TOP ISSUES TO FIX:")
            for i, issue in enumerate(issues, 1):
                severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡"}.get(issue["severity"], "⚪")
                print(f"{i}. {severity_icon} [{issue['layer'].split(':')[1].strip()}] {issue['message']}")
                if issue["fix_action"]:
                    print(f"   → {issue['fix_action']}")


class FullSystemDiagnostics:
    """Run complete diagnostic cycle across all layers."""
    
    def __init__(self):
        self.layer1 = Layer1Diagnostics()
        self.layer2 = Layer2Diagnostics()
        self.layer3 = Layer3Diagnostics()
        self.layer4 = Layer4Diagnostics()
    
    async def run_all(
        self,
        include_scenarios: bool = False,
        save_report: bool = True,
        report_dir: Path | None = None,
    ) -> FullSystemReport:
        """Run complete diagnostic cycle.
        
        Args:
            include_scenarios: If True, also run scenario-based tests
            save_report: If True, save report to file
            report_dir: Directory for reports (default: tests_sandbox/reports)
        """
        import time
        start = time.perf_counter()
        
        report = FullSystemReport()
        
        print("🔬 Running Layer 1: Input Validation...")
        report.layer1 = self.layer1.run_all()
        
        print("🔬 Running Layer 2: Domain Profile...")
        report.layer2 = self.layer2.run_all()
        
        print("🔬 Running Layer 3: ML Pipeline...")
        report.layer3 = self.layer3.run_all()
        
        print("🔬 Running Layer 4: API & Integration...")
        report.layer4 = self.layer4.run_all()
        
        report.duration_seconds = time.perf_counter() - start
        
        # Run scenario tests if requested
        if include_scenarios:
            print("\n🔬 Running Scenario Tests...")
            await self._run_scenarios()
        
        # Save report
        if save_report:
            report_dir = report_dir or Path(__file__).parent.parent / "reports"
            report_dir.mkdir(exist_ok=True)
            
            # Save JSON report
            json_path = report_dir / "diagnostic_report.json"
            with open(json_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            
            # Save HTML report
            html_path = report_dir / "diagnostic_report.html"
            self._generate_html_report(report, html_path)
            
            print(f"\n📁 Reports saved to: {report_dir}")
        
        return report
    
    async def _run_scenarios(self) -> None:
        """Run scenario-based tests."""
        builder = ScenarioBuilder(seed=42)
        scenarios = [
            builder.fine_dining_scenario(),
            builder.hotel_luxury_scenario(),
            builder.scenario_with_issues(),
            builder.edge_case_scenario(),
        ]
        
        for scenario in scenarios:
            print(f"  - {scenario.scenario_name}: {scenario.entity_count} entities")
    
    def _generate_html_report(self, report: FullSystemReport, path: Path) -> None:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ARBOR Diagnostic Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center; flex: 1; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ color: #666; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .layer {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
        .layer h3 {{ margin-top: 0; }}
        .test {{ padding: 8px; margin: 5px 0; background: white; border-radius: 4px; }}
        .test.pass {{ border-left: 3px solid #28a745; }}
        .test.fail {{ border-left: 3px solid #dc3545; }}
        .test.warn {{ border-left: 3px solid #ffc107; }}
        .fix {{ color: #666; font-style: italic; margin-left: 20px; }}
        .issues {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .issues h3 {{ color: #856404; margin-top: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 ARBOR Diagnostic Report</h1>
        <p>Generated: {report.timestamp}</p>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{report.total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value passed">{report.passed_tests}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{report.failed_tests}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.duration_seconds:.1f}s</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>
        
        <div class="issues">
            <h3>🎯 Top Issues to Fix</h3>
            {''.join(f'<p><strong>[{i["layer"].split(":")[1].strip()}]</strong> {i["message"]}<br><span class="fix">→ {i["fix_action"] or "No fix suggested"}</span></p>' for i in report.get_top_issues(5)) or '<p>No issues found!</p>'}
        </div>
        
        {"".join(self._layer_html(layer) for layer in [report.layer1, report.layer2, report.layer3, report.layer4] if layer)}
    </div>
</body>
</html>"""
        
        with open(path, "w") as f:
            f.write(html)
    
    def _layer_html(self, layer) -> str:
        """Generate HTML for a single layer."""
        tests_html = ""
        for result in layer.results:
            css_class = "pass" if result.passed else ("warn" if result.severity == "warning" else "fail")
            icon = "✅" if result.passed else ("⚠️" if result.severity == "warning" else "❌")
            fix = f'<div class="fix">FIX: {result.fix_action}</div>' if result.fix_action and not result.passed else ""
            tests_html += f'<div class="test {css_class}">{icon} <strong>{result.test_name}</strong>: {result.message}{fix}</div>'
        
        return f"""
        <div class="layer">
            <h3>{layer.layer_name}</h3>
            {tests_html}
        </div>
        """


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

async def run() -> FullSystemReport:
    """Run full system diagnostics."""
    diagnostics = FullSystemDiagnostics()
    return await diagnostics.run_all()


if __name__ == "__main__":
    report = asyncio.run(run())
    report.print_summary()
