"""Report Generator - Detailed test reports.

Generates comprehensive JSON and HTML reports for each test iteration.
Reports are saved before database destruction to preserve evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    message: str
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationReport:
    """Complete report for a single test iteration."""
    iteration_id: int
    seed: int
    vertical: str
    schema_name: str
    
    # Schema info
    table_count: int = 0
    total_columns: int = 0
    total_rows: int = 0
    fk_count: int = 0
    
    # Timing
    schema_generation_seconds: float = 0.0
    data_population_seconds: float = 0.0
    test_execution_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    total_seconds: float = 0.0
    
    # Memory
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_delta_mb: float = 0.0
    
    # Test results
    tests: list[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    
    # Schema details (for debugging)
    schema_details: dict[str, Any] = field(default_factory=dict)
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def success(self) -> bool:
        """True if all tests passed."""
        return self.failed == 0 and not self.errors
    
    @property
    def pass_rate(self) -> float:
        """Test pass rate as percentage."""
        total = self.passed + self.failed
        if total == 0:
            return 100.0
        return round(self.passed / total * 100, 1)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration_id": self.iteration_id,
            "seed": self.seed,
            "vertical": self.vertical,
            "schema_name": self.schema_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "summary": {
                "tables": self.table_count,
                "columns": self.total_columns,
                "rows": self.total_rows,
                "fk_relationships": self.fk_count,
            },
            "timing": {
                "schema_generation_s": self.schema_generation_seconds,
                "data_population_s": self.data_population_seconds,
                "test_execution_s": self.test_execution_seconds,
                "cleanup_s": self.cleanup_seconds,
                "total_s": self.total_seconds,
            },
            "memory": {
                "before_mb": self.memory_before_mb,
                "after_mb": self.memory_after_mb,
                "delta_mb": self.memory_delta_mb,
            },
            "tests": {
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "details": [
                    {
                        "name": t.test_name,
                        "passed": t.passed,
                        "message": t.message,
                        "duration_s": t.duration_seconds,
                    }
                    for t in self.tests
                ],
            },
            "schema_details": self.schema_details,
            "errors": self.errors,
        }


@dataclass
class AggregateReport:
    """Aggregate report across all iterations."""
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    
    # By vertical
    by_vertical: dict[str, dict] = field(default_factory=dict)
    
    # Timing stats
    avg_iteration_seconds: float = 0.0
    min_iteration_seconds: float = 0.0
    max_iteration_seconds: float = 0.0
    total_seconds: float = 0.0
    
    # Memory stats
    peak_memory_mb: float = 0.0
    avg_memory_delta_mb: float = 0.0
    
    # Unique schemas tested
    unique_schemas: int = 0
    total_tables_tested: int = 0
    total_rows_tested: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def success_rate(self) -> float:
        if self.total_iterations == 0:
            return 100.0
        return round(self.successful_iterations / self.total_iterations * 100, 1)
    
    @property
    def test_pass_rate(self) -> float:
        if self.total_tests == 0:
            return 100.0
        return round(self.total_passed / self.total_tests * 100, 1)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "iterations": self.total_iterations,
                "successful": self.successful_iterations,
                "failed": self.failed_iterations,
                "success_rate": self.success_rate,
            },
            "tests": {
                "total": self.total_tests,
                "passed": self.total_passed,
                "failed": self.total_failed,
                "pass_rate": self.test_pass_rate,
            },
            "coverage": {
                "unique_schemas": self.unique_schemas,
                "total_tables": self.total_tables_tested,
                "total_rows": self.total_rows_tested,
            },
            "timing": {
                "total_seconds": round(self.total_seconds, 2),
                "avg_per_iteration": round(self.avg_iteration_seconds, 3),
                "min_iteration": round(self.min_iteration_seconds, 3),
                "max_iteration": round(self.max_iteration_seconds, 3),
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "avg_delta_mb": round(self.avg_memory_delta_mb, 2),
            },
            "by_vertical": self.by_vertical,
        }


class ReportGenerator:
    """Generates and saves test reports."""
    
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._iterations: list[IterationReport] = []
    
    def add_iteration(self, report: IterationReport) -> None:
        """Add an iteration report."""
        self._iterations.append(report)
        
        # Save individual iteration report
        self._save_iteration_report(report)
    
    def _save_iteration_report(self, report: IterationReport) -> Path:
        """Save individual iteration report to JSON."""
        filename = f"iteration_{report.iteration_id:04d}_{report.vertical}.json"
        path = self.output_dir / "iterations" / filename
        path.parent.mkdir(exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return path
    
    def generate_aggregate_report(self) -> AggregateReport:
        """Generate aggregate report from all iterations."""
        if not self._iterations:
            return AggregateReport()
        
        report = AggregateReport()
        report.total_iterations = len(self._iterations)
        report.unique_schemas = len(self._iterations)
        
        timings = []
        memory_deltas = []
        
        for it in self._iterations:
            if it.success:
                report.successful_iterations += 1
            else:
                report.failed_iterations += 1
            
            report.total_tests += it.passed + it.failed
            report.total_passed += it.passed
            report.total_failed += it.failed
            
            report.total_tables_tested += it.table_count
            report.total_rows_tested += it.total_rows
            
            timings.append(it.total_seconds)
            memory_deltas.append(it.memory_delta_mb)
            
            if it.memory_after_mb > report.peak_memory_mb:
                report.peak_memory_mb = it.memory_after_mb
            
            # By vertical
            if it.vertical not in report.by_vertical:
                report.by_vertical[it.vertical] = {
                    "iterations": 0,
                    "passed": 0,
                    "failed": 0,
                }
            
            report.by_vertical[it.vertical]["iterations"] += 1
            if it.success:
                report.by_vertical[it.vertical]["passed"] += 1
            else:
                report.by_vertical[it.vertical]["failed"] += 1
        
        # Calculate stats
        report.total_seconds = sum(timings)
        report.avg_iteration_seconds = sum(timings) / len(timings) if timings else 0
        report.min_iteration_seconds = min(timings) if timings else 0
        report.max_iteration_seconds = max(timings) if timings else 0
        report.avg_memory_delta_mb = sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        
        return report
    
    def save_aggregate_report(self) -> tuple[Path, Path]:
        """Save aggregate report to JSON and HTML."""
        aggregate = self.generate_aggregate_report()
        
        # Save JSON
        json_path = self.output_dir / "aggregate_report.json"
        with open(json_path, "w") as f:
            json.dump(aggregate.to_dict(), f, indent=2)
        
        # Save HTML
        html_path = self.output_dir / "aggregate_report.html"
        self._generate_html_report(aggregate, html_path)
        
        return json_path, html_path
    
    def _generate_html_report(self, report: AggregateReport, path: Path) -> None:
        """Generate HTML aggregate report."""
        
        # Generate vertical rows
        vertical_rows = ""
        for vertical, stats in report.by_vertical.items():
            status = "✅" if stats["failed"] == 0 else "⚠️"
            vertical_rows += f"""
            <tr>
                <td>{status} {vertical}</td>
                <td>{stats['iterations']}</td>
                <td class="passed">{stats['passed']}</td>
                <td class="failed">{stats['failed']}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ARBOR Agnostic Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00ff88; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ background: #16213e; padding: 20px; border-radius: 12px; text-align: center; flex: 1; min-width: 150px; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: #00ff88; }}
        .stat-label {{ color: #888; margin-top: 5px; }}
        .passed {{ color: #00ff88; }}
        .failed {{ color: #ff4444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; color: #00ff88; }}
        .progress {{ background: #333; border-radius: 10px; height: 20px; overflow: hidden; }}
        .progress-bar {{ background: linear-gradient(90deg, #00ff88, #00cc66); height: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌳 ARBOR Schema-Agnostic Test Report</h1>
        <p style="color: #888;">Generated: {report.timestamp}</p>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{report.total_iterations}</div>
                <div class="stat-label">Iterations</div>
            </div>
            <div class="stat">
                <div class="stat-value passed">{report.success_rate}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.total_tables_tested}</div>
                <div class="stat-label">Tables Tested</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.total_rows_tested:,}</div>
                <div class="stat-label">Rows Tested</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.total_seconds:.1f}s</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>
        
        <h2>Test Results</h2>
        <div class="progress">
            <div class="progress-bar" style="width: {report.test_pass_rate}%"></div>
        </div>
        <p><span class="passed">{report.total_passed} passed</span> / <span class="failed">{report.total_failed} failed</span> ({report.test_pass_rate}%)</p>
        
        <h2>By Vertical</h2>
        <table>
            <tr>
                <th>Vertical</th>
                <th>Iterations</th>
                <th>Passed</th>
                <th>Failed</th>
            </tr>
            {vertical_rows}
        </table>
        
        <h2>Performance</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Avg. iteration time</td>
                <td>{report.avg_iteration_seconds:.3f}s</td>
            </tr>
            <tr>
                <td>Min iteration time</td>
                <td>{report.min_iteration_seconds:.3f}s</td>
            </tr>
            <tr>
                <td>Max iteration time</td>
                <td>{report.max_iteration_seconds:.3f}s</td>
            </tr>
            <tr>
                <td>Peak memory</td>
                <td>{report.peak_memory_mb:.1f} MB</td>
            </tr>
            <tr>
                <td>Avg. memory delta</td>
                <td>{report.avg_memory_delta_mb:.2f} MB</td>
            </tr>
        </table>
    </div>
</body>
</html>"""
        
        with open(path, "w") as f:
            f.write(html)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator(tmpdir)
        
        # Add sample iterations
        for i in range(5):
            report = IterationReport(
                iteration_id=i,
                seed=42 + i,
                vertical="restaurant" if i % 2 == 0 else "hotel",
                schema_name=f"test_schema_{i}",
                table_count=5 + i,
                total_rows=100 * (i + 1),
                total_seconds=0.5 + i * 0.1,
                passed=3,
                failed=1 if i == 2 else 0,
            )
            gen.add_iteration(report)
        
        json_path, html_path = gen.save_aggregate_report()
        
        print(f"JSON: {json_path}")
        print(f"HTML: {html_path}")
        
        agg = gen.generate_aggregate_report()
        print(f"\nSuccess rate: {agg.success_rate}%")
        print(f"Test pass rate: {agg.test_pass_rate}%")
