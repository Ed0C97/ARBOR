"""Performance benchmarks for ARBOR test sandbox.

Measures and tracks performance metrics across the system.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import tracemalloc

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generators import EntityFactory, DomainProfileFactory, ScenarioBuilder


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    samples: int
    total_time_ms: float
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_per_sec: float
    memory_peak_mb: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "samples": self.samples,
            "total_time_ms": round(self.total_time_ms, 2),
            "mean_time_ms": round(self.mean_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "throughput_per_sec": round(self.throughput_per_sec, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 2) if self.memory_peak_mb else None,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_info: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        print("\n📊 PERFORMANCE BENCHMARKS")
        print("═" * 70)
        
        print(f"{'Benchmark':<30} {'Mean(ms)':<12} {'Throughput':<15} {'Memory':<10}")
        print("─" * 70)
        
        for b in self.benchmarks:
            throughput = f"{b.throughput_per_sec:.0f}/s"
            memory = f"{b.memory_peak_mb:.1f}MB" if b.memory_peak_mb else "N/A"
            print(f"{b.name:<30} {b.mean_time_ms:<12.3f} {throughput:<15} {memory:<10}")
        
        print("─" * 70)


class PerformanceBenchmarks:
    """Performance benchmark suite."""
    
    def __init__(self, warmup_runs: int = 3):
        self.warmup_runs = warmup_runs
        self.entity_factory = EntityFactory()
        self.domain_factory = DomainProfileFactory()
    
    def benchmark(
        self,
        name: str,
        fn: Callable,
        samples: int = 100,
        measure_memory: bool = True,
    ) -> BenchmarkResult:
        """Run a benchmark.
        
        Args:
            name: Benchmark name
            fn: Function to benchmark
            samples: Number of iterations
            measure_memory: Track peak memory
        """
        # Warmup
        for _ in range(self.warmup_runs):
            fn()
        
        # Force garbage collection
        gc.collect()
        
        # Start memory tracking
        peak_memory = None
        if measure_memory:
            tracemalloc.start()
        
        # Run benchmark
        times_ms = []
        start_total = time.perf_counter()
        
        for _ in range(samples):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed)
        
        total_ms = (time.perf_counter() - start_total) * 1000
        
        # Get memory stats
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024 * 1024)  # MB
            tracemalloc.stop()
        
        # Calculate stats
        mean = statistics.mean(times_ms)
        std_dev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
        throughput = samples / (total_ms / 1000) if total_ms > 0 else 0
        
        return BenchmarkResult(
            name=name,
            samples=samples,
            total_time_ms=total_ms,
            mean_time_ms=mean,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            std_dev_ms=std_dev,
            throughput_per_sec=throughput,
            memory_peak_mb=peak_memory,
        )
    
    def run_all(self, samples: int = 100) -> BenchmarkReport:
        """Run all benchmarks."""
        import platform
        
        report = BenchmarkReport()
        
        # System info
        report.system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        
        print("📊 Running Performance Benchmarks...")
        
        # Entity generation
        print("  • Entity (single)...", end=" ", flush=True)
        result = self.benchmark(
            "Entity Generation (single)",
            lambda: self.entity_factory.restaurant(),
            samples=samples,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # Entity with vibe DNA
        print("  • Entity (with vibe)...", end=" ", flush=True)
        result = self.benchmark(
            "Entity with Vibe DNA",
            lambda: self.entity_factory.restaurant(profile="fine_dining"),
            samples=samples,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # Domain profile generation
        print("  • Domain Profile...", end=" ", flush=True)
        result = self.benchmark(
            "Domain Profile",
            lambda: self.domain_factory.generate(),
            samples=samples // 10 or 1,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # Batch 10
        print("  • Batch (10 entities)...", end=" ", flush=True)
        result = self.benchmark(
            "Batch Generation (10)",
            lambda: self.entity_factory.generate_batch("restaurant", count=10),
            samples=samples // 2 or 1,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # Batch 100
        print("  • Batch (100 entities)...", end=" ", flush=True)
        result = self.benchmark(
            "Batch Generation (100)",
            lambda: self.entity_factory.generate_batch("restaurant", count=100),
            samples=samples // 5 or 1,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # JSON serialization
        print("  • JSON Serialization (100 entities)...", end=" ", flush=True)
        entities = self.entity_factory.generate_batch("restaurant", count=100)
        result = self.benchmark(
            "JSON Serialization (100)",
            lambda: json.dumps(entities),
            samples=samples,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        # Scenario building
        print("  • Scenario Build...", end=" ", flush=True)
        builder = ScenarioBuilder()
        result = self.benchmark(
            "Scenario Build (50 entities)",
            lambda: builder.fine_dining_scenario(entity_count=50),
            samples=samples // 10 or 1,
        )
        report.benchmarks.append(result)
        print(f"✅ {result.mean_time_ms:.3f}ms")
        
        return report


async def run(samples: int = 100, save_report: bool = True) -> BenchmarkReport:
    """Run all benchmarks.
    
    Args:
        samples: Number of samples per benchmark
        save_report: Save report to file
    """
    benchmarks = PerformanceBenchmarks()
    report = benchmarks.run_all(samples=samples)
    
    if save_report:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / "benchmark_results.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\n📁 Report saved to: {report_path}")
    
    report.print_summary()
    
    return report


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
