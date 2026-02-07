#!/usr/bin/env python3
"""Master run script for ARBOR test sandbox diagnostics.

This script orchestrates all diagnostic tests and generates comprehensive reports.

Usage:
    python run_all_diagnostics.py                    # Run all layers
    python run_all_diagnostics.py --layer 1          # Run specific layer
    python run_all_diagnostics.py --quick            # Quick smoke test
    python run_all_diagnostics.py --benchmark        # Include performance benchmarks
    python run_all_diagnostics.py --scenarios        # Include scenario tests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics import (
    Layer1Diagnostics,
    Layer2Diagnostics,
    Layer3Diagnostics,
    Layer4Diagnostics,
    FullSystemDiagnostics,
)
from data_generators import ScenarioBuilder


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     🌳 ARBOR Test Sandbox - Full System Diagnostics 🌳       ║
║                                                               ║
║   Layer 1: Input Validation  │  Layer 3: ML Pipeline        ║
║   Layer 2: Domain Profile    │  Layer 4: API Integration    ║
╚═══════════════════════════════════════════════════════════════╝
""")


def run_single_layer(layer_num: int) -> dict[str, Any]:
    """Run diagnostics for a single layer.
    
    Args:
        layer_num: Layer number (1-4)
        
    Returns:
        Report dictionary
    """
    print(f"\n🔬 Running Layer {layer_num} Diagnostics...")
    
    layer_classes = {
        1: Layer1Diagnostics,
        2: Layer2Diagnostics,
        3: Layer3Diagnostics,
        4: Layer4Diagnostics,
    }
    
    if layer_num not in layer_classes:
        print(f"❌ Invalid layer: {layer_num}. Must be 1-4.")
        sys.exit(1)
    
    diagnostics = layer_classes[layer_num]()
    report = diagnostics.run_all()
    report.print_summary()
    
    return report.to_dict()


async def run_all_layers(
    include_scenarios: bool = False,
    save_report: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run diagnostics for all layers.
    
    Args:
        include_scenarios: Include scenario-based tests
        save_report: Save report to file
        verbose: Print verbose output
        
    Returns:
        Full report dictionary
    """
    print("\n🔬 Running Full System Diagnostics...")
    
    diagnostics = FullSystemDiagnostics()
    
    report = await diagnostics.run_all(
        include_scenarios=include_scenarios,
        save_report=save_report,
    )
    
    report.print_summary()
    
    return report.to_dict()


def run_quick_smoke_test() -> bool:
    """Run quick smoke test to verify basic functionality.
    
    Returns:
        True if all critical tests pass
    """
    print("\n⚡ Running Quick Smoke Test...")
    
    results = []
    
    # Layer 1: Just check JSON serialization
    print("  • Layer 1: Input Validation...", end=" ")
    try:
        layer1 = Layer1Diagnostics()
        entities = layer1.entity_factory.generate_batch("restaurant", count=5)
        json.dumps(entities)
        print("✅")
        results.append(True)
    except Exception as e:
        print(f"❌ {e}")
        results.append(False)
    
    # Layer 2: Just check profile generation
    print("  • Layer 2: Domain Profile...", end=" ")
    try:
        layer2 = Layer2Diagnostics()
        intake = layer2.domain_factory.generate()
        assert intake.domain_name
        print("✅")
        results.append(True)
    except Exception as e:
        print(f"❌ {e}")
        results.append(False)
    
    # Layer 3: Just check scoring works
    print("  • Layer 3: ML Pipeline...", end=" ")
    try:
        layer3 = Layer3Diagnostics()
        entity = layer3.entity_factory.restaurant()
        result = layer3.scoring_fn(entity)
        assert "overall_score" in result
        print("✅")
        results.append(True)
    except Exception as e:
        print(f"❌ {e}")
        results.append(False)
    
    # Layer 4: Just check mock API
    print("  • Layer 4: API Integration...", end=" ")
    try:
        layer4 = Layer4Diagnostics()
        health = layer4.client.health_check()
        assert health.get("status") == "healthy"
        print("✅")
        results.append(True)
    except Exception as e:
        print(f"❌ {e}")
        results.append(False)
    
    passed = all(results)
    print(f"\n{'✅ Smoke Test Passed!' if passed else '❌ Smoke Test Failed!'}")
    return passed


def run_benchmark(samples: int = 100) -> dict[str, Any]:
    """Run performance benchmarks.
    
    Args:
        samples: Number of samples for benchmarking
        
    Returns:
        Benchmark results
    """
    from data_generators import EntityFactory, DomainProfileFactory
    
    print(f"\n📊 Running Performance Benchmarks (n={samples})...")
    
    results = {}
    
    # Entity generation benchmark
    print("  • Entity Generation...", end=" ")
    factory = EntityFactory()
    start = time.perf_counter()
    for _ in range(samples):
        factory.restaurant()
    elapsed = time.perf_counter() - start
    results["entity_generation"] = {
        "samples": samples,
        "total_ms": round(elapsed * 1000, 1),
        "per_entity_ms": round(elapsed * 1000 / samples, 2),
    }
    print(f"✅ ({results['entity_generation']['per_entity_ms']}ms/entity)")
    
    # Domain profile generation benchmark
    print("  • Profile Generation...", end=" ")
    domain_factory = DomainProfileFactory()
    start = time.perf_counter()
    for _ in range(samples // 10 or 1):
        domain_factory.generate()
    elapsed = time.perf_counter() - start
    results["profile_generation"] = {
        "samples": samples // 10,
        "total_ms": round(elapsed * 1000, 1),
        "per_profile_ms": round(elapsed * 1000 / (samples // 10 or 1), 2),
    }
    print(f"✅ ({results['profile_generation']['per_profile_ms']}ms/profile)")
    
    # Batch generation benchmark
    print("  • Batch Generation (100 entities)...", end=" ")
    start = time.perf_counter()
    factory.generate_batch("restaurant", count=100)
    elapsed = time.perf_counter() - start
    results["batch_100"] = {
        "total_ms": round(elapsed * 1000, 1),
    }
    print(f"✅ ({results['batch_100']['total_ms']}ms)")
    
    # Large batch benchmark
    print("  • Large Batch (1000 entities)...", end=" ")
    start = time.perf_counter()
    factory.generate_batch("restaurant", count=1000)
    elapsed = time.perf_counter() - start
    results["batch_1000"] = {
        "total_ms": round(elapsed * 1000, 1),
    }
    print(f"✅ ({results['batch_1000']['total_ms']}ms)")
    
    # Scoring benchmark (mock)
    print("  • Scoring (mock)...", end=" ")
    layer3 = Layer3Diagnostics()
    entities = factory.generate_batch("restaurant", count=samples)
    start = time.perf_counter()
    for entity in entities:
        layer3.scoring_fn(entity)
    elapsed = time.perf_counter() - start
    results["scoring"] = {
        "samples": samples,
        "total_ms": round(elapsed * 1000, 1),
        "per_entity_ms": round(elapsed * 1000 / samples, 2),
    }
    print(f"✅ ({results['scoring']['per_entity_ms']}ms/entity)")
    
    # Summary
    print("\n📋 Benchmark Summary:")
    print(f"  Entity Generation:  {results['entity_generation']['per_entity_ms']}ms/entity")
    print(f"  Profile Generation: {results['profile_generation']['per_profile_ms']}ms/profile")
    print(f"  Batch 100:          {results['batch_100']['total_ms']}ms total")
    print(f"  Batch 1000:         {results['batch_1000']['total_ms']}ms total")
    print(f"  Scoring:            {results['scoring']['per_entity_ms']}ms/entity")
    
    # Save benchmark results
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    benchmark_path = reports_dir / "benchmark_results.json"
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Saved to: {benchmark_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARBOR Test Sandbox - Diagnostic Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_diagnostics.py                  Run all layers
  python run_all_diagnostics.py --layer 1        Run Layer 1 only
  python run_all_diagnostics.py --quick          Quick smoke test
  python run_all_diagnostics.py --benchmark      Run benchmarks
  python run_all_diagnostics.py --scenarios      Include scenario tests
        """
    )
    
    parser.add_argument(
        "--layer", "-l",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific layer (1-4)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick smoke test"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--scenarios", "-s",
        action="store_true",
        help="Include scenario-based tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't save report files"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    start_time = time.perf_counter()
    
    try:
        if args.quick:
            success = run_quick_smoke_test()
            sys.exit(0 if success else 1)
        
        if args.benchmark:
            run_benchmark()
            sys.exit(0)
        
        if args.layer:
            run_single_layer(args.layer)
            sys.exit(0)
        
        # Default: run all layers
        asyncio.run(run_all_layers(
            include_scenarios=args.scenarios,
            save_report=not args.no_report,
            verbose=args.verbose,
        ))
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"\n⏱️ Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
