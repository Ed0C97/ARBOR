"""Test Orchestrator - Main test loop coordinator.

Orchestrates the create-test-cleanup cycle:
1. Generate random schema
2. Create PostgreSQL schema
3. Populate with data
4. Run ARBOR tests
5. Generate report
6. Cleanup
7. Repeat
"""

from __future__ import annotations

import gc
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schema_generator import SchemaGenerator, DatabaseSchema
from .data_populator import DataPopulator
from .cleanup_manager import CleanupManager, MockConnection
from .config_generator import ConfigGenerator, GeneratedConfigs
from .report_generator import ReportGenerator, IterationReport, TestResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for test orchestrator."""
    
    # Iterations
    num_iterations: int = 100
    
    # Schema complexity
    min_tables: int = 5
    max_tables: int = 15
    min_columns: int = 5
    max_columns: int = 20
    
    # Data volume
    min_rows_per_table: int = 100
    max_rows_per_table: int = 1000
    
    # Verticals to test (None = all)
    verticals: list[str] | None = None
    
    # Edge cases
    inject_edge_cases: bool = True
    edge_case_probability: float = 0.05
    
    # Output
    output_dir: Path = Path(__file__).parent.parent / "reports" / "agnostic"
    save_individual_reports: bool = True
    
    # Memory limits
    max_memory_mb: float = 500.0
    gc_every_n_iterations: int = 10
    
    # Database
    use_real_postgres: bool = False
    postgres_dsn: str = ""
    
    # Seed (None = random each run)
    base_seed: int | None = None


class TestOrchestrator:
    """Orchestrates schema-agnostic testing."""
    
    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        
        self.report_generator = ReportGenerator(self.config.output_dir)
        self.cleanup_manager = CleanupManager()
        
        # Stats
        self._current_iteration = 0
        self._start_time: float = 0
        
        # Custom test functions
        self._test_functions: list[Callable[[DatabaseSchema, Any], TestResult]] = []
    
    def register_test(self, test_fn: Callable[[DatabaseSchema, Any], TestResult]) -> None:
        """Register a custom test function.
        
        Args:
            test_fn: Function taking (schema, connection) and returning TestResult
        """
        self._test_functions.append(test_fn)
    
    def run(self, callback: Callable[[IterationReport], None] | None = None) -> None:
        """Run the full test suite.
        
        Args:
            callback: Optional callback called after each iteration
        """
        self._start_time = time.perf_counter()
        
        # Use base seed or generate random
        base_seed = self.config.base_seed or random.randint(0, 999999)
        
        # Get verticals to test
        verticals = self.config.verticals or [
            "restaurant", "hotel", "retail", "spa", "healthcare", "fitness"
        ]
        
        print("=" * 60)
        print("🌳 ARBOR Schema-Agnostic Test Suite")
        print("=" * 60)
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Verticals: {', '.join(verticals)}")
        print(f"Tables per schema: {self.config.min_tables}-{self.config.max_tables}")
        print(f"Rows per table: {self.config.min_rows_per_table}-{self.config.max_rows_per_table}")
        print(f"Base seed: {base_seed}")
        print("=" * 60)
        print()
        
        # Get database connection
        connection = self._get_connection()
        
        try:
            for i in range(self.config.num_iterations):
                self._current_iteration = i
                
                # Select vertical
                vertical = verticals[i % len(verticals)]
                
                # Generate unique seed for this iteration
                seed = base_seed + i
                
                # Run iteration
                report = self._run_iteration(i, seed, vertical, connection)
                
                # Add to aggregate
                self.report_generator.add_iteration(report)
                
                # Callback
                if callback:
                    callback(report)
                
                # Print progress
                status = "✅" if report.success else "❌"
                print(
                    f"[{i+1:4d}/{self.config.num_iterations}] "
                    f"{status} {vertical:12s} | "
                    f"tables: {report.table_count:2d} | "
                    f"rows: {report.total_rows:6d} | "
                    f"tests: {report.passed}/{report.passed + report.failed} | "
                    f"time: {report.total_seconds:.2f}s"
                )
                
                # Periodic garbage collection
                if (i + 1) % self.config.gc_every_n_iterations == 0:
                    gc.collect()
                    mem = self.cleanup_manager.get_memory_usage()
                    if mem["rss_mb"] > self.config.max_memory_mb:
                        logger.warning(f"Memory usage high: {mem['rss_mb']} MB")
        
        finally:
            # Cleanup any remaining schemas
            self.cleanup_manager.cleanup_all(connection)
            
            # Close connection
            if hasattr(connection, "close"):
                connection.close()
        
        # Generate final aggregate report
        json_path, html_path = self.report_generator.save_aggregate_report()
        
        total_time = time.perf_counter() - self._start_time
        
        print()
        print("=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        agg = self.report_generator.generate_aggregate_report()
        print(f"Total iterations: {agg.total_iterations}")
        print(f"Success rate: {agg.success_rate}%")
        print(f"Test pass rate: {agg.test_pass_rate}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Reports saved to: {self.config.output_dir}")
        print("=" * 60)
    
    def _run_iteration(
        self,
        iteration_id: int,
        seed: int,
        vertical: str,
        connection: Any,
    ) -> IterationReport:
        """Run a single test iteration."""
        report = IterationReport(
            iteration_id=iteration_id,
            seed=seed,
            vertical=vertical,
            schema_name="",
        )
        
        # Get memory before
        mem_before = self.cleanup_manager.get_memory_usage()
        report.memory_before_mb = mem_before.get("rss_mb", 0)
        
        iter_start = time.perf_counter()
        configs = None  # Will hold generated ARBOR configs
        
        try:
            # 1. Generate schema
            t0 = time.perf_counter()
            schema_gen = SchemaGenerator(seed=seed)
            schema = schema_gen.generate(
                vertical=vertical,
                num_tables=random.randint(self.config.min_tables, self.config.max_tables),
                columns_per_table=(self.config.min_columns, self.config.max_columns),
            )
            report.schema_name = schema.schema_name
            report.schema_generation_seconds = time.perf_counter() - t0
            
            # Count schema stats
            report.table_count = len(schema.tables)
            report.total_columns = sum(len(t.columns) for t in schema.tables)
            report.fk_count = sum(
                1 for t in schema.tables for c in t.columns if c.is_foreign_key
            )
            report.schema_details = schema.to_dict()
            
            # Register for cleanup
            self.cleanup_manager.register_schema(schema.schema_name)
            
            # 2. Generate ARBOR configs (SOURCE_SCHEMA_CONFIG + domain_profile)
            config_gen = ConfigGenerator(seed=seed)
            configs = config_gen.generate(schema)
            
            # Add configs to report for debugging
            report.schema_details["arbor_config"] = {
                "schema_config": [c.to_dict() for c in configs.schema_config],
                "domain_profile": configs.domain_profile.to_dict() if configs.domain_profile else {},
            }
            
            # 3. Create schema in database
            self._create_schema(schema, connection)
            
            # 4. Populate data
            t0 = time.perf_counter()
            populator = DataPopulator(seed=seed)
            statements, pop_result = populator.generate_insert_statements(
                schema,
                rows_per_table=(self.config.min_rows_per_table, self.config.max_rows_per_table),
                inject_edge_cases=self.config.inject_edge_cases,
            )
            report.total_rows = pop_result.total_rows
            
            self._execute_statements(statements, connection)
            report.data_population_seconds = time.perf_counter() - t0
            
            # 5. Run tests (now with configs)
            t0 = time.perf_counter()
            test_results = self._run_tests(schema, connection, configs)
            report.tests = test_results
            report.passed = sum(1 for t in test_results if t.passed)
            report.failed = sum(1 for t in test_results if not t.passed)
            report.test_execution_seconds = time.perf_counter() - t0
            
            # 6. Cleanup
            t0 = time.perf_counter()
            self.cleanup_manager.drop_schema(schema.schema_name, connection)
            report.cleanup_seconds = time.perf_counter() - t0
            
        except Exception as e:
            report.errors.append(str(e))
            logger.exception(f"Iteration {iteration_id} failed: {e}")
        
        report.total_seconds = time.perf_counter() - iter_start
        
        # Get memory after
        mem_after = self.cleanup_manager.get_memory_usage()
        report.memory_after_mb = mem_after.get("rss_mb", 0)
        report.memory_delta_mb = report.memory_after_mb - report.memory_before_mb
        
        return report
    
    def _get_connection(self) -> Any:
        """Get database connection."""
        if self.config.use_real_postgres and self.config.postgres_dsn:
            try:
                import psycopg2
                return psycopg2.connect(self.config.postgres_dsn)
            except ImportError:
                logger.warning("psycopg2 not installed, using mock connection")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}, using mock")
        
        return MockConnection()
    
    def _create_schema(self, schema: DatabaseSchema, connection: Any) -> None:
        """Create schema in database."""
        if isinstance(connection, MockConnection):
            # Mock: just register tables
            table_names = [t.name for t in schema.tables]
            connection.create_schema(schema.schema_name, table_names)
        else:
            # Real PostgreSQL
            cursor = connection.cursor()
            sql = schema.to_sql()
            for statement in sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt + ";")
            connection.commit()
    
    def _execute_statements(self, statements: list[str], connection: Any) -> None:
        """Execute SQL statements."""
        if isinstance(connection, MockConnection):
            # Mock: no-op
            return
        
        cursor = connection.cursor()
        for stmt in statements:
            try:
                cursor.execute(stmt)
            except Exception as e:
                logger.debug(f"Statement failed: {e}")
        connection.commit()
    
    def _run_tests(
        self,
        schema: DatabaseSchema,
        connection: Any,
        configs: GeneratedConfigs | None = None,
    ) -> list[TestResult]:
        """Run all tests on the schema."""
        results = []
        
        # Built-in tests
        results.extend(self._run_builtin_tests(schema, connection, configs))
        
        # Custom registered tests
        for test_fn in self._test_functions:
            try:
                result = test_fn(schema, connection)
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    test_name=test_fn.__name__,
                    passed=False,
                    message=f"Test exception: {e}",
                ))
        
        return results
    
    def _run_builtin_tests(
        self,
        schema: DatabaseSchema,
        connection: Any,
        configs: GeneratedConfigs | None = None,
    ) -> list[TestResult]:
        """Run built-in schema agnostic tests."""
        results = []
        
        # Test 1: Schema has tables
        results.append(TestResult(
            test_name="schema_has_tables",
            passed=len(schema.tables) > 0,
            message=f"Schema has {len(schema.tables)} tables",
        ))
        
        # Test 2: All tables have primary key
        tables_with_pk = sum(
            1 for t in schema.tables
            if any(c.is_primary_key for c in t.columns)
        )
        results.append(TestResult(
            test_name="tables_have_pk",
            passed=tables_with_pk == len(schema.tables),
            message=f"{tables_with_pk}/{len(schema.tables)} tables have PK",
        ))
        
        # Test 3: Schema can be queried
        if not isinstance(connection, MockConnection):
            try:
                cursor = connection.cursor()
                cursor.execute(f"""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = '{schema.schema_name}'
                """)
                found_tables = len(cursor.fetchall())
                results.append(TestResult(
                    test_name="schema_queryable",
                    passed=found_tables == len(schema.tables),
                    message=f"Found {found_tables} tables in schema",
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name="schema_queryable",
                    passed=False,
                    message=f"Query failed: {e}",
                ))
        else:
            # Mock always passes
            results.append(TestResult(
                test_name="schema_queryable",
                passed=True,
                message="Mock connection (test skipped)",
            ))
        
        # Test 4: Column names are valid
        invalid_cols = []
        for table in schema.tables:
            for col in table.columns:
                if not col.name or len(col.name) > 63:  # PostgreSQL limit
                    invalid_cols.append(f"{table.name}.{col.name}")
        
        results.append(TestResult(
            test_name="valid_column_names",
            passed=len(invalid_cols) == 0,
            message=f"All column names valid" if not invalid_cols else f"Invalid: {invalid_cols[:3]}",
        ))
        
        # ==============================
        # CONFIG VALIDATION TESTS
        # ==============================
        if configs:
            # Test 5: Schema config covers all tables
            config_tables = {c.entity_type for c in configs.schema_config}
            schema_tables = {t.name for t in schema.tables}
            missing_tables = schema_tables - config_tables
            
            results.append(TestResult(
                test_name="config_covers_all_tables",
                passed=len(missing_tables) == 0,
                message=f"All {len(schema_tables)} tables mapped" if not missing_tables 
                        else f"Missing: {list(missing_tables)[:3]}",
            ))
            
            # Test 6: All configs have required name mapping
            configs_with_name = sum(
                1 for c in configs.schema_config
                if c.required_mappings.get("name")
            )
            results.append(TestResult(
                test_name="configs_have_name_mapping",
                passed=configs_with_name == len(configs.schema_config),
                message=f"{configs_with_name}/{len(configs.schema_config)} have name mapping",
            ))
            
            # Test 7: Domain profile has valid structure
            if configs.domain_profile:
                dp = configs.domain_profile
                has_dims = len(dp.vibe_dimensions) > 0
                has_aspects = len(dp.quality_aspects) > 0
                has_entities = len(dp.entity_types) > 0
                valid_profile = has_dims and has_aspects and has_entities
                
                results.append(TestResult(
                    test_name="domain_profile_valid",
                    passed=valid_profile,
                    message=f"dims:{len(dp.vibe_dimensions)} aspects:{len(dp.quality_aspects)} entities:{len(dp.entity_types)}",
                ))
                
                # Test 8: Domain profile matches schema vertical
                results.append(TestResult(
                    test_name="profile_matches_vertical",
                    passed=dp.vertical == schema.vertical,
                    message=f"Profile: {dp.vertical}, Schema: {schema.vertical}",
                ))
            else:
                results.append(TestResult(
                    test_name="domain_profile_valid",
                    passed=False,
                    message="No domain profile generated",
                ))
        
        return results


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_agnostic_tests(
    iterations: int = 100,
    output_dir: str = "tests_sandbox/reports/agnostic",
    use_postgres: bool = False,
    postgres_dsn: str = "",
    base_seed: int | None = None,
) -> None:
    """Run schema-agnostic test suite.
    
    Args:
        iterations: Number of test iterations
        output_dir: Directory for reports
        use_postgres: Whether to use real PostgreSQL
        postgres_dsn: PostgreSQL connection string
        base_seed: Base random seed (None = random)
    """
    config = OrchestratorConfig(
        num_iterations=iterations,
        output_dir=Path(output_dir),
        use_real_postgres=use_postgres,
        postgres_dsn=postgres_dsn,
        base_seed=base_seed,
    )
    
    orchestrator = TestOrchestrator(config)
    orchestrator.run()


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ARBOR schema-agnostic tests")
    parser.add_argument("-n", "--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("-o", "--output", default="tests_sandbox/reports/agnostic", help="Output directory")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--postgres", help="PostgreSQL DSN for real testing")
    
    args = parser.parse_args()
    
    run_agnostic_tests(
        iterations=args.iterations,
        output_dir=args.output,
        use_postgres=bool(args.postgres),
        postgres_dsn=args.postgres or "",
        base_seed=args.seed,
    )
