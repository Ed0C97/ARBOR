"""Cleanup Manager - Safe database destruction.

Handles complete cleanup of test schemas to prevent memory accumulation.
Uses DROP SCHEMA CASCADE for instant cleanup.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    schema_name: str
    success: bool
    tables_dropped: int
    error: str | None = None
    duration_seconds: float = 0.0


class CleanupManager:
    """Manages safe destruction of test databases."""
    
    def __init__(self, connection: Any = None):
        """Initialize cleanup manager.
        
        Args:
            connection: Database connection (psycopg2 or asyncpg)
        """
        self.connection = connection
        self._schemas_created: list[str] = []
    
    def register_schema(self, schema_name: str) -> None:
        """Register a schema for later cleanup."""
        if schema_name not in self._schemas_created:
            self._schemas_created.append(schema_name)
    
    def drop_schema(self, schema_name: str, conn: Any = None) -> CleanupResult:
        """Drop a schema and all its contents.
        
        Args:
            schema_name: Name of schema to drop
            conn: Optional connection override
        
        Returns:
            CleanupResult with status
        """
        import time
        start = time.perf_counter()
        
        connection = conn or self.connection
        
        if connection is None:
            return CleanupResult(
                schema_name=schema_name,
                success=False,
                tables_dropped=0,
                error="No database connection available",
            )
        
        try:
            cursor = connection.cursor()
            
            # Count tables first
            cursor.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = '{schema_name}'
            """)
            table_count = cursor.fetchone()[0]
            
            # Drop schema with cascade
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
            connection.commit()
            
            # Remove from tracking
            if schema_name in self._schemas_created:
                self._schemas_created.remove(schema_name)
            
            duration = time.perf_counter() - start
            
            logger.info(f"Dropped schema {schema_name} ({table_count} tables) in {duration:.3f}s")
            
            return CleanupResult(
                schema_name=schema_name,
                success=True,
                tables_dropped=table_count,
                duration_seconds=round(duration, 3),
            )
            
        except Exception as e:
            logger.error(f"Failed to drop schema {schema_name}: {e}")
            return CleanupResult(
                schema_name=schema_name,
                success=False,
                tables_dropped=0,
                error=str(e),
            )
    
    def cleanup_all(self, conn: Any = None) -> list[CleanupResult]:
        """Cleanup all registered schemas.
        
        Args:
            conn: Optional connection override
        
        Returns:
            List of cleanup results
        """
        results = []
        
        # Copy list since we modify it during iteration
        schemas = self._schemas_created.copy()
        
        for schema_name in schemas:
            result = self.drop_schema(schema_name, conn)
            results.append(result)
        
        # Force garbage collection
        self.force_memory_cleanup()
        
        return results
    
    def force_memory_cleanup(self) -> dict[str, int]:
        """Force Python garbage collection.
        
        Returns:
            Dict with garbage collection stats
        """
        # Run full garbage collection
        collected = {
            "gen0": gc.collect(0),
            "gen1": gc.collect(1),
            "gen2": gc.collect(2),
        }
        
        logger.debug(f"Garbage collected: {collected}")
        
        return collected
    
    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage.
        
        Returns:
            Dict with memory stats in MB
        """
        import os
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            
            return {
                "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
                "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
            }
        except ImportError:
            # psutil not available
            return {"rss_mb": -1, "vms_mb": -1, "percent": -1}
    
    @property
    def pending_schemas(self) -> list[str]:
        """Get list of schemas waiting for cleanup."""
        return self._schemas_created.copy()


class MockConnection:
    """Mock database connection for testing without real PostgreSQL."""
    
    def __init__(self):
        self._schemas: dict[str, list[str]] = {}
    
    def cursor(self):
        return MockCursor(self)
    
    def commit(self):
        pass
    
    def create_schema(self, schema_name: str, tables: list[str]):
        self._schemas[schema_name] = tables
    
    def drop_schema(self, schema_name: str):
        if schema_name in self._schemas:
            del self._schemas[schema_name]


class MockCursor:
    """Mock cursor for testing."""
    
    def __init__(self, connection: MockConnection):
        self.connection = connection
        self._result = None
    
    def execute(self, sql: str):
        sql_upper = sql.upper()
        
        if "COUNT(*)" in sql_upper and "INFORMATION_SCHEMA" in sql_upper:
            # Extract schema name
            for schema_name, tables in self.connection._schemas.items():
                if schema_name in sql:
                    self._result = [(len(tables),)]
                    return
            self._result = [(0,)]
        
        elif "DROP SCHEMA" in sql_upper:
            # Extract and drop schema
            for schema_name in list(self.connection._schemas.keys()):
                if schema_name in sql:
                    self.connection.drop_schema(schema_name)
                    break
    
    def fetchone(self):
        if self._result:
            return self._result[0]
        return (0,)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # Test with mock connection
    mock_conn = MockConnection()
    mock_conn.create_schema("test_schema_1", ["table1", "table2", "table3"])
    mock_conn.create_schema("test_schema_2", ["tableA", "tableB"])
    
    cleanup = CleanupManager(mock_conn)
    cleanup.register_schema("test_schema_1")
    cleanup.register_schema("test_schema_2")
    
    print(f"Pending schemas: {cleanup.pending_schemas}")
    
    results = cleanup.cleanup_all()
    
    for r in results:
        status = "✅" if r.success else "❌"
        print(f"{status} {r.schema_name}: {r.tables_dropped} tables dropped")
    
    print(f"Memory: {cleanup.get_memory_usage()}")
