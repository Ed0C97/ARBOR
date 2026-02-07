"""Data Populator - Fills tables with pseudo-realistic data.

Uses Faker to generate Italian-localized, domain-specific data.
Supports configurable row counts and edge case injection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timedelta
from faker import Faker

from .schema_generator import TableSchema, ColumnSchema, DatabaseSchema


# =============================================================================
# DATA GENERATORS PER COLUMN TYPE
# =============================================================================

class ColumnDataGenerator:
    """Generates fake data for different column types."""
    
    def __init__(self, seed: int = 42, locale: str = "it_IT"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fake = Faker(locale)
        self.fake.seed_instance(seed)
    
    def generate_value(
        self,
        column: ColumnSchema,
        row_index: int = 0,
        inject_edge_cases: bool = False,
    ) -> Any:
        """Generate a value for a column.
        
        Args:
            column: Column schema
            row_index: Current row index (for deterministic patterns)
            inject_edge_cases: Whether to inject edge cases (NULL, unicode, etc.)
        
        Returns:
            Generated value
        """
        # Handle nullable with chance of NULL
        if column.nullable and inject_edge_cases and self.rng.random() < 0.05:
            return None
        
        # Infer generator from column name first
        value = self._generate_by_name(column.name, row_index)
        if value is not None:
            return value
        
        # Fall back to type-based generation
        return self._generate_by_type(column.data_type, inject_edge_cases)
    
    def _generate_by_name(self, name: str, row_index: int) -> Any | None:
        """Generate value based on column name heuristics."""
        name_lower = name.lower()
        
        # Identifiers
        if name_lower == "id":
            return row_index + 1
        
        # Names
        if "name" in name_lower:
            if "first" in name_lower:
                return self.fake.first_name()
            elif "last" in name_lower:
                return self.fake.last_name()
            elif "company" in name_lower or "business" in name_lower:
                return self.fake.company()
            else:
                # Generic name - could be product, restaurant, etc.
                return self._generate_entity_name()
        
        # Contact info
        if "email" in name_lower:
            return self.fake.email()
        if "phone" in name_lower:
            return self.fake.phone_number()
        if "address" in name_lower:
            return self.fake.address().replace("\n", ", ")
        if "city" in name_lower:
            return self.fake.city()
        if "country" in name_lower:
            return "Italia"
        if "region" in name_lower:
            return self.fake.administrative_unit()
        if "zip" in name_lower or "postal" in name_lower or "cap" in name_lower:
            return self.fake.postcode()
        
        # URLs
        if "url" in name_lower or "website" in name_lower:
            return self.fake.url()
        if "instagram" in name_lower:
            return f"@{self.fake.user_name()}"
        
        # Prices and money
        if "price" in name_lower or "amount" in name_lower or "rate" in name_lower:
            return round(self.rng.uniform(5.0, 500.0), 2)
        if "total" in name_lower:
            return round(self.rng.uniform(10.0, 1000.0), 2)
        
        # Counts and quantities
        if "count" in name_lower or "quantity" in name_lower or "num" in name_lower:
            return self.rng.randint(1, 100)
        
        # Ratings and scores
        if "rating" in name_lower or "score" in name_lower:
            return round(self.rng.uniform(1.0, 5.0), 1)
        
        # Booleans
        if name_lower.startswith("is_") or name_lower.startswith("has_"):
            return self.rng.random() > 0.3  # 70% True
        
        # Descriptions and text
        if "description" in name_lower or "notes" in name_lower:
            return self.fake.paragraph(nb_sentences=2)
        if "comment" in name_lower:
            return self.fake.sentence()
        
        # Dates
        if "created" in name_lower:
            return self.fake.date_time_between(start_date="-2y", end_date="now")
        if "updated" in name_lower:
            return self.fake.date_time_between(start_date="-30d", end_date="now")
        if "date" in name_lower:
            return self.fake.date_between(start_date="-1y", end_date="+1y")
        
        # Categories and types
        if "category" in name_lower or "type" in name_lower:
            return self.rng.choice([
                "Standard", "Premium", "Basic", "VIP", "Economy", "Deluxe"
            ])
        
        # Status
        if "status" in name_lower:
            return self.rng.choice([
                "active", "pending", "completed", "cancelled", "archived"
            ])
        
        # Priority
        if "priority" in name_lower:
            return self.rng.randint(1, 5)
        
        return None
    
    def _generate_by_type(self, data_type: str, inject_edge_cases: bool) -> Any:
        """Generate value based on PostgreSQL type."""
        type_upper = data_type.upper()
        
        if "SERIAL" in type_upper or "INT" in type_upper or "BIGINT" in type_upper:
            return self.rng.randint(1, 10000)
        
        elif "DECIMAL" in type_upper or "NUMERIC" in type_upper or "FLOAT" in type_upper:
            return round(self.rng.uniform(0.0, 1000.0), 2)
        
        elif "BOOL" in type_upper:
            return self.rng.random() > 0.5
        
        elif "DATE" in type_upper and "TIME" not in type_upper:
            return self.fake.date_between(start_date="-2y", end_date="+1y")
        
        elif "TIME" in type_upper:
            return self.fake.date_time_between(start_date="-1y", end_date="now")
        
        elif "JSON" in type_upper:
            return self._generate_json_data()
        
        elif "TEXT" in type_upper:
            if inject_edge_cases and self.rng.random() < 0.1:
                return self._generate_edge_case_text()
            return self.fake.paragraph(nb_sentences=3)
        
        else:  # VARCHAR and others
            if inject_edge_cases and self.rng.random() < 0.1:
                return self._generate_edge_case_text()
            return self.fake.sentence()[:200]
    
    def _generate_entity_name(self) -> str:
        """Generate a realistic entity name (restaurant, hotel, etc.)."""
        templates = [
            f"Ristorante {self.fake.last_name()}",
            f"Trattoria {self.fake.first_name()}",
            f"Hotel {self.fake.last_name()}",
            f"Bar {self.fake.last_name()}",
            f"Pizzeria {self.fake.first_name()}",
            f"Osteria del {self.fake.last_name()}",
            f"Caffè {self.fake.first_name()}",
            f"Locanda {self.fake.last_name()}",
            f"Spa {self.fake.city()}",
            f"Centro {self.fake.last_name()}",
        ]
        return self.rng.choice(templates)
    
    def _generate_json_data(self) -> dict:
        """Generate random JSON data."""
        return {
            "key1": self.fake.word(),
            "key2": self.rng.randint(1, 100),
            "active": self.rng.random() > 0.5,
            "tags": [self.fake.word() for _ in range(self.rng.randint(1, 5))],
        }
    
    def _generate_edge_case_text(self) -> str:
        """Generate edge case text for testing."""
        cases = [
            # Unicode
            "Ristorante 日本 & Caffè ☕",
            "L'Osteria dell'Arco – Menù €€€",
            "Trattoria «Il Bello»",
            # Very long
            self.fake.paragraph(nb_sentences=10),
            # Special characters
            "Restaurant <Test> & 'Quotes' \"Double\"",
            # RTL
            "مطعم عربي - Arabic Restaurant",
            # Emoji
            "Pizza Place 🍕🍝 Italian Food",
            # Empty-ish
            "   ",
            "-",
        ]
        return self.rng.choice(cases)


# =============================================================================
# DATA POPULATOR
# =============================================================================

@dataclass
class PopulationResult:
    """Result of data population."""
    schema_name: str
    rows_by_table: dict[str, int]
    total_rows: int
    duration_seconds: float
    edge_cases_injected: int


class DataPopulator:
    """Populates database tables with realistic data."""
    
    def __init__(self, seed: int = 42, locale: str = "it_IT"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.generator = ColumnDataGenerator(seed, locale)
    
    def generate_insert_statements(
        self,
        schema: DatabaseSchema,
        rows_per_table: int | tuple[int, int] = 100,
        inject_edge_cases: bool = True,
    ) -> tuple[list[str], PopulationResult]:
        """Generate INSERT statements for all tables.
        
        Args:
            schema: Database schema to populate
            rows_per_table: Fixed count or (min, max) range
            inject_edge_cases: Whether to inject edge cases
        
        Returns:
            Tuple of (SQL statements, population result)
        """
        import time
        start = time.perf_counter()
        
        statements = []
        rows_by_table = {}
        edge_cases = 0
        
        # Sort tables to insert parents before children (FK order)
        ordered_tables = self._topological_sort(schema.tables)
        
        for table in ordered_tables:
            # Determine row count
            if isinstance(rows_per_table, tuple):
                num_rows = self.rng.randint(rows_per_table[0], rows_per_table[1])
            else:
                num_rows = rows_per_table
            
            # Generate rows
            table_statements, table_edge_cases = self._generate_table_inserts(
                schema.schema_name,
                table,
                num_rows,
                inject_edge_cases,
            )
            
            statements.extend(table_statements)
            rows_by_table[table.name] = num_rows
            edge_cases += table_edge_cases
        
        duration = time.perf_counter() - start
        
        result = PopulationResult(
            schema_name=schema.schema_name,
            rows_by_table=rows_by_table,
            total_rows=sum(rows_by_table.values()),
            duration_seconds=round(duration, 3),
            edge_cases_injected=edge_cases,
        )
        
        return statements, result
    
    def _generate_table_inserts(
        self,
        schema_name: str,
        table: TableSchema,
        num_rows: int,
        inject_edge_cases: bool,
    ) -> tuple[list[str], int]:
        """Generate INSERT statements for a single table."""
        statements = []
        edge_cases = 0
        
        # Get non-PK columns for insert
        columns = [c for c in table.columns if not c.is_primary_key or c.data_type != "SERIAL"]
        col_names = [c.name for c in columns]
        
        for row_idx in range(num_rows):
            values = []
            
            for col in columns:
                # Inject edge cases occasionally
                is_edge = inject_edge_cases and self.rng.random() < 0.05
                if is_edge:
                    edge_cases += 1
                
                value = self.generator.generate_value(col, row_idx, is_edge)
                values.append(self._sql_escape(value))
            
            cols_sql = ", ".join(f'"{c}"' for c in col_names)
            vals_sql = ", ".join(values)
            
            stmt = f'INSERT INTO "{schema_name}"."{table.name}" ({cols_sql}) VALUES ({vals_sql});'
            statements.append(stmt)
        
        return statements, edge_cases
    
    def _sql_escape(self, value: Any) -> str:
        """Escape value for SQL INSERT."""
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, datetime):
            return f"'{value.isoformat()}'"
        elif isinstance(value, dict):
            import json
            return f"'{json.dumps(value)}'"
        else:
            # Escape single quotes
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"
    
    def _topological_sort(self, tables: list[TableSchema]) -> list[TableSchema]:
        """Sort tables so parents come before children (for FK constraints)."""
        # Build dependency graph
        deps = {t.name: set() for t in tables}
        table_map = {t.name: t for t in tables}
        
        for table in tables:
            for col in table.columns:
                if col.is_foreign_key and col.references_table:
                    if col.references_table in deps:
                        deps[table.name].add(col.references_table)
        
        # Kahn's algorithm for topological sort
        result = []
        no_deps = [t for t in tables if not deps[t.name]]
        
        while no_deps:
            table = no_deps.pop(0)
            result.append(table)
            
            # Remove from dependencies
            for name, d in deps.items():
                d.discard(table.name)
                if not d and table_map[name] not in result and table_map[name] not in no_deps:
                    no_deps.append(table_map[name])
        
        # Add any remaining (circular deps)
        for table in tables:
            if table not in result:
                result.append(table)
        
        return result


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    from .schema_generator import SchemaGenerator
    
    gen = SchemaGenerator(seed=42)
    schema = gen.generate(vertical="restaurant", num_tables=3)
    
    pop = DataPopulator(seed=42)
    statements, result = pop.generate_insert_statements(schema, rows_per_table=5)
    
    print(f"Schema: {result.schema_name}")
    print(f"Total rows: {result.total_rows}")
    print(f"Edge cases: {result.edge_cases_injected}")
    print(f"Duration: {result.duration_seconds}s")
    print()
    print("Sample statements:")
    for stmt in statements[:3]:
        print(stmt[:100] + "...")
