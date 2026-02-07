"""Schema Generator - Creates random PostgreSQL schemas.

Generates realistic database schemas with:
- Domain-specific table names (restaurant, hotel, retail, etc.)
- Random column types (VARCHAR, INT, DATE, JSONB, DECIMAL, BOOLEAN)
- Foreign key relationships
- Configurable complexity
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal
from faker import Faker

fake = Faker("it_IT")


# =============================================================================
# DOMAIN TEMPLATES
# =============================================================================

# Table templates per vertical
DOMAIN_TEMPLATES = {
    "restaurant": {
        "tables": [
            "menu_items", "orders", "customers", "reservations", "tables",
            "staff", "suppliers", "inventory", "reviews", "categories",
            "promotions", "delivery_zones", "payment_methods", "allergens",
        ],
        "common_columns": ["name", "description", "price", "is_active", "created_at"],
    },
    "hotel": {
        "tables": [
            "rooms", "bookings", "guests", "room_types", "amenities",
            "staff", "services", "invoices", "reviews", "housekeeping",
            "events", "packages", "loyalty_members", "complaints",
        ],
        "common_columns": ["name", "description", "rate", "is_available", "created_at"],
    },
    "retail": {
        "tables": [
            "products", "orders", "customers", "categories", "inventory",
            "suppliers", "warehouses", "shipments", "returns", "promotions",
            "reviews", "wishlists", "carts", "payment_transactions",
        ],
        "common_columns": ["name", "description", "price", "sku", "created_at"],
    },
    "spa": {
        "tables": [
            "treatments", "appointments", "clients", "therapists", "rooms",
            "products", "packages", "memberships", "reviews", "gift_cards",
            "promotions", "equipment", "suppliers", "inventory",
        ],
        "common_columns": ["name", "description", "duration_minutes", "price", "created_at"],
    },
    "healthcare": {
        "tables": [
            "patients", "appointments", "doctors", "departments", "treatments",
            "prescriptions", "medical_records", "lab_results", "insurance",
            "billing", "rooms", "equipment", "suppliers", "staff",
        ],
        "common_columns": ["name", "description", "code", "is_active", "created_at"],
    },
    "fitness": {
        "tables": [
            "members", "classes", "trainers", "equipment", "memberships",
            "bookings", "schedules", "payments", "reviews", "programs",
            "achievements", "nutrition_plans", "body_metrics", "challenges",
        ],
        "common_columns": ["name", "description", "capacity", "is_active", "created_at"],
    },
}

# Column type weights (PostgreSQL types)
COLUMN_TYPES = {
    "VARCHAR(255)": 25,
    "TEXT": 10,
    "INTEGER": 20,
    "BIGINT": 5,
    "DECIMAL(10,2)": 15,
    "BOOLEAN": 10,
    "DATE": 5,
    "TIMESTAMP": 8,
    "JSONB": 2,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnSchema:
    """Definition of a database column."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: str | None = None
    references_column: str | None = None
    default: str | None = None
    
    def to_sql(self) -> str:
        """Generate SQL column definition."""
        parts = [f'"{self.name}"', self.data_type]
        
        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        elif not self.nullable:
            parts.append("NOT NULL")
        
        if self.default:
            parts.append(f"DEFAULT {self.default}")
        
        return " ".join(parts)


@dataclass
class TableSchema:
    """Definition of a database table."""
    name: str
    columns: list[ColumnSchema] = field(default_factory=list)
    
    def to_sql(self, schema_name: str = "public") -> str:
        """Generate CREATE TABLE SQL."""
        cols_sql = ",\n    ".join(col.to_sql() for col in self.columns)
        
        # Add foreign key constraints
        fk_constraints = []
        for col in self.columns:
            if col.is_foreign_key and col.references_table:
                fk_constraints.append(
                    f'FOREIGN KEY ("{col.name}") REFERENCES "{schema_name}"."{col.references_table}"("{col.references_column}")'
                )
        
        all_constraints = cols_sql
        if fk_constraints:
            all_constraints += ",\n    " + ",\n    ".join(fk_constraints)
        
        return f'CREATE TABLE "{schema_name}"."{self.name}" (\n    {all_constraints}\n);'


@dataclass
class DatabaseSchema:
    """Complete database schema definition."""
    schema_name: str
    vertical: str
    tables: list[TableSchema] = field(default_factory=list)
    seed: int = 42
    
    def to_sql(self) -> str:
        """Generate complete SQL for schema creation."""
        lines = [
            f'CREATE SCHEMA IF NOT EXISTS "{self.schema_name}";',
            "",
        ]
        
        for table in self.tables:
            lines.append(table.to_sql(self.schema_name))
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "schema_name": self.schema_name,
            "vertical": self.vertical,
            "tables": [
                {
                    "name": t.name,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.data_type,
                            "nullable": c.nullable,
                            "is_pk": c.is_primary_key,
                            "is_fk": c.is_foreign_key,
                        }
                        for c in t.columns
                    ],
                }
                for t in self.tables
            ],
            "seed": self.seed,
        }


# =============================================================================
# SCHEMA GENERATOR
# =============================================================================

class SchemaGenerator:
    """Generates random PostgreSQL schemas for testing."""
    
    def __init__(self, seed: int | None = None):
        """Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed or random.randint(0, 999999)
        self.rng = random.Random(self.seed)
        self.fake = Faker("it_IT")
        self.fake.seed_instance(self.seed)
    
    def generate(
        self,
        vertical: str | None = None,
        num_tables: int | None = None,
        columns_per_table: tuple[int, int] = (5, 20),
        include_relationships: bool = True,
    ) -> DatabaseSchema:
        """Generate a random database schema.
        
        Args:
            vertical: Domain vertical (restaurant, hotel, etc.). Random if None.
            num_tables: Number of tables. Random 5-15 if None.
            columns_per_table: Min/max columns per table.
            include_relationships: Whether to add foreign keys.
        
        Returns:
            Complete DatabaseSchema
        """
        # Select vertical
        if vertical is None:
            vertical = self.rng.choice(list(DOMAIN_TEMPLATES.keys()))
        
        template = DOMAIN_TEMPLATES.get(vertical, DOMAIN_TEMPLATES["restaurant"])
        
        # Determine number of tables
        if num_tables is None:
            num_tables = self.rng.randint(5, 15)
        
        num_tables = min(num_tables, len(template["tables"]))
        
        # Select tables
        selected_tables = self.rng.sample(template["tables"], num_tables)
        
        # Generate schema name
        schema_name = f"test_{vertical}_{self.seed}"
        
        # Create tables
        tables = []
        for table_name in selected_tables:
            table = self._generate_table(
                table_name,
                columns_per_table,
                template["common_columns"],
            )
            tables.append(table)
        
        # Add relationships
        if include_relationships and len(tables) > 1:
            self._add_relationships(tables)
        
        return DatabaseSchema(
            schema_name=schema_name,
            vertical=vertical,
            tables=tables,
            seed=self.seed,
        )
    
    def _generate_table(
        self,
        name: str,
        columns_range: tuple[int, int],
        common_columns: list[str],
    ) -> TableSchema:
        """Generate a single table schema."""
        columns = []
        
        # Primary key
        columns.append(ColumnSchema(
            name="id",
            data_type="SERIAL",
            nullable=False,
            is_primary_key=True,
        ))
        
        # Common columns from template
        for col_name in common_columns[:3]:  # Use first 3 common columns
            col_type = self._get_type_for_column(col_name)
            columns.append(ColumnSchema(
                name=col_name,
                data_type=col_type,
                nullable=col_name not in ["name", "created_at"],
            ))
        
        # Random additional columns
        num_extra = self.rng.randint(columns_range[0], columns_range[1]) - len(columns)
        
        for _ in range(max(0, num_extra)):
            col_name = self._generate_column_name()
            col_type = self._random_column_type()
            
            # Avoid duplicates
            existing_names = {c.name for c in columns}
            if col_name in existing_names:
                col_name = f"{col_name}_{self.rng.randint(1, 99)}"
            
            columns.append(ColumnSchema(
                name=col_name,
                data_type=col_type,
                nullable=self.rng.random() > 0.3,  # 70% nullable
            ))
        
        return TableSchema(name=name, columns=columns)
    
    def _get_type_for_column(self, col_name: str) -> str:
        """Infer column type from name."""
        name_lower = col_name.lower()
        
        if "id" in name_lower:
            return "INTEGER"
        elif "price" in name_lower or "rate" in name_lower or "amount" in name_lower:
            return "DECIMAL(10,2)"
        elif "is_" in name_lower or "has_" in name_lower:
            return "BOOLEAN"
        elif "date" in name_lower:
            return "DATE"
        elif "created" in name_lower or "updated" in name_lower or "time" in name_lower:
            return "TIMESTAMP"
        elif "description" in name_lower or "notes" in name_lower:
            return "TEXT"
        elif "count" in name_lower or "quantity" in name_lower:
            return "INTEGER"
        elif "data" in name_lower or "meta" in name_lower or "config" in name_lower:
            return "JSONB"
        else:
            return "VARCHAR(255)"
    
    def _random_column_type(self) -> str:
        """Select a random column type with weights."""
        types = list(COLUMN_TYPES.keys())
        weights = list(COLUMN_TYPES.values())
        return self.rng.choices(types, weights=weights, k=1)[0]
    
    def _generate_column_name(self) -> str:
        """Generate a realistic column name."""
        prefixes = [
            "total", "avg", "min", "max", "is", "has", "num", "count",
            "first", "last", "primary", "secondary", "external",
        ]
        
        nouns = [
            "value", "amount", "status", "type", "code", "number",
            "date", "time", "level", "score", "rating", "priority",
            "quantity", "size", "weight", "length", "width", "height",
            "url", "email", "phone", "address", "notes", "comment",
        ]
        
        # Sometimes use prefix
        if self.rng.random() > 0.6:
            return f"{self.rng.choice(prefixes)}_{self.rng.choice(nouns)}"
        else:
            return self.rng.choice(nouns)
    
    def _add_relationships(self, tables: list[TableSchema]) -> None:
        """Add foreign key relationships between tables."""
        # Add 1-3 relationships
        num_relationships = self.rng.randint(1, min(3, len(tables) - 1))
        
        for _ in range(num_relationships):
            # Pick parent and child tables
            parent_idx = self.rng.randint(0, len(tables) - 1)
            child_idx = self.rng.randint(0, len(tables) - 1)
            
            if parent_idx == child_idx:
                continue
            
            parent = tables[parent_idx]
            child = tables[child_idx]
            
            # Check if FK already exists
            fk_name = f"{parent.name}_id"
            if any(c.name == fk_name for c in child.columns):
                continue
            
            # Add foreign key column
            child.columns.append(ColumnSchema(
                name=fk_name,
                data_type="INTEGER",
                nullable=True,
                is_foreign_key=True,
                references_table=parent.name,
                references_column="id",
            ))


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    gen = SchemaGenerator(seed=42)
    schema = gen.generate(vertical="restaurant", num_tables=5)
    
    print(f"Generated schema: {schema.schema_name}")
    print(f"Vertical: {schema.vertical}")
    print(f"Tables: {len(schema.tables)}")
    print()
    print(schema.to_sql())
