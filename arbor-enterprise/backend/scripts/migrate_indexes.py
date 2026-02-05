"""Database Index Migrations.

TIER 2 - Point 5: Advanced Database Indexing Strategy

Creates optimized indexes for:
- PostgreSQL GIN indexes for JSONB (vibe_dna)
- PostgreSQL GIST indexes for geospatial (location)
- Neo4j constraints and indexes
"""

import asyncio
import logging
from typing import Any

import asyncpg
from neo4j import AsyncGraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# PostgreSQL Index Migrations
# ═══════════════════════════════════════════════════════════════════════════

POSTGRES_INDEXES = [
    # TIER 2 - Point 5: GIN index for JSONB vibe_dna
    {
        "name": "idx_enrichment_vibe_gin",
        "table": "arbor_enrichment",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_vibe_gin 
            ON arbor_enrichment USING GIN (vibe_dna jsonb_path_ops);
        """,
        "description": "GIN index for fast JSONB vibe_dna queries",
    },
    
    # TIER 2 - Point 5: GIST index for geospatial queries
    {
        "name": "idx_entity_geo",
        "table": "arbor_entity",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_geo 
            ON arbor_entity USING GIST (
                ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            ) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        """,
        "description": "GIST index for geospatial queries",
    },
    
    # Category + city composite for filtered searches
    {
        "name": "idx_entity_category_city",
        "table": "arbor_entity",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_category_city 
            ON arbor_entity (category, city) 
            WHERE deleted_at IS NULL;
        """,
        "description": "Composite index for category + city filters",
    },
    
    # Timestamp index for keyset pagination
    {
        "name": "idx_entity_pagination",
        "table": "arbor_entity",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_pagination 
            ON arbor_entity (created_at DESC, id DESC)
            WHERE deleted_at IS NULL;
        """,
        "description": "Index for efficient keyset pagination",
    },
    
    # Full-text search index
    {
        "name": "idx_entity_search",
        "table": "arbor_entity",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_search 
            ON arbor_entity USING GIN (
                to_tsvector('italian', coalesce(name, '') || ' ' || coalesce(description, ''))
            );
        """,
        "description": "GIN index for full-text search",
    },
    
    # Enrichment entity_id foreign key
    {
        "name": "idx_enrichment_entity_id",
        "table": "arbor_enrichment",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_entity_id 
            ON arbor_enrichment (entity_id);
        """,
        "description": "Index for enrichment lookups by entity",
    },
    
    # Last indexed for CDC/refresh queries
    {
        "name": "idx_enrichment_last_indexed",
        "table": "arbor_enrichment",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enrichment_last_indexed 
            ON arbor_enrichment (last_indexed)
            WHERE last_indexed IS NOT NULL;
        """,
        "description": "Index for finding stale embeddings",
    },
    
    # Audit log timestamp for retention
    {
        "name": "idx_audit_log_timestamp",
        "table": "audit_log",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_timestamp 
            ON audit_log (created_at DESC);
        """,
        "description": "Index for audit log queries and retention",
    },
    
    # Outbox for CDC relay
    {
        "name": "idx_outbox_unprocessed",
        "table": "outbox_events",
        "sql": """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outbox_unprocessed 
            ON outbox_events (created_at)
            WHERE processed_at IS NULL;
        """,
        "description": "Index for efficient outbox processing",
    },
]


async def apply_postgres_indexes() -> dict[str, Any]:
    """Apply all PostgreSQL indexes."""
    conn = await asyncpg.connect(settings.arbor_database_url)
    results = {"success": [], "failed": [], "skipped": []}
    
    try:
        for index in POSTGRES_INDEXES:
            try:
                # Check if index exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = $1
                    )
                """, index["name"])
                
                if exists:
                    results["skipped"].append(index["name"])
                    logger.info(f"Index {index['name']} already exists")
                    continue
                
                # Create index
                await conn.execute(index["sql"])
                results["success"].append(index["name"])
                logger.info(f"Created index: {index['name']}")
            
            except Exception as e:
                results["failed"].append({"name": index["name"], "error": str(e)})
                logger.error(f"Failed to create index {index['name']}: {e}")
    
    finally:
        await conn.close()
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Neo4j Index Migrations
# ═══════════════════════════════════════════════════════════════════════════

NEO4J_CONSTRAINTS = [
    # TIER 2 - Point 5: Unique constraint on Entity.id
    {
        "name": "constraint_entity_id_unique",
        "cypher": """
            CREATE CONSTRAINT constraint_entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """,
        "description": "Unique constraint on Entity.id",
    },
    
    # Unique constraint on Brand.id
    {
        "name": "constraint_brand_id_unique",
        "cypher": """
            CREATE CONSTRAINT constraint_brand_id_unique IF NOT EXISTS
            FOR (b:Brand) REQUIRE b.id IS UNIQUE
        """,
        "description": "Unique constraint on Brand.id",
    },
    
    # Unique constraint on Venue.id
    {
        "name": "constraint_venue_id_unique",
        "cypher": """
            CREATE CONSTRAINT constraint_venue_id_unique IF NOT EXISTS
            FOR (v:Venue) REQUIRE v.id IS UNIQUE
        """,
        "description": "Unique constraint on Venue.id",
    },
]

NEO4J_INDEXES = [
    # TIER 2 - Point 5: Category index
    {
        "name": "index_entity_category",
        "cypher": """
            CREATE INDEX index_entity_category IF NOT EXISTS
            FOR (e:Entity) ON (e.category)
        """,
        "description": "Index on Entity.category",
    },
    
    # City index
    {
        "name": "index_entity_city",
        "cypher": """
            CREATE INDEX index_entity_city IF NOT EXISTS
            FOR (e:Entity) ON (e.city)
        """,
        "description": "Index on Entity.city",
    },
    
    # TIER 2 - Point 5: Full-text search index
    {
        "name": "fulltext_entity_search",
        "cypher": """
            CREATE FULLTEXT INDEX fulltext_entity_search IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name, n.description]
        """,
        "description": "Full-text search index on Entity name and description",
    },
    
    # Vibe index for filtering
    {
        "name": "index_entity_vibe",
        "cypher": """
            CREATE INDEX index_entity_vibe IF NOT EXISTS
            FOR (e:Entity) ON (e.primary_vibe)
        """,
        "description": "Index on Entity.primary_vibe",
    },
    
    # Relationship type index
    {
        "name": "index_relationship_type",
        "cypher": """
            CREATE INDEX index_relationship_type IF NOT EXISTS
            FOR ()-[r:RELATED_TO]-() ON (r.relationship_type)
        """,
        "description": "Index on relationship type",
    },
]


async def apply_neo4j_indexes() -> dict[str, Any]:
    """Apply all Neo4j constraints and indexes."""
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    
    results = {"constraints": [], "indexes": [], "failed": []}
    
    try:
        async with driver.session() as session:
            # Apply constraints
            for constraint in NEO4J_CONSTRAINTS:
                try:
                    await session.run(constraint["cypher"])
                    results["constraints"].append(constraint["name"])
                    logger.info(f"Applied constraint: {constraint['name']}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Constraint {constraint['name']} already exists")
                    else:
                        results["failed"].append({"name": constraint["name"], "error": str(e)})
                        logger.error(f"Failed constraint {constraint['name']}: {e}")
            
            # Apply indexes
            for index in NEO4J_INDEXES:
                try:
                    await session.run(index["cypher"])
                    results["indexes"].append(index["name"])
                    logger.info(f"Applied index: {index['name']}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Index {index['name']} already exists")
                    else:
                        results["failed"].append({"name": index["name"], "error": str(e)})
                        logger.error(f"Failed index {index['name']}: {e}")
    
    finally:
        await driver.close()
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main Migration Runner
# ═══════════════════════════════════════════════════════════════════════════

async def run_all_migrations() -> dict[str, Any]:
    """Run all database index migrations."""
    logger.info("Starting database index migrations...")
    
    results = {
        "postgres": {},
        "neo4j": {},
    }
    
    # PostgreSQL indexes
    try:
        results["postgres"] = await apply_postgres_indexes()
        logger.info(f"PostgreSQL: {len(results['postgres'].get('success', []))} indexes created")
    except Exception as e:
        results["postgres"] = {"error": str(e)}
        logger.error(f"PostgreSQL migrations failed: {e}")
    
    # Neo4j indexes
    try:
        results["neo4j"] = await apply_neo4j_indexes()
        logger.info(f"Neo4j: {len(results['neo4j'].get('indexes', []))} indexes created")
    except Exception as e:
        results["neo4j"] = {"error": str(e)}
        logger.error(f"Neo4j migrations failed: {e}")
    
    logger.info("Database index migrations complete")
    return results


# CLI entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_all_migrations())
    print(f"\nMigration Results: {result}")
