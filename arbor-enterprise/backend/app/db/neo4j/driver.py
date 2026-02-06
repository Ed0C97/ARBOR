"""Neo4j async driver management.

TIER D1: Connection pooling with configurable pool size.
- max_connection_pool_size controls concurrent connections
- connection_acquisition_timeout prevents hanging under load
"""

from neo4j import AsyncGraphDatabase

from app.config import get_settings

settings = get_settings()

_driver = None


def get_neo4j_driver():
    """Return singleton Neo4j driver, or None if not configured.

    Uses connection pooling with max_connection_pool_size=50 and
    a 30s acquisition timeout to prevent connection starvation.
    """
    global _driver
    # Skip if Neo4j is not configured
    if not settings.neo4j_uri:
        return None
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30.0,
            max_transaction_retry_time=15.0,
        )
    return _driver


async def close_neo4j_driver():
    """Close the Neo4j driver on shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def init_neo4j_schema():
    """Initialize Neo4j constraints and indexes."""
    driver = get_neo4j_driver()
    if driver is None:
        return  # Neo4j not configured, skip

    constraints = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT abstract_id IF NOT EXISTS FOR (a:AbstractEntity) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT curator_id IF NOT EXISTS FOR (c:Curator) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT style_name IF NOT EXISTS FOR (s:Style) REQUIRE s.name IS UNIQUE",
    ]

    indexes = [
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_category IF NOT EXISTS FOR (e:Entity) ON (e.category)",
        "CREATE INDEX entity_city IF NOT EXISTS FOR (e:Entity) ON (e.city)",
        "CREATE INDEX abstract_name IF NOT EXISTS FOR (a:AbstractEntity) ON (a.name)",
    ]

    async with driver.session() as session:
        for stmt in constraints + indexes:
            await session.run(stmt)


async def check_neo4j_health() -> bool:
    """Health check for Neo4j service.

    TIER 5 - Point 22: Deep Health Checks
    Returns True if Neo4j is reachable and responsive.
    """
    driver = get_neo4j_driver()
    if driver is None:
        return False

    try:
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ping")
            record = await result.single()
            return record is not None and record["ping"] == 1
    except Exception:
        return False
