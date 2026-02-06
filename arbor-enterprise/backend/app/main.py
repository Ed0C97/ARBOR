"""A.R.B.O.R. Enterprise - FastAPI Application Entry Point.

TIER 2 - Point 7: Compiled Graph Singleton Pattern
- Pre-compile LangGraph at startup to avoid 200ms overhead per request

TIER 2 - Point 9: Asyncio.gather Service Initialization
- Concurrent startup of all services (DB, Qdrant, Neo4j, Redis)
- Reduces startup time from 15s+ to <5s
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.graphql_schema import create_graphql_app
from app.api.v1 import admin, curator, discover, entities, graph, search
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def _init_postgres():
    """Initialize PostgreSQL tables."""
    from app.db.postgres.connection import create_arbor_tables

    await create_arbor_tables()
    logger.info("PostgreSQL: ARBOR tables ready")


async def _init_qdrant():
    """Initialize Qdrant collections."""
    from app.db.qdrant.client import init_qdrant_collections

    await init_qdrant_collections()
    logger.info("Qdrant: Collections initialized")


async def _init_neo4j():
    """Initialize Neo4j schema."""
    from app.db.neo4j.driver import init_neo4j_schema

    await init_neo4j_schema()
    logger.info("Neo4j: Schema initialized")


async def _init_redis():
    """Initialize Redis connection."""
    from app.db.redis.client import get_redis_client

    client = await get_redis_client()
    if client:
        await client.ping()
        logger.info("Redis: Connection verified")
    else:
        logger.warning("Redis: Not configured or unavailable")


async def _compile_agent_graph(app: FastAPI):
    """Pre-compile LangGraph agent graph.

    TIER 2 - Point 7: Compiled Graph Singleton Pattern
    Pre-compiling avoids 200ms overhead per request.
    """
    from app.agents.graph import create_agent_graph

    # Compile the graph once at startup
    compiled_graph = create_agent_graph()
    app.state.compiled_graph = compiled_graph
    logger.info("LangGraph: Agent graph compiled")


async def _warmup_cache():
    """TIER D3: Initialize multi-tier cache singleton at startup.

    Pre-creates the L1/L2/L3 cache layers and bloom filter so the first
    request doesn't pay initialization overhead (~200ms).
    """
    from app.core.multi_tier_cache import get_multi_tier_cache

    cache = await get_multi_tier_cache()
    logger.info("Cache warmup: MultiTierCache initialized (%s)", cache.stats()["bloom_filter"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown.

    TIER 2 - Point 9: Concurrent service initialization
    Uses asyncio.gather for parallel startup (<5s total).
    """
    start_time = time.time()
    logger.info("Starting A.R.B.O.R. Enterprise...")

    # TIER 2 - Point 9: Concurrent initialization
    init_tasks = [
        _safe_init("PostgreSQL", _init_postgres()),
        _safe_init("Qdrant", _init_qdrant()),
        _safe_init("Neo4j", _init_neo4j()),
        _safe_init("Redis", _init_redis()),
    ]

    await asyncio.gather(*init_tasks)

    # TIER 2 - Point 7: Compile agent graph
    try:
        await _compile_agent_graph(app)
    except Exception as e:
        logger.warning(f"Agent graph compilation failed: {e}")

    # TIER D3: Cache warmup — initialize multi-tier cache singleton
    await _safe_init("CacheWarmup", _warmup_cache())

    elapsed = time.time() - start_time
    logger.info(f"A.R.B.O.R. Enterprise started in {elapsed:.2f}s")

    yield

    # Shutdown
    logger.info("Shutting down A.R.B.O.R. Enterprise...")

    # Concurrent cleanup
    from app.db.neo4j.driver import close_neo4j_driver
    from app.db.postgres.connection import close_all_connections
    from app.db.qdrant.client import close_qdrant_client
    from app.db.redis.client import close_redis_client
    from app.llm.gateway import close_cohere_client

    shutdown_tasks = [
        _safe_shutdown("Neo4j", close_neo4j_driver()),
        _safe_shutdown("Redis", close_redis_client()),
        _safe_shutdown("Qdrant", close_qdrant_client()),
        _safe_shutdown("PostgreSQL", close_all_connections()),
        _safe_shutdown("Cohere", close_cohere_client()),
    ]

    await asyncio.gather(*shutdown_tasks)
    logger.info("A.R.B.O.R. Enterprise shutdown complete")


async def _safe_init(name: str, coro):
    """Safely execute initialization, logging any failures."""
    try:
        await coro
    except Exception as e:
        logger.warning(f"{name} init failed (may not be running): {e}")


async def _safe_shutdown(name: str, coro):
    """Safely execute shutdown, logging any failures."""
    try:
        await coro
    except Exception as e:
        logger.warning(f"{name} shutdown error: {e}")


app = FastAPI(
    title="A.R.B.O.R. Enterprise API",
    description="Advanced Reasoning By Ontological Rules - Contextual Discovery Engine",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 routes
app.include_router(discover.router, prefix="/api/v1", tags=["Discovery"])
app.include_router(entities.router, prefix="/api/v1", tags=["Entities"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(graph.router, prefix="/api/v1", tags=["Graph"])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])
app.include_router(curator.router, prefix="/api/v1", tags=["Curator"])

# GraphQL (Strawberry) — mirrors REST endpoints with flexible queries
graphql_app = create_graphql_app()
app.include_router(graphql_app, prefix="/graphql", tags=["GraphQL"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.app_version, "service": "arbor-api"}


@app.get("/health/readiness")
async def readiness():
    """Deep health check endpoint.

    TIER 5 - Point 22: Deep Health Checks
    Verifies actual connectivity to all backend services.
    Returns 503 if a critical service is down.
    """
    from fastapi.responses import JSONResponse

    from app.db.neo4j.driver import check_neo4j_health
    from app.db.postgres.connection import (
        check_arbor_connection,
        check_magazine_connection,
    )
    from app.db.qdrant.client import check_qdrant_health
    from app.db.redis.client import check_redis_health

    checks = {}
    critical_failures = []

    # PostgreSQL (Magazine DB - critical)
    try:
        magazine_ok = await check_magazine_connection()
        checks["postgres_magazine"] = {"healthy": magazine_ok, "critical": True}
        if not magazine_ok:
            critical_failures.append("postgres_magazine")
    except Exception as e:
        checks["postgres_magazine"] = {"healthy": False, "critical": True, "error": str(e)}
        critical_failures.append("postgres_magazine")

    # PostgreSQL (ARBOR DB - critical)
    try:
        arbor_ok = await check_arbor_connection()
        checks["postgres_arbor"] = {"healthy": arbor_ok, "critical": True}
        if not arbor_ok:
            critical_failures.append("postgres_arbor")
    except Exception as e:
        checks["postgres_arbor"] = {"healthy": False, "critical": True, "error": str(e)}
        critical_failures.append("postgres_arbor")

    # Qdrant (critical for search)
    try:
        qdrant_status = await check_qdrant_health()
        checks["qdrant"] = {"healthy": qdrant_status.get("healthy", False), "critical": True}
        if not qdrant_status.get("healthy"):
            critical_failures.append("qdrant")
    except Exception as e:
        checks["qdrant"] = {"healthy": False, "critical": True, "error": str(e)}
        critical_failures.append("qdrant")

    # Neo4j (non-critical, app can degrade)
    try:
        neo4j_ok = await check_neo4j_health()
        checks["neo4j"] = {"healthy": neo4j_ok, "critical": False}
    except Exception as e:
        checks["neo4j"] = {"healthy": False, "critical": False, "error": str(e)}

    # Redis (non-critical, caching degrades gracefully)
    try:
        redis_status = await check_redis_health()
        checks["redis"] = {"healthy": redis_status.get("healthy", False), "critical": False}
    except Exception as e:
        checks["redis"] = {"healthy": False, "critical": False, "error": str(e)}

    # Overall status
    overall_healthy = len(critical_failures) == 0
    status_code = 200 if overall_healthy else 503

    response_data = {
        "status": "ready" if overall_healthy else "not_ready",
        "version": settings.app_version,
        "checks": checks,
    }

    if critical_failures:
        response_data["critical_failures"] = critical_failures

    return JSONResponse(content=response_data, status_code=status_code)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "A.R.B.O.R. Enterprise",
        "description": "Contextual Discovery Engine",
        "version": settings.app_version,
        "docs": "/docs",
    }


# ═══════════════════════════════════════════════════════════════════════════
# TIER D4: WebSocket for real-time entity updates
# ═══════════════════════════════════════════════════════════════════════════


class ConnectionManager:
    """Manages active WebSocket connections for real-time broadcasts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected (total=%d)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected (total=%d)", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        payload = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


ws_manager = ConnectionManager()


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """Real-time entity update stream via WebSocket.

    TIER D4: Clients connect here to receive live notifications for
    entity changes, curator decisions, and feedback events.

    Messages are JSON with structure:
        {"event": "entity_updated", "data": {...}, "timestamp": "..."}

    Clients can also send subscription filters:
        {"subscribe": ["entity_updated", "curator_decision"]}
    """
    await ws_manager.connect(websocket)
    subscriptions: set[str] = set()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if "subscribe" in msg and isinstance(msg["subscribe"], list):
                    subscriptions = set(msg["subscribe"])
                    await websocket.send_text(
                        json.dumps(
                            {
                                "event": "subscribed",
                                "channels": list(subscriptions),
                            }
                        )
                    )
                elif msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


def get_ws_manager() -> ConnectionManager:
    """Return the global WebSocket connection manager for broadcasting."""
    return ws_manager
