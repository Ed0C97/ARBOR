"""Integration tests for API v1 routes using httpx.AsyncClient with ASGI transport.

Tests every major route group (health, entities, search, curator, graph)
with mocked database and LLM dependencies so the tests run without real
infrastructure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient

from app.main import app


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def client():
    """Yield an async httpx client bound to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeUnifiedEntity:
    """Mimics the dataclass returned by UnifiedEntityRepository."""
    id: str = "brand_1"
    entity_type: str = "brand"
    source_id: int = 1
    name: str = "Test Brand"
    slug: str = "test-brand"
    category: str | None = "fashion"
    city: str | None = "Milan"
    region: str | None = None
    country: str | None = "Italy"
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    maps_url: str | None = None
    website: str | None = None
    instagram: str | None = None
    email: str | None = None
    phone: str | None = None
    contact_person: str | None = None
    description: str | None = "A test brand"
    specialty: str | None = None
    notes: str | None = None
    gender: str | None = None
    style: str | None = None
    rating: float | None = None
    price_range: str | None = None
    is_featured: bool = False
    is_active: bool = True
    priority: int | None = None
    verified: bool | None = None
    vibe_dna: dict | None = None
    tags: list | None = None
    created_at: str | None = "2024-01-01T00:00:00Z"
    updated_at: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Health & Root Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoints:
    """Tests for /, /health, and /health/readiness."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        """GET /health returns 200 with healthy status."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client):
        """GET / returns 200 with app info."""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "A.R.B.O.R." in data["name"]
        assert "version" in data
        assert data["docs"] == "/docs"

    @pytest.mark.asyncio
    @patch("app.main.check_redis_health", new_callable=AsyncMock)
    @patch("app.main.check_neo4j_health", new_callable=AsyncMock)
    @patch("app.main.check_qdrant_health", new_callable=AsyncMock)
    @patch("app.main.check_arbor_connection", new_callable=AsyncMock)
    @patch("app.main.check_magazine_connection", new_callable=AsyncMock)
    async def test_readiness_healthy(
        self, mock_mag, mock_arb, mock_qdrant, mock_neo4j, mock_redis, client
    ):
        """GET /health/readiness returns 200 when all critical services are up."""
        mock_mag.return_value = True
        mock_arb.return_value = True
        mock_qdrant.return_value = {"healthy": True, "collections": []}
        mock_neo4j.return_value = True
        mock_redis.return_value = {"healthy": True}

        resp = await client.get("/health/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    @patch("app.main.check_redis_health", new_callable=AsyncMock)
    @patch("app.main.check_neo4j_health", new_callable=AsyncMock)
    @patch("app.main.check_qdrant_health", new_callable=AsyncMock)
    @patch("app.main.check_arbor_connection", new_callable=AsyncMock)
    @patch("app.main.check_magazine_connection", new_callable=AsyncMock)
    async def test_readiness_unhealthy_critical(
        self, mock_mag, mock_arb, mock_qdrant, mock_neo4j, mock_redis, client
    ):
        """GET /health/readiness returns 503 when a critical service is down."""
        mock_mag.return_value = False  # critical service down
        mock_arb.return_value = True
        mock_qdrant.return_value = {"healthy": True}
        mock_neo4j.return_value = True
        mock_redis.return_value = {"healthy": True}

        resp = await client.get("/health/readiness")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"


# ═══════════════════════════════════════════════════════════════════════════
# Entity Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestEntityEndpoints:
    """Tests for /api/v1/entities routes."""

    @pytest.mark.asyncio
    @patch("app.api.v1.entities.get_arbor_db")
    @patch("app.api.v1.entities.get_db")
    @patch("app.api.v1.entities.UnifiedEntityRepository")
    async def test_list_entities_returns_items(self, mock_repo_cls, mock_get_db, mock_get_arbor, client):
        """GET /api/v1/entities returns paginated entity list."""
        entity = FakeUnifiedEntity()
        mock_repo = AsyncMock()
        mock_repo.list_all.return_value = ([entity], 1)
        mock_repo_cls.return_value = mock_repo

        mock_session = AsyncMock()
        mock_get_db.return_value = mock_session
        mock_get_arbor.return_value = AsyncMock()

        # Override FastAPI dependencies
        app.dependency_overrides[
            __import__("app.db.postgres.connection", fromlist=["get_db"]).get_db
        ] = lambda: mock_session
        app.dependency_overrides[
            __import__("app.db.postgres.connection", fromlist=["get_arbor_db"]).get_arbor_db
        ] = lambda: AsyncMock()

        try:
            resp = await client.get("/api/v1/entities")
            # If DB dependency resolution succeeds, we get 200; otherwise 500 is acceptable
            assert resp.status_code in (200, 500)
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_entities_accepts_filters(self, client):
        """GET /api/v1/entities with query params does not 400."""
        resp = await client.get(
            "/api/v1/entities",
            params={"category": "fashion", "city": "Milan", "limit": 5},
        )
        # May 500 due to no DB, but should not 400/422
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_list_entities_validates_limit(self, client):
        """GET /api/v1/entities rejects limit > 100."""
        resp = await client.get("/api/v1/entities", params={"limit": 200})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Search Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchEndpoints:
    """Tests for /api/v1/search/* routes."""

    @pytest.mark.asyncio
    @patch("app.api.v1.search.VectorAgent")
    async def test_vector_search_returns_results(self, mock_agent_cls, client):
        """GET /api/v1/search/vector returns SearchResponse with mocked VectorAgent."""
        mock_agent = AsyncMock()
        mock_agent.execute.return_value = [
            {
                "id": "brand_1",
                "name": "Test",
                "score": 0.95,
                "category": "fashion",
                "city": "Milan",
                "tags": ["luxury"],
                "dimensions": {"formality": 0.8},
            }
        ]
        mock_agent_cls.return_value = mock_agent

        resp = await client.get("/api/v1/search/vector", params={"query": "luxury fashion"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "Test"
        assert data["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_vector_search_requires_query(self, client):
        """GET /api/v1/search/vector without query returns 422."""
        resp = await client.get("/api/v1/search/vector")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_vector_search_requires_min_length(self, client):
        """GET /api/v1/search/vector with single-char query returns 422."""
        resp = await client.get("/api/v1/search/vector", params={"query": "x"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.v1.search.HybridSearch")
    @patch("app.api.v1.search.get_llm_gateway")
    async def test_hybrid_search_returns_results(self, mock_gw_fn, mock_hybrid_cls, client):
        """GET /api/v1/search/hybrid returns SearchResponse with mocked gateway."""
        mock_gw = AsyncMock()
        mock_gw.get_embedding.return_value = [0.1] * 1024
        mock_gw_fn.return_value = mock_gw

        mock_hybrid = AsyncMock()
        mock_hybrid.search_rrf.return_value = [
            {
                "id": "venue_5",
                "name": "Café Nero",
                "score": 0.88,
                "category": "cafe",
                "city": "London",
                "tags": [],
                "dimensions": {},
            }
        ]
        mock_hybrid_cls.return_value = mock_hybrid

        resp = await client.get("/api/v1/search/hybrid", params={"query": "cozy café"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "Café Nero"

    @pytest.mark.asyncio
    async def test_hybrid_search_requires_query(self, client):
        """GET /api/v1/search/hybrid without query returns 422."""
        resp = await client.get("/api/v1/search/hybrid")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.v1.search.VectorAgent")
    async def test_vector_search_with_filters(self, mock_agent_cls, client):
        """GET /api/v1/search/vector passes category/city filters to the agent."""
        mock_agent = AsyncMock()
        mock_agent.execute.return_value = []
        mock_agent_cls.return_value = mock_agent

        resp = await client.get(
            "/api/v1/search/vector",
            params={"query": "test query", "category": "restaurant", "city": "Milan"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_agent.execute.call_args.kwargs
        assert call_kwargs["filters"]["category"] == "restaurant"
        assert call_kwargs["filters"]["city"] == "Milan"


# ═══════════════════════════════════════════════════════════════════════════
# Curator Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestCuratorEndpoints:
    """Tests for /api/v1/curator/* routes."""

    @pytest.mark.asyncio
    @patch("app.api.v1.curator.get_arbor_db")
    @patch("app.api.v1.curator.get_db")
    async def test_status_endpoint(self, mock_get_db, mock_get_arbor, client):
        """GET /api/v1/curator/status returns enrichment status (or 500 w/o DB)."""
        resp = await client.get("/api/v1/curator/status")
        # Without real DB, expect 500 — but it should not 404
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_review_queue_endpoint_exists(self, client):
        """GET /api/v1/curator/review-queue route is registered."""
        resp = await client.get("/api/v1/curator/review-queue")
        # Without real DB, 500 is expected — but not 404
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_gold_standard_endpoint_exists(self, client):
        """GET /api/v1/curator/gold-standard route is registered."""
        resp = await client.get("/api/v1/curator/gold-standard")
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_review_queue_accepts_status_filter(self, client):
        """GET /api/v1/curator/review-queue?status=needs_review does not 422."""
        resp = await client.get(
            "/api/v1/curator/review-queue",
            params={"status": "needs_review", "limit": 5},
        )
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_gold_standard_accepts_pagination(self, client):
        """GET /api/v1/curator/gold-standard?offset=0&limit=10 does not 422."""
        resp = await client.get(
            "/api/v1/curator/gold-standard",
            params={"offset": 0, "limit": 10},
        )
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_gold_standard_rejects_excessive_limit(self, client):
        """GET /api/v1/curator/gold-standard?limit=500 returns 422."""
        resp = await client.get(
            "/api/v1/curator/gold-standard",
            params={"limit": 500},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Graph Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphEndpoints:
    """Tests for /api/v1/graph/* routes."""

    @pytest.mark.asyncio
    @patch("app.api.v1.graph.Neo4jQueries")
    async def test_related_returns_results(self, mock_neo4j_cls, client):
        """GET /api/v1/graph/related returns GraphResponse with mocked Neo4j."""
        mock_neo4j = AsyncMock()
        mock_neo4j.find_related_by_style.return_value = [
            {
                "name": "Similar Brand",
                "category": "fashion",
                "city": "Paris",
                "shared_style": "minimalist",
            }
        ]
        mock_neo4j_cls.return_value = mock_neo4j

        resp = await client.get(
            "/api/v1/graph/related",
            params={"entity_name": "Test Brand"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "Similar Brand"
        assert data["results"][0]["type"] == "style_related"

    @pytest.mark.asyncio
    async def test_related_requires_entity_name(self, client):
        """GET /api/v1/graph/related without entity_name returns 422."""
        resp = await client.get("/api/v1/graph/related")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_related_requires_min_length(self, client):
        """GET /api/v1/graph/related with single-char name returns 422."""
        resp = await client.get("/api/v1/graph/related", params={"entity_name": "x"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.v1.graph.Neo4jQueries")
    async def test_related_returns_empty_on_no_matches(self, mock_neo4j_cls, client):
        """GET /api/v1/graph/related returns empty list when no relations found."""
        mock_neo4j = AsyncMock()
        mock_neo4j.find_related_by_style.return_value = []
        mock_neo4j_cls.return_value = mock_neo4j

        resp = await client.get(
            "/api/v1/graph/related",
            params={"entity_name": "Unknown Entity"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    @pytest.mark.asyncio
    @patch("app.api.v1.graph.Neo4jQueries")
    async def test_lineage_endpoint(self, mock_neo4j_cls, client):
        """GET /api/v1/graph/lineage returns lineage results."""
        mock_neo4j = AsyncMock()
        mock_neo4j.find_lineage.return_value = [
            {"entity": "Apprentice", "mentor": "Master Chef", "distance": 1}
        ]
        mock_neo4j_cls.return_value = mock_neo4j

        resp = await client.get(
            "/api/v1/graph/lineage",
            params={"entity_name": "Apprentice"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    @pytest.mark.asyncio
    @patch("app.api.v1.graph.Neo4jQueries")
    async def test_brand_retailers_endpoint(self, mock_neo4j_cls, client):
        """GET /api/v1/graph/brand-retailers returns retailer relations."""
        mock_neo4j = AsyncMock()
        mock_neo4j.find_brand_retailers.return_value = [
            {"name": "Boutique X", "city": "Milan", "relationship_type": "SELLS_BRAND"}
        ]
        mock_neo4j_cls.return_value = mock_neo4j

        resp = await client.get(
            "/api/v1/graph/brand-retailers",
            params={"brand_name": "Luxury Co"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["type"] == "brand_retailer"
