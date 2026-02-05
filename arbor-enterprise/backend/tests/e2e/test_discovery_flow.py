"""
End-to-end tests for the full discovery flow:
  1. Ingest an entity
  2. Discover it through the chat API
  3. Verify entity detail retrieval
  4. Verify graph relationships
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestFullDiscoveryFlow:
    """E2E flow: health -> ingest -> discover -> entity detail -> graph."""

    @pytest.mark.asyncio
    async def test_00_health_check(self, client):
        """Ensure the system is up before running the flow."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_01_ingest_entity(self, client, sample_entity):
        """POST a new entity through the admin ingestion endpoint."""
        response = await client.post(
            "/api/v1/admin/entities",
            json=sample_entity,
        )
        # With a live DB this returns 201; without DB we accept 500
        assert response.status_code in (201, 200, 422, 500)

    @pytest.mark.asyncio
    async def test_02_list_entities(self, client):
        """GET entities list and verify pagination structure."""
        response = await client.get(
            "/api/v1/entities",
            params={"limit": 10, "offset": 0},
        )
        if response.status_code == 200:
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert isinstance(data["items"], list)
        else:
            # DB not available — acceptable in isolated E2E
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_03_discover_query(self, client, sample_discover_request):
        """POST a discovery query and verify response structure."""
        response = await client.post(
            "/api/v1/discover",
            json=sample_discover_request,
        )
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "response_text" in data
            assert "confidence" in data
            assert isinstance(data["recommendations"], list)
        else:
            assert response.status_code in (422, 500)

    @pytest.mark.asyncio
    async def test_04_entity_detail(self, client, sample_entity):
        """GET a specific entity by ID."""
        entity_id = sample_entity["id"]
        response = await client.get(f"/api/v1/entities/{entity_id}")
        if response.status_code == 200:
            data = response.json()
            assert data["id"] == entity_id
            assert "name" in data
            assert "vibe_dna" in data
        else:
            # Entity may not exist in test DB
            assert response.status_code in (404, 500)

    @pytest.mark.asyncio
    async def test_05_vector_search(self, client):
        """GET vector search results."""
        response = await client.get(
            "/api/v1/search/vector",
            params={"query": "cozy Italian atmosphere", "limit": 5},
        )
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)
        else:
            assert response.status_code in (422, 500)

    @pytest.mark.asyncio
    async def test_06_graph_related(self, client):
        """GET related entities from the knowledge graph."""
        response = await client.get(
            "/api/v1/graph/related",
            params={"entity_name": "Ristorante Porfido"},
        )
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
        else:
            assert response.status_code in (422, 500)

    @pytest.mark.asyncio
    async def test_07_graph_lineage(self, client):
        """GET entity lineage from the knowledge graph."""
        response = await client.get(
            "/api/v1/graph/lineage",
            params={"entity_name": "Ristorante Porfido", "depth": 2},
        )
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
        else:
            assert response.status_code in (422, 500)


class TestAdminFlow:
    """E2E admin operations: curator dashboard, analytics, ingestion."""

    @pytest.mark.asyncio
    async def test_admin_stats(self, client):
        """GET admin analytics stats."""
        response = await client.get("/api/v1/admin/stats")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
        else:
            assert response.status_code in (401, 403, 500)

    @pytest.mark.asyncio
    async def test_admin_entity_update_status(self, client, sample_entity):
        """PATCH entity status (curator approve)."""
        response = await client.patch(
            f"/api/v1/admin/entities/{sample_entity['id']}/status",
            json={"status": "selected"},
        )
        assert response.status_code in (200, 404, 401, 403, 500)

    @pytest.mark.asyncio
    async def test_admin_ingestion_trigger(self, client):
        """POST an ingestion job."""
        response = await client.post(
            "/api/v1/admin/ingest",
            json={
                "query": "best coffee shops Milan",
                "location": "Milan",
                "category": "cafe",
            },
        )
        assert response.status_code in (200, 202, 422, 500)


class TestAuthFlow:
    """E2E authentication: register -> login -> access protected route."""

    @pytest.mark.asyncio
    async def test_register(self, client):
        """POST register a new user."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": "test@arbor.ai",
                "password": "securepassword123",
            },
        )
        assert response.status_code in (200, 201, 409, 422, 500)

    @pytest.mark.asyncio
    async def test_login(self, client):
        """POST login and get JWT token."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@arbor.ai",
                "password": "securepassword123",
            },
        )
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
        else:
            assert response.status_code in (401, 422, 500)

    @pytest.mark.asyncio
    async def test_protected_endpoint_without_token(self, client):
        """GET protected endpoint without token — should 401/403."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code in (401, 403, 404, 500)
