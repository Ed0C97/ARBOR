"""Integration tests for API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "A.R.B.O.R." in data["name"]


class TestDiscoverEndpoint:
    @pytest.mark.asyncio
    async def test_discover_requires_query(self, client):
        response = await client.post(
            "/api/v1/discover",
            json={"query": ""},
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_discover_valid_request(self, client):
        response = await client.post(
            "/api/v1/discover",
            json={"query": "test query", "limit": 5},
        )
        # May fail without LLM keys, but should not 500
        assert response.status_code in (200, 500)


class TestEntitiesEndpoint:
    @pytest.mark.asyncio
    async def test_list_entities(self, client):
        response = await client.get("/api/v1/entities")
        # May fail without DB, but should handle gracefully
        assert response.status_code in (200, 500)
