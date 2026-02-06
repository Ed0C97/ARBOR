"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Async backend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Environment overrides for test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set safe environment variables for all tests."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "arbor_test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-unit-tests-min-32-chars")


# ---------------------------------------------------------------------------
# Mock database connections
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_postgres():
    """Return a mock async postgres connection."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="OK")
    return conn


@pytest.fixture
def mock_redis():
    """Return a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def mock_qdrant():
    """Return a mock Qdrant client."""
    client = MagicMock()
    client.search = AsyncMock(return_value=[])
    client.upsert = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_neo4j():
    """Return a mock Neo4j driver."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=MagicMock(data=lambda: []))
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entity():
    """Return a sample entity dictionary."""
    return {
        "id": "test-entity-001",
        "name": "Ristorante Porfido",
        "category": "restaurant",
        "city": "Milan",
        "address": "Via Monte Napoleone 10",
        "price_tier": 4,
        "status": "vetted",
        "verified": True,
        "description": "High-end Italian dining experience",
        "vibe_dna": {
            "dimensions": {
                "formality": 0.9,
                "craftsmanship": 0.95,
                "price_value": 0.7,
                "atmosphere": 0.85,
                "service_quality": 0.9,
                "exclusivity": 0.88,
            },
            "tags": ["fine-dining", "italian", "michelin"],
            "signature_items": ["Risotto allo Zafferano", "Ossobuco"],
            "target_audience": "Luxury dining enthusiasts",
            "summary": "An exquisite Italian fine-dining experience.",
        },
        "phone": "+39 02 1234567",
        "website": "https://porfido.example.com",
        "images": [],
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_vibe_dimensions():
    """Return sample vibe dimensions."""
    return {
        "formality": 0.8,
        "craftsmanship": 0.9,
        "price_value": 0.6,
        "atmosphere": 0.85,
        "service_quality": 0.75,
        "exclusivity": 0.7,
    }


@pytest.fixture
def sample_discover_request():
    """Return a sample discover request payload."""
    return {
        "query": "Romantic Italian restaurant in Milan",
        "location": "Milan",
        "category": "restaurant",
        "price_max": 4,
        "limit": 5,
    }
