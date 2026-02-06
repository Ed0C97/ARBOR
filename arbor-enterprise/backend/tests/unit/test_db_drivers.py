"""Unit tests for database driver modules: Neo4j, Qdrant, and Redis.

Tests the singleton lifecycle, health checks, schema initialization,
and collection management for each database driver, using mocks to
avoid real connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ═══════════════════════════════════════════════════════════════════════════
# Neo4j Driver Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNeo4jDriver:
    """Tests for app.db.neo4j.driver module."""

    def setup_method(self):
        """Reset the global _driver singleton before each test."""
        import app.db.neo4j.driver as mod
        mod._driver = None

    # ── get_neo4j_driver ──────────────────────────────────────────────

    @patch("app.db.neo4j.driver.settings")
    @patch("app.db.neo4j.driver.AsyncGraphDatabase")
    def test_get_driver_returns_driver_when_configured(self, mock_agd, mock_settings):
        """get_neo4j_driver creates and returns a driver when neo4j_uri is set."""
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_user = "neo4j"
        mock_settings.neo4j_password = "password"
        mock_agd.driver.return_value = MagicMock()

        from app.db.neo4j.driver import get_neo4j_driver

        driver = get_neo4j_driver()

        assert driver is not None
        mock_agd.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

    @patch("app.db.neo4j.driver.settings")
    def test_get_driver_returns_none_when_not_configured(self, mock_settings):
        """get_neo4j_driver returns None when neo4j_uri is empty."""
        mock_settings.neo4j_uri = ""

        from app.db.neo4j.driver import get_neo4j_driver

        assert get_neo4j_driver() is None

    @patch("app.db.neo4j.driver.settings")
    @patch("app.db.neo4j.driver.AsyncGraphDatabase")
    def test_get_driver_returns_singleton(self, mock_agd, mock_settings):
        """get_neo4j_driver returns the same instance on repeated calls."""
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_user = "neo4j"
        mock_settings.neo4j_password = "pw"
        mock_agd.driver.return_value = MagicMock()

        from app.db.neo4j.driver import get_neo4j_driver

        d1 = get_neo4j_driver()
        d2 = get_neo4j_driver()

        assert d1 is d2
        assert mock_agd.driver.call_count == 1

    # ── close_neo4j_driver ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_close_driver_closes_and_clears_singleton(self):
        """close_neo4j_driver awaits driver.close() and sets _driver to None."""
        import app.db.neo4j.driver as mod

        mock_driver = AsyncMock()
        mod._driver = mock_driver

        await mod.close_neo4j_driver()

        mock_driver.close.assert_awaited_once()
        assert mod._driver is None

    @pytest.mark.asyncio
    async def test_close_driver_noop_when_none(self):
        """close_neo4j_driver is a no-op when no driver exists."""
        import app.db.neo4j.driver as mod

        mod._driver = None
        await mod.close_neo4j_driver()  # should not raise

    # ── init_neo4j_schema ─────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.neo4j.driver.get_neo4j_driver")
    async def test_init_schema_runs_all_statements(self, mock_get_driver):
        """init_neo4j_schema runs all constraint and index statements."""
        mock_session = AsyncMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        from app.db.neo4j.driver import init_neo4j_schema

        await init_neo4j_schema()

        # 4 constraints + 4 indexes = 8 statements
        assert mock_session.run.await_count == 8

    @pytest.mark.asyncio
    @patch("app.db.neo4j.driver.get_neo4j_driver")
    async def test_init_schema_skips_when_not_configured(self, mock_get_driver):
        """init_neo4j_schema returns early when driver is None."""
        mock_get_driver.return_value = None

        from app.db.neo4j.driver import init_neo4j_schema

        await init_neo4j_schema()  # should not raise

    # ── check_neo4j_health ────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.neo4j.driver.get_neo4j_driver")
    async def test_health_check_returns_true_on_success(self, mock_get_driver):
        """check_neo4j_health returns True when ping query succeeds."""
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=1)

        mock_result = AsyncMock()
        mock_result.single.return_value = mock_record

        mock_session = AsyncMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        from app.db.neo4j.driver import check_neo4j_health

        assert await check_neo4j_health() is True

    @pytest.mark.asyncio
    @patch("app.db.neo4j.driver.get_neo4j_driver")
    async def test_health_check_returns_false_when_driver_none(self, mock_get_driver):
        """check_neo4j_health returns False when no driver is configured."""
        mock_get_driver.return_value = None

        from app.db.neo4j.driver import check_neo4j_health

        assert await check_neo4j_health() is False

    @pytest.mark.asyncio
    @patch("app.db.neo4j.driver.get_neo4j_driver")
    async def test_health_check_returns_false_on_exception(self, mock_get_driver):
        """check_neo4j_health returns False when session.run raises."""
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("connection refused")

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        from app.db.neo4j.driver import check_neo4j_health

        assert await check_neo4j_health() is False


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant Client Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQdrantClient:
    """Tests for app.db.qdrant.client module."""

    def setup_method(self):
        """Reset global singletons before each test."""
        import app.db.qdrant.client as mod
        mod._async_client = None
        mod._sync_client = None

    # ── get_async_qdrant_client ───────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.settings")
    @patch("app.db.qdrant.client.AsyncQdrantClient")
    async def test_get_async_client_creates_singleton(self, mock_cls, mock_settings):
        """get_async_qdrant_client creates and reuses an async client."""
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = ""
        mock_settings.qdrant_prefer_grpc = True
        mock_cls.return_value = MagicMock()

        from app.db.qdrant.client import get_async_qdrant_client

        c1 = await get_async_qdrant_client()
        c2 = await get_async_qdrant_client()

        assert c1 is c2
        assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.settings")
    async def test_get_async_client_returns_none_when_not_configured(self, mock_settings):
        """get_async_qdrant_client returns None when qdrant_url is empty."""
        mock_settings.qdrant_url = ""

        from app.db.qdrant.client import get_async_qdrant_client

        assert await get_async_qdrant_client() is None

    # ── init_qdrant_collections ───────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    async def test_init_collections_creates_missing_collections(self, mock_get_client):
        """init_qdrant_collections creates collections that do not already exist."""
        mock_client = AsyncMock()
        collections_resp = MagicMock()
        collections_resp.collections = []  # no existing collections
        mock_client.get_collections.return_value = collections_resp
        mock_get_client.return_value = mock_client

        from app.db.qdrant.client import init_qdrant_collections

        await init_qdrant_collections()

        # Should create entities_vectors and semantic_cache
        assert mock_client.create_collection.await_count == 2
        call_names = [
            call.kwargs["collection_name"]
            for call in mock_client.create_collection.call_args_list
        ]
        assert "entities_vectors" in call_names
        assert "semantic_cache" in call_names

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    async def test_init_collections_updates_existing(self, mock_get_client):
        """init_qdrant_collections updates existing collections' config."""
        mock_client = AsyncMock()
        existing_collection = MagicMock()
        existing_collection.name = "entities_vectors"
        existing_cache = MagicMock()
        existing_cache.name = "semantic_cache"
        collections_resp = MagicMock()
        collections_resp.collections = [existing_collection, existing_cache]
        mock_client.get_collections.return_value = collections_resp
        mock_get_client.return_value = mock_client

        from app.db.qdrant.client import init_qdrant_collections

        await init_qdrant_collections()

        # Should NOT create, but should update entities_vectors
        mock_client.create_collection.assert_not_awaited()
        mock_client.update_collection.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    async def test_init_collections_skips_when_not_configured(self, mock_get_client):
        """init_qdrant_collections is a no-op when the client is None."""
        mock_get_client.return_value = None

        from app.db.qdrant.client import init_qdrant_collections

        await init_qdrant_collections()  # should not raise

    # ── close_qdrant_client ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_close_client_closes_both_clients(self):
        """close_qdrant_client closes async and sync clients."""
        import app.db.qdrant.client as mod

        mock_async = AsyncMock()
        mock_sync = MagicMock()
        mod._async_client = mock_async
        mod._sync_client = mock_sync

        await mod.close_qdrant_client()

        mock_async.close.assert_awaited_once()
        mock_sync.close.assert_called_once()
        assert mod._async_client is None
        assert mod._sync_client is None

    # ── check_qdrant_health ───────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    @patch("app.db.qdrant.client.settings")
    async def test_health_check_returns_healthy(self, mock_settings, mock_get_client):
        """check_qdrant_health returns healthy status with collection names."""
        mock_settings.qdrant_prefer_grpc = True
        mock_client = AsyncMock()
        c1 = MagicMock()
        c1.name = "entities_vectors"
        collections_resp = MagicMock()
        collections_resp.collections = [c1]
        mock_client.get_collections.return_value = collections_resp
        mock_get_client.return_value = mock_client

        from app.db.qdrant.client import check_qdrant_health

        result = await check_qdrant_health()

        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert "entities_vectors" in result["collections"]

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    async def test_health_check_returns_not_configured(self, mock_get_client):
        """check_qdrant_health returns not_configured when client is None."""
        mock_get_client.return_value = None

        from app.db.qdrant.client import check_qdrant_health

        result = await check_qdrant_health()

        assert result["healthy"] is False
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    @patch("app.db.qdrant.client.get_async_qdrant_client")
    async def test_health_check_returns_unhealthy_on_error(self, mock_get_client):
        """check_qdrant_health returns unhealthy when get_collections raises."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("connection lost")
        mock_get_client.return_value = mock_client

        from app.db.qdrant.client import check_qdrant_health

        result = await check_qdrant_health()

        assert result["healthy"] is False
        assert result["status"] == "unhealthy"
        assert "connection lost" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# Redis Client Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRedisClient:
    """Tests for app.db.redis.client module."""

    def setup_method(self):
        """Reset the global _redis_client singleton before each test."""
        import app.db.redis.client as mod
        mod._redis_client = None

    # ── get_redis_client ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.redis.client.settings")
    @patch("app.db.redis.client.redis")
    async def test_get_client_creates_singleton(self, mock_redis_mod, mock_settings):
        """get_redis_client creates and returns a singleton client."""
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_mod.from_url.return_value = mock_client

        from app.db.redis.client import get_redis_client

        c1 = await get_redis_client()
        c2 = await get_redis_client()

        assert c1 is c2
        assert c1 is not None
        mock_client.ping.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.db.redis.client.settings")
    async def test_get_client_returns_none_when_not_configured(self, mock_settings):
        """get_redis_client returns None when redis_url is empty."""
        mock_settings.redis_url = ""

        from app.db.redis.client import get_redis_client

        assert await get_redis_client() is None

    @pytest.mark.asyncio
    @patch("app.db.redis.client.settings")
    @patch("app.db.redis.client.redis")
    async def test_get_client_returns_none_on_connection_failure(self, mock_redis_mod, mock_settings):
        """get_redis_client returns None when ping fails."""
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Connection refused")
        mock_redis_mod.from_url.return_value = mock_client

        from app.db.redis.client import get_redis_client

        assert await get_redis_client() is None

    # ── close_redis_client ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_close_client_closes_and_clears(self):
        """close_redis_client awaits client.close() and sets to None."""
        import app.db.redis.client as mod

        mock_client = AsyncMock()
        mod._redis_client = mock_client

        await mod.close_redis_client()

        mock_client.close.assert_awaited_once()
        assert mod._redis_client is None

    @pytest.mark.asyncio
    async def test_close_client_noop_when_none(self):
        """close_redis_client is a no-op when no client exists."""
        import app.db.redis.client as mod

        mod._redis_client = None
        await mod.close_redis_client()  # should not raise

    # ── check_redis_health ────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_health_check_returns_healthy(self, mock_get_client):
        """check_redis_health returns healthy when ping and info succeed."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.info.return_value = {"redis_version": "7.2.4"}
        mock_get_client.return_value = mock_client

        from app.db.redis.client import check_redis_health

        result = await check_redis_health()

        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert result["version"] == "7.2.4"

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_health_check_returns_not_configured(self, mock_get_client):
        """check_redis_health returns not_configured when client is None."""
        mock_get_client.return_value = None

        from app.db.redis.client import check_redis_health

        result = await check_redis_health()

        assert result["healthy"] is False
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_health_check_returns_unhealthy_on_error(self, mock_get_client):
        """check_redis_health returns unhealthy when ping raises."""
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("timeout")
        mock_get_client.return_value = mock_client

        from app.db.redis.client import check_redis_health

        result = await check_redis_health()

        assert result["healthy"] is False
        assert result["status"] == "unhealthy"
        assert "timeout" in result["error"]

    # ── RedisCache class ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_redis_cache_get_returns_parsed_json(self, mock_get_client):
        """RedisCache.get() parses JSON values from Redis."""
        import json
        mock_client = AsyncMock()
        mock_client.get.return_value = json.dumps({"key": "value"})
        mock_get_client.return_value = mock_client

        from app.db.redis.client import RedisCache

        cache = RedisCache()
        result = await cache.get("test-key")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_redis_cache_get_returns_none_when_no_client(self, mock_get_client):
        """RedisCache.get() returns None when Redis is not available."""
        mock_get_client.return_value = None

        from app.db.redis.client import RedisCache

        cache = RedisCache()
        result = await cache.get("test-key")

        assert result is None

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_redis_cache_set_serializes_dict(self, mock_get_client):
        """RedisCache.set() serializes dicts to JSON before storing."""
        import json
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        from app.db.redis.client import RedisCache

        cache = RedisCache()
        await cache.set("k", {"a": 1}, ttl=600)

        mock_client.set.assert_awaited_once()
        stored_value = mock_client.set.call_args[0][1]
        assert json.loads(stored_value) == {"a": 1}

    # ── SlidingWindowRateLimiter ──────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_rate_limiter_allows_when_redis_down(self, mock_get_client):
        """SlidingWindowRateLimiter fails open when Redis is unavailable."""
        mock_get_client.return_value = None

        from app.db.redis.client import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter()
        allowed, remaining = await limiter.is_allowed("key", 100, 60)

        assert allowed is True
        assert remaining == 100

    # ── IdempotencyStore ──────────────────────────────────────────────

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_idempotency_store_returns_none_for_new_key(self, mock_get_client):
        """IdempotencyStore.check_and_set() returns None for a previously unseen key."""
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        from app.db.redis.client import IdempotencyStore

        store = IdempotencyStore()
        result = await store.check_and_set("unique-key-123")

        assert result is None

    @pytest.mark.asyncio
    @patch("app.db.redis.client.get_redis_client")
    async def test_idempotency_store_returns_cached_for_existing_key(self, mock_get_client):
        """IdempotencyStore.check_and_set() returns cached response for a known key."""
        import json
        cached = {"status": "created", "id": 42}
        mock_client = AsyncMock()
        mock_client.get.return_value = json.dumps(cached)
        mock_get_client.return_value = mock_client

        from app.db.redis.client import IdempotencyStore

        store = IdempotencyStore()
        result = await store.check_and_set("existing-key")

        assert result == cached
