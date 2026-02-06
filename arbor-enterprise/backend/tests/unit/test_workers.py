"""Unit tests for Celery background workers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We must patch get_settings before importing the module so that it does not
# try to connect to a real broker at import time.
_mock_settings = MagicMock(
    celery_broker_url="redis://localhost:6379/1",
    celery_result_backend="redis://localhost:6379/2",
)

with patch("app.workers.background_jobs.get_settings", return_value=_mock_settings):
    with patch("app.workers.background_jobs.settings", _mock_settings):
        from app.workers.background_jobs import (
            _check_index_health_async,
            _cleanup_old_data_async,
            _refresh_stale_embeddings_async,
            _warmup_popular_queries_async,
            celery_app,
            check_index_health,
            cleanup_old_data,
            refresh_stale_embeddings,
            warmup_popular_queries,
        )


# ==========================================================================
# Celery App Configuration Tests
# ==========================================================================


class TestCeleryAppConfiguration:
    """Tests for the Celery app setup and configuration."""

    def test_celery_app_name(self):
        assert celery_app.main == "arbor_workers"

    def test_celery_serializer_is_json(self):
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"

    def test_celery_accept_content(self):
        assert "json" in celery_app.conf.accept_content

    def test_celery_timezone_is_utc(self):
        assert celery_app.conf.timezone == "UTC"
        assert celery_app.conf.enable_utc is True

    def test_celery_task_tracking_enabled(self):
        assert celery_app.conf.task_track_started is True

    def test_celery_task_time_limits(self):
        assert celery_app.conf.task_time_limit == 3600
        assert celery_app.conf.task_soft_time_limit == 3300

    def test_celery_fairness_settings(self):
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_celery_reliability_settings(self):
        assert celery_app.conf.task_acks_late is True
        assert celery_app.conf.task_reject_on_worker_lost is True


# ==========================================================================
# Beat Schedule Tests
# ==========================================================================


class TestBeatSchedule:
    """Tests for the Celery beat schedule definitions."""

    def test_beat_schedule_has_required_tasks(self):
        schedule = celery_app.conf.beat_schedule
        assert "refresh-stale-embeddings" in schedule
        assert "weekly-reindex-check" in schedule
        assert "cache-warmup" in schedule
        assert "cleanup-audit-logs" in schedule

    def test_refresh_embeddings_schedule(self):
        entry = celery_app.conf.beat_schedule["refresh-stale-embeddings"]
        assert entry["task"] == "app.workers.background_jobs.refresh_stale_embeddings"
        assert entry["options"]["queue"] == "batch"

    def test_weekly_reindex_schedule(self):
        entry = celery_app.conf.beat_schedule["weekly-reindex-check"]
        assert entry["task"] == "app.workers.background_jobs.check_index_health"
        assert entry["options"]["queue"] == "batch"

    def test_cache_warmup_schedule(self):
        entry = celery_app.conf.beat_schedule["cache-warmup"]
        assert entry["task"] == "app.workers.background_jobs.warmup_popular_queries"
        assert entry["options"]["queue"] == "default"

    def test_cleanup_schedule(self):
        entry = celery_app.conf.beat_schedule["cleanup-audit-logs"]
        assert entry["task"] == "app.workers.background_jobs.cleanup_old_data"
        assert entry["options"]["queue"] == "maintenance"


# ==========================================================================
# Task Registration Tests
# ==========================================================================


class TestTaskRegistration:
    """Tests for task function existence and registration."""

    def test_refresh_stale_embeddings_is_callable(self):
        assert callable(refresh_stale_embeddings)

    def test_warmup_popular_queries_is_callable(self):
        assert callable(warmup_popular_queries)

    def test_check_index_health_is_callable(self):
        assert callable(check_index_health)

    def test_cleanup_old_data_is_callable(self):
        assert callable(cleanup_old_data)

    def test_refresh_stale_embeddings_has_task_name(self):
        assert hasattr(refresh_stale_embeddings, "name")
        assert "refresh_stale_embeddings" in refresh_stale_embeddings.name

    def test_warmup_popular_queries_has_task_name(self):
        assert hasattr(warmup_popular_queries, "name")
        assert "warmup_popular_queries" in warmup_popular_queries.name

    def test_check_index_health_has_task_name(self):
        assert hasattr(check_index_health, "name")
        assert "check_index_health" in check_index_health.name

    def test_cleanup_old_data_has_task_name(self):
        assert hasattr(cleanup_old_data, "name")
        assert "cleanup_old_data" in cleanup_old_data.name

    def test_refresh_stale_embeddings_has_max_retries(self):
        # refresh_stale_embeddings is bind=True with max_retries=3
        assert refresh_stale_embeddings.max_retries == 3


# ==========================================================================
# Async Implementation Tests
# ==========================================================================


class TestRefreshStaleEmbeddingsAsync:
    """Tests for the _refresh_stale_embeddings_async implementation."""

    @patch("app.workers.background_jobs.get_async_qdrant_client")
    @patch("app.workers.background_jobs.get_llm_gateway")
    @patch("app.workers.background_jobs.get_arbor_session")
    async def test_returns_zero_when_no_stale_entities(
        self, mock_get_session, mock_get_llm, mock_get_qdrant
    ):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await _refresh_stale_embeddings_async(24)
        assert result == {"refreshed": 0, "failed": 0}


class TestWarmupPopularQueriesAsync:
    """Tests for the _warmup_popular_queries_async implementation."""

    @patch("app.workers.background_jobs.get_discovery_graph")
    async def test_warmup_counts_successes(self, mock_get_graph):
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"results": []})
        mock_get_graph.return_value = mock_graph

        result = await _warmup_popular_queries_async()
        assert "warmed" in result
        assert "total" in result
        assert result["total"] == 8  # 8 popular queries defined
        assert result["warmed"] <= result["total"]

    @patch("app.workers.background_jobs.get_discovery_graph")
    async def test_warmup_handles_failures_gracefully(self, mock_get_graph):
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_get_graph.return_value = mock_graph

        result = await _warmup_popular_queries_async()
        assert result["warmed"] == 0
        assert result["total"] == 8


class TestCheckIndexHealthAsync:
    """Tests for the _check_index_health_async implementation."""

    @patch("app.workers.background_jobs.get_async_qdrant_client")
    async def test_health_check_returns_collection_info(self, mock_get_client):
        mock_client = AsyncMock()
        mock_info = MagicMock()
        mock_info.status.name = "GREEN"
        mock_info.points_count = 1000
        mock_info.indexed_vectors_count = 1000
        mock_info.segments = []
        mock_client.get_collection = AsyncMock(return_value=mock_info)
        mock_get_client.return_value = mock_client

        result = await _check_index_health_async()
        assert "entities_vectors" in result
        assert "semantic_cache" in result
        assert result["entities_vectors"]["status"] == "GREEN"
        assert result["entities_vectors"]["points_count"] == 1000

    @patch("app.workers.background_jobs.get_async_qdrant_client")
    async def test_health_check_handles_error(self, mock_get_client):
        mock_client = AsyncMock()
        mock_client.get_collection = AsyncMock(side_effect=Exception("Connection refused"))
        mock_get_client.return_value = mock_client

        result = await _check_index_health_async()
        assert result["entities_vectors"]["status"] == "ERROR"
        assert "error" in result["entities_vectors"]
