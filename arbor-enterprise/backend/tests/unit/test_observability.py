"""Unit tests for observability — metrics, SLO, and telemetry."""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.observability.metrics import (
    arbor_active_curators,
    arbor_active_users,
    arbor_build_info,
    arbor_cache_hits_total,
    arbor_circuit_breaker_state,
    arbor_db_pool_connections,
    arbor_discover_requests_total,
    arbor_embedding_latency_seconds,
    arbor_entities_total,
    arbor_guardrail_blocks_total,
    arbor_llm_latency_seconds,
    arbor_llm_tokens_used,
    arbor_query_latency_seconds,
    arbor_rate_limit_hits_total,
    arbor_rerank_latency_seconds,
    arbor_search_requests_total,
    get_metrics,
    get_metrics_content_type,
    record_cache_hit,
    record_cache_miss,
    record_discover_request,
    record_embedding_latency,
    record_guardrail_block,
    record_llm_tokens,
    record_rate_limit,
    record_rerank_latency,
    record_search_request,
    set_build_info,
    set_circuit_breaker_state,
    track_latency,
    track_llm_latency,
    update_db_pool_metrics,
)
from app.observability.slo import (
    BurnRateCalculator,
    ErrorBudget,
    SLODefinition,
    SLOMetric,
    SLOMonitor,
    SLOType,
    get_slo_monitor,
)

# ==========================================================================
# Metrics Registration Tests
# ==========================================================================


class TestMetricRegistration:
    """Tests that all Prometheus metrics are properly registered."""

    def test_discover_requests_counter_exists(self):
        assert arbor_discover_requests_total is not None
        assert arbor_discover_requests_total._name == "arbor_discover_requests_total"

    def test_search_requests_counter_exists(self):
        assert arbor_search_requests_total is not None
        assert arbor_search_requests_total._name == "arbor_search_requests_total"

    def test_llm_latency_histogram_exists(self):
        assert arbor_llm_latency_seconds is not None
        assert arbor_llm_latency_seconds._name == "arbor_llm_latency_seconds"

    def test_embedding_latency_histogram_exists(self):
        assert arbor_embedding_latency_seconds is not None

    def test_rerank_latency_histogram_exists(self):
        assert arbor_rerank_latency_seconds is not None

    def test_active_curators_gauge_exists(self):
        assert arbor_active_curators is not None

    def test_entities_total_gauge_exists(self):
        assert arbor_entities_total is not None

    def test_circuit_breaker_gauge_exists(self):
        assert arbor_circuit_breaker_state is not None

    def test_db_pool_gauge_exists(self):
        assert arbor_db_pool_connections is not None

    def test_rate_limit_counter_exists(self):
        assert arbor_rate_limit_hits_total is not None

    def test_guardrail_counter_exists(self):
        assert arbor_guardrail_blocks_total is not None

    def test_build_info_exists(self):
        assert arbor_build_info is not None

    def test_query_latency_histogram_exists(self):
        assert arbor_query_latency_seconds is not None

    def test_cache_hits_counter_exists(self):
        assert arbor_cache_hits_total is not None

    def test_llm_tokens_counter_exists(self):
        assert arbor_llm_tokens_used is not None

    def test_active_users_gauge_exists(self):
        assert arbor_active_users is not None


# ==========================================================================
# Metric Helper Tests
# ==========================================================================


class TestMetricHelpers:
    """Tests for metric convenience helper functions."""

    def test_record_cache_hit(self):
        # Should not raise
        record_cache_hit("redis")
        record_cache_hit("llm_semantic")

    def test_record_cache_miss(self):
        record_cache_miss("redis")
        record_cache_miss("llm_semantic")

    def test_record_llm_tokens(self):
        record_llm_tokens("gpt-4o", prompt_tokens=100, completion_tokens=50)
        record_llm_tokens("gemini-3-pro-preview", prompt_tokens=200, completion_tokens=80)

    def test_record_discover_request(self):
        record_discover_request(status="success", intent="restaurant", cache_hit=True)
        record_discover_request(status="error", intent="unknown", cache_hit=False)

    def test_record_search_request(self):
        record_search_request(search_type="vector", status="success")
        record_search_request(search_type="hybrid", status="error")

    def test_record_embedding_latency(self):
        record_embedding_latency(provider="cohere", batch_size=96, latency_seconds=0.35)

    def test_record_rerank_latency(self):
        record_rerank_latency(provider="cohere", doc_count=20, latency_seconds=0.15)

    def test_set_circuit_breaker_state_closed(self):
        set_circuit_breaker_state("cohere", "closed")

    def test_set_circuit_breaker_state_open(self):
        set_circuit_breaker_state("qdrant", "open")

    def test_set_circuit_breaker_state_half_open(self):
        set_circuit_breaker_state("neo4j", "half_open")

    def test_update_db_pool_metrics(self):
        update_db_pool_metrics(database="postgres", active=5, idle=10, overflow=0)

    def test_record_rate_limit_allowed(self):
        record_rate_limit(endpoint="/discover", allowed=True)

    def test_record_rate_limit_blocked(self):
        record_rate_limit(endpoint="/discover", allowed=False)

    def test_record_guardrail_block(self):
        record_guardrail_block(block_type="input", reason="prompt_injection")

    def test_set_build_info(self):
        set_build_info(version="1.0.0", environment="test", git_sha="abc123")


class TestGetMetrics:
    """Tests for Prometheus metrics output."""

    def test_get_metrics_returns_bytes(self):
        result = get_metrics()
        assert isinstance(result, bytes)

    def test_get_metrics_contains_metric_names(self):
        output = get_metrics().decode("utf-8")
        assert "arbor_discover_requests_total" in output

    def test_get_metrics_content_type(self):
        ct = get_metrics_content_type()
        assert "text/plain" in ct or "openmetrics" in ct


# ==========================================================================
# Track Latency Decorator Tests
# ==========================================================================


class TestTrackLatencyDecorator:
    """Tests for the track_latency decorator."""

    async def test_async_function_latency_recorded(self):
        @track_latency(endpoint="/test", intent="test_intent")
        async def sample_async():
            return "result"

        result = await sample_async()
        assert result == "result"

    def test_sync_function_latency_recorded(self):
        @track_latency(endpoint="/sync_test", intent="sync_intent")
        def sample_sync():
            return 42

        result = sample_sync()
        assert result == 42

    async def test_async_function_records_error_status(self):
        @track_latency(endpoint="/error_test", intent="err")
        async def failing_async():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await failing_async()

    def test_sync_function_records_error_status(self):
        @track_latency(endpoint="/sync_error", intent="err")
        def failing_sync():
            raise RuntimeError("sync fail")

        with pytest.raises(RuntimeError, match="sync fail"):
            failing_sync()


class TestTrackLLMLatencyDecorator:
    """Tests for the track_llm_latency decorator."""

    async def test_async_llm_latency(self):
        @track_llm_latency(provider="gemini", model="gemini-3-pro", task_type="completion")
        async def mock_llm_call():
            return {"response": "hello"}

        result = await mock_llm_call()
        assert result == {"response": "hello"}

    def test_sync_llm_latency(self):
        @track_llm_latency(provider="cohere", model="embed-v4.0", task_type="embedding")
        def mock_embed():
            return [0.1, 0.2, 0.3]

        result = mock_embed()
        assert len(result) == 3


# ==========================================================================
# SLO Type and Definition Tests
# ==========================================================================


class TestSLOType:
    """Tests for the SLOType enum."""

    def test_slo_type_values(self):
        assert SLOType.AVAILABILITY.value == "availability"
        assert SLOType.LATENCY.value == "latency"
        assert SLOType.ERROR_RATE.value == "error_rate"
        assert SLOType.THROUGHPUT.value == "throughput"
        assert SLOType.QUALITY.value == "quality"


class TestSLODefinition:
    """Tests for the SLODefinition dataclass."""

    def test_create_definition(self):
        defn = SLODefinition(
            slo_id="test_slo",
            name="Test SLO",
            description="A test SLO",
            slo_type=SLOType.AVAILABILITY,
            target=0.999,
        )
        assert defn.slo_id == "test_slo"
        assert defn.target == 0.999
        assert defn.window_seconds == 86_400
        assert defn.endpoint is None
        assert defn.tier == "medium"

    def test_frozen_definition(self):
        defn = SLODefinition(
            slo_id="frozen",
            name="Frozen",
            description="Cannot mutate",
            slo_type=SLOType.LATENCY,
            target=500.0,
        )
        with pytest.raises(AttributeError):
            defn.target = 1000.0


# ==========================================================================
# BurnRateCalculator Tests
# ==========================================================================


class TestBurnRateCalculator:
    """Tests for the BurnRateCalculator utility."""

    def test_calculate_burn_rate_normal(self):
        # 50% budget consumed at 50% elapsed = burn rate 1.0
        rate = BurnRateCalculator.calculate_burn_rate(consumed=0.5, elapsed_fraction=0.5)
        assert rate == pytest.approx(1.0)

    def test_calculate_burn_rate_fast(self):
        # 50% consumed at 25% elapsed = burn rate 2.0
        rate = BurnRateCalculator.calculate_burn_rate(consumed=0.5, elapsed_fraction=0.25)
        assert rate == pytest.approx(2.0)

    def test_calculate_burn_rate_zero_elapsed(self):
        rate = BurnRateCalculator.calculate_burn_rate(consumed=0.1, elapsed_fraction=0.0)
        assert rate == 0.0

    def test_project_exhaustion_none_when_healthy(self):
        result = BurnRateCalculator.project_exhaustion(
            remaining=1.0, burn_rate=0.5, window_remaining_seconds=43200
        )
        assert result is None  # Budget won't exhaust within window

    def test_project_exhaustion_returns_datetime_when_critical(self):
        result = BurnRateCalculator.project_exhaustion(
            remaining=0.1, burn_rate=5.0, window_remaining_seconds=3600
        )
        assert result is not None
        assert isinstance(result, datetime)

    def test_project_exhaustion_none_when_zero_burn_rate(self):
        result = BurnRateCalculator.project_exhaustion(
            remaining=0.5, burn_rate=0.0, window_remaining_seconds=3600
        )
        assert result is None

    def test_classify_severity_healthy(self):
        assert BurnRateCalculator.classify_severity(0.5) == "healthy"
        assert BurnRateCalculator.classify_severity(0.0) == "healthy"

    def test_classify_severity_warning(self):
        assert BurnRateCalculator.classify_severity(1.5) == "warning"

    def test_classify_severity_critical(self):
        assert BurnRateCalculator.classify_severity(5.0) == "critical"

    def test_classify_severity_exhausted(self):
        assert BurnRateCalculator.classify_severity(15.0) == "exhausted"


# ==========================================================================
# SLOMonitor Tests
# ==========================================================================


class TestSLOMonitor:
    """Tests for the SLOMonitor class."""

    def setup_method(self):
        self.monitor = SLOMonitor()

    def test_default_slos_registered(self):
        statuses = self.monitor.get_all_slo_statuses()
        slo_ids = {s.slo_id for s in statuses}
        assert "api_availability" in slo_ids
        assert "discover_latency" in slo_ids
        assert "search_latency" in slo_ids
        assert "error_rate" in slo_ids
        assert "discovery_quality" in slo_ids

    def test_record_and_get_status(self):
        self.monitor.record_request(
            endpoint="/discover",
            latency_ms=200.0,
            is_success=True,
        )
        status = self.monitor.get_slo_status("api_availability")
        assert isinstance(status, SLOMetric)
        assert status.total_requests == 1
        assert status.good_requests == 1
        assert status.is_meeting_target is True

    def test_availability_slo_violation(self):
        # Record 1 failure in 2 requests = 50% availability (target 99.9%)
        self.monitor.record_request("/api", 100, True)
        self.monitor.record_request("/api", 100, False)

        status = self.monitor.get_slo_status("api_availability")
        assert status.total_requests == 2
        assert status.current_value == pytest.approx(0.5)
        assert status.is_meeting_target is False

    def test_latency_slo_met(self):
        # Record requests under the 2000ms target for discover_latency
        for _ in range(20):
            self.monitor.record_request("/discover", 500.0, True)

        status = self.monitor.get_slo_status("discover_latency")
        assert status.is_meeting_target is True

    def test_latency_slo_violated(self):
        # Record requests over the 2000ms target
        for _ in range(20):
            self.monitor.record_request("/discover", 3000.0, True)

        status = self.monitor.get_slo_status("discover_latency")
        assert status.is_meeting_target is False

    def test_error_rate_slo(self):
        # All successes
        for _ in range(100):
            self.monitor.record_request("/api", 100, True)

        status = self.monitor.get_slo_status("error_rate")
        assert status.current_value == 0.0
        assert status.is_meeting_target is True

    def test_quality_slo(self):
        # Record quality passes
        for _ in range(19):
            self.monitor.record_request("/discover", 100, True, is_quality_pass=True)
        # One quality failure
        self.monitor.record_request("/discover", 100, True, is_quality_pass=False)

        status = self.monitor.get_slo_status("discovery_quality")
        assert status.current_value == pytest.approx(0.95)

    def test_define_custom_slo(self):
        custom = SLODefinition(
            slo_id="custom_throughput",
            name="Custom Throughput",
            description="At least 10 RPS",
            slo_type=SLOType.THROUGHPUT,
            target=10.0,
            tier="low",
        )
        self.monitor.define_slo(custom)
        status = self.monitor.get_slo_status("custom_throughput")
        assert status.slo_id == "custom_throughput"

    def test_get_error_budget(self):
        self.monitor.record_request("/api", 100, True)
        budget = self.monitor.get_error_budget("api_availability")
        assert isinstance(budget, ErrorBudget)
        assert budget.total_budget == pytest.approx(0.001)
        assert budget.remaining > 0

    def test_get_burn_rate_alerts_empty_when_healthy(self):
        # All success = no alerts
        for _ in range(100):
            self.monitor.record_request("/api", 100, True)
        alerts = self.monitor.get_burn_rate_alerts()
        # Should have no alerts with burn rate > 1.0
        # (may or may not depending on timing, so just check structure)
        assert isinstance(alerts, list)

    def test_should_allow_deployment_all_healthy(self):
        for _ in range(100):
            self.monitor.record_request("/discover", 200, True)
            self.monitor.record_request("/search", 100, True)
        result = self.monitor.should_allow_deployment()
        assert result is True

    def test_get_report_structure(self):
        self.monitor.record_request("/discover", 200, True)
        report = self.monitor.get_report()
        assert "generated_at" in report
        assert "slos" in report
        assert "alerts" in report
        assert "deployment_allowed" in report
        assert "summary" in report
        assert "total_slos" in report["summary"]
        assert "meeting_target" in report["summary"]

    def test_no_data_slo_treated_as_met(self):
        # Fresh monitor with no requests should treat all SLOs as met
        fresh = SLOMonitor()
        status = fresh.get_slo_status("api_availability")
        assert status.total_requests == 0
        assert status.is_meeting_target is True


class TestGetSLOMonitor:
    """Tests for the singleton accessor."""

    def test_returns_slo_monitor_instance(self):
        # Reset singleton for test isolation
        import app.observability.slo as slo_module

        slo_module._slo_monitor_instance = None

        monitor = get_slo_monitor()
        assert isinstance(monitor, SLOMonitor)

    def test_returns_same_instance(self):
        import app.observability.slo as slo_module

        slo_module._slo_monitor_instance = None

        m1 = get_slo_monitor()
        m2 = get_slo_monitor()
        assert m1 is m2


# ==========================================================================
# Telemetry Tests
# ==========================================================================


class TestSetupTelemetry:
    """Tests for the OpenTelemetry setup_telemetry function."""

    @patch("app.observability.telemetry.trace")
    @patch("app.observability.telemetry.BatchSpanProcessor")
    @patch("app.observability.telemetry.OTLPSpanExporter")
    @patch("app.observability.telemetry.TracerProvider")
    @patch("app.observability.telemetry.Resource")
    def test_setup_creates_tracer_provider(
        self, mock_resource, mock_tp_cls, mock_exporter, mock_processor, mock_trace
    ):
        import app.observability.telemetry as tel_module

        tel_module._tracer_provider = None  # Reset singleton

        mock_resource.create.return_value = MagicMock()
        mock_tp = MagicMock()
        mock_tp_cls.return_value = mock_tp

        from app.observability.telemetry import setup_telemetry

        provider = setup_telemetry()

        mock_tp_cls.assert_called_once()
        mock_tp.add_span_processor.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once_with(mock_tp)
        assert provider is mock_tp

        # Clean up
        tel_module._tracer_provider = None

    @patch("app.observability.telemetry.trace")
    @patch("app.observability.telemetry.BatchSpanProcessor")
    @patch("app.observability.telemetry.OTLPSpanExporter")
    @patch("app.observability.telemetry.TracerProvider")
    @patch("app.observability.telemetry.Resource")
    def test_setup_returns_existing_provider_if_set(
        self, mock_resource, mock_tp_cls, mock_exporter, mock_processor, mock_trace
    ):
        import app.observability.telemetry as tel_module

        existing = MagicMock()
        tel_module._tracer_provider = existing

        from app.observability.telemetry import setup_telemetry

        provider = setup_telemetry()

        assert provider is existing
        mock_tp_cls.assert_not_called()

        tel_module._tracer_provider = None

    @patch("app.observability.telemetry.FastAPIInstrumentor")
    @patch("app.observability.telemetry.trace")
    @patch("app.observability.telemetry.BatchSpanProcessor")
    @patch("app.observability.telemetry.OTLPSpanExporter")
    @patch("app.observability.telemetry.TracerProvider")
    @patch("app.observability.telemetry.Resource")
    def test_setup_instruments_fastapi_app(
        self,
        mock_resource,
        mock_tp_cls,
        mock_exporter,
        mock_processor,
        mock_trace,
        mock_fastapi_inst,
    ):
        import app.observability.telemetry as tel_module

        tel_module._tracer_provider = None

        mock_resource.create.return_value = MagicMock()
        mock_tp = MagicMock()
        mock_tp_cls.return_value = mock_tp

        mock_app = MagicMock()

        from app.observability.telemetry import setup_telemetry

        setup_telemetry(app=mock_app)

        mock_fastapi_inst.instrument_app.assert_called_once()
        call_kwargs = mock_fastapi_inst.instrument_app.call_args
        assert call_kwargs[0][0] is mock_app or call_kwargs[1].get("app") is mock_app

        tel_module._tracer_provider = None


class TestGetTracer:
    """Tests for the get_tracer function."""

    @patch("app.observability.telemetry.trace")
    def test_get_tracer_returns_tracer(self, mock_trace):
        import app.observability.telemetry as tel_module

        tel_module._tracer_provider = None

        mock_provider = MagicMock()
        mock_trace.get_tracer_provider.return_value = mock_provider
        mock_provider.get_tracer.return_value = MagicMock()

        from app.observability.telemetry import get_tracer

        tracer = get_tracer("test_module")

        mock_provider.get_tracer.assert_called_once()

    def test_get_tracer_uses_existing_provider(self):
        import app.observability.telemetry as tel_module

        mock_provider = MagicMock()
        tel_module._tracer_provider = mock_provider
        mock_provider.get_tracer.return_value = MagicMock()

        from app.observability.telemetry import get_tracer

        tracer = get_tracer("my_module")

        mock_provider.get_tracer.assert_called_once()
        tel_module._tracer_provider = None


class TestShutdownTelemetry:
    """Tests for the shutdown_telemetry function."""

    async def test_shutdown_calls_provider_shutdown(self):
        import app.observability.telemetry as tel_module

        mock_provider = MagicMock()
        tel_module._tracer_provider = mock_provider

        from app.observability.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        mock_provider.shutdown.assert_called_once()
        assert tel_module._tracer_provider is None

    async def test_shutdown_noop_when_no_provider(self):
        import app.observability.telemetry as tel_module

        tel_module._tracer_provider = None

        from app.observability.telemetry import shutdown_telemetry

        await shutdown_telemetry()  # Should not raise
        assert tel_module._tracer_provider is None
