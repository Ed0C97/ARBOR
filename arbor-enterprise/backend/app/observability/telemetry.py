"""OpenTelemetry setup for distributed tracing across A.R.B.O.R. Enterprise."""

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_tracer_provider: TracerProvider | None = None


def setup_telemetry(app=None) -> TracerProvider:
    """Initialize OpenTelemetry with OTLP exporter and FastAPI instrumentation.

    Args:
        app: Optional FastAPI application instance to instrument.

    Returns:
        The configured TracerProvider.
    """
    global _tracer_provider

    if _tracer_provider is not None:
        return _tracer_provider

    resource = Resource.create(
        {
            "service.name": settings.app_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.app_env,
        }
    )

    _tracer_provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        insecure=settings.app_env == "development",
    )

    span_processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    _tracer_provider.add_span_processor(span_processor)

    trace.set_tracer_provider(_tracer_provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,/",
            tracer_provider=_tracer_provider,
        )
        logger.info("FastAPI OpenTelemetry instrumentation enabled")

    logger.info(
        "OpenTelemetry initialized: endpoint=%s, service=%s",
        settings.otel_exporter_otlp_endpoint,
        settings.app_name,
    )
    return _tracer_provider


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer from the global provider.

    Args:
        name: Logical name for the tracer, typically __name__.

    Returns:
        An OpenTelemetry Tracer instance.
    """
    provider = _tracer_provider or trace.get_tracer_provider()
    return provider.get_tracer(name, settings.app_version)


async def shutdown_telemetry() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("OpenTelemetry tracer provider shut down")
