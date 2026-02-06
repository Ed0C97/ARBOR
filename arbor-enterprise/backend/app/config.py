"""A.R.B.O.R. Enterprise - Application Configuration.

Dual-database architecture:
- DATABASE_URL         → magazine_h182 on Render (READ-ONLY: brands, venues)
- ARBOR_DATABASE_URL   → arbor_db (READ-WRITE: enrichments, gold standard, etc.)

LLM stack:
- Google Gemini gemini-3-pro-preview  → Text + Vision (primary)
- Cohere embed-v4.0                   → Embedding
- Cohere rerank-v4.0-pro              → Reranking

SECURITY NOTE (TIER 1 - Point 1):
All sensitive credentials MUST be loaded from environment variables.
In production, use GCP Secret Manager via app.core.secrets_manager.
Never hardcode credentials in this file.
"""

import logging
import os
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    SECURITY: All sensitive fields have no default values and MUST be set
    via environment variables or secrets manager. The application will fail
    to start if required secrets are missing.
    """

    # Application
    app_name: str = "arbor-enterprise"
    app_env: str = "development"
    app_debug: bool = True
    app_version: str = "1.0.0"
    app_secret_key: str = ""  # REQUIRED - no default

    # GCP Project ID (for Secret Manager in production)
    gcp_project_id: str = ""

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # -----------------------------------------------------------------------
    # DATABASE 1: magazine_h182 — READ-ONLY (brands, venues)
    # SECURITY: Load from environment - no hardcoded credentials
    # -----------------------------------------------------------------------
    database_url: str = ""  # REQUIRED - no default
    database_ssl: bool = True

    # -----------------------------------------------------------------------
    # DATABASE 2: arbor_db — READ-WRITE (enrichments, gold standard, feedback)
    # Hosted on Render (Virginia)
    # SECURITY: Load from environment - no hardcoded credentials
    # -----------------------------------------------------------------------
    arbor_database_url: str = ""  # REQUIRED - no default
    arbor_database_ssl: bool = True

    # -----------------------------------------------------------------------
    # Connection Pool Settings (TIER 2 - Point 6)
    # -----------------------------------------------------------------------
    db_pool_size: int = 40
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800  # 30 minutes
    db_pool_pre_ping: bool = True

    # -----------------------------------------------------------------------
    # Qdrant — vector search
    # -----------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "entities_vectors"
    qdrant_prefer_grpc: bool = True  # TIER 7 - Point 29: gRPC over REST

    # -----------------------------------------------------------------------
    # Neo4j — knowledge graph
    # -----------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # REQUIRED in production

    # -----------------------------------------------------------------------
    # Redis — caching & session store
    # -----------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # -----------------------------------------------------------------------
    # GOOGLE GEMINI — Text + Vision (primary LLM)
    # SECURITY: Load from environment - no hardcoded credentials
    # -----------------------------------------------------------------------
    google_api_key: str = ""  # REQUIRED - no default
    google_model: str = "gemini-3-pro-preview"

    # -----------------------------------------------------------------------
    # COHERE — Embedding + Reranking
    # SECURITY: Load from environment - no hardcoded credentials
    # -----------------------------------------------------------------------
    cohere_api_key: str = ""  # REQUIRED - no default
    cohere_embedding_model: str = "embed-v4.0"
    cohere_rerank_model: str = "rerank-v4.0-pro"

    # -----------------------------------------------------------------------
    # Fallback LLM providers (optional, leave empty if not needed)
    # -----------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    azure_api_key: str = ""
    azure_api_base: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    mistral_api_key: str = ""
    mistral_model: str = "mistral-large-latest"
    mistral_embedding_model: str = "mistral-embed"

    # -----------------------------------------------------------------------
    # LLM routing — which provider for which task
    # -----------------------------------------------------------------------
    llm_primary_provider: str = "google"        # Text generation → Gemini
    llm_vision_provider: str = "google"          # Image analysis → Gemini
    llm_embedding_provider: str = "cohere"       # Embedding → Cohere
    llm_fast_provider: str = "google"            # Fast tasks → Gemini

    # -----------------------------------------------------------------------
    # Google Maps (for venue scraping)
    # -----------------------------------------------------------------------
    google_maps_api_key: str = ""

    # -----------------------------------------------------------------------
    # Auth0
    # -----------------------------------------------------------------------
    auth0_domain: str = ""
    auth0_api_audience: str = ""

    # -----------------------------------------------------------------------
    # Observability
    # -----------------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_dlq_topic: str = "arbor-dlq"  # TIER 3 - Point 13: Dead Letter Queue

    # Celery (TIER 7 - Point 34: Background Jobs)
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Temporal
    temporal_host: str = "localhost:7233"

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "arbor-api"

    # -----------------------------------------------------------------------
    # Circuit Breaker Settings (TIER 3 - Point 10)
    # -----------------------------------------------------------------------
    circuit_breaker_fail_max: int = 5
    circuit_breaker_reset_timeout: int = 60  # seconds

    # -----------------------------------------------------------------------
    # Retry Settings (TIER 3 - Point 12)
    # -----------------------------------------------------------------------
    retry_max_attempts: int = 3
    retry_initial_wait: float = 0.5
    retry_max_wait: float = 5.0

    # -----------------------------------------------------------------------
    # Cache Settings (TIER 4 - Point 17)
    # -----------------------------------------------------------------------
    semantic_cache_threshold: float = 0.90  # Lowered from 0.95 for better hit rate
    cache_ttl: int = 3600  # 1 hour

    # -----------------------------------------------------------------------
    # Rate Limiting (TIER 6 - Point 24)
    # -----------------------------------------------------------------------
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # -----------------------------------------------------------------------
    # Node Timeouts (TIER 3 - Point 11)
    # -----------------------------------------------------------------------
    timeout_intent_node: float = 2.0
    timeout_search_node: float = 5.0
    timeout_synthesis_node: float = 15.0
    timeout_total_request: float = 25.0

    model_config = {
        "env_file": (".env", "../.env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("app_secret_key", "database_url", "arbor_database_url")
    @classmethod
    def validate_required_in_production(cls, v: str, info) -> str:
        """Validate that critical secrets are set in production."""
        env = os.getenv("APP_ENV", "development")
        if env == "production" and not v:
            raise ValueError(
                f"{info.field_name} is required in production. "
                f"Set it via environment variable or GCP Secret Manager."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance.

    In production, validates that all required secrets are available.
    """
    settings = Settings()

    # Log warnings for missing optional but recommended settings
    if settings.app_env == "production":
        if not settings.google_api_key:
            logger.warning("GOOGLE_API_KEY not set - LLM features will not work")
        if not settings.cohere_api_key:
            logger.warning("COHERE_API_KEY not set - embedding/reranking will not work")

    return settings
