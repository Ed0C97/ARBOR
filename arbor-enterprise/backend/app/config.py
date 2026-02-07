"""A.R.B.O.R. Enterprise - Application Configuration.

Dual-database architecture:
- SOURCE_DATABASE_URL  → Client's database (READ-ONLY: any tables)
- ARBOR_DATABASE_URL   → arbor_db (READ-WRITE: enrichments, gold standard, etc.)

SCHEMA-AGNOSTIC DESIGN:
The system reads table schemas from SOURCE_SCHEMA_CONFIG (JSON in ENV).
This allows ARBOR to work with ANY database/tables/columns without code changes.

LLM stack:
- Google Gemini gemini-3-pro-preview  → Text + Vision (primary)
- Cohere embed-v4.0                   → Embedding
- Cohere rerank-v4.0-pro              → Reranking

SECURITY NOTE (TIER 1 - Point 1):
All sensitive credentials MUST be loaded from environment variables.
In production, use GCP Secret Manager via app.core.secrets_manager.
Never hardcode credentials in this file.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA-AGNOSTIC CONFIGURATION
# =============================================================================


@dataclass
class EntityTypeConfig:
    """Configuration for a single entity type (table) in the source database.

    This defines how ARBOR maps a source table to its internal UnifiedEntity format.

    Example JSON in SOURCE_SCHEMA_CONFIG:
    {
        "entity_type": "product",
        "table_name": "products",
        "id_column": "id",
        "required_mappings": {"name": "product_name"},
        "optional_mappings": {"category": "cat_id", "city": null},
        "text_fields_for_embedding": ["product_name", "description"]
    }
    """

    entity_type: str  # Internal name: "product", "store", etc.
    table_name: str  # Actual table name in source DB
    id_column: str = "id"  # Primary key column

    # Maps ARBOR field → source column name (or None if not available)
    required_mappings: dict = field(default_factory=lambda: {"name": "name"})
    optional_mappings: dict = field(default_factory=dict)

    # Columns to concatenate for embedding generation
    text_fields_for_embedding: list = field(default_factory=lambda: ["name", "description"])

    # Filter for active/published records (optional)
    active_filter_column: str | None = None
    active_filter_value: str | bool | None = True

    @property
    def all_mappings(self) -> dict:
        """Return combined required + optional mappings."""
        return {**self.required_mappings, **self.optional_mappings}

    @classmethod
    def from_dict(cls, data: dict) -> "EntityTypeConfig":
        """Create from dictionary (parsed from JSON)."""
        return cls(
            entity_type=data["entity_type"],
            table_name=data["table_name"],
            id_column=data.get("id_column", "id"),
            required_mappings=data.get("required_mappings", {"name": "name"}),
            optional_mappings=data.get("optional_mappings", {}),
            text_fields_for_embedding=data.get("text_fields_for_embedding", ["name"]),
            active_filter_column=data.get("active_filter_column"),
            active_filter_value=data.get("active_filter_value", True),
        )


def parse_schema_config(config_str: str | None, config_file: str | None = None) -> list[EntityTypeConfig]:
    """Parse SOURCE_SCHEMA_CONFIG from ENV or file.

    Args:
        config_str: JSON string from SOURCE_SCHEMA_CONFIG env var
        config_file: Path to JSON file from SOURCE_SCHEMA_CONFIG_FILE env var

    Returns:
        List of EntityTypeConfig objects

    Example ENV:
        SOURCE_SCHEMA_CONFIG='[{"entity_type":"product","table_name":"products",...}]'
    """
    # Try file first
    if config_file:
        path = Path(config_file)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return [EntityTypeConfig.from_dict(item) for item in data]
        else:
            logger.warning(f"Schema config file not found: {config_file}")

    # Fall back to inline JSON
    if config_str:
        try:
            data = json.loads(config_str)
            return [EntityTypeConfig.from_dict(item) for item in data]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SOURCE_SCHEMA_CONFIG: {e}")

    # Default: backwards-compatible brands/venues schema
    logger.info("No SOURCE_SCHEMA_CONFIG found, using default brands/venues schema")
    return [
        EntityTypeConfig(
            entity_type="brand",
            table_name="brands",
            id_column="id",
            required_mappings={"name": "name", "slug": "slug", "category": "category"},
            optional_mappings={
                "city": "city",
                "region": "region",
                "country": "country",
                "address": "address",
                "latitude": "latitude",
                "longitude": "longitude",
                "website": "website",
                "instagram": "instagram",
                "email": "email",
                "phone": "phone",
                "description": "description",
                "specialty": "specialty",
                "notes": "notes",
                "gender": "gender",
                "style": "style",
                "rating": "rating",
                "is_featured": "is_featured",
                "is_active": "is_active",
                "priority": "priority",
            },
            text_fields_for_embedding=["name", "description", "specialty", "notes"],
            active_filter_column="is_active",
            active_filter_value=True,
        ),
        EntityTypeConfig(
            entity_type="venue",
            table_name="venues",
            id_column="id",
            required_mappings={"name": "name", "slug": "slug", "category": "category"},
            optional_mappings={
                "city": "city",
                "region": "region",
                "country": "country",
                "address": "address",
                "latitude": "latitude",
                "longitude": "longitude",
                "website": "website",
                "instagram": "instagram",
                "email": "email",
                "phone": "phone",
                "description": "description",
                "notes": "notes",
                "gender": "gender",
                "style": "style",
                "rating": "rating",
                "price_range": "price_range",
                "is_featured": "is_featured",
                "is_active": "is_active",
                "priority": "priority",
            },
            text_fields_for_embedding=["name", "description", "notes"],
            active_filter_column="is_active",
            active_filter_value=True,
        ),
    ]


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
    # SOURCE DATABASE — READ-ONLY (client's data)
    # SECURITY: Load from environment - no hardcoded credentials
    # -----------------------------------------------------------------------
    source_database_url: str = ""  # REQUIRED - client's database URL
    database_url: str = ""  # DEPRECATED: alias for source_database_url
    database_ssl: bool = True

    # Schema configuration (JSON or file path)
    source_schema_config: str = ""  # JSON array of entity type configs
    source_schema_config_file: str = ""  # Path to JSON config file

    # Domain profile configuration (JSON or file path)
    # Defines vibe dimensions, categories, prompts, and persona for the
    # discovery domain.  If not set, the system uses universal defaults.
    # Generate one using: python -m app.core.domain_profile_generator
    domain_profile_config: str = ""  # JSON domain profile
    domain_profile_config_file: str = ""  # Path to JSON profile file

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
    llm_primary_provider: str = "google"  # Text generation → Gemini
    llm_vision_provider: str = "google"  # Image analysis → Gemini
    llm_embedding_provider: str = "cohere"  # Embedding → Cohere
    llm_fast_provider: str = "google"  # Fast tasks → Gemini

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

    # Cached entity type configs (set after initialization)
    _entity_type_configs: list["EntityTypeConfig"] | None = None

    def get_entity_type_configs(self) -> list["EntityTypeConfig"]:
        """Get parsed entity type configurations.

        Returns cached configs or parses from ENV on first call.
        """
        if self._entity_type_configs is None:
            self._entity_type_configs = parse_schema_config(
                self.source_schema_config or None,
                self.source_schema_config_file or None,
            )
        return self._entity_type_configs

    def get_entity_types(self) -> list[str]:
        """Get list of configured entity type names."""
        return [config.entity_type for config in self.get_entity_type_configs()]

    def get_entity_config(self, entity_type: str) -> "EntityTypeConfig | None":
        """Get configuration for a specific entity type."""
        for config in self.get_entity_type_configs():
            if config.entity_type == entity_type:
                return config
        return None

    def get_domain_profile(self) -> dict | None:
        """Load domain profile from ENV or file, if configured.

        Returns:
            Parsed JSON dict ready for ``DomainExporter.import_domain_config``,
            or ``None`` if no profile is configured.
        """
        # Try file first
        if self.domain_profile_config_file:
            path = Path(self.domain_profile_config_file)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    logger.info(f"Loaded domain profile from file: {path}")
                    return data
            else:
                logger.warning(f"Domain profile file not found: {path}")

        # Inline JSON
        if self.domain_profile_config:
            try:
                data = json.loads(self.domain_profile_config)
                logger.info("Loaded domain profile from DOMAIN_PROFILE_CONFIG env")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse DOMAIN_PROFILE_CONFIG: {e}")

        return None

    def get_effective_source_db_url(self) -> str:
        """Get the effective source database URL (supports legacy database_url)."""
        return self.source_database_url or self.database_url

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
