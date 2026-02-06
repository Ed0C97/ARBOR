"""Centralized Secrets Management.

Factory pattern for loading secrets from:
- Environment variables (local/development)
- Google Cloud Secret Manager (production)

TIER 1 - Point 1: Centralized Secrets Management
"""

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""

    pass


class SecretManager:
    """Centralized secret management with factory pattern.

    In local environment: reads from os.environ
    In production: reads from Google Cloud Secret Manager

    Usage:
        manager = get_secret_manager()
        api_key = manager.get_secret("COHERE_API_KEY")
    """

    def __init__(self, project_id: str | None = None):
        self.env = os.getenv("APP_ENV", "local").lower()
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self._client = None
        self._cache: dict[str, str] = {}

        if self.env == "production" and not self.project_id:
            logger.warning(
                "GCP_PROJECT_ID not set in production mode. "
                "Falling back to environment variables."
            )

    def _get_gcp_client(self):
        """Lazy-load GCP Secret Manager client."""
        if self._client is None:
            try:
                from google.cloud import secretmanager

                self._client = secretmanager.SecretManagerServiceClient()
                logger.info("GCP Secret Manager client initialized")
            except ImportError:
                logger.error(
                    "google-cloud-secret-manager not installed. "
                    "Run: pip install google-cloud-secret-manager"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to initialize GCP Secret Manager: {e}")
                raise
        return self._client

    def get_secret(
        self,
        secret_id: str,
        version_id: str = "latest",
        required: bool = True,
        default: str | None = None,
    ) -> str | None:
        """Get a secret value.

        Args:
            secret_id: The secret identifier (e.g., "COHERE_API_KEY")
            version_id: Secret version (default: "latest")
            required: If True, raises SecretNotFoundError when missing
            default: Default value if secret not found and not required

        Returns:
            The secret value as a string

        Raises:
            SecretNotFoundError: If required secret is not found
        """
        # Check cache first
        cache_key = f"{secret_id}:{version_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        value = None

        # Local/development: read from environment
        if self.env in ("local", "development", "test"):
            value = os.getenv(secret_id)
            if value:
                logger.debug(f"Secret '{secret_id}' loaded from environment")
        # Production: read from GCP Secret Manager
        elif self.env == "production" and self.project_id:
            try:
                client = self._get_gcp_client()
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
                response = client.access_secret_version(request={"name": name})
                value = response.payload.data.decode("UTF-8")
                logger.debug(f"Secret '{secret_id}' loaded from GCP Secret Manager")
            except Exception as e:
                logger.warning(f"Failed to fetch secret '{secret_id}' from GCP: {e}")
                # Fallback to environment variable
                value = os.getenv(secret_id)
                if value:
                    logger.info(f"Secret '{secret_id}' loaded from environment (fallback)")
        else:
            # Fallback to environment for any other case
            value = os.getenv(secret_id)

        # Handle missing secrets
        if value is None:
            if required:
                raise SecretNotFoundError(
                    f"Required secret '{secret_id}' not found. "
                    f"Set it in environment or GCP Secret Manager."
                )
            return default

        # Cache the value
        self._cache[cache_key] = value
        return value

    def get_secrets_batch(
        self, secret_ids: list[str], required: bool = True
    ) -> dict[str, str | None]:
        """Get multiple secrets at once.

        Args:
            secret_ids: List of secret identifiers
            required: If True, raises on any missing secret

        Returns:
            Dictionary mapping secret_id to value
        """
        return {
            secret_id: self.get_secret(secret_id, required=required) for secret_id in secret_ids
        }

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()
        logger.debug("Secret cache cleared")

    def validate_required_secrets(self, secret_ids: list[str]) -> bool:
        """Validate that all required secrets are available.

        Args:
            secret_ids: List of required secret identifiers

        Returns:
            True if all secrets are available

        Raises:
            SecretNotFoundError: If any required secret is missing
        """
        missing = []
        for secret_id in secret_ids:
            try:
                self.get_secret(secret_id, required=True)
            except SecretNotFoundError:
                missing.append(secret_id)

        if missing:
            raise SecretNotFoundError(
                f"Missing required secrets: {', '.join(missing)}. "
                f"Application cannot start without these secrets."
            )
        return True


# Singleton instance
_manager: SecretManager | None = None


@lru_cache
def get_secret_manager() -> SecretManager:
    """Return singleton SecretManager instance."""
    global _manager
    if _manager is None:
        _manager = SecretManager()
    return _manager


def get_secret(
    secret_id: str,
    version_id: str = "latest",
    required: bool = True,
    default: str | None = None,
) -> str | None:
    """Convenience function to get a secret.

    Args:
        secret_id: The secret identifier
        version_id: Secret version (default: "latest")
        required: If True, raises SecretNotFoundError when missing
        default: Default value if secret not found and not required

    Returns:
        The secret value
    """
    return get_secret_manager().get_secret(
        secret_id=secret_id,
        version_id=version_id,
        required=required,
        default=default,
    )


# List of sensitive secrets that should NEVER have default values in config
SENSITIVE_SECRETS = [
    "DATABASE_URL",
    "ARBOR_DATABASE_URL",
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "APP_SECRET_KEY",
    "AUTH0_CLIENT_SECRET",
    "REDIS_PASSWORD",
    "NEO4J_PASSWORD",
    "QDRANT_API_KEY",
]
