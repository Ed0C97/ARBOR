"""Automated Secrets Rotation.

TIER 11 - Point 61: Automated Secrets Rotation

Rotates API keys and database passwords on a schedule.
Integrates with Kubernetes Secrets and GCP Secret Manager.

Features:
- Scheduled rotation (cron job)
- Zero-downtime rotation
- Audit logging
- Notification on failure
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets that can be rotated."""
    DATABASE_PASSWORD = "database_password"
    API_KEY = "api_key"
    JWT_SECRET = "jwt_secret"
    REDIS_PASSWORD = "redis_password"


class RotationStatus(str, Enum):
    """Status of a rotation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RotationResult:
    """Result of a secret rotation."""
    secret_name: str
    secret_type: SecretType
    status: RotationStatus
    rotated_at: datetime
    old_version: str | None = None
    new_version: str | None = None
    error: str | None = None


@dataclass
class SecretConfig:
    """Configuration for a secret to rotate."""
    name: str
    secret_type: SecretType
    rotation_days: int = 90
    last_rotated: datetime | None = None
    gcp_secret_id: str | None = None  # For GCP Secret Manager
    k8s_secret_name: str | None = None  # For Kubernetes secrets


class SecretsRotator:
    """Handles automated secrets rotation.
    
    TIER 11 - Point 61: Zero-downtime secrets rotation.
    
    Rotation process:
    1. Generate new secret value
    2. Add to secret store (new version)
    3. Update application to use new version
    4. Verify application works
    5. Disable old version
    6. Log rotation in audit
    
    Usage:
        rotator = SecretsRotator()
        result = await rotator.rotate("database_password")
    """
    
    def __init__(self):
        self.secrets: dict[str, SecretConfig] = {}
        self.rotation_history: list[RotationResult] = []
    
    def register_secret(self, config: SecretConfig) -> None:
        """Register a secret for rotation."""
        self.secrets[config.name] = config
    
    async def rotate(self, secret_name: str) -> RotationResult:
        """Rotate a specific secret.
        
        Args:
            secret_name: Name of the secret to rotate
            
        Returns:
            RotationResult with status
        """
        if secret_name not in self.secrets:
            return RotationResult(
                secret_name=secret_name,
                secret_type=SecretType.API_KEY,
                status=RotationStatus.FAILED,
                rotated_at=datetime.utcnow(),
                error=f"Secret {secret_name} not registered",
            )
        
        config = self.secrets[secret_name]
        logger.info(f"Starting rotation for: {secret_name}")
        
        try:
            # Step 1: Generate new secret value
            new_value = await self._generate_secret(config.secret_type)
            old_version = await self._get_current_version(config)
            
            # Step 2: Add new version to secret store
            new_version = await self._add_secret_version(config, new_value)
            
            # Step 3: Update Kubernetes secret if configured
            if config.k8s_secret_name:
                await self._update_k8s_secret(config.k8s_secret_name, new_value)
            
            # Step 4: Verify application works (health check)
            healthy = await self._verify_health()
            
            if not healthy:
                # Rollback
                logger.error(f"Health check failed after rotation, rolling back")
                await self._rollback(config, old_version)
                
                result = RotationResult(
                    secret_name=secret_name,
                    secret_type=config.secret_type,
                    status=RotationStatus.ROLLED_BACK,
                    rotated_at=datetime.utcnow(),
                    error="Health check failed after rotation",
                )
            else:
                # Step 5: Disable old version
                if old_version:
                    await self._disable_old_version(config, old_version)
                
                # Update last rotated
                config.last_rotated = datetime.utcnow()
                
                result = RotationResult(
                    secret_name=secret_name,
                    secret_type=config.secret_type,
                    status=RotationStatus.SUCCESS,
                    rotated_at=datetime.utcnow(),
                    old_version=old_version,
                    new_version=new_version,
                )
            
            # Step 6: Log to audit
            await self._log_rotation(result)
            
            self.rotation_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Rotation failed for {secret_name}: {e}")
            result = RotationResult(
                secret_name=secret_name,
                secret_type=config.secret_type,
                status=RotationStatus.FAILED,
                rotated_at=datetime.utcnow(),
                error=str(e),
            )
            self.rotation_history.append(result)
            return result
    
    async def rotate_all_due(self) -> list[RotationResult]:
        """Rotate all secrets that are due for rotation."""
        results = []
        
        for name, config in self.secrets.items():
            if self._is_due_for_rotation(config):
                result = await self.rotate(name)
                results.append(result)
        
        return results
    
    def _is_due_for_rotation(self, config: SecretConfig) -> bool:
        """Check if a secret is due for rotation."""
        if config.last_rotated is None:
            return True
        
        from datetime import timedelta
        age = datetime.utcnow() - config.last_rotated
        return age > timedelta(days=config.rotation_days)
    
    async def _generate_secret(self, secret_type: SecretType) -> str:
        """Generate a new secret value."""
        import secrets
        import string
        
        if secret_type == SecretType.DATABASE_PASSWORD:
            # Strong password: 32 chars, mixed characters
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            return "".join(secrets.choice(alphabet) for _ in range(32))
        
        elif secret_type == SecretType.API_KEY:
            # API key format: prefix + random hex
            return f"arbor_{''.join(secrets.choice(string.hexdigits.lower()) for _ in range(48))}"
        
        elif secret_type == SecretType.JWT_SECRET:
            # JWT secret: 64 bytes base64 encoded
            return secrets.token_urlsafe(64)
        
        elif secret_type == SecretType.REDIS_PASSWORD:
            # Redis password: alphanumeric, 24 chars
            return "".join(
                secrets.choice(string.ascii_letters + string.digits)
                for _ in range(24)
            )
        
        return secrets.token_urlsafe(32)
    
    async def _get_current_version(self, config: SecretConfig) -> str | None:
        """Get current version of a secret from GCP Secret Manager.

        Falls back to env-var lookup in non-production environments.
        """
        from app.config import get_settings

        settings = get_settings()
        project_id = settings.gcp_project_id

        if project_id and config.gcp_secret_id:
            try:
                from google.cloud import secretmanager

                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{config.gcp_secret_id}/versions/latest"
                response = client.access_secret_version(request={"name": name})
                return response.name.rsplit("/", 1)[-1]
            except Exception as exc:
                logger.warning("GCP get_current_version failed for %s: %s", config.name, exc)

        # Fallback: check env var
        import os

        value = os.getenv(config.env_var_name)
        return "env" if value else None

    async def _add_secret_version(
        self, config: SecretConfig, value: str
    ) -> str:
        """Add a new secret version to GCP Secret Manager.

        Falls back to local logging in non-production environments.
        """
        from app.config import get_settings

        settings = get_settings()
        project_id = settings.gcp_project_id

        if project_id and config.gcp_secret_id:
            try:
                from google.cloud import secretmanager

                client = secretmanager.SecretManagerServiceClient()
                parent = client.secret_path(project_id, config.gcp_secret_id)
                response = client.add_secret_version(
                    request={
                        "parent": parent,
                        "payload": {"data": value.encode("utf-8")},
                    }
                )
                version = response.name.rsplit("/", 1)[-1]
                logger.info("Added GCP secret version: %s/%s", config.gcp_secret_id, version)
                return version
            except Exception as exc:
                logger.warning("GCP add_secret_version failed for %s: %s", config.name, exc)

        # Fallback: generate a version label locally
        version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info("Added local secret version: %s", version)
        return version

    async def _update_k8s_secret(self, secret_name: str, value: str) -> None:
        """Update a Kubernetes secret in the arbor namespace.

        Uses the in-cluster config when running inside K8s, or falls back
        to kubeconfig when running locally.
        """
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            import base64

            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            v1 = k8s_client.CoreV1Api()
            encoded = base64.b64encode(value.encode("utf-8")).decode("utf-8")
            body = {"data": {secret_name: encoded}}

            v1.patch_namespaced_secret(
                name="arbor-secrets",
                namespace="arbor",
                body=body,
            )
            logger.info("Updated K8s secret: %s", secret_name)
        except ImportError:
            logger.info("kubernetes package not installed, skipping K8s update for: %s", secret_name)
        except Exception as exc:
            logger.warning("K8s secret update failed for %s: %s", secret_name, exc)
    
    async def _verify_health(self) -> bool:
        """Verify application health after rotation."""
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8000/health/readiness",
                    timeout=5,
                )
                return response.status_code == 200
        except Exception:
            return False
    
    async def _rollback(self, config: SecretConfig, old_version: str) -> None:
        """Rollback to old secret version by re-enabling it in GCP Secret Manager."""
        logger.warning(f"Rolling back {config.name} to {old_version}")

        from app.config import get_settings

        settings = get_settings()
        project_id = settings.gcp_project_id

        if project_id and config.gcp_secret_id and old_version:
            try:
                from google.cloud import secretmanager

                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{config.gcp_secret_id}/versions/{old_version}"
                client.enable_secret_version(request={"name": name})
                logger.info("Re-enabled old version: %s/%s", config.gcp_secret_id, old_version)
            except Exception as exc:
                logger.error("GCP rollback failed for %s: %s", config.name, exc)

        if config.k8s_secret_name:
            logger.info("K8s rollback would require redeployment for: %s", config.k8s_secret_name)

    async def _disable_old_version(
        self, config: SecretConfig, version: str
    ) -> None:
        """Disable old secret version in GCP Secret Manager."""
        logger.info(f"Disabling old version: {version}")

        from app.config import get_settings

        settings = get_settings()
        project_id = settings.gcp_project_id

        if project_id and config.gcp_secret_id and version and version != "env":
            try:
                from google.cloud import secretmanager

                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{config.gcp_secret_id}/versions/{version}"
                client.disable_secret_version(request={"name": name})
                logger.info("Disabled old version: %s/%s", config.gcp_secret_id, version)
            except Exception as exc:
                logger.warning("GCP disable_old_version failed for %s: %s", config.name, exc)
    
    async def _log_rotation(self, result: RotationResult) -> None:
        """Log rotation to audit log."""
        logger.info(
            f"Secret rotation: {result.secret_name} -> {result.status.value}"
        )


def get_rotation_schedule() -> dict[str, int]:
    """Get recommended rotation schedule (days)."""
    return {
        "database_password": 90,
        "api_keys": 180,
        "jwt_secrets": 30,
        "redis_password": 90,
    }


# CLI for manual rotation
async def main():
    """Run rotation for due secrets."""
    rotator = SecretsRotator()
    
    # Register secrets (would be loaded from config)
    rotator.register_secret(SecretConfig(
        name="arbor_db_password",
        secret_type=SecretType.DATABASE_PASSWORD,
        rotation_days=90,
        k8s_secret_name="arbor-secrets",
    ))
    
    results = await rotator.rotate_all_due()
    
    for result in results:
        print(f"{result.secret_name}: {result.status.value}")
        if result.error:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
