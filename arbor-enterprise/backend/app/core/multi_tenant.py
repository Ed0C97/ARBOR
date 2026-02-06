"""Multi-tenant architecture for A.R.B.O.R. Enterprise.

Provides tenant isolation, context management, and middleware for
multi-tenant request routing. Each tenant gets isolated data stores
(Qdrant collections, Redis key prefixes, PostgreSQL schemas) and
configurable rate limits based on their subscription tier.
"""

import contextvars
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tenant dataclass
# ---------------------------------------------------------------------------

VALID_TIERS = ("free", "pro", "enterprise")

DEFAULT_TIER_LIMITS: dict[str, dict[str, int]] = {
    "free": {"rpm": 10, "daily": 100, "max_entities": 1_000},
    "pro": {"rpm": 60, "daily": 5_000, "max_entities": 50_000},
    "enterprise": {"rpm": 1_000, "daily": 100_000, "max_entities": 10_000_000},
}

DEFAULT_TIER_FEATURES: dict[str, list[str]] = {
    "free": ["search", "browse"],
    "pro": ["search", "browse", "analytics", "export", "api_access"],
    "enterprise": [
        "search",
        "browse",
        "analytics",
        "export",
        "api_access",
        "custom_models",
        "sso",
        "audit_log",
        "dedicated_support",
    ],
}


@dataclass
class Tenant:
    """Represents a single tenant in the multi-tenant system."""

    id: str
    name: str
    slug: str
    tier: str = "free"
    config: dict[str, Any] = field(default_factory=dict)
    api_key_hash: str | None = None
    rate_limit_rpm: int | None = None
    rate_limit_daily: int | None = None
    max_entities: int | None = None
    features_enabled: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Apply tier defaults for any fields left unset."""
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier '{self.tier}'. Must be one of {VALID_TIERS}")

        tier_limits = DEFAULT_TIER_LIMITS[self.tier]
        if self.rate_limit_rpm is None:
            self.rate_limit_rpm = tier_limits["rpm"]
        if self.rate_limit_daily is None:
            self.rate_limit_daily = tier_limits["daily"]
        if self.max_entities is None:
            self.max_entities = tier_limits["max_entities"]
        if not self.features_enabled:
            self.features_enabled = list(DEFAULT_TIER_FEATURES[self.tier])

    @staticmethod
    def hash_api_key(raw_key: str) -> str:
        """Return a SHA-256 hex digest of a raw API key."""
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tenant context variable (async-safe, per-request)
# ---------------------------------------------------------------------------

_current_tenant_var: contextvars.ContextVar[Tenant | None] = contextvars.ContextVar(
    "current_tenant", default=None
)


class TenantContext:
    """Async-safe context holder for the current tenant.

    Uses ``contextvars.ContextVar`` so each async task / request handler
    sees its own tenant without interfering with concurrent requests.
    """

    @staticmethod
    def set_current_tenant(tenant: Tenant) -> contextvars.Token:
        """Bind *tenant* to the current async context.

        Returns a ``contextvars.Token`` that can be used to reset the
        variable to its previous value if needed.
        """
        logger.debug("Setting current tenant context: %s (%s)", tenant.slug, tenant.id)
        return _current_tenant_var.set(tenant)

    @staticmethod
    def get_current_tenant() -> Tenant | None:
        """Return the tenant bound to the current context, or ``None``."""
        return _current_tenant_var.get()

    @staticmethod
    def reset(token: contextvars.Token) -> None:
        """Reset the tenant context variable using a previously obtained token."""
        _current_tenant_var.reset(token)


# Convenience module-level aliases
set_current_tenant = TenantContext.set_current_tenant
get_current_tenant = TenantContext.get_current_tenant


# ---------------------------------------------------------------------------
# Tenant middleware (Starlette)
# ---------------------------------------------------------------------------

# Paths that do not require tenant resolution.
_SKIP_PATHS: set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class TenantMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that resolves the current tenant per-request.

    Resolution order:
    1. ``X-Tenant-ID`` header  -- looked up by tenant id.
    2. ``Authorization`` header with ``Bearer <api-key>`` -- looked up by
       hashed API key.

    If neither header yields an active tenant the request is rejected with
    a ``401`` JSON response.  Health-check endpoints are always allowed
    through without tenant context.
    """

    def __init__(self, app, manager: "TenantManager | None" = None) -> None:  # noqa: ANN001
        super().__init__(app)
        self._manager = manager

    @property
    def manager(self) -> "TenantManager":
        """Lazily resolve the TenantManager singleton."""
        if self._manager is None:
            self._manager = get_tenant_manager()
        return self._manager

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        """Process the request, resolve tenant, and set context."""
        # Allow health/infra endpoints through without tenant context.
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        tenant = self._resolve_tenant(request)

        if tenant is None:
            logger.warning(
                "Tenant resolution failed for %s %s",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Tenant identification required. "
                    "Provide X-Tenant-ID header or a valid API key."
                },
            )

        if not tenant.is_active:
            logger.warning("Request from inactive tenant: %s", tenant.slug)
            return JSONResponse(
                status_code=401,
                content={"detail": f"Tenant '{tenant.slug}' is deactivated."},
            )

        # Bind tenant to the async context for downstream handlers.
        token = set_current_tenant(tenant)
        try:
            response = await call_next(request)
            return response
        finally:
            TenantContext.reset(token)

    # -- helpers -------------------------------------------------------------

    def _resolve_tenant(self, request: Request) -> Tenant | None:
        """Attempt to resolve a Tenant from request headers."""
        # 1. Explicit tenant id header
        tenant_id = request.headers.get("x-tenant-id")
        if tenant_id:
            tenant = self.manager.get_tenant(tenant_id)
            if tenant is not None:
                return tenant

        # 2. API key in Authorization header (Bearer <key>)
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            raw_key = auth_header[7:].strip()
            if raw_key:
                tenant = self.manager.get_tenant_by_api_key(raw_key)
                if tenant is not None:
                    return tenant

        return None


# ---------------------------------------------------------------------------
# Tenant manager (singleton registry)
# ---------------------------------------------------------------------------


class TenantManager:
    """In-memory tenant registry with lookup and rate-limit helpers.

    Use the module-level :func:`get_tenant_manager` factory to obtain the
    singleton instance.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._api_key_index: dict[str, str] = {}  # hash -> tenant_id
        self._slug_index: dict[str, str] = {}  # slug -> tenant_id

        # Simple in-memory rate-limit counters: {tenant_id: [(timestamp, count)]}
        self._rpm_counters: dict[str, list[float]] = {}
        self._daily_counters: dict[str, list[float]] = {}

    # -- CRUD ----------------------------------------------------------------

    def register_tenant(self, tenant: Tenant) -> None:
        """Register a new tenant in the in-memory registry."""
        if tenant.id in self._tenants:
            logger.warning("Overwriting existing tenant: %s", tenant.id)

        self._tenants[tenant.id] = tenant

        # Index by slug
        self._slug_index[tenant.slug] = tenant.id

        # Index by API key hash (if provided)
        if tenant.api_key_hash:
            self._api_key_index[tenant.api_key_hash] = tenant.id

        logger.info(
            "Registered tenant '%s' (id=%s, tier=%s)",
            tenant.name,
            tenant.id,
            tenant.tier,
        )

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Look up a tenant by its unique identifier."""
        return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, raw_key: str) -> Tenant | None:
        """Look up a tenant by raw (unhashed) API key."""
        key_hash = Tenant.hash_api_key(raw_key)
        tenant_id = self._api_key_index.get(key_hash)
        if tenant_id is not None:
            return self._tenants.get(tenant_id)
        return None

    def get_tenant_by_slug(self, slug: str) -> Tenant | None:
        """Look up a tenant by its URL-friendly slug."""
        tenant_id = self._slug_index.get(slug)
        if tenant_id is not None:
            return self._tenants.get(tenant_id)
        return None

    def list_tenants(self, *, active_only: bool = False) -> list[Tenant]:
        """Return all registered tenants, optionally filtering by active status."""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return tenants

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Mark a tenant as inactive.

        Returns ``True`` if the tenant was found and deactivated, ``False``
        if no such tenant exists.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            logger.warning("Cannot deactivate unknown tenant: %s", tenant_id)
            return False

        tenant.is_active = False
        logger.info("Deactivated tenant '%s' (id=%s)", tenant.name, tenant.id)
        return True

    # -- Rate limiting -------------------------------------------------------

    def check_rate_limit(self, tenant_id: str) -> bool:
        """Check whether *tenant_id* is within its rate limits.

        Uses simple in-memory sliding-window counters.  Returns ``True``
        if the request is allowed, ``False`` if the tenant has exceeded
        its per-minute or daily quota.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            logger.warning("Rate-limit check for unknown tenant: %s", tenant_id)
            return False

        now = time.time()

        # --- Per-minute check ---
        rpm_window = self._rpm_counters.setdefault(tenant_id, [])
        # Purge entries older than 60 seconds
        cutoff_rpm = now - 60.0
        self._rpm_counters[tenant_id] = [ts for ts in rpm_window if ts > cutoff_rpm]
        if len(self._rpm_counters[tenant_id]) >= tenant.rate_limit_rpm:
            logger.warning(
                "Tenant %s exceeded RPM limit (%d)",
                tenant_id,
                tenant.rate_limit_rpm,
            )
            return False
        self._rpm_counters[tenant_id].append(now)

        # --- Daily check ---
        daily_window = self._daily_counters.setdefault(tenant_id, [])
        cutoff_daily = now - 86_400.0
        self._daily_counters[tenant_id] = [ts for ts in daily_window if ts > cutoff_daily]
        if len(self._daily_counters[tenant_id]) >= tenant.rate_limit_daily:
            logger.warning(
                "Tenant %s exceeded daily limit (%d)",
                tenant_id,
                tenant.rate_limit_daily,
            )
            return False
        self._daily_counters[tenant_id].append(now)

        return True


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_tenant_manager: TenantManager | None = None


def get_tenant_manager() -> TenantManager:
    """Return the global TenantManager singleton, creating it on first call."""
    global _tenant_manager  # noqa: PLW0603
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
        logger.info("Initialized TenantManager singleton")
    return _tenant_manager


# ---------------------------------------------------------------------------
# Tenant isolation utilities
# ---------------------------------------------------------------------------


class TenantIsolation:
    """Utilities for deriving tenant-scoped resource names.

    These helpers ensure that each tenant's data is stored in isolated
    namespaces across Qdrant, Redis, and PostgreSQL.
    """

    @staticmethod
    def get_collection_name(base_name: str, tenant: Tenant) -> str:
        """Return a Qdrant collection name scoped to *tenant*.

        Example::

            >>> t = Tenant(id="t1", name="Acme", slug="acme")
            >>> TenantIsolation.get_collection_name("entities", t)
            'acme_entities'
        """
        return f"{tenant.slug}_{base_name}"

    @staticmethod
    def get_cache_prefix(tenant: Tenant) -> str:
        """Return a Redis key prefix scoped to *tenant*.

        Example::

            >>> t = Tenant(id="t1", name="Acme", slug="acme")
            >>> TenantIsolation.get_cache_prefix(t)
            'tenant:acme:'
        """
        return f"tenant:{tenant.slug}:"

    @staticmethod
    def get_db_schema(tenant: Tenant) -> str:
        """Return a PostgreSQL schema name scoped to *tenant*.

        Example::

            >>> t = Tenant(id="t1", name="Acme", slug="acme")
            >>> TenantIsolation.get_db_schema(t)
            'tenant_acme'
        """
        return f"tenant_{tenant.slug}"
