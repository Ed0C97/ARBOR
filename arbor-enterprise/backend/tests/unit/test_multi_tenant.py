"""Unit tests for multi-tenant architecture and domain portability system.

Covers:
- app.core.multi_tenant: Tenant, TenantContext, TenantManager, TenantIsolation
- app.core.domain_portability: DomainConfig, DomainRegistry, DomainAdapter, DomainExporter
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.domain_portability import (
    DomainAdapter,
    DomainConfig,
    DomainExporter,
    DomainRegistry,
)
from app.core.multi_tenant import (
    DEFAULT_TIER_FEATURES,
    DEFAULT_TIER_LIMITS,
    Tenant,
    TenantContext,
    TenantIsolation,
    TenantManager,
)

# ═══════════════════════════════════════════════════════════════════════════
# TestTenant
# ═══════════════════════════════════════════════════════════════════════════


class TestTenant:
    """Tests for the Tenant dataclass."""

    def test_tenant_creation(self):
        """Creating a Tenant populates all expected fields."""
        tenant = Tenant(
            id="t-001",
            name="Acme Corp",
            slug="acme-corp",
            tier="pro",
        )

        assert tenant.id == "t-001"
        assert tenant.name == "Acme Corp"
        assert tenant.slug == "acme-corp"
        assert tenant.tier == "pro"
        assert tenant.is_active is True
        assert isinstance(tenant.created_at, float)
        assert tenant.config == {}

    def test_tier_defaults_applied(self):
        """A free-tier tenant receives the correct default rate limits and features."""
        tenant = Tenant(id="t-free", name="Free User", slug="free-user", tier="free")

        expected_limits = DEFAULT_TIER_LIMITS["free"]
        assert tenant.rate_limit_rpm == expected_limits["rpm"]
        assert tenant.rate_limit_daily == expected_limits["daily"]
        assert tenant.max_entities == expected_limits["max_entities"]
        assert tenant.features_enabled == DEFAULT_TIER_FEATURES["free"]

    def test_api_key_hashing(self):
        """hash_api_key produces a consistent SHA-256 hex digest."""
        raw_key = "arbor-secret-key-12345"
        expected = hashlib.sha256(raw_key.encode()).hexdigest()

        result = Tenant.hash_api_key(raw_key)

        assert result == expected
        assert len(result) == 64  # SHA-256 hex length
        # Deterministic: calling again gives the same hash
        assert Tenant.hash_api_key(raw_key) == result

    def test_inactive_tenant(self):
        """A tenant created with is_active=False remains inactive."""
        tenant = Tenant(
            id="t-inactive",
            name="Dormant Co",
            slug="dormant-co",
            is_active=False,
        )

        assert tenant.is_active is False


# ═══════════════════════════════════════════════════════════════════════════
# TestTenantContext
# ═══════════════════════════════════════════════════════════════════════════


class TestTenantContext:
    """Tests for the async-safe TenantContext holder."""

    def test_set_and_get_tenant(self):
        """Setting a tenant then getting it returns the same tenant."""
        tenant = Tenant(id="ctx-1", name="Context Co", slug="context-co")
        token = TenantContext.set_current_tenant(tenant)

        try:
            result = TenantContext.get_current_tenant()
            assert result is tenant
            assert result.id == "ctx-1"
        finally:
            TenantContext.reset(token)

    def test_get_returns_none_default(self):
        """Getting the current tenant without setting one returns None."""
        # Ensure we start from a clean state by saving and restoring
        previous = TenantContext.get_current_tenant()
        if previous is not None:
            # We are in a context where a tenant is set; create a clean
            # nested context by setting None manually and resetting after.
            pass

        # In a fresh contextvar scope, default is None.
        # We test this by resetting to a known state.
        tenant = Tenant(id="temp", name="Temp", slug="temp")
        token = TenantContext.set_current_tenant(tenant)
        TenantContext.reset(token)

        result = TenantContext.get_current_tenant()
        # After reset, should return whatever was there before (None in a clean run)
        assert result is None or result is previous

    def test_reset_restores_previous(self):
        """Resetting with a token restores the previous tenant context."""
        tenant_a = Tenant(id="a", name="A", slug="a")
        tenant_b = Tenant(id="b", name="B", slug="b")

        token_a = TenantContext.set_current_tenant(tenant_a)
        try:
            assert TenantContext.get_current_tenant() is tenant_a

            token_b = TenantContext.set_current_tenant(tenant_b)
            assert TenantContext.get_current_tenant() is tenant_b

            # Reset token_b -> should restore tenant_a
            TenantContext.reset(token_b)
            assert TenantContext.get_current_tenant() is tenant_a
        finally:
            TenantContext.reset(token_a)


# ═══════════════════════════════════════════════════════════════════════════
# TestTenantManager
# ═══════════════════════════════════════════════════════════════════════════


class TestTenantManager:
    """Tests for the TenantManager in-memory registry."""

    def _make_manager(self) -> TenantManager:
        """Return a fresh TenantManager instance."""
        return TenantManager()

    def test_register_and_retrieve(self):
        """Registering a tenant then retrieving by ID returns the same object."""
        mgr = self._make_manager()
        tenant = Tenant(id="t-reg", name="Registered", slug="registered")
        mgr.register_tenant(tenant)

        result = mgr.get_tenant("t-reg")
        assert result is tenant
        assert result.name == "Registered"

    def test_get_by_api_key(self):
        """Retrieving a tenant by raw API key works through hash lookup."""
        mgr = self._make_manager()
        raw_key = "super-secret-api-key-xyz"
        tenant = Tenant(
            id="t-api",
            name="API Tenant",
            slug="api-tenant",
            api_key_hash=Tenant.hash_api_key(raw_key),
        )
        mgr.register_tenant(tenant)

        result = mgr.get_tenant_by_api_key(raw_key)
        assert result is tenant
        assert result.id == "t-api"

        # Non-existent key returns None
        assert mgr.get_tenant_by_api_key("wrong-key") is None

    def test_get_by_slug(self):
        """Retrieving a tenant by slug returns the correct tenant."""
        mgr = self._make_manager()
        tenant = Tenant(id="t-slug", name="Slug Tenant", slug="slug-tenant")
        mgr.register_tenant(tenant)

        result = mgr.get_tenant_by_slug("slug-tenant")
        assert result is tenant

        assert mgr.get_tenant_by_slug("nonexistent") is None

    def test_list_tenants(self):
        """list_tenants returns all registered tenants."""
        mgr = self._make_manager()
        t1 = Tenant(id="t1", name="One", slug="one")
        t2 = Tenant(id="t2", name="Two", slug="two")
        t3 = Tenant(id="t3", name="Three", slug="three", is_active=False)

        mgr.register_tenant(t1)
        mgr.register_tenant(t2)
        mgr.register_tenant(t3)

        all_tenants = mgr.list_tenants()
        assert len(all_tenants) == 3

        active_only = mgr.list_tenants(active_only=True)
        assert len(active_only) == 2
        assert all(t.is_active for t in active_only)

    def test_deactivate_tenant(self):
        """Deactivating a tenant sets is_active to False and returns True."""
        mgr = self._make_manager()
        tenant = Tenant(id="t-deact", name="Deactivate Me", slug="deactivate-me")
        mgr.register_tenant(tenant)

        assert tenant.is_active is True

        result = mgr.deactivate_tenant("t-deact")
        assert result is True
        assert tenant.is_active is False

        # Deactivating an unknown tenant returns False
        assert mgr.deactivate_tenant("nonexistent") is False


# ═══════════════════════════════════════════════════════════════════════════
# TestTenantIsolation
# ═══════════════════════════════════════════════════════════════════════════


class TestTenantIsolation:
    """Tests for TenantIsolation resource-naming utilities."""

    def _make_tenant(self, slug: str = "acme") -> Tenant:
        return Tenant(id="t-iso", name="Isolation Test", slug=slug)

    def test_collection_name_prefixed(self):
        """get_collection_name prefixes the base name with the tenant slug."""
        tenant = self._make_tenant("acme")
        result = TenantIsolation.get_collection_name("entities", tenant)
        assert result == "acme_entities"

    def test_cache_prefix_format(self):
        """get_cache_prefix returns 'tenant:{slug}:' format."""
        tenant = self._make_tenant("acme")
        result = TenantIsolation.get_cache_prefix(tenant)
        assert result == "tenant:acme:"

    def test_db_schema_format(self):
        """get_db_schema returns 'tenant_{slug}' format."""
        tenant = self._make_tenant("acme")
        result = TenantIsolation.get_db_schema(tenant)
        assert result == "tenant_acme"


# ═══════════════════════════════════════════════════════════════════════════
# TestDomainConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainConfig:
    """Tests for DomainConfig and the pre-built fashion domain."""

    def _make_registry(self) -> DomainRegistry:
        """Return a fresh DomainRegistry (pre-loaded with fashion)."""
        return DomainRegistry()

    def test_fashion_domain_preregistered(self):
        """The fashion domain exists in the registry upon initialization."""
        registry = self._make_registry()
        config = registry.get_domain("fashion")
        assert config.domain_id == "fashion"
        assert config.name == "Fashion & Style"

    def test_domain_has_vibe_dimensions(self):
        """The fashion domain has the expected vibe dimensions."""
        registry = self._make_registry()
        config = registry.get_domain("fashion")

        assert isinstance(config.vibe_dimensions, list)
        assert len(config.vibe_dimensions) > 0

        expected_dims = {
            "formality",
            "craftsmanship",
            "price_value",
            "atmosphere",
            "service_quality",
            "exclusivity",
            "modernity",
        }
        assert set(config.vibe_dimensions) == expected_dims

    def test_domain_has_categories(self):
        """The fashion domain has a non-empty categories list."""
        registry = self._make_registry()
        config = registry.get_domain("fashion")

        assert isinstance(config.categories, list)
        assert len(config.categories) > 0
        assert "Designer" in config.categories
        assert "Boutique" in config.categories


# ═══════════════════════════════════════════════════════════════════════════
# TestDomainRegistry
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainRegistry:
    """Tests for the DomainRegistry singleton-style registry."""

    def _make_registry(self) -> DomainRegistry:
        return DomainRegistry()

    def _make_config(self, domain_id: str = "restaurants") -> DomainConfig:
        return DomainConfig(
            domain_id=domain_id,
            name="Restaurants & Dining",
            description="Restaurants, cafes, and dining venues.",
            vibe_dimensions=["ambiance", "cuisine_quality", "service"],
            categories=["Fine Dining", "Casual", "Fast Food"],
        )

    def test_register_and_retrieve(self):
        """Registering a new domain and retrieving it returns the same config."""
        registry = self._make_registry()
        config = self._make_config("restaurants")
        registry.register_domain(config)

        result = registry.get_domain("restaurants")
        assert result is config
        assert result.domain_id == "restaurants"
        assert result.name == "Restaurants & Dining"

    def test_active_domain_default(self):
        """The default active domain is 'fashion'."""
        registry = self._make_registry()
        active = registry.get_active_domain()
        assert active.domain_id == "fashion"

    def test_set_active_domain(self):
        """Changing the active domain updates what get_active_domain returns."""
        registry = self._make_registry()
        config = self._make_config("restaurants")
        registry.register_domain(config)

        registry.set_active_domain("restaurants")
        active = registry.get_active_domain()
        assert active.domain_id == "restaurants"

        # Setting back to fashion works too
        registry.set_active_domain("fashion")
        assert registry.get_active_domain().domain_id == "fashion"

    def test_list_domains(self):
        """list_domains returns all registered domains sorted by domain_id."""
        registry = self._make_registry()
        registry.register_domain(self._make_config("restaurants"))
        registry.register_domain(
            DomainConfig(
                domain_id="art",
                name="Art Galleries",
                description="Art galleries and exhibitions.",
            )
        )

        domains = registry.list_domains()
        assert len(domains) == 3  # fashion + restaurants + art
        domain_ids = [d.domain_id for d in domains]
        assert domain_ids == sorted(domain_ids)


# ═══════════════════════════════════════════════════════════════════════════
# TestDomainAdapter
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainAdapter:
    """Tests for the DomainAdapter data-mapping and validation utilities."""

    def _fashion_config(self) -> DomainConfig:
        """Return a minimal fashion-like domain config for tests."""
        return DomainConfig(
            domain_id="fashion",
            name="Fashion & Style",
            description="Fashion brands and boutiques.",
            entity_schema={
                "required": ["name", "slug", "category"],
                "optional": ["city", "website", "latitude"],
                "field_types": {
                    "name": "str",
                    "slug": "str",
                    "category": "str",
                    "city": "str",
                    "website": "str",
                    "latitude": "float",
                },
            },
            vibe_dimensions=["formality", "craftsmanship", "price_value"],
            categories=["Designer", "Boutique", "Concept Store"],
        )

    def test_adapt_entity_extracts_fields(self):
        """adapt_entity extracts and coerces fields from raw data per the schema."""
        adapter = DomainAdapter()
        config = self._fashion_config()

        raw_data = {
            "name": "Maison Artisan",
            "slug": "maison-artisan",
            "category": "Boutique",
            "city": "Paris",
            "latitude": "48.8566",
            "unrecognized_field": "should be dropped",
        }

        result = adapter.adapt_entity(raw_data, config)

        assert result["name"] == "Maison Artisan"
        assert result["slug"] == "maison-artisan"
        assert result["category"] == "Boutique"
        assert result["city"] == "Paris"
        assert result["latitude"] == 48.8566  # coerced str -> float
        assert "unrecognized_field" not in result

    def test_validate_entity_valid(self):
        """A valid entity passes validation with no errors."""
        adapter = DomainAdapter()
        config = self._fashion_config()

        entity = {
            "name": "Le Bon Marche",
            "slug": "le-bon-marche",
            "category": "Concept Store",
            "city": "Paris",
        }

        is_valid, errors = adapter.validate_entity(entity, config)
        assert is_valid is True
        assert errors == []

    def test_validate_entity_missing_required(self):
        """An entity missing a required field fails validation."""
        adapter = DomainAdapter()
        config = self._fashion_config()

        entity = {
            "name": "Incomplete Store",
            # "slug" is missing
            "category": "Designer",
        }

        is_valid, errors = adapter.validate_entity(entity, config)
        assert is_valid is False
        assert len(errors) >= 1
        assert any("slug" in e for e in errors)

    def test_adapt_vibe_dimensions(self):
        """adapt_vibe_dimensions translates scores between domains correctly."""
        adapter = DomainAdapter()

        source = DomainConfig(
            domain_id="fashion",
            name="Fashion",
            description="Fashion domain.",
            vibe_dimensions=["formality", "craftsmanship", "price_value"],
        )
        target = DomainConfig(
            domain_id="restaurants",
            name="Restaurants",
            description="Restaurant domain.",
            vibe_dimensions=["formality", "ambiance", "cuisine_quality"],
        )

        vibe_dna = {"formality": 80, "craftsmanship": 90, "price_value": 70}

        result = adapter.adapt_vibe_dimensions(vibe_dna, source, target)

        # "formality" is shared -- should carry over
        assert result["formality"] == 80
        # "ambiance" and "cuisine_quality" are target-only -- default to 50
        assert result["ambiance"] == 50
        assert result["cuisine_quality"] == 50
        # Source-only dimensions are dropped
        assert "craftsmanship" not in result
        assert "price_value" not in result


# ═══════════════════════════════════════════════════════════════════════════
# TestDomainExporter
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainExporter:
    """Tests for the DomainExporter serialization and deserialization."""

    def test_export_roundtrip(self):
        """Exporting a domain config then importing it back produces an equivalent config."""
        # Create a fresh registry with the pre-built fashion domain
        with (
            patch("app.core.domain_portability._registry", None),
            patch("app.core.domain_portability.get_domain_registry") as mock_get_registry,
        ):
            registry = DomainRegistry()
            mock_get_registry.return_value = registry

            exporter = DomainExporter()

            exported = exporter.export_domain_config("fashion")
            assert isinstance(exported, dict)
            assert exported["domain_id"] == "fashion"
            assert "vibe_dimensions" in exported
            assert "categories" in exported

            imported = exporter.import_domain_config(exported)
            assert isinstance(imported, DomainConfig)
            assert imported.domain_id == exported["domain_id"]
            assert imported.name == exported["name"]
            assert imported.description == exported["description"]
            assert imported.vibe_dimensions == exported["vibe_dimensions"]
            assert imported.categories == exported["categories"]

    def test_import_validates_required_keys(self):
        """Importing a dict missing 'domain_id' raises ValueError."""
        exporter = DomainExporter()

        incomplete_data = {
            "name": "Missing Domain ID",
            "description": "This should fail.",
        }

        with pytest.raises(ValueError, match="missing required keys"):
            exporter.import_domain_config(incomplete_data)
