"""Unit tests for API request/response validation logic."""

from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Validation helpers (inline to avoid heavy imports)
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "restaurant",
    "bar",
    "hotel",
    "shop",
    "cafe",
    "gallery",
    "spa",
    "club",
    "tailoring",
    "clothing",
}

VALID_STATUSES = {"pending", "vetted", "selected", "rejected"}


def validate_entity_payload(payload: dict) -> list[str]:
    """Validate an entity creation payload. Returns list of errors."""
    errors = []

    if not payload.get("name"):
        errors.append("name is required")
    elif len(payload["name"]) > 200:
        errors.append("name must be <= 200 characters")

    if not payload.get("category"):
        errors.append("category is required")
    elif payload["category"] not in VALID_CATEGORIES:
        errors.append(f"invalid category: {payload['category']}")

    tier = payload.get("price_tier")
    if tier is not None:
        if not isinstance(tier, int) or tier < 1 or tier > 5:
            errors.append("price_tier must be 1-5")

    status = payload.get("status", "pending")
    if status not in VALID_STATUSES:
        errors.append(f"invalid status: {status}")

    return errors


def validate_discover_payload(payload: dict) -> list[str]:
    """Validate a discover request payload. Returns list of errors."""
    errors = []

    query = payload.get("query", "")
    if not query or not query.strip():
        errors.append("query is required")
    elif len(query) > 2000:
        errors.append("query must be <= 2000 characters")

    limit = payload.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            errors.append("limit must be 1-50")

    price_max = payload.get("price_max")
    if price_max is not None:
        if not isinstance(price_max, int) or price_max < 1 or price_max > 5:
            errors.append("price_max must be 1-5")

    return errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityPayloadValidation:
    def test_valid_payload(self, sample_entity):
        errors = validate_entity_payload(sample_entity)
        assert errors == []

    def test_missing_name(self):
        errors = validate_entity_payload({"category": "restaurant"})
        assert any("name" in e for e in errors)

    def test_missing_category(self):
        errors = validate_entity_payload({"name": "Test"})
        assert any("category" in e for e in errors)

    def test_invalid_category(self):
        errors = validate_entity_payload({"name": "Test", "category": "xyz"})
        assert any("invalid category" in e for e in errors)

    def test_invalid_price_tier_too_high(self):
        errors = validate_entity_payload({"name": "Test", "category": "bar", "price_tier": 10})
        assert any("price_tier" in e for e in errors)

    def test_invalid_price_tier_zero(self):
        errors = validate_entity_payload({"name": "Test", "category": "bar", "price_tier": 0})
        assert any("price_tier" in e for e in errors)

    def test_valid_price_tier(self):
        errors = validate_entity_payload({"name": "Test", "category": "bar", "price_tier": 3})
        assert errors == []

    def test_invalid_status(self):
        errors = validate_entity_payload({"name": "Test", "category": "bar", "status": "unknown"})
        assert any("status" in e for e in errors)

    def test_name_too_long(self):
        errors = validate_entity_payload({"name": "A" * 201, "category": "bar"})
        assert any("200" in e for e in errors)


class TestDiscoverPayloadValidation:
    def test_valid_payload(self, sample_discover_request):
        errors = validate_discover_payload(sample_discover_request)
        assert errors == []

    def test_empty_query(self):
        errors = validate_discover_payload({"query": ""})
        assert any("query" in e for e in errors)

    def test_whitespace_query(self):
        errors = validate_discover_payload({"query": "   "})
        assert any("query" in e for e in errors)

    def test_query_too_long(self):
        errors = validate_discover_payload({"query": "a" * 2001})
        assert any("2000" in e for e in errors)

    def test_limit_too_high(self):
        errors = validate_discover_payload({"query": "test", "limit": 100})
        assert any("limit" in e for e in errors)

    def test_limit_zero(self):
        errors = validate_discover_payload({"query": "test", "limit": 0})
        assert any("limit" in e for e in errors)

    def test_valid_limit(self):
        errors = validate_discover_payload({"query": "test", "limit": 10})
        assert errors == []

    def test_invalid_price_max(self):
        errors = validate_discover_payload({"query": "test", "price_max": 6})
        assert any("price_max" in e for e in errors)
