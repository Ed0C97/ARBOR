"""Unit tests for caching logic."""

import hashlib
import json
import time

import pytest

# ---------------------------------------------------------------------------
# Cache helpers (inline to avoid importing redis/llm deps)
# ---------------------------------------------------------------------------


class InMemoryCache:
    """Simple in-memory cache for testing cache logic."""

    def __init__(self, default_ttl: int = 3600):
        self._store: dict[str, tuple[str, float]] = {}
        self.default_ttl = default_ttl

    def _make_key(self, prefix: str, params: dict) -> str:
        """Generate a deterministic cache key."""
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"

    def get(self, key: str) -> str | None:
        """Get a cached value, or None if expired/missing."""
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a cached value with TTL."""
        expiry = time.time() + (ttl or self.default_ttl)
        self._store[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCacheKeyGeneration:
    def setup_method(self):
        self.cache = InMemoryCache()

    def test_deterministic_key(self):
        key1 = self.cache._make_key("llm", {"query": "test", "model": "gpt-4"})
        key2 = self.cache._make_key("llm", {"query": "test", "model": "gpt-4"})
        assert key1 == key2

    def test_different_params_different_keys(self):
        key1 = self.cache._make_key("llm", {"query": "test1"})
        key2 = self.cache._make_key("llm", {"query": "test2"})
        assert key1 != key2

    def test_key_has_prefix(self):
        key = self.cache._make_key("embed", {"text": "hello"})
        assert key.startswith("embed:")

    def test_param_order_independent(self):
        key1 = self.cache._make_key("x", {"a": 1, "b": 2})
        key2 = self.cache._make_key("x", {"b": 2, "a": 1})
        assert key1 == key2


class TestCacheOperations:
    def setup_method(self):
        self.cache = InMemoryCache(default_ttl=60)

    def test_set_and_get(self):
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_get_missing_key(self):
        assert self.cache.get("nonexistent") is None

    def test_delete(self):
        self.cache.set("key1", "value1")
        assert self.cache.delete("key1") is True
        assert self.cache.get("key1") is None

    def test_delete_missing(self):
        assert self.cache.delete("nonexistent") is False

    def test_clear(self):
        self.cache.set("k1", "v1")
        self.cache.set("k2", "v2")
        assert self.cache.size == 2
        self.cache.clear()
        assert self.cache.size == 0

    def test_overwrite(self):
        self.cache.set("key1", "v1")
        self.cache.set("key1", "v2")
        assert self.cache.get("key1") == "v2"

    def test_ttl_expiry(self):
        self.cache.set("key1", "value1", ttl=0)  # Expire immediately
        # Give a tiny delay for time.time() to advance
        time.sleep(0.01)
        assert self.cache.get("key1") is None

    def test_custom_ttl(self):
        self.cache.set("key1", "value1", ttl=3600)
        assert self.cache.get("key1") == "value1"

    def test_size(self):
        assert self.cache.size == 0
        self.cache.set("k1", "v1")
        assert self.cache.size == 1
        self.cache.set("k2", "v2")
        assert self.cache.size == 2
