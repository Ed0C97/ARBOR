"""Chaos Engineering with Toxiproxy.

TIER 9 - Point 48: Chaos Engineering (Toxiproxy)

Client for injecting faults to test resilience:
- Network latency
- Connection timeouts
- Packet loss
- Service unavailability

Usage in tests:
    async def test_handles_db_latency():
        chaos = ChaosClient()
        await chaos.add_latency("postgres", latency_ms=2000)
        
        # Test that circuit breaker activates
        result = await discover("test query")
        assert result.cached  # Should fallback to cache
        
        await chaos.reset("postgres")
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ToxicType(str, Enum):
    """Types of chaos to inject."""
    LATENCY = "latency"
    TIMEOUT = "timeout"
    BANDWIDTH = "bandwidth"
    SLOW_CLOSE = "slow_close"
    SLICER = "slicer"
    LIMIT_DATA = "limit_data"


@dataclass
class Toxic:
    """A chaos toxic configuration."""
    name: str
    type: ToxicType
    stream: str = "downstream"  # upstream or downstream
    toxicity: float = 1.0  # Probability 0-1
    attributes: dict = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Proxy:
    """A Toxiproxy proxy configuration."""
    name: str
    listen: str  # e.g., "localhost:5433"
    upstream: str  # e.g., "localhost:5432"
    enabled: bool = True
    toxics: list[Toxic] = None
    
    def __post_init__(self):
        if self.toxics is None:
            self.toxics = []


class ChaosClient:
    """Client for Toxiproxy chaos injection.
    
    TIER 9 - Point 48: Chaos Engineering.
    
    Requires Toxiproxy server running (usually in Docker):
        docker run -d --name toxiproxy -p 8474:8474 -p 5433:5433 ghcr.io/shopify/toxiproxy
    
    Usage:
        chaos = ChaosClient()
        
        # Create proxy for PostgreSQL
        await chaos.create_proxy(Proxy(
            name="postgres",
            listen="localhost:5433",
            upstream="localhost:5432",
        ))
        
        # Inject 500ms latency
        await chaos.add_toxic("postgres", Toxic(
            name="latency",
            type=ToxicType.LATENCY,
            attributes={"latency": 500},
        ))
        
        # Run tests...
        
        # Reset
        await chaos.remove_toxic("postgres", "latency")
    """
    
    def __init__(self, base_url: str = "http://localhost:8474"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=10)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    # Proxy management
    
    async def create_proxy(self, proxy: Proxy) -> dict[str, Any]:
        """Create a new proxy."""
        try:
            response = await self.client.post(
                "/proxies",
                json={
                    "name": proxy.name,
                    "listen": proxy.listen,
                    "upstream": proxy.upstream,
                    "enabled": proxy.enabled,
                },
            )
            response.raise_for_status()
            logger.info(f"Created proxy: {proxy.name}")
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                logger.debug(f"Proxy {proxy.name} already exists")
                return await self.get_proxy(proxy.name)
            raise
    
    async def get_proxy(self, name: str) -> dict[str, Any] | None:
        """Get proxy configuration."""
        try:
            response = await self.client.get(f"/proxies/{name}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError:
            return None
    
    async def list_proxies(self) -> dict[str, dict]:
        """List all proxies."""
        response = await self.client.get("/proxies")
        response.raise_for_status()
        return response.json()
    
    async def delete_proxy(self, name: str) -> bool:
        """Delete a proxy."""
        try:
            response = await self.client.delete(f"/proxies/{name}")
            response.raise_for_status()
            logger.info(f"Deleted proxy: {name}")
            return True
        except httpx.HTTPStatusError:
            return False
    
    async def toggle_proxy(self, name: str, enabled: bool) -> bool:
        """Enable or disable a proxy."""
        try:
            response = await self.client.post(
                f"/proxies/{name}",
                json={"enabled": enabled},
            )
            response.raise_for_status()
            logger.info(f"Proxy {name} {'enabled' if enabled else 'disabled'}")
            return True
        except httpx.HTTPStatusError:
            return False
    
    # Toxic management
    
    async def add_toxic(self, proxy_name: str, toxic: Toxic) -> dict[str, Any]:
        """Add a toxic to a proxy."""
        response = await self.client.post(
            f"/proxies/{proxy_name}/toxics",
            json={
                "name": toxic.name,
                "type": toxic.type.value,
                "stream": toxic.stream,
                "toxicity": toxic.toxicity,
                "attributes": toxic.attributes,
            },
        )
        response.raise_for_status()
        logger.info(f"Added toxic {toxic.name} to {proxy_name}")
        return response.json()
    
    async def remove_toxic(self, proxy_name: str, toxic_name: str) -> bool:
        """Remove a toxic from a proxy."""
        try:
            response = await self.client.delete(
                f"/proxies/{proxy_name}/toxics/{toxic_name}"
            )
            response.raise_for_status()
            logger.info(f"Removed toxic {toxic_name} from {proxy_name}")
            return True
        except httpx.HTTPStatusError:
            return False
    
    async def list_toxics(self, proxy_name: str) -> list[dict]:
        """List all toxics for a proxy."""
        response = await self.client.get(f"/proxies/{proxy_name}/toxics")
        response.raise_for_status()
        return response.json()
    
    # Convenience methods
    
    async def add_latency(
        self,
        proxy_name: str,
        latency_ms: int = 1000,
        jitter_ms: int = 100,
        toxic_name: str = "latency",
    ) -> dict[str, Any]:
        """Add latency to a proxy."""
        return await self.add_toxic(
            proxy_name,
            Toxic(
                name=toxic_name,
                type=ToxicType.LATENCY,
                attributes={
                    "latency": latency_ms,
                    "jitter": jitter_ms,
                },
            ),
        )
    
    async def add_timeout(
        self,
        proxy_name: str,
        timeout_ms: int = 5000,
        toxic_name: str = "timeout",
    ) -> dict[str, Any]:
        """Add timeout (connection hang) to a proxy."""
        return await self.add_toxic(
            proxy_name,
            Toxic(
                name=toxic_name,
                type=ToxicType.TIMEOUT,
                attributes={"timeout": timeout_ms},
            ),
        )
    
    async def add_bandwidth_limit(
        self,
        proxy_name: str,
        rate_kb: int = 10,
        toxic_name: str = "bandwidth",
    ) -> dict[str, Any]:
        """Limit bandwidth to simulate slow network."""
        return await self.add_toxic(
            proxy_name,
            Toxic(
                name=toxic_name,
                type=ToxicType.BANDWIDTH,
                attributes={"rate": rate_kb},
            ),
        )
    
    async def reset(self, proxy_name: str) -> None:
        """Remove all toxics from a proxy."""
        toxics = await self.list_toxics(proxy_name)
        for toxic in toxics:
            await self.remove_toxic(proxy_name, toxic["name"])
        logger.info(f"Reset all toxics for {proxy_name}")
    
    async def reset_all(self) -> None:
        """Remove all toxics from all proxies."""
        proxies = await self.list_proxies()
        for proxy_name in proxies:
            await self.reset(proxy_name)


# Pre-configured proxies for ARBOR services
DEFAULT_PROXIES = {
    "postgres": Proxy(
        name="postgres",
        listen="localhost:5433",
        upstream="localhost:5432",
    ),
    "redis": Proxy(
        name="redis",
        listen="localhost:6380",
        upstream="localhost:6379",
    ),
    "qdrant": Proxy(
        name="qdrant",
        listen="localhost:6334",
        upstream="localhost:6333",
    ),
}


async def setup_chaos_proxies() -> ChaosClient:
    """Setup default chaos proxies for testing.
    
    Call at the start of chaos tests.
    """
    chaos = ChaosClient()
    
    for proxy in DEFAULT_PROXIES.values():
        try:
            await chaos.create_proxy(proxy)
        except Exception as e:
            logger.warning(f"Failed to create proxy {proxy.name}: {e}")
    
    return chaos


# Chaos test scenarios
class ChaosScenarios:
    """Pre-built chaos test scenarios.
    
    TIER 9 - Point 48: Common failure patterns.
    """
    
    def __init__(self, chaos: ChaosClient):
        self.chaos = chaos
    
    async def database_slowdown(self, duration_ms: int = 2000) -> None:
        """Simulate slow database queries."""
        await self.chaos.add_latency("postgres", latency_ms=duration_ms)
    
    async def cache_unavailable(self) -> None:
        """Simulate Redis outage."""
        await self.chaos.toggle_proxy("redis", enabled=False)
    
    async def vector_search_timeout(self) -> None:
        """Simulate Qdrant timeout."""
        await self.chaos.add_timeout("qdrant", timeout_ms=30000)
    
    async def network_partition(self) -> None:
        """Simulate network partition (all services down)."""
        for proxy_name in DEFAULT_PROXIES:
            await self.chaos.toggle_proxy(proxy_name, enabled=False)
    
    async def recover_all(self) -> None:
        """Recover from all chaos scenarios."""
        for proxy_name in DEFAULT_PROXIES:
            await self.chaos.toggle_proxy(proxy_name, enabled=True)
            await self.chaos.reset(proxy_name)
