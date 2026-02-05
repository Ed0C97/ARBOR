"""Qdrant vector database async client.

TIER 1 - Point 2: Async Event Loop Integrity
- Uses AsyncQdrantClient for non-blocking I/O

TIER 7 - Point 29: gRPC over REST
- prefer_grpc=True for 30%+ throughput improvement

TIER 7 - Point 31: HNSW Index Fine-Tuning
- m=16 (connections)
- ef_construct=128 (build accuracy)
- Scalar Int8 quantization for 4x RAM reduction
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    QuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Async client singleton
_async_client: AsyncQdrantClient | None = None

# Sync client singleton (for backwards compatibility)
_sync_client: QdrantClient | None = None


async def get_async_qdrant_client() -> AsyncQdrantClient | None:
    """Return singleton async Qdrant client.

    TIER 1: Uses AsyncQdrantClient for non-blocking I/O
    TIER 7: Uses gRPC for better performance
    """
    global _async_client

    if not settings.qdrant_url:
        logger.warning("Qdrant URL not configured")
        return None

    if _async_client is None:
        _async_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=settings.qdrant_prefer_grpc,  # TIER 7: gRPC over REST
            grpc_port=6334,  # Standard gRPC port
            timeout=30,
        )
        logger.info(
            f"Async Qdrant client initialized (grpc={settings.qdrant_prefer_grpc})"
        )

    return _async_client


def get_qdrant_client() -> QdrantClient | None:
    """Return singleton sync Qdrant client (backwards compatibility).

    Prefer get_async_qdrant_client() for new code.
    """
    global _sync_client

    if not settings.qdrant_url:
        return None

    if _sync_client is None:
        _sync_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

    return _sync_client


async def init_qdrant_collections():
    """Initialize Qdrant collections on startup.

    TIER 7 - Point 31: HNSW Index Fine-Tuning
    - m=16 connections per node
    - ef_construct=128 for build accuracy
    - Scalar Int8 quantization for RAM efficiency
    """
    client = await get_async_qdrant_client()
    if client is None:
        logger.warning("Qdrant not configured, skipping collection init")
        return

    try:
        collections_response = await client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        # HNSW configuration for better recall
        hnsw_config = HnswConfigDiff(
            m=16,  # Number of edges per node
            ef_construct=128,  # Build-time accuracy
            full_scan_threshold=10000,  # Use index for >10k vectors
        )

        # Scalar quantization for 4x RAM reduction with minimal precision loss
        quantization_config = QuantizationConfig(
            scalar=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,  # Keep quantized vectors in RAM
                )
            )
        )

        # Main entities collection (Cohere embed-v4.0 = 1024 dims)
        if "entities_vectors" not in existing_collections:
            await client.create_collection(
                collection_name="entities_vectors",
                vectors_config=VectorParams(
                    size=1024,  # Cohere embed-v4.0 dimension
                    distance=Distance.COSINE,
                ),
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=50000,
                ),
            )
            logger.info("Created 'entities_vectors' collection with HNSW optimization")
        else:
            # Update existing collection with optimized config
            try:
                await client.update_collection(
                    collection_name="entities_vectors",
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                )
                logger.info("Updated 'entities_vectors' collection config")
            except Exception as e:
                logger.debug(f"Could not update collection config: {e}")

        # Semantic cache collection
        if "semantic_cache" not in existing_collections:
            await client.create_collection(
                collection_name="semantic_cache",
                vectors_config=VectorParams(
                    size=1024,  # Cohere embed-v4.0 dimension
                    distance=Distance.COSINE,
                ),
                hnsw_config=hnsw_config,
            )
            logger.info("Created 'semantic_cache' collection")

        logger.info("Qdrant collections initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collections: {e}")
        raise


async def close_qdrant_client():
    """Close Qdrant client connections."""
    global _async_client, _sync_client

    if _async_client is not None:
        await _async_client.close()
        _async_client = None
        logger.info("Async Qdrant client closed")

    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
        logger.info("Sync Qdrant client closed")


async def check_qdrant_health() -> dict[str, Any]:
    """Health check for Qdrant service."""
    client = await get_async_qdrant_client()
    if client is None:
        return {"status": "not_configured", "healthy": False}

    try:
        collections = await client.get_collections()
        return {
            "status": "healthy",
            "healthy": True,
            "collections": [c.name for c in collections.collections],
            "grpc_enabled": settings.qdrant_prefer_grpc,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "healthy": False,
            "error": str(e),
        }

