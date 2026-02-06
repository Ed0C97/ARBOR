"""LLM Gateway — Gemini + Cohere stack.

Primary stack:
- Google Gemini gemini-3-pro-preview  → Text generation + Vision analysis
- Cohere embed-v4.0                   → Embedding generation
- Cohere rerank-v4.0-pro              → Result reranking

Uses LiteLLM for unified API + Cohere SDK directly for embedding/reranking
(because Cohere embed-v4.0 uses a special input format).

TIER 1 - Point 2: Async Event Loop Integrity
- Uses cohere.AsyncClientV2 for non-blocking I/O

TIER 2 - Point 4: Vector Ingestion Batch Pipeline
- Batched embedding with chunker for 96-item batches

TIER 3 - Point 10: Circuit Breakers Integration
TIER 3 - Point 12: Tenacity Retry Policies Integration
"""

import asyncio
import logging
from typing import Any

import cohere
from litellm import acompletion
from litellm.router import Router

from app.config import get_settings
from app.core.circuit import cohere_circuit, llm_circuit
from app.core.retry import retry_cohere, retry_llm

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Cohere Async Client (TIER 1 - Point 2)
# ═══════════════════════════════════════════════════════════════════════════

_async_cohere_client: cohere.AsyncClientV2 | None = None
_sync_cohere_client: cohere.ClientV2 | None = None


async def get_async_cohere_client() -> cohere.AsyncClientV2 | None:
    """Return singleton async Cohere client.

    TIER 1: Uses AsyncClientV2 for non-blocking I/O
    """
    global _async_cohere_client

    if not settings.cohere_api_key:
        logger.warning("Cohere API key not configured")
        return None

    if _async_cohere_client is None:
        _async_cohere_client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        logger.info("Async Cohere client initialized")

    return _async_cohere_client


def get_cohere_client() -> cohere.ClientV2 | None:
    """Return singleton sync Cohere client (backwards compatibility)."""
    global _sync_cohere_client

    if not settings.cohere_api_key:
        return None

    if _sync_cohere_client is None:
        _sync_cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)

    return _sync_cohere_client


async def close_cohere_client():
    """Close Cohere client connections."""
    global _async_cohere_client, _sync_cohere_client

    if _async_cohere_client is not None:
        await _async_cohere_client.close()
        _async_cohere_client = None
        logger.info("Async Cohere client closed")

    _sync_cohere_client = None


# ═══════════════════════════════════════════════════════════════════════════
# Batch Utilities (TIER 2 - Point 4)
# ═══════════════════════════════════════════════════════════════════════════


def chunker(iterable: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split iterable into chunks of specified size.

    TIER 2 - Point 4: Generator-based chunking for batch processing.

    Args:
        iterable: List to chunk
        chunk_size: Maximum items per chunk (Cohere max = 96)

    Returns:
        List of chunks
    """
    return [iterable[i : i + chunk_size] for i in range(0, len(iterable), chunk_size)]


# Maximum batch size for Cohere embedding API
COHERE_BATCH_SIZE = 96


# ═══════════════════════════════════════════════════════════════════════════
# Model registry for LiteLLM router (text + vision via Gemini)
# ═══════════════════════════════════════════════════════════════════════════


def _build_model_list() -> list[dict]:
    """Build model list from available API keys."""
    models = []

    # ── Google Gemini gemini-3-pro-preview (primary: text + vision) ───
    if settings.google_api_key:
        models.append(
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": f"gemini/{settings.google_model}",
                    "api_key": settings.google_api_key,
                },
                "tpm": 200000,
                "rpm": 2000,
            }
        )
        models.append(
            {
                "model_name": "vision",
                "litellm_params": {
                    "model": f"gemini/{settings.google_model}",
                    "api_key": settings.google_api_key,
                },
            }
        )
        # Same model for simple/fast — Gemini is the only text provider
        models.append(
            {
                "model_name": "simple",
                "litellm_params": {
                    "model": f"gemini/{settings.google_model}",
                    "api_key": settings.google_api_key,
                },
            }
        )
        models.append(
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": f"gemini/{settings.google_model}",
                    "api_key": settings.google_api_key,
                },
            }
        )

    # ── Fallback: Anthropic Claude (if configured) ────────────────────
    if settings.anthropic_api_key:
        models.append(
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": f"anthropic/{settings.anthropic_model}",
                    "api_key": settings.anthropic_api_key,
                },
            }
        )
        models.append(
            {
                "model_name": "vision",
                "litellm_params": {
                    "model": f"anthropic/{settings.anthropic_model}",
                    "api_key": settings.anthropic_api_key,
                },
            }
        )

    # ── Fallback: OpenAI (if configured) ──────────────────────────────
    if settings.openai_api_key:
        models.append(
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": f"openai/{settings.openai_model}",
                    "api_key": settings.openai_api_key,
                },
            }
        )

    # ── Fallback: Groq (if configured, for fast tasks) ───────────────
    if settings.groq_api_key:
        models.append(
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": f"groq/{settings.groq_model}",
                    "api_key": settings.groq_api_key,
                },
            }
        )

    return models


# ═══════════════════════════════════════════════════════════════════════════
# LLM Gateway
# ═══════════════════════════════════════════════════════════════════════════


class LLMGateway:
    """Unified LLM gateway.

    Text + Vision → Google Gemini gemini-3-pro-preview (via LiteLLM)
    Embedding     → Cohere embed-v4.0 (via native async SDK)
    Reranking     → Cohere rerank-v4.0-pro (via native async SDK)

    TIER 1 - Point 2: All I/O operations are async
    TIER 2 - Point 4: Batch embedding support
    TIER 3 - Point 10: Circuit breaker integration
    TIER 3 - Point 12: Retry policy integration

    Task types for complete():
    - "complex"        → primary model (scoring, synthesis)
    - "simple"         → lighter tasks (extraction)
    - "vision"         → image analysis
    - "fast"           → fastest available
    - "classification" → fast model
    - "extraction"     → simple model
    - "synthesis"      → primary model
    """

    def __init__(self):
        model_list = _build_model_list()
        if model_list:
            fallbacks = [
                {"primary": ["simple", "fast"]},
                {"vision": ["primary"]},
                {"simple": ["fast", "primary"]},
            ]
            self.router = Router(
                model_list=model_list,
                routing_strategy="latency-based-routing",
                fallbacks=fallbacks,
                num_retries=2,
                timeout=60,
            )
            model_names = sorted({m["model_name"] for m in model_list})
            logger.info(f"LLM Gateway initialized: {model_names}")
        else:
            self.router = None
            logger.warning("No LLM API keys configured!")

        # Track async Cohere client initialization status
        self._async_cohere_initialized = False
        self._async_cohere: cohere.AsyncClientV2 | None = None

        # Sync Cohere client for backwards compatibility
        if settings.cohere_api_key:
            self.cohere = get_cohere_client()
            logger.info(
                f"Cohere initialized: embedding={settings.cohere_embedding_model}, "
                f"reranking={settings.cohere_rerank_model}"
            )
        else:
            self.cohere = None
            logger.warning("No Cohere API key — embedding and reranking will not work")

    async def _get_async_cohere(self) -> cohere.AsyncClientV2 | None:
        """Lazy-initialize async Cohere client."""
        if not self._async_cohere_initialized:
            self._async_cohere = await get_async_cohere_client()
            self._async_cohere_initialized = True
        return self._async_cohere

    # ── Text Completion ───────────────────────────────────────────────

    @llm_circuit
    @retry_llm
    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        task_type: str = "complex",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Text completion via Gemini (or fallback providers).

        TIER 3: Protected by circuit breaker and retry policy.
        """
        if model is None:
            model = self._select_model(task_type)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        try:
            if self.router:
                response = await self.router.acompletion(**kwargs)
            else:
                response = await acompletion(**kwargs)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM completion failed (task={task_type}): {e}")
            raise

    async def complete_json(
        self,
        messages: list[dict],
        model: str | None = None,
        task_type: str = "complex",
    ) -> str:
        """Completion that returns JSON."""
        return await self.complete(
            messages=messages,
            model=model,
            task_type=task_type,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

    async def complete_vision(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Vision completion — sends images to Gemini gemini-3-pro-preview."""
        return await self.complete(
            messages=messages,
            task_type="vision",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Embedding via Cohere embed-v4.0 (ASYNC) ───────────────────────

    @cohere_circuit
    @retry_cohere
    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding using async Cohere embed-v4.0.

        TIER 1: Uses AsyncClientV2 for non-blocking I/O
        TIER 3: Protected by circuit breaker and retry policy.
        """
        cohere_client = await self._get_async_cohere()

        if not cohere_client:
            # Fallback to OpenAI text-embedding-3-small (1536 dims)
            logger.warning("Cohere not available, using OpenAI fallback for embeddings")
            try:
                from litellm import aembedding

                response = await aembedding(
                    model="text-embedding-3-small",
                    input=[text],
                )
                return response.data[0]["embedding"]
            except Exception as e:
                logger.error(f"OpenAI embedding fallback failed: {e}")
                raise RuntimeError("No embedding provider available")

        response = await cohere_client.embed(
            model=settings.cohere_embedding_model,
            input_type="search_document",
            embedding_types=["float"],
            texts=[text],
        )
        return response.embeddings.float_[0]

    @cohere_circuit
    @retry_cohere
    async def get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a search query (different input_type).

        TIER 1: Uses AsyncClientV2 for non-blocking I/O
        """
        cohere_client = await self._get_async_cohere()

        if not cohere_client:
            logger.warning("Cohere not available, using OpenAI fallback")
            try:
                from litellm import aembedding

                response = await aembedding(
                    model="text-embedding-3-small",
                    input=[query],
                )
                return response.data[0]["embedding"]
            except Exception as e:
                logger.error(f"OpenAI query embedding fallback failed: {e}")
                raise RuntimeError("No embedding provider available")

        response = await cohere_client.embed(
            model=settings.cohere_embedding_model,
            input_type="search_query",
            embedding_types=["float"],
            texts=[query],
        )
        return response.embeddings.float_[0]

    @cohere_circuit
    @retry_cohere
    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        TIER 2 - Point 4: Vector Ingestion Batch Pipeline
        - Chunks texts into batches of 96 (Cohere max)
        - Processes batches in parallel with asyncio.gather
        - Reduces latency from N*200ms to ~(N/96)*200ms

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        cohere_client = await self._get_async_cohere()
        if not cohere_client:
            # Fallback: process one by one
            return [await self.get_embedding(t) for t in texts]

        # Chunk into batches of 96
        batches = chunker(texts, COHERE_BATCH_SIZE)

        # Process batches in parallel
        async def embed_batch(batch: list[str]) -> list[list[float]]:
            response = await cohere_client.embed(
                model=settings.cohere_embedding_model,
                input_type="search_document",
                embedding_types=["float"],
                texts=batch,
            )
            return response.embeddings.float_

        tasks = [embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        # Flatten results
        return [emb for batch_result in results for emb in batch_result]

    @cohere_circuit
    @retry_cohere
    async def get_image_embedding(self, image_base64: str) -> list[float]:
        """Generate embedding for an image using async Cohere embed-v4.0.

        Args:
            image_base64: Base64 encoded image with data URI prefix
        """
        cohere_client = await self._get_async_cohere()
        if not cohere_client:
            raise RuntimeError("Cohere API key not configured")

        response = await cohere_client.embed(
            model=settings.cohere_embedding_model,
            input_type="image",
            embedding_types=["float"],
            inputs=[{"content": [{"type": "image_url", "image_url": {"url": image_base64}}]}],
        )
        return response.embeddings.float_[0]

    # ── Reranking via Cohere rerank-v4.0-pro (ASYNC) ──────────────────

    @cohere_circuit
    @retry_cohere
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10,
    ) -> list[dict]:
        """Rerank documents using async Cohere rerank-v4.0-pro.

        TIER 1: Uses AsyncClientV2 for non-blocking I/O
        TIER 4 - Point 19: Cohere Rerank v3 Integration
        """
        cohere_client = await self._get_async_cohere()

        if not cohere_client:
            logger.warning("Cohere not configured — returning documents in original order")
            return [
                {"index": i, "relevance_score": 1.0, "document": doc}
                for i, doc in enumerate(documents[:top_n])
            ]

        try:
            response = await cohere_client.rerank(
                model=settings.cohere_rerank_model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return [
                {
                    "index": r.index,
                    "relevance_score": r.relevance_score,
                    "document": documents[r.index],
                }
                for r in response.results
            ]

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Graceful degradation: return original order
            return [
                {"index": i, "relevance_score": 1.0, "document": doc}
                for i, doc in enumerate(documents[:top_n])
            ]

    @cohere_circuit
    @retry_cohere
    async def rerank_results(
        self,
        query: str,
        results: list[dict],
        top_n: int = 10,
        text_field: str = "name",
    ) -> list[dict]:
        """Rerank search results preserving full result objects.

        TIER 4 - Point 19: Cohere Rerank v3 Integration

        Pipeline: Retrieve 50 -> Rerank -> Keep Top 10 -> LLM Synthesis

        Args:
            query: User query string
            results: List of result dicts from search
            top_n: Number of top results to return
            text_field: Field to use for reranking text

        Returns:
            Reranked results with relevance scores added
        """
        if not results:
            return []

        # Build document texts for reranking
        documents = []
        for r in results:
            # Build rich text from multiple fields
            parts = []
            if r.get("name"):
                parts.append(r["name"])
            if r.get("category"):
                parts.append(f"[{r['category']}]")
            if r.get("description"):
                parts.append(r["description"][:500])
            if r.get("tags"):
                parts.append(", ".join(r["tags"][:5]))

            doc_text = " ".join(parts) if parts else str(r.get(text_field, ""))
            documents.append(doc_text)

        # Get reranked indices
        reranked = await self.rerank(query, documents, top_n=top_n)

        # Map back to original results with scores
        reranked_results = []
        for item in reranked:
            idx = item["index"]
            if idx < len(results):
                result = results[idx].copy()
                result["relevance_score"] = item["relevance_score"]
                result["original_rank"] = idx + 1
                reranked_results.append(result)

        logger.info(
            f"Reranked {len(results)} results -> top {len(reranked_results)} for: {query[:50]}"
        )

        return reranked_results

    # ── Model selection ───────────────────────────────────────────────

    def _select_model(self, task_type: str) -> str:
        """Select model name based on task type."""
        model_map = {
            "classification": "fast",
            "extraction": "simple",
            "synthesis": "primary",
            "complex": "primary",
            "simple": "simple",
            "vision": "vision",
            "fast": "fast",
        }
        return model_map.get(task_type, "primary")

    async def close(self):
        """Close all client connections."""
        await close_cohere_client()


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    """Return singleton LLM gateway."""
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
