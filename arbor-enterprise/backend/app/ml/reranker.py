"""Reranker using Cohere rerank-v4.0-pro.

Primary: Cohere rerank-v4.0-pro via native SDK (ClientV2)
Fallback: local CrossEncoder if Cohere is unavailable
"""

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridReranker:
    """Two-stage reranker for search results.

    1. **Cohere rerank-v4.0-pro** (primary) — neural reranking via API.
    2. **CrossEncoder** (fallback) — local model when Cohere is unavailable.

    Usage::

        reranker = HybridReranker()
        reranked = await reranker.rerank(
            query="cozy coffee shop in Roma Norte",
            documents=search_results,
            top_k=5,
        )
    """

    def __init__(self) -> None:
        self._cohere_client = None
        if settings.cohere_api_key:
            import cohere

            self._cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
            logger.info("Cohere reranker enabled (model=%s)", settings.cohere_rerank_model)

        # Lazy-loaded local CrossEncoder
        self._cross_encoder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
        text_key: str = "name",
    ) -> list[dict[str, Any]]:
        """Rerank a list of search result dicts by relevance to the query.

        Attempts Cohere first; falls back to the local CrossEncoder on failure.

        Args:
            query: The user's search query.
            documents: Search results, each a dict containing at least ``text_key``.
            top_k: Maximum number of results to return after reranking.
            text_key: Key in each document dict to use as the passage text.

        Returns:
            Sorted list of result dicts with an added ``rerank_score`` field.
        """
        if not documents:
            return []

        try:
            if self._cohere_client is not None:
                return self._rerank_cohere(query, documents, top_k, text_key)
        except Exception as exc:
            logger.warning("Cohere rerank failed, falling back to CrossEncoder: %s", exc)

        return self._rerank_cross_encoder(query, documents, top_k, text_key)

    # ------------------------------------------------------------------
    # Cohere reranking — rerank-v4.0-pro via ClientV2
    # ------------------------------------------------------------------

    def _rerank_cohere(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        text_key: str,
    ) -> list[dict[str, Any]]:
        """Rerank using Cohere rerank-v4.0-pro."""
        assert self._cohere_client is not None

        passages = [self._extract_text(doc, text_key) for doc in documents]

        response = self._cohere_client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=passages,
            top_n=top_k,
        )

        reranked: list[dict[str, Any]] = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            reranked.append(doc)

        logger.debug(
            "Cohere rerank: query='%s' candidates=%d returned=%d",
            query,
            len(documents),
            len(reranked),
        )
        return reranked

    # ------------------------------------------------------------------
    # CrossEncoder fallback
    # ------------------------------------------------------------------

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        text_key: str,
    ) -> list[dict[str, Any]]:
        """Rerank using a local CrossEncoder model (fallback)."""
        encoder = self._get_cross_encoder()

        pairs = [(query, self._extract_text(doc, text_key)) for doc in documents]
        scores = encoder.predict(pairs)

        scored_docs: list[tuple[float, dict[str, Any]]] = []
        for score, doc in zip(scores, documents):
            enriched = doc.copy()
            enriched["rerank_score"] = float(score)
            scored_docs.append((float(score), enriched))

        scored_docs.sort(key=lambda t: t[0], reverse=True)
        reranked = [doc for _, doc in scored_docs[:top_k]]

        logger.debug(
            "CrossEncoder rerank: query='%s' candidates=%d returned=%d",
            query,
            len(documents),
            len(reranked),
        )
        return reranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cross_encoder(self):
        """Lazy-load the local CrossEncoder model."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info("Loading local CrossEncoder model: %s", model_name)
            self._cross_encoder = CrossEncoder(model_name)
        return self._cross_encoder

    @staticmethod
    def _extract_text(doc: dict[str, Any], text_key: str) -> str:
        """Build a passage string from a result dict."""
        text = doc.get(text_key, "")
        description = doc.get("description", "")
        if description:
            return f"{text}. {description}" if text else description
        return text or ""


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_reranker: HybridReranker | None = None


def get_reranker() -> HybridReranker:
    """Return the singleton HybridReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = HybridReranker()
    return _reranker
