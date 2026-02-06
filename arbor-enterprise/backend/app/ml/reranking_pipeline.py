"""Advanced 4-stage reranking pipeline for ARBOR Enterprise search.

Implements a multi-stage funnel that progressively narrows candidates:
    Stage 1: Dense retrieval   (cosine similarity)       → top 100
    Stage 2: Sparse retrieval  (BM25)                    → top  50  (parallel, merged via RRF)
    Stage 3: Cross-encoder     (Cohere rerank-v4.0-pro)  → top  20
    Stage 4: LLM reranker      (Gemini)                  → top  10

Each stage is independently usable and the pipeline gracefully degrades
when external services are unavailable (circuit breakers + fallbacks).

Usage::

    pipeline = get_reranking_pipeline()

    # Full 4-stage rerank
    results = await pipeline.rerank(
        query="cozy rooftop bar in Roma Norte",
        query_embedding=embedding_vector,
        candidates=raw_candidates,
    )

    # Fast rerank (no API calls — stages 1+2 only)
    results = await pipeline.rerank_fast(
        query="cozy rooftop bar in Roma Norte",
        query_embedding=embedding_vector,
        candidates=raw_candidates,
    )
"""

import asyncio
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "it",
    "as",
    "was",
    "are",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "not",
    "no",
    "nor",
    "so",
    "if",
    "than",
    "that",
    "this",
    "these",
    "those",
    "then",
    "there",
    "here",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "too",
    "very",
    "just",
    "about",
    "above",
    "after",
    "again",
    "also",
    "any",
    "because",
    "before",
    "below",
    "between",
    "during",
    "into",
    "its",
    "me",
    "my",
    "our",
    "out",
    "over",
    "their",
    "them",
    "up",
    "we",
    "you",
    "your",
    "he",
    "she",
    "her",
    "him",
    "his",
    "i",
    "itself",
    "let",
    "many",
    "much",
    "must",
    "nor",
    "off",
    "once",
    "ours",
    "ourselves",
    "shall",
    "still",
    "through",
    "under",
    "until",
    "us",
    "while",
    "whom",
    "yours",
}


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data Classes
# ═══════════════════════════════════════════════════════════════════════════


class RankingStage(Enum):
    """Identifies a stage in the reranking pipeline."""

    DENSE_RETRIEVAL = "dense_retrieval"
    SPARSE_RETRIEVAL = "sparse_retrieval"
    CROSS_ENCODER = "cross_encoder"
    LLM_RERANKER = "llm_reranker"


@dataclass
class RankedResult:
    """A search result enriched with per-stage scoring information.

    Attributes:
        entity_id: Unique identifier for the entity.
        name: Display name of the entity.
        category: Entity category (e.g. ``cafe``, ``bar``).
        city: City where the entity is located.
        score: Final composite score after all pipeline stages.
        stage_scores: Mapping of stage name to the score produced at that stage.
        metadata: Arbitrary additional data carried through the pipeline.
        rank: Final 1-based rank position after reranking.
    """

    entity_id: str
    name: str
    category: str
    city: str
    score: float
    stage_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class ScoringSignal:
    """An individual scoring signal that contributed to a result's score.

    Useful for debugging and explainability.

    Attributes:
        signal_name: Human-readable name (e.g. ``cosine_similarity``).
        weight: Relative weight of this signal in the final score.
        score: Raw score value for this signal.
        explanation: Free-text explanation of how this score was computed.
    """

    signal_name: str
    weight: float
    score: float
    explanation: str


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Dense Retrieval (cosine similarity)
# ═══════════════════════════════════════════════════════════════════════════


class Stage1DenseRetrieval:
    """Score candidates by cosine similarity against the query embedding.

    This is the cheapest stage — pure vector math with no external calls.
    """

    async def rank(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int = 100,
    ) -> list[RankedResult]:
        """Rank candidates by cosine similarity to *query_embedding*.

        Args:
            query_embedding: The dense embedding vector for the user query.
            candidates: Raw candidate dicts.  Each must contain at minimum
                ``entity_id`` and ``name``.  If an ``embedding`` key is
                present it will be used for similarity; otherwise the
                candidate's existing ``score`` is preserved.
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of :class:`RankedResult` (descending by score).
        """
        scored: list[RankedResult] = []

        for candidate in candidates:
            embedding = candidate.get("embedding")

            if embedding is not None and query_embedding:
                sim = self._cosine_similarity(query_embedding, embedding)
            else:
                # Fall back to any pre-existing score
                sim = float(candidate.get("score", 0.0))

            result = RankedResult(
                entity_id=str(candidate.get("entity_id", candidate.get("id", ""))),
                name=candidate.get("name", ""),
                category=candidate.get("category", ""),
                city=candidate.get("city", ""),
                score=sim,
                stage_scores={RankingStage.DENSE_RETRIEVAL.value: sim},
                metadata={
                    k: v
                    for k, v in candidate.items()
                    if k
                    not in {"embedding", "entity_id", "id", "name", "category", "city", "score"}
                },
            )
            scored.append(result)

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]

        for i, result in enumerate(top):
            result.rank = i + 1

        logger.debug(
            "Stage1 Dense: %d candidates -> top %d (max_score=%.4f)",
            len(candidates),
            len(top),
            top[0].score if top else 0.0,
        )
        return top

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero magnitude.
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot / (mag_a * mag_b)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Sparse Retrieval (BM25)
# ═══════════════════════════════════════════════════════════════════════════


class Stage2SparseRetrieval:
    """BM25-based sparse retrieval stage.

    Builds an in-memory inverted index and scores candidates using the
    Okapi BM25 formula with parameters k1=1.2, b=0.75.

    Usage::

        stage2 = Stage2SparseRetrieval()
        stage2.add_documents([
            {"id": "e1", "name": "Cafe Amor", "description": "cozy coffee"},
            ...
        ])
        results = stage2.rank("cozy coffee shop", candidates)
    """

    # BM25 hyper-parameters
    K1: float = 1.2
    B: float = 0.75

    def __init__(self) -> None:
        # doc_id -> list[str] (tokenised terms for each document)
        self._doc_tokens: dict[str, list[str]] = {}
        # term -> set of doc_ids containing the term
        self._inverted_index: dict[str, set[str]] = defaultdict(set)
        # Total number of indexed documents
        self._doc_count: int = 0
        # Average document length (in tokens)
        self._avg_doc_len: float = 0.0
        # doc_id -> document length
        self._doc_lengths: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Index a batch of documents for BM25 scoring.

        Each document dict should have an ``id`` key and one or more text
        fields (``name``, ``description``, ``category``, ``tags``).  All
        text fields are concatenated and tokenised.

        Args:
            docs: List of document dicts to index.
        """
        for doc in docs:
            doc_id = str(doc.get("id", doc.get("entity_id", "")))
            if not doc_id:
                continue

            text_parts: list[str] = []
            for text_field in ("name", "description", "category"):
                val = doc.get(text_field, "")
                if val:
                    text_parts.append(str(val))

            tags = doc.get("tags", [])
            if isinstance(tags, list):
                text_parts.extend(str(t) for t in tags)

            full_text = " ".join(text_parts)
            tokens = self._tokenize(full_text)

            self._doc_tokens[doc_id] = tokens
            self._doc_lengths[doc_id] = len(tokens)

            for token in set(tokens):
                self._inverted_index[token].add(doc_id)

        self._doc_count = len(self._doc_tokens)
        total_len = sum(self._doc_lengths.values())
        self._avg_doc_len = total_len / self._doc_count if self._doc_count > 0 else 0.0

        logger.debug(
            "BM25 index updated: %d docs, avg_len=%.1f, vocabulary=%d",
            self._doc_count,
            self._avg_doc_len,
            len(self._inverted_index),
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int = 50,
    ) -> list[RankedResult]:
        """Rank candidates by BM25 relevance to *query_text*.

        If the inverted index is empty (no documents indexed), candidates
        are indexed on-the-fly before scoring.

        Args:
            query_text: The user's search query in plain text.
            candidates: Raw candidate dicts.
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of :class:`RankedResult` (descending by BM25 score).
        """
        # Lazy-index if we have no documents yet
        if self._doc_count == 0 and candidates:
            self.add_documents(candidates)

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            # No meaningful query tokens — return candidates in original order
            return self._passthrough(candidates, top_k)

        scored: list[RankedResult] = []

        for candidate in candidates:
            doc_id = str(candidate.get("entity_id", candidate.get("id", "")))

            # Compute BM25 score
            bm25_score = self._bm25_score(doc_id, query_tokens)

            result = RankedResult(
                entity_id=doc_id,
                name=candidate.get("name", ""),
                category=candidate.get("category", ""),
                city=candidate.get("city", ""),
                score=bm25_score,
                stage_scores={RankingStage.SPARSE_RETRIEVAL.value: bm25_score},
                metadata={
                    k: v
                    for k, v in candidate.items()
                    if k
                    not in {"embedding", "entity_id", "id", "name", "category", "city", "score"}
                },
            )
            scored.append(result)

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]

        for i, result in enumerate(top):
            result.rank = i + 1

        logger.debug(
            "Stage2 BM25: %d candidates -> top %d (max_score=%.4f)",
            len(candidates),
            len(top),
            top[0].score if top else 0.0,
        )
        return top

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _bm25_score(self, doc_id: str, query_tokens: list[str]) -> float:
        """Compute the BM25 score for a single document against query tokens.

        Okapi BM25 formula::

            score = sum over query terms of:
                IDF(t) * ( tf * (k1 + 1) ) / ( tf + k1 * (1 - b + b * dl / avgdl) )

        where:
            IDF(t) = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
            tf     = term frequency in the document
            dl     = document length
            avgdl  = average document length
        """
        doc_tokens = self._doc_tokens.get(doc_id)
        if doc_tokens is None:
            return 0.0

        dl = self._doc_lengths.get(doc_id, 0)
        score = 0.0

        # Pre-compute term frequencies for this document
        tf_map: dict[str, int] = {}
        for token in doc_tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        for term in query_tokens:
            # Document frequency for this term
            n_t = len(self._inverted_index.get(term, set()))
            if n_t == 0:
                continue

            # IDF with smoothing
            idf = math.log((self._doc_count - n_t + 0.5) / (n_t + 0.5) + 1.0)

            # Term frequency in this document
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue

            # BM25 TF normalisation
            denominator = (
                tf + self.K1 * (1.0 - self.B + self.B * dl / self._avg_doc_len)
                if self._avg_doc_len > 0
                else tf + self.K1
            )

            score += idf * (tf * (self.K1 + 1.0)) / denominator

        return score

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, and remove stopwords.

        Args:
            text: Raw input text.

        Returns:
            List of cleaned tokens.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _passthrough(candidates: list[dict[str, Any]], top_k: int) -> list[RankedResult]:
        """Return candidates in their original order when BM25 cannot score."""
        results: list[RankedResult] = []
        for i, candidate in enumerate(candidates[:top_k]):
            results.append(
                RankedResult(
                    entity_id=str(candidate.get("entity_id", candidate.get("id", ""))),
                    name=candidate.get("name", ""),
                    category=candidate.get("category", ""),
                    city=candidate.get("city", ""),
                    score=float(candidate.get("score", 0.0)),
                    stage_scores={RankingStage.SPARSE_RETRIEVAL.value: 0.0},
                    metadata={},
                    rank=i + 1,
                )
            )
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Cross-Encoder (Cohere rerank-v4.0-pro)
# ═══════════════════════════════════════════════════════════════════════════


class Stage3CrossEncoder:
    """Cross-encoder reranking via Cohere rerank-v4.0-pro.

    Primary:  Cohere rerank API (via async SDK) protected by circuit breaker.
    Fallback: Simple keyword-overlap scoring when the API is unavailable.
    """

    def __init__(self) -> None:
        self._cohere_client = None
        self._circuit_breaker = None

    async def _get_cohere_client(self):
        """Lazy-initialise the async Cohere client."""
        if self._cohere_client is None and settings.cohere_api_key:
            try:
                import cohere

                self._cohere_client = cohere.AsyncClientV2(
                    api_key=settings.cohere_api_key,
                )
                logger.info("Stage3 CrossEncoder: Cohere client initialised")
            except Exception as exc:
                logger.warning("Stage3 CrossEncoder: Failed to init Cohere client: %s", exc)
        return self._cohere_client

    async def _get_circuit_breaker(self):
        """Lazy-load the cohere circuit breaker."""
        if self._circuit_breaker is None:
            from app.core.circuit import cohere_circuit

            self._circuit_breaker = cohere_circuit
        return self._circuit_breaker

    async def rank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int = 20,
    ) -> list[RankedResult]:
        """Rerank candidates using cross-encoder scoring.

        Attempts Cohere rerank API first.  If Cohere is unavailable or the
        circuit breaker is open, falls back to keyword-overlap scoring.

        Args:
            query: User query string.
            candidates: Results from earlier pipeline stages.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of :class:`RankedResult`.
        """
        if not candidates:
            return []

        try:
            return await self._rank_cohere(query, candidates, top_k)
        except Exception as exc:
            logger.warning("Stage3 CrossEncoder: Cohere rerank failed, using fallback: %s", exc)
            return self._rank_keyword_overlap(query, candidates, top_k)

    # ------------------------------------------------------------------
    # Cohere reranking
    # ------------------------------------------------------------------

    async def _rank_cohere(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Rerank via Cohere rerank-v4.0-pro wrapped with circuit breaker."""
        client = await self._get_cohere_client()
        if client is None:
            raise RuntimeError("Cohere client not available")

        breaker = await self._get_circuit_breaker()

        # Build document strings for Cohere
        documents: list[str] = []
        for r in candidates:
            parts = [r.name]
            if r.category:
                parts.append(f"[{r.category}]")
            if r.city:
                parts.append(f"in {r.city}")
            desc = r.metadata.get("description", "")
            if desc:
                parts.append(str(desc)[:500])
            documents.append(" ".join(parts))

        # Call Cohere with circuit breaker protection
        async with breaker:
            response = await client.rerank(
                model=settings.cohere_rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(candidates)),
            )

        reranked: list[RankedResult] = []
        for result in response.results:
            idx = result.index
            if idx < len(candidates):
                r = RankedResult(
                    entity_id=candidates[idx].entity_id,
                    name=candidates[idx].name,
                    category=candidates[idx].category,
                    city=candidates[idx].city,
                    score=result.relevance_score,
                    stage_scores={
                        **candidates[idx].stage_scores,
                        RankingStage.CROSS_ENCODER.value: result.relevance_score,
                    },
                    metadata=candidates[idx].metadata,
                    rank=0,
                )
                reranked.append(r)

        # Assign ranks
        for i, r in enumerate(reranked):
            r.rank = i + 1

        logger.debug(
            "Stage3 Cohere: %d candidates -> top %d",
            len(candidates),
            len(reranked),
        )
        return reranked

    # ------------------------------------------------------------------
    # Keyword-overlap fallback
    # ------------------------------------------------------------------

    def _rank_keyword_overlap(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Fallback scorer based on keyword overlap between query and candidate text.

        This is a simple Jaccard-like overlap score that requires no external
        API.  It is only used when Cohere is unavailable.
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            # Cannot score — preserve existing order
            return candidates[:top_k]

        scored: list[tuple[float, RankedResult]] = []

        for r in candidates:
            doc_text = f"{r.name} {r.category} {r.city} {r.metadata.get('description', '')}"
            doc_tokens = set(self._tokenize(doc_text))

            if not doc_tokens:
                overlap_score = 0.0
            else:
                intersection = len(query_tokens & doc_tokens)
                union = len(query_tokens | doc_tokens)
                overlap_score = intersection / union if union > 0 else 0.0

            scored.append((overlap_score, r))

        scored.sort(key=lambda t: t[0], reverse=True)

        reranked: list[RankedResult] = []
        for i, (overlap_score, r) in enumerate(scored[:top_k]):
            result = RankedResult(
                entity_id=r.entity_id,
                name=r.name,
                category=r.category,
                city=r.city,
                score=overlap_score,
                stage_scores={
                    **r.stage_scores,
                    RankingStage.CROSS_ENCODER.value: overlap_score,
                },
                metadata=r.metadata,
                rank=i + 1,
            )
            reranked.append(result)

        logger.debug(
            "Stage3 Fallback (keyword overlap): %d candidates -> top %d",
            len(candidates),
            len(reranked),
        )
        return reranked

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, remove stopwords."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: LLM Reranker (Gemini)
# ═══════════════════════════════════════════════════════════════════════════


class Stage4LLMReranker:
    """LLM-based reranker using Gemini to evaluate relevance.

    This is the most expensive stage and is only applied to a small number
    of top candidates (typically <= 20).  It sends each candidate to the
    LLM with a relevance-rating prompt and parses the numeric score.

    Fallback: pass-through (preserve existing order) if the LLM call fails.
    """

    PROMPT_TEMPLATE: str = (
        "Rate the relevance of the following entity to the user's query on "
        "a scale of 0 to 100, where 0 means completely irrelevant and 100 "
        "means a perfect match.\n\n"
        "Query: {query}\n\n"
        "Entity:\n"
        "  Name: {name}\n"
        "  Category: {category}\n"
        "  City: {city}\n"
        "  Description: {description}\n\n"
        "Respond with ONLY a single integer between 0 and 100. "
        "Do not include any other text."
    )

    async def rank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int = 10,
    ) -> list[RankedResult]:
        """Rerank candidates using LLM relevance scoring.

        Args:
            query: User query string.
            candidates: Results from earlier pipeline stages.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of :class:`RankedResult`.
        """
        if not candidates:
            return []

        try:
            return await self._rank_with_llm(query, candidates, top_k)
        except Exception as exc:
            logger.warning("Stage4 LLMReranker: LLM scoring failed, passing through: %s", exc)
            return self._passthrough(candidates, top_k)

    # ------------------------------------------------------------------
    # LLM scoring
    # ------------------------------------------------------------------

    async def _rank_with_llm(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Score each candidate by asking Gemini to rate its relevance."""
        from app.llm.gateway import get_llm_gateway

        gateway = get_llm_gateway()

        # Score candidates concurrently (bounded to avoid rate-limit storms)
        semaphore = asyncio.Semaphore(5)

        async def score_candidate(r: RankedResult) -> tuple[RankedResult, float]:
            async with semaphore:
                prompt = self.PROMPT_TEMPLATE.format(
                    query=query,
                    name=r.name,
                    category=r.category,
                    city=r.city,
                    description=str(r.metadata.get("description", ""))[:300],
                )
                messages = [{"role": "user", "content": prompt}]

                try:
                    response = await gateway.complete(
                        messages=messages,
                        task_type="fast",
                        temperature=0.0,
                        max_tokens=10,
                    )
                    llm_score = self._parse_score(response)
                except Exception as exc:
                    logger.debug("LLM scoring failed for entity=%s: %s", r.entity_id, exc)
                    # Preserve existing score as a fallback
                    llm_score = r.score * 100.0

                return r, llm_score

        tasks = [score_candidate(r) for r in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored: list[RankedResult] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Stage4 scoring task failed: %s", result)
                continue
            candidate, llm_score = result
            normalised = llm_score / 100.0  # Normalise to 0-1 range

            reranked = RankedResult(
                entity_id=candidate.entity_id,
                name=candidate.name,
                category=candidate.category,
                city=candidate.city,
                score=normalised,
                stage_scores={
                    **candidate.stage_scores,
                    RankingStage.LLM_RERANKER.value: normalised,
                },
                metadata=candidate.metadata,
                rank=0,
            )
            scored.append(reranked)

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]

        for i, r in enumerate(top):
            r.rank = i + 1

        logger.debug(
            "Stage4 LLM: %d candidates -> top %d",
            len(candidates),
            len(top),
        )
        return top

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score (0-100) from the LLM response.

        Searches for the first integer in the response string and clamps
        it to the [0, 100] range.

        Args:
            response: Raw text response from the LLM.

        Returns:
            A float score between 0.0 and 100.0.
        """
        if not response:
            return 50.0

        match = re.search(r"\d+", response.strip())
        if match:
            score = float(match.group())
            return max(0.0, min(100.0, score))

        return 50.0

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _passthrough(candidates: list[RankedResult], top_k: int) -> list[RankedResult]:
        """Return candidates in their existing order (fallback)."""
        top = candidates[:top_k]
        for i, r in enumerate(top):
            r.rank = i + 1
            r.stage_scores[RankingStage.LLM_RERANKER.value] = r.score
        return top


# ═══════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion (RRF) Merger
# ═══════════════════════════════════════════════════════════════════════════


class RRFMerger:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF produces a single fused ranking from N ranked lists by scoring
    each item as::

        fused_score = sum over lists L of:
            weight_L * 1 / (k + rank_in_L)

    The constant *k* (default 60) controls how much emphasis is placed on
    top-ranked items versus the full list.

    Usage::

        merger = RRFMerger()
        fused = merger.merge(
            ranked_lists=[dense_results, sparse_results],
            weights=[0.6, 0.4],
        )
    """

    def merge(
        self,
        ranked_lists: list[list[RankedResult]],
        weights: list[float] | None = None,
        k: int = 60,
    ) -> list[RankedResult]:
        """Fuse multiple ranked lists into a single list via RRF.

        Args:
            ranked_lists: Two or more ranked result lists.
            weights: Per-list weight multipliers.  Defaults to uniform
                weighting (1.0 for each list).
            k: RRF constant (default 60).

        Returns:
            A single sorted list of :class:`RankedResult`.
        """
        if not ranked_lists:
            return []

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        if len(weights) != len(ranked_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"ranked_lists length ({len(ranked_lists)})"
            )

        # entity_id -> (best RankedResult, fused_score)
        fused_scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, RankedResult] = {}
        merged_stage_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for list_idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
            for result in ranked_list:
                eid = result.entity_id
                rrf_score = weight * (1.0 / (k + result.rank))
                fused_scores[eid] += rrf_score

                # Keep the result object with the richest metadata
                if eid not in best_result or len(result.stage_scores) > len(
                    best_result[eid].stage_scores
                ):
                    best_result[eid] = result

                # Merge stage scores from all lists
                merged_stage_scores[eid].update(result.stage_scores)

        # Build final list
        fused: list[RankedResult] = []
        for eid, rrf_score in fused_scores.items():
            base = best_result[eid]
            merged = RankedResult(
                entity_id=eid,
                name=base.name,
                category=base.category,
                city=base.city,
                score=rrf_score,
                stage_scores=merged_stage_scores[eid],
                metadata=base.metadata,
                rank=0,
            )
            fused.append(merged)

        fused.sort(key=lambda r: r.score, reverse=True)

        for i, r in enumerate(fused):
            r.rank = i + 1

        logger.debug(
            "RRF Merge: %d lists -> %d unique entities",
            len(ranked_lists),
            len(fused),
        )
        return fused


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _StageStats:
    """Internal statistics for a single pipeline stage execution."""

    stage: str
    latency_ms: float
    input_count: int
    output_count: int


class RerankingPipeline:
    """Orchestrates the full 4-stage reranking pipeline.

    The pipeline narrows candidates through progressively more expensive
    stages:

    1. **Dense retrieval** (cosine similarity) → top 100
    2. **Sparse retrieval** (BM25) → top 50, merged with Stage 1 via RRF
    3. **Cross-encoder** (Cohere rerank) → top 20
    4. **LLM reranker** (Gemini) → top 10

    Stages can be selectively disabled via a config dict.

    Usage::

        pipeline = get_reranking_pipeline()
        results = await pipeline.rerank(query, embedding, candidates)
    """

    def __init__(
        self,
        stages: list[RankingStage] | None = None,
    ) -> None:
        """Initialise the pipeline with the desired stages.

        Args:
            stages: List of stages to execute.  Defaults to all four stages.
        """
        self.stages = stages or [
            RankingStage.DENSE_RETRIEVAL,
            RankingStage.SPARSE_RETRIEVAL,
            RankingStage.CROSS_ENCODER,
            RankingStage.LLM_RERANKER,
        ]

        self._stage1 = Stage1DenseRetrieval()
        self._stage2 = Stage2SparseRetrieval()
        self._stage3 = Stage3CrossEncoder()
        self._stage4 = Stage4LLMReranker()
        self._merger = RRFMerger()

        # Per-invocation statistics
        self._last_stats: list[_StageStats] = []

        logger.info(
            "RerankingPipeline initialised: stages=%s",
            [s.value for s in self.stages],
        )

    # ------------------------------------------------------------------
    # Full reranking
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[RankedResult]:
        """Execute the full multi-stage reranking pipeline.

        Args:
            query: User query string.
            query_embedding: Dense embedding vector for the query.
            candidates: Raw candidate dicts from initial retrieval.
            config: Optional configuration overrides::

                {
                    "stage1_top_k": 100,
                    "stage2_top_k": 50,
                    "stage3_top_k": 20,
                    "stage4_top_k": 10,
                    "skip_stages": [],      # list of RankingStage values to skip
                    "rrf_weights": [0.6, 0.4],  # dense vs sparse weights
                }

        Returns:
            Final reranked list of :class:`RankedResult`.
        """
        if not candidates:
            return []

        config = config or {}
        stage1_top_k = config.get("stage1_top_k", 100)
        stage2_top_k = config.get("stage2_top_k", 50)
        stage3_top_k = config.get("stage3_top_k", 20)
        stage4_top_k = config.get("stage4_top_k", 10)
        skip_stages = set(config.get("skip_stages", []))
        rrf_weights = config.get("rrf_weights", [0.6, 0.4])

        self._last_stats = []

        # ── Stages 1 & 2: Dense + Sparse (run in parallel) ──────────

        merged_results: list[RankedResult] = []

        run_dense = (
            RankingStage.DENSE_RETRIEVAL in self.stages
            and RankingStage.DENSE_RETRIEVAL.value not in skip_stages
        )
        run_sparse = (
            RankingStage.SPARSE_RETRIEVAL in self.stages
            and RankingStage.SPARSE_RETRIEVAL.value not in skip_stages
        )

        if run_dense and run_sparse:
            # Run both in parallel
            t0 = time.perf_counter()
            dense_task = self._stage1.rank(query_embedding, candidates, top_k=stage1_top_k)
            sparse_task = asyncio.get_event_loop().run_in_executor(
                None, self._stage2.rank, query, candidates, stage2_top_k
            )

            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task, return_exceptions=True
            )

            # Handle failures gracefully
            if isinstance(dense_results, Exception):
                logger.warning("Stage1 failed: %s", dense_results)
                dense_results = []
            if isinstance(sparse_results, Exception):
                logger.warning("Stage2 failed: %s", sparse_results)
                sparse_results = []

            dense_ms = (time.perf_counter() - t0) * 1000

            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.DENSE_RETRIEVAL.value,
                    latency_ms=dense_ms,
                    input_count=len(candidates),
                    output_count=len(dense_results),
                )
            )
            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.SPARSE_RETRIEVAL.value,
                    latency_ms=dense_ms,  # ran in parallel, same wall time
                    input_count=len(candidates),
                    output_count=len(sparse_results),
                )
            )

            # Merge via RRF
            if dense_results and sparse_results:
                merged_results = self._merger.merge(
                    ranked_lists=[dense_results, sparse_results],
                    weights=rrf_weights,
                )
            elif dense_results:
                merged_results = dense_results
            else:
                merged_results = sparse_results

        elif run_dense:
            t0 = time.perf_counter()
            merged_results = await self._stage1.rank(
                query_embedding, candidates, top_k=stage1_top_k
            )
            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.DENSE_RETRIEVAL.value,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_count=len(candidates),
                    output_count=len(merged_results),
                )
            )

        elif run_sparse:
            t0 = time.perf_counter()
            merged_results = self._stage2.rank(query, candidates, top_k=stage2_top_k)
            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.SPARSE_RETRIEVAL.value,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_count=len(candidates),
                    output_count=len(merged_results),
                )
            )

        else:
            # No retrieval stages — convert candidates to RankedResults
            merged_results = [
                RankedResult(
                    entity_id=str(c.get("entity_id", c.get("id", ""))),
                    name=c.get("name", ""),
                    category=c.get("category", ""),
                    city=c.get("city", ""),
                    score=float(c.get("score", 0.0)),
                    stage_scores={},
                    metadata={},
                    rank=i + 1,
                )
                for i, c in enumerate(candidates)
            ]

        if not merged_results:
            return []

        # ── Stage 3: Cross-Encoder ──────────────────────────────────

        stage3_input = merged_results
        if (
            RankingStage.CROSS_ENCODER in self.stages
            and RankingStage.CROSS_ENCODER.value not in skip_stages
            and len(stage3_input) > 1
        ):
            t0 = time.perf_counter()
            stage3_output = await self._stage3.rank(query, stage3_input, top_k=stage3_top_k)
            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.CROSS_ENCODER.value,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_count=len(stage3_input),
                    output_count=len(stage3_output),
                )
            )
            stage3_input = stage3_output

        # ── Stage 4: LLM Reranker ──────────────────────────────────

        stage4_input = stage3_input
        if (
            RankingStage.LLM_RERANKER in self.stages
            and RankingStage.LLM_RERANKER.value not in skip_stages
            and len(stage4_input) > 1
        ):
            t0 = time.perf_counter()
            stage4_output = await self._stage4.rank(query, stage4_input, top_k=stage4_top_k)
            self._last_stats.append(
                _StageStats(
                    stage=RankingStage.LLM_RERANKER.value,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_count=len(stage4_input),
                    output_count=len(stage4_output),
                )
            )
            return stage4_output

        return stage3_input

    # ------------------------------------------------------------------
    # Fast reranking (no API calls)
    # ------------------------------------------------------------------

    async def rerank_fast(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
    ) -> list[RankedResult]:
        """Fast reranking using only Stages 1 and 2 (no external API calls).

        Ideal for latency-sensitive paths where sub-50ms reranking is needed.

        Args:
            query: User query string.
            query_embedding: Dense embedding vector for the query.
            candidates: Raw candidate dicts.

        Returns:
            Reranked list of :class:`RankedResult`.
        """
        return await self.rerank(
            query=query,
            query_embedding=query_embedding,
            candidates=candidates,
            config={
                "skip_stages": [
                    RankingStage.CROSS_ENCODER.value,
                    RankingStage.LLM_RERANKER.value,
                ],
                "stage1_top_k": 50,
                "stage2_top_k": 50,
            },
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Return per-stage latency and candidate-count statistics.

        Returns:
            Dict with ``stages`` list and ``total_latency_ms``.
        """
        total_latency = sum(s.latency_ms for s in self._last_stats)
        return {
            "stages": [
                {
                    "stage": s.stage,
                    "latency_ms": round(s.latency_ms, 2),
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                }
                for s in self._last_stats
            ],
            "total_latency_ms": round(total_latency, 2),
            "stages_executed": len(self._last_stats),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_pipeline: Optional[RerankingPipeline] = None


def get_reranking_pipeline() -> RerankingPipeline:
    """Return the singleton RerankingPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RerankingPipeline()
    return _pipeline
