"""Predictive Prefetching Engine for ARBOR Enterprise.

Anticipates the user's next query based on session behaviour patterns and
pre-computes results so that subsequent searches return near-instantly.

The engine observes three main prediction strategies:

1. **Pattern matching** - detects common query transition sequences learned
   from the fashion / lifestyle discovery domain (e.g. city search followed
   by neighbourhood search followed by specific-entity lookup).
2. **Entity drill-down** - when a user clicks on entities from a result set,
   the engine predicts detail queries for those entities.
3. **Category broadening & refinement** - after a category-scoped search the
   engine predicts related categories in the same city as well as narrower
   filter refinements of the original query.

Pre-computed results are held in an in-memory TTL cache and served on cache
hit, dramatically reducing perceived latency for predictable navigation
flows.

Usage::

    orchestrator = get_prefetch_orchestrator()
    # After a search completes:
    await orchestrator.on_query_complete(
        user_id="u_42",
        query="cafes in Milan",
        intent="search",
        results=[{"entity_id": "e_1", "name": "Cafe Napoli"}, ...],
    )
    # Before running the next search, check the cache:
    cached = await orchestrator.check_prefetch("cafe Cafe Napoli details")
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SessionEvent:
    """A single recorded event within a user's browsing session.

    Attributes:
        user_id: Unique identifier for the user.
        query: The raw query string submitted by the user.
        intent: Classified intent of the query (e.g. ``search``, ``detail``,
            ``compare``).
        timestamp: Unix epoch when the event occurred.
        results_count: Number of results returned for this query.
        clicked_entity_ids: Entity IDs the user clicked on from the result
            set.
        session_id: Identifier grouping events into a contiguous session.
    """

    user_id: str
    query: str
    intent: str
    timestamp: float
    results_count: int
    clicked_entity_ids: list[str]
    session_id: str


@dataclass
class PrefetchCandidate:
    """A predicted future query that should be pre-computed.

    Attributes:
        predicted_query: The query string we expect the user to issue next.
        confidence: Estimated probability (0-1) that this prediction is
            correct.
        strategy: Which prediction strategy produced this candidate
            (``pattern``, ``drill_down``, ``broadening``, ``refinement``).
        prefetch_key: Deterministic cache key derived from the predicted
            query.
        source_session_id: The session that triggered this prediction.
    """

    predicted_query: str
    confidence: float
    strategy: str
    prefetch_key: str
    source_session_id: str


@dataclass
class PrefetchResult:
    """A pre-computed result set held in the prefetch cache.

    Attributes:
        prefetch_key: Cache key that identifies this entry.
        query: The query whose results are stored.
        results: Pre-computed result dicts.
        computed_at: Unix epoch when the results were computed.
        ttl: Time-to-live in seconds (default 300 = 5 minutes).
        hit_count: Number of times this entry has been served from cache.
    """

    prefetch_key: str
    query: str
    results: list[dict]
    computed_at: float
    ttl: int = 300
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_prefetch_key(query: str) -> str:
    """Return a deterministic cache key for *query*.

    The key is a hex digest of the lowercased, whitespace-normalised query
    so that minor formatting differences do not cause cache misses.
    """
    normalised = " ".join(query.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _normalise_query(query: str) -> str:
    """Lowercase and collapse whitespace for comparison purposes."""
    return " ".join(query.lower().split())


# ---------------------------------------------------------------------------
# Session predictor
# ---------------------------------------------------------------------------


class SessionPredictor:
    """Predicts a user's next queries from their recent session history.

    Maintains a sliding window of the most recent events for each user and
    applies multiple heuristic strategies to generate a ranked list of
    :class:`PrefetchCandidate` objects.

    Args:
        window_size: Maximum number of events to retain per user.
    """

    # Common category groupings in the fashion / lifestyle domain
    RELATED_CATEGORIES: dict[str, list[str]] = {
        "restaurants": ["bars", "cafes", "bistros"],
        "cafes": ["bakeries", "restaurants", "tea rooms"],
        "bars": ["clubs", "restaurants", "lounges"],
        "boutiques": ["concept stores", "showrooms", "ateliers"],
        "hotels": ["hostels", "resorts", "apartments"],
        "galleries": ["museums", "studios", "exhibitions"],
        "brands": ["designers", "labels", "ateliers"],
    }

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        # user_id -> deque of SessionEvent (most recent last)
        self._user_events: dict[str, deque[SessionEvent]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_event(self, event: SessionEvent) -> None:
        """Append *event* to the user's session history.

        Args:
            event: The session event to record.
        """
        self._user_events[event.user_id].append(event)
        logger.debug(
            "Recorded session event: user=%s query=%r intent=%s",
            event.user_id,
            event.query,
            event.intent,
        )

    def predict_next_queries(
        self,
        user_id: str,
        top_k: int = 3,
    ) -> list[PrefetchCandidate]:
        """Return up to *top_k* predicted next queries for *user_id*.

        Predictions are produced by several strategies and then merged,
        de-duplicated, and sorted by confidence (descending).

        Args:
            user_id: The user to predict for.
            top_k: Maximum number of candidates to return.

        Returns:
            A list of :class:`PrefetchCandidate` sorted by descending
            confidence.
        """
        events = self._user_events.get(user_id)
        if not events:
            return []

        last_event = events[-1]
        candidates: list[PrefetchCandidate] = []

        # Strategy 1 - Pattern matching (common query sequences)
        candidates.extend(self._predict_from_patterns(last_event, events))

        # Strategy 2 - Entity drill-down from clicked results
        candidates.extend(self._predict_from_clicked(last_event))

        # Strategy 3 - Refinement (narrowing the previous query)
        candidates.extend(self._predict_from_refinement(last_event))

        # Strategy 4 - Category broadening (related categories, same city)
        candidates.extend(self._predict_from_broadening(last_event))

        # De-duplicate by prefetch_key, keeping highest confidence
        seen: dict[str, PrefetchCandidate] = {}
        for c in candidates:
            existing = seen.get(c.prefetch_key)
            if existing is None or c.confidence > existing.confidence:
                seen[c.prefetch_key] = c
        unique = sorted(seen.values(), key=lambda c: c.confidence, reverse=True)

        selected = unique[:top_k]
        logger.debug(
            "Predicted %d candidates for user=%s (from %d raw)",
            len(selected),
            user_id,
            len(candidates),
        )
        return selected

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _get_session_patterns(self) -> dict[str, list[str]]:
        """Return hard-coded common query transition patterns.

        Keys are simplified intent/template descriptors; values are lists of
        follow-up query templates.  Templates may contain placeholders such
        as ``{city}``, ``{category}``, ``{brand}``, ``{A}``, ``{B}``.

        Returns:
            A dict mapping trigger patterns to predicted follow-up
            templates.
        """
        return {
            # "brands in {city}" -> detail & category pivots
            "brands in {city}": [
                "{category} in {city}",
                "{brand} details",
            ],
            # "compare {A} vs {B}" -> individual details
            "compare {a} vs {b}": [
                "{a} details",
                "{b} details",
            ],
            # "{category} {city}" -> drill into neighbourhood or add "best"
            "{category} {city}": [
                "{category} {neighbourhood}",
                "best {category} {city}",
            ],
            # city search -> neighbourhood search -> specific entity
            "search {city}": [
                "{category} in {city}",
                "things to do in {city}",
            ],
        }

    def _predict_from_patterns(
        self,
        last_event: SessionEvent,
        events: deque[SessionEvent],
    ) -> list[PrefetchCandidate]:
        """Predict follow-up queries using known transition patterns.

        Attempts to match the last query against hard-coded templates and
        generates concrete predictions by filling in template variables
        extracted from the query.

        Args:
            last_event: The most recent event.
            events: Full event window (used for multi-step pattern matching).

        Returns:
            A list of :class:`PrefetchCandidate` from pattern matches.
        """
        candidates: list[PrefetchCandidate] = []
        query_lower = last_event.query.lower().strip()

        # --- "brands in {city}" ----------------------------------------
        brands_match = re.match(r"brands?\s+in\s+(.+)", query_lower)
        if brands_match:
            city = brands_match.group(1).strip()
            for cat in ("boutiques", "restaurants", "cafes"):
                predicted = f"{cat} in {city}"
                candidates.append(
                    PrefetchCandidate(
                        predicted_query=predicted,
                        confidence=0.55,
                        strategy="pattern",
                        prefetch_key=_make_prefetch_key(predicted),
                        source_session_id=last_event.session_id,
                    )
                )

        # --- "compare {A} vs {B}" --------------------------------------
        compare_match = re.match(r"compare\s+(.+?)\s+vs\.?\s+(.+)", query_lower)
        if compare_match:
            entity_a = compare_match.group(1).strip()
            entity_b = compare_match.group(2).strip()
            for entity in (entity_a, entity_b):
                predicted = f"{entity} details"
                candidates.append(
                    PrefetchCandidate(
                        predicted_query=predicted,
                        confidence=0.70,
                        strategy="pattern",
                        prefetch_key=_make_prefetch_key(predicted),
                        source_session_id=last_event.session_id,
                    )
                )

        # --- "{category} {city}" (two-token heuristic) -----------------
        tokens = query_lower.split()
        if len(tokens) == 2:
            category, city = tokens
            predicted_best = f"best {category} {city}"
            candidates.append(
                PrefetchCandidate(
                    predicted_query=predicted_best,
                    confidence=0.45,
                    strategy="pattern",
                    prefetch_key=_make_prefetch_key(predicted_best),
                    source_session_id=last_event.session_id,
                )
            )

        # --- multi-step: city search -> category in city ---------------
        if len(events) >= 2:
            prev_event = events[-2]
            prev_lower = prev_event.query.lower().strip()
            city_match = re.match(r"(?:search|explore|discover)\s+(.+)", prev_lower)
            if city_match:
                city = city_match.group(1).strip()
                # If the latest query is already a category+city, predict
                # neighbourhood drill-down
                cat_city_match = re.match(rf"(\w+)\s+(?:in\s+)?{re.escape(city)}", query_lower)
                if cat_city_match:
                    cat = cat_city_match.group(1)
                    for nbhd in ("centro", "downtown", "old town"):
                        predicted = f"{cat} in {nbhd} {city}"
                        candidates.append(
                            PrefetchCandidate(
                                predicted_query=predicted,
                                confidence=0.40,
                                strategy="pattern",
                                prefetch_key=_make_prefetch_key(predicted),
                                source_session_id=last_event.session_id,
                            )
                        )

        return candidates

    # ------------------------------------------------------------------
    # Entity drill-down
    # ------------------------------------------------------------------

    def _predict_from_clicked(
        self,
        last_event: SessionEvent,
    ) -> list[PrefetchCandidate]:
        """Predict detail queries for entities the user clicked.

        When a user clicks on specific entities from a result list, there is
        a high probability they will next request the full details for one of
        those entities.

        Args:
            last_event: The most recent event (whose clicked entity IDs
                are inspected).

        Returns:
            A list of :class:`PrefetchCandidate` for entity detail lookups.
        """
        candidates: list[PrefetchCandidate] = []
        if not last_event.clicked_entity_ids:
            return candidates

        for idx, entity_id in enumerate(last_event.clicked_entity_ids):
            # Confidence decreases for entities clicked later (lower signal)
            confidence = max(0.30, 0.80 - idx * 0.15)
            predicted = f"entity detail {entity_id}"
            candidates.append(
                PrefetchCandidate(
                    predicted_query=predicted,
                    confidence=confidence,
                    strategy="drill_down",
                    prefetch_key=_make_prefetch_key(predicted),
                    source_session_id=last_event.session_id,
                )
            )

        logger.debug(
            "drill_down: %d candidates from %d clicked entities",
            len(candidates),
            len(last_event.clicked_entity_ids),
        )
        return candidates

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _predict_from_refinement(
        self,
        last_event: SessionEvent,
    ) -> list[PrefetchCandidate]:
        """Predict refined (narrower) versions of the last query.

        If the last search returned many results, the user is likely to add
        qualifiers such as a price tier, a style tag, or a neighbourhood
        filter.

        Args:
            last_event: The most recent event.

        Returns:
            A list of :class:`PrefetchCandidate` for refined queries.
        """
        candidates: list[PrefetchCandidate] = []
        query = last_event.query.strip()

        # Only refine broad searches that returned multiple results
        if last_event.results_count < 5:
            return candidates

        refinement_suffixes = [
            ("affordable", 0.40),
            ("luxury", 0.35),
            ("top rated", 0.45),
            ("near me", 0.30),
            ("open now", 0.30),
        ]

        for suffix, confidence in refinement_suffixes:
            # Skip if the original query already contains the refinement
            if suffix in query.lower():
                continue
            predicted = f"{query} {suffix}"
            candidates.append(
                PrefetchCandidate(
                    predicted_query=predicted,
                    confidence=confidence,
                    strategy="refinement",
                    prefetch_key=_make_prefetch_key(predicted),
                    source_session_id=last_event.session_id,
                )
            )

        logger.debug(
            "refinement: %d candidates from query=%r (results_count=%d)",
            len(candidates),
            query,
            last_event.results_count,
        )
        return candidates

    # ------------------------------------------------------------------
    # Category broadening
    # ------------------------------------------------------------------

    def _predict_from_broadening(
        self,
        last_event: SessionEvent,
    ) -> list[PrefetchCandidate]:
        """Predict queries for related categories in the same city.

        After searching for e.g. "Italian restaurants Milan", the engine
        predicts that the user might also look for "bars Milan" or
        "cafes Milan".

        Args:
            last_event: The most recent event.

        Returns:
            A list of :class:`PrefetchCandidate` for broadened queries.
        """
        candidates: list[PrefetchCandidate] = []
        query_lower = last_event.query.lower().strip()

        # Try to extract a (category, city) pair from the query
        # Handles patterns: "{category} in {city}", "{adj} {category} {city}"
        cat_city_match = re.match(r"(?:\w+\s+)?(\w+)\s+(?:in\s+)?(\w+)$", query_lower)
        if not cat_city_match:
            return candidates

        category = cat_city_match.group(1)
        city = cat_city_match.group(2)

        related = self.RELATED_CATEGORIES.get(category, [])
        for idx, related_cat in enumerate(related):
            confidence = max(0.25, 0.50 - idx * 0.10)
            predicted = f"{related_cat} in {city}"
            candidates.append(
                PrefetchCandidate(
                    predicted_query=predicted,
                    confidence=confidence,
                    strategy="broadening",
                    prefetch_key=_make_prefetch_key(predicted),
                    source_session_id=last_event.session_id,
                )
            )

        logger.debug(
            "broadening: %d candidates for category=%s city=%s",
            len(candidates),
            category,
            city,
        )
        return candidates


# ---------------------------------------------------------------------------
# Prefetch cache
# ---------------------------------------------------------------------------


class PrefetchCache:
    """In-memory TTL cache for pre-computed search results.

    Entries are evicted lazily on access and eagerly via
    :meth:`evict_expired`.  Basic statistics (hit/miss counts) are tracked
    so that cache effectiveness can be monitored.
    """

    def __init__(self) -> None:
        self._store: dict[str, PrefetchResult] = {}
        self._total_hits: int = 0
        self._total_misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        prefetch_key: str,
        query: str,
        results: list[dict],
        ttl: int = 300,
    ) -> None:
        """Store pre-computed *results* under *prefetch_key*.

        If an entry with the same key already exists it is overwritten.

        Args:
            prefetch_key: Cache key for this entry.
            query: The query whose results are being cached.
            results: Pre-computed result dicts.
            ttl: Time-to-live in seconds (default 300).
        """
        self._store[prefetch_key] = PrefetchResult(
            prefetch_key=prefetch_key,
            query=query,
            results=results,
            computed_at=time.time(),
            ttl=ttl,
            hit_count=0,
        )
        logger.debug(
            "Cache store: key=%s query=%r results=%d ttl=%ds",
            prefetch_key,
            query,
            len(results),
            ttl,
        )

    def get(self, prefetch_key: str) -> PrefetchResult | None:
        """Retrieve a cached result by exact *prefetch_key*.

        Returns ``None`` if the key is absent or the entry has expired
        (in which case it is evicted immediately).

        Args:
            prefetch_key: The cache key to look up.

        Returns:
            The cached :class:`PrefetchResult` or ``None``.
        """
        entry = self._store.get(prefetch_key)
        if entry is None:
            self._total_misses += 1
            return None

        if time.time() - entry.computed_at > entry.ttl:
            del self._store[prefetch_key]
            self._total_misses += 1
            logger.debug("Cache expired: key=%s", prefetch_key)
            return None

        entry.hit_count += 1
        self._total_hits += 1
        logger.debug(
            "Cache hit: key=%s query=%r hit_count=%d",
            prefetch_key,
            entry.query,
            entry.hit_count,
        )
        return entry

    def get_by_query(self, query: str) -> PrefetchResult | None:
        """Retrieve a cached result by fuzzy-matching against *query*.

        The lookup normalises both the incoming query and stored queries to
        lowercase with collapsed whitespace and checks for an exact
        normalised match.  If no exact match is found, a simple
        substring-containment check is used as a fallback.

        Args:
            query: The query to match against cached entries.

        Returns:
            The best-matching :class:`PrefetchResult` or ``None``.
        """
        target = _normalise_query(query)
        now = time.time()

        # Pass 1: exact normalised match
        for entry in self._store.values():
            if now - entry.computed_at > entry.ttl:
                continue
            if _normalise_query(entry.query) == target:
                entry.hit_count += 1
                self._total_hits += 1
                return entry

        # Pass 2: substring containment (either direction)
        for entry in self._store.values():
            if now - entry.computed_at > entry.ttl:
                continue
            cached_norm = _normalise_query(entry.query)
            if target in cached_norm or cached_norm in target:
                entry.hit_count += 1
                self._total_hits += 1
                logger.debug(
                    "Cache fuzzy hit: query=%r matched cached=%r",
                    query,
                    entry.query,
                )
                return entry

        self._total_misses += 1
        return None

    def evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            The number of entries evicted.
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._store.items() if now - entry.computed_at > entry.ttl
        ]
        for key in expired_keys:
            del self._store[key]

        if expired_keys:
            logger.info("Evicted %d expired prefetch entries", len(expired_keys))
        return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            A dict containing ``size``, ``hit_rate``, and
            ``avg_ttl_remaining`` (in seconds).
        """
        now = time.time()
        total_lookups = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total_lookups if total_lookups > 0 else 0.0

        ttl_remaining: list[float] = []
        for entry in self._store.values():
            remaining = entry.ttl - (now - entry.computed_at)
            if remaining > 0:
                ttl_remaining.append(remaining)

        avg_ttl_remaining = sum(ttl_remaining) / len(ttl_remaining) if ttl_remaining else 0.0

        return {
            "size": len(self._store),
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(hit_rate, 4),
            "avg_ttl_remaining": round(avg_ttl_remaining, 1),
        }


# ---------------------------------------------------------------------------
# Prefetch orchestrator
# ---------------------------------------------------------------------------


class PrefetchOrchestrator:
    """Coordinates prediction, prefetching, and cache management.

    Ties together a :class:`SessionPredictor` (to forecast the next query)
    and a :class:`PrefetchCache` (to store pre-computed results) with an
    async task pool that executes background prefetch jobs.

    Args:
        predictor: The session predictor instance.
        cache: The prefetch cache instance.
        max_concurrent_prefetches: Upper limit on simultaneously running
            background prefetch tasks.
    """

    def __init__(
        self,
        predictor: SessionPredictor,
        cache: PrefetchCache,
        max_concurrent_prefetches: int = 3,
    ) -> None:
        self._predictor = predictor
        self._cache = cache
        self._max_concurrent = max_concurrent_prefetches
        self._active_tasks: list[asyncio.Task] = []  # type: ignore[type-arg]
        self._total_prefetches: int = 0
        self._total_hits: int = 0
        self._total_predictions: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_query_complete(
        self,
        user_id: str,
        query: str,
        intent: str,
        results: list[dict[str, Any]],
        session_id: str = "",
        clicked_entity_ids: list[str] | None = None,
    ) -> list[PrefetchCandidate]:
        """Handle a completed query by recording the event and prefetching.

        This is the main entry point called after every search. It:

        1. Records the query as a :class:`SessionEvent`.
        2. Asks the predictor for the top candidate queries.
        3. Launches background tasks to prefetch the predicted queries.

        Args:
            user_id: The user who issued the query.
            query: The query string.
            intent: Classified query intent.
            results: The results returned for this query.
            session_id: Optional session identifier.
            clicked_entity_ids: Entity IDs clicked so far in this session.

        Returns:
            The list of :class:`PrefetchCandidate` that were submitted for
            background prefetching.
        """
        event = SessionEvent(
            user_id=user_id,
            query=query,
            intent=intent,
            timestamp=time.time(),
            results_count=len(results),
            clicked_entity_ids=clicked_entity_ids or [],
            session_id=session_id or f"session_{user_id}_{int(time.time())}",
        )
        self._predictor.record_event(event)

        candidates = self._predictor.predict_next_queries(user_id)
        self._total_predictions += len(candidates)

        # Evict expired cache entries periodically
        self._cache.evict_expired()

        # Launch prefetch tasks (up to the concurrency limit)
        self._cleanup_finished_tasks()
        slots_available = self._max_concurrent - len(self._active_tasks)
        launched: list[PrefetchCandidate] = []

        for candidate in candidates:
            if slots_available <= 0:
                break
            # Skip if already cached
            if self._cache.get(candidate.prefetch_key) is not None:
                continue
            task = asyncio.create_task(self._prefetch_query(candidate))
            self._active_tasks.append(task)
            launched.append(candidate)
            slots_available -= 1

        if launched:
            logger.info(
                "Launched %d prefetch tasks for user=%s (query=%r)",
                len(launched),
                user_id,
                query,
            )

        return launched

    async def check_prefetch(self, query: str) -> list[dict] | None:
        """Check whether pre-computed results exist for *query*.

        Performs an exact key lookup first, then falls back to a fuzzy
        query match.

        Args:
            query: The query to check.

        Returns:
            The pre-computed result list if a cache hit occurs, otherwise
            ``None``.
        """
        prefetch_key = _make_prefetch_key(query)

        # Try exact key match
        result = self._cache.get(prefetch_key)
        if result is not None:
            self._total_hits += 1
            logger.info("Prefetch cache HIT (exact): query=%r", query)
            return result.results

        # Try fuzzy query match
        result = self._cache.get_by_query(query)
        if result is not None:
            self._total_hits += 1
            logger.info("Prefetch cache HIT (fuzzy): query=%r", query)
            return result.results

        logger.debug("Prefetch cache MISS: query=%r", query)
        return None

    async def _prefetch_query(self, candidate: PrefetchCandidate) -> None:
        """Execute a single prefetch job for *candidate*.

        Runs the real search pipeline: embedding generation via the LLM
        gateway followed by RRF hybrid search on Qdrant, then stores the
        results in the prefetch cache.

        Args:
            candidate: The prefetch candidate describing which query to
                pre-compute.
        """
        try:
            logger.debug(
                "Prefetching: query=%r strategy=%s confidence=%.2f",
                candidate.predicted_query,
                candidate.strategy,
                candidate.confidence,
            )

            from app.db.qdrant.hybrid_search import HybridSearch
            from app.llm.gateway import get_llm_gateway

            gateway = get_llm_gateway()
            hybrid = HybridSearch()

            # Generate embedding for the predicted query
            query_embedding = await gateway.get_embedding(candidate.predicted_query)

            # Execute RRF hybrid search
            results = await hybrid.search_rrf(
                query_vector=query_embedding,
                query_text=candidate.predicted_query,
                limit=10,
            )

            # Annotate results with prefetch metadata
            for r in results:
                r["prefetched"] = True
                r["strategy"] = candidate.strategy

            self._cache.store(
                prefetch_key=candidate.prefetch_key,
                query=candidate.predicted_query,
                results=results,
                ttl=300,
            )
            self._total_prefetches += 1

            logger.debug(
                "Prefetch complete: query=%r results=%d",
                candidate.predicted_query,
                len(results),
            )

        except Exception:
            logger.exception("Prefetch failed for query=%r", candidate.predicted_query)

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics for the orchestrator.

        Returns:
            A dict containing prefetch counts, hit rate, prediction volume,
            and the underlying cache stats.
        """
        hit_rate = self._total_hits / self._total_prefetches if self._total_prefetches > 0 else 0.0
        return {
            "total_prefetches": self._total_prefetches,
            "total_hits": self._total_hits,
            "prefetch_hit_rate": round(hit_rate, 4),
            "total_predictions": self._total_predictions,
            "active_tasks": len([t for t in self._active_tasks if not t.done()]),
            "cache": self._cache.stats(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_finished_tasks(self) -> None:
        """Remove completed tasks from the active task list."""
        self._active_tasks = [t for t in self._active_tasks if not t.done()]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_orchestrator: PrefetchOrchestrator | None = None


def get_prefetch_orchestrator() -> PrefetchOrchestrator:
    """Return the singleton :class:`PrefetchOrchestrator` instance.

    On first call, initialises a :class:`SessionPredictor` and
    :class:`PrefetchCache` with default settings.

    Returns:
        The global PrefetchOrchestrator.
    """
    global _orchestrator
    if _orchestrator is None:
        predictor = SessionPredictor(window_size=20)
        cache = PrefetchCache()
        _orchestrator = PrefetchOrchestrator(
            predictor=predictor,
            cache=cache,
            max_concurrent_prefetches=3,
        )
        logger.info("PrefetchOrchestrator initialised (singleton)")
    return _orchestrator
