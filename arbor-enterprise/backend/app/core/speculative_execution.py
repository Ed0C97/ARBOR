"""Speculative Execution Engine for ARBOR Enterprise.

Launches multiple search/retrieval strategies in parallel and returns the
first one that meets the quality threshold.  Inspired by speculative execution
in CPU design: bet on several strategies at once, commit the winner, discard
the rest.

Architecture:
    SpeculativeExecutor
        |-- DenseSearchStrategy      (Qdrant vector search)
        |-- HybridSearchStrategy     (vector + metadata RRF)
        |-- GraphRAGStrategy         (Neo4j knowledge graph traversal)
        |-- FullPipelineStrategy     (IntentRouter -> agents -> Curator)
        |
        +-- SpeculationArbiter       (quality gate)
        +-- StrategySelector         (learned routing)

The executor fires all (or selected) strategies concurrently.  As each
completes, the arbiter evaluates whether the result meets the quality bar.
The first result that passes is committed and all remaining tasks are
cancelled, saving latency and compute.

A StrategySelector learns over time which strategies perform best for
different (intent, complexity) pairs so that the preferred ordering adapts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ===========================================================================
# Strategy Result
# ===========================================================================


@dataclass
class StrategyResult:
    """Outcome produced by a single execution strategy.

    Attributes:
        strategy_name: Identifier of the strategy that produced this result.
        results: List of result dicts (entities, documents, summaries, etc.).
        confidence: Model/heuristic confidence in the quality of the results,
                    in the range [0.0, 1.0].
        latency_ms: Wall-clock time the strategy took, in milliseconds.
        cost_estimate: Estimated dollar cost of the strategy execution.
        is_complete: Whether the strategy ran to completion without being
                     cancelled or timing out.
        error: Optional error message if the strategy failed.
    """

    strategy_name: str
    results: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    is_complete: bool = False
    error: str | None = None


# ===========================================================================
# Abstract Execution Strategy
# ===========================================================================


class ExecutionStrategy(ABC):
    """Base class for all retrieval/search strategies.

    Each concrete strategy encapsulates one approach for answering a user
    query.  Strategies are designed to be cheap to instantiate and stateless
    so that they can be shared across concurrent executions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short identifier for this strategy."""

    @property
    @abstractmethod
    def estimated_cost(self) -> float:
        """Estimated dollar cost per invocation."""

    @abstractmethod
    async def execute(self, query: str, context: dict[str, Any] | None = None) -> StrategyResult:
        """Run the strategy against the given *query*.

        Args:
            query: The user's natural-language query.
            context: Optional dict carrying intent classification, filters,
                     user preferences, embeddings, etc.

        Returns:
            A fully populated :class:`StrategyResult`.
        """


# ===========================================================================
# Concrete Strategies
# ===========================================================================


class DenseSearchStrategy(ExecutionStrategy):
    """Pure dense vector search via Qdrant.

    Performs a semantic similarity lookup against the entities_vectors
    collection.  Cheapest and fastest strategy but may miss keyword-heavy
    or graph-relational queries.
    """

    @property
    def name(self) -> str:
        return "dense_search"

    @property
    def estimated_cost(self) -> float:
        return 0.001

    async def execute(self, query: str, context: dict[str, Any] | None = None) -> StrategyResult:
        start = time.perf_counter()
        context = context or {}

        try:
            from app.agents.vector_agent import VectorAgent

            agent = VectorAgent()
            raw_results = await agent.execute(
                query=query,
                filters=context.get("filters", {}),
                limit=context.get("limit", 10),
            )

            # Normalise results to list[dict]
            results: list[dict] = []
            if isinstance(raw_results, list):
                results = [r if isinstance(r, dict) else {"data": r} for r in raw_results]

            # Heuristic confidence: based on result count and top-score if present
            confidence = self._compute_confidence(results)

            elapsed_ms = (time.perf_counter() - start) * 1000
            return StrategyResult(
                strategy_name=self.name,
                results=results,
                confidence=confidence,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=True,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("DenseSearchStrategy failed: %s", exc)
            return StrategyResult(
                strategy_name=self.name,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=False,
                error=str(exc),
            )

    @staticmethod
    def _compute_confidence(results: list[dict]) -> float:
        """Heuristic confidence based on result quality signals."""
        if not results:
            return 0.0

        # Base confidence scales with result count (diminishing returns)
        count_factor = min(len(results) / 5, 1.0) * 0.5

        # Boost from top result score if present
        top_score = results[0].get("score", 0.0) if results else 0.0
        score_factor = min(float(top_score), 1.0) * 0.5

        return min(count_factor + score_factor, 1.0)


class HybridSearchStrategy(ExecutionStrategy):
    """Hybrid vector + metadata search with Reciprocal Rank Fusion.

    Combines dense vector similarity from Qdrant with structured metadata
    filtering, merging results via RRF.  Slightly more expensive than pure
    dense search but handles keyword-rich and filtered queries better.
    """

    @property
    def name(self) -> str:
        return "hybrid_search"

    @property
    def estimated_cost(self) -> float:
        return 0.002

    async def execute(self, query: str, context: dict[str, Any] | None = None) -> StrategyResult:
        start = time.perf_counter()
        context = context or {}

        try:
            from app.db.qdrant.hybrid_search import HybridSearch

            hybrid = HybridSearch(collection=settings.qdrant_collection)

            # Obtain query embedding from context or generate one
            query_vector = context.get("query_embedding")
            if query_vector is None:
                query_vector = await self._embed_query(query)

            if query_vector is None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                return StrategyResult(
                    strategy_name=self.name,
                    latency_ms=elapsed_ms,
                    cost_estimate=self.estimated_cost,
                    is_complete=False,
                    error="Unable to generate query embedding",
                )

            raw_results = hybrid.search(
                query_vector=query_vector,
                query_text=query,
                limit=context.get("limit", 10),
                category=context.get("filters", {}).get("category"),
                city=context.get("filters", {}).get("city"),
            )

            results: list[dict] = []
            if isinstance(raw_results, list):
                results = [r if isinstance(r, dict) else {"data": r} for r in raw_results]

            confidence = self._compute_confidence(results)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return StrategyResult(
                strategy_name=self.name,
                results=results,
                confidence=confidence,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=True,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("HybridSearchStrategy failed: %s", exc)
            return StrategyResult(
                strategy_name=self.name,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=False,
                error=str(exc),
            )

    @staticmethod
    async def _embed_query(query: str) -> list[float] | None:
        """Generate an embedding vector for the query via the LLM gateway."""
        try:
            from app.llm.gateway import get_llm_gateway

            gateway = get_llm_gateway()
            embedding = await gateway.embed(query)
            return embedding
        except Exception as exc:
            logger.warning("HybridSearchStrategy: embedding generation failed: %s", exc)
            return None

    @staticmethod
    def _compute_confidence(results: list[dict]) -> float:
        """Heuristic confidence combining count and RRF scores."""
        if not results:
            return 0.0

        count_factor = min(len(results) / 5, 1.0) * 0.4

        # Average RRF/fusion score across results
        scores = [float(r.get("score", r.get("rrf_score", 0.0))) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_factor = min(avg_score, 1.0) * 0.6

        return min(count_factor + score_factor, 1.0)


class GraphRAGStrategy(ExecutionStrategy):
    """Knowledge graph traversal with LLM-enhanced reasoning via Neo4j.

    Uses the GraphRAG module to perform Cypher queries, local entity
    expansion, and global knowledge graph search with LLM summarisation.
    Higher cost due to graph traversal + LLM reasoning but excels at
    relational and comparison queries.
    """

    @property
    def name(self) -> str:
        return "graph_rag"

    @property
    def estimated_cost(self) -> float:
        return 0.005

    async def execute(self, query: str, context: dict[str, Any] | None = None) -> StrategyResult:
        start = time.perf_counter()
        context = context or {}

        try:
            from app.db.neo4j.graphrag import GraphRAG

            graph_rag = GraphRAG()
            query_type = context.get("graph_query_type", "auto")
            raw_result = await graph_rag.smart_query(query, query_type=query_type)

            # Normalise: smart_query returns a dict with a "results" key
            results: list[dict] = []
            if isinstance(raw_result, dict):
                inner = raw_result.get("results", [])
                if isinstance(inner, list):
                    results = [r if isinstance(r, dict) else {"data": r} for r in inner]
                else:
                    results = [raw_result]
            elif isinstance(raw_result, list):
                results = [r if isinstance(r, dict) else {"data": r} for r in raw_result]

            confidence = self._compute_confidence(results, raw_result)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return StrategyResult(
                strategy_name=self.name,
                results=results,
                confidence=confidence,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=True,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("GraphRAGStrategy failed: %s", exc)
            return StrategyResult(
                strategy_name=self.name,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=False,
                error=str(exc),
            )

    @staticmethod
    def _compute_confidence(results: list[dict], raw_result: Any = None) -> float:
        """Confidence from graph result density and summary presence."""
        if not results:
            return 0.0

        count_factor = min(len(results) / 3, 1.0) * 0.4

        # Boost if the raw result contains an LLM-generated summary
        has_summary = isinstance(raw_result, dict) and bool(
            raw_result.get("summary") or raw_result.get("answer")
        )
        summary_factor = 0.4 if has_summary else 0.1

        # Small boost for graph traversal type (local/global vs simple cypher)
        query_type = raw_result.get("type", "cypher") if isinstance(raw_result, dict) else "cypher"
        type_boost = 0.2 if query_type in ("local", "global") else 0.1

        return min(count_factor + summary_factor + type_boost, 1.0)


class FullPipelineStrategy(ExecutionStrategy):
    """Complete agent swarm: IntentRouter -> all agents -> Curator.

    The most expensive strategy but produces the highest quality, curated
    response.  Runs the full LangGraph agent graph including intent
    classification, parallel vector/metadata/graph search, optional
    historian pass, and final curation.
    """

    @property
    def name(self) -> str:
        return "full_pipeline"

    @property
    def estimated_cost(self) -> float:
        return 0.05

    async def execute(self, query: str, context: dict[str, Any] | None = None) -> StrategyResult:
        start = time.perf_counter()
        context = context or {}

        try:
            from app.agents.graph import create_agent_graph

            graph = create_agent_graph(
                session=context.get("session"),
                arbor_session=context.get("arbor_session"),
            )

            initial_state = {
                "user_query": query,
                "user_location": context.get("user_location"),
                "user_preferences": context.get("user_preferences", {}),
                "intent": "",
                "intent_confidence": 0.0,
                "entities_mentioned": [],
                "filters": context.get("filters", {}),
                "vector_results": [],
                "metadata_results": [],
                "graph_results": [],
                "final_response": "",
                "recommendations": [],
                "confidence_score": 0.0,
                "sources_used": [],
            }

            # Run the compiled LangGraph
            final_state = await graph.ainvoke(initial_state)

            # Extract results from final state
            recommendations = final_state.get("recommendations", [])
            results: list[dict] = []
            if isinstance(recommendations, list):
                results = [r if isinstance(r, dict) else {"data": r} for r in recommendations]

            # Include the curated response as a meta-result if results are empty
            if not results and final_state.get("final_response"):
                results = [{"final_response": final_state["final_response"]}]

            confidence = final_state.get("confidence_score", 0.0)
            if not confidence and results:
                confidence = 0.8  # Full pipeline inherently higher quality

            elapsed_ms = (time.perf_counter() - start) * 1000

            return StrategyResult(
                strategy_name=self.name,
                results=results,
                confidence=float(confidence),
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=True,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("FullPipelineStrategy failed: %s", exc)
            return StrategyResult(
                strategy_name=self.name,
                latency_ms=elapsed_ms,
                cost_estimate=self.estimated_cost,
                is_complete=False,
                error=str(exc),
            )


# ===========================================================================
# Speculation Arbiter
# ===========================================================================


class SpeculationArbiter:
    """Evaluates whether a :class:`StrategyResult` meets the quality bar.

    Configurable thresholds allow tuning the trade-off between latency
    (accepting a fast-but-lower-quality result) and quality (waiting for
    a more expensive strategy).

    Args:
        min_confidence: Minimum confidence score to pass.  Default 0.6.
        min_results: Minimum number of results to pass.  Default 1.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_results: int = 1,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_results = min_results

    def evaluate(self, result: StrategyResult) -> bool:
        """Return ``True`` if *result* meets all quality criteria."""
        if result.error is not None:
            return False

        if not result.is_complete:
            return False

        if result.confidence < self.min_confidence:
            return False

        if len(result.results) < self.min_results:
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"SpeculationArbiter(min_confidence={self.min_confidence}, "
            f"min_results={self.min_results})"
        )


# ===========================================================================
# Speculative Executor
# ===========================================================================


class SpeculativeExecutor:
    """Orchestrates parallel strategy execution with first-past-the-post wins.

    Launches all configured strategies concurrently.  As each strategy
    completes, the arbiter evaluates the result.  The first result that
    passes the quality gate is returned immediately and all remaining
    in-flight tasks are cancelled.

    If no strategy passes within *timeout_seconds*, the best available
    result is returned.

    Args:
        strategies: Ordered list of strategies to execute.
        arbiter: Quality gate evaluator.
        timeout_seconds: Maximum wall-clock time before returning best-effort.
    """

    def __init__(
        self,
        strategies: list[ExecutionStrategy],
        arbiter: SpeculationArbiter | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.strategies = strategies
        self.arbiter = arbiter or SpeculationArbiter()
        self.timeout_seconds = timeout_seconds

        # Performance tracking
        self._win_counts: dict[str, int] = defaultdict(int)
        self._total_executions: int = 0
        self._latency_accum: dict[str, list[float]] = defaultdict(list)
        self._cost_accum: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    # ----- core execution ---------------------------------------------------

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> StrategyResult:
        """Launch all strategies concurrently and return the first winner.

        Args:
            query: User query.
            context: Shared context dict (filters, embeddings, sessions, ...).

        Returns:
            The first :class:`StrategyResult` that passes the arbiter, or
            the best-effort result if the timeout is reached.
        """
        self._total_executions += 1
        context = context or {}

        # Create a completion event so we can signal the winner
        winner_event = asyncio.Event()
        winner_result: list[StrategyResult] = []  # mutable container for closure
        all_results: list[StrategyResult] = []

        async def _run_strategy(strategy: ExecutionStrategy) -> None:
            """Execute one strategy and check with the arbiter."""
            try:
                result = await strategy.execute(query, context)
            except Exception as exc:
                result = StrategyResult(
                    strategy_name=strategy.name,
                    is_complete=False,
                    error=str(exc),
                    cost_estimate=strategy.estimated_cost,
                )

            all_results.append(result)

            # Record stats
            async with self._lock:
                self._latency_accum[strategy.name].append(result.latency_ms)
                self._cost_accum[strategy.name].append(result.cost_estimate)

            # Check quality gate
            if self.arbiter.evaluate(result) and not winner_event.is_set():
                winner_result.append(result)
                winner_event.set()

        # Launch all strategies as tasks
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(_run_strategy(strategy), name=f"spec_{strategy.name}")
            for strategy in self.strategies
        ]

        try:
            # Wait for either a winner or timeout
            await asyncio.wait_for(winner_event.wait(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "SpeculativeExecutor: timeout (%.1fs) reached with %d/%d strategies complete",
                self.timeout_seconds,
                len(all_results),
                len(self.strategies),
            )

        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Suppress CancelledErrors from cancelled tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        # Determine the result to return
        if winner_result:
            chosen = winner_result[0]
            logger.info(
                "SpeculativeExecutor: winner=%s confidence=%.2f latency=%.1fms",
                chosen.strategy_name,
                chosen.confidence,
                chosen.latency_ms,
            )
        else:
            # No strategy passed the arbiter; return the best available
            chosen = self._select_best(all_results)
            logger.info(
                "SpeculativeExecutor: no winner, best-effort=%s confidence=%.2f",
                chosen.strategy_name,
                chosen.confidence,
            )

        # Track win
        async with self._lock:
            self._win_counts[chosen.strategy_name] += 1

        return chosen

    async def execute_with_fallback(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        preferred_strategy: str | None = None,
    ) -> StrategyResult:
        """Try a preferred strategy first; fall back to speculative execution.

        If *preferred_strategy* is specified and produces a passing result
        within 2 seconds, it is returned immediately.  Otherwise the
        remaining strategies are launched speculatively.

        Args:
            query: User query.
            context: Shared context dict.
            preferred_strategy: Name of the strategy to try first.

        Returns:
            A :class:`StrategyResult`.
        """
        context = context or {}
        preferred_timeout = 2.0

        if preferred_strategy:
            strategy = self._find_strategy(preferred_strategy)
            if strategy is not None:
                try:
                    result = await asyncio.wait_for(
                        strategy.execute(query, context),
                        timeout=preferred_timeout,
                    )

                    # Record stats
                    async with self._lock:
                        self._latency_accum[strategy.name].append(result.latency_ms)
                        self._cost_accum[strategy.name].append(result.cost_estimate)

                    if self.arbiter.evaluate(result):
                        async with self._lock:
                            self._win_counts[strategy.name] += 1
                            self._total_executions += 1
                        logger.info(
                            "SpeculativeExecutor: preferred=%s passed in %.1fms",
                            strategy.name,
                            result.latency_ms,
                        )
                        return result

                    logger.info(
                        "SpeculativeExecutor: preferred=%s did not pass arbiter "
                        "(confidence=%.2f, results=%d), falling back to speculative",
                        strategy.name,
                        result.confidence,
                        len(result.results),
                    )

                except asyncio.TimeoutError:
                    logger.info(
                        "SpeculativeExecutor: preferred=%s timed out (%.1fs), "
                        "falling back to speculative",
                        preferred_strategy,
                        preferred_timeout,
                    )

        # Fall back to full speculative execution with remaining strategies
        remaining = [s for s in self.strategies if s.name != preferred_strategy]
        if not remaining:
            remaining = self.strategies  # fallback to all if preferred was the only one

        executor = SpeculativeExecutor(
            strategies=remaining,
            arbiter=self.arbiter,
            timeout_seconds=self.timeout_seconds,
        )
        result = await executor.execute(query, context)

        # Merge stats from the sub-executor
        async with self._lock:
            self._total_executions += 1
            for name, latencies in executor._latency_accum.items():
                self._latency_accum[name].extend(latencies)
            for name, costs in executor._cost_accum.items():
                self._cost_accum[name].extend(costs)
            for name, count in executor._win_counts.items():
                self._win_counts[name] += count

        return result

    # ----- statistics -------------------------------------------------------

    def get_strategy_stats(self) -> dict[str, Any]:
        """Return per-strategy performance statistics.

        Returns a dict keyed by strategy name with:
            - win_rate: fraction of total executions won
            - avg_latency_ms: mean latency
            - avg_cost: mean cost
            - total_wins: absolute win count
        """
        stats: dict[str, Any] = {
            "total_executions": self._total_executions,
            "strategies": {},
        }

        for strategy in self.strategies:
            name = strategy.name
            wins = self._win_counts.get(name, 0)
            latencies = self._latency_accum.get(name, [])
            costs = self._cost_accum.get(name, [])

            stats["strategies"][name] = {
                "total_wins": wins,
                "win_rate": (wins / self._total_executions) if self._total_executions > 0 else 0.0,
                "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
                "avg_cost": (sum(costs) / len(costs)) if costs else 0.0,
                "executions": len(latencies),
            }

        return stats

    # ----- internal helpers -------------------------------------------------

    def _find_strategy(self, name: str) -> ExecutionStrategy | None:
        """Look up a strategy by name."""
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None

    @staticmethod
    def _select_best(results: list[StrategyResult]) -> StrategyResult:
        """Select the best result from a list based on confidence and completeness.

        Prefers complete results with the highest confidence.  If no
        complete results exist, returns the one with the highest confidence
        anyway (even if incomplete).
        """
        if not results:
            return StrategyResult(
                strategy_name="none",
                error="No strategies produced results",
            )

        # Sort: complete first, then by confidence descending
        ranked = sorted(
            results,
            key=lambda r: (r.is_complete, r.confidence),
            reverse=True,
        )
        return ranked[0]


# ===========================================================================
# Strategy Selector (learned routing)
# ===========================================================================


class StrategySelector:
    """Learned routing that tracks historical performance per (intent, complexity).

    Over time, builds a profile of which strategies work best for different
    kinds of queries, enabling the executor to skip strategies that are
    unlikely to succeed for a given intent type.

    The selector records outcomes (latency and quality) and uses them to
    rank strategies for future queries with matching characteristics.

    Attributes:
        _performance_log: Maps (intent_type, complexity_bucket) to a list
                          of per-strategy performance records.
    """

    def __init__(self, all_strategies: list[ExecutionStrategy] | None = None) -> None:
        self._all_strategies = all_strategies or []

        # (intent_type, complexity_bucket) -> strategy_name -> [records]
        self._performance_log: dict[tuple[str, str], dict[str, list[dict[str, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        self._lock = asyncio.Lock()

    # ----- public API -------------------------------------------------------

    def select_strategies(
        self,
        query: str,
        intent: str = "DISCOVERY",
    ) -> list[ExecutionStrategy]:
        """Return strategies ordered by expected performance for this query.

        Args:
            query: User query (used to determine complexity).
            intent: Classified intent type (e.g. DISCOVERY, COMPARISON).

        Returns:
            Strategies ordered from most-promising to least-promising.
        """
        complexity = self._estimate_complexity(query)
        key = (intent.upper(), complexity)

        # Look up historical performance for this (intent, complexity)
        perf_map = self._performance_log.get(key)

        if not perf_map:
            # No history: return strategies in default order (cheapest first)
            return sorted(self._all_strategies, key=lambda s: s.estimated_cost)

        # Compute a score for each strategy: quality / (latency_norm + 1)
        strategy_scores: dict[str, float] = {}
        for name, records in perf_map.items():
            if not records:
                continue
            avg_quality = sum(r["quality"] for r in records) / len(records)
            avg_latency = sum(r["latency"] for r in records) / len(records)
            # Normalise latency to seconds for scoring
            latency_norm = avg_latency / 1000.0
            strategy_scores[name] = avg_quality / (latency_norm + 0.1)

        # Sort strategies by score descending, falling back to default order
        def sort_key(s: ExecutionStrategy) -> float:
            return strategy_scores.get(s.name, 0.0)

        ranked = sorted(self._all_strategies, key=sort_key, reverse=True)
        return ranked

    async def record_outcome(
        self,
        strategy_name: str,
        intent: str,
        latency: float,
        quality: float,
        query: str = "",
    ) -> None:
        """Record an observed strategy outcome for learning.

        Args:
            strategy_name: Which strategy produced the result.
            intent: Intent classification of the query.
            latency: Observed latency in milliseconds.
            quality: Observed quality/confidence score [0, 1].
            query: Original query (used for complexity estimation).
        """
        complexity = self._estimate_complexity(query)
        key = (intent.upper(), complexity)

        async with self._lock:
            records = self._performance_log[key][strategy_name]
            records.append({"latency": latency, "quality": quality})

            # Keep a sliding window of the most recent records per strategy
            max_history = 100
            if len(records) > max_history:
                self._performance_log[key][strategy_name] = records[-max_history:]

        logger.debug(
            "StrategySelector: recorded %s for (%s, %s) latency=%.1fms quality=%.2f",
            strategy_name,
            intent,
            complexity,
            latency,
            quality,
        )

    def get_performance_summary(self) -> dict[str, Any]:
        """Return a summary of all recorded performance data."""
        summary: dict[str, Any] = {}
        for (intent, complexity), strategies in self._performance_log.items():
            key_label = f"{intent}:{complexity}"
            summary[key_label] = {}
            for name, records in strategies.items():
                if records:
                    summary[key_label][name] = {
                        "count": len(records),
                        "avg_latency_ms": sum(r["latency"] for r in records) / len(records),
                        "avg_quality": sum(r["quality"] for r in records) / len(records),
                    }
        return summary

    # ----- internal helpers -------------------------------------------------

    @staticmethod
    def _estimate_complexity(query: str) -> str:
        """Bucket a query into a complexity category.

        Simple heuristic based on token count and structural signals.
        """
        words = query.split()
        word_count = len(words)

        # Check for comparison/multi-entity signals
        comparison_keywords = {"vs", "versus", "compare", "compared", "between", "differ"}
        has_comparison = bool(set(w.lower() for w in words) & comparison_keywords)

        if has_comparison or word_count > 20:
            return "complex"
        elif word_count > 8:
            return "moderate"
        else:
            return "simple"


# ===========================================================================
# Singleton accessor
# ===========================================================================

_speculative_executor: SpeculativeExecutor | None = None
_executor_lock = asyncio.Lock()


async def get_speculative_executor() -> SpeculativeExecutor:
    """Return (or create) the application-wide :class:`SpeculativeExecutor`.

    Lazily instantiates the executor with all four default strategies and
    a standard arbiter configuration.  Thread-safe via asyncio lock.
    """
    global _speculative_executor

    if _speculative_executor is not None:
        return _speculative_executor

    async with _executor_lock:
        # Double-check inside the lock
        if _speculative_executor is not None:
            return _speculative_executor

        strategies: list[ExecutionStrategy] = [
            DenseSearchStrategy(),
            HybridSearchStrategy(),
            GraphRAGStrategy(),
            FullPipelineStrategy(),
        ]

        arbiter = SpeculationArbiter(
            min_confidence=0.6,
            min_results=1,
        )

        _speculative_executor = SpeculativeExecutor(
            strategies=strategies,
            arbiter=arbiter,
            timeout_seconds=settings.timeout_total_request,
        )

        logger.info(
            "SpeculativeExecutor initialised: strategies=%s, timeout=%.1fs, arbiter=%r",
            [s.name for s in strategies],
            settings.timeout_total_request,
            arbiter,
        )

        return _speculative_executor
