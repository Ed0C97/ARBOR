"""Streaming Inference with Early Exit for ARBOR Enterprise.

Implements a multi-stage inference pipeline that can terminate early when
a confidence threshold is reached, saving compute and reducing latency for
straightforward queries.

The pipeline is modelled as a sequence of :class:`InferenceStage` objects,
each with an associated cost, latency, and quality contribution.  After
every stage the :class:`ConfidenceEstimator` evaluates the partial results;
if confidence is above the configured threshold the pipeline exits early
and returns whatever has been computed so far.

Pre-configured stages for the ARBOR discovery platform::

    cache_check       →  cost 0.000, ~5 ms     (instant if cached)
    dense_retrieval   →  cost 0.001, ~50 ms     (Qdrant vector search)
    reranking         →  cost 0.003, ~200 ms    (Cohere cross-encoder)
    agent_synthesis   →  cost 0.050, ~2000 ms   (Gemini agent swarm)

Usage::

    engine = get_streaming_engine()

    # Stream partial results
    async for partial in engine.execute("rooftop bars in Milan", context={}):
        print(partial["stage"], partial["confidence"])

    # Or get the first result that meets the confidence bar
    result = await engine.execute_with_early_exit("cozy cafes", context={})
    print(result["exit_stage"], result["total_latency_ms"])
"""

import asyncio
import logging
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferenceStage:
    """A single stage in the streaming inference pipeline.

    Attributes:
        name: Human-readable identifier for the stage.
        cost: Estimated cost in USD per execution of this stage.
        avg_latency_ms: Average wall-clock time in milliseconds.
        quality_contribution: How much this stage improves result quality
            on a [0, 1] scale (used by the confidence estimator).
    """

    name: str
    cost: float
    avg_latency_ms: float
    quality_contribution: float


@dataclass
class EarlyExitDecision:
    """The result of a confidence check after a pipeline stage.

    Attributes:
        should_exit: Whether the pipeline should stop here.
        confidence: Estimated confidence in the partial results so far.
        stage_name: Name of the stage that was just completed.
        reason: Human-readable explanation for the decision.
    """

    should_exit: bool
    confidence: float
    stage_name: str
    reason: str


# ---------------------------------------------------------------------------
# Pre-configured stages
# ---------------------------------------------------------------------------

DEFAULT_STAGES: list[InferenceStage] = [
    InferenceStage(
        name="cache_check",
        cost=0.0,
        avg_latency_ms=5.0,
        quality_contribution=0.1,
    ),
    InferenceStage(
        name="dense_retrieval",
        cost=0.001,
        avg_latency_ms=50.0,
        quality_contribution=0.35,
    ),
    InferenceStage(
        name="reranking",
        cost=0.003,
        avg_latency_ms=200.0,
        quality_contribution=0.30,
    ),
    InferenceStage(
        name="agent_synthesis",
        cost=0.05,
        avg_latency_ms=2000.0,
        quality_contribution=0.25,
    ),
]


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


class ConfidenceEstimator:
    """Estimate confidence in partial results after each inference stage.

    The estimator uses stage-specific heuristics:

    - **cache_check** -- confidence is 1.0 on a cache hit, 0.0 on miss.
    - **dense_retrieval** -- confidence based on top retrieval score and the
      number of results above a relevance floor.
    - **reranking** -- confidence based on score spread (gap between rank 1
      and rank 2) and absolute top score.
    - **agent_synthesis** -- confidence based on response length and the
      number of entity mentions in the synthesised text.
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Args:
            thresholds: Per-stage confidence thresholds.  If a stage's
                estimated confidence exceeds its threshold, the pipeline
                may exit early.  Defaults are tuned for the ARBOR platform.
        """
        self._thresholds = thresholds or {
            "cache_check": 0.95,
            "dense_retrieval": 0.80,
            "reranking": 0.75,
            "agent_synthesis": 0.60,
        }
        logger.info(
            "ConfidenceEstimator initialised with thresholds: %s", self._thresholds
        )

    def estimate(
        self,
        partial_results: dict[str, Any],
        stage: InferenceStage,
    ) -> float:
        """Estimate confidence in the partial results after *stage*.

        Args:
            partial_results: Accumulated results so far.  Keys depend on
                the stage:

                - ``cache_hit`` (bool) -- set by ``cache_check``.
                - ``retrieval_scores`` (list[float]) -- set by ``dense_retrieval``.
                - ``rerank_scores`` (list[float]) -- set by ``reranking``.
                - ``synthesis_text`` (str) -- set by ``agent_synthesis``.
                - ``entity_mentions`` (int) -- set by ``agent_synthesis``.

            stage: The stage that was just completed.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if stage.name == "cache_check":
            return self._estimate_cache(partial_results)
        elif stage.name == "dense_retrieval":
            return self._estimate_retrieval(partial_results)
        elif stage.name == "reranking":
            return self._estimate_reranking(partial_results)
        elif stage.name == "agent_synthesis":
            return self._estimate_synthesis(partial_results)
        else:
            # Unknown stage -- fall back to quality-contribution heuristic
            cumulative = partial_results.get("_cumulative_quality", 0.0)
            return min(cumulative, 1.0)

    def get_threshold(self, stage_name: str) -> float:
        """Return the confidence threshold for a given stage.

        Args:
            stage_name: Name of the inference stage.

        Returns:
            The threshold value, defaulting to 0.8 if not configured.
        """
        return self._thresholds.get(stage_name, 0.8)

    # -- Stage-specific estimators ------------------------------------------

    @staticmethod
    def _estimate_cache(partial_results: dict[str, Any]) -> float:
        """Cache hit → full confidence, miss → zero."""
        if partial_results.get("cache_hit", False):
            return 1.0
        return 0.0

    @staticmethod
    def _estimate_retrieval(partial_results: dict[str, Any]) -> float:
        """Confidence from dense retrieval scores.

        High confidence when the top score is strong and multiple results
        clear the relevance floor.
        """
        scores = partial_results.get("retrieval_scores", [])
        if not scores:
            return 0.1

        top_score = max(scores)
        relevance_floor = 0.5
        strong_results = sum(1 for s in scores if s >= relevance_floor)
        count_factor = min(strong_results / 5.0, 1.0)  # saturates at 5+

        confidence = 0.6 * top_score + 0.4 * count_factor
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _estimate_reranking(partial_results: dict[str, Any]) -> float:
        """Confidence from reranked scores.

        High confidence when rank-1 is clearly separated from rank-2 (large
        score spread) and the absolute top score is high.
        """
        scores = partial_results.get("rerank_scores", [])
        if not scores:
            return 0.1

        sorted_scores = sorted(scores, reverse=True)
        top = sorted_scores[0]

        if len(sorted_scores) >= 2:
            spread = top - sorted_scores[1]
        else:
            spread = top  # only one result

        spread_factor = min(spread / 0.3, 1.0)  # large gap → confident
        confidence = 0.5 * top + 0.5 * spread_factor
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _estimate_synthesis(partial_results: dict[str, Any]) -> float:
        """Confidence from the agent-synthesised response.

        Longer, entity-rich responses indicate the agent found enough
        information to give a quality answer.
        """
        text = partial_results.get("synthesis_text", "")
        entity_mentions = partial_results.get("entity_mentions", 0)

        length_factor = min(len(text) / 500.0, 1.0)  # saturates at 500 chars
        entity_factor = min(entity_mentions / 3.0, 1.0)  # saturates at 3+

        confidence = 0.4 * length_factor + 0.6 * entity_factor
        return min(max(confidence, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Streaming inference engine
# ---------------------------------------------------------------------------


class StreamingInferenceEngine:
    """Multi-stage inference pipeline with streaming partial results and
    early exit capability.

    The engine iterates through a sequence of inference stages, yielding
    partial results after each.  A :class:`ConfidenceEstimator` is
    consulted after every stage; if confidence exceeds the threshold the
    pipeline exits early, saving cost and latency.
    """

    def __init__(
        self,
        stages: Optional[list[InferenceStage]] = None,
        confidence_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            stages: Ordered list of inference stages.  Defaults to the
                ARBOR platform stages (cache, retrieval, reranking, synthesis).
            confidence_threshold: Global confidence threshold.  If the
                estimated confidence after a stage exceeds this, the pipeline
                exits early.  Per-stage thresholds in the
                :class:`ConfidenceEstimator` can override this.
        """
        self._stages = stages or list(DEFAULT_STAGES)
        self._confidence_threshold = confidence_threshold
        self._estimator = ConfidenceEstimator()

        # Execution statistics
        self._stats: dict[str, Any] = {
            "total_executions": 0,
            "exit_stage_counts": defaultdict(int),
            "total_latency_ms": 0.0,
            "total_cost": 0.0,
            "stages_executed_total": 0,
        }

        logger.info(
            "StreamingInferenceEngine initialised with %d stages, "
            "confidence_threshold=%.2f",
            len(self._stages),
            self._confidence_threshold,
        )

    # -- Public API: streaming execution ------------------------------------

    async def execute(
        self,
        query: str,
        context: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute the inference pipeline, yielding partial results.

        After each stage, yields a dict containing the stage name, the
        partial results accumulated so far, the confidence estimate, and
        the early-exit decision.

        Args:
            query: The user's search query.
            context: Additional context (user profile, session, etc.).

        Yields:
            Dicts with keys ``stage``, ``partial_results``, ``confidence``,
            ``exit_decision``, ``latency_ms``, ``cost``.
        """
        execution_id = uuid.uuid4().hex[:10]
        partial_results: dict[str, Any] = {
            "query": query,
            "context": context,
            "_cumulative_quality": 0.0,
            "results": [],
        }

        total_latency = 0.0
        total_cost = 0.0

        logger.info(
            "[%s] Starting streaming execution for query: '%s'",
            execution_id,
            query[:80],
        )

        for stage in self._stages:
            stage_start = time.monotonic()

            # Simulate stage execution
            stage_results = await self._execute_stage(stage, query, context, partial_results)
            partial_results.update(stage_results)
            partial_results["_cumulative_quality"] += stage.quality_contribution

            stage_latency = (time.monotonic() - stage_start) * 1000
            total_latency += stage_latency
            total_cost += stage.cost

            # Estimate confidence
            confidence = self._estimator.estimate(partial_results, stage)

            # Decide whether to exit
            stage_threshold = min(
                self._confidence_threshold,
                self._estimator.get_threshold(stage.name),
            )
            should_exit = confidence >= stage_threshold

            if should_exit:
                reason = (
                    f"Confidence {confidence:.3f} >= threshold {stage_threshold:.3f} "
                    f"after stage '{stage.name}'"
                )
            else:
                reason = (
                    f"Confidence {confidence:.3f} < threshold {stage_threshold:.3f} "
                    f"— continuing to next stage"
                )

            exit_decision = EarlyExitDecision(
                should_exit=should_exit,
                confidence=confidence,
                stage_name=stage.name,
                reason=reason,
            )

            yield {
                "execution_id": execution_id,
                "stage": stage.name,
                "partial_results": partial_results,
                "confidence": round(confidence, 4),
                "exit_decision": exit_decision,
                "latency_ms": round(stage_latency, 2),
                "cumulative_latency_ms": round(total_latency, 2),
                "cost": stage.cost,
                "cumulative_cost": round(total_cost, 6),
            }

            if should_exit:
                logger.info(
                    "[%s] Early exit at stage '%s' (confidence=%.3f, "
                    "latency=%.1fms, cost=$%.4f)",
                    execution_id,
                    stage.name,
                    confidence,
                    total_latency,
                    total_cost,
                )
                self._record_execution(stage.name, total_latency, total_cost)
                return

        # Reached the end of all stages without early exit
        logger.info(
            "[%s] Completed all stages (latency=%.1fms, cost=$%.4f)",
            execution_id,
            total_latency,
            total_cost,
        )
        self._record_execution(self._stages[-1].name, total_latency, total_cost)

    # -- Public API: early-exit convenience ---------------------------------

    async def execute_with_early_exit(
        self,
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the pipeline and return the first result meeting confidence.

        Unlike :meth:`execute`, this does not stream.  It returns a single
        result dict once the confidence threshold is met or all stages have
        been exhausted.

        Args:
            query: The user's search query.
            context: Additional context.

        Returns:
            Dict containing ``results``, ``exit_stage``, ``stages_executed``,
            ``total_latency_ms``, ``total_cost``, ``confidence``.
        """
        stages_executed: list[str] = []
        final_result: dict[str, Any] = {}

        async for partial in self.execute(query, context):
            stages_executed.append(partial["stage"])
            final_result = partial

            if partial["exit_decision"].should_exit:
                break

        return {
            "query": query,
            "results": final_result.get("partial_results", {}).get("results", []),
            "exit_stage": final_result.get("stage", "unknown"),
            "stages_executed": stages_executed,
            "stages_count": len(stages_executed),
            "total_latency_ms": final_result.get("cumulative_latency_ms", 0.0),
            "total_cost": final_result.get("cumulative_cost", 0.0),
            "confidence": final_result.get("confidence", 0.0),
            "execution_id": final_result.get("execution_id", ""),
        }

    # -- Statistics ---------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate execution statistics.

        Returns:
            Dict containing:
            - ``total_executions``: total pipeline runs.
            - ``avg_exit_stage``: average numeric index of the exit stage.
            - ``exit_stage_distribution``: how often each stage was the exit.
            - ``avg_latency_ms``: mean total latency.
            - ``avg_cost``: mean cost per execution.
            - ``compute_saved_pct``: estimated percentage of compute saved
              via early exit compared to always running all stages.
        """
        total = self._stats["total_executions"]
        if total == 0:
            return {
                "total_executions": 0,
                "avg_exit_stage": 0,
                "exit_stage_distribution": {},
                "avg_latency_ms": 0.0,
                "avg_cost": 0.0,
                "compute_saved_pct": 0.0,
            }

        # Compute average exit stage index
        stage_index = {s.name: i for i, s in enumerate(self._stages)}
        weighted_index_sum = sum(
            stage_index.get(name, len(self._stages) - 1) * count
            for name, count in self._stats["exit_stage_counts"].items()
        )
        avg_exit_index = weighted_index_sum / total

        # Max possible cost and latency (all stages)
        max_cost = sum(s.cost for s in self._stages)
        max_latency = sum(s.avg_latency_ms for s in self._stages)

        avg_latency = self._stats["total_latency_ms"] / total
        avg_cost = self._stats["total_cost"] / total

        # Compute saved as % of maximum
        compute_saved = (
            (1.0 - avg_cost / max_cost) * 100 if max_cost > 0 else 0.0
        )

        return {
            "total_executions": total,
            "avg_exit_stage": round(avg_exit_index, 2),
            "exit_stage_distribution": dict(self._stats["exit_stage_counts"]),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_cost": round(avg_cost, 6),
            "compute_saved_pct": round(compute_saved, 1),
            "total_stages": len(self._stages),
            "max_possible_cost": max_cost,
            "max_possible_latency_ms": max_latency,
        }

    # -- Internal: real stage execution -------------------------------------

    async def _execute_stage(
        self,
        stage: InferenceStage,
        query: str,
        context: dict[str, Any],
        partial_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single inference stage using real services.

        Each stage calls the corresponding production service:
        - cache_check → SemanticCache
        - dense_retrieval → LLMGateway (embedding) + HybridSearch (Qdrant)
        - reranking → LLMGateway.rerank_results (Cohere cross-encoder)
        - agent_synthesis → LLMGateway.complete (Gemini agent synthesis)

        Args:
            stage: The stage to execute.
            query: The original query.
            context: Additional context.
            partial_results: Results accumulated from previous stages.

        Returns:
            Dict of new keys to merge into ``partial_results``.
        """
        if stage.name == "cache_check":
            return await self._stage_cache_check(query, context)
        elif stage.name == "dense_retrieval":
            return await self._stage_dense_retrieval(query)
        elif stage.name == "reranking":
            return await self._stage_reranking(query, partial_results)
        elif stage.name == "agent_synthesis":
            return await self._stage_synthesis(query, partial_results)
        else:
            return {}

    @staticmethod
    async def _stage_cache_check(
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check the semantic cache for a previous response."""
        try:
            from app.llm.cache import SemanticCache
            cache = SemanticCache()
            result = await cache.check_cache(query)

            if result.hit and result.response:
                logger.debug("Streaming cache HIT for: %s", query[:50])
                return {
                    "cache_hit": True,
                    "results": [
                        {
                            "entity_id": "cached",
                            "name": "Cached Response",
                            "score": result.score or 0.95,
                            "source": "cache",
                            "response": result.response,
                        }
                    ],
                    # Store embedding for reuse in dense_retrieval
                    "_query_embedding": result.query_embedding,
                }
            else:
                return {
                    "cache_hit": False,
                    "_query_embedding": result.query_embedding,
                }
        except Exception as exc:
            logger.warning("Cache check failed: %s", exc)
            return {"cache_hit": False}

    @staticmethod
    async def _stage_dense_retrieval(query: str) -> dict[str, Any]:
        """Execute real dense vector retrieval via Qdrant."""
        try:
            from app.llm.gateway import get_llm_gateway
            from app.db.qdrant.hybrid_search import HybridSearch

            gateway = get_llm_gateway()
            hybrid = HybridSearch()

            # Generate embedding
            query_embedding = await gateway.get_embedding(query)

            # Execute RRF hybrid search
            results = await hybrid.search_rrf(
                query_vector=query_embedding,
                query_text=query,
                limit=20,  # Retrieve more for reranking stage
            )

            scores = [r.get("score", 0.0) for r in results]

            # Annotate results
            for r in results:
                r["source"] = "dense_retrieval"
                r["entity_id"] = r.get("id", "")

            return {
                "retrieval_scores": sorted(scores, reverse=True),
                "results": results,
                "retrieval_count": len(results),
                "_query_embedding": query_embedding,
            }
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)
            return {
                "retrieval_scores": [],
                "results": [],
                "retrieval_count": 0,
            }

    @staticmethod
    async def _stage_reranking(
        query: str,
        partial_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute real cross-encoder reranking via Cohere."""
        existing_results = partial_results.get("results", [])
        if not existing_results:
            return {"rerank_scores": [], "results": []}

        try:
            from app.llm.gateway import get_llm_gateway
            gateway = get_llm_gateway()

            # Use the gateway's rerank_results which preserves full result objects
            reranked = await gateway.rerank_results(
                query=query,
                results=existing_results,
                top_n=min(10, len(existing_results)),
                text_field="name",
            )

            scores = [r.get("rerank_score", r.get("score", 0.0)) for r in reranked]

            return {
                "rerank_scores": sorted(scores, reverse=True),
                "results": reranked,
            }
        except Exception as exc:
            logger.warning("Reranking failed, preserving original order: %s", exc)
            # Fallback: keep original results
            scores = [r.get("score", 0.0) for r in existing_results]
            return {
                "rerank_scores": sorted(scores, reverse=True),
                "results": existing_results,
            }

    @staticmethod
    async def _stage_synthesis(
        query: str,
        partial_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute real LLM agent synthesis via Gemini."""
        results = partial_results.get("results", [])
        top_names = [r.get("name", "Entity") for r in results[:5]]
        entity_mentions = len(top_names)

        try:
            from app.llm.gateway import get_llm_gateway
            gateway = get_llm_gateway()

            # Build context from results
            context_lines = []
            for i, r in enumerate(results[:5], 1):
                context_lines.append(
                    f"{i}. {r.get('name', 'Unknown')} "
                    f"(Score: {r.get('score', 0):.2f}, "
                    f"Category: {r.get('category', 'N/A')}, "
                    f"City: {r.get('city', 'N/A')})"
                )
            context_text = "\n".join(context_lines)

            synthesis_text = await gateway.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are ARBOR, a curated discovery assistant for fashion "
                            "brands and venues. Provide a concise, insightful response "
                            "based on the search results."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: '{query}'\n\nTop results:\n{context_text}\n\n"
                            f"Provide a 2-3 sentence curated recommendation."
                        ),
                    },
                ],
                task_type="synthesis",
                temperature=0.7,
            )

            return {
                "synthesis_text": synthesis_text,
                "entity_mentions": entity_mentions,
                "results": results,
            }
        except Exception as exc:
            logger.warning("LLM synthesis failed, using fallback: %s", exc)
            # Fallback to template-based synthesis
            if top_names:
                synthesis_text = (
                    f"Based on your search for '{query}', I recommend "
                    f"{', '.join(top_names[:3])}. "
                    f"These match your criteria with strong relevance scores."
                )
            else:
                synthesis_text = (
                    f"No results found for '{query}'. "
                    f"Try broadening your search criteria."
                )

            return {
                "synthesis_text": synthesis_text,
                "entity_mentions": entity_mentions,
                "results": results,
            }

    # -- Internal bookkeeping -----------------------------------------------

    def _record_execution(
        self,
        exit_stage: str,
        total_latency_ms: float,
        total_cost: float,
    ) -> None:
        """Record statistics for a completed pipeline execution."""
        self._stats["total_executions"] += 1
        self._stats["exit_stage_counts"][exit_stage] += 1
        self._stats["total_latency_ms"] += total_latency_ms
        self._stats["total_cost"] += total_cost

        stage_index = {s.name: i for i, s in enumerate(self._stages)}
        stages_run = stage_index.get(exit_stage, len(self._stages) - 1) + 1
        self._stats["stages_executed_total"] += stages_run


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[StreamingInferenceEngine] = None


def get_streaming_engine() -> StreamingInferenceEngine:
    """Return the singleton :class:`StreamingInferenceEngine` instance.

    The engine is created on first call with default stages and reused
    thereafter.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StreamingInferenceEngine()
    return _engine_instance
