"""Cost-Aware Mixture of Experts Query Router for ARBOR Enterprise.

Routes queries to the optimal execution path based on complexity, cost,
and quality requirements.  Each user tier (free / pro / enterprise) has
a cost and latency budget; the router selects the execution plan that
maximises expected quality within those constraints.

Architecture overview::

    Query
      │
      ▼
    QueryAnalyzer  ──►  complexity + features
      │
      ▼
    MoERouter
      ├── filter plans by RoutingPolicy (tier budget)
      ├── score candidates via BanditOptimizer (Thompson Sampling)
      └── return best ExecutionPlan

Pre-configured plans (one per complexity level):

    TRIVIAL   →  cache-only lookup, no LLM
    SIMPLE    →  Qdrant dense search + Cohere rerank, no LLM synthesis
    MODERATE  →  Hybrid search + cross-encoder + Gemini synthesis
    COMPLEX   →  Full 4-stage reranking + full agent swarm + Gemini
    EXPERT    →  Everything + graph reasoning + LLM reranking

Usage::

    router = get_moe_router()
    plan   = router.route("cozy rooftop bar near Roma Norte", user_tier="pro")
    # → ExecutionPlan(plan_id="moderate", llm_provider="google", ...)
"""

import logging
import math
import random
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class QueryComplexity(str, Enum):
    """Discrete complexity levels for incoming queries.

    Each level maps to one or more candidate execution plans with
    progressively richer (and more expensive) retrieval + generation
    pipelines.
    """

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionPlan:
    """A fully-specified execution plan for a single query.

    Attributes:
        plan_id: Unique identifier for this plan template (e.g. ``"moderate"``).
        llm_provider: LLM provider key (``"google"``, ``"openai"``, ``"none"``).
        llm_model: Specific model identifier (e.g. ``"gemini-3-pro-preview"``).
        retrieval_stages: Ordered list of retrieval stage names to execute.
        agent_config: Arbitrary agent-swarm configuration dict.
        estimated_cost: Estimated cost in USD for this plan.
        estimated_latency_ms: Estimated wall-clock latency in milliseconds.
        expected_quality: Expected quality score in [0, 1].
        reason: Human-readable explanation of why this plan was selected.
    """

    plan_id: str
    llm_provider: str
    llm_model: str
    retrieval_stages: list[str] = field(default_factory=list)
    agent_config: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    expected_quality: float = 0.0
    reason: str = ""


@dataclass
class RoutingPolicy:
    """Per-tier constraints that govern plan selection.

    Attributes:
        tier: User tier key (``"free"``, ``"pro"``, ``"enterprise"``).
        max_cost_per_query: Hard ceiling on cost (USD) per query.
        max_latency_ms: Hard ceiling on acceptable latency (ms).
        min_quality: Minimum acceptable quality score [0, 1].
        allow_llm_reranking: Whether Stage-4 LLM reranking is permitted.
        allow_full_pipeline: Whether the full 4-stage + agent pipeline
            is permitted.
    """

    tier: str
    max_cost_per_query: float
    max_latency_ms: float
    min_quality: float
    allow_llm_reranking: bool
    allow_full_pipeline: bool


# ═══════════════════════════════════════════════════════════════════════════
# Query Analyzer
# ═══════════════════════════════════════════════════════════════════════════


# Keywords used by the analyzer to detect domain specificity and filters.
_QUESTION_WORDS: set[str] = {
    "what", "where", "which", "who", "when", "how", "why", "is", "are",
    "can", "could", "would", "should", "do", "does", "did", "recommend",
    "suggest", "find", "show", "list", "compare", "rank",
}

_FILTER_KEYWORDS: set[str] = {
    "near", "in", "under", "below", "above", "between", "within",
    "cheaper", "cheapest", "best", "top", "most", "least", "only",
    "open", "closed", "vegan", "vegetarian", "gluten-free", "organic",
    "budget", "luxury", "mid-range", "price", "rating", "stars",
    "outdoor", "indoor", "rooftop", "pet-friendly", "wifi",
}

_FASHION_KEYWORDS: set[str] = {
    "fashion", "style", "designer", "brand", "collection", "runway",
    "couture", "streetwear", "vintage", "sustainable", "luxury",
    "accessories", "shoes", "sneakers", "denim", "leather", "silk",
    "boutique", "atelier", "wardrobe", "trend", "aesthetic", "outfit",
    "wear", "clothing", "apparel", "textile", "fabric",
}

_FOOD_KEYWORDS: set[str] = {
    "restaurant", "cafe", "bar", "bistro", "taco", "sushi", "pizza",
    "brunch", "cocktail", "wine", "coffee", "bakery", "pastry",
    "mezcal", "tequila", "omakase", "ramen", "dim sum", "tapas",
    "michelin", "chef", "kitchen", "menu", "dish", "cuisine",
    "food", "dining", "eatery", "gastropub", "cantina",
}

_VAGUE_INDICATORS: set[str] = {
    "something", "anything", "somewhere", "whatever", "stuff",
    "things", "good", "nice", "cool", "fun", "interesting",
    "vibes", "vibe", "chill", "awesome", "great", "place",
}


class QueryAnalyzer:
    """Analyses an incoming query to determine complexity and features.

    Uses lightweight heuristics (no LLM calls) to produce a feature dict
    consumed by :class:`MoERouter` for plan selection.

    Usage::

        analyzer = QueryAnalyzer()
        features = analyzer.analyze("best rooftop bar in Roma Norte with mezcal")
        # → {
        #     "complexity": QueryComplexity.MODERATE,
        #     "entity_count": 1,
        #     "filter_count": 2,
        #     "ambiguity_score": 0.15,
        #     "domain_specificity": 0.6,
        # }
    """

    def analyze(self, query: str) -> dict[str, Any]:
        """Analyse *query* and return a complexity feature dict.

        Args:
            query: Raw user query string.

        Returns:
            Dict with keys ``complexity``, ``entity_count``,
            ``filter_count``, ``ambiguity_score``, ``domain_specificity``.
        """
        query_lower = query.lower().strip()
        tokens = re.findall(r"[a-z0-9]+", query_lower)

        entity_count = self._count_entities(query)
        filter_count = self._count_filters(tokens)
        ambiguity = self._measure_ambiguity(query_lower, tokens)
        domain_spec = self._measure_domain_specificity(tokens)
        question_count = self._count_question_words(tokens)

        # ── Complexity scoring ────────────────────────────────────────
        # A simple weighted score that maps to a discrete complexity level.
        score = 0.0

        # Length contribution (longer queries tend to be more complex)
        word_count = len(tokens)
        if word_count <= 3:
            score += 0.0
        elif word_count <= 7:
            score += 1.0
        elif word_count <= 15:
            score += 2.0
        else:
            score += 3.0

        # Entity and filter contributions
        score += min(entity_count, 3) * 0.8
        score += min(filter_count, 5) * 0.5
        score += question_count * 0.3

        # Ambiguity pushes complexity *up* (harder to resolve)
        score += ambiguity * 2.0

        # Domain specificity pushes complexity *down* slightly (easier to
        # route to the right index) but only for moderate specificity
        if domain_spec > 0.4:
            score -= 0.3

        complexity = self._score_to_complexity(score)

        features = {
            "complexity": complexity,
            "entity_count": entity_count,
            "filter_count": filter_count,
            "ambiguity_score": round(ambiguity, 3),
            "domain_specificity": round(domain_spec, 3),
        }

        logger.debug(
            "QueryAnalyzer: query=%r → complexity=%s (score=%.2f) features=%s",
            query[:60],
            complexity.value,
            score,
            features,
        )
        return features

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _count_entities(self, query: str) -> int:
        """Detect likely entity mentions (brand / venue names).

        Heuristic: count capitalised multi-word sequences that look like
        proper nouns.  Also counts quoted strings as entity references.

        Args:
            query: Raw query string.

        Returns:
            Estimated entity count (>= 0).
        """
        count = 0

        # Quoted strings are almost always entity references
        quoted = re.findall(r'"([^"]+)"', query)
        count += len(quoted)

        # Capitalised words that are not at the start of a sentence and
        # are not common question/filter words
        words = query.split()
        for i, word in enumerate(words):
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if not clean:
                continue
            if clean[0].isupper() and i > 0:
                lower = clean.lower()
                if (
                    lower not in _QUESTION_WORDS
                    and lower not in _FILTER_KEYWORDS
                    and len(clean) > 1
                ):
                    count += 1

        return count

    @staticmethod
    def _count_filters(tokens: list[str]) -> int:
        """Count filter keywords present in the tokenised query."""
        return sum(1 for t in tokens if t in _FILTER_KEYWORDS)

    @staticmethod
    def _count_question_words(tokens: list[str]) -> int:
        """Count question / intent words in the tokenised query."""
        return sum(1 for t in tokens if t in _QUESTION_WORDS)

    @staticmethod
    def _measure_ambiguity(query_lower: str, tokens: list[str]) -> float:
        """Score how ambiguous / vague a query is.

        Vague queries contain words like *something*, *stuff*, *vibes* and
        tend to be short with few concrete nouns.

        Args:
            query_lower: Lowercased query string.
            tokens: Pre-tokenised query.

        Returns:
            Ambiguity score in [0.0, 1.0].
        """
        if not tokens:
            return 1.0

        vague_count = sum(1 for t in tokens if t in _VAGUE_INDICATORS)
        vague_ratio = vague_count / len(tokens)

        # Short queries with vague words are highly ambiguous
        length_penalty = max(0.0, 1.0 - len(tokens) / 10.0)

        ambiguity = 0.5 * vague_ratio + 0.5 * length_penalty
        return min(1.0, ambiguity)

    @staticmethod
    def _measure_domain_specificity(tokens: list[str]) -> float:
        """Score how domain-specific (fashion / food) a query is.

        Queries heavy with ARBOR domain keywords are easier to route
        because they clearly target the fashion or food verticals.

        Args:
            tokens: Pre-tokenised query.

        Returns:
            Domain specificity score in [0.0, 1.0].
        """
        if not tokens:
            return 0.0

        domain_tokens = _FASHION_KEYWORDS | _FOOD_KEYWORDS
        domain_count = sum(1 for t in tokens if t in domain_tokens)
        return min(1.0, domain_count / max(len(tokens), 1))

    @staticmethod
    def _score_to_complexity(score: float) -> QueryComplexity:
        """Map a continuous complexity score to a discrete level.

        Thresholds (tuned empirically):

        ======  =================
        Score   Complexity
        ======  =================
        < 1.0   TRIVIAL
        < 2.5   SIMPLE
        < 4.5   MODERATE
        < 7.0   COMPLEX
        >= 7.0  EXPERT
        ======  =================
        """
        if score < 1.0:
            return QueryComplexity.TRIVIAL
        if score < 2.5:
            return QueryComplexity.SIMPLE
        if score < 4.5:
            return QueryComplexity.MODERATE
        if score < 7.0:
            return QueryComplexity.COMPLEX
        return QueryComplexity.EXPERT


# ═══════════════════════════════════════════════════════════════════════════
# Cost Model
# ═══════════════════════════════════════════════════════════════════════════


class CostModel:
    """Tracks per-operation costs and estimates total plan cost.

    All costs are in USD.  The model maintains a running total so that
    cumulative spending can be monitored over the lifetime of the process.

    Usage::

        cost_model = CostModel()
        plan_cost  = cost_model.estimate_plan_cost(plan)
        report     = cost_model.get_cost_report()
    """

    # Default cost table (USD).  Rates are per-unit as indicated.
    DEFAULT_COSTS: dict[str, float] = {
        "gemini_pro": 0.00125,       # per 1K tokens
        "gpt4o": 0.005,              # per 1K tokens
        "cohere_embed": 0.0001,      # per call
        "cohere_rerank": 0.002,      # per call
        "qdrant_search": 0.0001,     # per call
        "neo4j_query": 0.0002,       # per call
    }

    # Mapping from retrieval stage names to the cost-table operations
    # that they invoke (with estimated call counts per invocation).
    _STAGE_OPERATION_MAP: dict[str, list[tuple[str, int]]] = {
        "cache_lookup": [],
        "qdrant_dense": [("qdrant_search", 1), ("cohere_embed", 1)],
        "qdrant_hybrid": [("qdrant_search", 2), ("cohere_embed", 1)],
        "cohere_rerank": [("cohere_rerank", 1)],
        "cross_encoder": [("cohere_rerank", 1)],
        "gemini_synthesis": [("gemini_pro", 1)],
        "gpt4o_synthesis": [("gpt4o", 1)],
        "llm_reranking": [("gemini_pro", 2)],
        "graph_reasoning": [("neo4j_query", 3), ("gemini_pro", 1)],
        "agent_swarm": [("gemini_pro", 3)],
    }

    def __init__(
        self,
        cost_overrides: dict[str, float] | None = None,
    ) -> None:
        self.costs: dict[str, float] = {**self.DEFAULT_COSTS}
        if cost_overrides:
            self.costs.update(cost_overrides)

        # Running totals
        self._total_cost: float = 0.0
        self._operation_counts: dict[str, int] = defaultdict(int)
        self._operation_costs: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_plan_cost(self, plan: ExecutionPlan) -> float:
        """Estimate the total USD cost for executing *plan*.

        Walks the plan's ``retrieval_stages`` list and sums up the
        per-operation costs using the stage-to-operation mapping.

        Args:
            plan: The execution plan to cost.

        Returns:
            Estimated cost in USD.
        """
        total = 0.0
        for stage in plan.retrieval_stages:
            ops = self._STAGE_OPERATION_MAP.get(stage, [])
            for op_name, call_count in ops:
                unit_cost = self.costs.get(op_name, 0.0)
                total += unit_cost * call_count
        return total

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_operation(self, operation: str, count: int = 1) -> None:
        """Record that *operation* was invoked *count* times.

        Args:
            operation: Operation key from the cost table.
            count: Number of invocations.
        """
        unit_cost = self.costs.get(operation, 0.0)
        cost = unit_cost * count
        self._total_cost += cost
        self._operation_counts[operation] += count
        self._operation_costs[operation] += cost

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_cost_report(self) -> dict[str, Any]:
        """Return a cost report with totals and per-operation breakdown.

        Returns:
            Dict with ``total_cost``, ``operations`` (per-op breakdown),
            and ``cost_table`` (the unit rates in effect).
        """
        operations: list[dict[str, Any]] = []
        for op_name in sorted(self._operation_counts.keys()):
            operations.append({
                "operation": op_name,
                "count": self._operation_counts[op_name],
                "total_cost": round(self._operation_costs[op_name], 6),
                "unit_cost": self.costs.get(op_name, 0.0),
            })

        return {
            "total_cost": round(self._total_cost, 6),
            "operations": operations,
            "cost_table": dict(self.costs),
        }

    def reset(self) -> None:
        """Reset running totals (useful for testing)."""
        self._total_cost = 0.0
        self._operation_counts.clear()
        self._operation_costs.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Bandit Optimizer (Thompson Sampling)
# ═══════════════════════════════════════════════════════════════════════════


class BanditOptimizer:
    """Multi-armed bandit using Thompson Sampling (Beta-Bernoulli).

    Each *arm* corresponds to an execution plan.  The reward signal is
    the quality/cost ratio observed after plan execution.  Over time the
    bandit learns which plan delivers the best quality-per-dollar and
    explores less-tried plans with appropriate probability.

    Usage::

        bandit = BanditOptimizer(arms=["trivial", "simple", "moderate"])
        chosen = bandit.select()           # sample from Beta posteriors
        bandit.update(chosen, reward=0.85) # quality/cost ratio

    Attributes:
        alpha: Per-arm success count (Beta parameter).
        beta_param: Per-arm failure count (Beta parameter).
    """

    def __init__(self, arms: list[str]) -> None:
        """Initialise with uniform Beta(1, 1) priors for each arm.

        Args:
            arms: List of arm identifiers (plan IDs).
        """
        self.arms: list[str] = list(arms)
        # Beta(1, 1) = Uniform(0, 1) — non-informative prior
        self.alpha: dict[str, float] = {arm: 1.0 for arm in arms}
        self.beta_param: dict[str, float] = {arm: 1.0 for arm in arms}

    def select(self, context: dict[str, Any] | None = None) -> str:
        """Select an arm by sampling from each arm's Beta posterior.

        The arm with the highest sampled value wins.  This naturally
        balances exploration (uncertain arms have wide distributions)
        and exploitation (arms with high observed rewards have high
        means).

        Args:
            context: Optional context dict (reserved for future
                contextual-bandit extensions).

        Returns:
            The arm ID (plan_id) with the highest Thompson sample.
        """
        best_arm = self.arms[0]
        best_sample = -1.0

        for arm in self.arms:
            a = self.alpha.get(arm, 1.0)
            b = self.beta_param.get(arm, 1.0)

            # Sample from Beta(a, b)
            sample = random.betavariate(a, b)

            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        logger.debug(
            "BanditOptimizer: selected arm=%s (sample=%.4f)",
            best_arm,
            best_sample,
        )
        return best_arm

    def update(self, arm_id: str, reward: float) -> None:
        """Update the posterior for *arm_id* given the observed *reward*.

        The reward is clamped to [0, 1] and treated as a Bernoulli-like
        signal: values closer to 1 increase alpha (successes), values
        closer to 0 increase beta (failures).

        Args:
            arm_id: The arm that was pulled.
            reward: Observed reward signal in [0, 1].
        """
        if arm_id not in self.alpha:
            logger.warning("BanditOptimizer: unknown arm %r, skipping update", arm_id)
            return

        reward = max(0.0, min(1.0, reward))

        # Fractional update: reward contributes proportionally to both
        # alpha and beta so that partially-good outcomes are captured.
        self.alpha[arm_id] += reward
        self.beta_param[arm_id] += (1.0 - reward)

        logger.debug(
            "BanditOptimizer: updated arm=%s reward=%.3f → alpha=%.2f beta=%.2f",
            arm_id,
            reward,
            self.alpha[arm_id],
            self.beta_param[arm_id],
        )

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return per-arm posterior statistics.

        Returns:
            Dict mapping arm_id to ``{alpha, beta, mean, variance}``.
        """
        stats: dict[str, dict[str, float]] = {}
        for arm in self.arms:
            a = self.alpha[arm]
            b = self.beta_param[arm]
            mean = a / (a + b)
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            stats[arm] = {
                "alpha": round(a, 3),
                "beta": round(b, 3),
                "mean": round(mean, 4),
                "variance": round(variance, 6),
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# MoE Router
# ═══════════════════════════════════════════════════════════════════════════


# Pre-configured tier policies.
_TIER_POLICIES: dict[str, RoutingPolicy] = {
    "free": RoutingPolicy(
        tier="free",
        max_cost_per_query=0.005,
        max_latency_ms=3000.0,
        min_quality=0.3,
        allow_llm_reranking=False,
        allow_full_pipeline=False,
    ),
    "pro": RoutingPolicy(
        tier="pro",
        max_cost_per_query=0.02,
        max_latency_ms=8000.0,
        min_quality=0.5,
        allow_llm_reranking=True,
        allow_full_pipeline=False,
    ),
    "enterprise": RoutingPolicy(
        tier="enterprise",
        max_cost_per_query=0.10,
        max_latency_ms=15000.0,
        min_quality=0.7,
        allow_llm_reranking=True,
        allow_full_pipeline=True,
    ),
}


class MoERouter:
    """Mixture-of-Experts query router.

    Combines :class:`QueryAnalyzer`, :class:`CostModel`, and
    :class:`BanditOptimizer` to select the best execution plan for each
    incoming query, subject to the user's tier constraints.

    Usage::

        router = get_moe_router()

        plan = router.route(
            query="best mezcal bar in Condesa",
            user_tier="pro",
        )
        print(plan.plan_id, plan.estimated_cost)

        # After execution, record outcomes so the bandit can learn:
        router.record_outcome(
            plan_id=plan.plan_id,
            actual_cost=0.0035,
            actual_latency=1200.0,
            quality_score=0.82,
        )
    """

    def __init__(self) -> None:
        self._analyzer = QueryAnalyzer()
        self._cost_model = CostModel()

        # ── Register execution plans per complexity ───────────────────
        self._plans: dict[str, ExecutionPlan] = {}
        self._complexity_plans: dict[QueryComplexity, list[str]] = defaultdict(list)

        self._register_default_plans()

        # ── Bandit optimiser (one arm per plan) ───────────────────────
        plan_ids = list(self._plans.keys())
        self._bandit = BanditOptimizer(arms=plan_ids)

        # ── Outcome history for stats and optimisation ────────────────
        self._outcomes: dict[str, list[dict[str, float]]] = defaultdict(list)

        logger.info(
            "MoERouter initialised: %d plans, bandit arms=%s",
            len(self._plans),
            plan_ids,
        )

    # ------------------------------------------------------------------
    # Plan registration
    # ------------------------------------------------------------------

    def _register_default_plans(self) -> None:
        """Register the built-in execution plans."""

        # TRIVIAL: cache-only lookup, no LLM
        self._register_plan(
            plan=ExecutionPlan(
                plan_id="trivial",
                llm_provider="none",
                llm_model="none",
                retrieval_stages=["cache_lookup"],
                agent_config={},
                estimated_cost=0.0,
                estimated_latency_ms=50.0,
                expected_quality=0.4,
                reason="Cache-only lookup for trivial queries",
            ),
            complexities=[QueryComplexity.TRIVIAL],
        )

        # SIMPLE: Qdrant dense search + Cohere rerank, no LLM synthesis
        self._register_plan(
            plan=ExecutionPlan(
                plan_id="simple",
                llm_provider="none",
                llm_model="none",
                retrieval_stages=["qdrant_dense", "cohere_rerank"],
                agent_config={},
                estimated_cost=0.0,  # will be computed
                estimated_latency_ms=500.0,
                expected_quality=0.6,
                reason="Dense retrieval + reranking without LLM synthesis",
            ),
            complexities=[QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE],
        )

        # MODERATE: Hybrid search + cross-encoder + Gemini synthesis
        self._register_plan(
            plan=ExecutionPlan(
                plan_id="moderate",
                llm_provider="google",
                llm_model=settings.google_model,
                retrieval_stages=[
                    "qdrant_hybrid",
                    "cross_encoder",
                    "gemini_synthesis",
                ],
                agent_config={"max_agent_steps": 3},
                estimated_cost=0.0,
                estimated_latency_ms=2000.0,
                expected_quality=0.75,
                reason="Hybrid search with cross-encoder and Gemini synthesis",
            ),
            complexities=[
                QueryComplexity.SIMPLE,
                QueryComplexity.MODERATE,
            ],
        )

        # COMPLEX: Full 4-stage reranking + full agent swarm + Gemini
        self._register_plan(
            plan=ExecutionPlan(
                plan_id="complex",
                llm_provider="google",
                llm_model=settings.google_model,
                retrieval_stages=[
                    "qdrant_hybrid",
                    "cross_encoder",
                    "llm_reranking",
                    "agent_swarm",
                    "gemini_synthesis",
                ],
                agent_config={
                    "max_agent_steps": 8,
                    "enable_tool_use": True,
                },
                estimated_cost=0.0,
                estimated_latency_ms=5000.0,
                expected_quality=0.88,
                reason="Full 4-stage reranking with agent swarm and Gemini",
            ),
            complexities=[
                QueryComplexity.MODERATE,
                QueryComplexity.COMPLEX,
            ],
        )

        # EXPERT: Everything + graph reasoning + LLM reranking
        self._register_plan(
            plan=ExecutionPlan(
                plan_id="expert",
                llm_provider="google",
                llm_model=settings.google_model,
                retrieval_stages=[
                    "qdrant_hybrid",
                    "cross_encoder",
                    "llm_reranking",
                    "graph_reasoning",
                    "agent_swarm",
                    "gemini_synthesis",
                ],
                agent_config={
                    "max_agent_steps": 15,
                    "enable_tool_use": True,
                    "enable_reflection": True,
                    "enable_graph_reasoning": True,
                },
                estimated_cost=0.0,
                estimated_latency_ms=10000.0,
                expected_quality=0.95,
                reason="Full pipeline with graph reasoning and LLM reranking",
            ),
            complexities=[
                QueryComplexity.COMPLEX,
                QueryComplexity.EXPERT,
            ],
        )

        # Back-fill estimated costs from the cost model
        for plan in self._plans.values():
            plan.estimated_cost = self._cost_model.estimate_plan_cost(plan)

    def _register_plan(
        self,
        plan: ExecutionPlan,
        complexities: list[QueryComplexity],
    ) -> None:
        """Register *plan* as a candidate for the given complexity levels.

        Args:
            plan: The execution plan template.
            complexities: Complexity levels at which this plan is eligible.
        """
        self._plans[plan.plan_id] = plan
        for complexity in complexities:
            if plan.plan_id not in self._complexity_plans[complexity]:
                self._complexity_plans[complexity].append(plan.plan_id)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        user_tier: str = "free",
        intent: str | None = None,
    ) -> ExecutionPlan:
        """Route *query* to the optimal execution plan.

        Steps:

        1. Analyse query complexity via :class:`QueryAnalyzer`.
        2. Gather candidate plans for that complexity level.
        3. Filter candidates by the user's tier constraints.
        4. Use the bandit to pick the best plan (exploration/exploitation).
        5. Return a copy of the selected plan with a unique invocation ID.

        Args:
            query: Raw user query string.
            user_tier: One of ``"free"``, ``"pro"``, ``"enterprise"``.
            intent: Optional pre-classified intent (unused today, reserved).

        Returns:
            An :class:`ExecutionPlan` ready for execution.
        """
        t0 = time.perf_counter()

        # 1. Analyse
        features = self._analyzer.analyze(query)
        complexity: QueryComplexity = features["complexity"]

        # 2. Candidate plans
        candidate_ids = list(self._complexity_plans.get(complexity, []))
        if not candidate_ids:
            # Fallback: use the "simple" plan
            candidate_ids = ["simple"]

        # 3. Filter by tier policy
        policy = _TIER_POLICIES.get(user_tier, _TIER_POLICIES["free"])
        eligible_ids = self._filter_by_policy(candidate_ids, policy)

        if not eligible_ids:
            # If nothing is eligible after filtering, fall back to the
            # cheapest plan that satisfies cost constraints.
            eligible_ids = self._cheapest_fallback(policy)

        # 4. Bandit selection among eligible plans
        selected_id = self._bandit_select(eligible_ids)

        # 5. Build final plan
        template = self._plans[selected_id]
        plan = ExecutionPlan(
            plan_id=template.plan_id,
            llm_provider=template.llm_provider,
            llm_model=template.llm_model,
            retrieval_stages=list(template.retrieval_stages),
            agent_config=dict(template.agent_config),
            estimated_cost=template.estimated_cost,
            estimated_latency_ms=template.estimated_latency_ms,
            expected_quality=template.expected_quality,
            reason=(
                f"Complexity={complexity.value}, tier={user_tier}, "
                f"candidates={len(candidate_ids)}, eligible={len(eligible_ids)}. "
                f"{template.reason}"
            ),
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "MoERouter: query=%r → plan=%s (cost=$%.4f, latency=%.0fms, "
            "quality=%.2f) routing_time=%.1fms",
            query[:60],
            plan.plan_id,
            plan.estimated_cost,
            plan.estimated_latency_ms,
            plan.expected_quality,
            elapsed_ms,
        )
        return plan

    def _filter_by_policy(
        self,
        candidate_ids: list[str],
        policy: RoutingPolicy,
    ) -> list[str]:
        """Filter candidate plans to those that satisfy *policy* constraints.

        Args:
            candidate_ids: Plan IDs to evaluate.
            policy: The tier routing policy.

        Returns:
            Filtered list of plan IDs.
        """
        eligible: list[str] = []

        for pid in candidate_ids:
            plan = self._plans.get(pid)
            if plan is None:
                continue

            # Cost ceiling
            if plan.estimated_cost > policy.max_cost_per_query:
                continue

            # Latency ceiling
            if plan.estimated_latency_ms > policy.max_latency_ms:
                continue

            # Quality floor
            if plan.expected_quality < policy.min_quality:
                continue

            # LLM reranking gate
            if not policy.allow_llm_reranking and "llm_reranking" in plan.retrieval_stages:
                continue

            # Full pipeline gate
            if not policy.allow_full_pipeline and "agent_swarm" in plan.retrieval_stages:
                continue

            eligible.append(pid)

        return eligible

    def _cheapest_fallback(self, policy: RoutingPolicy) -> list[str]:
        """Return the single cheapest plan that fits within *policy* cost.

        This is the last-resort fallback when no complexity-matched plan
        passes the tier filter.
        """
        sorted_plans = sorted(
            self._plans.values(),
            key=lambda p: p.estimated_cost,
        )
        for plan in sorted_plans:
            if plan.estimated_cost <= policy.max_cost_per_query:
                return [plan.plan_id]

        # Absolute fallback: the cheapest plan regardless of policy
        if sorted_plans:
            return [sorted_plans[0].plan_id]
        return ["trivial"]

    def _bandit_select(self, eligible_ids: list[str]) -> str:
        """Use the bandit to select among *eligible_ids*.

        If only one plan is eligible the bandit is bypassed.
        """
        if len(eligible_ids) == 1:
            return eligible_ids[0]

        # Create a temporary bandit view restricted to eligible arms
        best_arm = eligible_ids[0]
        best_sample = -1.0

        for arm in eligible_ids:
            a = self._bandit.alpha.get(arm, 1.0)
            b = self._bandit.beta_param.get(arm, 1.0)
            sample = random.betavariate(a, b)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    # ------------------------------------------------------------------
    # Outcome recording & learning
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        plan_id: str,
        actual_cost: float,
        actual_latency: float,
        quality_score: float,
    ) -> None:
        """Record the observed outcome of executing *plan_id*.

        Updates the bandit optimizer and stores the outcome for later
        analysis via :meth:`get_routing_stats`.

        Args:
            plan_id: The plan that was executed.
            actual_cost: Actual cost incurred (USD).
            actual_latency: Actual wall-clock latency (ms).
            quality_score: Observed quality score [0, 1].
        """
        self._outcomes[plan_id].append({
            "actual_cost": actual_cost,
            "actual_latency": actual_latency,
            "quality_score": quality_score,
            "timestamp": time.time(),
        })

        # Reward = quality / cost ratio, normalised to [0, 1].
        # Higher quality and lower cost yield higher reward.
        if actual_cost > 0:
            raw_reward = quality_score / (actual_cost * 1000)
        else:
            raw_reward = quality_score

        reward = max(0.0, min(1.0, raw_reward))
        self._bandit.update(plan_id, reward)

        logger.debug(
            "MoERouter: recorded outcome plan=%s cost=$%.4f latency=%.0fms "
            "quality=%.3f reward=%.3f",
            plan_id,
            actual_cost,
            actual_latency,
            quality_score,
            reward,
        )

    def get_routing_stats(self) -> dict[str, Any]:
        """Return per-plan usage, average cost, and average quality.

        Returns:
            Dict keyed by plan_id with usage counts, averages, and
            bandit posterior stats.
        """
        stats: dict[str, Any] = {}

        for plan_id in self._plans:
            outcomes = self._outcomes.get(plan_id, [])
            n = len(outcomes)

            if n > 0:
                avg_cost = sum(o["actual_cost"] for o in outcomes) / n
                avg_latency = sum(o["actual_latency"] for o in outcomes) / n
                avg_quality = sum(o["quality_score"] for o in outcomes) / n
            else:
                avg_cost = 0.0
                avg_latency = 0.0
                avg_quality = 0.0

            stats[plan_id] = {
                "usage_count": n,
                "avg_cost": round(avg_cost, 6),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_quality": round(avg_quality, 4),
            }

        stats["bandit"] = self._bandit.get_stats()
        stats["cost_report"] = self._cost_model.get_cost_report()

        return stats

    def optimize_routing(self) -> dict[str, Any]:
        """Adjust plan selection parameters based on historical outcomes.

        Implements a simple bandit-based optimisation step:

        1. For each plan with recorded outcomes, re-compute the
           quality/cost reward.
        2. Feed the rewards back into the bandit so that its Beta
           posteriors sharpen.
        3. Prune outcomes older than the most recent 200 per plan to
           keep memory bounded.

        Returns:
            Summary dict with per-plan adjustments.
        """
        adjustments: dict[str, dict[str, Any]] = {}

        for plan_id, outcomes in self._outcomes.items():
            if not outcomes:
                continue

            # Only consider recent outcomes (last 200)
            recent = outcomes[-200:]
            self._outcomes[plan_id] = recent

            # Re-compute average reward and feed back into bandit
            total_reward = 0.0
            for outcome in recent[-20:]:  # use last 20 for recency
                cost = outcome["actual_cost"]
                quality = outcome["quality_score"]
                if cost > 0:
                    raw = quality / (cost * 1000)
                else:
                    raw = quality
                reward = max(0.0, min(1.0, raw))
                total_reward += reward

            avg_reward = total_reward / min(20, len(recent))

            # Adjust bandit prior slightly towards recent performance
            # (soft reset to avoid being stuck on stale data)
            decay = 0.95
            self._bandit.alpha[plan_id] = max(
                1.0,
                self._bandit.alpha[plan_id] * decay + avg_reward,
            )
            self._bandit.beta_param[plan_id] = max(
                1.0,
                self._bandit.beta_param[plan_id] * decay + (1.0 - avg_reward),
            )

            adjustments[plan_id] = {
                "recent_outcomes": len(recent),
                "avg_reward": round(avg_reward, 4),
                "new_alpha": round(self._bandit.alpha[plan_id], 3),
                "new_beta": round(self._bandit.beta_param[plan_id], 3),
            }

        logger.info(
            "MoERouter: optimise_routing complete, adjusted %d plans",
            len(adjustments),
        )
        return {"adjustments": adjustments}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def analyzer(self) -> QueryAnalyzer:
        """Expose the query analyzer for external use."""
        return self._analyzer

    @property
    def cost_model(self) -> CostModel:
        """Expose the cost model for external use."""
        return self._cost_model


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_moe_router: Optional[MoERouter] = None


def get_moe_router() -> MoERouter:
    """Return the singleton :class:`MoERouter` instance.

    Thread-safe for CPython (GIL protects the check-and-set).
    """
    global _moe_router
    if _moe_router is None:
        _moe_router = MoERouter()
    return _moe_router
