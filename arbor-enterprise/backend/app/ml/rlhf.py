"""Reinforcement Learning from Human Feedback (RLHF) for ARBOR Enterprise discovery.

Learns a reward model from pairwise human preferences over search/discovery
responses, then uses it to select and optimise candidate responses.  A
lightweight Constitutional AI layer enforces hard and soft constraints so that
optimised outputs remain safe and grounded.

Usage::

    optimizer = get_rlhf_optimizer()
    optimizer.collect_preference(
        query="best cafes in CDMX",
        response_a=candidate_1,
        response_b=candidate_2,
        preference="a",
        annotator="ann_01",
    )
    optimizer.train_reward_model()
    best = optimizer.optimize_response(query, candidates)
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A single pairwise preference annotation.

    An annotator is shown two responses to the same query and indicates which
    one they prefer (or declares a tie).

    Attributes:
        query: The original discovery/search query.
        response_a: First candidate response dict.
        response_b: Second candidate response dict.
        preference: Which response is preferred - ``"a"``, ``"b"``, or ``"tie"``.
        confidence: Annotator self-reported confidence in [0.0, 1.0].
        annotator_id: Unique identifier for the human annotator.
        created_at: UTC timestamp of annotation creation.
    """

    query: str
    response_a: dict
    response_b: dict
    preference: str  # "a", "b", "tie"
    confidence: float = 1.0
    annotator_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.preference not in ("a", "b", "tie"):
            raise ValueError(f"preference must be 'a', 'b', or 'tie', got '{self.preference}'")
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class RewardSignal:
    """Scalar reward with a per-dimension breakdown.

    Attributes:
        query: The query that generated the response.
        response: The response dict being scored.
        reward: Aggregate reward in [0.0, 1.0].
        components: Per-dimension reward breakdown (e.g. ``{"relevance": 0.8}``).
        source: Origin of the signal - ``"explicit"`` (human), ``"implicit"``
            (behavioural), or ``"model"`` (reward-model prediction).
    """

    query: str
    response: dict
    reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    source: str = "model"  # "explicit", "implicit", "model"

    def __post_init__(self) -> None:
        self.reward = max(0.0, min(1.0, self.reward))
        if self.source not in ("explicit", "implicit", "model"):
            raise ValueError(
                f"source must be 'explicit', 'implicit', or 'model', got '{self.source}'"
            )


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class RewardModel:
    """Learned reward function trained from pairwise human preferences.

    Maintains a lightweight weight vector over five interpretable reward
    dimensions.  Training uses a simple gradient rule: for each preference
    pair, feature differences between the preferred and rejected responses
    are computed and the weight vector is nudged in the direction that would
    increase the preferred response's score.
    """

    # Reward dimensions and their initial weights
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "result_count": 0.15,
        "confidence": 0.25,
        "diversity": 0.20,
        "relevance_keywords": 0.25,
        "personalization": 0.15,
    }

    def __init__(self) -> None:
        self._weights: dict[str, float] = dict(self._DEFAULT_WEIGHTS)
        self._training_iterations: int = 0
        self._learning_rate: float = 0.01
        logger.info("RewardModel initialised with default weights")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, preference_pairs: list[PreferencePair]) -> dict[str, Any]:
        """Train the reward model from a list of pairwise preferences.

        For each pair the method:
        1. Extracts features for both responses.
        2. Computes the element-wise feature difference (preferred - rejected).
        3. Updates weights via ``w += lr * confidence * diff``.

        Ties are skipped (no gradient).

        Args:
            preference_pairs: Annotated preference pairs to learn from.

        Returns:
            A dict summarising the training run (pairs used, iterations, weights).
        """
        pairs_used = 0
        for pair in preference_pairs:
            if pair.preference == "tie":
                continue

            if pair.preference == "a":
                preferred, rejected = pair.response_a, pair.response_b
            else:
                preferred, rejected = pair.response_b, pair.response_a

            feat_preferred = self._extract_features(pair.query, preferred)
            feat_rejected = self._extract_features(pair.query, rejected)

            # Gradient step: increase weight on dimensions where preferred > rejected
            for dim in self._weights:
                diff = feat_preferred.get(dim, 0.0) - feat_rejected.get(dim, 0.0)
                self._weights[dim] += self._learning_rate * pair.confidence * diff
                # Clamp weights to stay positive
                self._weights[dim] = max(0.01, self._weights[dim])

            pairs_used += 1
            self._training_iterations += 1

        # Normalise weights so they sum to 1
        self._normalise_weights()

        logger.info(
            "RewardModel trained: pairs_used=%d total_iterations=%d",
            pairs_used,
            self._training_iterations,
        )

        return {
            "pairs_used": pairs_used,
            "total_iterations": self._training_iterations,
            "weights": dict(self._weights),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_reward(self, query: str, response: dict) -> RewardSignal:
        """Score a single response using the learned reward function.

        Args:
            query: The originating query string.
            response: A candidate response dict.

        Returns:
            A :class:`RewardSignal` with the aggregate reward and per-dimension
            component breakdown.
        """
        features = self._extract_features(query, response)
        components: dict[str, float] = {}
        aggregate = 0.0

        for dim, weight in self._weights.items():
            score = features.get(dim, 0.0)
            components[dim] = round(score, 4)
            aggregate += weight * score

        aggregate = max(0.0, min(1.0, aggregate))

        return RewardSignal(
            query=query,
            response=response,
            reward=round(aggregate, 4),
            components=components,
            source="model",
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, response_a: dict, response_b: dict, query: str) -> str:
        """Determine which response the reward model prefers.

        Args:
            response_a: First candidate response.
            response_b: Second candidate response.
            query: The originating query.

        Returns:
            ``"a"`` if response_a is preferred, ``"b"`` otherwise.
        """
        reward_a = self.predict_reward(query, response_a).reward
        reward_b = self.predict_reward(query, response_b).reward
        return "a" if reward_a >= reward_b else "b"

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """Return a copy of the current weight vector."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, query: str, response: dict) -> dict[str, float]:
        """Compute normalised feature scores for a response.

        All features are scaled to approximately [0, 1].

        Args:
            query: The originating query string.
            response: A candidate response dict.

        Returns:
            Dict mapping dimension name to score.
        """
        features: dict[str, float] = {}

        # 1. result_count - normalised by a soft-cap of 20
        results = response.get("results", response.get("entities", []))
        count = len(results) if isinstance(results, list) else 0
        features["result_count"] = min(1.0, count / 20.0)

        # 2. confidence - average confidence across results, or response-level
        conf = response.get("confidence", 0.0)
        if isinstance(results, list) and results:
            conf_values = [
                r.get("confidence", r.get("score", 0.5)) for r in results if isinstance(r, dict)
            ]
            conf = sum(conf_values) / len(conf_values) if conf_values else conf
        features["confidence"] = max(0.0, min(1.0, float(conf)))

        # 3. diversity - number of unique categories / styles present
        if isinstance(results, list) and results:
            categories = set()
            for r in results:
                if isinstance(r, dict):
                    cat = r.get("category", r.get("type", ""))
                    if cat:
                        categories.add(cat)
            features["diversity"] = min(1.0, len(categories) / 8.0)
        else:
            features["diversity"] = 0.0

        # 4. relevance_keywords - fraction of query tokens found in response text
        query_tokens = set(re.findall(r"\w+", query.lower()))
        response_text = str(response).lower()
        if query_tokens:
            matched = sum(1 for t in query_tokens if t in response_text)
            features["relevance_keywords"] = matched / len(query_tokens)
        else:
            features["relevance_keywords"] = 0.0

        # 5. personalization - presence of personalisation metadata
        has_personalization = (
            "personalization_boost" in response
            or "personalized_score" in response
            or response.get("personalized", False)
        )
        features["personalization"] = 1.0 if has_personalization else 0.0

        return features

    def _normalise_weights(self) -> None:
        """Normalise weights so they sum to 1.0."""
        total = sum(self._weights.values())
        if total > 0:
            for dim in self._weights:
                self._weights[dim] /= total


# ---------------------------------------------------------------------------
# Constitutional AI
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionalConstraint:
    """A single constitutional constraint that responses must satisfy.

    Attributes:
        name: Short unique identifier for the constraint.
        description: Human-readable explanation.
        check_fn: Callable ``(query, response) -> bool``.  Returns ``True``
            when the constraint is **satisfied** (no violation).
        severity: ``"hard"`` constraints block the response; ``"soft"``
            constraints produce warnings.
    """

    name: str
    description: str
    check_fn: Callable[[str, dict], bool]
    severity: str = "hard"  # "hard" or "soft"

    def __post_init__(self) -> None:
        if self.severity not in ("hard", "soft"):
            raise ValueError(f"severity must be 'hard' or 'soft', got '{self.severity}'")


class ConstitutionalAI:
    """Enforce a set of constitutional constraints on discovery responses.

    Pre-registers a default set of safety and quality constraints.  Additional
    constraints can be added at runtime.
    """

    # Words / phrases that should never appear in recommendations
    _HARMFUL_PATTERNS: list[str] = [
        "illegal",
        "dangerous",
        "harmful",
        "exploit",
        "hack",
        "malware",
        "phishing",
        "scam",
    ]

    def __init__(self) -> None:
        self._constraints: list[ConstitutionalConstraint] = []
        self._register_defaults()
        logger.info(
            "ConstitutionalAI initialised with %d default constraints",
            len(self._constraints),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: ConstitutionalConstraint) -> None:
        """Register a new constitutional constraint.

        Args:
            constraint: The constraint to add.
        """
        self._constraints.append(constraint)
        logger.debug("Constraint added: name=%s severity=%s", constraint.name, constraint.severity)

    def check(self, query: str, response: dict) -> list[dict[str, Any]]:
        """Run all registered constraints against a response.

        Args:
            query: The originating query string.
            response: The candidate response dict.

        Returns:
            A list of violation dicts.  Each dict contains ``name``,
            ``description``, ``severity``, and ``passed`` keys.  An empty
            list means no violations were found.
        """
        violations: list[dict[str, Any]] = []
        for constraint in self._constraints:
            try:
                passed = constraint.check_fn(query, response)
            except Exception as exc:
                logger.warning("Constraint %s raised an exception: %s", constraint.name, exc)
                passed = False

            if not passed:
                violations.append(
                    {
                        "name": constraint.name,
                        "description": constraint.description,
                        "severity": constraint.severity,
                        "passed": False,
                    }
                )

        if violations:
            logger.warning(
                "Constitutional check found %d violation(s) for query='%s'",
                len(violations),
                query[:80],
            )

        return violations

    # ------------------------------------------------------------------
    # Default constraints
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register the built-in constitutional constraints."""

        # 1. No harmful content
        self.add_constraint(
            ConstitutionalConstraint(
                name="no_harmful_content",
                description=(
                    "Response must not contain harmful, dangerous, or " "exploitative language."
                ),
                check_fn=self._check_no_harmful_content,
                severity="hard",
            )
        )

        # 2. Factual grounding
        self.add_constraint(
            ConstitutionalConstraint(
                name="factual_grounding",
                description=(
                    "Response must reference entities that actually exist "
                    "in the knowledge base (entities should have IDs)."
                ),
                check_fn=self._check_factual_grounding,
                severity="hard",
            )
        )

        # 3. Relevance
        self.add_constraint(
            ConstitutionalConstraint(
                name="relevance",
                description=("Response must be relevant to the original query."),
                check_fn=self._check_relevance,
                severity="soft",
            )
        )

        # 4. Diversity
        self.add_constraint(
            ConstitutionalConstraint(
                name="diversity",
                description=(
                    "Recommendations should include a variety of options, "
                    "not just repeated categories."
                ),
                check_fn=self._check_diversity,
                severity="soft",
            )
        )

    # ------------------------------------------------------------------
    # Constraint implementations
    # ------------------------------------------------------------------

    def _check_no_harmful_content(self, query: str, response: dict) -> bool:
        """Return True if the response contains no harmful language."""
        response_text = str(response).lower()
        for pattern in self._HARMFUL_PATTERNS:
            if pattern in response_text:
                logger.debug("Harmful content detected: pattern='%s'", pattern)
                return False
        return True

    @staticmethod
    def _check_factual_grounding(query: str, response: dict) -> bool:
        """Return True if all referenced entities have an ID field."""
        results = response.get("results", response.get("entities", []))
        if not isinstance(results, list) or not results:
            # No results to check - pass vacuously
            return True
        for result in results:
            if isinstance(result, dict):
                has_id = result.get("id") or result.get("entity_id") or result.get("_id")
                if not has_id:
                    return False
        return True

    @staticmethod
    def _check_relevance(query: str, response: dict) -> bool:
        """Return True if at least one query token appears in the response."""
        if not query.strip():
            return True
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return True
        response_text = str(response).lower()
        matched = sum(1 for t in query_tokens if t in response_text)
        # At least 25 % of query tokens should appear
        return (matched / len(query_tokens)) >= 0.25

    @staticmethod
    def _check_diversity(query: str, response: dict) -> bool:
        """Return True if results span more than one category (when >=3 results)."""
        results = response.get("results", response.get("entities", []))
        if not isinstance(results, list) or len(results) < 3:
            return True
        categories = set()
        for result in results:
            if isinstance(result, dict):
                cat = result.get("category", result.get("type", ""))
                if cat:
                    categories.add(cat)
        # With 3+ results we expect at least 2 distinct categories
        return len(categories) >= 2


# ---------------------------------------------------------------------------
# RLHF optimizer
# ---------------------------------------------------------------------------


class RLHFOptimizer:
    """End-to-end RLHF pipeline for optimising discovery responses.

    Combines a :class:`RewardModel` for scoring candidates with a
    :class:`ConstitutionalAI` layer for safety enforcement.

    Usage::

        optimizer = get_rlhf_optimizer()
        optimizer.collect_preference(query, resp_a, resp_b, "a", "ann_01")
        optimizer.train_reward_model()
        best = optimizer.optimize_response(query, candidates)
    """

    def __init__(self) -> None:
        self._reward_model = RewardModel()
        self._constitutional = ConstitutionalAI()
        self._preference_pairs: list[PreferencePair] = []
        self._training_runs: int = 0
        logger.info("RLHFOptimizer initialised")

    # ------------------------------------------------------------------
    # Preference collection
    # ------------------------------------------------------------------

    def collect_preference(
        self,
        query: str,
        response_a: dict,
        response_b: dict,
        preference: str,
        annotator: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Record a new human preference annotation.

        Args:
            query: The originating query.
            response_a: First candidate response.
            response_b: Second candidate response.
            preference: ``"a"``, ``"b"``, or ``"tie"``.
            annotator: Identifier for the annotator.
            confidence: Annotator confidence in [0.0, 1.0].
        """
        pair = PreferencePair(
            query=query,
            response_a=response_a,
            response_b=response_b,
            preference=preference,
            confidence=confidence,
            annotator_id=annotator,
        )
        self._preference_pairs.append(pair)

        logger.debug(
            "Preference collected: query='%s' preference=%s annotator=%s " "total_pairs=%d",
            query[:60],
            preference,
            annotator,
            len(self._preference_pairs),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_reward_model(self, min_pairs: int = 10) -> dict[str, Any]:
        """Train the reward model from collected preference pairs.

        Args:
            min_pairs: Minimum number of pairs required before training.

        Returns:
            A dict with training statistics, or an error message if
            insufficient data is available.
        """
        if len(self._preference_pairs) < min_pairs:
            msg = (
                f"Insufficient preference pairs: have {len(self._preference_pairs)}, "
                f"need {min_pairs}"
            )
            logger.warning(msg)
            return {"error": msg, "pairs_available": len(self._preference_pairs)}

        result = self._reward_model.train(self._preference_pairs)
        self._training_runs += 1
        result["training_run"] = self._training_runs

        logger.info(
            "Reward model training run %d complete: %s",
            self._training_runs,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_response(self, query: str, response: dict) -> RewardSignal:
        """Score a single response using the learned reward model.

        Args:
            query: The originating query.
            response: A candidate response dict.

        Returns:
            A :class:`RewardSignal` with aggregate and per-dimension scores.
        """
        return self._reward_model.predict_reward(query, response)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_response(
        self,
        query: str,
        candidate_responses: list[dict],
    ) -> dict[str, Any]:
        """Select the best candidate response.

        Scores all candidates with the reward model, filters out those that
        violate hard constitutional constraints, and returns the highest-scoring
        compliant candidate.

        Args:
            query: The originating query.
            candidate_responses: List of candidate response dicts.

        Returns:
            A dict containing the ``best_response``, its ``reward_signal``,
            any ``constitutional_violations``, and the number of
            ``candidates_evaluated``.
        """
        if not candidate_responses:
            return {
                "best_response": None,
                "reward_signal": None,
                "constitutional_violations": [],
                "candidates_evaluated": 0,
            }

        scored_candidates: list[dict[str, Any]] = []

        for idx, candidate in enumerate(candidate_responses):
            signal = self._reward_model.predict_reward(query, candidate)
            violations = self._constitutional.check(query, candidate)
            hard_violations = [v for v in violations if v["severity"] == "hard"]
            soft_violations = [v for v in violations if v["severity"] == "soft"]

            scored_candidates.append(
                {
                    "index": idx,
                    "response": candidate,
                    "reward": signal.reward,
                    "signal": signal,
                    "hard_violations": hard_violations,
                    "soft_violations": soft_violations,
                    "compliant": len(hard_violations) == 0,
                }
            )

        # Prefer compliant candidates; fall back to best overall if none comply
        compliant = [c for c in scored_candidates if c["compliant"]]
        pool = compliant if compliant else scored_candidates
        best = max(pool, key=lambda c: c["reward"])

        all_violations = best["hard_violations"] + best["soft_violations"]

        logger.info(
            "Optimised response: query='%s' candidates=%d compliant=%d "
            "best_reward=%.4f violations=%d",
            query[:60],
            len(candidate_responses),
            len(compliant),
            best["reward"],
            len(all_violations),
        )

        return {
            "best_response": best["response"],
            "reward_signal": best["signal"],
            "constitutional_violations": all_violations,
            "candidates_evaluated": len(candidate_responses),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_training_stats(self) -> dict[str, Any]:
        """Return a summary of the RLHF pipeline state.

        Returns:
            Dict with preference pair count, training runs, reward model
            weights, and annotator breakdown.
        """
        annotators: dict[str, int] = {}
        preference_distribution: dict[str, int] = {"a": 0, "b": 0, "tie": 0}

        for pair in self._preference_pairs:
            annotators[pair.annotator_id] = annotators.get(pair.annotator_id, 0) + 1
            preference_distribution[pair.preference] = (
                preference_distribution.get(pair.preference, 0) + 1
            )

        return {
            "total_preference_pairs": len(self._preference_pairs),
            "training_runs": self._training_runs,
            "reward_model_weights": self._reward_model.get_weights(),
            "annotator_counts": annotators,
            "preference_distribution": preference_distribution,
            "constitutional_constraints": len(self._constitutional._constraints),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_optimizer: RLHFOptimizer | None = None


def get_rlhf_optimizer() -> RLHFOptimizer:
    """Return the singleton :class:`RLHFOptimizer` instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = RLHFOptimizer()
    return _optimizer
