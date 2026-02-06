"""Real-Time Knowledge Graph Reasoning Engine for ARBOR Enterprise.

Goes beyond simple graph traversal to perform inference, link prediction,
and rule induction over the ARBOR knowledge graph.  Supports five
reasoning modes:

- **Link prediction** -- TransE-inspired transitive closure and
  pattern-based candidate generation (shared vibe_dna, co-location).
- **Rule induction** -- Discovers Horn-clause-style rules from observed
  entity-pair patterns and applies them to predict new relationships.
- **Temporal reasoning** -- Detects trends in entity timelines and
  extrapolates future attribute trajectories.
- **Counterfactual reasoning** -- Estimates the impact of hypothetically
  adding or removing edges in the knowledge graph.
- **Abductive reasoning** -- Generates plausible explanations for
  observed user-entity interactions from graph context.

Usage::

    engine = get_kg_reasoning_engine()
    results = await engine.reason("entity-42", entity_data)
    inferred = engine.get_inferred_relationships("entity-42")
    rules = engine.get_discovered_rules()
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReasoningType(str, Enum):
    """Supported knowledge-graph reasoning modes."""

    LINK_PREDICTION = "link_prediction"
    RULE_INDUCTION = "rule_induction"
    TEMPORAL = "temporal"
    COUNTERFACTUAL = "counterfactual"
    ABDUCTIVE = "abductive"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferredRelationship:
    """A relationship inferred by one of the reasoning engines.

    Attributes:
        source_id: Identifier of the source entity.
        target_id: Identifier of the target entity.
        rel_type: Predicted relationship label (e.g. ``SIMILAR_TO``,
            ``COMPETES_WITH``).
        confidence: Score in [0, 1] indicating inference certainty.
        reasoning_type: The reasoning mode that produced this inference.
        evidence: List of human-readable evidence strings supporting the
            inference.
        explanation: One-line natural-language summary of the reasoning.
    """

    source_id: str
    target_id: str
    rel_type: str
    confidence: float
    reasoning_type: ReasoningType
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class GraphRule:
    """A Horn-clause-style rule discovered via rule induction.

    An antecedent is a list of graph pattern strings such as
    ``["A -HAS_STYLE-> S", "B -HAS_STYLE-> S"]``.  The consequent is
    the predicted relationship, e.g. ``"A -SIMILAR_TO-> B"``.

    Attributes:
        rule_id: Unique identifier for the rule.
        antecedent: List of pattern strings forming the rule body.
        consequent: Predicted relationship pattern (rule head).
        confidence: Fraction of antecedent matches where the consequent
            also holds.
        support_count: Number of entity pairs that satisfy the full rule.
        discovered_at: Timestamp when the rule was first discovered.
    """

    rule_id: str
    antecedent: list[str]
    consequent: str
    confidence: float
    support_count: int
    discovered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector (must be same length as *a*).

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 when either vector
        has zero magnitude or the vectors differ in length.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        a: First vector.
        b: Second vector (must be same length as *a*).

    Returns:
        Euclidean distance.  Returns ``float('inf')`` when vectors
        differ in length.
    """
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Link Predictor
# ---------------------------------------------------------------------------


class LinkPredictor:
    """Predicts missing links in the knowledge graph.

    Employs three complementary strategies:

    1. **TransE-inspired transitive closure** -- if ``(A, rel, B)`` and
       ``(B, rel, C)`` both exist, predict ``(A, rel_transitive, C)``.
    2. **Vibe-DNA pattern** -- entities with similar ``vibe_dna`` vectors
       sharing at least one style are predicted as ``SIMILAR_TO``.
    3. **Co-location pattern** -- entities in the same city with the
       same category are predicted as ``COMPETES_WITH``.
    """

    # Relationship types eligible for transitive closure
    _TRANSITIVE_REL_TYPES: set[str] = {
        "SIMILAR_TO",
        "SIMILAR_VIBE",
        "INFLUENCED_BY",
        "INSPIRED_BY",
    }

    def predict_links(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        all_entities: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[InferredRelationship]:
        """Predict missing links for *entity_id*.

        Runs all prediction strategies, merges results, deduplicates
        (keeping the highest-confidence prediction for each target), and
        returns the top *top_k* predictions sorted by confidence.

        Args:
            entity_id: The entity to predict links for.
            entity_data: Attribute dict for the source entity.
            all_entities: List of all entity dicts in the graph.
            top_k: Maximum number of predictions to return.

        Returns:
            Up to *top_k* :class:`InferredRelationship` objects sorted
            by confidence descending.
        """
        predictions: list[InferredRelationship] = []

        # Strategy 1: TransE-inspired transitive closure
        predictions.extend(
            self._transitive_closure(entity_id, entity_data, all_entities)
        )

        # Strategy 2: shared vibe_dna + style => SIMILAR_TO
        predictions.extend(
            self._vibe_style_pattern(entity_id, entity_data, all_entities)
        )

        # Strategy 3: same city + same category => COMPETES_WITH
        predictions.extend(
            self._colocation_pattern(entity_id, entity_data, all_entities)
        )

        # Deduplicate: keep highest confidence per (target_id, rel_type)
        best: dict[tuple[str, str], InferredRelationship] = {}
        for pred in predictions:
            key = (pred.target_id, pred.rel_type)
            existing = best.get(key)
            if existing is None or pred.confidence > existing.confidence:
                best[key] = pred

        deduplicated = sorted(
            best.values(), key=lambda r: r.confidence, reverse=True
        )

        result = deduplicated[:top_k]
        logger.info(
            "LinkPredictor: %d raw predictions -> %d deduplicated -> "
            "top %d for entity %s",
            len(predictions),
            len(deduplicated),
            len(result),
            entity_id,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy: transitive closure
    # ------------------------------------------------------------------

    def _transitive_closure(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        all_entities: list[dict[str, Any]],
    ) -> list[InferredRelationship]:
        """TransE-inspired: if (A, r, B) and (B, r, C) then (A, r, C).

        Scans the ``relationships`` field of the source entity and of
        each intermediate entity to discover two-hop paths that close
        transitively.
        """
        predictions: list[InferredRelationship] = []

        source_rels = entity_data.get("relationships", [])
        if not isinstance(source_rels, list):
            return predictions

        # Build a quick lookup: entity_id -> entity_data
        entity_index: dict[str, dict[str, Any]] = {
            str(e.get("id", "")): e for e in all_entities if e.get("id")
        }

        for rel in source_rels:
            if not isinstance(rel, dict):
                continue

            rel_type = rel.get("type", rel.get("rel_type", ""))
            intermediate_id = str(
                rel.get("target_id", rel.get("target", ""))
            )

            if not rel_type or not intermediate_id:
                continue
            if rel_type not in self._TRANSITIVE_REL_TYPES:
                continue

            # Look at intermediate entity's relationships
            intermediate = entity_index.get(intermediate_id)
            if intermediate is None:
                continue

            inter_rels = intermediate.get("relationships", [])
            if not isinstance(inter_rels, list):
                continue

            for inter_rel in inter_rels:
                if not isinstance(inter_rel, dict):
                    continue

                inter_rel_type = inter_rel.get(
                    "type", inter_rel.get("rel_type", "")
                )
                target_id = str(
                    inter_rel.get("target_id", inter_rel.get("target", ""))
                )

                if not target_id or target_id == entity_id:
                    continue
                if inter_rel_type != rel_type:
                    continue

                # Score the transitive candidate
                confidence = self._score_candidate(
                    entity_data,
                    entity_index.get(target_id, {}),
                    rel_type,
                )

                predictions.append(
                    InferredRelationship(
                        source_id=entity_id,
                        target_id=target_id,
                        rel_type=rel_type,
                        confidence=confidence,
                        reasoning_type=ReasoningType.LINK_PREDICTION,
                        evidence=[
                            f"{entity_id} -{rel_type}-> {intermediate_id}",
                            f"{intermediate_id} -{rel_type}-> {target_id}",
                        ],
                        explanation=(
                            f"Transitive: {entity_id} is connected to "
                            f"{target_id} via {intermediate_id} through "
                            f"{rel_type}"
                        ),
                    )
                )

        return predictions

    # ------------------------------------------------------------------
    # Strategy: vibe + style similarity
    # ------------------------------------------------------------------

    def _vibe_style_pattern(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        all_entities: list[dict[str, Any]],
    ) -> list[InferredRelationship]:
        """Entities with similar vibe_dna sharing a style => SIMILAR_TO."""
        predictions: list[InferredRelationship] = []

        source_vibe = entity_data.get("vibe_dna")
        source_styles = entity_data.get("styles", [])

        if not source_vibe or not isinstance(source_vibe, list):
            return predictions
        if not source_styles or not isinstance(source_styles, list):
            return predictions

        source_style_set = {str(s).lower() for s in source_styles}

        for other in all_entities:
            other_id = str(other.get("id", ""))
            if not other_id or other_id == entity_id:
                continue

            other_vibe = other.get("vibe_dna")
            other_styles = other.get("styles", [])

            if not other_vibe or not isinstance(other_vibe, list):
                continue
            if not other_styles or not isinstance(other_styles, list):
                continue

            # Check for shared style
            other_style_set = {str(s).lower() for s in other_styles}
            shared_styles = source_style_set & other_style_set
            if not shared_styles:
                continue

            # Compute vibe similarity
            vibe_sim = _cosine_similarity(source_vibe, other_vibe)
            if vibe_sim < 0.6:
                continue

            # Confidence is a blend of vibe similarity and style overlap
            style_overlap_ratio = len(shared_styles) / max(
                len(source_style_set), len(other_style_set), 1
            )
            confidence = 0.6 * vibe_sim + 0.4 * style_overlap_ratio
            confidence = max(0.0, min(1.0, confidence))

            predictions.append(
                InferredRelationship(
                    source_id=entity_id,
                    target_id=other_id,
                    rel_type="SIMILAR_TO",
                    confidence=round(confidence, 4),
                    reasoning_type=ReasoningType.LINK_PREDICTION,
                    evidence=[
                        f"Vibe DNA cosine similarity: {vibe_sim:.4f}",
                        f"Shared styles: {', '.join(sorted(shared_styles))}",
                    ],
                    explanation=(
                        f"Entities share styles "
                        f"({', '.join(sorted(shared_styles))}) and have "
                        f"similar vibe DNA (cosine={vibe_sim:.3f})"
                    ),
                )
            )

        return predictions

    # ------------------------------------------------------------------
    # Strategy: co-location competitors
    # ------------------------------------------------------------------

    def _colocation_pattern(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        all_entities: list[dict[str, Any]],
    ) -> list[InferredRelationship]:
        """Same city + same category => COMPETES_WITH."""
        predictions: list[InferredRelationship] = []

        source_city = str(entity_data.get("city", "")).strip().lower()
        source_category = str(entity_data.get("category", "")).strip().lower()

        if not source_city or not source_category:
            return predictions

        for other in all_entities:
            other_id = str(other.get("id", ""))
            if not other_id or other_id == entity_id:
                continue

            other_city = str(other.get("city", "")).strip().lower()
            other_category = str(other.get("category", "")).strip().lower()

            if other_city != source_city:
                continue
            if other_category != source_category:
                continue

            # Base confidence from co-location; boost if vibe_dna is also close
            confidence = 0.55

            source_vibe = entity_data.get("vibe_dna")
            other_vibe = other.get("vibe_dna")
            if (
                source_vibe
                and other_vibe
                and isinstance(source_vibe, list)
                and isinstance(other_vibe, list)
            ):
                vibe_sim = _cosine_similarity(source_vibe, other_vibe)
                # Boost confidence proportionally but cap at 0.9
                confidence = min(0.9, confidence + 0.25 * vibe_sim)

            predictions.append(
                InferredRelationship(
                    source_id=entity_id,
                    target_id=other_id,
                    rel_type="COMPETES_WITH",
                    confidence=round(confidence, 4),
                    reasoning_type=ReasoningType.LINK_PREDICTION,
                    evidence=[
                        f"Same city: {source_city}",
                        f"Same category: {source_category}",
                    ],
                    explanation=(
                        f"Both entities are {source_category} in "
                        f"{source_city}, suggesting competition"
                    ),
                )
            )

        return predictions

    # ------------------------------------------------------------------
    # Scoring helper
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        rel_type: str,
    ) -> float:
        """Score a candidate relationship between *source* and *target*.

        Combines vibe-DNA similarity and category alignment into a
        single confidence value.

        Args:
            source: Source entity attribute dict.
            target: Target entity attribute dict.
            rel_type: Relationship type label.

        Returns:
            Confidence score in [0, 1].
        """
        score = 0.5  # Base score for transitive candidates

        # Vibe similarity boost
        source_vibe = source.get("vibe_dna")
        target_vibe = target.get("vibe_dna")
        if (
            source_vibe
            and target_vibe
            and isinstance(source_vibe, list)
            and isinstance(target_vibe, list)
        ):
            vibe_sim = _cosine_similarity(source_vibe, target_vibe)
            score += 0.3 * vibe_sim

        # Category alignment boost
        source_cat = str(source.get("category", "")).lower()
        target_cat = str(target.get("category", "")).lower()
        if source_cat and target_cat and source_cat == target_cat:
            score += 0.1

        return max(0.0, min(1.0, round(score, 4)))


# ---------------------------------------------------------------------------
# Rule Inductor
# ---------------------------------------------------------------------------


class RuleInductor:
    """Discovers and applies Horn-clause-style rules from graph patterns.

    Examines pairs of entities that share a known relationship and
    identifies common structural patterns in their attributes.  When a
    pattern occurs frequently enough (above ``min_support``) and with
    sufficient predictive accuracy (above ``min_confidence``), it is
    promoted to a :class:`GraphRule`.

    Currently recognises two rule templates:

    - ``shared_style + shared_city => SIMILAR_TO``
    - ``shared_category + shared_city => COMPETES_WITH``
    """

    def __init__(
        self,
        min_support: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._rule_counter: int = 0

    # ------------------------------------------------------------------
    # Rule discovery
    # ------------------------------------------------------------------

    def induce_rules(
        self,
        entity_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        relationships: list[dict[str, Any]],
    ) -> list[GraphRule]:
        """Discover rules from observed entity pairs and their relationships.

        For each entity pair that shares a relationship, checks which
        attribute-level patterns hold (shared style, shared city, shared
        category).  Patterns that recur above ``min_support`` with
        accuracy above ``min_confidence`` become rules.

        Args:
            entity_pairs: List of ``(entity_a_data, entity_b_data)``
                tuples representing connected entity pairs.
            relationships: List of relationship dicts, each containing
                ``source_id``, ``target_id``, and ``type`` keys.

        Returns:
            List of discovered :class:`GraphRule` objects.
        """
        # Build a set of known relationships keyed by (source, target)
        rel_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("source_id", ""))
            tgt = str(rel.get("target_id", ""))
            rtype = rel.get("type", rel.get("rel_type", ""))
            if src and tgt and rtype:
                rel_index[(src, tgt)].add(rtype)
                rel_index[(tgt, src)].add(rtype)

        # Count pattern occurrences
        # pattern key: (frozenset(antecedent_strings), consequent_string)
        pattern_support: Counter[
            tuple[tuple[str, ...], str]
        ] = Counter()
        pattern_total: Counter[tuple[str, ...]] = Counter()

        for entity_a, entity_b in entity_pairs:
            a_id = str(entity_a.get("id", ""))
            b_id = str(entity_b.get("id", ""))
            if not a_id or not b_id:
                continue

            pair_rels = rel_index.get((a_id, b_id), set())

            # Detect antecedent patterns
            antecedent_parts: list[str] = []

            # Shared style?
            a_styles = {
                str(s).lower()
                for s in (entity_a.get("styles") or [])
                if isinstance(entity_a.get("styles"), list)
            }
            b_styles = {
                str(s).lower()
                for s in (entity_b.get("styles") or [])
                if isinstance(entity_b.get("styles"), list)
            }
            shared_styles = a_styles & b_styles
            if shared_styles:
                antecedent_parts.append("A -HAS_STYLE-> S")
                antecedent_parts.append("B -HAS_STYLE-> S")

            # Shared city?
            a_city = str(entity_a.get("city", "")).strip().lower()
            b_city = str(entity_b.get("city", "")).strip().lower()
            if a_city and b_city and a_city == b_city:
                antecedent_parts.append("A -IS_IN-> City")
                antecedent_parts.append("B -IS_IN-> City")

            # Shared category?
            a_cat = str(entity_a.get("category", "")).strip().lower()
            b_cat = str(entity_b.get("category", "")).strip().lower()
            if a_cat and b_cat and a_cat == b_cat:
                antecedent_parts.append("A -HAS_CATEGORY-> C")
                antecedent_parts.append("B -HAS_CATEGORY-> C")

            if not antecedent_parts:
                continue

            antecedent_key = tuple(sorted(antecedent_parts))
            pattern_total[antecedent_key] += 1

            # Check which consequents hold
            for consequent_rel in ("SIMILAR_TO", "COMPETES_WITH"):
                if consequent_rel in pair_rels:
                    consequent_str = f"A -{consequent_rel}-> B"
                    pattern_support[
                        (antecedent_key, consequent_str)
                    ] += 1

        # Filter by min_support and min_confidence
        rules: list[GraphRule] = []

        for (ant_key, consequent), support in pattern_support.items():
            total = pattern_total.get(ant_key, 0)
            if total == 0:
                continue
            confidence = support / total

            if support < self._min_support:
                continue
            if confidence < self._min_confidence:
                continue

            self._rule_counter += 1
            rule = GraphRule(
                rule_id=f"rule_{self._rule_counter:04d}",
                antecedent=list(ant_key),
                consequent=consequent,
                confidence=round(confidence, 4),
                support_count=support,
            )
            rules.append(rule)

        logger.info(
            "RuleInductor: discovered %d rules from %d entity pairs "
            "(min_support=%d, min_confidence=%.2f)",
            len(rules),
            len(entity_pairs),
            self._min_support,
            self._min_confidence,
        )
        return rules

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    def apply_rule(
        self,
        rule: GraphRule,
        entities: list[dict[str, Any]],
    ) -> list[InferredRelationship]:
        """Apply a discovered rule to a set of entities.

        For every pair of entities that satisfies the rule's antecedent,
        generates an :class:`InferredRelationship` with the rule's
        consequent.

        Args:
            rule: The rule to apply.
            entities: List of entity dicts to evaluate.

        Returns:
            List of inferred relationships.
        """
        inferred: list[InferredRelationship] = []

        # Parse the consequent to extract rel_type
        # Expected format: "A -REL_TYPE-> B"
        consequent_rel = self._parse_rel_type(rule.consequent)
        if not consequent_rel:
            logger.warning(
                "Could not parse consequent '%s' in rule %s",
                rule.consequent,
                rule.rule_id,
            )
            return inferred

        # Determine required antecedent checks
        needs_shared_style = (
            "A -HAS_STYLE-> S" in rule.antecedent
            and "B -HAS_STYLE-> S" in rule.antecedent
        )
        needs_shared_city = (
            "A -IS_IN-> City" in rule.antecedent
            and "B -IS_IN-> City" in rule.antecedent
        )
        needs_shared_category = (
            "A -HAS_CATEGORY-> C" in rule.antecedent
            and "B -HAS_CATEGORY-> C" in rule.antecedent
        )

        # Check all pairs
        for i, entity_a in enumerate(entities):
            a_id = str(entity_a.get("id", ""))
            if not a_id:
                continue

            for j, entity_b in enumerate(entities):
                if j <= i:
                    continue

                b_id = str(entity_b.get("id", ""))
                if not b_id:
                    continue

                # Check antecedent conditions
                if needs_shared_style:
                    a_styles = {
                        str(s).lower()
                        for s in (entity_a.get("styles") or [])
                        if isinstance(entity_a.get("styles"), list)
                    }
                    b_styles = {
                        str(s).lower()
                        for s in (entity_b.get("styles") or [])
                        if isinstance(entity_b.get("styles"), list)
                    }
                    if not (a_styles & b_styles):
                        continue

                if needs_shared_city:
                    a_city = str(entity_a.get("city", "")).strip().lower()
                    b_city = str(entity_b.get("city", "")).strip().lower()
                    if not a_city or a_city != b_city:
                        continue

                if needs_shared_category:
                    a_cat = str(
                        entity_a.get("category", "")
                    ).strip().lower()
                    b_cat = str(
                        entity_b.get("category", "")
                    ).strip().lower()
                    if not a_cat or a_cat != b_cat:
                        continue

                # All antecedent conditions met - produce inference
                evidence_parts = [
                    f"Rule {rule.rule_id}: {' AND '.join(rule.antecedent)} "
                    f"=> {rule.consequent}",
                    f"Rule confidence: {rule.confidence:.4f} "
                    f"(support={rule.support_count})",
                ]

                inferred.append(
                    InferredRelationship(
                        source_id=a_id,
                        target_id=b_id,
                        rel_type=consequent_rel,
                        confidence=round(rule.confidence, 4),
                        reasoning_type=ReasoningType.RULE_INDUCTION,
                        evidence=evidence_parts,
                        explanation=(
                            f"Inferred via rule {rule.rule_id}: entities "
                            f"satisfy {len(rule.antecedent)} antecedent "
                            f"conditions"
                        ),
                    )
                )

        logger.info(
            "RuleInductor.apply_rule(%s): %d inferred relationships "
            "from %d entities",
            rule.rule_id,
            len(inferred),
            len(entities),
        )
        return inferred

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rel_type(pattern: str) -> str:
        """Extract the relationship type from a pattern string.

        Expects the format ``"A -REL_TYPE-> B"`` and returns
        ``"REL_TYPE"``.

        Args:
            pattern: A consequent pattern string.

        Returns:
            The extracted relationship type, or an empty string if
            parsing fails.
        """
        # Pattern: "A -REL_TYPE-> B"
        try:
            start = pattern.index("-") + 1
            end = pattern.index("->")
            return pattern[start:end].strip()
        except ValueError:
            return ""


# ---------------------------------------------------------------------------
# Temporal Reasoner
# ---------------------------------------------------------------------------


class TemporalReasoner:
    """Detects trends and predicts future states from entity timelines.

    Analyses how entity attributes (particularly connection counts and
    vibe-DNA dimensions) change over time, classifying entities into
    trend categories and extrapolating short-term futures via linear
    regression.
    """

    def detect_trends(
        self,
        entity_timeline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect trends from a sequence of entity snapshots.

        Each snapshot is a dict that must contain at least a
        ``timestamp`` key (ISO-8601 string or datetime) and may contain
        ``connection_count`` (int) and ``vibe_dna`` (list[float]).

        Args:
            entity_timeline: Chronologically ordered list of entity
                snapshots.

        Returns:
            Dict with keys:

            - ``trend`` -- one of ``"rising"``, ``"declining"``,
              ``"stable"``, ``"insufficient_data"``.
            - ``connection_trend`` -- slope of connection count over
              time.
            - ``vibe_drift`` -- average per-dimension change in
              ``vibe_dna`` between first and last snapshot.
            - ``snapshot_count`` -- number of snapshots analysed.
        """
        if len(entity_timeline) < 2:
            return {
                "trend": "insufficient_data",
                "connection_trend": 0.0,
                "vibe_drift": 0.0,
                "snapshot_count": len(entity_timeline),
            }

        # Connection count trend
        connection_counts: list[float] = []
        timestamps: list[float] = []

        for snapshot in entity_timeline:
            ts = snapshot.get("timestamp")
            count = snapshot.get("connection_count")

            if ts is not None:
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        continue
                if isinstance(ts, datetime):
                    timestamps.append(ts.timestamp())

            if count is not None:
                try:
                    connection_counts.append(float(count))
                except (TypeError, ValueError):
                    pass

        # Compute connection trend via simple linear slope
        connection_slope = 0.0
        if len(connection_counts) >= 2:
            n = len(connection_counts)
            indices = list(range(n))
            mean_x = sum(indices) / n
            mean_y = sum(connection_counts) / n
            numerator = sum(
                (x - mean_x) * (y - mean_y)
                for x, y in zip(indices, connection_counts)
            )
            denominator = sum((x - mean_x) ** 2 for x in indices)
            if denominator > 0:
                connection_slope = numerator / denominator

        # Vibe drift: average per-dimension change
        vibe_drift = 0.0
        first_vibe = entity_timeline[0].get("vibe_dna")
        last_vibe = entity_timeline[-1].get("vibe_dna")

        if (
            first_vibe
            and last_vibe
            and isinstance(first_vibe, list)
            and isinstance(last_vibe, list)
            and len(first_vibe) == len(last_vibe)
        ):
            diffs = [abs(a - b) for a, b in zip(last_vibe, first_vibe)]
            vibe_drift = sum(diffs) / len(diffs) if diffs else 0.0

        # Classify overall trend
        if connection_slope > 0.5:
            trend = "rising"
        elif connection_slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        result = {
            "trend": trend,
            "connection_trend": round(connection_slope, 4),
            "vibe_drift": round(vibe_drift, 4),
            "snapshot_count": len(entity_timeline),
        }

        logger.debug(
            "TemporalReasoner.detect_trends: %d snapshots -> trend=%s "
            "(slope=%.4f, drift=%.4f)",
            len(entity_timeline),
            trend,
            connection_slope,
            vibe_drift,
        )
        return result

    def predict_future_state(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        history: list[dict[str, Any]],
        horizon_steps: int = 3,
    ) -> dict[str, Any]:
        """Predict future entity state via linear extrapolation.

        Uses the entity's historical vibe-DNA snapshots to fit a
        per-dimension linear trend and extrapolates *horizon_steps*
        into the future.

        Args:
            entity_id: Entity identifier.
            entity_data: Current entity attribute dict.
            history: Chronologically ordered list of past snapshots,
                each containing a ``vibe_dna`` field.
            horizon_steps: Number of steps into the future to
                extrapolate.

        Returns:
            Dict with keys:

            - ``entity_id`` -- echo of the input.
            - ``predicted_vibe_dna`` -- extrapolated vibe vector.
            - ``predicted_connection_trend`` -- extrapolated connection
              trajectory.
            - ``horizon_steps`` -- echo of the input.
            - ``confidence`` -- confidence in the prediction (higher
              with more history).
        """
        # Collect vibe_dna history
        vibe_history: list[list[float]] = []
        connection_history: list[float] = []

        for snapshot in history:
            vibe = snapshot.get("vibe_dna")
            if vibe and isinstance(vibe, list):
                vibe_history.append(vibe)
            conn = snapshot.get("connection_count")
            if conn is not None:
                try:
                    connection_history.append(float(conn))
                except (TypeError, ValueError):
                    pass

        # Include current state
        current_vibe = entity_data.get("vibe_dna")
        if current_vibe and isinstance(current_vibe, list):
            vibe_history.append(current_vibe)

        current_conn = entity_data.get("connection_count")
        if current_conn is not None:
            try:
                connection_history.append(float(current_conn))
            except (TypeError, ValueError):
                pass

        # Predict vibe_dna
        predicted_vibe: list[float] = []
        if len(vibe_history) >= 2:
            n_dims = len(vibe_history[0])
            for dim in range(n_dims):
                values = [
                    v[dim] for v in vibe_history if dim < len(v)
                ]
                if len(values) >= 2:
                    slope = self._linear_slope(values)
                    last_val = values[-1]
                    predicted = last_val + slope * horizon_steps
                    # Clamp to [0, 1] for normalised vibe dimensions
                    predicted_vibe.append(
                        round(max(0.0, min(1.0, predicted)), 4)
                    )
                else:
                    predicted_vibe.append(
                        round(vibe_history[-1][dim], 4)
                        if dim < len(vibe_history[-1])
                        else 0.0
                    )
        elif current_vibe and isinstance(current_vibe, list):
            predicted_vibe = [round(v, 4) for v in current_vibe]

        # Predict connection trend
        predicted_conn_slope = 0.0
        if len(connection_history) >= 2:
            predicted_conn_slope = self._linear_slope(connection_history)

        # Confidence: more history = more confident
        confidence = min(1.0, len(vibe_history) / 10.0)

        result = {
            "entity_id": entity_id,
            "predicted_vibe_dna": predicted_vibe,
            "predicted_connection_trend": round(predicted_conn_slope, 4),
            "horizon_steps": horizon_steps,
            "confidence": round(confidence, 4),
        }

        logger.debug(
            "TemporalReasoner.predict_future_state(%s): %d vibe snapshots, "
            "confidence=%.4f",
            entity_id,
            len(vibe_history),
            confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        """Compute the slope of a simple linear regression on *values*.

        The independent variable is the positional index (0, 1, 2, ...).

        Args:
            values: Sequence of observed values.

        Returns:
            Slope of the best-fit line.  Returns 0.0 if fewer than two
            values are provided.
        """
        n = len(values)
        if n < 2:
            return 0.0

        indices = list(range(n))
        mean_x = sum(indices) / n
        mean_y = sum(values) / n

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(indices, values)
        )
        denominator = sum((x - mean_x) ** 2 for x in indices)

        if denominator == 0:
            return 0.0
        return numerator / denominator


# ---------------------------------------------------------------------------
# Counterfactual Reasoner
# ---------------------------------------------------------------------------


class CounterfactualReasoner:
    """Estimates the impact of hypothetical graph mutations.

    Answers "what-if" questions: how would the entity's graph position
    change if a specific edge were added or removed?
    """

    def what_if_remove_edge(
        self,
        entity_id: str,
        edge_type: str,
        target_id: str,
        graph_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate impact of removing an edge.

        Analyses the entity's neighbourhood in *graph_state* and
        computes connectivity and recommendation impact scores under
        the counterfactual scenario where the edge
        ``(entity_id)-[edge_type]->(target_id)`` is removed.

        Args:
            entity_id: Source entity of the edge to remove.
            edge_type: Relationship type of the edge to remove.
            target_id: Target entity of the edge to remove.
            graph_state: Dict representing the current graph.  Expected
                keys: ``entities`` (dict[str, entity_data]),
                ``relationships`` (list[dict]).

        Returns:
            Dict with keys:

            - ``entity_id``, ``removed_edge`` -- echo of inputs.
            - ``connectivity_before`` / ``connectivity_after`` --
              connection counts.
            - ``connectivity_impact`` -- change in connections.
            - ``affected_paths`` -- number of two-hop paths disrupted.
            - ``recommendation_impact`` -- estimated change to
              recommendation score.
            - ``severity`` -- ``"high"``, ``"medium"``, or ``"low"``.
        """
        entities = graph_state.get("entities", {})
        relationships = graph_state.get("relationships", [])

        # Current connectivity: count edges involving entity_id
        current_edges = [
            r for r in relationships
            if isinstance(r, dict)
            and (
                str(r.get("source_id", "")) == entity_id
                or str(r.get("target_id", "")) == entity_id
            )
        ]
        connectivity_before = len(current_edges)

        # After removal: exclude the specific edge
        remaining_edges = [
            r for r in current_edges
            if not (
                str(r.get("source_id", "")) == entity_id
                and str(r.get("target_id", "")) == target_id
                and r.get("type", r.get("rel_type", "")) == edge_type
            )
        ]
        connectivity_after = len(remaining_edges)

        # Count affected two-hop paths through the removed edge
        # Paths: entity_id -> target_id -> X  (these are now unreachable
        # via this edge)
        target_onward = [
            r for r in relationships
            if isinstance(r, dict)
            and str(r.get("source_id", "")) == target_id
            and str(r.get("target_id", "")) != entity_id
        ]
        affected_paths = len(target_onward)

        # Recommendation impact heuristic: based on edge type importance
        edge_type_weights: dict[str, float] = {
            "SIMILAR_TO": 0.8,
            "SIMILAR_VIBE": 0.7,
            "COMPETES_WITH": 0.5,
            "NEAR": 0.4,
            "SELLS_AT": 0.6,
            "AVAILABLE_AT": 0.6,
            "EMBEDDING_NEIGHBOR": 0.7,
            "INFLUENCED_BY": 0.5,
            "INSPIRED_BY": 0.5,
        }
        edge_weight = edge_type_weights.get(edge_type, 0.3)

        # Scale by proportion of connections lost
        if connectivity_before > 0:
            connectivity_ratio = (
                (connectivity_before - connectivity_after)
                / connectivity_before
            )
        else:
            connectivity_ratio = 0.0

        recommendation_impact = round(
            edge_weight * connectivity_ratio * -1.0, 4
        )

        # Severity classification
        if abs(recommendation_impact) > 0.5:
            severity = "high"
        elif abs(recommendation_impact) > 0.2:
            severity = "medium"
        else:
            severity = "low"

        result = {
            "entity_id": entity_id,
            "removed_edge": {
                "type": edge_type,
                "target_id": target_id,
            },
            "connectivity_before": connectivity_before,
            "connectivity_after": connectivity_after,
            "connectivity_impact": connectivity_after - connectivity_before,
            "affected_paths": affected_paths,
            "recommendation_impact": recommendation_impact,
            "severity": severity,
        }

        logger.debug(
            "CounterfactualReasoner.what_if_remove_edge(%s -[%s]-> %s): "
            "severity=%s, rec_impact=%.4f",
            entity_id,
            edge_type,
            target_id,
            severity,
            recommendation_impact,
        )
        return result

    def what_if_add_edge(
        self,
        entity_id: str,
        edge_type: str,
        target_id: str,
        graph_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate impact of adding an edge.

        Analyses the entity's neighbourhood and estimates how adding
        the edge ``(entity_id)-[edge_type]->(target_id)`` would change
        connectivity and recommendations.

        Args:
            entity_id: Source entity of the proposed edge.
            edge_type: Relationship type of the proposed edge.
            target_id: Target entity of the proposed edge.
            graph_state: Dict representing the current graph.  Expected
                keys: ``entities`` (dict[str, entity_data]),
                ``relationships`` (list[dict]).

        Returns:
            Dict with keys:

            - ``entity_id``, ``added_edge`` -- echo of inputs.
            - ``connectivity_before`` / ``connectivity_after`` --
              connection counts.
            - ``connectivity_impact`` -- change in connections.
            - ``new_paths`` -- number of new two-hop paths unlocked.
            - ``recommendation_impact`` -- estimated positive change to
              recommendation score.
            - ``redundancy`` -- ``True`` if a similar edge already exists.
        """
        entities = graph_state.get("entities", {})
        relationships = graph_state.get("relationships", [])

        # Current connectivity
        current_edges = [
            r for r in relationships
            if isinstance(r, dict)
            and (
                str(r.get("source_id", "")) == entity_id
                or str(r.get("target_id", "")) == entity_id
            )
        ]
        connectivity_before = len(current_edges)
        connectivity_after = connectivity_before + 1

        # Check for redundancy: does a similar edge already exist?
        redundancy = any(
            isinstance(r, dict)
            and str(r.get("source_id", "")) == entity_id
            and str(r.get("target_id", "")) == target_id
            and r.get("type", r.get("rel_type", "")) == edge_type
            for r in relationships
        )

        # New two-hop paths unlocked via the target
        target_connections = [
            r for r in relationships
            if isinstance(r, dict)
            and str(r.get("source_id", "")) == target_id
            and str(r.get("target_id", "")) != entity_id
        ]
        # Exclude paths to entities already directly connected
        existing_neighbours = {
            str(r.get("target_id", ""))
            for r in current_edges
            if isinstance(r, dict)
            and str(r.get("source_id", "")) == entity_id
        }
        new_paths = sum(
            1
            for r in target_connections
            if str(r.get("target_id", "")) not in existing_neighbours
        )

        # Recommendation impact heuristic
        edge_type_weights: dict[str, float] = {
            "SIMILAR_TO": 0.8,
            "SIMILAR_VIBE": 0.7,
            "COMPETES_WITH": 0.5,
            "NEAR": 0.4,
            "SELLS_AT": 0.6,
            "AVAILABLE_AT": 0.6,
            "EMBEDDING_NEIGHBOR": 0.7,
            "INFLUENCED_BY": 0.5,
            "INSPIRED_BY": 0.5,
        }
        edge_weight = edge_type_weights.get(edge_type, 0.3)

        # New-path diversity bonus
        path_bonus = min(0.3, new_paths * 0.05)
        recommendation_impact = round(edge_weight * 0.3 + path_bonus, 4)

        if redundancy:
            recommendation_impact = round(recommendation_impact * 0.2, 4)

        result = {
            "entity_id": entity_id,
            "added_edge": {
                "type": edge_type,
                "target_id": target_id,
            },
            "connectivity_before": connectivity_before,
            "connectivity_after": connectivity_after,
            "connectivity_impact": 1,
            "new_paths": new_paths,
            "recommendation_impact": recommendation_impact,
            "redundancy": redundancy,
        }

        logger.debug(
            "CounterfactualReasoner.what_if_add_edge(%s -[%s]-> %s): "
            "new_paths=%d, rec_impact=%.4f, redundancy=%s",
            entity_id,
            edge_type,
            target_id,
            new_paths,
            recommendation_impact,
            redundancy,
        )
        return result


# ---------------------------------------------------------------------------
# Abductive Reasoner
# ---------------------------------------------------------------------------


class AbductiveReasoner:
    """Generates plausible explanations for observed user-entity interactions.

    Given an observation (e.g. "user liked entity X"), inspects the
    graph context to enumerate possible causal explanations: shared
    styles with previously liked entities, geographic preference,
    category affinity, vibe alignment, and graph-based connections.
    """

    def explain_observation(
        self,
        observation: dict[str, Any],
        graph_state: dict[str, Any],
    ) -> list[str]:
        """Generate possible explanations for an observation.

        The observation dict should contain:

        - ``user_id`` (str): Identifier of the acting user.
        - ``entity_id`` (str): Identifier of the entity involved.
        - ``action`` (str): What happened (e.g. ``"liked"``,
          ``"visited"``, ``"saved"``).
        - ``user_history`` (list[dict], optional): Previously
          interacted entity dicts.
        - ``user_profile`` (dict, optional): Learned user preferences.

        Args:
            observation: Dict describing the observed event.
            graph_state: Dict representing the graph with keys
                ``entities`` (dict[str, entity_data]) and
                ``relationships`` (list[dict]).

        Returns:
            List of natural-language explanation strings, ordered from
            most to least likely.
        """
        explanations: list[str] = []

        entity_id = str(observation.get("entity_id", ""))
        action = observation.get("action", "interacted with")
        user_history = observation.get("user_history", [])
        user_profile = observation.get("user_profile", {})

        entities = graph_state.get("entities", {})
        relationships = graph_state.get("relationships", [])

        # Get the target entity data
        entity_data = entities.get(entity_id, {})
        entity_name = entity_data.get("name", entity_id)
        entity_styles = set()
        raw_styles = entity_data.get("styles", [])
        if isinstance(raw_styles, list):
            entity_styles = {str(s).lower() for s in raw_styles}
        entity_city = str(entity_data.get("city", "")).strip().lower()
        entity_category = str(
            entity_data.get("category", "")
        ).strip().lower()

        # Explanation 1: shared style with previously liked entities
        for hist_entity in user_history:
            hist_name = hist_entity.get("name", "unknown")
            hist_styles = set()
            raw_hist_styles = hist_entity.get("styles", [])
            if isinstance(raw_hist_styles, list):
                hist_styles = {str(s).lower() for s in raw_hist_styles}

            shared = entity_styles & hist_styles
            if shared:
                explanations.append(
                    f"Because {entity_name} shares style "
                    f"({', '.join(sorted(shared))}) with "
                    f"{hist_name} which user previously {action}"
                )

        # Explanation 2: city preference
        if entity_city:
            city_prefs = user_profile.get("city_preferences", {})
            if city_prefs.get(entity_city, 0.0) > 0.0:
                explanations.append(
                    f"Because {entity_name} is in user's preferred city "
                    f"({entity_city})"
                )

            # Also check history for city matches
            for hist_entity in user_history:
                hist_city = str(
                    hist_entity.get("city", "")
                ).strip().lower()
                if hist_city == entity_city:
                    explanations.append(
                        f"Because {entity_name} is in {entity_city}, "
                        f"where user previously explored "
                        f"{hist_entity.get('name', 'other entities')}"
                    )
                    break

        # Explanation 3: category affinity
        if entity_category:
            cat_affinities = user_profile.get("category_affinities", {})
            if cat_affinities.get(entity_category, 0.0) > 0.0:
                explanations.append(
                    f"Because user has shown affinity for "
                    f"{entity_category} entities and {entity_name} is a "
                    f"{entity_category}"
                )

        # Explanation 4: vibe alignment with user preference
        entity_vibe = entity_data.get("vibe_dna")
        user_vibe_pref = user_profile.get("vibe_preferences")
        if (
            entity_vibe
            and user_vibe_pref
            and isinstance(entity_vibe, list)
            and isinstance(user_vibe_pref, list)
        ):
            vibe_sim = _cosine_similarity(entity_vibe, user_vibe_pref)
            if vibe_sim > 0.6:
                explanations.append(
                    f"Because {entity_name}'s vibe DNA aligns with "
                    f"user's vibe preferences (similarity={vibe_sim:.3f})"
                )

        # Explanation 5: graph connection to a previously liked entity
        entity_rels = [
            r for r in relationships
            if isinstance(r, dict)
            and (
                str(r.get("source_id", "")) == entity_id
                or str(r.get("target_id", "")) == entity_id
            )
        ]
        hist_ids = {
            str(h.get("id", "")) for h in user_history if h.get("id")
        }
        for rel in entity_rels:
            src = str(rel.get("source_id", ""))
            tgt = str(rel.get("target_id", ""))
            rel_type = rel.get("type", rel.get("rel_type", ""))
            connected_id = tgt if src == entity_id else src

            if connected_id in hist_ids:
                connected_name = entities.get(
                    connected_id, {}
                ).get("name", connected_id)
                explanations.append(
                    f"Because {entity_name} is connected to "
                    f"{connected_name} (via {rel_type}), which user "
                    f"previously {action}"
                )

        # Explanation 6: popularity
        popularity = entity_data.get("popularity_score")
        if popularity is not None:
            try:
                popularity = float(popularity)
                if popularity > 0.7:
                    explanations.append(
                        f"Because {entity_name} is highly popular "
                        f"(score={popularity:.2f}) in the community"
                    )
            except (TypeError, ValueError):
                pass

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_explanations: list[str] = []
        for exp in explanations:
            if exp not in seen:
                seen.add(exp)
                unique_explanations.append(exp)

        logger.debug(
            "AbductiveReasoner.explain_observation(entity=%s): "
            "%d explanations generated",
            entity_id,
            len(unique_explanations),
        )
        return unique_explanations


# ---------------------------------------------------------------------------
# KG Reasoning Engine (orchestrator)
# ---------------------------------------------------------------------------


class KGReasoningEngine:
    """Orchestrates all knowledge-graph reasoning capabilities.

    Combines :class:`LinkPredictor`, :class:`RuleInductor`,
    :class:`TemporalReasoner`, :class:`CounterfactualReasoner`, and
    :class:`AbductiveReasoner` into a single facade.

    Usage::

        engine = get_kg_reasoning_engine()
        results = await engine.reason("entity-42", entity_data)
        inferred = engine.get_inferred_relationships("entity-42")
        rules = engine.get_discovered_rules()
    """

    def __init__(self) -> None:
        self.link_predictor = LinkPredictor()
        self.rule_inductor = RuleInductor()
        self.temporal_reasoner = TemporalReasoner()
        self.counterfactual_reasoner = CounterfactualReasoner()
        self.abductive_reasoner = AbductiveReasoner()

        # Caches for inferred state
        self._inferred_relationships: dict[
            str, list[InferredRelationship]
        ] = defaultdict(list)
        self._discovered_rules: list[GraphRule] = []

        logger.info("KGReasoningEngine initialised")

    # ------------------------------------------------------------------
    # Main reasoning entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        reasoning_types: Optional[list[ReasoningType]] = None,
        all_entities: Optional[list[dict[str, Any]]] = None,
        graph_state: Optional[dict[str, Any]] = None,
        entity_timeline: Optional[list[dict[str, Any]]] = None,
        entity_pairs: Optional[
            list[tuple[dict[str, Any], dict[str, Any]]]
        ] = None,
        relationships: Optional[list[dict[str, Any]]] = None,
        observation: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one or more reasoning modes for the given entity.

        By default, all reasoning types are executed.  Pass a subset
        via *reasoning_types* to restrict which engines run.

        Args:
            entity_id: Entity to reason about.
            entity_data: Current attribute dict for the entity.
            reasoning_types: List of :class:`ReasoningType` values to
                run.  ``None`` means all.
            all_entities: List of all entity dicts (needed for link
                prediction and rule induction).
            graph_state: Graph dict with ``entities`` and
                ``relationships`` keys (needed for counterfactual and
                abductive reasoning).
            entity_timeline: Chronological snapshots for temporal
                reasoning.
            entity_pairs: Connected entity pairs for rule induction.
            relationships: Relationship dicts for rule induction.
            observation: Observation dict for abductive reasoning.

        Returns:
            Dict with keys corresponding to each executed reasoning
            type, plus a ``summary`` key with aggregate statistics.
        """
        if reasoning_types is None:
            reasoning_types = list(ReasoningType)

        all_entities = all_entities or []
        graph_state = graph_state or {"entities": {}, "relationships": []}
        entity_timeline = entity_timeline or []
        entity_pairs = entity_pairs or []
        relationships = relationships or []

        results: dict[str, Any] = {}

        # --- Link prediction ---
        if ReasoningType.LINK_PREDICTION in reasoning_types:
            predicted_links = self.link_predictor.predict_links(
                entity_id, entity_data, all_entities
            )
            results["link_prediction"] = {
                "predictions": [
                    {
                        "source_id": p.source_id,
                        "target_id": p.target_id,
                        "rel_type": p.rel_type,
                        "confidence": p.confidence,
                        "explanation": p.explanation,
                        "evidence": p.evidence,
                    }
                    for p in predicted_links
                ],
                "count": len(predicted_links),
            }
            # Cache
            self._inferred_relationships[entity_id].extend(predicted_links)

        # --- Rule induction ---
        if ReasoningType.RULE_INDUCTION in reasoning_types:
            discovered_rules: list[GraphRule] = []
            rule_inferred: list[InferredRelationship] = []

            if entity_pairs and relationships:
                discovered_rules = self.rule_inductor.induce_rules(
                    entity_pairs, relationships
                )
                self._discovered_rules.extend(discovered_rules)

                # Apply each discovered rule to all entities
                for rule in discovered_rules:
                    rule_inferred.extend(
                        self.rule_inductor.apply_rule(rule, all_entities)
                    )

                # Cache inferred from rules
                for inf in rule_inferred:
                    self._inferred_relationships[inf.source_id].append(inf)

            results["rule_induction"] = {
                "discovered_rules": [
                    {
                        "rule_id": r.rule_id,
                        "antecedent": r.antecedent,
                        "consequent": r.consequent,
                        "confidence": r.confidence,
                        "support_count": r.support_count,
                    }
                    for r in discovered_rules
                ],
                "inferred_count": len(rule_inferred),
                "rules_discovered": len(discovered_rules),
            }

        # --- Temporal reasoning ---
        if ReasoningType.TEMPORAL in reasoning_types:
            trends = self.temporal_reasoner.detect_trends(entity_timeline)
            future_state = self.temporal_reasoner.predict_future_state(
                entity_id, entity_data, entity_timeline
            )
            results["temporal"] = {
                "trends": trends,
                "future_state": future_state,
            }

        # --- Counterfactual reasoning ---
        if ReasoningType.COUNTERFACTUAL in reasoning_types:
            # Run counterfactual analysis on all existing edges
            entity_rels = [
                r
                for r in graph_state.get("relationships", [])
                if isinstance(r, dict)
                and str(r.get("source_id", "")) == entity_id
            ]

            counterfactual_results: list[dict[str, Any]] = []
            for rel in entity_rels[:5]:  # Limit to 5 to avoid explosion
                rel_type = rel.get("type", rel.get("rel_type", ""))
                target = str(rel.get("target_id", ""))
                if rel_type and target:
                    cf = self.counterfactual_reasoner.what_if_remove_edge(
                        entity_id, rel_type, target, graph_state
                    )
                    counterfactual_results.append(cf)

            results["counterfactual"] = {
                "analyses": counterfactual_results,
                "count": len(counterfactual_results),
            }

        # --- Abductive reasoning ---
        if ReasoningType.ABDUCTIVE in reasoning_types:
            if observation:
                explanations = self.abductive_reasoner.explain_observation(
                    observation, graph_state
                )
            else:
                # Build a default observation from context
                explanations = self.abductive_reasoner.explain_observation(
                    {
                        "entity_id": entity_id,
                        "action": "liked",
                        "user_history": [],
                        "user_profile": {},
                    },
                    graph_state,
                )
            results["abductive"] = {
                "explanations": explanations,
                "count": len(explanations),
            }

        # --- Summary ---
        total_inferred = sum(
            len(rels)
            for rels in self._inferred_relationships.values()
        )
        results["summary"] = {
            "entity_id": entity_id,
            "reasoning_types_executed": [rt.value for rt in reasoning_types],
            "total_inferred_relationships": total_inferred,
            "total_discovered_rules": len(self._discovered_rules),
        }

        logger.info(
            "KGReasoningEngine.reason(%s): executed %d reasoning types, "
            "%d total inferred relationships, %d rules discovered",
            entity_id,
            len(reasoning_types),
            total_inferred,
            len(self._discovered_rules),
        )
        return results

    # ------------------------------------------------------------------
    # Query cached state
    # ------------------------------------------------------------------

    def get_inferred_relationships(
        self,
        entity_id: str,
    ) -> list[InferredRelationship]:
        """Return all inferred relationships for *entity_id*.

        Args:
            entity_id: Entity to query.

        Returns:
            List of :class:`InferredRelationship` objects, sorted by
            confidence descending.
        """
        rels = self._inferred_relationships.get(entity_id, [])
        return sorted(rels, key=lambda r: r.confidence, reverse=True)

    def get_discovered_rules(self) -> list[GraphRule]:
        """Return all rules discovered across all reasoning sessions.

        Returns:
            List of :class:`GraphRule` objects, sorted by confidence
            descending.
        """
        return sorted(
            self._discovered_rules,
            key=lambda r: r.confidence,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine: Optional[KGReasoningEngine] = None


def get_kg_reasoning_engine() -> KGReasoningEngine:
    """Return the singleton KGReasoningEngine instance."""
    global _engine
    if _engine is None:
        _engine = KGReasoningEngine()
    return _engine
