"""Causal Inference Engine for ARBOR Enterprise.

Provides causal reasoning capabilities for the discovery platform,
enabling the system to move beyond correlational insights (``users who
clicked X also clicked Y``) toward genuine causal understanding
(``recommending X *caused* the user to convert``).

The module implements three complementary layers:

1. **Structural causal model** -- a directed acyclic graph (DAG) encoding
   the assumed data-generating process for the recommendation pipeline
   (recommendation position, entity quality, user preference, click,
   conversion).
2. **Uplift estimation** -- difference-in-means estimation of the Average
   Treatment Effect (ATE) of being recommended on downstream outcomes.
3. **Counterfactual explanation** -- human-readable sentences of the form
   ``User converted BECAUSE recommendation was at position 1``.

Usage::

    engine = get_causal_engine()
    effect = engine.estimate_recommendation_effect(
        entity_id="e_42",
        interactions=[...],
    )
    print(effect.ate, effect.p_value)

    explanation = engine.explain_conversion(
        user_id="u_7", entity_id="e_42", context={...}
    )
    print(explanation)
    # "User converted BECAUSE recommendation was at position 1 (vs avg position 3)"
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CausalVariable:
    """A single variable in a structural causal model.

    Attributes:
        name: Human-readable identifier for the variable.
        var_type: Role in the causal graph -- one of ``"treatment"``,
            ``"outcome"``, ``"confounder"``, or ``"instrument"``.
        values: Possible values / levels the variable can take.
    """

    name: str
    var_type: str  # "treatment", "outcome", "confounder", "instrument"
    values: list = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_types = {"treatment", "outcome", "confounder", "instrument"}
        if self.var_type not in valid_types:
            raise ValueError(f"var_type must be one of {valid_types}, got '{self.var_type}'")


@dataclass
class CausalEffect:
    """Estimated causal effect of a treatment on an outcome.

    Attributes:
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        ate: Average Treatment Effect -- the mean difference in outcome
            between treatment and control groups.
        confidence_interval: 95 % confidence interval for the ATE as
            ``(lower, upper)``.
        p_value: Two-sided p-value for the null hypothesis ATE = 0.
        sample_size: Total number of observations used in the estimate.
    """

    treatment: str
    outcome: str
    ate: float
    confidence_interval: tuple[float, float]
    p_value: float
    sample_size: int


# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------


class CausalGraph:
    """Directed acyclic graph encoding causal relationships.

    The graph stores :class:`CausalVariable` nodes and directed edges
    ``(parent → child)`` representing direct causal influence.  A built-in
    cycle check ensures the graph remains a valid DAG.
    """

    def __init__(self) -> None:
        self._variables: dict[str, CausalVariable] = {}
        self._edges: dict[str, set[str]] = defaultdict(set)  # parent → children
        self._reverse_edges: dict[str, set[str]] = defaultdict(set)  # child → parents

    # -- Mutation -----------------------------------------------------------

    def add_variable(self, variable: CausalVariable) -> None:
        """Register a variable in the graph.

        Args:
            variable: The :class:`CausalVariable` to add.

        Raises:
            ValueError: If a variable with the same name already exists.
        """
        if variable.name in self._variables:
            raise ValueError(f"Variable '{variable.name}' already exists in graph")
        self._variables[variable.name] = variable
        logger.debug("Added causal variable: %s (%s)", variable.name, variable.var_type)

    def add_edge(self, from_var: str, to_var: str) -> None:
        """Add a directed edge ``from_var → to_var``.

        Args:
            from_var: Name of the parent (cause) variable.
            to_var: Name of the child (effect) variable.

        Raises:
            KeyError: If either variable has not been added to the graph.
            ValueError: If adding the edge would create a cycle.
        """
        for name in (from_var, to_var):
            if name not in self._variables:
                raise KeyError(f"Variable '{name}' not found in graph")

        # Tentatively add edge and check for cycles
        self._edges[from_var].add(to_var)
        self._reverse_edges[to_var].add(from_var)

        if not self.is_valid():
            # Roll back
            self._edges[from_var].discard(to_var)
            self._reverse_edges[to_var].discard(from_var)
            raise ValueError(f"Adding edge {from_var} → {to_var} would create a cycle")

        logger.debug("Added causal edge: %s → %s", from_var, to_var)

    # -- Queries ------------------------------------------------------------

    def get_parents(self, var: str) -> list[str]:
        """Return the direct parents (causes) of *var*.

        Args:
            var: Variable name.

        Returns:
            Sorted list of parent variable names.
        """
        return sorted(self._reverse_edges.get(var, set()))

    def get_children(self, var: str) -> list[str]:
        """Return the direct children (effects) of *var*.

        Args:
            var: Variable name.

        Returns:
            Sorted list of child variable names.
        """
        return sorted(self._edges.get(var, set()))

    def get_confounders(self, treatment: str, outcome: str) -> list[str]:
        """Identify common causes (confounders) of *treatment* and *outcome*.

        A confounder is any variable that is an ancestor of **both** the
        treatment and the outcome, or any variable explicitly typed as
        ``"confounder"`` that has a directed path to both.

        For the purposes of this implementation we use a simplified
        heuristic: a variable is considered a confounder if it is a parent
        of the treatment OR the outcome and is typed ``"confounder"``, or
        if it is a common ancestor of both.

        Args:
            treatment: Treatment variable name.
            outcome: Outcome variable name.

        Returns:
            Sorted list of confounder variable names.
        """
        treatment_ancestors = self._get_ancestors(treatment)
        outcome_ancestors = self._get_ancestors(outcome)

        # Common ancestors
        common = treatment_ancestors & outcome_ancestors

        # Also include any explicitly-typed confounders that parent either
        typed_confounders: set[str] = set()
        for name, var in self._variables.items():
            if var.var_type == "confounder":
                parents_of_treatment = self._get_ancestors(treatment)
                parents_of_outcome = self._get_ancestors(outcome)
                if name in parents_of_treatment or name in parents_of_outcome:
                    typed_confounders.add(name)

        return sorted(common | typed_confounders)

    def is_valid(self) -> bool:
        """Check that the graph is a DAG (contains no directed cycles).

        Uses Kahn's algorithm (topological sort) to detect cycles.

        Returns:
            ``True`` if the graph is acyclic, ``False`` otherwise.
        """
        in_degree: dict[str, int] = {name: 0 for name in self._variables}
        for parent, children in self._edges.items():
            for child in children:
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for child in self._edges.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited == len(self._variables)

    def get_variable(self, name: str) -> CausalVariable | None:
        """Retrieve a variable by name, or ``None`` if not present."""
        return self._variables.get(name)

    @property
    def variables(self) -> list[str]:
        """All variable names in insertion order."""
        return list(self._variables.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """All directed edges as ``(parent, child)`` pairs."""
        result: list[tuple[str, str]] = []
        for parent, children in sorted(self._edges.items()):
            for child in sorted(children):
                result.append((parent, child))
        return result

    # -- Internal helpers ---------------------------------------------------

    def _get_ancestors(self, var: str) -> set[str]:
        """Return all ancestors of *var* (transitive closure of parents)."""
        ancestors: set[str] = set()
        stack = list(self._reverse_edges.get(var, set()))
        while stack:
            node = stack.pop()
            if node not in ancestors:
                ancestors.add(node)
                stack.extend(self._reverse_edges.get(node, set()))
        return ancestors


# ---------------------------------------------------------------------------
# Pre-built ARBOR causal graph
# ---------------------------------------------------------------------------


def build_arbor_causal_graph() -> CausalGraph:
    """Construct the default causal graph for ARBOR's discovery platform.

    Structure::

        user_preference ──►  recommendation  ◄── entity_quality
              │                     │                   │
              │                     ▼                   │
              │                   click  ◄──── position │
              │                     │                   │
              ▼                     ▼                   ▼
            conversion  ◄────── conversion

    Variables:
        - recommendation (treatment): whether / how the entity was recommended
        - click (outcome): whether the user clicked
        - conversion (outcome): whether the user converted (saved / booked)
        - user_preference (confounder): pre-existing user taste profile
        - entity_quality (confounder): intrinsic quality of the entity
        - position (confounder): rank position in the result list

    Returns:
        A fully-wired :class:`CausalGraph`.
    """
    graph = CausalGraph()

    # Variables
    graph.add_variable(
        CausalVariable(
            name="recommendation",
            var_type="treatment",
            values=["shown", "not_shown"],
        )
    )
    graph.add_variable(
        CausalVariable(
            name="click",
            var_type="outcome",
            values=["clicked", "not_clicked"],
        )
    )
    graph.add_variable(
        CausalVariable(
            name="conversion",
            var_type="outcome",
            values=["converted", "not_converted"],
        )
    )
    graph.add_variable(
        CausalVariable(
            name="user_preference",
            var_type="confounder",
            values=["high_affinity", "medium_affinity", "low_affinity"],
        )
    )
    graph.add_variable(
        CausalVariable(
            name="entity_quality",
            var_type="confounder",
            values=["high", "medium", "low"],
        )
    )
    graph.add_variable(
        CausalVariable(
            name="position",
            var_type="confounder",
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    )

    # Edges (cause → effect)
    graph.add_edge("recommendation", "click")
    graph.add_edge("click", "conversion")
    graph.add_edge("user_preference", "recommendation")
    graph.add_edge("user_preference", "click")
    graph.add_edge("user_preference", "conversion")
    graph.add_edge("entity_quality", "recommendation")
    graph.add_edge("entity_quality", "click")
    graph.add_edge("entity_quality", "conversion")
    graph.add_edge("position", "click")

    logger.info(
        "Built ARBOR causal graph: %d variables, %d edges",
        len(graph.variables),
        len(graph.edges),
    )
    return graph


# ---------------------------------------------------------------------------
# Uplift estimation
# ---------------------------------------------------------------------------


class UpliftEstimator:
    """Estimate the causal uplift (ATE) of a treatment using simple
    difference-in-means.

    This is the most straightforward estimator: it computes the mean outcome
    for the treatment group, the mean outcome for the control group, and
    reports the difference together with a Wald-type confidence interval.
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self._z = self._z_score(confidence_level)

    # -- Public API ---------------------------------------------------------

    def estimate_uplift(
        self,
        treatment_group: list[float],
        control_group: list[float],
        outcome_metric: str = "conversion",
    ) -> CausalEffect:
        """Compute the ATE via difference-in-means.

        Args:
            treatment_group: Outcome values for the treatment group.
            control_group: Outcome values for the control group.
            outcome_metric: Name of the outcome being measured.

        Returns:
            A :class:`CausalEffect` with the estimated ATE, confidence
            interval, and p-value.

        Raises:
            ValueError: If either group is empty.
        """
        if not treatment_group or not control_group:
            raise ValueError("Both treatment and control groups must be non-empty")

        n_t = len(treatment_group)
        n_c = len(control_group)
        mean_t = sum(treatment_group) / n_t
        mean_c = sum(control_group) / n_c
        ate = mean_t - mean_c

        var_t = self._variance(treatment_group, mean_t)
        var_c = self._variance(control_group, mean_c)

        se = math.sqrt(var_t / n_t + var_c / n_c) if (n_t > 1 and n_c > 1) else 0.0

        ci_lower = ate - self._z * se
        ci_upper = ate + self._z * se

        p_value = self._compute_p_value(ate, se)

        effect = CausalEffect(
            treatment="recommendation",
            outcome=outcome_metric,
            ate=round(ate, 6),
            confidence_interval=(round(ci_lower, 6), round(ci_upper, 6)),
            p_value=round(p_value, 6),
            sample_size=n_t + n_c,
        )

        logger.info(
            "Uplift estimate: ATE=%.4f, CI=(%.4f, %.4f), p=%.4f, n=%d",
            effect.ate,
            effect.confidence_interval[0],
            effect.confidence_interval[1],
            effect.p_value,
            effect.sample_size,
        )
        return effect

    def estimate_per_entity(
        self,
        entity_id: str,
        interactions: list[dict[str, Any]],
    ) -> float:
        """Estimate the incremental impact of a specific entity.

        Computes the difference in conversion rate when this entity was
        recommended vs. the baseline conversion rate across all interactions.

        Args:
            entity_id: Identifier of the entity.
            interactions: List of interaction dicts, each containing at least
                ``entity_id``, ``action``, and ``was_recommended``.

        Returns:
            The incremental conversion rate lift attributable to the entity.
        """
        entity_interactions = [i for i in interactions if i.get("entity_id") == entity_id]
        other_interactions = [i for i in interactions if i.get("entity_id") != entity_id]

        if not entity_interactions:
            logger.warning("No interactions found for entity %s", entity_id)
            return 0.0

        entity_conversions = sum(1 for i in entity_interactions if i.get("action") == "convert")
        entity_rate = entity_conversions / len(entity_interactions)

        if other_interactions:
            other_conversions = sum(1 for i in other_interactions if i.get("action") == "convert")
            baseline_rate = other_conversions / len(other_interactions)
        else:
            baseline_rate = 0.0

        uplift = entity_rate - baseline_rate

        logger.debug(
            "Entity %s uplift: %.4f (entity_rate=%.4f, baseline=%.4f)",
            entity_id,
            uplift,
            entity_rate,
            baseline_rate,
        )
        return round(uplift, 6)

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _variance(values: list[float], mean: float) -> float:
        """Compute sample variance (Bessel-corrected)."""
        n = len(values)
        if n < 2:
            return 0.0
        return sum((x - mean) ** 2 for x in values) / (n - 1)

    @staticmethod
    def _z_score(confidence_level: float) -> float:
        """Approximate z-score for a given confidence level.

        Uses common lookup values for standard levels and falls back to the
        rational approximation of the inverse normal CDF for others.
        """
        lookup = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        if confidence_level in lookup:
            return lookup[confidence_level]
        # Rational approximation (Abramowitz & Stegun 26.2.23)
        p = (1 + confidence_level) / 2
        t = math.sqrt(-2 * math.log(1 - p))
        return round(
            t
            - (2.515517 + 0.802853 * t + 0.010328 * t**2)
            / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3),
            4,
        )

    @staticmethod
    def _compute_p_value(ate: float, se: float) -> float:
        """Two-sided p-value using the normal approximation."""
        if se == 0:
            return 0.0 if ate != 0 else 1.0
        z = abs(ate) / se
        # Approximation of 2 * (1 - Phi(z)) using the complementary error function
        p = math.erfc(z / math.sqrt(2))
        return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Counterfactual explanations
# ---------------------------------------------------------------------------


class CounterfactualExplainer:
    """Generate human-readable counterfactual explanations for user actions.

    Each explanation answers the question *"Why did this outcome happen?"*
    by constructing a contrast between the observed situation and a
    hypothetical alternative.
    """

    # Default baseline statistics used when the real baselines are unknown.
    _DEFAULTS: dict[str, Any] = {
        "avg_position": 3.0,
        "avg_vibe_score": 0.65,
        "avg_click_rate": 0.12,
        "avg_conversion_rate": 0.04,
    }

    def explain(
        self,
        user_id: str,
        entity_id: str,
        action: str,
        context: dict[str, Any],
    ) -> str:
        """Generate a counterfactual explanation for an observed action.

        Args:
            user_id: The user who performed the action.
            entity_id: The entity the action was performed on.
            action: What happened (``"click"``, ``"convert"``, ``"dismiss"``).
            context: Contextual information -- may include ``position``,
                ``vibe_score``, ``category``, ``was_recommended``, etc.

        Returns:
            A human-readable explanation sentence.
        """
        explanations: list[str] = []

        # Position effect
        position = context.get("position")
        if position is not None:
            avg_pos = context.get("avg_position", self._DEFAULTS["avg_position"])
            if position < avg_pos:
                explanations.append(
                    f"User {action}ed BECAUSE recommendation was at position "
                    f"{position} (vs avg position {avg_pos:.0f})"
                )
            elif position > avg_pos:
                explanations.append(
                    f"User {action}ed DESPITE recommendation being at position "
                    f"{position} (vs avg position {avg_pos:.0f})"
                )

        # Vibe score effect
        vibe_score = context.get("vibe_score")
        if vibe_score is not None:
            avg_vibe = context.get("avg_vibe_score", self._DEFAULTS["avg_vibe_score"])
            if vibe_score > avg_vibe + 0.15:
                explanations.append(
                    f"Entity's high vibe score ({vibe_score:.2f}) "
                    f"strongly contributed to the {action}"
                )
            elif vibe_score < avg_vibe - 0.15:
                explanations.append(
                    f"User would NOT have {action}ed if entity had lower vibe score"
                )

        # Category affinity
        user_preferred = context.get("user_preferred_categories", [])
        entity_category = context.get("category")
        if entity_category and entity_category in user_preferred:
            explanations.append(f"User has strong affinity for '{entity_category}' category")

        # Was-recommended counterfactual
        was_recommended = context.get("was_recommended", True)
        if was_recommended and action in ("click", "convert"):
            explanations.append(
                "Without the recommendation, the user likely would NOT have "
                f"discovered entity {entity_id}"
            )

        if not explanations:
            explanations.append(
                f"User {action}ed entity {entity_id} -- " f"no strong causal factor identified"
            )

        explanation = "; ".join(explanations)
        logger.info(
            "Counterfactual explanation for user=%s entity=%s: %s",
            user_id,
            entity_id,
            explanation,
        )
        return explanation

    def what_would_happen(
        self,
        user_id: str,
        hypothetical_changes: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict what would happen under hypothetical changes.

        Simulates the effect of changing recommendation parameters (position,
        vibe score, category) on expected click and conversion probabilities.

        Args:
            user_id: The user to simulate for.
            hypothetical_changes: Dict of parameter changes, e.g.
                ``{"position": 1, "vibe_score": 0.9}``.

        Returns:
            Dict with predicted ``click_probability``, ``conversion_probability``,
            and a list of ``effects`` describing each change.
        """
        base_click = self._DEFAULTS["avg_click_rate"]
        base_convert = self._DEFAULTS["avg_conversion_rate"]
        effects: list[str] = []

        click_multiplier = 1.0
        convert_multiplier = 1.0

        # Position change
        new_position = hypothetical_changes.get("position")
        if new_position is not None:
            avg_pos = self._DEFAULTS["avg_position"]
            # Higher position (lower number) → more clicks
            position_effect = (avg_pos - new_position) / avg_pos
            click_multiplier += position_effect * 0.5
            effects.append(
                f"Moving to position {new_position} would "
                f"{'increase' if position_effect > 0 else 'decrease'} "
                f"click rate by ~{abs(position_effect * 50):.0f}%"
            )

        # Vibe score change
        new_vibe = hypothetical_changes.get("vibe_score")
        if new_vibe is not None:
            avg_vibe = self._DEFAULTS["avg_vibe_score"]
            vibe_effect = (new_vibe - avg_vibe) / max(avg_vibe, 0.01)
            click_multiplier += vibe_effect * 0.3
            convert_multiplier += vibe_effect * 0.4
            effects.append(
                f"Vibe score of {new_vibe:.2f} would "
                f"{'increase' if vibe_effect > 0 else 'decrease'} "
                f"conversion rate by ~{abs(vibe_effect * 40):.0f}%"
            )

        # Category match change
        new_category = hypothetical_changes.get("category")
        user_prefs = hypothetical_changes.get("user_preferred_categories", [])
        if new_category and new_category in user_prefs:
            click_multiplier += 0.25
            convert_multiplier += 0.35
            effects.append(
                f"Category '{new_category}' matches user preference — "
                f"expected +25% clicks, +35% conversions"
            )
        elif new_category:
            effects.append(
                f"Category '{new_category}' does not match user preference — "
                f"minimal impact expected"
            )

        predicted_click = min(max(base_click * click_multiplier, 0.0), 1.0)
        predicted_convert = min(max(base_convert * convert_multiplier, 0.0), 1.0)

        result = {
            "user_id": user_id,
            "hypothetical_changes": hypothetical_changes,
            "click_probability": round(predicted_click, 4),
            "conversion_probability": round(predicted_convert, 4),
            "effects": effects,
            "baseline_click_rate": base_click,
            "baseline_conversion_rate": base_convert,
        }

        logger.info(
            "Counterfactual prediction for user=%s: click=%.4f convert=%.4f",
            user_id,
            predicted_click,
            predicted_convert,
        )
        return result


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class CausalInferenceEngine:
    """Top-level orchestrator that wires together the causal graph, uplift
    estimator, and counterfactual explainer for ARBOR's discovery platform.
    """

    def __init__(self) -> None:
        self._graph: CausalGraph | None = None
        self._estimator = UpliftEstimator()
        self._explainer = CounterfactualExplainer()
        logger.info("CausalInferenceEngine initialised")

    # -- Graph construction -------------------------------------------------

    def build_graph(self) -> CausalGraph:
        """Build (or return cached) pre-configured causal graph.

        Returns:
            The ARBOR platform causal graph.
        """
        if self._graph is None:
            self._graph = build_arbor_causal_graph()
        return self._graph

    # -- Effect estimation --------------------------------------------------

    def estimate_recommendation_effect(
        self,
        entity_id: str,
        interactions: list[dict[str, Any]],
    ) -> CausalEffect:
        """Estimate the causal effect of recommending a specific entity.

        Splits interactions into treatment (entity was recommended and shown)
        and control (entity was not recommended) groups, then estimates the
        ATE on conversion.

        Args:
            entity_id: The entity whose recommendation effect to estimate.
            interactions: Full interaction log, each dict containing at least
                ``entity_id``, ``was_recommended`` (bool), and ``action``.

        Returns:
            A :class:`CausalEffect` describing the recommendation uplift.
        """
        treatment_outcomes: list[float] = []
        control_outcomes: list[float] = []

        for interaction in interactions:
            outcome = 1.0 if interaction.get("action") == "convert" else 0.0
            if interaction.get("entity_id") == entity_id and interaction.get(
                "was_recommended", False
            ):
                treatment_outcomes.append(outcome)
            elif interaction.get("entity_id") == entity_id:
                control_outcomes.append(outcome)

        # If we have no control group for this entity, use interactions with
        # other entities as a proxy baseline.
        if not control_outcomes:
            control_outcomes = [
                1.0 if i.get("action") == "convert" else 0.0
                for i in interactions
                if i.get("entity_id") != entity_id
            ]

        if not treatment_outcomes:
            logger.warning(
                "No treatment interactions for entity %s; returning zero effect",
                entity_id,
            )
            return CausalEffect(
                treatment="recommendation",
                outcome="conversion",
                ate=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                sample_size=0,
            )

        return self._estimator.estimate_uplift(
            treatment_group=treatment_outcomes,
            control_group=control_outcomes,
            outcome_metric="conversion",
        )

    # -- Explanation --------------------------------------------------------

    def explain_conversion(
        self,
        user_id: str,
        entity_id: str,
        context: dict[str, Any],
    ) -> str:
        """Generate a human-readable causal explanation for a conversion.

        Args:
            user_id: The user who converted.
            entity_id: The entity they converted on.
            context: Contextual information (position, vibe_score, category,
                user preferences, etc.).

        Returns:
            An explanation string.
        """
        return self._explainer.explain(
            user_id=user_id,
            entity_id=entity_id,
            action="convert",
            context=context,
        )

    # -- Actionable insights ------------------------------------------------

    def get_actionable_insights(
        self,
        entity_data: dict[str, Any],
        interactions: list[dict[str, Any]],
    ) -> list[str]:
        """Derive actionable business insights from causal analysis.

        Examines entity performance data and interaction patterns to produce
        a set of plain-English recommendations for platform operators.

        Args:
            entity_data: Dict with entity metadata (``entity_id``, ``name``,
                ``category``, ``vibe_score``, ``avg_position``, etc.).
            interactions: Interaction log for this entity.

        Returns:
            A list of insight strings.
        """
        insights: list[str] = []
        entity_id = entity_data.get("entity_id", "unknown")
        entity_name = entity_data.get("name", entity_id)

        # -- Uplift insight
        if interactions:
            uplift = self._estimator.estimate_per_entity(entity_id, interactions)
            if uplift > 0.05:
                insights.append(
                    f"'{entity_name}' has a strong positive causal uplift of "
                    f"{uplift:.1%} on conversions — consider featuring it more "
                    f"prominently."
                )
            elif uplift < -0.02:
                insights.append(
                    f"'{entity_name}' shows negative uplift ({uplift:.1%}) — "
                    f"recommendations may not be driving conversions for this entity."
                )

        # -- Position insight
        avg_position = entity_data.get("avg_position")
        vibe_score = entity_data.get("vibe_score")
        if avg_position is not None and avg_position > 5:
            insights.append(
                f"'{entity_name}' is typically shown at position {avg_position:.0f}. "
                f"Moving it higher could increase click-through by ~"
                f"{min((avg_position - 1) * 10, 80):.0f}%."
            )

        # -- Vibe quality insight
        if vibe_score is not None:
            if vibe_score > 0.85:
                insights.append(
                    f"'{entity_name}' has an exceptional vibe score ({vibe_score:.2f}) "
                    f"— it performs well regardless of recommendation position."
                )
            elif vibe_score < 0.4:
                insights.append(
                    f"'{entity_name}' has a low vibe score ({vibe_score:.2f}) — "
                    f"improving entity data quality could boost conversions."
                )

        # -- Interaction volume insight
        entity_interactions = [i for i in interactions if i.get("entity_id") == entity_id]
        total = len(entity_interactions)
        if total < 10:
            insights.append(
                f"'{entity_name}' has only {total} interactions — "
                f"causal estimates are uncertain. More data needed."
            )
        elif total > 100:
            convert_count = sum(1 for i in entity_interactions if i.get("action") == "convert")
            rate = convert_count / total
            if rate > 0.15:
                insights.append(
                    f"'{entity_name}' has a high conversion rate ({rate:.1%}) "
                    f"across {total} interactions — a strong performer."
                )

        if not insights:
            insights.append(
                f"No strong causal signals detected for '{entity_name}'. " f"Continue monitoring."
            )

        logger.info(
            "Generated %d actionable insights for entity %s",
            len(insights),
            entity_id,
        )
        return insights


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: CausalInferenceEngine | None = None


def get_causal_engine() -> CausalInferenceEngine:
    """Return the singleton :class:`CausalInferenceEngine` instance.

    The engine is created on first call and reused thereafter.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CausalInferenceEngine()
    return _engine_instance
