"""Recommendation explainability module for ARBOR Enterprise.

Generates human-readable explanations for why entities were recommended
to a user.  Each recommendation is decomposed into weighted explanation
factors (style match, category match, location, vibe similarity,
popularity, personalization, graph connection) and a natural-language
summary is synthesised from the top-contributing factors.

Usage::

    engine = get_explainability_engine()
    explanation = engine.explain_recommendation(
        entity_data={"name": "Cafe Moya", "category": "cafe", ...},
        query="minimalist cafes in Milan",
        user_profile=user_profile_dict,
        search_scores={"text_score": 0.82, "rerank_score": 0.91},
        rank=1,
    )
    print(explanation.summary)
    # "Recommended because it matches your interest in minimalist style
    #  and is located in Milan"
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Factor type constants
# ---------------------------------------------------------------------------

FACTOR_STYLE_MATCH = "style_match"
FACTOR_CATEGORY_MATCH = "category_match"
FACTOR_LOCATION = "location"
FACTOR_VIBE_SIMILARITY = "vibe_similarity"
FACTOR_POPULARITY = "popularity"
FACTOR_PERSONALIZATION = "personalization"
FACTOR_GRAPH_CONNECTION = "graph_connection"

VALID_FACTOR_TYPES = {
    FACTOR_STYLE_MATCH,
    FACTOR_CATEGORY_MATCH,
    FACTOR_LOCATION,
    FACTOR_VIBE_SIMILARITY,
    FACTOR_POPULARITY,
    FACTOR_PERSONALIZATION,
    FACTOR_GRAPH_CONNECTION,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExplanationFactor:
    """A single contributing factor in a recommendation explanation.

    Attributes:
        factor_type: One of the recognised factor types (e.g.
            ``"style_match"``, ``"category_match"``, ``"location"``,
            ``"vibe_similarity"``, ``"popularity"``,
            ``"personalization"``, ``"graph_connection"``).
        weight: Importance of this factor in the range [0.0, 1.0].
        description: Human-readable description of this factor's
            contribution.
        details: Arbitrary metadata about how the factor was computed.
    """

    factor_type: str
    weight: float
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(1.0, self.weight))


@dataclass
class RecommendationExplanation:
    """Complete explanation for a single recommended entity.

    Attributes:
        entity_id: Identifier of the recommended entity.
        entity_name: Display name of the entity.
        overall_score: Combined recommendation score in [0.0, 1.0].
        rank: 1-based position in the recommendation list.
        factors: Ordered list of explanation factors, sorted by weight
            descending.
        summary: One-line human-readable explanation.
        confidence: Confidence in the explanation's accuracy, derived
            from the number and strength of contributing factors.
    """

    entity_id: str
    entity_name: str
    overall_score: float
    rank: int
    factors: list[ExplanationFactor] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Vibe intent keywords
# ---------------------------------------------------------------------------

# Mapping of vibe dimension names to representative query keywords.
# Used to infer which vibe dimensions the user is interested in based
# on the words that appear in their search query.
_VIBE_KEYWORDS: dict[str, list[str]] = {
    "minimalist": ["minimalist", "minimal", "simple", "clean", "understated"],
    "luxurious": ["luxury", "luxurious", "premium", "high-end", "upscale", "exclusive"],
    "cozy": ["cozy", "cosy", "warm", "intimate", "homey", "snug"],
    "modern": ["modern", "contemporary", "sleek", "cutting-edge", "innovative"],
    "vintage": ["vintage", "retro", "classic", "old-school", "antique", "heritage"],
    "artistic": ["artistic", "creative", "artsy", "art", "gallery", "design"],
    "bohemian": ["bohemian", "boho", "eclectic", "free-spirited"],
    "industrial": ["industrial", "raw", "warehouse", "loft", "urban"],
    "natural": ["natural", "organic", "eco", "green", "sustainable", "earthy"],
    "playful": ["playful", "fun", "quirky", "whimsical", "colorful"],
}


# ---------------------------------------------------------------------------
# Explainability engine
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """Generates human-readable explanations for entity recommendations.

    Analyses the relationship between a search query, user profile, and
    entity data to identify the factors that contributed most to a
    recommendation.  Each factor is scored and described, and a
    natural-language summary is produced.

    Usage::

        engine = get_explainability_engine()
        explanation = engine.explain_recommendation(entity_data, query, ...)
    """

    def __init__(self) -> None:
        logger.info("ExplainabilityEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_recommendation(
        self,
        entity_data: dict[str, Any],
        query: str,
        user_profile: Optional[dict[str, Any]] = None,
        search_scores: Optional[dict[str, float]] = None,
        rank: int = 1,
    ) -> RecommendationExplanation:
        """Generate a full explanation for why *entity_data* was recommended.

        Analyses query-entity text relevance, vibe alignment, category
        match, geographic relevance, personalization boost, and graph
        connections.  Returns a :class:`RecommendationExplanation` with
        all contributing factors and a natural-language summary.

        Args:
            entity_data: Dict describing the entity.  Expected keys
                include ``id``, ``name``, ``category``, ``styles``,
                ``city``, ``vibe_dna``, ``description``,
                ``popularity_score``, ``relationships``.
            query: The user's original search query string.
            user_profile: Optional dict of learned user preferences
                (style_preferences, category_affinities, city_preferences).
            search_scores: Optional dict of raw search/rerank scores
                (e.g. ``text_score``, ``rerank_score``).
            rank: 1-based position in the result list.

        Returns:
            A :class:`RecommendationExplanation` instance.
        """
        search_scores = search_scores or {}
        entity_id = str(entity_data.get("id", "unknown"))
        entity_name = str(entity_data.get("name", "Unknown Entity"))

        factors: list[ExplanationFactor] = []

        # 1. Query-entity text relevance
        text_factor = self._compute_text_relevance(query, entity_data)
        if text_factor.weight > 0.0:
            factors.append(text_factor)

        # 2. Vibe DNA alignment with query intent
        vibe_factor = self._compute_vibe_alignment(query, entity_data)
        if vibe_factor.weight > 0.0:
            factors.append(vibe_factor)

        # 3. Category match
        category_factor = self._compute_category_match(query, entity_data)
        if category_factor.weight > 0.0:
            factors.append(category_factor)

        # 4. Geographic relevance
        geo_factor = self._compute_geo_relevance(query, entity_data)
        if geo_factor.weight > 0.0:
            factors.append(geo_factor)

        # 5. Personalization boost
        personalization_factor = self._compute_personalization_factor(user_profile, entity_data)
        if personalization_factor is not None:
            factors.append(personalization_factor)

        # 6. Graph connections
        graph_factor = self._compute_graph_connections(entity_data)
        if graph_factor is not None:
            factors.append(graph_factor)

        # 7. Popularity signal
        popularity_factor = self._compute_popularity(entity_data)
        if popularity_factor is not None:
            factors.append(popularity_factor)

        # Sort factors by weight descending
        factors.sort(key=lambda f: f.weight, reverse=True)

        # Overall score: use rerank_score > text_score > weighted factor avg
        overall_score = search_scores.get(
            "rerank_score",
            search_scores.get("text_score", 0.0),
        )
        if overall_score == 0.0 and factors:
            total_weight = sum(f.weight for f in factors)
            overall_score = total_weight / len(factors) if factors else 0.0
        overall_score = max(0.0, min(1.0, overall_score))

        # Confidence: based on factor count and strength
        confidence = self._compute_confidence(factors)

        # Summary
        summary = self._generate_summary(factors, entity_name)

        explanation = RecommendationExplanation(
            entity_id=entity_id,
            entity_name=entity_name,
            overall_score=round(overall_score, 4),
            rank=rank,
            factors=factors,
            summary=summary,
            confidence=round(confidence, 4),
        )

        logger.debug(
            "Explanation generated: entity=%s rank=%d factors=%d confidence=%.4f",
            entity_id,
            rank,
            len(factors),
            confidence,
        )

        return explanation

    def explain_batch(
        self,
        entities: list[dict[str, Any]],
        query: str,
        user_profile: Optional[dict[str, Any]] = None,
        search_scores: Optional[list[dict[str, float]]] = None,
    ) -> list[RecommendationExplanation]:
        """Generate explanations for all entities in a search response.

        This is a convenience wrapper that calls
        :meth:`explain_recommendation` for each entity, assigning the
        appropriate rank and per-entity search scores.

        Args:
            entities: List of entity dicts to explain.
            query: The user's original search query.
            user_profile: Optional dict of learned user preferences.
            search_scores: Optional list of score dicts, one per entity,
                aligned by index.  If ``None``, empty score dicts are
                used.

        Returns:
            List of :class:`RecommendationExplanation` objects, one per
            entity, in the same order as *entities*.
        """
        explanations: list[RecommendationExplanation] = []
        scores_list = search_scores or [{}] * len(entities)

        for idx, entity_data in enumerate(entities):
            entity_scores = scores_list[idx] if idx < len(scores_list) else {}
            explanation = self.explain_recommendation(
                entity_data=entity_data,
                query=query,
                user_profile=user_profile,
                search_scores=entity_scores,
                rank=idx + 1,
            )
            explanations.append(explanation)

        logger.info(
            "Batch explanation complete: %d entities explained for query='%s'",
            len(explanations),
            query[:80],
        )

        return explanations

    # ------------------------------------------------------------------
    # Factor computation helpers
    # ------------------------------------------------------------------

    def _compute_text_relevance(
        self,
        query: str,
        entity_data: dict[str, Any],
    ) -> ExplanationFactor:
        """Measure keyword overlap between query and entity text fields.

        Tokenises the query and several entity text fields (name,
        description, category, styles, city) and computes a normalised
        overlap score.

        Args:
            query: The search query.
            entity_data: Entity dict.

        Returns:
            An :class:`ExplanationFactor` of type ``style_match`` (text
            relevance is surfaced as a style match when keywords align
            with entity attributes).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return ExplanationFactor(
                factor_type=FACTOR_STYLE_MATCH,
                weight=0.0,
                description="No query keywords to match.",
                details={"query_tokens": [], "matched_tokens": []},
            )

        # Build a bag of words from the entity's searchable fields
        entity_text_parts: list[str] = []
        for text_field in ("name", "description", "category", "city"):
            value = entity_data.get(text_field, "")
            if isinstance(value, str):
                entity_text_parts.append(value)

        styles = entity_data.get("styles", [])
        if isinstance(styles, list):
            entity_text_parts.extend(str(s) for s in styles)

        entity_tokens = self._tokenize(" ".join(entity_text_parts))

        if not entity_tokens:
            return ExplanationFactor(
                factor_type=FACTOR_STYLE_MATCH,
                weight=0.0,
                description="Entity has no searchable text.",
                details={"query_tokens": list(query_tokens), "matched_tokens": []},
            )

        matched = query_tokens & entity_tokens
        weight = len(matched) / len(query_tokens) if query_tokens else 0.0

        matched_list = sorted(matched)
        description = (
            f"Query keywords {matched_list} found in entity attributes"
            if matched_list
            else "No direct keyword matches"
        )

        return ExplanationFactor(
            factor_type=FACTOR_STYLE_MATCH,
            weight=round(weight, 4),
            description=description,
            details={
                "query_tokens": sorted(query_tokens),
                "entity_tokens_sample": sorted(entity_tokens)[:20],
                "matched_tokens": matched_list,
                "overlap_ratio": round(weight, 4),
            },
        )

    def _compute_vibe_alignment(
        self,
        query: str,
        entity_data: dict[str, Any],
    ) -> ExplanationFactor:
        """Measure alignment between inferred query vibe intent and entity vibe DNA.

        Maps query keywords to vibe dimensions, then computes cosine
        similarity between the inferred query vibe vector and the
        entity's ``vibe_dna`` vector.

        Args:
            query: The search query.
            entity_data: Entity dict with an optional ``vibe_dna`` field.

        Returns:
            An :class:`ExplanationFactor` of type ``vibe_similarity``.
        """
        entity_vibe = entity_data.get("vibe_dna")
        if not entity_vibe or not isinstance(entity_vibe, (list, dict)):
            return ExplanationFactor(
                factor_type=FACTOR_VIBE_SIMILARITY,
                weight=0.0,
                description="Entity has no vibe DNA data.",
                details={},
            )

        # Infer query vibe intent from keywords
        query_lower = query.lower()
        query_intent: dict[str, float] = {}
        matched_vibes: list[str] = []

        for dimension, keywords in _VIBE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    query_intent[dimension] = 1.0
                    matched_vibes.append(dimension)
                    break

        if not query_intent:
            return ExplanationFactor(
                factor_type=FACTOR_VIBE_SIMILARITY,
                weight=0.0,
                description="No vibe intent detected in query.",
                details={"query": query},
            )

        # Compute alignment
        if isinstance(entity_vibe, dict):
            # Dict-based vibe_dna: compare shared dimensions
            alignment_scores: list[float] = []
            for dimension, intent_weight in query_intent.items():
                entity_dim_score = float(entity_vibe.get(dimension, 0.0))
                alignment_scores.append(entity_dim_score * intent_weight)

            weight = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
        else:
            # List-based vibe_dna: build query vector in same order and
            # compute cosine similarity
            dimensions = sorted(_VIBE_KEYWORDS.keys())
            query_vector = [query_intent.get(d, 0.0) for d in dimensions]

            # If entity vector length matches our dimension count, compare
            if len(entity_vibe) == len(dimensions):
                weight = self._cosine_similarity(query_vector, entity_vibe)
            else:
                # Fallback: average of matched vibe dimensions (positional)
                weight = sum(
                    float(entity_vibe[i])
                    for i, d in enumerate(dimensions)
                    if d in query_intent and i < len(entity_vibe)
                ) / len(query_intent)

        weight = max(0.0, min(1.0, weight))

        description = f"Entity vibe aligns with query intent: {', '.join(matched_vibes)}"

        return ExplanationFactor(
            factor_type=FACTOR_VIBE_SIMILARITY,
            weight=round(weight, 4),
            description=description,
            details={
                "matched_vibes": matched_vibes,
                "alignment_score": round(weight, 4),
            },
        )

    def _compute_category_match(
        self,
        query: str,
        entity_data: dict[str, Any],
    ) -> ExplanationFactor:
        """Check whether the entity's category appears in the query.

        Performs case-insensitive matching and also handles common
        pluralisation (e.g. ``"cafes"`` matching ``"cafe"``).

        Args:
            query: The search query.
            entity_data: Entity dict with an optional ``category`` field.

        Returns:
            An :class:`ExplanationFactor` of type ``category_match``.
        """
        category = entity_data.get("category", "")
        if not category or not isinstance(category, str):
            return ExplanationFactor(
                factor_type=FACTOR_CATEGORY_MATCH,
                weight=0.0,
                description="Entity has no category.",
                details={},
            )

        query_lower = query.lower()
        category_lower = category.lower().strip()

        # Direct match or plural match
        is_match = (
            category_lower in query_lower
            or f"{category_lower}s" in query_lower
            or f"{category_lower}es" in query_lower
            or (category_lower.endswith("s") and category_lower[:-1] in query_lower)
        )

        if is_match:
            weight = 0.9
            description = f"Entity category '{category}' matches the query"
        else:
            weight = 0.0
            description = f"Entity category '{category}' does not appear in the query"

        return ExplanationFactor(
            factor_type=FACTOR_CATEGORY_MATCH,
            weight=weight,
            description=description,
            details={
                "entity_category": category,
                "query": query,
                "is_match": is_match,
            },
        )

    def _compute_geo_relevance(
        self,
        query: str,
        entity_data: dict[str, Any],
    ) -> ExplanationFactor:
        """Determine whether the entity is geographically relevant to the query.

        Checks if the entity's ``city`` or ``country`` appears in the
        query string.

        Args:
            query: The search query.
            entity_data: Entity dict with optional ``city`` and
                ``country`` fields.

        Returns:
            An :class:`ExplanationFactor` of type ``location``.
        """
        city = entity_data.get("city", "")
        country = entity_data.get("country", "")
        neighborhood = entity_data.get("neighborhood", "")

        query_lower = query.lower()
        matched_locations: list[str] = []

        for location_value in (city, country, neighborhood):
            if location_value and isinstance(location_value, str):
                if location_value.lower().strip() in query_lower:
                    matched_locations.append(location_value.strip())

        if matched_locations:
            weight = min(1.0, 0.85 + 0.05 * (len(matched_locations) - 1))
            description = f"Located in {', '.join(matched_locations)}, mentioned in the query"
        else:
            weight = 0.0
            location_parts = [
                loc for loc in (city, neighborhood, country) if loc and isinstance(loc, str)
            ]
            location_str = ", ".join(location_parts) if location_parts else "unknown"
            description = f"Entity location ({location_str}) not mentioned in the query"

        return ExplanationFactor(
            factor_type=FACTOR_LOCATION,
            weight=round(weight, 4),
            description=description,
            details={
                "entity_city": city,
                "entity_country": country,
                "entity_neighborhood": neighborhood,
                "matched_locations": matched_locations,
            },
        )

    def _compute_personalization_factor(
        self,
        user_profile: Optional[dict[str, Any]],
        entity_data: dict[str, Any],
    ) -> Optional[ExplanationFactor]:
        """Compute personalization contribution if a user profile exists.

        Checks style preference overlap, category affinity, and city
        preference from the user's learned profile against the entity's
        attributes.

        Args:
            user_profile: Dict of user preferences (or ``None``).
            entity_data: Entity dict.

        Returns:
            An :class:`ExplanationFactor` of type ``personalization``,
            or ``None`` if no user profile is provided or if the profile
            has no interactions yet.
        """
        if user_profile is None:
            return None

        interaction_count = user_profile.get("interaction_count", 0)
        if interaction_count == 0:
            return None

        style_prefs: dict[str, float] = user_profile.get("style_preferences", {})
        category_affinities: dict[str, float] = user_profile.get("category_affinities", {})
        city_prefs: dict[str, float] = user_profile.get("city_preferences", {})

        # Style overlap
        entity_styles = entity_data.get("styles", [])
        style_scores: list[float] = []
        matched_styles: list[str] = []
        if isinstance(entity_styles, list):
            for style in entity_styles:
                pref = style_prefs.get(str(style), 0.0)
                if pref > 0.0:
                    style_scores.append(pref)
                    matched_styles.append(str(style))

        style_avg = sum(style_scores) / len(style_scores) if style_scores else 0.0

        # Category affinity
        entity_category = entity_data.get("category", "")
        category_score = category_affinities.get(entity_category, 0.0)

        # City preference
        entity_city = entity_data.get("city", "")
        city_score = city_prefs.get(entity_city, 0.0)

        # Weighted combination (mirrors PersonalizationEngine._compute_boost)
        weight = 0.5 * style_avg + 0.3 * category_score + 0.2 * city_score
        weight = max(0.0, min(1.0, weight))

        if weight == 0.0:
            return None

        # Confidence scaling: more interactions = more reliable profile
        profile_confidence = min(1.0, interaction_count / 20.0)
        weight *= profile_confidence

        description_parts: list[str] = []
        if matched_styles:
            description_parts.append(
                f"matches your preference for {', '.join(matched_styles)} style"
            )
        if category_score > 0.0:
            description_parts.append(f"you frequently explore {entity_category} entities")
        if city_score > 0.0:
            description_parts.append(f"you have shown interest in {entity_city}")

        description = (
            "Personalised: " + "; ".join(description_parts)
            if description_parts
            else "Personalised based on your past interactions"
        )

        return ExplanationFactor(
            factor_type=FACTOR_PERSONALIZATION,
            weight=round(weight, 4),
            description=description,
            details={
                "matched_styles": matched_styles,
                "style_score": round(style_avg, 4),
                "category_score": round(category_score, 4),
                "city_score": round(city_score, 4),
                "interaction_count": interaction_count,
                "profile_confidence": round(profile_confidence, 4),
            },
        )

    def _compute_graph_connections(
        self,
        entity_data: dict[str, Any],
    ) -> Optional[ExplanationFactor]:
        """Assess whether graph relationships contributed to the recommendation.

        Checks the entity's ``relationships`` field for connections that
        may have boosted its ranking (e.g. ``SIMILAR_VIBE``, ``NEAR``,
        ``SELLS_AT``).

        Args:
            entity_data: Entity dict with an optional ``relationships``
                field (list of dicts with ``type`` and ``target_name``).

        Returns:
            An :class:`ExplanationFactor` of type ``graph_connection``,
            or ``None`` if the entity has no relationships.
        """
        relationships = entity_data.get("relationships", [])
        if not relationships or not isinstance(relationships, list):
            return None

        rel_types: list[str] = []
        connected_names: list[str] = []
        for rel in relationships:
            if isinstance(rel, dict):
                rel_type = rel.get("type", rel.get("rel_type", ""))
                target = rel.get("target_name", rel.get("target", ""))
                if rel_type:
                    rel_types.append(str(rel_type))
                if target:
                    connected_names.append(str(target))

        if not rel_types:
            return None

        # Weight increases with more connections, capped at 1.0
        weight = min(1.0, 0.3 + 0.1 * len(rel_types))

        # Unique relationship types for description
        unique_types = sorted(set(rel_types))
        connected_sample = connected_names[:3]

        if connected_sample:
            description = (
                f"Connected to {', '.join(connected_sample)} " f"via {', '.join(unique_types)}"
            )
        else:
            description = (
                f"Has {len(rel_types)} graph connection(s) " f"({', '.join(unique_types)})"
            )

        return ExplanationFactor(
            factor_type=FACTOR_GRAPH_CONNECTION,
            weight=round(weight, 4),
            description=description,
            details={
                "relationship_count": len(rel_types),
                "relationship_types": unique_types,
                "connected_entities": connected_names[:5],
            },
        )

    def _compute_popularity(
        self,
        entity_data: dict[str, Any],
    ) -> Optional[ExplanationFactor]:
        """Generate a popularity factor if the entity has a popularity score.

        Args:
            entity_data: Entity dict with an optional
                ``popularity_score`` field (float in [0, 1]).

        Returns:
            An :class:`ExplanationFactor` of type ``popularity``, or
            ``None`` if no popularity score is present.
        """
        popularity = entity_data.get("popularity_score")
        if popularity is None:
            return None

        try:
            popularity = float(popularity)
        except (TypeError, ValueError):
            return None

        popularity = max(0.0, min(1.0, popularity))

        if popularity < 0.3:
            return None

        # Weight scaled modestly so popularity alone doesn't dominate
        weight = popularity * 0.6

        if popularity >= 0.8:
            tier = "highly popular"
        elif popularity >= 0.5:
            tier = "popular"
        else:
            tier = "moderately popular"

        description = f"This entity is {tier} in the community"

        return ExplanationFactor(
            factor_type=FACTOR_POPULARITY,
            weight=round(weight, 4),
            description=description,
            details={
                "popularity_score": round(popularity, 4),
                "tier": tier,
            },
        )

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        factors: list[ExplanationFactor],
        entity_name: str,
    ) -> str:
        """Generate a one-line natural-language summary from the top factors.

        Selects up to three highest-weighted factors and constructs a
        human-readable sentence explaining why the entity was recommended.

        Args:
            factors: List of explanation factors (should already be sorted
                by weight descending).
            entity_name: Display name of the entity.

        Returns:
            A natural-language summary string.
        """
        if not factors:
            return f"{entity_name} was included in results."

        # Take top 3 factors with weight > 0
        top_factors = [f for f in factors if f.weight > 0.0][:3]

        if not top_factors:
            return f"{entity_name} was included in results."

        phrases: list[str] = []
        for factor in top_factors:
            phrase = self._factor_to_phrase(factor)
            if phrase:
                phrases.append(phrase)

        if not phrases:
            return f"{entity_name} was included in results."

        # Join phrases naturally
        if len(phrases) == 1:
            reason_text = phrases[0]
        elif len(phrases) == 2:
            reason_text = f"{phrases[0]} and {phrases[1]}"
        else:
            reason_text = f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"

        return f"Recommended because {reason_text}"

    @staticmethod
    def _factor_to_phrase(factor: ExplanationFactor) -> str:
        """Convert a single factor into a short phrase for the summary.

        Args:
            factor: The explanation factor to convert.

        Returns:
            A short descriptive phrase, or an empty string if no
            suitable phrase can be generated.
        """
        details = factor.details

        if factor.factor_type == FACTOR_STYLE_MATCH:
            matched = details.get("matched_tokens", [])
            if matched:
                return f"it matches your interest in {', '.join(matched[:3])}"
            return "it matches your search terms"

        if factor.factor_type == FACTOR_VIBE_SIMILARITY:
            vibes = details.get("matched_vibes", [])
            if vibes:
                return f"its vibe aligns with {', '.join(vibes[:2])}"
            return "its vibe aligns with your query"

        if factor.factor_type == FACTOR_CATEGORY_MATCH:
            category = details.get("entity_category", "")
            if category:
                return f"it is a {category}"
            return "its category matches your search"

        if factor.factor_type == FACTOR_LOCATION:
            locations = details.get("matched_locations", [])
            if locations:
                return f"it is located in {', '.join(locations)}"
            return "its location is relevant"

        if factor.factor_type == FACTOR_PERSONALIZATION:
            matched_styles = details.get("matched_styles", [])
            if matched_styles:
                return f"it matches your preference for " f"{', '.join(matched_styles[:2])} style"
            return "it aligns with your past preferences"

        if factor.factor_type == FACTOR_GRAPH_CONNECTION:
            connected = details.get("connected_entities", [])
            if connected:
                return f"it is connected to {connected[0]}"
            return "it has relevant connections in the knowledge graph"

        if factor.factor_type == FACTOR_POPULARITY:
            tier = details.get("tier", "popular")
            return f"it is {tier}"

        return ""

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(factors: list[ExplanationFactor]) -> float:
        """Derive explanation confidence from factor count and strength.

        More factors and higher weights increase confidence.  The
        formula uses a saturating function so that having many weak
        factors does not produce unrealistically high confidence.

        Args:
            factors: List of explanation factors.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if not factors:
            return 0.0

        non_zero = [f for f in factors if f.weight > 0.0]
        if not non_zero:
            return 0.0

        avg_weight = sum(f.weight for f in non_zero) / len(non_zero)
        count_signal = 1.0 - math.exp(-0.5 * len(non_zero))

        confidence = 0.6 * avg_weight + 0.4 * count_signal
        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenise text into a set of lowercase alphabetic tokens.

        Strips punctuation and filters out tokens shorter than 2
        characters (common stop-word fragments).

        Args:
            text: Input text.

        Returns:
            Set of unique lowercase tokens.
        """
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return {t for t in tokens if len(t) >= 2}

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector (must be same length as *a*).

        Returns:
            Cosine similarity in [-1, 1].  Returns 0.0 when either
            vector has zero magnitude.
        """
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    """Return the singleton ExplainabilityEngine instance."""
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine
