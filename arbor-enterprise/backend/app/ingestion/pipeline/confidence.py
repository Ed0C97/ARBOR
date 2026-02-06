"""Confidence Scoring and Disagreement Detection — Layer 4.

Analyzes per-dimension confidence based on:
1. Number of sources that contributed data for each dimension
2. Agreement between sources (low spread = high confidence)
3. Quality of source data (reviews > inferred, photos > text)

Also detects disagreements and flags items for curator review.
"""

import logging
import statistics

from app.ingestion.pipeline.schemas import (
    DimensionName,
    DimensionScore,
    FactSheet,
    ReviewQueueItem,
    ReviewStatus,
    ScoredVibeDNA,
)

logger = logging.getLogger(__name__)

# Disagreement threshold: if source scores differ by more than this, flag it
DISAGREEMENT_THRESHOLD = 25

# Minimum confidence threshold: below this, flag for review
MIN_CONFIDENCE_THRESHOLD = 0.35

# Priority weights for review queue
PRIORITY_WEIGHTS = {
    "is_featured": 3.0,
    "low_confidence": 2.0,
    "disagreement": 2.5,
    "few_sources": 1.5,
    "high_traffic": 1.0,
}


class ConfidenceAnalyzer:
    """Analyze and enrich confidence scores for scored Vibe DNA."""

    def analyze(
        self,
        scored: ScoredVibeDNA,
        fact_sheet: FactSheet,
        is_featured: bool = False,
    ) -> ScoredVibeDNA:
        """Enrich scored Vibe DNA with detailed confidence analysis.

        Modifies the scored object in place and returns it.
        """
        # Compute per-dimension confidence adjustments
        for dim in scored.dimensions:
            self._analyze_dimension_confidence(dim, fact_sheet)

        # Detect disagreements
        disagreements = self._detect_disagreements(scored)

        # Update overall confidence
        confidences = [d.confidence for d in scored.dimensions]
        scored.overall_confidence = statistics.mean(confidences) if confidences else 0.0

        # Determine if review is needed
        review_reasons = list(scored.review_reasons)  # Keep existing reasons

        if disagreements:
            dim_names = [d for d in disagreements]
            review_reasons.append(f"Source disagreement on: {', '.join(dim_names)}")

        low_conf_dims = [
            d.dimension for d in scored.dimensions if d.confidence < MIN_CONFIDENCE_THRESHOLD
        ]
        if low_conf_dims:
            review_reasons.append(f"Low confidence on: {', '.join(low_conf_dims)}")

        if scored.sources_count < 2:
            review_reasons.append("Insufficient data sources (only 1)")

        if is_featured and scored.overall_confidence < 0.6:
            review_reasons.append("Featured entity with below-average confidence")

        scored.needs_review = len(review_reasons) > 0
        scored.review_reasons = review_reasons

        return scored

    def create_review_item(
        self,
        scored: ScoredVibeDNA,
        fact_sheet: FactSheet,
        entity_type: str,
        source_id: int,
        name: str,
        category: str,
        is_featured: bool = False,
    ) -> ReviewQueueItem | None:
        """Create a review queue item if the entity needs review.

        Returns None if no review is needed.
        """
        if not scored.needs_review:
            return None

        # Calculate priority
        priority = 0.0
        if is_featured:
            priority += PRIORITY_WEIGHTS["is_featured"]

        low_conf_count = sum(
            1 for d in scored.dimensions if d.confidence < MIN_CONFIDENCE_THRESHOLD
        )
        priority += low_conf_count * PRIORITY_WEIGHTS["low_confidence"] * 0.3

        disagreement_count = sum(1 for d in scored.dimensions if d.has_disagreement)
        priority += disagreement_count * PRIORITY_WEIGHTS["disagreement"] * 0.4

        if scored.sources_count < 2:
            priority += PRIORITY_WEIGHTS["few_sources"]

        return ReviewQueueItem(
            entity_id=scored.entity_id,
            entity_type=entity_type,
            source_id=source_id,
            name=name,
            category=category,
            scored_vibe=scored,
            fact_sheet=fact_sheet,
            reasons=scored.review_reasons,
            priority=round(priority, 2),
            status=ReviewStatus.NEEDS_REVIEW,
        )

    def _analyze_dimension_confidence(self, dim: DimensionScore, fact_sheet: FactSheet) -> None:
        """Adjust dimension confidence based on available evidence."""
        # Source coverage bonus
        source_count = len(fact_sheet.sources_used)
        source_bonus = min(source_count * 0.1, 0.3)

        # Fact density bonus: more facts = more data = higher confidence
        relevant_facts = self._count_relevant_facts(dim.dimension, fact_sheet)
        fact_bonus = min(relevant_facts * 0.05, 0.2)

        # Adjust confidence
        base_confidence = dim.confidence
        adjusted = min(1.0, base_confidence + source_bonus + fact_bonus)

        # Penalty for single source
        if source_count <= 1:
            adjusted *= 0.7

        dim.confidence = round(adjusted, 3)

    def _detect_disagreements(self, scored: ScoredVibeDNA) -> list[str]:
        """Detect dimensions where sources disagree significantly."""
        disagreements = []
        for dim in scored.dimensions:
            if len(dim.source_scores) < 2:
                continue

            values = list(dim.source_scores.values())
            spread = max(values) - min(values)
            dim.spread = spread

            if spread > DISAGREEMENT_THRESHOLD:
                dim.has_disagreement = True
                disagreements.append(dim.dimension)
                # Reduce confidence when sources disagree
                dim.confidence = max(0.1, dim.confidence * 0.6)

        return disagreements

    def _count_relevant_facts(self, dimension: str, fact_sheet: FactSheet) -> int:
        """Count facts relevant to a specific dimension."""
        relevance_map = {
            "formality": ["interior_element", "lighting", "furniture_style", "brand_signal"],
            "craftsmanship": ["material", "material_visible", "specialty"],
            "price_value": ["price_point", "price_tier", "price_tier_estimated"],
            "atmosphere": ["lighting", "layout", "decor_element", "crowd_density"],
            "service_quality": ["service_indicator"],
            "exclusivity": ["brand_presentation", "neighborhood_type", "featured"],
            "modernity": ["furniture_style", "decor_element"],
        }
        relevant_types = set(relevance_map.get(dimension, []))
        count = 0
        for fact in fact_sheet.all_facts():
            if fact.fact_type in relevant_types:
                count += 1
        return count
