"""Core data structures for the 5-layer enrichment pipeline.

Defines FactSheet, ScoredVibeDNA, ConfidenceScore, and GoldStandard types
used throughout the pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Data sources for fact extraction."""

    DATABASE = "database"
    GOOGLE_REVIEWS = "google_reviews"
    GOOGLE_PHOTOS = "google_photos"
    INSTAGRAM = "instagram"
    WEBSITE = "website"
    STREET_VIEW = "street_view"
    MENU = "menu"
    CURATOR = "curator"


class ReviewStatus(str, Enum):
    """Curator review status for enrichments."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class DimensionName(str, Enum):
    """Canonical Vibe DNA dimension names."""

    FORMALITY = "formality"
    CRAFTSMANSHIP = "craftsmanship"
    PRICE_VALUE = "price_value"
    ATMOSPHERE = "atmosphere"
    SERVICE_QUALITY = "service_quality"
    EXCLUSIVITY = "exclusivity"
    MODERNITY = "modernity"

    @classmethod
    def all_names(cls) -> list[str]:
        return [d.value for d in cls]


# ---------------------------------------------------------------------------
# Layer 1: Source data
# ---------------------------------------------------------------------------


@dataclass
class SourceData:
    """Raw data collected from a single source."""

    source_type: SourceType
    raw_text: str = ""
    images: list[str] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CollectedSources:
    """All data collected for a single entity across sources."""

    entity_id: str
    entity_type: str
    source_id: int
    name: str
    category: str
    city: str | None = None
    sources: list[SourceData] = field(default_factory=list)

    def get_source(self, source_type: SourceType) -> SourceData | None:
        for s in self.sources:
            if s.source_type == source_type:
                return s
        return None


# ---------------------------------------------------------------------------
# Layer 2: Fact sheets
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFact:
    """A single objective fact extracted from a source."""

    fact_type: str  # e.g. "material", "price_point", "interior_element"
    value: str  # e.g. "Italian leather", "$250", "exposed brick"
    source: SourceType = SourceType.DATABASE
    confidence: float = 1.0  # 0.0 - 1.0


@dataclass
class FactSheet:
    """Comprehensive fact sheet for an entity, aggregated from all analyzers.

    Contains objective, verifiable facts — NOT scores. Scores are derived
    in Layer 3 by the calibrated scoring engine.
    """

    entity_id: str
    name: str
    category: str

    # Objective facts grouped by category
    materials: list[ExtractedFact] = field(default_factory=list)
    price_points: list[ExtractedFact] = field(default_factory=list)
    interior_elements: list[ExtractedFact] = field(default_factory=list)
    service_indicators: list[ExtractedFact] = field(default_factory=list)
    audience_indicators: list[ExtractedFact] = field(default_factory=list)
    brand_signals: list[ExtractedFact] = field(default_factory=list)
    location_context: list[ExtractedFact] = field(default_factory=list)

    # Visual analysis output
    visual_style: str = ""
    visual_tags: list[str] = field(default_factory=list)
    visual_summary: str = ""

    # Menu / product analysis
    avg_price: float | None = None
    price_range_label: str = ""  # "$", "$$", "$$$", "$$$$"
    signature_items: list[str] = field(default_factory=list)

    # Text analysis summary
    review_sentiment: float = 0.0  # -1.0 to 1.0
    review_count: int = 0
    common_themes: list[str] = field(default_factory=list)

    # Sources used
    sources_used: list[SourceType] = field(default_factory=list)

    def all_facts(self) -> list[ExtractedFact]:
        """Return all facts flattened."""
        return (
            self.materials
            + self.price_points
            + self.interior_elements
            + self.service_indicators
            + self.audience_indicators
            + self.brand_signals
            + self.location_context
        )

    def to_scoring_text(self) -> str:
        """Convert fact sheet to a text block for the scoring LLM."""
        lines = [
            f"Entity: {self.name}",
            f"Category: {self.category}",
        ]
        if self.materials:
            lines.append(f"Materials: {', '.join(f.value for f in self.materials)}")
        if self.price_points:
            lines.append(f"Price points: {', '.join(f.value for f in self.price_points)}")
        if self.avg_price is not None:
            lines.append(f"Average price: ${self.avg_price:.0f}")
        if self.price_range_label:
            lines.append(f"Price tier: {self.price_range_label}")
        if self.interior_elements:
            lines.append(f"Interior: {', '.join(f.value for f in self.interior_elements)}")
        if self.service_indicators:
            lines.append(f"Service: {', '.join(f.value for f in self.service_indicators)}")
        if self.audience_indicators:
            lines.append(f"Audience: {', '.join(f.value for f in self.audience_indicators)}")
        if self.brand_signals:
            lines.append(f"Brand: {', '.join(f.value for f in self.brand_signals)}")
        if self.location_context:
            lines.append(f"Location: {', '.join(f.value for f in self.location_context)}")
        if self.visual_tags:
            lines.append(f"Visual style: {', '.join(self.visual_tags)}")
        if self.visual_summary:
            lines.append(f"Visual summary: {self.visual_summary}")
        if self.common_themes:
            lines.append(f"Review themes: {', '.join(self.common_themes)}")
        if self.signature_items:
            lines.append(f"Signature items: {', '.join(self.signature_items)}")
        if self.review_count > 0:
            lines.append(
                f"Review sentiment: {self.review_sentiment:.2f} ({self.review_count} reviews)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 3: Scored Vibe DNA with confidence
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """A single dimension score with per-source breakdown and confidence."""

    dimension: str
    score: int  # 0-100
    confidence: float  # 0.0 - 1.0

    # Per-source scores that were aggregated
    source_scores: dict[str, int] = field(default_factory=dict)
    # Disagreement flag: True if sources strongly disagree
    has_disagreement: bool = False
    # Spread of source scores (std dev)
    spread: float = 0.0


@dataclass
class ScoredVibeDNA:
    """Complete Vibe DNA with confidence and provenance for every dimension."""

    entity_id: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    signature_items: list[str] = field(default_factory=list)
    target_audience: str = "General"
    visual_style: str = ""
    summary: str = ""

    # Overall quality metrics
    overall_confidence: float = 0.0
    sources_count: int = 0
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    # Calibration metadata
    calibrated: bool = False
    calibration_reference_ids: list[str] = field(default_factory=list)

    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_vibe_dna_dict(self) -> dict:
        """Convert to the vibe_dna JSONB format stored in ArborEnrichment."""
        return {
            "dimensions": {d.dimension: d.score for d in self.dimensions},
            "confidence": {d.dimension: round(d.confidence, 3) for d in self.dimensions},
            "disagreements": {
                d.dimension: d.has_disagreement for d in self.dimensions if d.has_disagreement
            },
            "tags": self.tags,
            "signature_items": self.signature_items,
            "target_audience": self.target_audience,
            "visual_style": self.visual_style,
            "summary": self.summary,
            "overall_confidence": round(self.overall_confidence, 3),
            "sources_count": self.sources_count,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "calibrated": self.calibrated,
            "scored_at": self.scored_at.isoformat(),
        }

    def get_dimension(self, name: str) -> DimensionScore | None:
        for d in self.dimensions:
            if d.dimension == name:
                return d
        return None


# ---------------------------------------------------------------------------
# Layer 3: Gold Standard
# ---------------------------------------------------------------------------


@dataclass
class GoldStandardEntity:
    """A manually curated reference entity for calibration.

    Curators assign ground-truth scores that the scoring engine
    uses as few-shot examples and calibration anchors.
    """

    entity_id: str
    entity_type: str
    source_id: int
    name: str
    category: str
    city: str | None = None

    # Curator-assigned ground truth scores (0-100 per dimension)
    ground_truth_scores: dict[str, int] = field(default_factory=dict)
    # Curator-assigned tags
    ground_truth_tags: list[str] = field(default_factory=list)
    # Curator notes explaining the scores
    curator_notes: str = ""
    # Who curated it
    curated_by: str = ""
    curated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_few_shot_example(self, fact_sheet_text: str) -> str:
        """Format as a few-shot example for the scoring LLM."""
        scores_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.ground_truth_scores.items()))
        tags_str = ", ".join(self.ground_truth_tags[:10])
        return (
            f"### Example: {self.name} ({self.category})\n"
            f"FACTS:\n{fact_sheet_text}\n\n"
            f"SCORES: {scores_str}\n"
            f"TAGS: {tags_str}\n"
            f"REASONING: {self.curator_notes}\n"
        )


# ---------------------------------------------------------------------------
# Layer 4: Curator review queue item
# ---------------------------------------------------------------------------


@dataclass
class ReviewQueueItem:
    """An enrichment that needs human review."""

    entity_id: str
    entity_type: str
    source_id: int
    name: str
    category: str

    # The scored vibe DNA awaiting review
    scored_vibe: ScoredVibeDNA | None = None
    fact_sheet: FactSheet | None = None

    # Why it needs review
    reasons: list[str] = field(default_factory=list)
    priority: float = 0.0  # Higher = more urgent

    # Review state
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    reviewer_notes: str = ""

    # Curator overrides (if any)
    overridden_scores: dict[str, int] = field(default_factory=dict)
    overridden_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 5: Drift detection
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Report on scoring drift compared to the gold standard."""

    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    gold_standard_count: int = 0
    avg_mae: float = 0.0  # Mean Absolute Error vs gold standard
    per_dimension_mae: dict[str, float] = field(default_factory=dict)
    drifted_dimensions: list[str] = field(default_factory=list)
    needs_recalibration: bool = False
