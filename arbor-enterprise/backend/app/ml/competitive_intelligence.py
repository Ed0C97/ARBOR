"""Competitive Intelligence Engine for analysing the competitive landscape.

Provides tools for identifying competitors, segmenting markets, computing
market position scores, and generating SWOT analyses for fashion brands and
venues tracked by the ARBOR platform.

Usage::

    engine = get_competitive_engine()
    analysis = engine.analyze_competitors(
        entity_id="e_101",
        entity_data={"name": "Café Omotesandō", "category": "cafe",
                     "city": "Tokyo", "vibe_dna": {"minimalist": 0.9, "cozy": 0.4}},
        all_entities=[...],
        top_k=10,
    )
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _vibe_similarity(dna1: dict[str, float], dna2: dict[str, float]) -> float:
    """Compute cosine similarity between two vibe-DNA dictionaries.

    Each dictionary maps dimension names (e.g. ``"minimalist"``,
    ``"bohemian"``) to float weights.  Only dimensions present in *both*
    dictionaries contribute to the dot product; dimensions unique to one
    side still contribute to that vector's magnitude.

    Args:
        dna1: First vibe-DNA mapping.
        dna2: Second vibe-DNA mapping.

    Returns:
        A float in [0.0, 1.0].  Returns 0.0 if either vector has zero
        magnitude.
    """
    all_keys = set(dna1.keys()) | set(dna2.keys())
    if not all_keys:
        return 0.0

    dot = 0.0
    mag1 = 0.0
    mag2 = 0.0
    for key in all_keys:
        v1 = dna1.get(key, 0.0)
        v2 = dna2.get(key, 0.0)
        dot += v1 * v2
        mag1 += v1 * v1
        mag2 += v2 * v2

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    similarity = dot / (math.sqrt(mag1) * math.sqrt(mag2))
    return max(0.0, min(1.0, similarity))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CompetitorProfile:
    """Profile summarising a single competitor entity.

    Attributes:
        entity_id: Unique identifier of the competitor entity.
        name: Human-readable name.
        category: Entity category (e.g. ``"cafe"``, ``"boutique"``).
        city: City where the entity is located.
        vibe_dna: Mapping of vibe dimensions to float weights.
        tags: Descriptive tags associated with the entity.
        strengths: Vibe dimensions where the competitor outperforms the
            reference entity.
        weaknesses: Vibe dimensions where the competitor underperforms.
        market_position: Qualitative label (``"leader"``, ``"challenger"``,
            ``"follower"``, ``"niche"``).
        similarity_score: Cosine similarity of vibe DNA to the reference
            entity (0.0 - 1.0).
    """

    entity_id: str
    name: str
    category: str
    city: str
    vibe_dna: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    market_position: str = "follower"
    similarity_score: float = 0.0


@dataclass
class MarketSegment:
    """Describes a distinct market segment derived from entity clustering.

    Attributes:
        segment_id: Unique identifier for the segment.
        name: Short human-readable segment name.
        description: Longer narrative description.
        category: Entity category that defines the segment.
        geographic_scope: Geographic level (e.g. a city or country name).
        entity_count: Number of entities belonging to this segment.
        avg_vibe_dna: Averaged vibe-DNA profile across segment members.
        dominant_styles: Most frequently occurring style tags in the segment.
        growth_trend: One of ``"growing"``, ``"stable"``, or ``"declining"``.
    """

    segment_id: str
    name: str
    description: str
    category: str
    geographic_scope: str
    entity_count: int = 0
    avg_vibe_dna: dict[str, float] = field(default_factory=dict)
    dominant_styles: list[str] = field(default_factory=list)
    growth_trend: str = "stable"


@dataclass
class CompetitiveAnalysis:
    """Complete competitive analysis report for a single entity.

    Attributes:
        entity_id: Identifier of the entity being analysed.
        entity_name: Human-readable name of the entity.
        competitors: Ranked list of competitor profiles.
        market_segments: Relevant market segments the entity participates in.
        market_position_score: Numeric position score (0 - 100).
        unique_differentiators: Vibe dimensions or tags that set the entity
            apart from its competitors.
        overlap_areas: Dimensions where the entity closely matches many
            competitors (high crowding risk).
        opportunities: Strategic opportunities identified by gap analysis.
        threats: Competitive threats the entity faces.
        generated_at: Timestamp when the analysis was produced.
    """

    entity_id: str
    entity_name: str
    competitors: list[CompetitorProfile] = field(default_factory=list)
    market_segments: list[MarketSegment] = field(default_factory=list)
    market_position_score: float = 0.0
    unique_differentiators: list[str] = field(default_factory=list)
    overlap_areas: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)
    threats: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Competitive Intelligence Engine
# ---------------------------------------------------------------------------


class CompetitiveIntelligenceEngine:
    """Orchestrates competitive analysis across the ARBOR entity graph.

    Provides methods for competitor identification, market segmentation,
    gap analysis, market-position scoring, and SWOT generation.

    Usage::

        engine = get_competitive_engine()
        analysis = engine.analyze_competitors("e_101", entity_data, all_entities)
    """

    def __init__(self) -> None:
        logger.info("CompetitiveIntelligenceEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_competitors(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        all_entities: list[dict[str, Any]],
        top_k: int = 10,
    ) -> CompetitiveAnalysis:
        """Produce a full competitive analysis for the given entity.

        The method:

        1. Filters ``all_entities`` by overlapping category and city to
           find potential competitors.
        2. Ranks candidates by vibe-DNA cosine similarity.
        3. Identifies per-competitor strengths and weaknesses.
        4. Builds market segments and computes the entity's market position.
        5. Runs gap analysis and SWOT generation.

        Args:
            entity_id: Unique identifier for the entity being analysed.
            entity_data: Dict describing the entity.  Expected keys:
                ``name`` (str), ``category`` (str), ``city`` (str),
                ``vibe_dna`` (dict[str, float]), ``tags`` (list[str]).
            all_entities: Complete list of entity dicts in the platform.
            top_k: Maximum number of competitors to return.

        Returns:
            A :class:`CompetitiveAnalysis` instance.
        """
        entity_name: str = entity_data.get("name", "Unknown")
        entity_category: str = entity_data.get("category", "")
        entity_city: str = entity_data.get("city", "")
        entity_vibe: dict[str, float] = entity_data.get("vibe_dna", {})
        entity_tags: list[str] = entity_data.get("tags", [])

        # 1. Find candidate competitors (same category OR same city)
        candidates: list[dict[str, Any]] = []
        for ent in all_entities:
            eid = ent.get("entity_id", ent.get("id", ""))
            if eid == entity_id:
                continue
            same_category = ent.get("category", "") == entity_category
            same_city = ent.get("city", "") == entity_city
            if same_category or same_city:
                candidates.append(ent)

        # 2. Rank by vibe-DNA similarity
        scored_candidates: list[tuple[float, dict[str, Any]]] = []
        for cand in candidates:
            cand_vibe = cand.get("vibe_dna", {})
            sim = _vibe_similarity(entity_vibe, cand_vibe)
            scored_candidates.append((sim, cand))

        scored_candidates.sort(key=lambda t: t[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # 3. Build CompetitorProfile objects with strengths / weaknesses
        competitors: list[CompetitorProfile] = []
        for sim, cand in top_candidates:
            cand_vibe = cand.get("vibe_dna", {})
            strengths, weaknesses = self._compare_vibe_dimensions(entity_vibe, cand_vibe)
            position = self._classify_market_position(sim, cand, candidates)
            profile = CompetitorProfile(
                entity_id=cand.get("entity_id", cand.get("id", "")),
                name=cand.get("name", "Unknown"),
                category=cand.get("category", ""),
                city=cand.get("city", ""),
                vibe_dna=cand_vibe,
                tags=cand.get("tags", []),
                strengths=strengths,
                weaknesses=weaknesses,
                market_position=position,
                similarity_score=round(sim, 4),
            )
            competitors.append(profile)

        # 4. Build market segments
        segments = self.identify_market_segments(all_entities)

        # 5. Market position score
        position_score = self.compute_market_position(entity_data, competitors)

        # 6. Differentiators and overlap
        unique_differentiators = self._find_differentiators(entity_vibe, entity_tags, competitors)
        overlap_areas = self._find_overlap_areas(entity_vibe, competitors)

        # 7. Gap analysis and SWOT
        opportunities = self.find_market_gaps(entity_data, segments)
        swot = self.generate_swot(entity_data, competitors, segments)

        analysis = CompetitiveAnalysis(
            entity_id=entity_id,
            entity_name=entity_name,
            competitors=competitors,
            market_segments=segments,
            market_position_score=round(position_score, 2),
            unique_differentiators=unique_differentiators,
            overlap_areas=overlap_areas,
            opportunities=opportunities,
            threats=swot.get("threats", []),
            generated_at=datetime.now(UTC),
        )

        logger.info(
            "Competitive analysis complete: entity=%s competitors=%d segments=%d "
            "position_score=%.1f",
            entity_id,
            len(competitors),
            len(segments),
            position_score,
        )
        return analysis

    # ------------------------------------------------------------------
    # Market segmentation
    # ------------------------------------------------------------------

    def identify_market_segments(
        self,
        entities: list[dict[str, Any]],
        min_segment_size: int = 3,
    ) -> list[MarketSegment]:
        """Group entities into market segments by category and city.

        Each unique (category, city) pair with at least *min_segment_size*
        entities forms a segment.  The segment's average vibe DNA and
        dominant styles are computed from its members.

        Args:
            entities: List of entity dicts.
            min_segment_size: Minimum number of entities for a group to be
                considered a segment.

        Returns:
            List of :class:`MarketSegment` instances.
        """
        # Group entities by (category, city)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for ent in entities:
            key = (ent.get("category", "unknown"), ent.get("city", "unknown"))
            groups.setdefault(key, []).append(ent)

        segments: list[MarketSegment] = []
        for (category, city), members in groups.items():
            if len(members) < min_segment_size:
                continue

            avg_dna = self._average_vibe_dna(members)
            dominant = self._dominant_styles(members)
            growth = self._estimate_growth_trend(members)

            seg = MarketSegment(
                segment_id=f"seg_{category}_{city}".lower().replace(" ", "_"),
                name=f"{category.title()} in {city.title()}",
                description=(
                    f"Market segment for {category} entities located in "
                    f"{city}, comprising {len(members)} entities."
                ),
                category=category,
                geographic_scope=city,
                entity_count=len(members),
                avg_vibe_dna=avg_dna,
                dominant_styles=dominant,
                growth_trend=growth,
            )
            segments.append(seg)

        logger.debug(
            "Identified %d market segments from %d entities (min_size=%d)",
            len(segments),
            len(entities),
            min_segment_size,
        )
        return segments

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def find_market_gaps(
        self,
        entity_data: dict[str, Any],
        segments: list[MarketSegment],
    ) -> list[str]:
        """Identify underserved niches that the entity could exploit.

        A gap is flagged when:

        * A category-city combination has very few entities (small segment
          or no segment at all).
        * The entity's vibe DNA has strong dimensions that no existing
          segment emphasises.

        Args:
            entity_data: Dict describing the entity under analysis.
            segments: Pre-computed market segments.

        Returns:
            A list of human-readable opportunity strings.
        """
        entity_category: str = entity_data.get("category", "")
        entity_city: str = entity_data.get("city", "")
        entity_vibe: dict[str, float] = entity_data.get("vibe_dna", {})

        gaps: list[str] = []

        # 1. Look for low-density segments in the entity's category
        category_segments = [s for s in segments if s.category == entity_category]
        if not category_segments:
            gaps.append(
                f"No established segment for category '{entity_category}' — "
                f"opportunity to define the market."
            )
        else:
            small = [s for s in category_segments if s.entity_count < 5]
            for seg in small:
                gaps.append(
                    f"Underserved segment '{seg.name}' with only "
                    f"{seg.entity_count} entities — room to grow."
                )

        # 2. Vibe dimensions the entity excels in but segments lack
        for dim, weight in entity_vibe.items():
            if weight < 0.5:
                continue
            dim_present_in_segments = any(seg.avg_vibe_dna.get(dim, 0.0) >= 0.3 for seg in segments)
            if not dim_present_in_segments:
                gaps.append(
                    f"Strong '{dim}' vibe (weight={weight:.2f}) has no dominant "
                    f"segment — potential niche differentiator."
                )

        # 3. Geographic gap: entity's city is underrepresented
        city_segments = [s for s in segments if s.geographic_scope == entity_city]
        if not city_segments:
            gaps.append(
                f"City '{entity_city}' has no established market segments — "
                f"first-mover advantage possible."
            )

        logger.debug("Found %d market gaps for entity", len(gaps))
        return gaps

    # ------------------------------------------------------------------
    # Market position scoring
    # ------------------------------------------------------------------

    def compute_market_position(
        self,
        entity_data: dict[str, Any],
        competitors: list[CompetitorProfile],
    ) -> float:
        """Score the entity's overall market position on a 0 - 100 scale.

        The score is a weighted combination of three factors:

        * **Distinctiveness** (40%) — how different the entity's vibe DNA
          is from the average of its competitors.  Higher distinctiveness
          earns a higher score.
        * **Competitive density** (30%) — fewer direct competitors (high
          similarity) is better.  Many close competitors drag the score
          down.
        * **Quality signal** (30%) — derived from optional ``rating`` or
          ``quality_score`` fields in *entity_data*.

        Args:
            entity_data: Dict describing the entity.
            competitors: List of competitor profiles.

        Returns:
            A float in [0.0, 100.0].
        """
        if not competitors:
            return 75.0  # No competitors → strong default position

        entity_vibe: dict[str, float] = entity_data.get("vibe_dna", {})

        # --- Distinctiveness (40%) ---
        avg_competitor_dna = self._average_vibe_dna([{"vibe_dna": c.vibe_dna} for c in competitors])
        avg_similarity = _vibe_similarity(entity_vibe, avg_competitor_dna)
        # Lower similarity → more distinctive → higher score
        distinctiveness_score = (1.0 - avg_similarity) * 100.0

        # --- Competitive density (30%) ---
        close_competitors = sum(1 for c in competitors if c.similarity_score >= 0.7)
        # Fewer close competitors is better; cap at 10 for normalisation
        density_score = max(0.0, 100.0 - (close_competitors / max(len(competitors), 1)) * 100.0)

        # --- Quality signal (30%) ---
        quality_raw = float(entity_data.get("quality_score", entity_data.get("rating", 3.0)))
        # Normalise to 0-100 assuming a 1-5 rating scale
        quality_score = max(0.0, min(100.0, (quality_raw - 1.0) / 4.0 * 100.0))

        position = 0.40 * distinctiveness_score + 0.30 * density_score + 0.30 * quality_score

        logger.debug(
            "Market position: distinctiveness=%.1f density=%.1f quality=%.1f " "=> position=%.1f",
            distinctiveness_score,
            density_score,
            quality_score,
            position,
        )
        return max(0.0, min(100.0, position))

    # ------------------------------------------------------------------
    # SWOT generation
    # ------------------------------------------------------------------

    def generate_swot(
        self,
        entity_data: dict[str, Any],
        competitors: list[CompetitorProfile],
        segments: list[MarketSegment],
    ) -> dict[str, list[str]]:
        """Generate a SWOT analysis for the entity.

        Args:
            entity_data: Dict describing the entity.
            competitors: List of competitor profiles.
            segments: List of market segments.

        Returns:
            A dict with keys ``"strengths"``, ``"weaknesses"``,
            ``"opportunities"``, ``"threats"``, each mapping to a list of
            human-readable strings.
        """
        entity_vibe: dict[str, float] = entity_data.get("vibe_dna", {})
        entity_category: str = entity_data.get("category", "")
        entity_tags: list[str] = entity_data.get("tags", [])

        strengths: list[str] = []
        weaknesses: list[str] = []
        opportunities: list[str] = []
        threats: list[str] = []

        # --- Strengths ---
        # Vibe dimensions where the entity significantly exceeds competitors
        if competitors:
            avg_comp_dna = self._average_vibe_dna([{"vibe_dna": c.vibe_dna} for c in competitors])
            for dim, weight in entity_vibe.items():
                comp_weight = avg_comp_dna.get(dim, 0.0)
                if weight - comp_weight >= 0.2:
                    strengths.append(
                        f"Strong '{dim}' vibe ({weight:.2f} vs competitor avg "
                        f"{comp_weight:.2f})"
                    )
        # Unique tags
        competitor_tags: set[str] = set()
        for c in competitors:
            competitor_tags.update(c.tags)
        unique_tags = [t for t in entity_tags if t not in competitor_tags]
        if unique_tags:
            strengths.append(f"Unique positioning tags: {', '.join(unique_tags[:5])}")

        # --- Weaknesses ---
        if competitors:
            for dim, comp_weight in avg_comp_dna.items():
                entity_weight = entity_vibe.get(dim, 0.0)
                if comp_weight - entity_weight >= 0.2:
                    weaknesses.append(
                        f"Lagging in '{dim}' vibe ({entity_weight:.2f} vs "
                        f"competitor avg {comp_weight:.2f})"
                    )

        if not entity_vibe:
            weaknesses.append("No vibe DNA profile — limits discoverability")

        # --- Opportunities ---
        opportunities.extend(self.find_market_gaps(entity_data, segments))

        # Growing segments in the entity's category
        for seg in segments:
            if seg.category == entity_category and seg.growth_trend == "growing":
                opportunities.append(f"Growing segment '{seg.name}' — align to capture demand")

        # --- Threats ---
        close_count = sum(1 for c in competitors if c.similarity_score >= 0.8)
        if close_count >= 3:
            threats.append(
                f"{close_count} competitors with >=80% vibe similarity — " f"high substitution risk"
            )

        leaders = [c for c in competitors if c.market_position == "leader"]
        if leaders:
            leader_names = ", ".join(c.name for c in leaders[:3])
            threats.append(f"Market leaders present: {leader_names}")

        declining_segments = [
            s for s in segments if s.category == entity_category and s.growth_trend == "declining"
        ]
        for seg in declining_segments:
            threats.append(f"Declining segment '{seg.name}' — demand erosion risk")

        swot = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats,
        }

        logger.debug(
            "SWOT generated: S=%d W=%d O=%d T=%d",
            len(strengths),
            len(weaknesses),
            len(opportunities),
            len(threats),
        )
        return swot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_vibe_dimensions(
        reference_dna: dict[str, float],
        competitor_dna: dict[str, float],
        threshold: float = 0.15,
    ) -> tuple[list[str], list[str]]:
        """Identify dimensions where the competitor is stronger or weaker.

        A dimension is a *strength* of the competitor if the competitor's
        weight exceeds the reference by at least *threshold*, and a
        *weakness* if it falls short by the same margin.

        Args:
            reference_dna: Vibe-DNA of the entity being analysed.
            competitor_dna: Vibe-DNA of the competitor.
            threshold: Minimum absolute difference to flag.

        Returns:
            Tuple of (strengths, weaknesses) as lists of dimension names.
        """
        all_dims = set(reference_dna.keys()) | set(competitor_dna.keys())
        strengths: list[str] = []
        weaknesses: list[str] = []

        for dim in all_dims:
            ref_val = reference_dna.get(dim, 0.0)
            comp_val = competitor_dna.get(dim, 0.0)
            diff = comp_val - ref_val
            if diff >= threshold:
                strengths.append(dim)
            elif diff <= -threshold:
                weaknesses.append(dim)

        return strengths, weaknesses

    @staticmethod
    def _classify_market_position(
        similarity: float,
        candidate: dict[str, Any],
        all_candidates: list[dict[str, Any]],
    ) -> str:
        """Assign a qualitative market-position label to a competitor.

        Heuristic:

        * ``"leader"`` — high quality score (>= 4.0) and many tags.
        * ``"challenger"`` — moderate quality and above-average tag count.
        * ``"niche"`` — low similarity (< 0.4) to the reference entity,
          indicating a differentiated positioning.
        * ``"follower"`` — everyone else.

        Args:
            similarity: Cosine similarity to the reference entity.
            candidate: Entity dict for the competitor.
            all_candidates: All competitor candidates (for relative stats).

        Returns:
            One of ``"leader"``, ``"challenger"``, ``"niche"``, ``"follower"``.
        """
        quality = float(candidate.get("quality_score", candidate.get("rating", 3.0)))
        tag_count = len(candidate.get("tags", []))
        avg_tags = sum(len(c.get("tags", [])) for c in all_candidates) / max(len(all_candidates), 1)

        if quality >= 4.0 and tag_count >= avg_tags:
            return "leader"
        if quality >= 3.5 and tag_count >= avg_tags * 0.8:
            return "challenger"
        if similarity < 0.4:
            return "niche"
        return "follower"

    @staticmethod
    def _average_vibe_dna(
        entities: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute the element-wise mean vibe-DNA across a list of entities.

        Args:
            entities: List of entity dicts, each expected to have a
                ``vibe_dna`` key.

        Returns:
            A dict mapping each encountered dimension to its mean weight.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        for ent in entities:
            dna = ent.get("vibe_dna", {})
            for dim, val in dna.items():
                totals[dim] = totals.get(dim, 0.0) + val
                counts[dim] = counts.get(dim, 0) + 1

        avg: dict[str, float] = {}
        for dim in totals:
            avg[dim] = round(totals[dim] / counts[dim], 4)
        return avg

    @staticmethod
    def _dominant_styles(
        entities: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[str]:
        """Return the most common style tags across a group of entities.

        Args:
            entities: List of entity dicts with optional ``tags`` or
                ``styles`` keys.
            top_n: Number of top styles to return.

        Returns:
            List of style tag strings, most common first.
        """
        counter: Counter[str] = Counter()
        for ent in entities:
            tags = ent.get("tags", ent.get("styles", []))
            counter.update(tags)
        return [tag for tag, _ in counter.most_common(top_n)]

    @staticmethod
    def _estimate_growth_trend(
        entities: list[dict[str, Any]],
    ) -> str:
        """Heuristically estimate whether a segment is growing, stable, or declining.

        Uses the ``created_at`` field (if present) to judge how many
        entities were added recently.  Falls back to ``"stable"`` when
        temporal data is unavailable.

        Args:
            entities: Segment member dicts.

        Returns:
            One of ``"growing"``, ``"stable"``, ``"declining"``.
        """
        now = datetime.now(UTC)
        recent_count = 0
        dated_count = 0

        for ent in entities:
            created_raw = ent.get("created_at")
            if created_raw is None:
                continue
            if isinstance(created_raw, str):
                try:
                    created = datetime.fromisoformat(created_raw)
                except ValueError:
                    continue
            elif isinstance(created_raw, datetime):
                created = created_raw
            else:
                continue

            dated_count += 1
            # Consider "recent" if created within the last 90 days
            if (now - created).days <= 90:
                recent_count += 1

        if dated_count == 0:
            return "stable"

        recent_ratio = recent_count / dated_count
        if recent_ratio >= 0.4:
            return "growing"
        if recent_ratio <= 0.1:
            return "declining"
        return "stable"

    @staticmethod
    def _find_differentiators(
        entity_vibe: dict[str, float],
        entity_tags: list[str],
        competitors: list[CompetitorProfile],
    ) -> list[str]:
        """Identify what makes the entity unique compared to competitors.

        A vibe dimension is a differentiator if the entity scores highly
        on it (>= 0.5) while the average competitor score is low (< 0.25).
        A tag is a differentiator if no competitor shares it.

        Args:
            entity_vibe: The entity's vibe-DNA mapping.
            entity_tags: The entity's tags.
            competitors: List of competitor profiles.

        Returns:
            List of differentiator description strings.
        """
        differentiators: list[str] = []

        if competitors:
            avg_comp: dict[str, float] = {}
            for c in competitors:
                for dim, val in c.vibe_dna.items():
                    avg_comp[dim] = avg_comp.get(dim, 0.0) + val
            for dim in avg_comp:
                avg_comp[dim] /= len(competitors)

            for dim, weight in entity_vibe.items():
                if weight >= 0.5 and avg_comp.get(dim, 0.0) < 0.25:
                    differentiators.append(f"Distinctive '{dim}' vibe")

        # Unique tags
        competitor_tags: set[str] = set()
        for c in competitors:
            competitor_tags.update(c.tags)
        for tag in entity_tags:
            if tag not in competitor_tags:
                differentiators.append(f"Unique tag: {tag}")

        return differentiators

    @staticmethod
    def _find_overlap_areas(
        entity_vibe: dict[str, float],
        competitors: list[CompetitorProfile],
    ) -> list[str]:
        """Find vibe dimensions where the entity and many competitors cluster.

        Overlap is flagged when the entity and a majority of competitors
        all score above 0.4 on the same dimension.

        Args:
            entity_vibe: The entity's vibe-DNA mapping.
            competitors: List of competitor profiles.

        Returns:
            List of overlap-area dimension names.
        """
        if not competitors:
            return []

        overlaps: list[str] = []
        threshold = 0.4
        majority = len(competitors) / 2.0

        for dim, weight in entity_vibe.items():
            if weight < threshold:
                continue
            matching = sum(1 for c in competitors if c.vibe_dna.get(dim, 0.0) >= threshold)
            if matching >= majority:
                overlaps.append(dim)

        return overlaps


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine: CompetitiveIntelligenceEngine | None = None


def get_competitive_engine() -> CompetitiveIntelligenceEngine:
    """Return the singleton CompetitiveIntelligenceEngine instance."""
    global _engine
    if _engine is None:
        _engine = CompetitiveIntelligenceEngine()
    return _engine
