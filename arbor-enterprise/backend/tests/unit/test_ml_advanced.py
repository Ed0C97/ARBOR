"""Unit tests for advanced ML modules: federated learning, competitive intelligence,
and graph auto-expansion.

Tests are designed to instantiate classes directly with mock/inline data and have
no external service dependencies (Neo4j, Qdrant, etc.).
"""

import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Federated Learning imports
# ---------------------------------------------------------------------------
from app.ml.federated_learning import (
    GradientAggregator,
    FederatedLearningCoordinator,
    DifferentialPrivacy,
    ModelUpdate,
    FederatedModel,
    AggregationStrategy,
)

# ---------------------------------------------------------------------------
# Competitive Intelligence imports
# ---------------------------------------------------------------------------
from app.ml.competitive_intelligence import (
    CompetitiveIntelligenceEngine,
    _vibe_similarity,
)

# ---------------------------------------------------------------------------
# Graph Expansion imports
# ---------------------------------------------------------------------------
from app.ml.graph_expansion import (
    _cosine_similarity,
    _haversine_km,
    RelationshipCandidate,
    AutoExpansionScheduler,
)


# =========================================================================
# TestGradientAggregator
# =========================================================================


class TestGradientAggregator:
    """Tests for the GradientAggregator class."""

    def setup_method(self):
        self.aggregator = GradientAggregator()

    def test_fed_avg_averages_correctly(self):
        """Two updates with equal sample counts produce a simple average."""
        updates = [
            ModelUpdate(
                tenant_id="t1",
                round_number=0,
                gradient_updates={"layer1": [2.0, 4.0]},
                sample_count=100,
            ),
            ModelUpdate(
                tenant_id="t2",
                round_number=0,
                gradient_updates={"layer1": [4.0, 6.0]},
                sample_count=100,
            ),
        ]
        result = self.aggregator.aggregate(updates, AggregationStrategy.FED_AVG)

        # With equal sample counts (100 each), each weight = 100/200 = 0.5
        # layer1[0] = 2.0*0.5 + 4.0*0.5 = 3.0
        # layer1[1] = 4.0*0.5 + 6.0*0.5 = 5.0
        assert "layer1" in result
        assert len(result["layer1"]) == 2
        assert result["layer1"][0] == pytest.approx(3.0)
        assert result["layer1"][1] == pytest.approx(5.0)

    def test_weighted_avg_respects_sample_count(self):
        """Larger sample count has more influence on the weighted average."""
        updates = [
            ModelUpdate(
                tenant_id="t1",
                round_number=0,
                gradient_updates={"layer1": [10.0]},
                sample_count=900,
            ),
            ModelUpdate(
                tenant_id="t2",
                round_number=0,
                gradient_updates={"layer1": [0.0]},
                sample_count=100,
            ),
        ]
        result = self.aggregator.aggregate(updates, AggregationStrategy.WEIGHTED_AVG)

        # t1 weight = 900/1000 = 0.9, t2 weight = 100/1000 = 0.1
        # layer1[0] = 10.0*0.9 + 0.0*0.1 = 9.0
        assert result["layer1"][0] == pytest.approx(9.0)

    def test_handles_single_update(self):
        """A single update returns its own gradients (weight = 1.0)."""
        updates = [
            ModelUpdate(
                tenant_id="t1",
                round_number=0,
                gradient_updates={"layer1": [1.5, -2.5, 3.0]},
                sample_count=50,
            ),
        ]
        result = self.aggregator.aggregate(updates, AggregationStrategy.FED_AVG)

        assert result["layer1"][0] == pytest.approx(1.5)
        assert result["layer1"][1] == pytest.approx(-2.5)
        assert result["layer1"][2] == pytest.approx(3.0)

    def test_handles_empty_updates(self):
        """An empty list of updates raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate an empty list"):
            self.aggregator.aggregate([], AggregationStrategy.FED_AVG)


# =========================================================================
# TestFederatedCoordinator
# =========================================================================


class TestFederatedCoordinator:
    """Tests for the FederatedLearningCoordinator class."""

    def setup_method(self):
        self.coordinator = FederatedLearningCoordinator()

    def test_register_model(self):
        """Model is created with initial weights and stored."""
        initial_weights = {"layer1": [0.0, 0.0, 0.0]}
        model = self.coordinator.register_model(
            model_id="test_model",
            model_type="recommendation",
            initial_weights=initial_weights,
        )

        assert isinstance(model, FederatedModel)
        assert model.model_id == "test_model"
        assert model.model_type == "recommendation"
        assert model.global_weights == {"layer1": [0.0, 0.0, 0.0]}
        assert model.version == 0

    def test_submit_update(self):
        """Update is stored for the registered model."""
        self.coordinator.register_model(
            model_id="m1",
            model_type="rec",
            initial_weights={"layer1": [0.0]},
        )
        accepted = self.coordinator.submit_update(
            model_id="m1",
            tenant_id="tenant_A",
            gradient_updates={"layer1": [0.1]},
            sample_count=100,
        )

        assert accepted is True
        status = self.coordinator.get_round_status("m1")
        assert status["pending_updates"] == 1

    def test_aggregation_round(self):
        """Aggregation produces a new version with updated weights."""
        self.coordinator.register_model(
            model_id="m1",
            model_type="rec",
            initial_weights={"layer1": [0.0, 0.0]},
        )
        self.coordinator.submit_update(
            model_id="m1",
            tenant_id="t1",
            gradient_updates={"layer1": [1.0, 2.0]},
            sample_count=100,
        )
        self.coordinator.submit_update(
            model_id="m1",
            tenant_id="t2",
            gradient_updates={"layer1": [3.0, 4.0]},
            sample_count=100,
        )

        model = self.coordinator.run_aggregation_round("m1")

        assert model is not None
        assert model.version == 1
        # Weights = initial + aggregated = [0,0] + [2,3] = [2,3]
        assert model.global_weights["layer1"][0] == pytest.approx(2.0)
        assert model.global_weights["layer1"][1] == pytest.approx(3.0)
        assert model.last_aggregated_at is not None

    def test_min_updates_threshold(self):
        """Aggregation is skipped if too few updates are pending."""
        self.coordinator.register_model(
            model_id="m1",
            model_type="rec",
            initial_weights={"layer1": [0.0]},
        )
        self.coordinator.submit_update(
            model_id="m1",
            tenant_id="t1",
            gradient_updates={"layer1": [1.0]},
            sample_count=100,
        )

        # Default min_updates is 2, but we only submitted 1
        result = self.coordinator.run_aggregation_round("m1", min_updates=2)

        assert result is None

    def test_get_model_for_tenant(self):
        """Returns a copy of global weights, not a reference."""
        self.coordinator.register_model(
            model_id="m1",
            model_type="rec",
            initial_weights={"layer1": [1.0, 2.0, 3.0]},
        )

        weights = self.coordinator.get_model_for_tenant("m1", "tenant_X")

        assert weights == {"layer1": [1.0, 2.0, 3.0]}

        # Mutating the returned dict should not affect the stored model
        weights["layer1"][0] = 999.0
        stored = self.coordinator.get_model("m1")
        assert stored.global_weights["layer1"][0] == 1.0


# =========================================================================
# TestDifferentialPrivacy
# =========================================================================


class TestDifferentialPrivacy:
    """Tests for the DifferentialPrivacy class."""

    def setup_method(self):
        self.dp = DifferentialPrivacy()

    def test_add_noise_changes_values(self):
        """Noised gradients differ from the original values."""
        gradients = {"layer1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        noisy = self.dp.add_noise(gradients, epsilon=1.0, delta=1e-5)

        assert "layer1" in noisy
        assert len(noisy["layer1"]) == 5

        # At least one value should differ (probabilistically near-certain
        # given Gaussian noise with sigma > 0).
        differs = any(
            noisy["layer1"][i] != gradients["layer1"][i]
            for i in range(5)
        )
        assert differs, "Expected at least one value to change after noise injection"

    def test_clip_gradients_respects_norm(self):
        """Clipped gradients have L2 norm <= max_norm."""
        # Vector [3.0, 4.0] has L2 norm = 5.0
        gradients = {"layer1": [3.0, 4.0]}
        max_norm = 1.0
        clipped = self.dp.clip_gradients(gradients, max_norm=max_norm)

        l2 = math.sqrt(sum(v * v for v in clipped["layer1"]))
        assert l2 <= max_norm + 1e-9

    def test_clip_preserves_small_gradients(self):
        """Gradients already below max_norm are unchanged."""
        # Vector [0.3, 0.4] has L2 norm = 0.5
        gradients = {"layer1": [0.3, 0.4]}
        max_norm = 1.0
        clipped = self.dp.clip_gradients(gradients, max_norm=max_norm)

        assert clipped["layer1"][0] == pytest.approx(0.3)
        assert clipped["layer1"][1] == pytest.approx(0.4)


# =========================================================================
# TestCompetitiveIntelligence
# =========================================================================


class TestCompetitiveIntelligence:
    """Tests for the CompetitiveIntelligenceEngine class."""

    def setup_method(self):
        self.engine = CompetitiveIntelligenceEngine()

    def _make_entity(
        self,
        entity_id: str,
        name: str,
        category: str,
        city: str,
        vibe_dna: dict | None = None,
        tags: list | None = None,
        rating: float = 3.0,
    ) -> dict:
        return {
            "entity_id": entity_id,
            "name": name,
            "category": category,
            "city": city,
            "vibe_dna": vibe_dna or {},
            "tags": tags or [],
            "rating": rating,
        }

    def test_analyze_competitors_finds_similar(self):
        """Entities in the same category and city are found as competitors."""
        target = self._make_entity(
            "e1", "Target Cafe", "cafe", "Tokyo",
            vibe_dna={"minimalist": 0.8, "cozy": 0.5},
        )
        competitor = self._make_entity(
            "e2", "Rival Cafe", "cafe", "Tokyo",
            vibe_dna={"minimalist": 0.7, "cozy": 0.6},
        )
        unrelated = self._make_entity(
            "e3", "Shoe Store", "retail", "Paris",
            vibe_dna={"edgy": 0.9},
        )
        all_entities = [target, competitor, unrelated]

        analysis = self.engine.analyze_competitors(
            entity_id="e1",
            entity_data=target,
            all_entities=all_entities,
            top_k=10,
        )

        competitor_ids = [c.entity_id for c in analysis.competitors]
        assert "e2" in competitor_ids
        assert "e3" not in competitor_ids

    def test_market_segments_grouped(self):
        """Entities are grouped by category + city into market segments."""
        entities = [
            self._make_entity(f"e{i}", f"Cafe {i}", "cafe", "Tokyo")
            for i in range(5)
        ]

        segments = self.engine.identify_market_segments(entities, min_segment_size=3)

        assert len(segments) >= 1
        cafe_tokyo = [s for s in segments if s.category == "cafe" and s.geographic_scope == "Tokyo"]
        assert len(cafe_tokyo) == 1
        assert cafe_tokyo[0].entity_count == 5

    def test_market_position_score_range(self):
        """Market position score is between 0 and 100."""
        target = self._make_entity(
            "e1", "Target", "cafe", "Tokyo",
            vibe_dna={"minimalist": 0.8},
            rating=4.0,
        )
        competitors = [
            CompetitorProfile(
                entity_id=f"c{i}",
                name=f"Comp {i}",
                category="cafe",
                city="Tokyo",
                vibe_dna={"minimalist": 0.5 + i * 0.1},
                similarity_score=0.6 + i * 0.05,
            )
            for i in range(5)
        ]

        score = self.engine.compute_market_position(target, competitors)

        assert 0.0 <= score <= 100.0

    def test_swot_has_all_sections(self):
        """SWOT analysis dict has strengths, weaknesses, opportunities, threats keys."""
        target = self._make_entity(
            "e1", "Target", "cafe", "Tokyo",
            vibe_dna={"minimalist": 0.9, "cozy": 0.2},
            tags=["specialty-coffee"],
        )
        competitors = [
            CompetitorProfile(
                entity_id="c1",
                name="Comp 1",
                category="cafe",
                city="Tokyo",
                vibe_dna={"minimalist": 0.3, "cozy": 0.8},
                tags=["latte-art"],
                similarity_score=0.5,
            ),
        ]
        segments = self.engine.identify_market_segments(
            [target, self._make_entity("e2", "X", "cafe", "Tokyo")],
            min_segment_size=1,
        )

        swot = self.engine.generate_swot(target, competitors, segments)

        assert "strengths" in swot
        assert "weaknesses" in swot
        assert "opportunities" in swot
        assert "threats" in swot
        assert isinstance(swot["strengths"], list)
        assert isinstance(swot["weaknesses"], list)
        assert isinstance(swot["opportunities"], list)
        assert isinstance(swot["threats"], list)

    def test_vibe_similarity_identical(self):
        """Identical vibe_dna dictionaries give a similarity of 1.0."""
        dna = {"minimalist": 0.9, "cozy": 0.4, "bohemian": 0.6}
        similarity = _vibe_similarity(dna, dna)
        assert similarity == pytest.approx(1.0)


# We need CompetitorProfile for the tests above
from app.ml.competitive_intelligence import CompetitorProfile


# =========================================================================
# TestGraphExpansion
# =========================================================================


class TestGraphExpansion:
    """Tests for graph expansion math helpers and data classes."""

    def test_cosine_similarity_identical(self):
        """Identical vectors return cosine similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors return cosine similarity of 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_haversine_same_point(self):
        """Same coordinates return 0.0 km distance."""
        dist = _haversine_km(45.4642, 9.1900, 45.4642, 9.1900)
        assert dist == pytest.approx(0.0)

    def test_haversine_known_distance(self):
        """Milan to Rome is approximately 480 km (within 10% tolerance)."""
        # Milan: 45.4642 N, 9.1900 E
        # Rome: 41.9028 N, 12.4964 E
        dist = _haversine_km(45.4642, 9.1900, 41.9028, 12.4964)
        assert dist == pytest.approx(480.0, rel=0.10)

    def test_relationship_candidate_creation(self):
        """RelationshipCandidate stores all fields correctly."""
        candidate = RelationshipCandidate(
            source_id="entity-1",
            target_id="entity-2",
            rel_type="SIMILAR_VIBE",
            confidence=0.92,
            evidence="High cosine similarity",
            discovered_by="style_similarity",
        )

        assert candidate.source_id == "entity-1"
        assert candidate.target_id == "entity-2"
        assert candidate.rel_type == "SIMILAR_VIBE"
        assert candidate.confidence == 0.92
        assert candidate.evidence == "High cosine similarity"
        assert candidate.discovered_by == "style_similarity"


# =========================================================================
# TestAutoExpansionScheduler
# =========================================================================


class TestAutoExpansionScheduler:
    """Tests for the AutoExpansionScheduler class."""

    def setup_method(self):
        self.scheduler = AutoExpansionScheduler()

    def test_schedule_adds_to_queue(self):
        """Scheduling entity IDs adds them to the internal queue."""
        self.scheduler.schedule_expansion(["e1", "e2", "e3"])
        stats = self.scheduler.stats()
        assert stats["queue_size"] == 3

    def test_schedule_deduplicates(self):
        """Scheduling the same ID twice only adds it once."""
        self.scheduler.schedule_expansion(["e1", "e2"])
        self.scheduler.schedule_expansion(["e2", "e3"])
        stats = self.scheduler.stats()
        assert stats["queue_size"] == 3

    def test_stats_returns_counts(self):
        """Stats dict has queue_size and total_processed keys."""
        stats = self.scheduler.stats()
        assert "queue_size" in stats
        assert "total_processed" in stats
        assert stats["queue_size"] == 0
        assert stats["total_processed"] == 0
