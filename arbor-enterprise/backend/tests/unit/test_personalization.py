"""Unit tests for the personalization engine and explainability engine.

Tests cover:
- UserProfile data class defaults and attribute storage
- PreferenceLearner update logic (learning rates, decay, position effects)
- PersonalizationEngine profile management and result re-ranking
- ExplainabilityEngine factor computation, summaries, and batch explanations
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Inline helpers to avoid importing app.config (which needs env vars)
# ---------------------------------------------------------------------------

def _make_user_profile(user_id="u_test", **kwargs):
    """Create a UserProfile without triggering app.config imports."""
    from app.ml.personalization import UserProfile
    return UserProfile(user_id=user_id, **kwargs)


def _make_learner():
    from app.ml.personalization import PreferenceLearner
    return PreferenceLearner()


def _make_engine():
    from app.ml.personalization import PersonalizationEngine
    return PersonalizationEngine()


def _make_explainability_engine():
    from app.ml.explainability import ExplainabilityEngine
    return ExplainabilityEngine()


# ===================================================================
# TestUserProfile
# ===================================================================


class TestUserProfile:
    """Tests for the UserProfile data class."""

    def test_default_profile_creation(self):
        """A new profile has empty preference dicts and interaction_count=0."""
        profile = _make_user_profile(user_id="u_new")

        assert profile.user_id == "u_new"
        assert profile.style_preferences == {}
        assert profile.category_affinities == {}
        assert profile.city_preferences == {}
        assert profile.interaction_count == 0

    def test_decay_factor_default(self):
        """The decay_factor defaults to 0.95."""
        profile = _make_user_profile()

        assert profile.decay_factor == 0.95

    def test_profile_stores_preferences(self):
        """Style, category, and city preferences are stored correctly."""
        profile = _make_user_profile(
            style_preferences={"minimalist": 0.8, "cozy": 0.3},
            category_affinities={"cafe": 0.6},
            city_preferences={"Milan": 0.9},
        )

        assert profile.style_preferences["minimalist"] == 0.8
        assert profile.style_preferences["cozy"] == 0.3
        assert profile.category_affinities["cafe"] == 0.6
        assert profile.city_preferences["Milan"] == 0.9

    def test_price_tier_default(self):
        """The default price_tier is 2.5 (midpoint of 1-4 range)."""
        profile = _make_user_profile()

        assert profile.price_tier == 2.5


# ===================================================================
# TestPreferenceLearner
# ===================================================================


class TestPreferenceLearner:
    """Tests for the PreferenceLearner update logic."""

    def test_click_updates_style_preferences(self):
        """Clicking an entity with style 'minimalist' increases that weight."""
        learner = _make_learner()
        profile = _make_user_profile()

        entity_data = {
            "styles": ["minimalist"],
            "category": "cafe",
            "city": "Milan",
        }

        learner.update_from_feedback(profile, entity_data, action="click", position=0)

        assert "minimalist" in profile.style_preferences
        assert profile.style_preferences["minimalist"] > 0.0

    def test_convert_has_higher_learning_rate(self):
        """A 'convert' action produces larger weight changes than 'click'."""
        learner = _make_learner()

        profile_click = _make_user_profile(user_id="u_click")
        profile_convert = _make_user_profile(user_id="u_convert")

        entity_data = {
            "styles": ["industrial"],
            "category": "bar",
            "city": "Berlin",
        }

        learner.update_from_feedback(profile_click, entity_data, action="click", position=0)
        learner.update_from_feedback(profile_convert, entity_data, action="convert", position=0)

        click_weight = profile_click.style_preferences.get("industrial", 0.0)
        convert_weight = profile_convert.style_preferences.get("industrial", 0.0)

        assert convert_weight > click_weight, (
            f"convert weight ({convert_weight}) should exceed click weight ({click_weight})"
        )

    def test_dismiss_decreases_preferences(self):
        """A 'dismiss' action reduces existing preference weights."""
        learner = _make_learner()
        profile = _make_user_profile(
            style_preferences={"vintage": 0.8},
            category_affinities={"restaurant": 0.7},
            city_preferences={"Rome": 0.6},
        )

        entity_data = {
            "styles": ["vintage"],
            "category": "restaurant",
            "city": "Rome",
        }

        learner.update_from_feedback(profile, entity_data, action="dismiss", position=0)

        # After dismiss + decay, the weight should be lower than the original
        # Original was 0.8, decay (0.95) gives 0.76, then dismiss reduces further
        assert profile.style_preferences["vintage"] < 0.8
        assert profile.category_affinities["restaurant"] < 0.7
        assert profile.city_preferences["Rome"] < 0.6

    def test_position_affects_learning(self):
        """Lower position (further down the list) produces a stronger signal."""
        learner = _make_learner()

        profile_pos0 = _make_user_profile(user_id="u_pos0")
        profile_pos10 = _make_user_profile(user_id="u_pos10")

        entity_data = {
            "styles": ["modern"],
            "category": "gallery",
            "city": "Tokyo",
        }

        learner.update_from_feedback(profile_pos0, entity_data, action="click", position=0)
        learner.update_from_feedback(profile_pos10, entity_data, action="click", position=10)

        weight_pos0 = profile_pos0.style_preferences.get("modern", 0.0)
        weight_pos10 = profile_pos10.style_preferences.get("modern", 0.0)

        assert weight_pos10 > weight_pos0, (
            f"Position 10 weight ({weight_pos10}) should exceed position 0 weight ({weight_pos0})"
        )

    def test_decay_reduces_old_preferences(self):
        """Calling update applies decay to existing preference weights."""
        learner = _make_learner()
        profile = _make_user_profile(
            style_preferences={"bohemian": 1.0},
        )

        # Interact with a completely different entity so the 'bohemian' weight
        # only gets decayed, not reinforced.
        entity_data = {
            "styles": ["industrial"],
            "category": "bar",
            "city": "London",
        }

        learner.update_from_feedback(profile, entity_data, action="click", position=0)

        # After one decay pass, 1.0 * 0.95 = 0.95
        assert profile.style_preferences["bohemian"] == pytest.approx(0.95, abs=0.001)

    def test_category_affinity_updates(self):
        """Category preference updates when an interaction occurs."""
        learner = _make_learner()
        profile = _make_user_profile()

        entity_data = {
            "styles": [],
            "category": "bookshop",
            "city": "",
        }

        learner.update_from_feedback(profile, entity_data, action="save", position=0)

        assert "bookshop" in profile.category_affinities
        assert profile.category_affinities["bookshop"] > 0.0
        assert profile.interaction_count == 1


# ===================================================================
# TestPersonalizationEngine
# ===================================================================


class TestPersonalizationEngine:
    """Tests for the PersonalizationEngine orchestration layer."""

    def test_get_user_profile_creates_new(self):
        """Requesting a profile for an unknown user creates a fresh one."""
        engine = _make_engine()
        profile = engine.get_user_profile("u_unknown")

        assert profile.user_id == "u_unknown"
        assert profile.interaction_count == 0
        assert profile.style_preferences == {}

    def test_get_user_profile_returns_existing(self):
        """A second request for the same user returns the same profile object."""
        engine = _make_engine()
        p1 = engine.get_user_profile("u_42")
        p2 = engine.get_user_profile("u_42")

        assert p1 is p2

    def test_record_interaction_updates_profile(self):
        """Recording an interaction changes the profile's weights and count."""
        engine = _make_engine()

        entity_data = {
            "styles": ["minimalist", "modern"],
            "category": "cafe",
            "city": "CDMX",
        }

        engine.record_interaction(
            user_id="u_1",
            entity_id="e_101",
            entity_data=entity_data,
            action="save",
            position=3,
        )

        profile = engine.get_user_profile("u_1")
        assert profile.interaction_count == 1
        assert profile.style_preferences.get("minimalist", 0.0) > 0.0
        assert profile.category_affinities.get("cafe", 0.0) > 0.0
        assert profile.city_preferences.get("CDMX", 0.0) > 0.0

    def test_personalize_results_reorders(self):
        """Results are reordered based on learned user preferences."""
        engine = _make_engine()

        # Build up a strong preference for 'minimalist' style and 'cafe' category
        for _ in range(5):
            engine.record_interaction(
                user_id="u_pref",
                entity_id="e_200",
                entity_data={
                    "styles": ["minimalist"],
                    "category": "cafe",
                    "city": "Milan",
                },
                action="convert",
                position=0,
            )

        # Result A: matches preferences. Result B: does not.
        results = [
            {
                "id": "B",
                "name": "Loud Bar",
                "styles": ["industrial"],
                "category": "bar",
                "city": "Berlin",
                "score": 0.9,
            },
            {
                "id": "A",
                "name": "Quiet Cafe",
                "styles": ["minimalist"],
                "category": "cafe",
                "city": "Milan",
                "score": 0.5,
            },
        ]

        reranked = engine.personalize_results("u_pref", results, boost_factor=0.8)

        # The matching entity (A) should be boosted to the top despite lower
        # original score.
        assert reranked[0]["id"] == "A", (
            f"Expected 'A' first but got '{reranked[0]['id']}'"
        )

    def test_personalize_results_no_profile(self):
        """Without any recorded interactions the original order is preserved."""
        engine = _make_engine()

        results = [
            {"id": "X", "name": "First", "score": 0.8},
            {"id": "Y", "name": "Second", "score": 0.6},
        ]

        reranked = engine.personalize_results("u_fresh", results)

        assert [r["id"] for r in reranked] == ["X", "Y"]

    def test_boost_factor_controls_strength(self):
        """A higher boost_factor changes the relative ordering more aggressively.

        The personalized_score formula is:
            (1 - boost_factor) * original_score + boost_factor * boost

        A higher boost_factor amplifies the *difference* caused by personalization.
        We verify this by checking that the gap between the matching entity's
        personalized score and the non-matching entity's personalized score
        shifts in favour of the match as boost_factor increases.
        """
        engine = _make_engine()

        # Build mild preference for 'cozy'
        engine.record_interaction(
            user_id="u_boost",
            entity_id="e_300",
            entity_data={
                "styles": ["cozy"],
                "category": "restaurant",
                "city": "Paris",
            },
            action="click",
            position=0,
        )

        results = [
            {
                "id": "Non-match",
                "styles": ["industrial"],
                "category": "bar",
                "city": "NYC",
                "score": 0.9,
            },
            {
                "id": "Match",
                "styles": ["cozy"],
                "category": "restaurant",
                "city": "Paris",
                "score": 0.5,
            },
        ]

        low_boost = engine.personalize_results("u_boost", results, boost_factor=0.1)
        high_boost = engine.personalize_results("u_boost", results, boost_factor=0.9)

        def _gap(ranked_results):
            """Return (match_score - non_match_score)."""
            scores = {r["id"]: r["personalized_score"] for r in ranked_results}
            return scores["Match"] - scores["Non-match"]

        low_gap = _gap(low_boost)
        high_gap = _gap(high_boost)

        # With low boost the non-match entity dominates (gap is negative).
        # With high boost the personalization signal narrows or reverses the gap.
        assert high_gap > low_gap, (
            f"high-boost gap ({high_gap}) should be larger (more favourable to "
            f"Match) than low-boost gap ({low_gap})"
        )


# ===================================================================
# TestExplainabilityEngine
# ===================================================================


class TestExplainabilityEngine:
    """Tests for the ExplainabilityEngine."""

    def test_explain_recommendation_returns_factors(self):
        """The explanation contains a non-empty factors list."""
        engine = _make_explainability_engine()

        entity_data = {
            "id": "e_1",
            "name": "Cafe Moya",
            "category": "cafe",
            "styles": ["minimalist"],
            "city": "Milan",
            "description": "A minimalist cafe in the heart of Milan",
        }

        explanation = engine.explain_recommendation(
            entity_data=entity_data,
            query="minimalist cafes in Milan",
            search_scores={"text_score": 0.82, "rerank_score": 0.91},
            rank=1,
        )

        assert explanation.factors is not None
        assert len(explanation.factors) > 0
        assert explanation.entity_id == "e_1"
        assert explanation.entity_name == "Cafe Moya"

    def test_text_relevance_high_for_matching_query(self):
        """When query keywords match entity text, a high-weight factor is produced."""
        engine = _make_explainability_engine()

        entity_data = {
            "id": "e_2",
            "name": "Minimalist Cafe",
            "category": "cafe",
            "styles": ["minimalist"],
            "city": "Milan",
            "description": "A minimal cafe experience",
        }

        explanation = engine.explain_recommendation(
            entity_data=entity_data,
            query="minimalist cafe Milan",
            rank=1,
        )

        from app.ml.explainability import FACTOR_STYLE_MATCH

        style_factors = [
            f for f in explanation.factors if f.factor_type == FACTOR_STYLE_MATCH
        ]
        assert len(style_factors) >= 1
        # All query tokens appear in entity text, so weight should be high
        assert style_factors[0].weight > 0.5

    def test_category_match_detected(self):
        """Entity category appearing in the query produces a category_match factor."""
        engine = _make_explainability_engine()

        entity_data = {
            "id": "e_3",
            "name": "Bella Notte",
            "category": "restaurant",
            "styles": [],
            "city": "Florence",
            "description": "Fine Italian dining",
        }

        explanation = engine.explain_recommendation(
            entity_data=entity_data,
            query="best restaurant in Florence",
            rank=1,
        )

        from app.ml.explainability import FACTOR_CATEGORY_MATCH

        category_factors = [
            f for f in explanation.factors if f.factor_type == FACTOR_CATEGORY_MATCH
        ]
        assert len(category_factors) == 1
        assert category_factors[0].weight > 0.0
        assert category_factors[0].details["is_match"] is True

    def test_geo_relevance_detected(self):
        """Entity city appearing in the query produces a location factor."""
        engine = _make_explainability_engine()

        entity_data = {
            "id": "e_4",
            "name": "Tokyo Ramen House",
            "category": "restaurant",
            "styles": [],
            "city": "Tokyo",
            "description": "Authentic ramen",
        }

        explanation = engine.explain_recommendation(
            entity_data=entity_data,
            query="ramen restaurants in Tokyo",
            rank=1,
        )

        from app.ml.explainability import FACTOR_LOCATION

        geo_factors = [
            f for f in explanation.factors if f.factor_type == FACTOR_LOCATION
        ]
        assert len(geo_factors) == 1
        assert geo_factors[0].weight > 0.0
        assert "Tokyo" in geo_factors[0].details["matched_locations"]

    def test_summary_generation(self):
        """The summary is a non-empty string that mentions the entity name."""
        engine = _make_explainability_engine()

        entity_data = {
            "id": "e_5",
            "name": "Sunset Lounge",
            "category": "bar",
            "styles": ["modern"],
            "city": "Barcelona",
            "description": "Rooftop bar with sunset views",
        }

        explanation = engine.explain_recommendation(
            entity_data=entity_data,
            query="modern bars in Barcelona",
            rank=1,
        )

        assert isinstance(explanation.summary, str)
        assert len(explanation.summary) > 0
        # The summary should either start with "Recommended because" (when
        # factors are found) or mention the entity name in the fallback.
        assert "Recommended because" in explanation.summary or "Sunset Lounge" in explanation.summary

    def test_explain_batch_returns_all(self):
        """Batch explanation returns one explanation per entity."""
        engine = _make_explainability_engine()

        entities = [
            {
                "id": "e_10",
                "name": "Alpha Cafe",
                "category": "cafe",
                "styles": ["minimalist"],
                "city": "Milan",
                "description": "A cafe",
            },
            {
                "id": "e_11",
                "name": "Beta Bar",
                "category": "bar",
                "styles": ["modern"],
                "city": "Berlin",
                "description": "A bar",
            },
            {
                "id": "e_12",
                "name": "Gamma Gallery",
                "category": "gallery",
                "styles": ["artistic"],
                "city": "Paris",
                "description": "An art gallery",
            },
        ]

        explanations = engine.explain_batch(
            entities=entities,
            query="cafes and bars",
        )

        assert len(explanations) == 3

        # Each explanation should have the correct entity_id and rank
        assert explanations[0].entity_id == "e_10"
        assert explanations[0].rank == 1
        assert explanations[1].entity_id == "e_11"
        assert explanations[1].rank == 2
        assert explanations[2].entity_id == "e_12"
        assert explanations[2].rank == 3

        # Each explanation should have a summary
        for expl in explanations:
            assert isinstance(expl.summary, str)
            assert len(expl.summary) > 0
