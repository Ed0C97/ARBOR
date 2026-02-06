"""Personalization engine for learning user preferences and re-ranking results.

Tracks per-user style affinities, category preferences, city preferences, and
price-tier signals.  Uses an exponential moving average to blend new interaction
signals into the profile so that recent behaviour is weighted more heavily than
old behaviour.

Usage::

    engine = get_personalization_engine()
    engine.record_interaction(
        user_id="u_42",
        entity_id="e_101",
        entity_data={"styles": ["minimalist"], "category": "cafe", "city": "CDMX"},
        action="save",
        position=2,
    )
    reranked = engine.personalize_results("u_42", raw_results, boost_factor=0.3)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """Compact representation of learned user preferences.

    Attributes:
        user_id: Unique identifier for the user.
        style_preferences: Mapping of style tag to learned weight (0-1 range).
        category_affinities: Mapping of entity category to learned weight.
        city_preferences: Mapping of city name to learned weight.
        price_tier: Preferred price tier as an exponentially smoothed float
            (1 = budget ... 4 = luxury).  Defaults to the midpoint.
        interaction_count: Total number of recorded interactions, used for
            confidence scaling.
        decay_factor: Multiplicative decay applied to all weights before each
            update so that stale preferences fade over time.
    """

    user_id: str
    style_preferences: dict[str, float] = field(default_factory=dict)
    category_affinities: dict[str, float] = field(default_factory=dict)
    city_preferences: dict[str, float] = field(default_factory=dict)
    price_tier: float = 2.5
    interaction_count: int = 0
    decay_factor: float = 0.95


# ---------------------------------------------------------------------------
# Preference learner
# ---------------------------------------------------------------------------


class PreferenceLearner:
    """Stateless helper that applies a single interaction to a UserProfile.

    Learning rates per action type dictate how aggressively each signal
    updates the profile.  A position-aware modifier amplifies signals from
    results that required the user to scroll further down the list (implying
    a stronger intent signal).
    """

    # Action -> base learning rate
    ACTION_RATES: dict[str, float] = {
        "convert": 0.3,
        "save": 0.2,
        "click": 0.1,
        "dismiss": -0.05,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_feedback(
        self,
        profile: UserProfile,
        entity_data: dict[str, Any],
        action: str,
        position: int = 0,
    ) -> None:
        """Update *profile* in-place based on a single user interaction.

        The method:
        1. Applies the decay factor to all existing weights.
        2. Computes a position-aware learning rate.
        3. Blends style, category, and city signals into the profile using
           an exponential moving average.
        4. Increments the interaction counter.

        Args:
            profile: The user profile to update.
            entity_data: Dict describing the entity the user interacted with.
                Expected keys: ``styles`` (list[str]), ``category`` (str),
                ``city`` (str), ``price_tier`` (int | float, optional).
            action: Interaction type (``click``, ``save``, ``convert``,
                ``dismiss``).
            position: 0-based rank position of the result in the list.
        """
        learning_rate = self._position_adjusted_rate(action, position)

        # 1. Decay existing weights so old preferences fade
        self._apply_decay(profile)

        # 2. Update style preferences
        styles: list[str] = entity_data.get("styles", [])
        for style in styles:
            old = profile.style_preferences.get(style, 0.0)
            profile.style_preferences[style] = self._ema(old, learning_rate)

        # 3. Update category affinity
        category: str = entity_data.get("category", "")
        if category:
            old = profile.category_affinities.get(category, 0.0)
            profile.category_affinities[category] = self._ema(old, learning_rate)

        # 4. Update city preference
        city: str = entity_data.get("city", "")
        if city:
            old = profile.city_preferences.get(city, 0.0)
            profile.city_preferences[city] = self._ema(old, learning_rate)

        # 5. Update price tier (if available in entity data)
        entity_price = entity_data.get("price_tier")
        if entity_price is not None:
            alpha = abs(learning_rate)
            profile.price_tier = (1.0 - alpha) * profile.price_tier + alpha * float(
                entity_price
            )

        profile.interaction_count += 1

        logger.debug(
            "Profile updated: user=%s action=%s position=%d lr=%.4f interactions=%d",
            profile.user_id,
            action,
            position,
            learning_rate,
            profile.interaction_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _position_adjusted_rate(self, action: str, position: int) -> float:
        """Return the learning rate scaled by a position modifier.

        Higher positions (further down the list) amplify the signal because
        the user had to actively scroll past earlier results.

        Args:
            action: Interaction type.
            position: 0-based rank position.

        Returns:
            Adjusted learning rate (may be negative for ``dismiss``).
        """
        base_rate = self.ACTION_RATES.get(action, 0.05)
        # Position modifier: position 0 -> 1.0, position 10 -> ~1.4
        position_modifier = 1.0 + math.log1p(position) * 0.2
        return base_rate * position_modifier

    @staticmethod
    def _ema(old_value: float, learning_rate: float) -> float:
        """Apply a single-step exponential moving average blend.

        For positive rates the weight moves toward 1.0; for negative rates
        (dismiss) it moves toward 0.0.  The result is clamped to [0, 1].

        Args:
            old_value: Current weight in the profile.
            learning_rate: Signed step size.

        Returns:
            Updated weight clamped to [0.0, 1.0].
        """
        new_value = old_value + learning_rate * (1.0 - old_value) if learning_rate >= 0 else old_value + learning_rate * old_value
        return max(0.0, min(1.0, new_value))

    @staticmethod
    def _apply_decay(profile: UserProfile) -> None:
        """Multiply all preference weights by the profile's decay factor.

        This ensures that preferences which are not reinforced will
        gradually fade toward zero.

        Args:
            profile: The user profile whose weights will be decayed.
        """
        decay = profile.decay_factor
        for key in profile.style_preferences:
            profile.style_preferences[key] *= decay
        for key in profile.category_affinities:
            profile.category_affinities[key] *= decay
        for key in profile.city_preferences:
            profile.city_preferences[key] *= decay


# ---------------------------------------------------------------------------
# Personalization engine
# ---------------------------------------------------------------------------


class PersonalizationEngine:
    """Orchestrates user-profile management and result re-ranking.

    Maintains an in-memory store of :class:`UserProfile` objects and exposes
    methods to record interactions and to boost / re-rank search results based
    on learned preferences.

    Usage::

        engine = get_personalization_engine()
        reranked = engine.personalize_results("u_42", results)
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}
        self._learner = PreferenceLearner()
        logger.info("PersonalizationEngine initialised (in-memory store)")

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Return the profile for *user_id*, creating one if it does not exist.

        Args:
            user_id: Unique user identifier.

        Returns:
            The existing or newly created :class:`UserProfile`.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
            logger.debug("Created new profile for user=%s", user_id)
        return self._profiles[user_id]

    # ------------------------------------------------------------------
    # Recording interactions
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        user_id: str,
        entity_id: str,
        entity_data: dict[str, Any],
        action: str,
        position: int = 0,
    ) -> None:
        """Record a user interaction and update the corresponding profile.

        Args:
            user_id: Unique user identifier.
            entity_id: Identifier of the entity the user interacted with.
            entity_data: Dict describing the entity (styles, category, city, etc.).
            action: Interaction type (``click``, ``save``, ``convert``, ``dismiss``).
            position: 0-based rank position of the result.
        """
        profile = self.get_user_profile(user_id)
        self._learner.update_from_feedback(profile, entity_data, action, position)

        logger.debug(
            "Interaction recorded: user=%s entity=%s action=%s position=%d",
            user_id,
            entity_id,
            action,
            position,
        )

    # ------------------------------------------------------------------
    # Result re-ranking
    # ------------------------------------------------------------------

    def personalize_results(
        self,
        user_id: str,
        results: list[dict[str, Any]],
        boost_factor: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Re-rank *results* based on the user's learned preferences.

        Each result receives a ``personalization_boost`` score which is a
        weighted combination of style match, category match, and city match.
        The original ordering score (if present) is blended with the boost
        to produce a ``personalized_score``.

        Args:
            user_id: Unique user identifier.
            results: List of entity dicts to re-rank.
            boost_factor: How much weight to give the personalization boost
                relative to the original score (0.0 = no personalisation,
                1.0 = fully personalised).

        Returns:
            A new list of result dicts sorted by ``personalized_score``
            (descending), each enriched with ``personalization_boost`` and
            ``personalized_score`` fields.
        """
        profile = self.get_user_profile(user_id)

        # If the profile has no interactions yet, return results unchanged
        if profile.interaction_count == 0:
            logger.debug(
                "No interactions for user=%s, returning original order", user_id
            )
            return results

        scored: list[dict[str, Any]] = []
        for result in results:
            enriched = result.copy()
            boost = self._compute_boost(profile, result)
            enriched["personalization_boost"] = round(boost, 4)

            original_score = float(result.get("score", result.get("rerank_score", 0.5)))
            personalized = (1.0 - boost_factor) * original_score + boost_factor * boost
            enriched["personalized_score"] = round(personalized, 4)

            scored.append(enriched)

        scored.sort(key=lambda r: r["personalized_score"], reverse=True)

        logger.debug(
            "Personalised results for user=%s: count=%d boost_factor=%.2f",
            user_id,
            len(scored),
            boost_factor,
        )
        return scored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_boost(
        self, profile: UserProfile, result: dict[str, Any]
    ) -> float:
        """Compute a personalisation boost score for a single result.

        The boost is the weighted average of three sub-scores:

        * **Style match** (weight 0.5) - average of profile weights for
          each style tag present on the entity.
        * **Category match** (weight 0.3) - profile weight for the
          entity's category.
        * **City match** (weight 0.2) - profile weight for the entity's
          city.

        Args:
            profile: The user's learned preference profile.
            result: A single entity dict.

        Returns:
            A float in the range [0.0, 1.0].
        """
        style_score = 0.0
        category_score = 0.0
        city_score = 0.0

        # Style match
        styles: list[str] = result.get("styles", [])
        if styles:
            matches = [profile.style_preferences.get(s, 0.0) for s in styles]
            style_score = sum(matches) / len(matches)

        # Category match
        category: str = result.get("category", "")
        if category:
            category_score = profile.category_affinities.get(category, 0.0)

        # City match
        city: str = result.get("city", "")
        if city:
            city_score = profile.city_preferences.get(city, 0.0)

        # Weighted combination
        boost = 0.5 * style_score + 0.3 * category_score + 0.2 * city_score
        return max(0.0, min(1.0, boost))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine: Optional[PersonalizationEngine] = None


def get_personalization_engine() -> PersonalizationEngine:
    """Return the singleton PersonalizationEngine instance."""
    global _engine
    if _engine is None:
        _engine = PersonalizationEngine()
    return _engine
