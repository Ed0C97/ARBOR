"""Synthetic Data Generation for ARBOR Enterprise.

Generates realistic-looking but entirely fictional entities, users, and
interaction logs for testing, development, and privacy-safe analytics.

The generators produce data that is statistically similar to real platform
data (category distributions, vibe score ranges, interaction patterns)
without ever copying real records.  A :class:`DataValidator` verifies that
synthetic outputs are both distributionally faithful and sufficiently
distant from any real data.

Usage::

    entity_gen = get_synthetic_generator()
    entities = entity_gen.generate_batch(100, distribution={"cafe": 30, "boutique": 40, "gallery": 30})

    user_gen = SyntheticUserGenerator()
    users = user_gen.generate_batch(50)

    interaction_gen = SyntheticInteractionGenerator()
    interactions = interaction_gen.generate_interactions(users, entities, count=500)

    validator = DataValidator()
    report = validator.validate_statistical_similarity(entities, real_stats)
"""

import logging
import math
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Constants & templates
# ---------------------------------------------------------------------------

# Name templates per category.  ``{adj}`` and ``{noun}`` are filled from the
# adjective / noun pools below.
NAME_TEMPLATES: dict[str, list[str]] = {
    "cafe": [
        "{adj} Cafe",
        "Cafe {noun}",
        "{noun} Coffee House",
        "The {adj} Bean",
        "{adj} Brew",
        "Cafe de {noun}",
    ],
    "boutique": [
        "Atelier {adj} {noun}",
        "Maison {noun}",
        "{adj} Studio",
        "The {noun} Boutique",
        "{adj} & {noun}",
        "House of {noun}",
    ],
    "gallery": [
        "{noun} Gallery",
        "Galerie {adj}",
        "The {adj} Space",
        "{noun} Art House",
        "Studio {noun}",
        "{adj} Contemporary",
    ],
    "restaurant": [
        "{adj} Table",
        "Trattoria {noun}",
        "The {noun} Kitchen",
        "{adj} & Co.",
        "Osteria {noun}",
        "{noun} Dining Room",
    ],
    "bar": [
        "The {adj} {noun}",
        "{noun} Lounge",
        "{adj} Spirits",
        "Bar {noun}",
        "The {noun} Room",
        "{adj} Cocktail Club",
    ],
    "hotel": [
        "Hotel {noun}",
        "The {adj} House",
        "{noun} Suites",
        "Maison {adj}",
        "The {noun} Hotel",
        "{adj} Residences",
    ],
}

ADJECTIVES = [
    "Golden",
    "Velvet",
    "Hidden",
    "Azure",
    "Artisan",
    "Nordic",
    "Crimson",
    "Ivory",
    "Urban",
    "Rustic",
    "Verdant",
    "Sapphire",
    "Copper",
    "Lunar",
    "Solar",
    "Misty",
    "Serene",
    "Bold",
    "Wild",
    "Noble",
    "Polished",
    "Raw",
    "Lush",
    "Vivid",
]

NOUNS = [
    "Oak",
    "Bloom",
    "Stone",
    "Luna",
    "Flora",
    "Ember",
    "Atlas",
    "Sage",
    "Onyx",
    "Ivy",
    "Lark",
    "Marble",
    "Fern",
    "Dusk",
    "Prism",
    "Celeste",
    "Raven",
    "Cedar",
    "Aura",
    "Jade",
    "Pearl",
    "Moss",
    "Flint",
    "Silk",
]

CITIES = [
    "Milan",
    "Paris",
    "Tokyo",
    "New York",
    "London",
    "Barcelona",
    "Berlin",
    "Copenhagen",
    "Mexico City",
    "Seoul",
    "Amsterdam",
    "Lisbon",
    "Melbourne",
    "Stockholm",
    "Montreal",
]

TAGS_BY_CATEGORY: dict[str, list[str]] = {
    "cafe": [
        "specialty-coffee",
        "minimalist",
        "third-wave",
        "brunch",
        "laptop-friendly",
        "courtyard",
        "plant-filled",
        "industrial",
        "artisanal-pastries",
        "matcha",
        "pour-over",
        "quiet",
    ],
    "boutique": [
        "designer",
        "vintage",
        "sustainable",
        "avant-garde",
        "streetwear",
        "handmade",
        "curated",
        "emerging-designers",
        "luxury",
        "gender-neutral",
        "slow-fashion",
        "denim",
    ],
    "gallery": [
        "contemporary",
        "photography",
        "sculpture",
        "digital-art",
        "installation",
        "emerging-artists",
        "mixed-media",
        "abstract",
        "figurative",
        "pop-art",
        "site-specific",
        "immersive",
    ],
    "restaurant": [
        "farm-to-table",
        "fine-dining",
        "tasting-menu",
        "casual",
        "seafood",
        "plant-based",
        "omakase",
        "wine-pairing",
        "open-kitchen",
        "rooftop",
        "historic-building",
        "michelin",
    ],
    "bar": [
        "craft-cocktails",
        "speakeasy",
        "rooftop",
        "wine-bar",
        "natural-wine",
        "live-music",
        "jazz",
        "dive",
        "mezcal",
        "sake",
        "tiki",
        "aperitivo",
    ],
    "hotel": [
        "boutique-hotel",
        "design-hotel",
        "historic",
        "eco-friendly",
        "rooftop-pool",
        "spa",
        "art-collection",
        "minimalist",
        "converted-warehouse",
        "seaside",
        "city-center",
        "luxury",
    ],
}

# Vibe DNA dimensions (used across the platform)
VIBE_DIMENSIONS = [
    "cozy",
    "edgy",
    "luxurious",
    "minimalist",
    "eclectic",
    "vintage",
    "modern",
    "artistic",
    "rustic",
    "futuristic",
]

# Category-specific vibe biases: which dimensions tend to score higher.
VIBE_BIAS: dict[str, dict[str, float]] = {
    "cafe": {"cozy": 0.3, "minimalist": 0.2, "artistic": 0.1},
    "boutique": {"edgy": 0.2, "luxurious": 0.2, "modern": 0.15},
    "gallery": {"artistic": 0.35, "modern": 0.15, "eclectic": 0.1},
    "restaurant": {"luxurious": 0.2, "cozy": 0.15, "rustic": 0.1},
    "bar": {"edgy": 0.2, "vintage": 0.15, "eclectic": 0.1},
    "hotel": {"luxurious": 0.3, "modern": 0.15, "minimalist": 0.1},
}

# User segments
USER_SEGMENTS = {
    "luxury_seeker": {
        "preferred_categories": ["boutique", "restaurant", "hotel"],
        "vibe_affinity": {"luxurious": 0.8, "modern": 0.6, "minimalist": 0.5},
        "price_tier": "high",
        "style_keywords": ["refined", "exclusive", "premium"],
    },
    "budget_conscious": {
        "preferred_categories": ["cafe", "bar", "gallery"],
        "vibe_affinity": {"cozy": 0.7, "vintage": 0.6, "rustic": 0.5},
        "price_tier": "low",
        "style_keywords": ["affordable", "authentic", "hidden-gem"],
    },
    "trend_follower": {
        "preferred_categories": ["boutique", "cafe", "bar"],
        "vibe_affinity": {"edgy": 0.8, "modern": 0.7, "futuristic": 0.5},
        "price_tier": "medium",
        "style_keywords": ["trending", "instagram-worthy", "new"],
    },
    "classic_taste": {
        "preferred_categories": ["restaurant", "gallery", "hotel"],
        "vibe_affinity": {"vintage": 0.7, "artistic": 0.6, "luxurious": 0.5},
        "price_tier": "medium",
        "style_keywords": ["timeless", "heritage", "established"],
    },
}

# Interaction action weights — relative probability of each action type.
ACTION_WEIGHTS: dict[str, float] = {
    "click": 0.50,
    "view": 0.25,
    "save": 0.15,
    "convert": 0.07,
    "dismiss": 0.03,
}


# ---------------------------------------------------------------------------
# Synthetic entity generation
# ---------------------------------------------------------------------------


class SyntheticEntityGenerator:
    """Generate realistic-looking synthetic entity records.

    Each entity has a name derived from category-specific templates, a
    plausible ``vibe_dna`` vector, a set of tags, a description, and
    standard metadata (city, rating, price tier).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        logger.info("SyntheticEntityGenerator initialised (seed=%s)", seed)

    def generate_entity(
        self,
        category: str | None = None,
        city: str | None = None,
    ) -> dict[str, Any]:
        """Generate a single synthetic entity.

        Args:
            category: Entity category.  If ``None``, one is chosen at random
                from the available templates.
            city: City to assign.  If ``None``, one is chosen at random.

        Returns:
            A dict representing a full entity record.
        """
        if category is None:
            category = self._rng.choice(list(NAME_TEMPLATES.keys()))
        if city is None:
            city = self._rng.choice(CITIES)

        name = self._generate_name(category)
        vibe_dna = self._generate_vibe_dna(category)
        tags = self._generate_tags(category)
        description = self._generate_description(name, category, city, tags)

        entity = {
            "entity_id": f"syn_{uuid.uuid4().hex[:12]}",
            "name": name,
            "category": category,
            "city": city,
            "vibe_dna": vibe_dna,
            "vibe_score": round(sum(vibe_dna.values()) / len(vibe_dna), 4),
            "tags": tags,
            "description": description,
            "rating": round(self._rng.uniform(3.5, 5.0), 1),
            "price_tier": self._rng.choice(["$", "$$", "$$$", "$$$$"]),
            "is_verified": self._rng.random() < 0.7,
            "created_at": datetime.now(UTC).isoformat(),
            "synthetic": True,
        }

        logger.debug("Generated synthetic entity: %s (%s, %s)", name, category, city)
        return entity

    def generate_batch(
        self,
        count: int,
        distribution: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a batch of synthetic entities.

        Args:
            count: Total number of entities to generate.
            distribution: Optional mapping of ``category → count``.  If the
                values do not sum to *count*, remaining entities are assigned
                random categories.  If ``None``, categories are uniform-random.

        Returns:
            A list of entity dicts.
        """
        entities: list[dict[str, Any]] = []

        if distribution:
            for category, cat_count in distribution.items():
                for _ in range(cat_count):
                    entities.append(self.generate_entity(category=category))
            remaining = count - len(entities)
            for _ in range(max(remaining, 0)):
                entities.append(self.generate_entity())
        else:
            for _ in range(count):
                entities.append(self.generate_entity())

        logger.info(
            "Generated batch of %d synthetic entities (requested %d)",
            len(entities),
            count,
        )
        return entities[:count]

    # -- Internal helpers ---------------------------------------------------

    def _generate_name(self, category: str) -> str:
        """Build a name from category templates."""
        templates = NAME_TEMPLATES.get(category, NAME_TEMPLATES["cafe"])
        template = self._rng.choice(templates)
        adj = self._rng.choice(ADJECTIVES)
        noun = self._rng.choice(NOUNS)
        return template.format(adj=adj, noun=noun)

    def _generate_vibe_dna(self, category: str) -> dict[str, float]:
        """Generate a vibe DNA vector with category-specific bias.

        Each dimension gets a base random value in [0.1, 0.7] plus any
        category bias, then is clipped to [0, 1].
        """
        bias = VIBE_BIAS.get(category, {})
        dna: dict[str, float] = {}
        for dim in VIBE_DIMENSIONS:
            base = self._rng.uniform(0.1, 0.7)
            boosted = base + bias.get(dim, 0.0)
            dna[dim] = round(min(max(boosted, 0.0), 1.0), 3)
        return dna

    def _generate_tags(self, category: str) -> list[str]:
        """Select a realistic subset of tags for the category."""
        available = TAGS_BY_CATEGORY.get(category, TAGS_BY_CATEGORY["cafe"])
        tag_count = self._rng.randint(3, min(7, len(available)))
        return self._rng.sample(available, tag_count)

    def _generate_description(
        self,
        name: str,
        category: str,
        city: str,
        tags: list[str],
    ) -> str:
        """Compose a short synthetic description."""
        openings = [
            f"{name} is a {category} located in the heart of {city}.",
            f"Nestled in {city}, {name} offers a distinctive {category} experience.",
            f"Discover {name}, a curated {category} in {city}.",
            f"A {city}-based {category}, {name} stands out for its unique character.",
        ]
        tag_phrase = ", ".join(tags[:3]) if tags else category
        middles = [
            f"Known for its {tag_phrase} aesthetic,",
            f"Featuring {tag_phrase},",
            f"With a focus on {tag_phrase},",
        ]
        endings = [
            "it has become a favourite among locals and visitors alike.",
            "it draws a discerning crowd seeking something special.",
            "it delivers an experience that resonates with the culturally curious.",
            "it embodies the spirit of modern discovery.",
        ]
        return (
            f"{self._rng.choice(openings)} "
            f"{self._rng.choice(middles)} "
            f"{self._rng.choice(endings)}"
        )


# ---------------------------------------------------------------------------
# Synthetic user generation
# ---------------------------------------------------------------------------


class SyntheticUserGenerator:
    """Generate realistic synthetic user profiles.

    Users are assigned to one of four behavioural segments (luxury_seeker,
    budget_conscious, trend_follower, classic_taste), each with characteristic
    category preferences, vibe affinities, and price sensitivity.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        logger.info("SyntheticUserGenerator initialised (seed=%s)", seed)

    def generate_user(
        self,
        segment: str | None = None,
    ) -> dict[str, Any]:
        """Generate a single synthetic user profile.

        Args:
            segment: Behavioural segment.  If ``None``, one is chosen at
                random from the defined segments.

        Returns:
            A dict representing a user profile.
        """
        if segment is None:
            segment = self._rng.choice(list(USER_SEGMENTS.keys()))

        template = USER_SEGMENTS.get(segment, USER_SEGMENTS["classic_taste"])

        # Add noise to vibe affinity so users within a segment still differ
        vibe_affinity: dict[str, float] = {}
        for dim in VIBE_DIMENSIONS:
            base = template["vibe_affinity"].get(dim, 0.3)
            noisy = base + self._rng.gauss(0, 0.1)
            vibe_affinity[dim] = round(min(max(noisy, 0.0), 1.0), 3)

        # Pick a subset of preferred categories with some randomness
        preferred = list(template["preferred_categories"])
        if self._rng.random() < 0.3:
            extra = self._rng.choice(list(NAME_TEMPLATES.keys()))
            if extra not in preferred:
                preferred.append(extra)

        user = {
            "user_id": f"syn_u_{uuid.uuid4().hex[:10]}",
            "segment": segment,
            "preferred_categories": preferred,
            "vibe_affinity": vibe_affinity,
            "price_tier": template["price_tier"],
            "style_keywords": list(template["style_keywords"]),
            "interaction_count": self._rng.randint(5, 200),
            "account_age_days": self._rng.randint(7, 730),
            "city": self._rng.choice(CITIES),
            "created_at": datetime.now(UTC).isoformat(),
            "synthetic": True,
        }

        logger.debug("Generated synthetic user: %s (segment=%s)", user["user_id"], segment)
        return user

    def generate_batch(self, count: int) -> list[dict[str, Any]]:
        """Generate a batch of synthetic users.

        Args:
            count: Number of user profiles to generate.

        Returns:
            A list of user profile dicts.
        """
        users = [self.generate_user() for _ in range(count)]
        logger.info("Generated batch of %d synthetic users", len(users))
        return users


# ---------------------------------------------------------------------------
# Synthetic interaction generation
# ---------------------------------------------------------------------------


class SyntheticInteractionGenerator:
    """Generate realistic interaction logs between users and entities.

    Interaction likelihood is driven by user-entity affinity: the higher the
    affinity, the more likely the user is to click, save, and convert.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        logger.info("SyntheticInteractionGenerator initialised (seed=%s)", seed)

    def generate_interactions(
        self,
        users: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        """Generate a set of synthetic user-entity interactions.

        Users with higher affinity for an entity are sampled more often, and
        higher-affinity pairs are more likely to produce saves / conversions.

        Args:
            users: List of user profile dicts.
            entities: List of entity record dicts.
            count: Number of interactions to generate.

        Returns:
            A list of interaction dicts, each containing ``user_id``,
            ``entity_id``, ``action``, ``position``, ``timestamp``, etc.
        """
        if not users or not entities:
            logger.warning("Cannot generate interactions: users or entities empty")
            return []

        interactions: list[dict[str, Any]] = []

        for _ in range(count):
            user = self._rng.choice(users)
            entity = self._rng.choice(entities)
            affinity = self._compute_affinity(user, entity)

            action = self._choose_action(affinity)
            position = self._rng.randint(1, 10)
            was_recommended = self._rng.random() < 0.7

            interaction = {
                "interaction_id": f"syn_i_{uuid.uuid4().hex[:12]}",
                "user_id": user.get("user_id", "unknown"),
                "entity_id": entity.get("entity_id", "unknown"),
                "action": action,
                "position": position,
                "was_recommended": was_recommended,
                "affinity_score": round(affinity, 4),
                "session_id": f"syn_s_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(UTC).isoformat(),
                "time_to_action_ms": round(self._rng.uniform(200, 15000), 1),
                "synthetic": True,
            }
            interactions.append(interaction)

        logger.info(
            "Generated %d synthetic interactions from %d users x %d entities",
            len(interactions),
            len(users),
            len(entities),
        )
        return interactions

    def _compute_affinity(
        self,
        user: dict[str, Any],
        entity: dict[str, Any],
    ) -> float:
        """Compute affinity between a user and an entity.

        Affinity is the weighted combination of:
        1. **Category match** (0.4) -- is the entity's category in the user's
           preferred list?
        2. **Vibe overlap** (0.4) -- cosine-like similarity between user vibe
           affinity and entity vibe DNA.
        3. **City match** (0.2) -- bonus when user and entity are in the same city.

        Args:
            user: User profile dict.
            entity: Entity record dict.

        Returns:
            Affinity score in [0.0, 1.0].
        """
        # Category match
        preferred = user.get("preferred_categories", [])
        entity_cat = entity.get("category", "")
        category_score = 1.0 if entity_cat in preferred else 0.2

        # Vibe overlap
        user_vibe = user.get("vibe_affinity", {})
        entity_vibe = entity.get("vibe_dna", {})
        if user_vibe and entity_vibe:
            dot = sum(user_vibe.get(d, 0.0) * entity_vibe.get(d, 0.0) for d in VIBE_DIMENSIONS)
            mag_u = math.sqrt(sum(v**2 for v in user_vibe.values())) or 1.0
            mag_e = math.sqrt(sum(v**2 for v in entity_vibe.values())) or 1.0
            vibe_score = dot / (mag_u * mag_e)
        else:
            vibe_score = 0.5

        # City match
        city_score = 1.0 if user.get("city") == entity.get("city") else 0.3

        affinity = 0.4 * category_score + 0.4 * vibe_score + 0.2 * city_score
        return min(max(affinity, 0.0), 1.0)

    def _choose_action(self, affinity: float) -> str:
        """Choose an interaction action biased by affinity.

        Higher affinity increases the probability of high-value actions
        (save, convert) relative to low-value ones (view, dismiss).

        Args:
            affinity: User-entity affinity in [0.0, 1.0].

        Returns:
            Action string (``"click"``, ``"view"``, ``"save"``, ``"convert"``,
            or ``"dismiss"``).
        """
        # Adjust weights: boost high-value actions proportional to affinity
        adjusted: dict[str, float] = {}
        for action, base_weight in ACTION_WEIGHTS.items():
            if action in ("convert", "save"):
                adjusted[action] = base_weight * (0.5 + affinity)
            elif action == "dismiss":
                adjusted[action] = base_weight * (1.5 - affinity)
            else:
                adjusted[action] = base_weight

        total = sum(adjusted.values())
        r = self._rng.random() * total
        cumulative = 0.0
        for action, weight in adjusted.items():
            cumulative += weight
            if r <= cumulative:
                return action
        return "click"  # fallback


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate synthetic data for statistical fidelity and privacy safety.

    Provides two checks:

    1. **Statistical similarity** -- are the distributions of categories,
       cities, and vibe scores in the synthetic data close to the real stats?
    2. **Privacy** -- does any synthetic entity match a real entity too
       closely (risking re-identification)?
    """

    def __init__(self, similarity_threshold: float = 0.15) -> None:
        """
        Args:
            similarity_threshold: Maximum allowed KL-divergence-like distance
                between real and synthetic distributions for the data to be
                considered statistically similar.
        """
        self._threshold = similarity_threshold
        logger.info(
            "DataValidator initialised (similarity_threshold=%.2f)",
            similarity_threshold,
        )

    def validate_statistical_similarity(
        self,
        synthetic: list[dict[str, Any]],
        real_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare distributional properties of synthetic data to real stats.

        Args:
            synthetic: List of synthetic entity dicts.
            real_stats: Summary statistics of real data.  Expected keys:
                ``category_distribution`` (dict[str, float] — proportions),
                ``city_distribution`` (dict[str, float]),
                ``avg_vibe_score`` (float),
                ``vibe_score_std`` (float).

        Returns:
            A report dict containing per-dimension distances and an overall
            ``is_similar`` boolean.
        """
        report: dict[str, Any] = {
            "sample_size": len(synthetic),
            "checks": {},
            "is_similar": True,
        }

        if not synthetic:
            report["is_similar"] = False
            report["error"] = "No synthetic data to validate"
            return report

        # -- Category distribution
        syn_cats: dict[str, int] = defaultdict(int)
        for e in synthetic:
            syn_cats[e.get("category", "unknown")] += 1
        total = len(synthetic)
        syn_cat_dist = {k: v / total for k, v in syn_cats.items()}

        real_cat_dist = real_stats.get("category_distribution", {})
        cat_distance = self._distribution_distance(syn_cat_dist, real_cat_dist)
        report["checks"]["category_distribution"] = {
            "distance": round(cat_distance, 4),
            "pass": cat_distance <= self._threshold,
            "synthetic": syn_cat_dist,
            "real": real_cat_dist,
        }
        if cat_distance > self._threshold:
            report["is_similar"] = False

        # -- City distribution
        syn_cities: dict[str, int] = defaultdict(int)
        for e in synthetic:
            syn_cities[e.get("city", "unknown")] += 1
        syn_city_dist = {k: v / total for k, v in syn_cities.items()}

        real_city_dist = real_stats.get("city_distribution", {})
        city_distance = self._distribution_distance(syn_city_dist, real_city_dist)
        report["checks"]["city_distribution"] = {
            "distance": round(city_distance, 4),
            "pass": city_distance <= self._threshold,
            "synthetic": syn_city_dist,
            "real": real_city_dist,
        }
        if city_distance > self._threshold:
            report["is_similar"] = False

        # -- Vibe score mean / std
        vibe_scores = [e.get("vibe_score", 0.0) for e in synthetic]
        syn_mean = sum(vibe_scores) / len(vibe_scores)
        syn_std = math.sqrt(
            sum((v - syn_mean) ** 2 for v in vibe_scores) / max(len(vibe_scores) - 1, 1)
        )

        real_mean = real_stats.get("avg_vibe_score", 0.5)
        real_std = real_stats.get("vibe_score_std", 0.15)

        mean_diff = abs(syn_mean - real_mean)
        std_diff = abs(syn_std - real_std)

        report["checks"]["vibe_score"] = {
            "synthetic_mean": round(syn_mean, 4),
            "real_mean": real_mean,
            "mean_difference": round(mean_diff, 4),
            "synthetic_std": round(syn_std, 4),
            "real_std": real_std,
            "std_difference": round(std_diff, 4),
            "pass": mean_diff <= self._threshold and std_diff <= self._threshold,
        }
        if mean_diff > self._threshold or std_diff > self._threshold:
            report["is_similar"] = False

        logger.info(
            "Statistical similarity validation: is_similar=%s (cat=%.4f, city=%.4f, "
            "vibe_mean_diff=%.4f)",
            report["is_similar"],
            cat_distance,
            city_distance,
            mean_diff,
        )
        return report

    def validate_privacy(
        self,
        synthetic: list[dict[str, Any]],
        real: list[dict[str, Any]],
    ) -> bool:
        """Verify that no synthetic entity is too similar to a real one.

        Two entities are considered ``too similar`` if they share the same
        category, city, and have a name edit-distance ratio below 0.3
        (meaning the names are very close).

        Args:
            synthetic: List of synthetic entity dicts.
            real: List of real entity dicts.

        Returns:
            ``True`` if privacy is preserved (no overly similar pairs),
            ``False`` if any synthetic entity is too close to a real one.
        """
        for syn_entity in synthetic:
            syn_name = syn_entity.get("name", "").lower()
            syn_cat = syn_entity.get("category", "")
            syn_city = syn_entity.get("city", "")

            for real_entity in real:
                real_name = real_entity.get("name", "").lower()
                real_cat = real_entity.get("category", "")
                real_city = real_entity.get("city", "")

                # Quick filter: skip if category or city differ
                if syn_cat != real_cat or syn_city != real_city:
                    continue

                # Name similarity via normalised edit distance
                distance = self._levenshtein_distance(syn_name, real_name)
                max_len = max(len(syn_name), len(real_name), 1)
                similarity = 1 - (distance / max_len)

                if similarity > 0.7:
                    logger.warning(
                        "Privacy risk: synthetic '%s' too similar to real '%s' "
                        "(similarity=%.2f)",
                        syn_entity.get("name"),
                        real_entity.get("name"),
                        similarity,
                    )
                    return False

        logger.info("Privacy validation passed for %d synthetic entities", len(synthetic))
        return True

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _distribution_distance(
        dist_a: dict[str, float],
        dist_b: dict[str, float],
    ) -> float:
        """Compute symmetric Jensen-Shannon-like distance between two dists.

        Uses total variation distance (half L1 norm) for simplicity and
        robustness to zero-probability categories.
        """
        all_keys = set(dist_a.keys()) | set(dist_b.keys())
        if not all_keys:
            return 0.0
        return 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in all_keys)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return DataValidator._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_generator_instance: SyntheticEntityGenerator | None = None


def get_synthetic_generator() -> SyntheticEntityGenerator:
    """Return the singleton :class:`SyntheticEntityGenerator` instance.

    The generator is created on first call and reused thereafter.
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SyntheticEntityGenerator()
    return _generator_instance
