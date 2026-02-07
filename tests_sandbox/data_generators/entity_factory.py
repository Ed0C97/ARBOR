"""Entity Factory - Generate realistic entities for testing.

Generates entities (restaurants, hotels, etc.) with realistic data
for testing the ML pipeline and scoring engine.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from faker import Faker


# ═══════════════════════════════════════════════════════════════════════════
# Entity templates per category
# ═══════════════════════════════════════════════════════════════════════════

RESTAURANT_PROFILES = {
    "fine_dining": {
        "price_tier_range": (4, 5),
        "vibe_profile": {
            "formality": (0.8, 1.0),
            "craftsmanship": (0.85, 1.0),
            "price_value": (0.3, 0.5),  # Lower because expensive
            "atmosphere": (0.8, 1.0),
            "service_quality": (0.85, 1.0),
            "exclusivity": (0.8, 1.0),
        },
        "name_patterns": [
            "Ristorante {surname}",
            "{surname}",
            "La {noun}",
            "Il {noun}",
        ],
        "categories": ["fine_dining", "contemporary", "italian", "french"],
        "tags": ["michelin", "tasting-menu", "wine-pairing", "reservation-required"],
    },
    "trattoria": {
        "price_tier_range": (2, 3),
        "vibe_profile": {
            "formality": (0.3, 0.5),
            "craftsmanship": (0.6, 0.8),
            "price_value": (0.7, 0.9),
            "atmosphere": (0.6, 0.8),
            "service_quality": (0.5, 0.7),
            "exclusivity": (0.2, 0.4),
        },
        "name_patterns": [
            "Trattoria {surname}",
            "Trattoria da {name}",
            "Osteria {surname}",
            "La Vecchia {noun}",
        ],
        "categories": ["trattoria", "traditional", "regional"],
        "tags": ["family-run", "homemade-pasta", "local-wines", "traditional"],
    },
    "casual": {
        "price_tier_range": (1, 2),
        "vibe_profile": {
            "formality": (0.1, 0.3),
            "craftsmanship": (0.4, 0.6),
            "price_value": (0.8, 1.0),
            "atmosphere": (0.4, 0.6),
            "service_quality": (0.4, 0.6),
            "exclusivity": (0.1, 0.2),
        },
        "name_patterns": [
            "Pizzeria {surname}",
            "{name}'s",
            "Bar {surname}",
        ],
        "categories": ["pizzeria", "fast_casual", "bar"],
        "tags": ["quick-service", "takeaway", "budget-friendly"],
    },
}

HOTEL_PROFILES = {
    "luxury": {
        "star_rating_range": (5, 5),
        "price_tier_range": (5, 5),
        "vibe_profile": {
            "room_quality": (0.9, 1.0),
            "location": (0.8, 1.0),
            "service_level": (0.9, 1.0),
            "facilities": (0.85, 1.0),
            "design_atmosphere": (0.85, 1.0),
            "uniqueness": (0.7, 0.9),
        },
        "name_patterns": [
            "{brand} {city}",
            "Grand Hotel {surname}",
            "Palazzo {surname}",
            "Hotel {surname}",
        ],
        "categories": ["luxury", "5_star", "palace"],
        "tags": ["spa", "fine-dining", "concierge", "rooftop"],
    },
    "boutique": {
        "star_rating_range": (4, 4),
        "price_tier_range": (3, 4),
        "vibe_profile": {
            "room_quality": (0.7, 0.85),
            "location": (0.7, 0.9),
            "service_level": (0.7, 0.85),
            "facilities": (0.5, 0.7),
            "design_atmosphere": (0.8, 0.95),
            "uniqueness": (0.8, 1.0),
        },
        "name_patterns": [
            "Hotel {adjective} {noun}",
            "{surname} Boutique Hotel",
            "The {noun}",
        ],
        "categories": ["boutique", "design", "4_star"],
        "tags": ["design-hotel", "instagram-worthy", "artsy", "local-character"],
    },
    "business": {
        "star_rating_range": (3, 4),
        "price_tier_range": (2, 3),
        "vibe_profile": {
            "room_quality": (0.6, 0.75),
            "location": (0.7, 0.9),
            "service_level": (0.6, 0.75),
            "facilities": (0.6, 0.8),
            "design_atmosphere": (0.4, 0.6),
            "uniqueness": (0.2, 0.4),
        },
        "name_patterns": [
            "{brand} {city}",
            "Business Hotel {surname}",
            "Hotel {city} Central",
        ],
        "categories": ["business", "3_star", "4_star"],
        "tags": ["business-center", "meeting-rooms", "airport-shuttle"],
    },
}

# Vocabulary for name generation
NOUNS_IT = [
    "Pergola", "Terrazza", "Corte", "Cascina", "Locanda", "Cantina",
    "Bottega", "Torre", "Villa", "Vigna", "Fontana", "Olivo",
]

ADJECTIVES_IT = [
    "Antica", "Bella", "Piccola", "Grande", "Verde", "Bianca",
    "Dorata", "Nascosta", "Segreta", "Reale",
]

LUXURY_BRANDS = [
    "Four Seasons", "Bulgari", "Armani", "Mandarin Oriental",
    "Park Hyatt", "St. Regis", "Ritz-Carlton", "Aman",
]

BUSINESS_BRANDS = [
    "NH", "Hilton", "Marriott", "Novotel", "Holiday Inn",
    "Best Western", "Radisson", "AC Hotels",
]


# ═══════════════════════════════════════════════════════════════════════════
# Entity Factory
# ═══════════════════════════════════════════════════════════════════════════

class EntityFactory:
    """Factory for generating realistic entities for testing.
    
    Usage:
        factory = EntityFactory(locale="it_IT", seed=42)
        
        # Generate a single restaurant
        restaurant = factory.restaurant(profile="fine_dining")
        
        # Generate multiple entities
        entities = factory.generate_batch(
            entity_type="restaurant",
            count=50,
            profile_distribution={"fine_dining": 0.2, "trattoria": 0.5, "casual": 0.3}
        )
    """
    
    def __init__(self, locale: str = "it_IT", seed: int | None = None):
        """Initialize the factory.
        
        Args:
            locale: Faker locale
            seed: Random seed for reproducibility
        """
        self.faker = Faker(locale)
        self.locale = locale
        
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
    
    def restaurant(
        self,
        profile: str = "fine_dining",
        city: str | None = None,
        with_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a realistic restaurant entity.
        
        Args:
            profile: One of "fine_dining", "trattoria", "casual"
            city: Specific city, or random if None
            with_issues: List of data quality issues to introduce:
                - "missing_description": No description
                - "missing_address": No address
                - "invalid_price_tier": Price tier out of range
                - "incomplete_vibe": Some vibe dimensions missing
        
        Returns:
            Restaurant entity dict
        """
        config = RESTAURANT_PROFILES.get(profile, RESTAURANT_PROFILES["casual"])
        city = city or self.faker.city()
        
        # Generate name
        name = self._generate_name(config["name_patterns"])
        
        # Generate vibe dimensions
        vibe_dna = self._generate_vibe_dimensions(config["vibe_profile"])
        
        # Base entity
        entity = {
            "id": str(uuid.uuid4()),
            "source_id": random.randint(10000, 99999),
            "entity_type": "restaurant",
            "name": name,
            "category": random.choice(config["categories"]),
            "city": city,
            "address": f"{self.faker.street_address()}, {city}",
            "price_tier": random.randint(*config["price_tier_range"]),
            "status": "vetted",
            "verified": True,
            "description": self._generate_description("restaurant", profile),
            "vibe_dna": {
                "dimensions": vibe_dna,
                "tags": random.sample(config["tags"], k=min(3, len(config["tags"]))),
                "signature_items": self._generate_signature_items("restaurant"),
                "target_audience": self._generate_target_audience("restaurant", profile),
                "summary": self._generate_summary("restaurant", name, profile),
            },
            "phone": self.faker.phone_number(),
            "website": f"https://{name.lower().replace(' ', '-')}.{self.faker.tld()}",
            "images": [f"https://images.example.com/{uuid.uuid4()}.jpg" for _ in range(3)],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "generated_by": "EntityFactory",
                "profile": profile,
                "locale": self.locale,
            },
        }
        
        # Introduce issues if requested
        if with_issues:
            entity = self._introduce_issues(entity, with_issues)
        
        return entity
    
    def hotel(
        self,
        profile: str = "luxury",
        city: str | None = None,
        with_issues: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a realistic hotel entity.
        
        Args:
            profile: One of "luxury", "boutique", "business"
            city: Specific city, or random if None
            with_issues: List of data quality issues to introduce
        
        Returns:
            Hotel entity dict
        """
        config = HOTEL_PROFILES.get(profile, HOTEL_PROFILES["business"])
        city = city or self.faker.city()
        
        # Generate name
        name = self._generate_hotel_name(config["name_patterns"], city, profile)
        
        # Generate vibe dimensions
        vibe_dna = self._generate_vibe_dimensions(config["vibe_profile"])
        
        entity = {
            "id": str(uuid.uuid4()),
            "source_id": random.randint(10000, 99999),
            "entity_type": "hotel",
            "name": name,
            "category": random.choice(config["categories"]),
            "city": city,
            "address": f"{self.faker.street_address()}, {city}",
            "price_tier": random.randint(*config["price_tier_range"]),
            "star_rating": random.randint(*config["star_rating_range"]),
            "status": "vetted",
            "verified": True,
            "description": self._generate_description("hotel", profile),
            "vibe_dna": {
                "dimensions": vibe_dna,
                "tags": random.sample(config["tags"], k=min(3, len(config["tags"]))),
                "amenities": self._generate_amenities(profile),
                "target_audience": self._generate_target_audience("hotel", profile),
                "summary": self._generate_summary("hotel", name, profile),
            },
            "phone": self.faker.phone_number(),
            "website": f"https://{name.lower().replace(' ', '-')}.{self.faker.tld()}",
            "images": [f"https://images.example.com/{uuid.uuid4()}.jpg" for _ in range(5)],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "generated_by": "EntityFactory",
                "profile": profile,
                "locale": self.locale,
            },
        }
        
        if with_issues:
            entity = self._introduce_issues(entity, with_issues)
        
        return entity
    
    def generate_batch(
        self,
        entity_type: str,
        count: int,
        profile_distribution: dict[str, float] | None = None,
        city: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a batch of entities.
        
        Args:
            entity_type: "restaurant" or "hotel"
            count: Number of entities to generate
            profile_distribution: Dict of profile -> probability (must sum to 1.0)
            city: Constrain all entities to this city
        
        Returns:
            List of entity dicts
        """
        if entity_type == "restaurant":
            profiles = list(RESTAURANT_PROFILES.keys())
            generator = self.restaurant
        elif entity_type == "hotel":
            profiles = list(HOTEL_PROFILES.keys())
            generator = self.hotel
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        if profile_distribution is None:
            # Default: equal distribution
            profile_distribution = {p: 1.0 / len(profiles) for p in profiles}
        
        entities = []
        profile_list = list(profile_distribution.keys())
        weights = list(profile_distribution.values())
        
        for _ in range(count):
            profile = random.choices(profile_list, weights=weights)[0]
            entity = generator(profile=profile, city=city)
            entities.append(entity)
        
        return entities
    
    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────
    
    def _generate_name(self, patterns: list[str]) -> str:
        """Generate a name from patterns."""
        pattern = random.choice(patterns)
        
        return pattern.format(
            name=self.faker.first_name(),
            surname=self.faker.last_name(),
            noun=random.choice(NOUNS_IT),
            adjective=random.choice(ADJECTIVES_IT),
        )
    
    def _generate_hotel_name(self, patterns: list[str], city: str, profile: str) -> str:
        """Generate a hotel name."""
        pattern = random.choice(patterns)
        
        brand = random.choice(LUXURY_BRANDS if profile == "luxury" else BUSINESS_BRANDS)
        
        return pattern.format(
            brand=brand,
            city=city,
            surname=self.faker.last_name(),
            noun=random.choice(NOUNS_IT),
            adjective=random.choice(ADJECTIVES_IT),
        )
    
    def _generate_vibe_dimensions(self, profile: dict[str, tuple[float, float]]) -> dict[str, float]:
        """Generate vibe dimensions based on profile ranges."""
        dimensions = {}
        for dim_name, (low, high) in profile.items():
            dimensions[dim_name] = round(random.uniform(low, high), 2)
        return dimensions
    
    def _generate_description(self, entity_type: str, profile: str) -> str:
        """Generate a realistic description."""
        if entity_type == "restaurant":
            if profile == "fine_dining":
                return self.faker.paragraph(nb_sentences=3) + " " + random.choice([
                    "Esperienza gastronomica d'eccellenza.",
                    "Cucina raffinata e servizio impeccabile.",
                    "Un viaggio culinario indimenticabile.",
                ])
            elif profile == "trattoria":
                return random.choice([
                    "Cucina casalinga della tradizione italiana.",
                    "I sapori autentici della nonna.",
                    "Ricette tramandate di generazione in generazione.",
                ])
            else:
                return self.faker.paragraph(nb_sentences=2)
        else:
            return self.faker.paragraph(nb_sentences=3)
    
    def _generate_signature_items(self, entity_type: str) -> list[str]:
        """Generate signature items."""
        if entity_type == "restaurant":
            items = [
                "Risotto allo Zafferano", "Ossobuco alla Milanese",
                "Tagliatelle al Ragù", "Cotoletta alla Milanese",
                "Tiramisù della Casa", "Panna Cotta",
                "Tortelli di Zucca", "Brasato al Barolo",
            ]
            return random.sample(items, k=random.randint(2, 4))
        return []
    
    def _generate_amenities(self, profile: str) -> list[str]:
        """Generate hotel amenities."""
        base = ["wifi", "room_service", "concierge"]
        
        if profile == "luxury":
            return base + ["spa", "pool", "fine_dining", "fitness", "valet"]
        elif profile == "boutique":
            return base + ["rooftop_bar", "art_gallery", "bike_rental"]
        else:
            return base + ["business_center", "meeting_rooms"]
    
    def _generate_target_audience(self, entity_type: str, profile: str) -> str:
        """Generate target audience description."""
        audiences = {
            ("restaurant", "fine_dining"): "Appassionati di alta cucina e buongustai esigenti",
            ("restaurant", "trattoria"): "Famiglie e amanti della cucina tradizionale",
            ("restaurant", "casual"): "Clientela giovane in cerca di qualità a buon prezzo",
            ("hotel", "luxury"): "Viaggiatori d'élite in cerca di esperienze esclusive",
            ("hotel", "boutique"): "Viaggiatori alla ricerca di design e atmosfera unica",
            ("hotel", "business"): "Professionisti e viaggiatori d'affari",
        }
        return audiences.get((entity_type, profile), "Clientela generale")
    
    def _generate_summary(self, entity_type: str, name: str, profile: str) -> str:
        """Generate entity summary."""
        return f"{name}: {self._generate_target_audience(entity_type, profile).lower()}."
    
    def _introduce_issues(self, entity: dict, issues: list[str]) -> dict:
        """Introduce data quality issues for testing."""
        if "missing_description" in issues:
            entity["description"] = ""
        
        if "missing_address" in issues:
            entity["address"] = ""
        
        if "invalid_price_tier" in issues:
            entity["price_tier"] = 99  # Invalid
        
        if "incomplete_vibe" in issues:
            # Remove some dimensions
            dims = entity.get("vibe_dna", {}).get("dimensions", {})
            if dims and len(dims) > 2:
                keys = list(dims.keys())
                del dims[keys[0]]
        
        return entity


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_restaurants(
    count: int,
    profile_distribution: dict[str, float] | None = None,
    locale: str = "it_IT",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to generate restaurants.
    
    Args:
        count: Number of restaurants
        profile_distribution: Dict of profile -> probability
        locale: Faker locale
        seed: Random seed
    
    Returns:
        List of restaurant entities
    """
    factory = EntityFactory(locale=locale, seed=seed)
    return factory.generate_batch("restaurant", count, profile_distribution)


def generate_hotels(
    count: int,
    profile_distribution: dict[str, float] | None = None,
    locale: str = "it_IT",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to generate hotels.
    
    Args:
        count: Number of hotels
        profile_distribution: Dict of profile -> probability
        locale: Faker locale
        seed: Random seed
    
    Returns:
        List of hotel entities
    """
    factory = EntityFactory(locale=locale, seed=seed)
    return factory.generate_batch("hotel", count, profile_distribution)
