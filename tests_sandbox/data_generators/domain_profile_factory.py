"""Domain Profile Factory - Generate realistic DomainIntake for testing.

Uses Faker for realistic data and Hypothesis for property-based testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from faker import Faker
from hypothesis import strategies as st

# Import ARBOR domain types
import sys
sys.path.insert(0, str(__file__).replace("tests_sandbox/data_generators/domain_profile_factory.py", "arbor-enterprise/backend"))

try:
    from app.core.domain_profile_service import (
        DomainIntake,
        DomainVertical,
        EntityTypeSpec,
        QualityAspect,
        VERTICAL_QUALITY_ASPECTS,
    )
except ImportError:
    # Fallback definitions for standalone testing
    from enum import Enum
    
    class DomainVertical(str, Enum):
        FOOD_DINING = "food_dining"
        FASHION_RETAIL = "fashion_retail"
        HOSPITALITY_HOTELS = "hospitality_hotels"
        REAL_ESTATE = "real_estate"
        HEALTH_WELLNESS = "health_wellness"
        OTHER = "other"
    
    @dataclass
    class EntityTypeSpec:
        name: str
        description: str = ""
        example_entities: list[str] = field(default_factory=list)
    
    @dataclass
    class QualityAspect:
        aspect_id: str
        importance: int = 3
        what_makes_it_great: str = ""
        what_makes_it_poor: str = ""
    
    @dataclass
    class DomainIntake:
        domain_name: str
        vertical: DomainVertical = DomainVertical.OTHER
        geographic_focus: str = ""
        language: str = "en"
        target_audience_description: str = ""
        audience_expertise_level: str = "mixed"
        entity_types: list[EntityTypeSpec] = field(default_factory=list)
        quality_aspects: list[QualityAspect] = field(default_factory=list)
        advisor_tone: str = "warm_expert"
        known_categories: list[str] = field(default_factory=list)
        sample_best_entities: list[str] = field(default_factory=list)
        sample_average_entities: list[str] = field(default_factory=list)
    
    VERTICAL_QUALITY_ASPECTS = {}


# ═══════════════════════════════════════════════════════════════════════════
# Vertical-specific data
# ═══════════════════════════════════════════════════════════════════════════

VERTICAL_DATA = {
    DomainVertical.FOOD_DINING: {
        "domain_names": [
            "Fine Dining {city}",
            "Guida Gastronomica {city}",
            "{city} Restaurant Guide",
            "Cucina d'Elite {city}",
            "Gourmet {city}",
        ],
        "entity_types": [
            {"name": "ristorante", "description": "Ristorante con servizio al tavolo"},
            {"name": "trattoria", "description": "Cucina tradizionale italiana"},
            {"name": "wine_bar", "description": "Bar con selezione di vini"},
            {"name": "pasticceria", "description": "Dolci e pasticceria artigianale"},
        ],
        "categories": [
            "fine_dining", "trattoria", "osteria", "pizzeria", "sushi",
            "seafood", "steakhouse", "vegetarian", "wine_bar", "cocktail_bar"
        ],
        "best_entities": [
            "Osteria Francescana", "Cracco", "D'O", "Piazza Duomo",
            "Le Calandre", "Da Vittorio", "Reale", "Uliassi"
        ],
        "average_entities": [
            "Trattoria da Mario", "Pizzeria Bella Napoli", 
            "Ristorante Il Giardino", "Osteria del Sole"
        ],
        "audiences": [
            "Food enthusiasts and critics",
            "Luxury dining seekers",
            "Tourists looking for authentic experiences",
            "Business professionals for client entertainment"
        ],
    },
    DomainVertical.HOSPITALITY_HOTELS: {
        "domain_names": [
            "Luxury Hotels {city}",
            "Hotel Guide {city}",
            "Boutique Hotels {city}",
            "Hospitality Excellence {city}",
        ],
        "entity_types": [
            {"name": "hotel", "description": "Full-service hotel"},
            {"name": "boutique_hotel", "description": "Design-focused boutique hotel"},
            {"name": "resort", "description": "Resort with spa and facilities"},
            {"name": "bed_and_breakfast", "description": "Intimate B&B experience"},
        ],
        "categories": [
            "luxury", "boutique", "business", "resort", "b&b",
            "design_hotel", "historic", "wellness", "family", "airport"
        ],
        "best_entities": [
            "Four Seasons Milano", "Bulgari Hotel", "Armani Hotel",
            "Mandarin Oriental", "Park Hyatt", "Aman Venice"
        ],
        "average_entities": [
            "Hotel Centrale", "Best Western City", 
            "NH Collection", "Holiday Inn Express"
        ],
        "audiences": [
            "Luxury travelers",
            "Business executives",
            "Honeymooners",
            "Conference organizers"
        ],
    },
    DomainVertical.FASHION_RETAIL: {
        "domain_names": [
            "Fashion Guide {city}",
            "Style Directory {city}",
            "Luxury Shopping {city}",
            "Boutique Finder {city}",
        ],
        "entity_types": [
            {"name": "boutique", "description": "Independent fashion boutique"},
            {"name": "flagship_store", "description": "Brand flagship store"},
            {"name": "concept_store", "description": "Multi-brand concept store"},
            {"name": "vintage_shop", "description": "Vintage and second-hand"},
        ],
        "categories": [
            "luxury", "contemporary", "streetwear", "vintage", "accessories",
            "shoes", "jewelry", "menswear", "womenswear", "sustainable"
        ],
        "best_entities": [
            "10 Corso Como", "Antonia", "Excelsior Milano",
            "Luisa Via Roma", "Biffi Boutiques"
        ],
        "average_entities": [
            "Zara", "H&M", "Mango", "COS"
        ],
        "audiences": [
            "Fashion-forward shoppers",
            "Luxury brand seekers",
            "Sustainable fashion advocates",
            "Style influencers"
        ],
    },
}

# Default data for other verticals
DEFAULT_VERTICAL_DATA = {
    "domain_names": [
        "Directory {city}",
        "Guide {city}",
        "Finder {city}",
    ],
    "entity_types": [
        {"name": "provider", "description": "Service provider"},
        {"name": "establishment", "description": "Business establishment"},
    ],
    "categories": ["premium", "standard", "budget", "specialized"],
    "best_entities": ["Top Provider A", "Excellence B", "Best C"],
    "average_entities": ["Standard D", "Regular E", "Common F"],
    "audiences": ["General consumers", "Professionals", "Enthusiasts"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Domain Profile Factory
# ═══════════════════════════════════════════════════════════════════════════

class DomainProfileFactory:
    """Factory for generating realistic DomainIntake objects.
    
    Usage:
        factory = DomainProfileFactory(locale="it_IT", seed=42)
        
        # Generate a random domain intake
        intake = factory.generate()
        
        # Generate for a specific vertical
        intake = factory.generate(vertical=DomainVertical.FOOD_DINING)
        
        # Generate with intentional errors (for testing validation)
        intake = factory.generate_with_errors(error_types=["missing_fields", "invalid_values"])
    """
    
    def __init__(self, locale: str = "it_IT", seed: int | None = None):
        """Initialize the factory.
        
        Args:
            locale: Faker locale for generating localized data
            seed: Random seed for reproducibility
        """
        self.faker = Faker(locale)
        self.locale = locale
        
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
    
    def generate(
        self,
        vertical: DomainVertical | None = None,
        city: str | None = None,
        language: str | None = None,
    ) -> DomainIntake:
        """Generate a realistic DomainIntake.
        
        Args:
            vertical: Specific vertical to use, or random if None
            city: City for geographic focus, or random if None
            language: Language code, or derived from locale if None
        
        Returns:
            A complete, valid DomainIntake object
        """
        # Choose vertical
        if vertical is None:
            vertical = random.choice(list(DomainVertical))
        
        # Get vertical-specific data
        vdata = VERTICAL_DATA.get(vertical, DEFAULT_VERTICAL_DATA)
        
        # Generate city and language
        city = city or self.faker.city()
        language = language or self.locale.split("_")[0]
        
        # Generate domain name
        domain_template = random.choice(vdata["domain_names"])
        domain_name = domain_template.format(city=city)
        
        # Generate entity types
        entity_types = []
        for et in vdata["entity_types"]:
            spec = EntityTypeSpec(
                name=et["name"],
                description=et["description"],
                example_entities=random.sample(
                    vdata.get("best_entities", []) + vdata.get("average_entities", []),
                    k=min(3, len(vdata.get("best_entities", []) + vdata.get("average_entities", [])))
                ) if vdata.get("best_entities") else []
            )
            entity_types.append(spec)
        
        # Generate quality aspects with realistic ratings
        quality_aspects = self._generate_quality_aspects(vertical)
        
        # Choose audience
        audience = random.choice(vdata["audiences"])
        expertise = random.choice(["novice", "intermediate", "expert", "mixed"])
        
        # Choose tone
        tone = random.choice(["formal_expert", "warm_expert", "casual_friend", "enthusiastic_guide"])
        
        return DomainIntake(
            domain_name=domain_name,
            vertical=vertical,
            geographic_focus=f"{city}, Italy" if "it" in self.locale else city,
            language=language,
            target_audience_description=audience,
            audience_expertise_level=expertise,
            entity_types=entity_types,
            quality_aspects=quality_aspects,
            advisor_tone=tone,
            known_categories=random.sample(vdata["categories"], k=min(5, len(vdata["categories"]))),
            sample_best_entities=vdata.get("best_entities", [])[:3],
            sample_average_entities=vdata.get("average_entities", [])[:2],
        )
    
    def generate_with_errors(
        self,
        error_types: list[str] | None = None,
        vertical: DomainVertical | None = None,
    ) -> tuple[DomainIntake, list[str]]:
        """Generate a DomainIntake with intentional errors for testing validation.
        
        Args:
            error_types: Types of errors to introduce. Options:
                - "missing_fields": Leave required fields empty
                - "invalid_values": Use out-of-range values
                - "wrong_types": Use wrong data types (as much as possible)
                - "duplicate_dimensions": Duplicate dimension IDs
                - "empty_entity_types": No entity types
            vertical: Specific vertical, or random if None
        
        Returns:
            Tuple of (DomainIntake with errors, list of error descriptions)
        """
        if error_types is None:
            error_types = ["missing_fields"]
        
        # Start with valid intake and then corrupt it
        intake = self.generate(vertical=vertical)
        errors_introduced = []
        
        if "missing_fields" in error_types:
            intake.domain_name = ""
            errors_introduced.append("domain_name is empty")
        
        if "invalid_values" in error_types:
            # Add quality aspect with invalid importance
            intake.quality_aspects.append(
                QualityAspect(
                    aspect_id="invalid_aspect",
                    importance=10,  # Should be 1-5
                    what_makes_it_great="",
                    what_makes_it_poor="",
                )
            )
            errors_introduced.append("quality_aspect importance=10 (should be 1-5)")
        
        if "empty_entity_types" in error_types:
            intake.entity_types = []
            errors_introduced.append("entity_types is empty")
        
        if "duplicate_dimensions" in error_types:
            # Add duplicate quality aspect
            if intake.quality_aspects:
                duplicate = QualityAspect(
                    aspect_id=intake.quality_aspects[0].aspect_id,
                    importance=3,
                    what_makes_it_great="Duplicate",
                    what_makes_it_poor="Duplicate",
                )
                intake.quality_aspects.append(duplicate)
                errors_introduced.append(f"duplicate aspect_id: {duplicate.aspect_id}")
        
        return intake, errors_introduced
    
    def _generate_quality_aspects(self, vertical: DomainVertical) -> list[QualityAspect]:
        """Generate quality aspects with realistic importance ratings."""
        aspects = []
        
        # Get vertical-specific aspects or defaults
        aspect_templates = VERTICAL_QUALITY_ASPECTS.get(vertical, [])
        if not aspect_templates:
            # Use generic aspects
            aspect_templates = [
                {"id": "core_quality", "label": "Core quality"},
                {"id": "experience", "label": "Overall experience"},
                {"id": "price_value", "label": "Price-to-value"},
                {"id": "service", "label": "Service quality"},
            ]
        
        for template in aspect_templates:
            # Realistic distribution: most aspects are important (3-5)
            importance = random.choices(
                [1, 2, 3, 4, 5],
                weights=[5, 10, 25, 35, 25]  # Higher weight for 4
            )[0]
            
            aspect = QualityAspect(
                aspect_id=template["id"],
                importance=importance,
                what_makes_it_great=self._generate_quality_description("great", template["id"]) if importance >= 4 else "",
                what_makes_it_poor=self._generate_quality_description("poor", template["id"]) if importance >= 4 else "",
            )
            aspects.append(aspect)
        
        return aspects
    
    def _generate_quality_description(self, quality: str, aspect_id: str) -> str:
        """Generate a realistic quality description."""
        great_phrases = [
            "Exceptional attention to detail",
            "World-class standards",
            "Consistently outstanding",
            "Sets the industry benchmark",
            "Unforgettable experience",
        ]
        
        poor_phrases = [
            "Lacks basic standards",
            "Inconsistent and unreliable",
            "Falls below expectations",
            "Poor attention to detail",
            "Forgettable or disappointing",
        ]
        
        return random.choice(great_phrases if quality == "great" else poor_phrases)


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis strategies
# ═══════════════════════════════════════════════════════════════════════════

@st.composite
def domain_intake_strategy(draw, vertical: DomainVertical | None = None):
    """Hypothesis strategy for generating DomainIntake objects.
    
    Usage with Hypothesis:
        from hypothesis import given
        
        @given(domain_intake_strategy())
        def test_profile_generation(intake):
            result = service.generate_profile(intake)
            assert result.is_valid
    """
    factory = DomainProfileFactory(seed=None)  # Let Hypothesis control randomness
    
    if vertical is None:
        vertical = draw(st.sampled_from(list(DomainVertical)))
    
    return factory.generate(vertical=vertical)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_domain_intake(
    vertical: DomainVertical | None = None,
    locale: str = "it_IT",
    seed: int | None = None,
) -> DomainIntake:
    """Convenience function to generate a single DomainIntake.
    
    Args:
        vertical: Specific vertical, or random if None
        locale: Faker locale
        seed: Random seed for reproducibility
    
    Returns:
        A complete, valid DomainIntake
    """
    factory = DomainProfileFactory(locale=locale, seed=seed)
    return factory.generate(vertical=vertical)


def generate_domain_intakes(
    count: int,
    verticals: list[DomainVertical] | None = None,
    locale: str = "it_IT",
    seed: int | None = None,
) -> list[DomainIntake]:
    """Generate multiple DomainIntake objects.
    
    Args:
        count: Number of intakes to generate
        verticals: Limit to specific verticals, or use all if None
        locale: Faker locale
        seed: Random seed for reproducibility
    
    Returns:
        List of DomainIntake objects
    """
    factory = DomainProfileFactory(locale=locale, seed=seed)
    intakes = []
    
    available_verticals = verticals or list(DomainVertical)
    
    for i in range(count):
        vertical = available_verticals[i % len(available_verticals)]
        intakes.append(factory.generate(vertical=vertical))
    
    return intakes
