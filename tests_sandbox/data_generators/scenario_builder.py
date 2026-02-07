"""Scenario Builder - Combine generators to create complete test scenarios.

Creates realistic combinations of domains, entities, and edge cases
for comprehensive system testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .domain_profile_factory import DomainProfileFactory, generate_domain_intake
from .entity_factory import EntityFactory


@dataclass
class TestScenario:
    """A complete test scenario with domain, entities, and metadata."""
    
    scenario_id: str
    scenario_name: str
    description: str
    domain_intake: Any  # DomainIntake
    entities: list[dict[str, Any]]
    expected_outcomes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)


class ScenarioBuilder:
    """Build complete test scenarios for system testing.
    
    Usage:
        builder = ScenarioBuilder(locale="it_IT", seed=42)
        
        # Build a complete fine dining scenario
        scenario = builder.fine_dining_scenario(entity_count=50)
        
        # Build a scenario with data quality issues
        scenario = builder.scenario_with_issues(
            issue_types=["missing_fields", "invalid_values"]
        )
        
        # Build stress test scenario
        scenario = builder.stress_test_scenario(entity_count=1000)
    """
    
    def __init__(self, locale: str = "it_IT", seed: int | None = None):
        """Initialize the builder.
        
        Args:
            locale: Faker locale
            seed: Random seed for reproducibility
        """
        self.locale = locale
        self.seed = seed
        self.domain_factory = DomainProfileFactory(locale=locale, seed=seed)
        self.entity_factory = EntityFactory(locale=locale, seed=seed)
        self._scenario_counter = 0
    
    def _next_id(self) -> str:
        """Generate next scenario ID."""
        self._scenario_counter += 1
        return f"scenario_{self._scenario_counter:04d}"
    
    def fine_dining_scenario(
        self,
        entity_count: int = 50,
        city: str = "Milano",
    ) -> TestScenario:
        """Build a fine dining restaurant scenario.
        
        Creates a coherent scenario with:
        - Fine dining domain profile
        - Mix of fine dining, trattoria, and casual restaurants
        - Expected score distributions
        """
        from .domain_profile_factory import DomainVertical
        
        domain_intake = self.domain_factory.generate(
            vertical=DomainVertical.FOOD_DINING,
            city=city,
            language="it",
        )
        
        # Generate entities with realistic distribution
        entities = self.entity_factory.generate_batch(
            entity_type="restaurant",
            count=entity_count,
            profile_distribution={
                "fine_dining": 0.2,  # 20% fine dining
                "trattoria": 0.5,    # 50% trattoria
                "casual": 0.3,       # 30% casual
            },
            city=city,
        )
        
        return TestScenario(
            scenario_id=self._next_id(),
            scenario_name=f"Fine Dining {city}",
            description="Complete fine dining scenario with mixed restaurant quality",
            domain_intake=domain_intake,
            entities=entities,
            expected_outcomes={
                "fine_dining_avg_score": (85, 100),  # Expected range
                "trattoria_avg_score": (60, 80),
                "casual_avg_score": (40, 65),
                "top_10_should_be_fine_dining": True,
            },
            metadata={
                "locale": self.locale,
                "seed": self.seed,
                "city": city,
            },
        )
    
    def hotel_luxury_scenario(
        self,
        entity_count: int = 30,
        city: str = "Roma",
    ) -> TestScenario:
        """Build a luxury hotel scenario."""
        from .domain_profile_factory import DomainVertical
        
        domain_intake = self.domain_factory.generate(
            vertical=DomainVertical.HOSPITALITY_HOTELS,
            city=city,
            language="it",
        )
        
        entities = self.entity_factory.generate_batch(
            entity_type="hotel",
            count=entity_count,
            profile_distribution={
                "luxury": 0.3,
                "boutique": 0.4,
                "business": 0.3,
            },
            city=city,
        )
        
        return TestScenario(
            scenario_id=self._next_id(),
            scenario_name=f"Luxury Hotels {city}",
            description="Luxury and boutique hotel scenario",
            domain_intake=domain_intake,
            entities=entities,
            expected_outcomes={
                "luxury_avg_score": (85, 100),
                "boutique_avg_score": (70, 90),
                "business_avg_score": (50, 70),
            },
            metadata={
                "locale": self.locale,
                "seed": self.seed,
                "city": city,
            },
        )
    
    def scenario_with_issues(
        self,
        issue_types: list[str] | None = None,
        entity_count: int = 50,
        issue_ratio: float = 0.2,  # 20% of entities have issues
    ) -> TestScenario:
        """Build a scenario with intentional data quality issues.
        
        Args:
            issue_types: Types of issues to introduce
            entity_count: Total number of entities
            issue_ratio: Ratio of entities with issues (0.0 - 1.0)
        
        Returns:
            Scenario with mixed clean and problematic data
        """
        if issue_types is None:
            issue_types = ["missing_description", "invalid_price_tier"]
        
        from .domain_profile_factory import DomainVertical
        
        domain_intake, domain_errors = self.domain_factory.generate_with_errors(
            error_types=["missing_fields"] if "missing_fields" in issue_types else None,
            vertical=DomainVertical.FOOD_DINING,
        )
        
        # Generate clean entities
        clean_count = int(entity_count * (1 - issue_ratio))
        issue_count = entity_count - clean_count
        
        clean_entities = self.entity_factory.generate_batch(
            entity_type="restaurant",
            count=clean_count,
        )
        
        # Generate entities with issues
        issue_entities = []
        for _ in range(issue_count):
            # Pick random issue types for each entity
            entity_issues = random.sample(issue_types, k=random.randint(1, len(issue_types)))
            entity = self.entity_factory.restaurant(with_issues=entity_issues)
            issue_entities.append(entity)
        
        all_entities = clean_entities + issue_entities
        random.shuffle(all_entities)
        
        return TestScenario(
            scenario_id=self._next_id(),
            scenario_name="Data Quality Issues Scenario",
            description=f"Scenario with {issue_ratio*100:.0f}% data quality issues",
            domain_intake=domain_intake,
            entities=all_entities,
            expected_outcomes={
                "expected_validation_errors": issue_count,
                "clean_entity_count": clean_count,
                "issue_types": issue_types,
            },
            metadata={
                "issue_types": issue_types,
                "issue_ratio": issue_ratio,
                "domain_errors": domain_errors,
            },
        )
    
    def stress_test_scenario(
        self,
        entity_count: int = 1000,
        diverse: bool = True,
    ) -> TestScenario:
        """Build a stress test scenario with many entities.
        
        Args:
            entity_count: Number of entities (can be very large)
            diverse: If True, mix different entity types and profiles
        
        Returns:
            Large scenario for load/performance testing
        """
        from .domain_profile_factory import DomainVertical
        
        domain_intake = self.domain_factory.generate(
            vertical=DomainVertical.FOOD_DINING,
        )
        
        entities = []
        
        if diverse:
            # Mix restaurants and hotels
            restaurant_count = int(entity_count * 0.7)
            hotel_count = entity_count - restaurant_count
            
            entities.extend(self.entity_factory.generate_batch(
                entity_type="restaurant",
                count=restaurant_count,
            ))
            entities.extend(self.entity_factory.generate_batch(
                entity_type="hotel",
                count=hotel_count,
            ))
            random.shuffle(entities)
        else:
            entities = self.entity_factory.generate_batch(
                entity_type="restaurant",
                count=entity_count,
            )
        
        return TestScenario(
            scenario_id=self._next_id(),
            scenario_name=f"Stress Test ({entity_count} entities)",
            description=f"Large scale stress test with {entity_count} entities",
            domain_intake=domain_intake,
            entities=entities,
            expected_outcomes={
                "should_complete": True,
                "max_memory_mb": 1024,
                "max_duration_seconds": 300,
            },
            metadata={
                "stress_test": True,
                "entity_count": entity_count,
                "diverse": diverse,
            },
        )
    
    def edge_case_scenario(self) -> TestScenario:
        """Build a scenario with edge cases for boundary testing.
        
        Includes:
        - Empty strings
        - Very long strings
        - Unicode/emoji
        - Extreme values
        - Special characters
        """
        from .domain_profile_factory import DomainVertical
        
        domain_intake = self.domain_factory.generate(
            vertical=DomainVertical.FOOD_DINING,
        )
        
        entities = []
        
        # Normal baseline
        entities.extend(self.entity_factory.generate_batch(
            entity_type="restaurant",
            count=10,
        ))
        
        # Edge case: Empty name (should fail validation)
        edge1 = self.entity_factory.restaurant()
        edge1["name"] = ""
        edge1["_edge_case"] = "empty_name"
        entities.append(edge1)
        
        # Edge case: Very long name
        edge2 = self.entity_factory.restaurant()
        edge2["name"] = "A" * 500
        edge2["_edge_case"] = "long_name"
        entities.append(edge2)
        
        # Edge case: Unicode/emoji in name
        edge3 = self.entity_factory.restaurant()
        edge3["name"] = "Ristorante 🍕 La 日本 Bella"
        edge3["_edge_case"] = "unicode_name"
        entities.append(edge3)
        
        # Edge case: Special characters
        edge4 = self.entity_factory.restaurant()
        edge4["name"] = "L'Osteria dell'Arco (Centro) - N°1"
        edge4["_edge_case"] = "special_chars"
        entities.append(edge4)
        
        # Edge case: Extreme price tier
        edge5 = self.entity_factory.restaurant()
        edge5["price_tier"] = 0
        edge5["_edge_case"] = "zero_price_tier"
        entities.append(edge5)
        
        # Edge case: Negative vibe scores
        edge6 = self.entity_factory.restaurant()
        edge6["vibe_dna"]["dimensions"]["formality"] = -0.5
        edge6["_edge_case"] = "negative_vibe_score"
        entities.append(edge6)
        
        # Edge case: Vibe score > 1.0
        edge7 = self.entity_factory.restaurant()
        edge7["vibe_dna"]["dimensions"]["craftsmanship"] = 1.5
        edge7["_edge_case"] = "vibe_score_overflow"
        entities.append(edge7)
        
        return TestScenario(
            scenario_id=self._next_id(),
            scenario_name="Edge Cases Scenario",
            description="Boundary testing with edge cases",
            domain_intake=domain_intake,
            entities=entities,
            expected_outcomes={
                "should_handle_gracefully": True,
                "expected_validation_failures": 6,  # Edge cases that should fail
            },
            metadata={
                "edge_cases": [
                    "empty_name",
                    "long_name",
                    "unicode_name",
                    "special_chars",
                    "zero_price_tier",
                    "negative_vibe_score",
                    "vibe_score_overflow",
                ],
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def build_standard_test_suite(
    locale: str = "it_IT",
    seed: int = 42,
) -> list[TestScenario]:
    """Build a standard suite of test scenarios.
    
    Returns a comprehensive set of scenarios for complete system testing:
    - Fine dining scenario
    - Hotel luxury scenario
    - Data quality issues scenario
    - Edge cases scenario
    - Stress test scenario (smaller)
    """
    builder = ScenarioBuilder(locale=locale, seed=seed)
    
    return [
        builder.fine_dining_scenario(entity_count=50),
        builder.hotel_luxury_scenario(entity_count=30),
        builder.scenario_with_issues(entity_count=30, issue_ratio=0.3),
        builder.edge_case_scenario(),
        builder.stress_test_scenario(entity_count=100, diverse=True),  # Smaller for standard suite
    ]
