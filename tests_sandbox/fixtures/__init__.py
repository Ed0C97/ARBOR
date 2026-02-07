"""Fixtures package for ARBOR test sandbox.

Pre-defined test data for consistent test runs.
"""

# Sample entities for quick testing
SAMPLE_RESTAURANT = {
    "id": "rest-001",
    "name": "Ristorante Test",
    "entity_type": "restaurant",
    "category": "Fine Dining",
    "city": "Milano",
    "description": "A sample restaurant for testing",
    "price_tier": 4,
    "vibe_dna": {
        "dimensions": {
            "food_quality": 0.9,
            "ambiance": 0.85,
            "service": 0.8,
            "value": 0.7,
        }
    }
}

SAMPLE_HOTEL = {
    "id": "hotel-001",
    "name": "Hotel Test",
    "entity_type": "hotel",
    "category": "Luxury",
    "city": "Roma",
    "description": "A sample hotel for testing",
    "stars": 5,
    "vibe_dna": {
        "dimensions": {
            "comfort": 0.95,
            "service": 0.9,
            "location": 0.85,
            "amenities": 0.88,
        }
    }
}

__all__ = ["SAMPLE_RESTAURANT", "SAMPLE_HOTEL"]
