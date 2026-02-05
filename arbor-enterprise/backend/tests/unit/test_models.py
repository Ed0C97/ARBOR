"""Unit tests for SQLAlchemy models."""

import uuid

from app.db.postgres.models import Entity, AbstractEntity, Curator


class TestEntityModel:
    def test_entity_defaults(self):
        entity = Entity(name="Test Shop", category="tailoring")
        assert entity.status == "pending"
        assert entity.verified is False

    def test_entity_fields(self):
        entity = Entity(
            name="Test Shop",
            category="tailoring",
            city="Roma",
            price_tier=3,
            vibe_dna={"dimensions": {"formality": 80}},
        )
        assert entity.name == "Test Shop"
        assert entity.city == "Roma"
        assert entity.price_tier == 3
        assert entity.vibe_dna["dimensions"]["formality"] == 80


class TestAbstractEntityModel:
    def test_abstract_entity_creation(self):
        brand = AbstractEntity(
            name="Kiton",
            category="clothing",
            origin_country="Italy",
        )
        assert brand.name == "Kiton"
        assert brand.origin_country == "Italy"
