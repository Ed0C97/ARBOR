"""Seed database with sample lifestyle entities for testing."""

import asyncio
import uuid

# Sample entities for the lifestyle domain
SAMPLE_ENTITIES = [
    {
        "name": "Sartoria Napoli",
        "category": "tailoring",
        "city": "Napoli",
        "address": "Via Toledo 123, 80134 Napoli NA",
        "price_tier": 4,
        "description": "Traditional Neapolitan tailor specializing in spalla scesa construction",
        "vibe_dna": {
            "dimensions": {
                "formality": 85,
                "craftsmanship": 95,
                "price_value": 70,
                "atmosphere": 80,
                "exclusivity": 85,
                "service_quality": 90,
            },
            "tags": ["neapolitan", "handmade", "full-canvas", "bespoke"],
            "target_audience": "Expert",
        },
        "status": "icon",
    },
    {
        "name": "Calzoleria Classica",
        "category": "footwear",
        "city": "Firenze",
        "address": "Piazza della Signoria 45, 50122 Firenze FI",
        "price_tier": 5,
        "description": "Bespoke shoemaker with over 50 years of tradition",
        "vibe_dna": {
            "dimensions": {
                "formality": 90,
                "craftsmanship": 98,
                "price_value": 60,
                "atmosphere": 75,
                "exclusivity": 95,
                "service_quality": 95,
            },
            "tags": ["goodyear-welted", "bespoke", "handmade", "classic"],
            "target_audience": "Expert",
        },
        "status": "selected",
    },
    {
        "name": "Bottega del Gusto",
        "category": "food_drink",
        "city": "Roma",
        "address": "Via del Corso 88, 00186 Roma RM",
        "price_tier": 3,
        "description": "Curated wine bar with artisanal food pairings",
        "vibe_dna": {
            "dimensions": {
                "formality": 55,
                "craftsmanship": 80,
                "price_value": 85,
                "atmosphere": 90,
                "exclusivity": 60,
                "service_quality": 85,
            },
            "tags": ["wine", "artisanal", "casual-elegance", "local"],
            "target_audience": "Enthusiast",
        },
        "status": "vetted",
    },
    {
        "name": "Cravatte di Seta",
        "category": "accessories",
        "city": "Milano",
        "address": "Via Montenapoleone 12, 20121 Milano MI",
        "price_tier": 4,
        "description": "Seven-fold silk ties handmade in Como",
        "vibe_dna": {
            "dimensions": {
                "formality": 80,
                "craftsmanship": 92,
                "price_value": 75,
                "atmosphere": 70,
                "exclusivity": 80,
                "service_quality": 80,
            },
            "tags": ["silk", "seven-fold", "como", "handmade"],
            "target_audience": "Enthusiast",
        },
        "status": "selected",
    },
    {
        "name": "Profumeria Artigianale",
        "category": "fragrance",
        "city": "Firenze",
        "address": "Via Tornabuoni 67, 50123 Firenze FI",
        "price_tier": 4,
        "description": "Artisanal perfumery with custom scent creation",
        "vibe_dna": {
            "dimensions": {
                "formality": 70,
                "craftsmanship": 90,
                "price_value": 65,
                "atmosphere": 95,
                "exclusivity": 90,
                "service_quality": 95,
            },
            "tags": ["niche", "artisanal", "custom", "sensorial"],
            "target_audience": "HighSpender",
        },
        "status": "vetted",
    },
]


async def seed():
    """Seed the database with sample data."""
    from app.config import get_settings
    from app.db.postgres.connection import async_session_factory
    from app.db.postgres.models import Entity

    print("Seeding database with sample entities...")

    async with async_session_factory() as session:
        for data in SAMPLE_ENTITIES:
            entity = Entity(
                id=uuid.uuid4(),
                **data,
            )
            session.add(entity)

        await session.commit()
        print(f"Seeded {len(SAMPLE_ENTITIES)} entities successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
