"""Temporal.io activity definitions for durable workflows."""

import logging

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def scrape_entity(source_url: str, scraper_type: str = "google_maps") -> dict:
    """Activity: Scrape a single entity from source URL."""
    from app.ingestion.scrapers.google_maps import GoogleMapsScraper
    from app.ingestion.scrapers.web_generic import WebGenericScraper

    scrapers = {
        "google_maps": GoogleMapsScraper,
        "web": WebGenericScraper,
    }
    scraper_cls = scrapers.get(scraper_type, WebGenericScraper)
    scraper = scraper_cls()

    raw = await scraper.scrape_single(source_url)
    if raw is None:
        raise ValueError(f"Failed to scrape: {source_url}")

    return {
        "name": raw.name,
        "address": raw.address,
        "phone": raw.phone,
        "website": raw.website,
        "reviews": raw.reviews,
        "images": raw.images,
        "source_url": raw.source_url,
        "source_type": raw.source_type,
        "latitude": raw.latitude,
        "longitude": raw.longitude,
        "raw_data": raw.raw_data,
    }


@activity.defn
async def analyze_with_vision(images: list[str]) -> dict:
    """Activity: Analyze images with GPT-4o Vision."""
    from app.ingestion.analyzers.vision import VisionAnalyzer

    analyzer = VisionAnalyzer()
    return await analyzer.analyze_images(images)


@activity.defn
async def extract_vibe(reviews: list[str], entity_name: str) -> dict:
    """Activity: Extract Vibe DNA from reviews."""
    from app.ingestion.analyzers.vibe_extractor import VibeExtractor

    extractor = VibeExtractor()
    return await extractor.extract_vibe(reviews, entity_name)


@activity.defn
async def generate_embedding(text: str) -> list[float]:
    """Activity: Generate text embedding."""
    from app.ingestion.analyzers.embedding import EmbeddingGenerator

    gen = EmbeddingGenerator()
    return await gen.generate(text)


@activity.defn
async def save_to_qdrant(entity_id: str, vector: list[float], payload: dict) -> None:
    """Activity: Save vector to Qdrant."""
    from app.db.qdrant.collections import QdrantCollections

    qdrant = QdrantCollections()
    qdrant.upsert_vector(entity_id, vector, payload)


@activity.defn
async def save_to_neo4j(entity_id: str, name: str, category: str, city: str | None) -> None:
    """Activity: Create entity node in Neo4j."""
    from app.db.neo4j.queries import Neo4jQueries

    neo4j = Neo4jQueries()
    await neo4j.create_entity_node(entity_id, name, category, city)
