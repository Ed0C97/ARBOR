"""Master Ingestor - orchestrates the full ingestion pipeline."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from app.db.neo4j.queries import Neo4jQueries
from app.db.qdrant.collections import QdrantCollections
from app.ingestion.analyzers.embedding import EmbeddingGenerator
from app.ingestion.analyzers.vibe_extractor import VibeExtractor
from app.ingestion.analyzers.vision import VisionAnalyzer
from app.ingestion.scrapers.base import BaseScraper, RawEntity

logger = logging.getLogger(__name__)


@dataclass
class ProcessedEntity:
    """Fully processed entity ready for database insertion."""

    raw: RawEntity
    vibe_from_reviews: dict = field(default_factory=dict)
    vibe_from_images: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    final_vibe_dna: dict = field(default_factory=dict)
    entity_id: str = ""


class MasterIngestor:
    """Orchestrate the full entity ingestion pipeline."""

    def __init__(
        self,
        scraper: BaseScraper,
        vision: VisionAnalyzer | None = None,
        vibe_extractor: VibeExtractor | None = None,
        embedding_gen: EmbeddingGenerator | None = None,
        qdrant: QdrantCollections | None = None,
        neo4j: Neo4jQueries | None = None,
    ):
        self.scraper = scraper
        self.vision = vision or VisionAnalyzer()
        self.vibe_extractor = vibe_extractor or VibeExtractor()
        self.embedding_gen = embedding_gen or EmbeddingGenerator()
        self.qdrant = qdrant or QdrantCollections()
        self.neo4j = neo4j or Neo4jQueries()

    async def ingest(
        self, query: str, location: str, category: str
    ) -> list[ProcessedEntity]:
        """Run the full ingestion pipeline for a query."""
        logger.info(f"Starting ingestion: '{query}' in {location} [{category}]")

        # Step 1: Scraping
        raw_entities = await self.scraper.scrape(query, location)
        logger.info(f"Scraped {len(raw_entities)} entities")

        processed = []
        for raw in raw_entities:
            try:
                result = await self._process_single(raw, category)
                processed.append(result)
            except Exception as e:
                logger.error(f"Failed to process {raw.name}: {e}")

        logger.info(f"Successfully processed {len(processed)}/{len(raw_entities)} entities")
        return processed

    async def ingest_single(self, source_url: str, category: str = "") -> ProcessedEntity | None:
        """Ingest a single entity from its URL."""
        raw = await self.scraper.scrape_single(source_url)
        if not raw:
            logger.warning(f"Could not scrape: {source_url}")
            return None

        return await self._process_single(raw, category)

    async def _process_single(self, raw: RawEntity, category: str) -> ProcessedEntity:
        """Process a single raw entity through the full pipeline."""
        entity_id = str(uuid.uuid4())

        # Step 2: Parallel analysis - Vision + Reviews
        vibe_images, vibe_reviews = await asyncio.gather(
            self.vision.analyze_images(raw.images),
            self.vibe_extractor.extract_vibe(raw.reviews, raw.name),
        )

        # Step 3: Merge Vibe DNA
        final_vibe = self._merge_vibes(vibe_images, vibe_reviews)

        # Step 4: Generate embedding
        embedding_data = {
            "name": raw.name,
            "category": category or raw.category or "",
            "tags": final_vibe.get("tags", []),
            "description": raw.raw_data.get("description", ""),
            "summary": final_vibe.get("summary", ""),
            "visual_summary": final_vibe.get("visual_summary", ""),
        }
        embedding = await self.embedding_gen.create_entity_embedding(embedding_data)

        # Step 5: Store in Qdrant
        payload = {
            "entity_id": entity_id,
            "name": raw.name,
            "category": category or raw.category or "",
            "city": self._extract_city(raw.address),
            "price_tier": final_vibe.get("price_tier"),
            "dimensions": final_vibe.get("dimensions", {}),
            "tags": final_vibe.get("tags", []),
            "status": "pending",
        }
        self.qdrant.upsert_vector(entity_id, embedding, payload)

        # Step 6: Create Neo4j node
        await self.neo4j.create_entity_node(
            entity_id=entity_id,
            name=raw.name,
            category=category or raw.category or "",
            city=self._extract_city(raw.address),
        )

        # Link to style if detected
        style = vibe_images.get("style") or final_vibe.get("visual_style")
        if style and style != "unknown":
            await self.neo4j.create_style_node(style)
            await self.neo4j.create_style_relationship(entity_id, style)

        return ProcessedEntity(
            raw=raw,
            vibe_from_reviews=vibe_reviews,
            vibe_from_images=vibe_images,
            embedding=embedding,
            final_vibe_dna=final_vibe,
            entity_id=entity_id,
        )

    def _merge_vibes(self, vibe_images: dict, vibe_reviews: dict) -> dict:
        """Merge Vibe DNA from images and reviews."""
        img_dims = vibe_images.get("dimensions", {})
        rev_dims = vibe_reviews.get("dimensions", {})
        all_keys = set(img_dims.keys()) | set(rev_dims.keys())

        merged_dims = {}
        for key in all_keys:
            img_val = img_dims.get(key, 50)
            rev_val = rev_dims.get(key, 50)
            # Reviews more reliable for price/service
            if key in ("price_value", "service_quality"):
                merged_dims[key] = int(0.3 * img_val + 0.7 * rev_val)
            else:
                merged_dims[key] = int(0.5 * img_val + 0.5 * rev_val)

        return {
            "dimensions": merged_dims,
            "tags": list(set(vibe_images.get("tags", []) + vibe_reviews.get("tags", []))),
            "signature_items": vibe_reviews.get("signature_items", []),
            "target_audience": vibe_reviews.get("target_audience", "General"),
            "visual_style": vibe_images.get("style", ""),
            "visual_summary": vibe_images.get("visual_summary", ""),
            "summary": vibe_reviews.get("summary", ""),
        }

    def _extract_city(self, address: str | None) -> str | None:
        """Extract city from address string (simple heuristic)."""
        if not address:
            return None
        parts = [p.strip() for p in address.split(",")]
        if len(parts) >= 2:
            return parts[-2].strip()
        return None
