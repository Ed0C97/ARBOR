"""Embedding generation for entity vectors."""

import logging

from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for entities."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        return await self.gateway.get_embedding(text)

    async def create_entity_embedding(self, entity_data: dict) -> list[float]:
        """Create a rich embedding combining multiple entity attributes."""
        text_parts = [
            entity_data.get("name", ""),
            entity_data.get("category", ""),
            " ".join(entity_data.get("tags", [])),
            entity_data.get("description", ""),
            entity_data.get("curator_notes", ""),
            entity_data.get("summary", ""),
            entity_data.get("visual_summary", ""),
        ]
        combined_text = " | ".join(filter(None, text_parts))

        if not combined_text.strip():
            combined_text = entity_data.get("name", "unknown entity")

        return await self.generate(combined_text)
