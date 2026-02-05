"""Vector Agent - semantic search via Qdrant."""

import logging

from app.db.qdrant.hybrid_search import HybridSearch
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


class VectorAgent:
    """Search entities by semantic similarity in Qdrant."""

    def __init__(self):
        self.search = HybridSearch()
        self.gateway = get_llm_gateway()

    async def execute(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search for entities matching the query semantically."""
        # Generate query embedding
        query_embedding = await self.gateway.get_embedding(query)

        # Extract filter params
        category = (filters or {}).get("category")
        city = (filters or {}).get("city")

        results = self.search.search(
            query_vector=query_embedding,
            query_text=query,
            limit=limit,
            category=category,
            city=city,
        )

        logger.info(f"Vector search returned {len(results)} results for: {query[:50]}")
        return results
