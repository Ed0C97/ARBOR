"""Historian Agent - knowledge graph navigation via Neo4j."""

import logging

from app.db.neo4j.queries import Neo4jQueries

logger = logging.getLogger(__name__)


class HistorianAgent:
    """Navigate the Knowledge Graph for relationships and lineage."""

    def __init__(self):
        self.neo4j = Neo4jQueries()

    async def execute(
        self,
        intent: str,
        entities_mentioned: list[str],
        filters: dict | None = None,
    ) -> list[dict]:
        """Execute graph queries based on intent and entities."""
        results = []

        if intent == "HISTORY":
            for entity_name in entities_mentioned:
                lineage = await self.neo4j.find_lineage(entity_name)
                results.extend(
                    [
                        {
                            "type": "lineage",
                            "entity": r.get("entity"),
                            "mentor": r.get("mentor"),
                            "distance": r.get("distance"),
                        }
                        for r in lineage
                    ]
                )

        # Always try to find style-related entities
        for entity_name in entities_mentioned:
            related = await self.neo4j.find_related_by_style(entity_name)
            results.extend(
                [
                    {
                        "type": "style_related",
                        "name": r.get("name"),
                        "category": r.get("category"),
                        "city": r.get("city"),
                        "shared_style": r.get("shared_style"),
                    }
                    for r in related
                ]
            )

        # Check for brand queries
        style = (filters or {}).get("style")
        if style:
            brand_results = await self.neo4j.find_brand_retailers(
                style, (filters or {}).get("city")
            )
            results.extend(
                [
                    {
                        "type": "brand_retailer",
                        "name": r.get("name"),
                        "city": r.get("city"),
                        "relationship": r.get("relationship_type"),
                    }
                    for r in brand_results
                ]
            )

        logger.info(f"Graph search returned {len(results)} results")
        return results
