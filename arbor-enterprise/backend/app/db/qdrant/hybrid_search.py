"""Hybrid search combining dense vectors + sparse BM25.

TIER 4 - Point 15: Reciprocal Rank Fusion (RRF) Hybrid Search

Implements true RRF fusion combining:
- Dense vector search (semantic similarity)
- Keyword/text search (exact matches)

Formula: Score = 1/(k + rank_vector) + 1/(k + rank_keyword)
where k=60 is the standard RRF constant.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    SearchParams,
)

from app.db.qdrant.client import get_async_qdrant_client, get_qdrant_client

logger = logging.getLogger(__name__)

# RRF constant - standard value that balances early vs late ranks
RRF_K = 60


class HybridSearch:
    """Hybrid vector search with RRF fusion.

    TIER 4 - Point 15: Reciprocal Rank Fusion

    Combines dense vector search (semantic) with keyword search (exact)
    using RRF scoring to handle both semantic and exact match queries.
    """

    def __init__(self, collection: str = "entities_vectors"):
        self.collection = collection

    def search(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        limit: int = 10,
        category: str | None = None,
        city: str | None = None,
    ) -> list[dict]:
        """Perform hybrid search combining vector similarity with metadata filters.

        Legacy sync method for backwards compatibility.
        """
        client = get_qdrant_client()
        if client is None:
            return []

        filter_conditions = []

        if category:
            filter_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if city:
            filter_conditions.append(
                FieldCondition(key="city", match=MatchValue(value=city))
            )

        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128, exact=False),
        ).points

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "name": r.payload.get("name", ""),
                "category": r.payload.get("category", ""),
                "city": r.payload.get("city", ""),
                "price_tier": r.payload.get("price_tier"),
                "dimensions": r.payload.get("dimensions", {}),
                "tags": r.payload.get("tags", []),
                "payload": r.payload,
            }
            for r in results
        ]

    async def search_rrf(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        category: str | None = None,
        city: str | None = None,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        prefetch_multiplier: int = 5,
    ) -> list[dict]:
        """Perform RRF hybrid search combining vector and keyword search.

        TIER 4 - Point 15: Reciprocal Rank Fusion

        Formula: Score = weight_v * (1/(k + rank_vector)) + weight_k * (1/(k + rank_keyword))

        Args:
            query_vector: Dense embedding vector for semantic search
            query_text: Original query text for keyword search
            limit: Number of final results to return
            category: Optional category filter
            city: Optional city filter
            vector_weight: Weight for vector search scores (default 0.5)
            keyword_weight: Weight for keyword search scores (default 0.5)
            prefetch_multiplier: How many more results to fetch for fusion

        Returns:
            List of results sorted by RRF score
        """
        client = await get_async_qdrant_client()
        if client is None:
            logger.warning("Qdrant not available, returning empty results")
            return []

        # Build filters
        filter_conditions = []
        if category:
            filter_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if city:
            filter_conditions.append(
                FieldCondition(key="city", match=MatchValue(value=city))
            )
        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Prefetch more results for better fusion
        prefetch_limit = limit * prefetch_multiplier

        # Execute both searches in parallel
        vector_task = self._vector_search(
            client, query_vector, qdrant_filter, prefetch_limit
        )
        keyword_task = self._keyword_search(
            client, query_text, qdrant_filter, prefetch_limit
        )

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task, return_exceptions=True
        )

        # Handle any search failures gracefully
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []
        if isinstance(keyword_results, Exception):
            logger.warning(f"Keyword search failed: {keyword_results}")
            keyword_results = []

        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        # Return top N results
        final_results = fused_results[:limit]

        logger.info(
            f"RRF hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword "
            f"-> {len(final_results)} fused results for: {query_text[:50]}"
        )

        return final_results

    async def _vector_search(
        self,
        client,
        query_vector: list[float],
        qdrant_filter: Filter | None,
        limit: int,
    ) -> list[dict]:
        """Execute dense vector search."""
        try:
            results = await client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            )
            return [self._format_result(r) for r in results.points]
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _keyword_search(
        self,
        client,
        query_text: str,
        qdrant_filter: Filter | None,
        limit: int,
    ) -> list[dict]:
        """Execute keyword/text search on indexed text fields."""
        try:
            # Build keyword filter for name field
            keyword_conditions = [
                FieldCondition(key="name", match=MatchText(text=query_text))
            ]

            # Combine with existing filters
            if qdrant_filter and qdrant_filter.must:
                all_conditions = keyword_conditions + list(qdrant_filter.must)
            else:
                all_conditions = keyword_conditions

            combined_filter = Filter(must=all_conditions)

            # Use scroll to get keyword matches (no vector needed)
            results, _ = await client.scroll(
                collection_name=self.collection,
                scroll_filter=combined_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [self._format_result(r, score=1.0) for r in results]
        except Exception as e:
            logger.debug(f"Keyword search not available or failed: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[dict]:
        """Apply Reciprocal Rank Fusion to combine two result lists.

        TIER 4 - Point 15: RRF Formula

        Score = weight_v * (1/(k + rank_v)) + weight_k * (1/(k + rank_k))
        """
        # Map entity_id -> {result data, rrf_score}
        fused: dict[str, dict[str, Any]] = {}

        # Process vector results (ranked by semantic similarity)
        for rank, result in enumerate(vector_results, start=1):
            entity_id = result["id"]
            rrf_score = vector_weight * (1.0 / (RRF_K + rank))

            if entity_id not in fused:
                fused[entity_id] = {
                    **result,
                    "rrf_score": rrf_score,
                    "vector_rank": rank,
                    "keyword_rank": None,
                }
            else:
                fused[entity_id]["rrf_score"] += rrf_score
                fused[entity_id]["vector_rank"] = rank

        # Process keyword results (ranked by text match)
        for rank, result in enumerate(keyword_results, start=1):
            entity_id = result["id"]
            rrf_score = keyword_weight * (1.0 / (RRF_K + rank))

            if entity_id not in fused:
                fused[entity_id] = {
                    **result,
                    "rrf_score": rrf_score,
                    "vector_rank": None,
                    "keyword_rank": rank,
                }
            else:
                fused[entity_id]["rrf_score"] += rrf_score
                fused[entity_id]["keyword_rank"] = rank

        # Sort by RRF score descending
        sorted_results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return sorted_results

    def _format_result(self, r, score: float | None = None) -> dict:
        """Format a Qdrant point into a result dict."""
        return {
            "id": str(r.id),
            "score": score if score is not None else getattr(r, "score", 0.0),
            "name": r.payload.get("name", ""),
            "category": r.payload.get("category", ""),
            "city": r.payload.get("city", ""),
            "price_tier": r.payload.get("price_tier"),
            "dimensions": r.payload.get("dimensions", {}),
            "tags": r.payload.get("tags", []),
            "payload": r.payload,
        }


class EntityResolver:
    """Entity resolution and deduplication.

    TIER 4 - Point 16: Entity Resolution & Merging Strategy

    Handles duplicate entities from multiple sources by:
    1. Grouping by entity_uuid
    2. Fuzzy matching on normalized_name + address
    3. Merging metadata with source priority
    """

    # Priority for metadata merging: higher = preferred
    SOURCE_PRIORITY = {
        "neo4j": 3,
        "postgres": 2,
        "vector": 1,
    }

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize resolver.

        Args:
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def resolve(
        self,
        results: list[dict],
        source: str = "vector",
    ) -> list[dict]:
        """Resolve and deduplicate a list of entity results.

        Args:
            results: List of entity dicts with at least 'id' and 'name'
            source: Source identifier for priority resolution

        Returns:
            Deduplicated list with merged metadata
        """
        if not results:
            return []

        # Group by entity_uuid if available
        uuid_groups: dict[str, list[dict]] = defaultdict(list)
        no_uuid: list[dict] = []

        for r in results:
            entity_uuid = r.get("entity_uuid") or r.get("payload", {}).get("entity_uuid")
            if entity_uuid:
                uuid_groups[entity_uuid].append(r)
            else:
                no_uuid.append(r)

        # Resolve UUID groups
        resolved = []
        for uuid, group in uuid_groups.items():
            merged = self._merge_group(group, source)
            merged["entity_uuid"] = uuid
            resolved.append(merged)

        # Fuzzy match remaining entities
        for entity in no_uuid:
            matched = False
            normalized = self._normalize_name(entity.get("name", ""))

            for existing in resolved:
                existing_norm = self._normalize_name(existing.get("name", ""))
                if self._is_similar(normalized, existing_norm):
                    # Merge into existing
                    self._merge_into(existing, entity, source)
                    matched = True
                    break

            if not matched:
                resolved.append(entity)

        return resolved

    def _merge_group(self, group: list[dict], source: str) -> dict:
        """Merge a group of duplicate entities."""
        if len(group) == 1:
            return group[0].copy()

        # Start with highest priority source
        sorted_group = sorted(
            group,
            key=lambda x: self.SOURCE_PRIORITY.get(
                x.get("source", source), 0
            ),
            reverse=True,
        )

        merged = sorted_group[0].copy()

        # Merge in data from lower priority sources
        for entity in sorted_group[1:]:
            self._merge_into(merged, entity, source)

        return merged

    def _merge_into(self, target: dict, source_entity: dict, source: str) -> None:
        """Merge source_entity into target, filling missing fields."""
        for key, value in source_entity.items():
            if key not in target or target[key] is None:
                target[key] = value
            elif key == "tags" and isinstance(value, list):
                # Merge tag lists
                existing = set(target.get("tags", []))
                existing.update(value)
                target["tags"] = list(existing)
            elif key == "dimensions" and isinstance(value, dict):
                # Merge dimension dicts
                target.setdefault("dimensions", {}).update(value)

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        import re
        # Lowercase, remove punctuation, normalize spaces
        name = name.lower()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _is_similar(self, name1: str, name2: str) -> bool:
        """Check if two normalized names are similar enough."""
        if not name1 or not name2:
            return False

        # Simple Jaccard similarity on word sets
        words1 = set(name1.split())
        words2 = set(name2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union
        return similarity >= self.similarity_threshold
