"""Qdrant collection management and operations."""

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    SearchParams,
)

from app.db.qdrant.client import get_qdrant_client


class QdrantCollections:
    """Manage Qdrant vector operations."""

    def __init__(self):
        self.collection = "entities_vectors"

    def upsert_vector(
        self,
        entity_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        """Insert or update a vector point."""
        client = get_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client not configured")
        client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=entity_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[dict]:
        """Search for similar vectors with optional filters."""
        client = get_qdrant_client()
        if client is None:
            return []

        qdrant_filter = self._build_filter(filter_conditions) if filter_conditions else None

        results = client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        ).points

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def delete_vector(self, entity_id: str) -> None:
        """Delete a vector point by ID."""
        client = get_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client not configured")
        client.delete(
            collection_name=self.collection,
            points_selector=[entity_id],
        )

    def _build_filter(self, conditions: dict) -> Filter:
        """Build Qdrant filter from dict conditions."""
        must_conditions = []

        if "category" in conditions:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=conditions["category"]))
            )
        if "city" in conditions:
            must_conditions.append(
                FieldCondition(key="city", match=MatchValue(value=conditions["city"]))
            )
        if "price_max" in conditions:
            must_conditions.append(
                FieldCondition(key="price_tier", range=Range(lte=conditions["price_max"]))
            )
        if "price_min" in conditions:
            must_conditions.append(
                FieldCondition(key="price_tier", range=Range(gte=conditions["price_min"]))
            )
        if "status" in conditions:
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=conditions["status"]))
            )

        return Filter(must=must_conditions) if must_conditions else None
