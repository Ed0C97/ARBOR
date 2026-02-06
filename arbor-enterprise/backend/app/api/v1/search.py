"""Search endpoints for vector and hybrid search."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.agents.vector_agent import VectorAgent
from app.core.security import get_optional_user

router = APIRouter()


class VectorSearchRequest(BaseModel):
    query: str
    category: str | None = None
    city: str | None = None
    limit: int = 10


class SearchResultItem(BaseModel):
    id: str
    name: str
    score: float
    category: str = ""
    city: str = ""
    tags: list[str] = []
    dimensions: dict = {}


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


@router.get("/search/vector", response_model=SearchResponse)
async def vector_search(
    query: str = Query(..., min_length=2),
    category: str | None = None,
    city: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    user: dict | None = Depends(get_optional_user),
):
    """Search entities by semantic similarity."""
    agent = VectorAgent()
    filters = {}
    if category:
        filters["category"] = category
    if city:
        filters["city"] = city

    results = await agent.execute(query=query, filters=filters, limit=limit)

    items = [
        SearchResultItem(
            id=r.get("id", ""),
            name=r.get("name", ""),
            score=r.get("score", 0),
            category=r.get("category", ""),
            city=r.get("city", ""),
            tags=r.get("tags", []),
            dimensions=r.get("dimensions", {}),
        )
        for r in results
    ]

    return SearchResponse(results=items, total=len(items))


@router.get("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(
    query: str = Query(..., min_length=2),
    category: str | None = None,
    city: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    user: dict | None = Depends(get_optional_user),
):
    """Search entities using RRF hybrid search (dense vector + keyword).

    Combines semantic embedding search with keyword matching via Reciprocal
    Rank Fusion for higher-quality results than either method alone.
    """
    from app.llm.gateway import get_llm_gateway
    from app.db.qdrant.hybrid_search import HybridSearch

    gateway = get_llm_gateway()
    hybrid = HybridSearch()

    # Generate query embedding
    query_embedding = await gateway.get_embedding(query)

    # Build Qdrant filter conditions
    qdrant_filter = None
    if category or city:
        conditions = []
        if category:
            conditions.append({"key": "category", "match": {"value": category}})
        if city:
            conditions.append({"key": "city", "match": {"value": city}})
        qdrant_filter = {"must": conditions}

    # Execute RRF hybrid search
    results = await hybrid.search_rrf(
        query_vector=query_embedding,
        query_text=query,
        limit=limit,
        qdrant_filter=qdrant_filter,
    )

    items = [
        SearchResultItem(
            id=r.get("id", ""),
            name=r.get("name", ""),
            score=r.get("score", 0),
            category=r.get("category", ""),
            city=r.get("city", ""),
            tags=r.get("tags", []),
            dimensions=r.get("dimensions", {}),
        )
        for r in results
    ]

    return SearchResponse(results=items, total=len(items))
