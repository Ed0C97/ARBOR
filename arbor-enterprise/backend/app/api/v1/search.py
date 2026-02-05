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
