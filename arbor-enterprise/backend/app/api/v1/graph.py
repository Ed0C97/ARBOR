"""Knowledge graph query endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.core.security import get_optional_user
from app.db.neo4j.queries import Neo4jQueries

router = APIRouter()


class GraphRelation(BaseModel):
    type: str
    name: str = ""
    category: str = ""
    city: str = ""
    extra: dict = {}


class GraphResponse(BaseModel):
    results: list[GraphRelation]
    total: int


@router.get("/graph/related", response_model=GraphResponse)
async def get_related_entities(
    entity_name: str = Query(..., min_length=2),
    user: dict | None = Depends(get_optional_user),
):
    """Find entities related by style in the knowledge graph."""
    neo4j = Neo4jQueries()
    results = await neo4j.find_related_by_style(entity_name)

    items = [
        GraphRelation(
            type="style_related",
            name=r.get("name", ""),
            category=r.get("category", ""),
            city=r.get("city", ""),
            extra={"shared_style": r.get("shared_style", "")},
        )
        for r in results
    ]

    return GraphResponse(results=items, total=len(items))


@router.get("/graph/lineage", response_model=GraphResponse)
async def get_lineage(
    entity_name: str = Query(..., min_length=2),
    depth: int = Query(3, ge=1, le=5),
    user: dict | None = Depends(get_optional_user),
):
    """Find the training/mentorship lineage of an entity."""
    neo4j = Neo4jQueries()
    results = await neo4j.find_lineage(entity_name, depth)

    items = [
        GraphRelation(
            type="lineage",
            name=r.get("mentor", ""),
            extra={"distance": r.get("distance", 0), "entity": r.get("entity", "")},
        )
        for r in results
    ]

    return GraphResponse(results=items, total=len(items))


@router.get("/graph/brand-retailers", response_model=GraphResponse)
async def get_brand_retailers(
    brand_name: str = Query(..., min_length=2),
    city: str | None = None,
    user: dict | None = Depends(get_optional_user),
):
    """Find retailers that sell a specific brand."""
    neo4j = Neo4jQueries()
    results = await neo4j.find_brand_retailers(brand_name, city)

    items = [
        GraphRelation(
            type="brand_retailer",
            name=r.get("name", ""),
            city=r.get("city", ""),
            extra={"relationship": r.get("relationship_type", "")},
        )
        for r in results
    ]

    return GraphResponse(results=items, total=len(items))


@router.get("/graph/full")
async def get_full_graph(
    limit: int = Query(200, ge=1, le=1000),
    user: dict | None = Depends(get_optional_user),
):
    """Get the full knowledge graph (nodes + edges) for visualization."""
    neo4j = Neo4jQueries()
    return await neo4j.get_full_graph(limit=limit)
