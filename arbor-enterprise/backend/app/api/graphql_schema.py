"""GraphQL Federation Layer for A.R.B.O.R. Enterprise.

Provides a Strawberry-based GraphQL API that mirrors the REST endpoints,
enabling clients to query entities, discover recommendations, traverse the
knowledge graph, and record feedback through a single flexible endpoint.

Uses Strawberry (modern Python GraphQL library) with FastAPI integration.
Resolvers query real PostgreSQL repositories, Qdrant vector search, Neo4j
knowledge graph, and the LLM agent pipeline.
"""

import asyncio
import logging
from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.types import Info

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.repository import (
    UnifiedEntityRepository,
    FeedbackRepository,
    EnrichmentRepository,
)
from app.llm.gateway import get_llm_gateway
from app.db.qdrant.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphQL Types
# ---------------------------------------------------------------------------


@strawberry.type
class EntityType:
    """Unified representation of a brand or venue entity."""

    id: str
    name: str
    slug: str
    category: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    style: Optional[str] = None
    description: Optional[str] = None
    is_featured: bool = False
    is_active: bool = True
    entity_type: str = "brand"  # "brand" | "venue"
    vibe_dna: Optional[JSON] = None
    tags: Optional[list[str]] = None


@strawberry.type
class DiscoveryResponse:
    """Response from a natural-language discovery query."""

    query: str
    response_text: str
    recommendations: list[EntityType]
    confidence_score: float
    sources_used: list[str]


@strawberry.type
class SearchResult:
    """A single search result with relevance scoring."""

    entity: EntityType
    score: float
    match_reason: str


@strawberry.type
class GraphNode:
    """A node in the ARBOR knowledge graph."""

    id: str
    name: str
    labels: list[str]
    properties: Optional[JSON] = None


@strawberry.type
class GraphEdge:
    """A directed edge (relationship) in the knowledge graph."""

    source: str
    target: str
    rel_type: str
    properties: Optional[JSON] = None


@strawberry.type
class KnowledgeGraph:
    """A sub-graph consisting of nodes and edges."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


@strawberry.type
class PaginatedEntities:
    """Cursor-paginated entity list."""

    items: list[EntityType]
    cursor: Optional[str] = None
    has_more: bool = False
    total: int = 0


# ---------------------------------------------------------------------------
# Helpers: UnifiedEntity → EntityType conversion
# ---------------------------------------------------------------------------


def _unified_to_graphql(entity) -> EntityType:
    """Convert a UnifiedEntity dataclass to a Strawberry EntityType."""
    return EntityType(
        id=entity.id,
        name=entity.name,
        slug=entity.slug,
        category=entity.category,
        city=entity.city,
        country=entity.country,
        style=entity.style,
        description=entity.description,
        is_featured=entity.is_featured,
        is_active=entity.is_active,
        entity_type=entity.entity_type,
        vibe_dna=entity.vibe_dna,
        tags=entity.tags,
    )


def _search_result_to_graphql(result: dict) -> EntityType:
    """Convert a Qdrant search result dict to a Strawberry EntityType."""
    return EntityType(
        id=str(result.get("id", "")),
        name=result.get("name", "Unknown"),
        slug=result.get("name", "").lower().replace(" ", "-"),
        category=result.get("category"),
        city=result.get("city"),
        country=None,
        style=None,
        description=None,
        is_featured=False,
        is_active=True,
        entity_type="entity",
        vibe_dna=result.get("dimensions"),
        tags=result.get("tags"),
    )


async def _get_sessions():
    """Create async sessions for both databases."""
    mag = magazine_session_factory() if magazine_session_factory else None
    arb = arbor_session_factory() if arbor_session_factory else None
    return mag, arb


# ---------------------------------------------------------------------------
# Query Resolvers
# ---------------------------------------------------------------------------


@strawberry.type
class Query:
    """Root query type for A.R.B.O.R. Enterprise GraphQL API."""

    @strawberry.field(description="List entities with optional filters and pagination.")
    async def entities(
        self,
        limit: int = 20,
        offset: int = 0,
        category: Optional[str] = None,
        city: Optional[str] = None,
        style: Optional[str] = None,
    ) -> list[EntityType]:
        logger.info(
            "GraphQL entities query: limit=%d offset=%d category=%s city=%s style=%s",
            limit, offset, category, city, style,
        )
        mag_session, arb_session = await _get_sessions()
        if mag_session is None:
            return []

        try:
            async with mag_session as session:
                async with arb_session as a_session if arb_session else session as a_session:
                    repo = UnifiedEntityRepository(session, a_session)
                    entities, _total = await repo.list_all(
                        category=category,
                        city=city,
                        style=style,
                        offset=offset,
                        limit=limit,
                    )
                    return [_unified_to_graphql(e) for e in entities]
        except Exception as exc:
            logger.exception("GraphQL entities resolver error: %s", exc)
            return []

    @strawberry.field(description="Get a single entity by ID and optional type.")
    async def entity(
        self,
        id: str,
        entity_type: Optional[str] = None,
    ) -> Optional[EntityType]:
        logger.info("GraphQL entity query: id=%s entity_type=%s", id, entity_type)
        mag_session, arb_session = await _get_sessions()
        if mag_session is None:
            return None

        try:
            async with mag_session as session:
                async with arb_session as a_session if arb_session else session as a_session:
                    repo = UnifiedEntityRepository(session, a_session)
                    entity = await repo.get_by_composite_id(id)
                    return _unified_to_graphql(entity) if entity else None
        except Exception as exc:
            logger.exception("GraphQL entity resolver error: %s", exc)
            return None

    @strawberry.field(description="Natural-language discovery query.")
    async def discover(
        self,
        query: str,
        limit: int = 5,
    ) -> DiscoveryResponse:
        logger.info("GraphQL discover query: query=%r limit=%d", query, limit)
        sources_used: list[str] = []

        try:
            gateway = get_llm_gateway()
            hybrid = HybridSearch()

            # Get embedding and run vector search
            query_embedding = await gateway.get_embedding(query)
            vector_results = await hybrid.search_rrf(
                query_vector=query_embedding,
                query_text=query,
                limit=limit,
            )
            sources_used.append("vector_search")

            recommendations = [_search_result_to_graphql(r) for r in vector_results[:limit]]

            # Use LLM to generate a natural language response
            if recommendations:
                top_names = [r.name for r in recommendations[:3]]
                response_text = await gateway.complete(
                    messages=[
                        {"role": "system", "content": "You are ARBOR, a curated discovery assistant for fashion brands and venues. Be concise."},
                        {"role": "user", "content": f"Based on search results for '{query}', recommend: {', '.join(top_names)}. Write a 2-sentence recommendation."},
                    ],
                    task_type="simple",
                    temperature=0.7,
                )
                sources_used.append("llm_synthesis")
            else:
                response_text = f"No results found for '{query}'. Try broadening your search."

            confidence = max((r.get("score", 0) for r in vector_results), default=0.0) if vector_results else 0.0

            return DiscoveryResponse(
                query=query,
                response_text=response_text,
                recommendations=recommendations,
                confidence_score=round(confidence, 4),
                sources_used=sources_used,
            )
        except Exception as exc:
            logger.exception("GraphQL discover resolver error: %s", exc)
            return DiscoveryResponse(
                query=query,
                response_text=f"Error processing discovery query: {exc}",
                recommendations=[],
                confidence_score=0.0,
                sources_used=sources_used,
            )

    @strawberry.field(description="Search entities with scoring and match reasons.")
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        city: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        logger.info(
            "GraphQL search query: query=%r category=%s city=%s limit=%d",
            query, category, city, limit,
        )
        try:
            gateway = get_llm_gateway()
            hybrid = HybridSearch()

            query_embedding = await gateway.get_embedding(query)
            results = await hybrid.search_rrf(
                query_vector=query_embedding,
                query_text=query,
                limit=limit,
                category=category,
                city=city,
            )

            search_results = []
            for r in results:
                entity = _search_result_to_graphql(r)
                score = r.get("rrf_score", r.get("score", 0.0))
                # Build a match reason from available data
                reasons = []
                if r.get("vector_rank"):
                    reasons.append(f"semantic rank #{r['vector_rank']}")
                if r.get("keyword_rank"):
                    reasons.append(f"keyword rank #{r['keyword_rank']}")
                if r.get("tags"):
                    reasons.append(f"tags: {', '.join(r['tags'][:3])}")
                match_reason = "; ".join(reasons) if reasons else "Matched by RRF hybrid search"

                search_results.append(SearchResult(
                    entity=entity,
                    score=round(score, 4),
                    match_reason=match_reason,
                ))
            return search_results
        except Exception as exc:
            logger.exception("GraphQL search resolver error: %s", exc)
            return []

    @strawberry.field(description="Retrieve a sub-graph around an entity from the knowledge graph.")
    async def knowledge_graph(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> KnowledgeGraph:
        logger.info("GraphQL knowledge_graph query: entity_id=%s depth=%d", entity_id, depth)
        try:
            from app.db.neo4j.graphrag import GraphRAG

            graphrag = GraphRAG()
            raw = await graphrag.smart_query(
                question=f"Find entities related to {entity_id}",
                query_type="neighborhood",
            )

            nodes: list[GraphNode] = []
            edges: list[GraphEdge] = []
            seen_ids: set[str] = set()

            for record in (raw if isinstance(raw, list) else [raw]):
                if isinstance(record, dict):
                    node_id = str(record.get("id", record.get("name", "")))
                    if node_id and node_id not in seen_ids:
                        seen_ids.add(node_id)
                        nodes.append(GraphNode(
                            id=node_id,
                            name=record.get("name", node_id),
                            labels=record.get("labels", ["Entity"]),
                            properties={k: v for k, v in record.items()
                                        if k not in ("id", "name", "labels")},
                        ))

            return KnowledgeGraph(nodes=nodes, edges=edges)
        except Exception as exc:
            logger.warning("Knowledge graph query failed, returning empty: %s", exc)
            return KnowledgeGraph(nodes=[], edges=[])

    @strawberry.field(description="Find entities related to the given entity by name.")
    async def related_entities(
        self,
        entity_name: str,
    ) -> list[EntityType]:
        logger.info("GraphQL related_entities query: entity_name=%r", entity_name)
        try:
            gateway = get_llm_gateway()
            hybrid = HybridSearch()

            # Search for entities similar to the given name
            query_embedding = await gateway.get_embedding(entity_name)
            results = hybrid.search(
                query_vector=query_embedding,
                query_text=entity_name,
                limit=5,
            )

            # Filter out exact match (the entity itself)
            related = [
                _search_result_to_graphql(r)
                for r in results
                if r.get("name", "").lower() != entity_name.lower()
            ]
            return related
        except Exception as exc:
            logger.exception("GraphQL related_entities error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Mutation Resolvers
# ---------------------------------------------------------------------------


@strawberry.type
class Mutation:
    """Root mutation type for A.R.B.O.R. Enterprise GraphQL API."""

    @strawberry.mutation(description="Record user feedback for an entity (click, save, dismiss, etc.).")
    async def record_feedback(
        self,
        user_id: str,
        entity_id: str,
        entity_type: str,
        action: str,
        position: Optional[int] = None,
        query: Optional[str] = None,
    ) -> bool:
        logger.info(
            "GraphQL record_feedback: user=%s entity=%s type=%s action=%s pos=%s query=%r",
            user_id, entity_id, entity_type, action, position, query,
        )
        allowed_actions = {"click", "save", "dismiss", "share", "view", "purchase"}
        if action not in allowed_actions:
            logger.warning("Unknown feedback action: %s", action)
            return False

        if arbor_session_factory is None:
            logger.warning("ARBOR database not configured, cannot record feedback")
            return False

        try:
            async with arbor_session_factory() as session:
                repo = FeedbackRepository(session)
                await repo.create(
                    user_id=user_id,
                    entity_type=entity_type,
                    source_id=int(entity_id.split("_")[-1]) if "_" in entity_id else 0,
                    action=action,
                    position=position,
                    query=query,
                )
                await session.commit()
            return True
        except Exception as exc:
            logger.exception("Failed to record feedback: %s", exc)
            return False

    @strawberry.mutation(description="Trigger an enrichment pipeline run for an entity.")
    async def trigger_enrichment(
        self,
        entity_type: str,
        source_id: str,
    ) -> bool:
        logger.info(
            "GraphQL trigger_enrichment: entity_type=%s source_id=%s",
            entity_type, source_id,
        )
        if entity_type not in ("brand", "venue"):
            logger.warning("Invalid entity_type for enrichment: %s", entity_type)
            return False

        if arbor_session_factory is None:
            logger.warning("ARBOR database not configured, cannot trigger enrichment")
            return False

        try:
            async with arbor_session_factory() as session:
                repo = EnrichmentRepository(session)
                await repo.upsert(
                    entity_type=entity_type,
                    source_id=int(source_id),
                    status="pending",
                )
                await session.commit()
            return True
        except Exception as exc:
            logger.exception("Failed to trigger enrichment: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Subscription (Redis Pub/Sub based)
# ---------------------------------------------------------------------------


@strawberry.type
class Subscription:
    """Root subscription type for A.R.B.O.R. Enterprise GraphQL API.

    Strawberry supports async generators for subscriptions.
    Listens to Redis Pub/Sub for entity update events, falling back to
    periodic polling when Redis is unavailable.
    """

    @strawberry.subscription(description="Subscribe to entity update events.")
    async def entity_updated(
        self,
        entity_type: Optional[str] = None,
    ) -> EntityType:  # type: ignore[override]
        """Yield entity updates as they occur via Redis Pub/Sub."""
        logger.info("GraphQL subscription entity_updated: entity_type=%s", entity_type)

        try:
            from app.db.redis.client import RedisCache
            redis_cache = RedisCache()
            pubsub = redis_cache.client.pubsub()
            channel = f"entity_updates:{entity_type}" if entity_type else "entity_updates:*"
            await pubsub.psubscribe(channel)

            async for message in pubsub.listen():
                if message["type"] in ("message", "pmessage"):
                    import json
                    data = json.loads(message["data"])
                    entity_id = data.get("entity_id", "")
                    # Fetch the updated entity from the database
                    mag_session, arb_session = await _get_sessions()
                    if mag_session:
                        async with mag_session as session:
                            async with arb_session as a_session if arb_session else session as a_session:
                                repo = UnifiedEntityRepository(session, a_session)
                                entity = await repo.get_by_composite_id(entity_id)
                                if entity:
                                    yield _unified_to_graphql(entity)
        except Exception as exc:
            logger.warning("Redis Pub/Sub unavailable, falling back to polling: %s", exc)
            # Fallback: poll for recent updates
            while True:
                await asyncio.sleep(10)
                mag_session, arb_session = await _get_sessions()
                if mag_session:
                    try:
                        async with mag_session as session:
                            async with arb_session as a_session if arb_session else session as a_session:
                                repo = UnifiedEntityRepository(session, a_session)
                                entities, _ = await repo.list_all(
                                    entity_type=entity_type,
                                    limit=1,
                                )
                                for e in entities:
                                    yield _unified_to_graphql(e)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Schema & App Factory
# ---------------------------------------------------------------------------


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)


def create_graphql_app() -> GraphQLRouter:
    """Create a Strawberry GraphQL router for mounting in the FastAPI app.

    Usage in main.py::

        from app.api.graphql_schema import create_graphql_app

        graphql_app = create_graphql_app()
        app.include_router(graphql_app, prefix="/graphql")

    This exposes:
      - ``/graphql``  -- GraphQL endpoint (POST for queries/mutations)
      - ``/graphql``  -- GraphiQL IDE in the browser (GET, dev mode)
    """
    return GraphQLRouter(schema, path="")
