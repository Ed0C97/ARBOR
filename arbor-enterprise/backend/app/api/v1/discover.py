"""Main discovery endpoint - the core of A.R.B.O.R.

TIER 7 - Point 33: Server-Sent Events (SSE) Streaming

Provides both traditional REST response and streaming response for
real-time "typewriter" effect in the frontend.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import create_agent_graph
from app.core.rate_limiter import check_rate_limit
from app.core.security import get_optional_user
from app.db.postgres.connection import get_arbor_db, get_db
from app.llm.cache import get_semantic_cache
from app.llm.guardrails import get_guardrails
from app.observability.metrics import record_discover_request

logger = logging.getLogger(__name__)
router = APIRouter()


class DiscoverRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    location: str | None = None
    category: str | None = None
    price_max: int | None = Field(None, ge=1, le=5)
    limit: int = Field(5, ge=1, le=20)


class RecommendationItem(BaseModel):
    id: str
    name: str
    score: float
    category: str = ""
    city: str = ""
    tags: list[str] = []
    dimensions: dict = {}


class DiscoverResponse(BaseModel):
    recommendations: list[RecommendationItem]
    response_text: str
    confidence: float
    query_intent: str
    latency_ms: int
    cached: bool = False


@router.post("/discover", response_model=DiscoverResponse)
async def discover(
    request_body: DiscoverRequest,
    request: Request,
    user: dict | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Main discovery endpoint. Send a natural language query to get curated recommendations."""
    start_time = time.time()

    # Rate limiting
    await check_rate_limit(request, user)

    # Guardrails check
    guardrails = get_guardrails()
    allowed, reason = await guardrails.check_input(request_body.query)
    if not allowed:
        return DiscoverResponse(
            recommendations=[],
            response_text=f"I can't process that request: {reason}",
            confidence=0.0,
            query_intent="BLOCKED",
            latency_ms=int((time.time() - start_time) * 1000),
        )

    # Check semantic cache
    cache = get_semantic_cache()
    cached_response = await cache.get(request_body.query)
    if cached_response:
        latency = int((time.time() - start_time) * 1000)
        return DiscoverResponse(
            recommendations=[],
            response_text=cached_response,
            confidence=0.9,
            query_intent="CACHED",
            latency_ms=latency,
            cached=True,
        )

    # Build agent state
    state = {
        "user_query": request_body.query,
        "user_location": request_body.location,
        "user_preferences": {},
        "intent": "",
        "intent_confidence": 0.0,
        "entities_mentioned": [],
        "filters": {},
        "vector_results": [],
        "metadata_results": [],
        "graph_results": [],
        "final_response": "",
        "recommendations": [],
        "confidence_score": 0.0,
        "sources_used": [],
    }

    # Add explicit filters
    if request_body.category:
        state["filters"]["category"] = request_body.category
    if request_body.price_max:
        state["filters"]["price_max"] = request_body.price_max

    # Execute agent graph
    agent_graph = create_agent_graph(session, arbor_session)
    result = await agent_graph.ainvoke(state)

    # Cache the response
    if result.get("final_response"):
        await cache.set(request_body.query, result["final_response"])

    latency = int((time.time() - start_time) * 1000)

    recommendations = [
        RecommendationItem(
            id=r.get("id", ""),
            name=r.get("name", ""),
            score=r.get("score", 0),
            category=r.get("category", ""),
            city=r.get("city", ""),
            tags=r.get("tags", []),
            dimensions=r.get("dimensions", {}),
        )
        for r in result.get("recommendations", [])[: request_body.limit]
    ]

    # Record metrics
    record_discover_request(
        status="success",
        intent=result.get("intent", "unknown"),
        cache_hit=False,
    )

    return DiscoverResponse(
        recommendations=recommendations,
        response_text=result.get("final_response", ""),
        confidence=result.get("confidence_score", 0.0),
        query_intent=result.get("intent", "UNKNOWN"),
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# TIER 7 - Point 33: Server-Sent Events (SSE) Streaming
# ---------------------------------------------------------------------------


class StreamingDiscoverRequest(BaseModel):
    """Request model for streaming discovery."""

    query: str = Field(..., min_length=2, max_length=1000)
    location: str | None = None
    category: str | None = None
    price_max: int | None = Field(None, ge=1, le=5)
    limit: int = Field(5, ge=1, le=20)


async def stream_discovery_response(
    query: str,
    location: str | None,
    category: str | None,
    price_max: int | None,
    limit: int,
    session: AsyncSession,
    arbor_session: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for discovery response.

    TIER 7 - Point 33: SSE Streaming

    Yields events in the format:
    - event: status (processing stages)
    - event: recommendation (each result as it's found)
    - event: text (streaming text chunks)
    - event: done (final event with full response)

    Frontend can parse these with EventSource API.
    """
    start_time = time.time()

    # Send initial status
    yield f"event: status\ndata: {json.dumps({'stage': 'started', 'message': 'Processing your query...'})}\n\n"

    # Guardrails check
    guardrails = get_guardrails()
    result = await guardrails.check_input(query)

    if not result.allowed:
        yield f"event: error\ndata: {json.dumps({'message': result.reason})}\n\n"
        yield f"event: done\ndata: {json.dumps({'success': False})}\n\n"
        return

    yield f"event: status\ndata: {json.dumps({'stage': 'validated', 'message': 'Query validated'})}\n\n"

    # Check semantic cache
    cache = get_semantic_cache()
    cache_result = await cache.check_cache(query)

    if cache_result.hit:
        yield f"event: status\ndata: {json.dumps({'stage': 'cached', 'message': 'Found cached response'})}\n\n"

        # Stream cached response character by character (typewriter effect)
        response_text = cache_result.response or ""
        for i in range(0, len(response_text), 10):  # 10 chars at a time
            chunk = response_text[i : i + 10]
            yield f"event: text\ndata: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.02)  # Small delay for effect

        latency = int((time.time() - start_time) * 1000)
        yield f"event: done\ndata: {json.dumps({'success': True, 'cached': True, 'latency_ms': latency})}\n\n"

        record_discover_request(status="success", intent="cached", cache_hit=True)
        return

    yield f"event: status\ndata: {json.dumps({'stage': 'searching', 'message': 'Searching for recommendations...'})}\n\n"

    # Build agent state
    state = {
        "user_query": query,
        "user_location": location,
        "user_preferences": {},
        "intent": "",
        "intent_confidence": 0.0,
        "entities_mentioned": [],
        "filters": {},
        "vector_results": [],
        "metadata_results": [],
        "graph_results": [],
        "final_response": "",
        "recommendations": [],
        "confidence_score": 0.0,
        "sources_used": [],
    }

    if category:
        state["filters"]["category"] = category
    if price_max:
        state["filters"]["price_max"] = price_max

    # Execute agent graph
    try:
        agent_graph = create_agent_graph(session, arbor_session)
        result = await agent_graph.ainvoke(state)
    except Exception as e:
        logger.error(f"Agent graph error: {e}")
        yield f"event: error\ndata: {json.dumps({'message': 'An error occurred while processing your request'})}\n\n"
        yield f"event: done\ndata: {json.dumps({'success': False})}\n\n"
        record_discover_request(status="error", intent="unknown", cache_hit=False)
        return

    rec_count = len(result.get("recommendations", []))
    yield f"event: status\ndata: {json.dumps({'stage': 'found', 'message': f'Found {rec_count} recommendations'})}\n\n"

    # Stream recommendations one by one
    recommendations = result.get("recommendations", [])[:limit]
    for i, rec in enumerate(recommendations):
        rec_data = {
            "index": i,
            "id": rec.get("id", ""),
            "name": rec.get("name", ""),
            "score": rec.get("score", 0),
            "category": rec.get("category", ""),
            "city": rec.get("city", ""),
            "tags": rec.get("tags", []),
        }
        yield f"event: recommendation\ndata: {json.dumps(rec_data)}\n\n"
        await asyncio.sleep(0.1)  # Small delay between recommendations

    yield f"event: status\ndata: {json.dumps({'stage': 'synthesizing', 'message': 'Generating response...'})}\n\n"

    # Stream the response text (typewriter effect)
    response_text = result.get("final_response", "")
    if response_text:
        # Stream in small chunks for typewriter effect
        chunk_size = 5  # Characters per chunk
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i : i + chunk_size]
            yield f"event: text\ndata: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.015)  # ~66 chars/second typing speed

        # Cache the response
        await cache.set(query, response_text)

    latency = int((time.time() - start_time) * 1000)

    # Send final event with complete data
    final_data = {
        "success": True,
        "cached": False,
        "latency_ms": latency,
        "intent": result.get("intent", "UNKNOWN"),
        "confidence": result.get("confidence_score", 0.0),
        "total_recommendations": len(recommendations),
    }
    yield f"event: done\ndata: {json.dumps(final_data)}\n\n"

    record_discover_request(
        status="success",
        intent=result.get("intent", "unknown"),
        cache_hit=False,
    )


@router.post("/discover/stream")
async def discover_stream(
    request_body: StreamingDiscoverRequest,
    request: Request,
    user: dict | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_db),
    arbor_session: AsyncSession = Depends(get_arbor_db),
):
    """Streaming discovery endpoint using Server-Sent Events.

    TIER 7 - Point 33: Server-Sent Events (SSE) Streaming

    Returns a stream of events:
    - status: Processing stage updates
    - recommendation: Individual recommendations as they're found
    - text: Response text chunks (typewriter effect)
    - error: Error messages if something goes wrong
    - done: Final event with summary data

    Frontend usage:
    ```javascript
    const eventSource = new EventSource('/api/v1/discover/stream', {
        method: 'POST',
        body: JSON.stringify({ query: '...' })
    });

    eventSource.addEventListener('text', (e) => {
        const { chunk } = JSON.parse(e.data);
        appendToResponse(chunk);
    });
    ```
    """
    # Rate limiting
    await check_rate_limit(request, user)

    return StreamingResponse(
        stream_discovery_response(
            query=request_body.query,
            location=request_body.location,
            category=request_body.category,
            price_max=request_body.price_max,
            limit=request_body.limit,
            session=session,
            arbor_session=arbor_session,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
