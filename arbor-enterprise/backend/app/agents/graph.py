"""LangGraph orchestration - the agent swarm graph."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.graph import END, StateGraph

from app.agents.curator import CuratorAgent
from app.agents.historian_agent import HistorianAgent
from app.agents.metadata_agent import MetadataAgent
from app.agents.router import IntentRouter
from app.agents.state import AgentState
from app.agents.vector_agent import VectorAgent


def create_agent_graph(session: AsyncSession | None = None, arbor_session: AsyncSession | None = None):
    """Create and compile the LangGraph agent swarm.

    Flow: Entry -> IntentRouter -> [VectorAgent, MetadataAgent] -> (HistorianAgent?) -> Curator -> End
    """
    router = IntentRouter()
    vector_agent = VectorAgent()
    metadata_agent = MetadataAgent(session, arbor_session)
    historian = HistorianAgent()
    curator = CuratorAgent()

    # --- Node functions ---

    async def route_intent(state: AgentState) -> dict:
        """Classify user intent."""
        result = await router.classify(state["user_query"])
        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
            "entities_mentioned": result.get("entities_mentioned", []),
            "filters": result.get("filters", {}),
        }

    async def search_vectors(state: AgentState) -> dict:
        """Semantic vector search."""
        results = await vector_agent.execute(
            query=state["user_query"],
            filters=state.get("filters", {}),
            limit=10,
        )
        return {"vector_results": results, "sources_used": ["qdrant"]}

    async def search_metadata(state: AgentState) -> dict:
        """Structured metadata search."""
        results = await metadata_agent.execute(
            filters=state.get("filters", {}),
            limit=5,
        )
        return {"metadata_results": results, "sources_used": ["postgres"]}

    async def search_graph(state: AgentState) -> dict:
        """Knowledge graph search."""
        results = await historian.execute(
            intent=state["intent"],
            entities_mentioned=state.get("entities_mentioned", []),
            filters=state.get("filters"),
        )
        return {"graph_results": results, "sources_used": ["neo4j"]}

    async def synthesize_response(state: AgentState) -> dict:
        """Generate final curated response."""
        result = await curator.synthesize(
            query=state["user_query"],
            vector_results=state.get("vector_results", []),
            graph_results=state.get("graph_results", []),
            metadata_results=state.get("metadata_results", []),
            intent=state["intent"],
        )
        return {
            "final_response": result["response_text"],
            "recommendations": result["recommendations"],
            "confidence_score": result["confidence"],
        }

    # --- Conditional edges ---

    def should_search_graph(state: AgentState) -> str:
        """Decide whether to search the knowledge graph."""
        if state.get("intent") in ("HISTORY", "COMPARISON"):
            return "search_graph"
        if state.get("entities_mentioned"):
            return "search_graph"
        return "synthesize"

    # --- Build graph ---

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("search_vectors", search_vectors)
    workflow.add_node("search_metadata", search_metadata)
    workflow.add_node("search_graph", search_graph)
    workflow.add_node("synthesize", synthesize_response)

    # Set entry
    workflow.set_entry_point("route_intent")

    # Edges: after routing, search vectors and metadata
    workflow.add_edge("route_intent", "search_vectors")
    workflow.add_edge("route_intent", "search_metadata")

    # After vector search, conditionally search graph
    workflow.add_conditional_edges(
        "search_vectors",
        should_search_graph,
        {
            "search_graph": "search_graph",
            "synthesize": "synthesize",
        },
    )

    # Metadata always goes to synthesize
    workflow.add_edge("search_metadata", "synthesize")

    # Graph search goes to synthesize
    workflow.add_edge("search_graph", "synthesize")

    # Synthesize ends
    workflow.add_edge("synthesize", END)

    return workflow.compile()
