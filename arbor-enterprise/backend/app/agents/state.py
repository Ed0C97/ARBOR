"""Agent state definition for LangGraph orchestration."""

from typing import Annotated, TypedDict

from operator import add


class AgentState(TypedDict):
    """Shared state between all agents in the swarm."""

    # User input
    user_query: str
    user_location: str | None
    user_preferences: dict

    # Intent classification
    intent: str  # DISCOVERY, COMPARISON, DETAIL, HISTORY, NAVIGATION, GENERAL
    intent_confidence: float
    entities_mentioned: list[str]
    filters: dict

    # Intermediate results (additive - each agent appends)
    vector_results: Annotated[list, add]
    metadata_results: Annotated[list, add]
    graph_results: Annotated[list, add]

    # Final output
    final_response: str
    recommendations: list[dict]
    confidence_score: float
    sources_used: Annotated[list[str], add]
