"""Agentic Workflows - autonomous multi-step agent workflows for ARBOR Enterprise.

Extends the base agent swarm with complex, iterative workflow patterns:
- Deep Research: iterative search-analyze-refine loops
- Trend Analysis: entity data aggregation and trend extraction
- Entity Comparison: multi-dimensional comparison matrices
- Market Mapping: geographic and categorical market analysis
- Style Exploration: vibe-dimension guided discovery

Each workflow is orchestrated as a dedicated LangGraph StateGraph with its own
topology, allowing cycles (for iterative refinement) and conditional branching.
"""

import asyncio
import logging
import operator
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared gather helper — calls real VectorAgent + MetadataAgent
# ---------------------------------------------------------------------------

async def _gather_entities_for_query(
    sub_query: str,
    limit: int = 10,
    source_label: str = "gather",
) -> tuple[list[dict], list[dict]]:
    """Execute a sub-query against VectorAgent and MetadataAgent.

    Returns:
        (findings_list, entities_list) — findings contain query+results,
        entities are the raw result dicts for downstream analysis.
    """
    from app.agents.vector_agent import VectorAgent
    from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
    from app.agents.metadata_agent import MetadataAgent

    vector_results: list[dict] = []
    metadata_results: list[dict] = []

    # --- Vector search (semantic) ---
    try:
        vector_agent = VectorAgent()
        vector_results = await vector_agent.execute(
            query=sub_query, filters=None, limit=limit,
        )
    except Exception as exc:
        logger.warning("VectorAgent failed for '%s': %s", sub_query[:60], exc)

    # --- Metadata search (structured) ---
    try:
        if magazine_session_factory:
            async with magazine_session_factory() as session:
                arb_ctx = arbor_session_factory() if arbor_session_factory else None
                if arb_ctx:
                    async with arb_ctx as arb_session:
                        metadata_agent = MetadataAgent(session, arb_session)
                        metadata_results = await metadata_agent.execute(
                            filters={"search": sub_query}, limit=limit,
                        )
                else:
                    metadata_agent = MetadataAgent(session, None)
                    metadata_results = await metadata_agent.execute(
                        filters={"search": sub_query}, limit=limit,
                    )
    except Exception as exc:
        logger.warning("MetadataAgent failed for '%s': %s", sub_query[:60], exc)

    # Merge and deduplicate by id
    combined: dict[str, dict] = {}
    for r in vector_results + metadata_results:
        rid = r.get("id", r.get("name", ""))
        if rid not in combined:
            combined[rid] = r

    all_results = list(combined.values())

    finding = {
        "query": sub_query,
        "type": source_label,
        "results": all_results,
        "vector_count": len(vector_results),
        "metadata_count": len(metadata_results),
        "source": source_label,
    }

    return [finding], all_results

# ---------------------------------------------------------------------------
# LangGraph imports - graceful fallback when not installed
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "__end__"
    logger.warning(
        "langgraph not installed - agentic workflows will use fallback execution"
    )


# ===========================================================================
# Workflow Type Enum
# ===========================================================================


class WorkflowType(str, Enum):
    """Available autonomous workflow types."""

    DEEP_RESEARCH = "deep_research"
    TREND_ANALYSIS = "trend_analysis"
    ENTITY_COMPARISON = "entity_comparison"
    MARKET_MAPPING = "market_mapping"
    STYLE_EXPLORATION = "style_exploration"


# ===========================================================================
# Workflow State
# ===========================================================================


class WorkflowState(TypedDict):
    """Shared state for agentic workflows.

    Follows the same Annotated[list, operator.add] convention used in
    ``AgentState`` so that each node *appends* rather than overwrites
    list fields.
    """

    # Workflow identity
    workflow_type: str
    original_query: str

    # Decomposed sub-queries produced by the QueryDecomposer
    sub_queries: list[str]

    # Iteration control
    iteration: int
    max_iterations: int

    # Accumulative results (each node appends)
    intermediate_findings: Annotated[list, operator.add]
    entities_discovered: Annotated[list, operator.add]

    # Final output
    synthesis: str
    is_complete: bool


# ===========================================================================
# Query Decomposer
# ===========================================================================


class QueryDecomposer:
    """Decompose a complex user query into targeted sub-queries.

    Uses workflow-specific templates to break a broad question into
    narrower, searchable sub-queries that the downstream agents can
    handle individually.
    """

    # Template strategies keyed by WorkflowType
    _TEMPLATES: dict[str, list[str]] = {
        WorkflowType.DEEP_RESEARCH: [
            "{query}",
            "{query} key characteristics",
            "{query} notable examples",
        ],
        WorkflowType.TREND_ANALYSIS: [
            "{query} trending now",
            "{query} emerging styles",
            "{query} popular categories",
            "{query} geographic hotspots",
        ],
        WorkflowType.ENTITY_COMPARISON: [
            # Handled specially - needs entity extraction
        ],
        WorkflowType.MARKET_MAPPING: [
            "{query} market overview",
            "{query} key players",
            "{query} geographic distribution",
            "{query} price segments",
        ],
        WorkflowType.STYLE_EXPLORATION: [
            "{query} aesthetic",
            "{query} vibe",
            "{query} similar styles",
            "{query} brands and venues",
        ],
    }

    def decompose(self, query: str, workflow_type: WorkflowType) -> list[str]:
        """Break *query* into sub-queries appropriate for *workflow_type*.

        For ENTITY_COMPARISON the method attempts to split the query on
        comparison keywords (``vs``, ``versus``, ``compared to``, ``or``)
        and generates per-entity sub-queries plus a comparison sub-query.

        Returns:
            A list of sub-query strings (always at least one).
        """
        if workflow_type == WorkflowType.ENTITY_COMPARISON:
            return self._decompose_comparison(query)

        templates = self._TEMPLATES.get(workflow_type, ["{query}"])
        sub_queries = [t.format(query=query) for t in templates]

        logger.info(
            "QueryDecomposer: %s -> %d sub-queries for %s",
            query[:60],
            len(sub_queries),
            workflow_type,
        )
        return sub_queries

    # ------------------------------------------------------------------
    # Comparison-specific decomposition
    # ------------------------------------------------------------------

    _COMPARISON_SEPARATORS = (" vs ", " versus ", " compared to ", " or ")

    def _decompose_comparison(self, query: str) -> list[str]:
        """Split a comparison query into per-entity + diff sub-queries."""
        query_lower = query.lower()

        for sep in self._COMPARISON_SEPARATORS:
            if sep in query_lower:
                idx = query_lower.index(sep)
                part_a = query[:idx].strip()
                part_b = query[idx + len(sep):].strip()
                return [
                    part_a,
                    part_b,
                    f"Differences between {part_a} and {part_b}",
                ]

        # Fallback: treat the whole query as a single sub-query
        logger.debug(
            "QueryDecomposer: no comparison separator found in '%s'", query[:60]
        )
        return [query]


# ===========================================================================
# Deep Research Agent
# ===========================================================================


class DeepResearchAgent:
    """Iteratively search, analyze gaps, and refine queries.

    Each iteration:
    1. Execute current sub-queries against vector/metadata stores.
    2. Analyze findings to identify knowledge gaps.
    3. Generate follow-up sub-queries targeting the gaps.
    4. Repeat until ``max_iterations`` is reached or no gaps remain.
    """

    def __init__(self, default_max_iterations: int = 3):
        self._default_max_iterations = default_max_iterations

    async def research(self, state: dict) -> dict:
        """Run one iteration of the research loop.

        Returns a state-update dict that the graph will merge.
        """
        iteration = state.get("iteration", 0)
        sub_queries = state.get("sub_queries", [])
        prior_findings: list = state.get("intermediate_findings", [])

        logger.info(
            "DeepResearchAgent: iteration %d, %d sub-queries",
            iteration,
            len(sub_queries),
        )

        # -- Step 1: execute sub-queries via VectorAgent + MetadataAgent ---
        new_findings: list[dict] = []
        new_entities: list[dict] = []
        for sq in sub_queries:
            try:
                findings, entities = await _gather_entities_for_query(
                    sq, limit=10, source_label="deep_research",
                )
                for f in findings:
                    f["iteration"] = iteration
                new_findings.extend(findings)
                new_entities.extend(entities)
            except Exception as exc:
                logger.warning("Research sub-query failed '%s': %s", sq[:50], exc)
                new_findings.append({
                    "query": sq,
                    "iteration": iteration,
                    "results": [],
                    "source": "deep_research",
                })

        # -- Step 2: identify gaps ------------------------------------------
        all_findings = list(prior_findings) + new_findings
        gaps = self._identify_gaps(all_findings)

        # -- Step 3: generate follow-up sub-queries -------------------------
        follow_up_queries = gaps[:3]  # cap follow-ups per iteration

        return {
            "intermediate_findings": new_findings,
            "entities_discovered": new_entities,
            "sub_queries": follow_up_queries if follow_up_queries else sub_queries,
            "iteration": iteration + 1,
        }

    # ------------------------------------------------------------------

    def _identify_gaps(self, findings: list[dict]) -> list[str]:
        """Identify knowledge gaps in the accumulated findings.

        Uses a heuristic pass first (empty results, low diversity) and then
        calls the LLM for deeper gap analysis if available.

        Returns:
            A list of follow-up query strings targeting the gaps.
        """
        gaps: list[str] = []

        # Heuristic: empty result sets → rephrase query
        for finding in findings:
            results = finding.get("results", [])
            query = finding.get("query", "")
            if not results and query:
                gaps.append(f"{query} alternative sources")

        # Heuristic: low category diversity → broaden
        all_categories: set[str] = set()
        for finding in findings:
            for r in finding.get("results", []):
                cat = r.get("category", "")
                if cat:
                    all_categories.add(cat)
        if len(all_categories) <= 1 and findings:
            original = findings[0].get("query", "")
            if original:
                gaps.append(f"{original} different categories and styles")

        # LLM-powered gap analysis (async → run via event loop if available)
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            # Schedule LLM gap analysis as a fire-and-forget coroutine
            # that we await synchronously via a helper
            llm_gaps = loop.run_until_complete(self._llm_identify_gaps(findings))
            gaps.extend(llm_gaps)
        except RuntimeError:
            # Already inside an async context or no loop — try direct call
            pass

        return gaps

    async def _llm_identify_gaps(self, findings: list[dict]) -> list[str]:
        """Use the LLM to analyze findings and suggest follow-up queries."""
        try:
            from app.llm.gateway import get_llm_gateway

            gateway = get_llm_gateway()

            summary_lines = []
            for f in findings[:10]:
                query = f.get("query", "?")
                count = len(f.get("results", []))
                cats = {r.get("category", "") for r in f.get("results", []) if r.get("category")}
                summary_lines.append(
                    f"- Query '{query}': {count} results, categories: {', '.join(cats) or 'none'}"
                )
            summary = "\n".join(summary_lines)

            response = await gateway.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research analyst. Given search findings, identify "
                            "knowledge gaps and suggest 1-3 specific follow-up search queries "
                            "to fill those gaps. Return ONLY the queries, one per line."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Current findings:\n{summary}\n\nSuggest follow-up queries:",
                    },
                ],
                task_type="simple",
                temperature=0.3,
            )

            return [
                line.strip().lstrip("- ").lstrip("0123456789.)")
                for line in response.strip().splitlines()
                if line.strip() and len(line.strip()) > 5
            ][:3]
        except Exception as exc:
            logger.debug("LLM gap analysis failed: %s", exc)
            return []

    def _should_continue(self, state: dict) -> bool:
        """Decide whether another research iteration is warranted.

        Stops when:
        - ``max_iterations`` reached
        - No new gaps were found (sub_queries empty)
        - Explicitly marked complete
        """
        if state.get("is_complete", False):
            return False

        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", self._default_max_iterations)
        if iteration >= max_iter:
            return False

        sub_queries = state.get("sub_queries", [])
        if not sub_queries:
            return False

        return True


# ===========================================================================
# Trend Analysis Agent
# ===========================================================================


class TrendAnalysisAgent:
    """Analyze entity data to surface trends.

    Examines categories, styles, geographic concentrations, and temporal
    patterns across the discovered entities.
    """

    async def analyze(self, state: dict) -> dict:
        """Analyze accumulated entities for trends.

        Returns:
            State update with trend findings appended to
            ``intermediate_findings``.
        """
        entities = state.get("entities_discovered", [])
        logger.info(
            "TrendAnalysisAgent: analyzing %d entities for trends", len(entities)
        )

        trends = self._extract_trends(entities)

        trend_finding = {
            "type": "trend_analysis",
            "trends": trends,
            "entity_count": len(entities),
            "source": "trend_agent",
        }

        return {
            "intermediate_findings": [trend_finding],
        }

    # ------------------------------------------------------------------

    def _extract_trends(self, entities: list[dict]) -> dict:
        """Extract trend signals from a list of entity dicts.

        Returns a dict with keys:
        - ``category_distribution`` : {category: count}
        - ``city_distribution``     : {city: count}
        - ``popular_tags``          : {tag: count}
        - ``price_range_spread``    : {range: count}
        - ``emerging``              : list[str]  (categories with few but recent entities)
        """
        category_counts: dict[str, int] = {}
        city_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        price_counts: dict[str, int] = {}

        for entity in entities:
            # Category
            cat = entity.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            # City
            city = entity.get("city", "unknown")
            city_counts[city] = city_counts.get(city, 0) + 1

            # Tags
            for tag in entity.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Price range
            price = entity.get("price_range") or entity.get("price_tier", "unknown")
            price_counts[price] = price_counts.get(price, 0) + 1

        # Identify "emerging" categories (present but low count, suggesting
        # nascent trends rather than established ones)
        total = len(entities) if entities else 1
        emerging = [
            cat
            for cat, count in category_counts.items()
            if 0 < count <= max(total * 0.1, 2)
        ]

        return {
            "category_distribution": category_counts,
            "city_distribution": city_counts,
            "popular_tags": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "price_range_spread": price_counts,
            "emerging": emerging,
        }


# ===========================================================================
# Entity Comparison Agent
# ===========================================================================


class EntityComparisonAgent:
    """Compare two or more entities across multiple dimensions.

    Dimensions include: vibe DNA, category, price tier, geographic
    location, tags, and any custom attributes.
    """

    async def compare(self, state: dict) -> dict:
        """Build a comparison matrix and append it as a finding.

        Returns:
            State update with comparison result in ``intermediate_findings``.
        """
        entities = state.get("entities_discovered", [])
        logger.info(
            "EntityComparisonAgent: comparing %d entities", len(entities)
        )

        if len(entities) < 2:
            comparison = {
                "type": "entity_comparison",
                "error": "Need at least 2 entities to compare",
                "entities_provided": len(entities),
                "matrix": {},
                "source": "comparison_agent",
            }
        else:
            matrix = self._build_comparison_matrix(entities)
            comparison = {
                "type": "entity_comparison",
                "matrix": matrix,
                "entity_count": len(entities),
                "source": "comparison_agent",
            }

        return {
            "intermediate_findings": [comparison],
        }

    # ------------------------------------------------------------------

    def _build_comparison_matrix(self, entities: list[dict]) -> dict:
        """Build a structured comparison across standard dimensions.

        Returns::

            {
                "entities": [ {name, category, ...}, ... ],
                "dimensions": {
                    "category":    [val_a, val_b, ...],
                    "city":        [val_a, val_b, ...],
                    "price_range": [val_a, val_b, ...],
                    "vibe_dna":    [dict_a, dict_b, ...],
                    "tags":        [list_a, list_b, ...],
                },
                "shared_tags":   [...],
                "unique_tags":   {entity_name: [...]},
            }
        """
        dimension_keys = ["category", "city", "price_range", "vibe_dna", "tags"]
        dimensions: dict[str, list[Any]] = {k: [] for k in dimension_keys}
        entity_summaries: list[dict] = []

        for entity in entities:
            entity_summaries.append(
                {
                    "name": entity.get("name", "Unknown"),
                    "id": entity.get("id", ""),
                    "category": entity.get("category", ""),
                    "city": entity.get("city", ""),
                }
            )
            for key in dimension_keys:
                val = entity.get(key)
                if key == "tags" and val is None:
                    val = []
                dimensions[key].append(val)

        # Shared vs unique tags
        all_tag_sets = [set(t) for t in dimensions.get("tags", []) if isinstance(t, list)]
        shared_tags: list[str] = []
        unique_tags: dict[str, list[str]] = {}

        if len(all_tag_sets) >= 2:
            shared = all_tag_sets[0]
            for ts in all_tag_sets[1:]:
                shared = shared & ts
            shared_tags = sorted(shared)

            for i, entity in enumerate(entities):
                name = entity.get("name", f"entity_{i}")
                if i < len(all_tag_sets):
                    unique_tags[name] = sorted(all_tag_sets[i] - set(shared_tags))

        return {
            "entities": entity_summaries,
            "dimensions": dimensions,
            "shared_tags": shared_tags,
            "unique_tags": unique_tags,
        }


# ===========================================================================
# Synthesizer (shared final step for all workflows)
# ===========================================================================


class WorkflowSynthesizer:
    """Produce a final synthesis from accumulated workflow findings.

    Uses the LLM gateway to generate a natural language summary from
    the structured ``intermediate_findings``, with a fallback to
    formatted text if the LLM is unavailable.
    """

    async def synthesize(self, state: dict) -> dict:
        """Synthesize intermediate findings into a final response.

        Returns:
            State update setting ``synthesis`` and ``is_complete``.
        """
        findings = state.get("intermediate_findings", [])
        workflow_type = state.get("workflow_type", "unknown")
        original_query = state.get("original_query", "")
        entities = state.get("entities_discovered", [])

        logger.info(
            "WorkflowSynthesizer: synthesizing %d findings for '%s'",
            len(findings),
            workflow_type,
        )

        # Build structured context for the LLM
        context_parts = [
            f"Workflow: {workflow_type}",
            f"Query: {original_query}",
            f"Findings collected: {len(findings)}",
            f"Entities discovered: {len(entities)}",
            f"Iterations completed: {state.get('iteration', 0)}",
        ]

        # Entity names for context
        entity_names = [e.get("name", "Unknown") for e in entities[:20]]
        if entity_names:
            context_parts.append(f"Top entities: {', '.join(entity_names[:10])}")

        # Trend data
        for finding in findings:
            if "trends" in finding:
                trends = finding["trends"]
                top_cats = list(trends.get("category_distribution", {}).keys())[:5]
                if top_cats:
                    context_parts.append(f"Top categories: {', '.join(top_cats)}")
                top_cities = list(trends.get("city_distribution", {}).keys())[:5]
                if top_cities:
                    context_parts.append(f"Top cities: {', '.join(top_cities)}")
            if "matrix" in finding and finding["matrix"]:
                compared = [e.get("name", "?") for e in finding["matrix"].get("entities", [])]
                if compared:
                    context_parts.append(f"Compared: {', '.join(compared)}")

        context_text = "\n".join(context_parts)

        # Try LLM synthesis, fall back to structured text
        try:
            from app.llm.gateway import get_llm_gateway
            gateway = get_llm_gateway()
            synthesis_text = await gateway.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are ARBOR, a curated discovery assistant for fashion brands "
                            "and venues. Synthesize the workflow findings into a concise, "
                            "insightful response. Be specific about entities and trends."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Synthesize these workflow findings into a response "
                            f"for the query: '{original_query}'\n\n{context_text}"
                        ),
                    },
                ],
                task_type="synthesis",
                temperature=0.7,
            )
        except Exception as exc:
            logger.warning("LLM synthesis failed, using structured fallback: %s", exc)
            synthesis_text = context_text

        return {
            "synthesis": synthesis_text,
            "is_complete": True,
        }


# ===========================================================================
# Workflow Orchestrator
# ===========================================================================


class WorkflowOrchestrator:
    """Create and execute LangGraph workflows for each ``WorkflowType``.

    Each workflow type maps to a distinct graph topology:

    - **DEEP_RESEARCH**: decompose -> search -> analyze -> (loop?) -> synthesize
    - **TREND_ANALYSIS**: decompose -> gather -> extract_trends -> synthesize
    - **ENTITY_COMPARISON**: decompose -> gather -> compare -> synthesize
    - **MARKET_MAPPING**: decompose -> gather -> extract_trends -> synthesize
    - **STYLE_EXPLORATION**: decompose -> search -> analyze -> synthesize

    When ``langgraph`` is unavailable the orchestrator falls back to a
    sequential in-process execution of the same node functions.
    """

    def __init__(self):
        self._decomposer = QueryDecomposer()
        self._researcher = DeepResearchAgent()
        self._trend_agent = TrendAnalysisAgent()
        self._comparison_agent = EntityComparisonAgent()
        self._synthesizer = WorkflowSynthesizer()
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def create_workflow(self, workflow_type: WorkflowType):
        """Build and compile a LangGraph ``StateGraph`` for *workflow_type*.

        Returns:
            A compiled LangGraph runnable, or ``None`` if LangGraph is not
            available (in which case ``run`` will use the fallback path).
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning(
                "LangGraph unavailable - workflow '%s' will use fallback execution",
                workflow_type,
            )
            return None

        if workflow_type == WorkflowType.DEEP_RESEARCH:
            return self._build_deep_research_graph()
        elif workflow_type == WorkflowType.TREND_ANALYSIS:
            return self._build_trend_analysis_graph()
        elif workflow_type == WorkflowType.ENTITY_COMPARISON:
            return self._build_entity_comparison_graph()
        elif workflow_type == WorkflowType.MARKET_MAPPING:
            return self._build_market_mapping_graph()
        elif workflow_type == WorkflowType.STYLE_EXPLORATION:
            return self._build_style_exploration_graph()
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    # ---- individual graph builders ------------------------------------

    def _build_deep_research_graph(self):
        """DEEP_RESEARCH: decompose -> research -> check_complete -> synthesize (with loop)."""
        graph = StateGraph(WorkflowState)

        # -- node functions (closures over self) --
        decomposer = self._decomposer
        researcher = self._researcher
        synthesizer = self._synthesizer

        async def decompose_node(state: WorkflowState) -> dict:
            sub_queries = decomposer.decompose(
                state["original_query"], WorkflowType.DEEP_RESEARCH
            )
            return {"sub_queries": sub_queries, "iteration": 0}

        async def research_node(state: WorkflowState) -> dict:
            return await researcher.research(state)

        async def synthesize_node(state: WorkflowState) -> dict:
            return await synthesizer.synthesize(state)

        def should_continue(state: WorkflowState) -> str:
            if researcher._should_continue(state):
                return "research"
            return "synthesize"

        # -- wire graph --
        graph.add_node("decompose", decompose_node)
        graph.add_node("research", research_node)
        graph.add_node("synthesize", synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "research")
        graph.add_conditional_edges(
            "research",
            should_continue,
            {"research": "research", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _build_trend_analysis_graph(self):
        """TREND_ANALYSIS: decompose -> gather -> extract_trends -> synthesize."""
        graph = StateGraph(WorkflowState)

        decomposer = self._decomposer
        trend_agent = self._trend_agent
        synthesizer = self._synthesizer

        async def decompose_node(state: WorkflowState) -> dict:
            sub_queries = decomposer.decompose(
                state["original_query"], WorkflowType.TREND_ANALYSIS
            )
            return {"sub_queries": sub_queries, "iteration": 0}

        async def gather_node(state: WorkflowState) -> dict:
            """Gather entities for each sub-query via VectorAgent + MetadataAgent."""
            all_findings: list[dict] = []
            all_entities: list[dict] = []
            for sq in state.get("sub_queries", []):
                findings, entities = await _gather_entities_for_query(
                    sq, limit=10, source_label="trend_gather",
                )
                all_findings.extend(findings)
                all_entities.extend(entities)
            return {
                "intermediate_findings": all_findings,
                "entities_discovered": all_entities,
            }

        async def extract_trends_node(state: WorkflowState) -> dict:
            return await trend_agent.analyze(state)

        async def synthesize_node(state: WorkflowState) -> dict:
            return await synthesizer.synthesize(state)

        graph.add_node("decompose", decompose_node)
        graph.add_node("gather", gather_node)
        graph.add_node("extract_trends", extract_trends_node)
        graph.add_node("synthesize", synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "gather")
        graph.add_edge("gather", "extract_trends")
        graph.add_edge("extract_trends", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _build_entity_comparison_graph(self):
        """ENTITY_COMPARISON: decompose -> gather -> compare -> synthesize."""
        graph = StateGraph(WorkflowState)

        decomposer = self._decomposer
        comparison_agent = self._comparison_agent
        synthesizer = self._synthesizer

        async def decompose_node(state: WorkflowState) -> dict:
            sub_queries = decomposer.decompose(
                state["original_query"], WorkflowType.ENTITY_COMPARISON
            )
            return {"sub_queries": sub_queries, "iteration": 0}

        async def gather_node(state: WorkflowState) -> dict:
            """Gather entity data for each side of the comparison via real agents."""
            all_findings: list[dict] = []
            all_entities: list[dict] = []
            for sq in state.get("sub_queries", []):
                findings, entities = await _gather_entities_for_query(
                    sq, limit=10, source_label="comparison_gather",
                )
                all_findings.extend(findings)
                all_entities.extend(entities)
            return {
                "intermediate_findings": all_findings,
                "entities_discovered": all_entities,
            }

        async def compare_node(state: WorkflowState) -> dict:
            return await comparison_agent.compare(state)

        async def synthesize_node(state: WorkflowState) -> dict:
            return await synthesizer.synthesize(state)

        graph.add_node("decompose", decompose_node)
        graph.add_node("gather", gather_node)
        graph.add_node("compare", compare_node)
        graph.add_node("synthesize", synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "gather")
        graph.add_edge("gather", "compare")
        graph.add_edge("compare", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _build_market_mapping_graph(self):
        """MARKET_MAPPING: decompose -> gather -> extract_trends -> synthesize.

        Re-uses the trend analysis topology because market mapping is
        fundamentally a geographic/categorical trend extraction.
        """
        graph = StateGraph(WorkflowState)

        decomposer = self._decomposer
        trend_agent = self._trend_agent
        synthesizer = self._synthesizer

        async def decompose_node(state: WorkflowState) -> dict:
            sub_queries = decomposer.decompose(
                state["original_query"], WorkflowType.MARKET_MAPPING
            )
            return {"sub_queries": sub_queries, "iteration": 0}

        async def gather_node(state: WorkflowState) -> dict:
            """Gather market data for each sub-query via real agents."""
            all_findings: list[dict] = []
            all_entities: list[dict] = []
            for sq in state.get("sub_queries", []):
                findings, entities = await _gather_entities_for_query(
                    sq, limit=15, source_label="market_gather",
                )
                all_findings.extend(findings)
                all_entities.extend(entities)
            return {
                "intermediate_findings": all_findings,
                "entities_discovered": all_entities,
            }

        async def extract_trends_node(state: WorkflowState) -> dict:
            return await trend_agent.analyze(state)

        async def synthesize_node(state: WorkflowState) -> dict:
            return await synthesizer.synthesize(state)

        graph.add_node("decompose", decompose_node)
        graph.add_node("gather", gather_node)
        graph.add_node("extract_trends", extract_trends_node)
        graph.add_node("synthesize", synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "gather")
        graph.add_edge("gather", "extract_trends")
        graph.add_edge("extract_trends", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _build_style_exploration_graph(self):
        """STYLE_EXPLORATION: decompose -> research -> analyze_trends -> synthesize."""
        graph = StateGraph(WorkflowState)

        decomposer = self._decomposer
        researcher = self._researcher
        trend_agent = self._trend_agent
        synthesizer = self._synthesizer

        async def decompose_node(state: WorkflowState) -> dict:
            sub_queries = decomposer.decompose(
                state["original_query"], WorkflowType.STYLE_EXPLORATION
            )
            return {"sub_queries": sub_queries, "iteration": 0}

        async def research_node(state: WorkflowState) -> dict:
            return await researcher.research(state)

        async def analyze_trends_node(state: WorkflowState) -> dict:
            return await trend_agent.analyze(state)

        async def synthesize_node(state: WorkflowState) -> dict:
            return await synthesizer.synthesize(state)

        graph.add_node("decompose", decompose_node)
        graph.add_node("research", research_node)
        graph.add_node("analyze_trends", analyze_trends_node)
        graph.add_node("synthesize", synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "research")
        graph.add_edge("research", "analyze_trends")
        graph.add_edge("analyze_trends", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        workflow_type: WorkflowType,
        query: str,
        max_iterations: int = 3,
    ) -> dict:
        """Execute a full workflow and return the final state.

        Args:
            workflow_type: Which workflow topology to use.
            query: The user's original natural-language query.
            max_iterations: Cap for iterative workflows (DEEP_RESEARCH).

        Returns:
            The final ``WorkflowState`` as a dict, including ``synthesis``
            and ``entities_discovered``.
        """
        logger.info(
            "WorkflowOrchestrator.run: type=%s query='%s' max_iter=%d",
            workflow_type,
            query[:80],
            max_iterations,
        )

        initial_state: dict = {
            "workflow_type": workflow_type.value,
            "original_query": query,
            "sub_queries": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "intermediate_findings": [],
            "entities_discovered": [],
            "synthesis": "",
            "is_complete": False,
        }

        compiled_graph = self.create_workflow(workflow_type)

        if compiled_graph is not None:
            # LangGraph execution
            result = await compiled_graph.ainvoke(initial_state)
            return dict(result)
        else:
            # Fallback: sequential execution without LangGraph
            return await self._fallback_run(workflow_type, initial_state)

    # ------------------------------------------------------------------
    # Fallback execution (no LangGraph)
    # ------------------------------------------------------------------

    async def _fallback_run(
        self, workflow_type: WorkflowType, state: dict
    ) -> dict:
        """Execute the workflow nodes sequentially without LangGraph.

        This allows the module to function (with reduced orchestration
        capabilities) even when ``langgraph`` is not installed.
        """
        logger.info("Fallback execution for workflow '%s'", workflow_type)

        # Step 1: Decompose
        sub_queries = self._decomposer.decompose(
            state["original_query"], workflow_type
        )
        state["sub_queries"] = sub_queries
        state["iteration"] = 0

        # Step 2: Workflow-specific middle steps
        if workflow_type == WorkflowType.DEEP_RESEARCH:
            while self._researcher._should_continue(state):
                update = await self._researcher.research(state)
                state["intermediate_findings"] = state.get(
                    "intermediate_findings", []
                ) + update.get("intermediate_findings", [])
                state["sub_queries"] = update.get("sub_queries", [])
                state["iteration"] = update.get("iteration", state["iteration"] + 1)

        elif workflow_type in (
            WorkflowType.TREND_ANALYSIS,
            WorkflowType.MARKET_MAPPING,
        ):
            # Gather real data first
            for sq in state["sub_queries"]:
                findings, entities = await _gather_entities_for_query(
                    sq, limit=10, source_label=f"{workflow_type.value}_gather",
                )
                state["intermediate_findings"] = state.get(
                    "intermediate_findings", []
                ) + findings
                state["entities_discovered"] = state.get(
                    "entities_discovered", []
                ) + entities
            # Then analyze trends
            trend_update = await self._trend_agent.analyze(state)
            state["intermediate_findings"] = state.get(
                "intermediate_findings", []
            ) + trend_update.get("intermediate_findings", [])

        elif workflow_type == WorkflowType.ENTITY_COMPARISON:
            # Gather real data first
            for sq in state["sub_queries"]:
                findings, entities = await _gather_entities_for_query(
                    sq, limit=10, source_label="comparison_gather",
                )
                state["intermediate_findings"] = state.get(
                    "intermediate_findings", []
                ) + findings
                state["entities_discovered"] = state.get(
                    "entities_discovered", []
                ) + entities
            # Then compare
            comp_update = await self._comparison_agent.compare(state)
            state["intermediate_findings"] = state.get(
                "intermediate_findings", []
            ) + comp_update.get("intermediate_findings", [])

        elif workflow_type == WorkflowType.STYLE_EXPLORATION:
            research_update = await self._researcher.research(state)
            state["intermediate_findings"] = state.get(
                "intermediate_findings", []
            ) + research_update.get("intermediate_findings", [])
            state["iteration"] = research_update.get(
                "iteration", state["iteration"] + 1
            )
            trend_update = await self._trend_agent.analyze(state)
            state["intermediate_findings"] = state.get(
                "intermediate_findings", []
            ) + trend_update.get("intermediate_findings", [])

        # Step 3: Synthesize
        synth_update = await self._synthesizer.synthesize(state)
        state["synthesis"] = synth_update.get("synthesis", "")
        state["is_complete"] = synth_update.get("is_complete", True)

        return state


# ===========================================================================
# Singleton accessor
# ===========================================================================

_orchestrator_instance: WorkflowOrchestrator | None = None


def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Return the singleton ``WorkflowOrchestrator`` instance.

    Thread-safe via module-level global; the orchestrator itself is
    stateless between ``run()`` calls.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = WorkflowOrchestrator()
        logger.info("WorkflowOrchestrator singleton created")
    return _orchestrator_instance
