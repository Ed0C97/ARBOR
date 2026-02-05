"""Curator Agent - synthesizes results into final response."""

import logging

from app.llm.gateway import get_llm_gateway
from app.llm.prompts import CURATOR_PERSONA, load_prompt

logger = logging.getLogger(__name__)


class CuratorAgent:
    """Synthesize results from all agents into a curated response."""

    def __init__(self):
        self.gateway = get_llm_gateway()
        try:
            self.system_prompt = load_prompt(CURATOR_PERSONA)
        except FileNotFoundError:
            self.system_prompt = self._default_prompt()

    async def synthesize(
        self,
        query: str,
        vector_results: list[dict],
        graph_results: list[dict],
        metadata_results: list[dict],
        intent: str,
    ) -> dict:
        """Generate the final curated response."""
        context = self._format_context(vector_results, graph_results, metadata_results)

        prompt = f"""Available Context Data:
{context}

User Query: {query}
Detected Intent: {intent}

Based on the above context, provide your curated response. Include:
1. Direct answer to the user's question
2. Top recommendations with match scores (if applicable)
3. Any relevant insights from the knowledge graph
4. Honest assessment of confidence in the recommendations"""

        response = await self.gateway.complete(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            task_type="synthesis",
            temperature=0.7,
        )

        # Build recommendation list
        recommendations = self._build_recommendations(vector_results, metadata_results)

        return {
            "response_text": response,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(vector_results, graph_results),
        }

    def _format_context(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        metadata_results: list[dict],
    ) -> str:
        """Format all results into a context string for the LLM."""
        sections = []

        if vector_results:
            sections.append("SEMANTIC SEARCH RESULTS:")
            for i, vr in enumerate(vector_results[:5], 1):
                sections.append(
                    f"  {i}. {vr.get('name', 'Unknown')} "
                    f"(Match: {vr.get('score', 0):.0%}) "
                    f"Category: {vr.get('category', 'N/A')} "
                    f"City: {vr.get('city', 'N/A')} "
                    f"Tags: {', '.join(vr.get('tags', []))}"
                )

        if graph_results:
            sections.append("\nKNOWLEDGE GRAPH RESULTS:")
            for gr in graph_results[:5]:
                if gr.get("type") == "lineage":
                    sections.append(
                        f"  - {gr.get('entity')} trained by {gr.get('mentor')}"
                    )
                elif gr.get("type") == "style_related":
                    sections.append(
                        f"  - {gr.get('name')} shares style: {gr.get('shared_style')}"
                    )
                elif gr.get("type") == "brand_retailer":
                    sections.append(
                        f"  - {gr.get('name')} ({gr.get('relationship')})"
                    )

        if metadata_results:
            sections.append("\nSTRUCTURED DATA:")
            for mr in metadata_results[:3]:
                sections.append(
                    f"  - {mr.get('name')} [{mr.get('category')}] "
                    f"Price tier: {mr.get('price_tier', 'N/A')}"
                )

        return "\n".join(sections) if sections else "No data available for this query."

    def _build_recommendations(
        self, vector_results: list[dict], metadata_results: list[dict]
    ) -> list[dict]:
        """Build deduplicated recommendation list."""
        seen_names = set()
        recommendations = []

        for vr in vector_results[:10]:
            name = vr.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                recommendations.append(
                    {
                        "id": vr.get("id", ""),
                        "name": name,
                        "score": vr.get("score", 0),
                        "category": vr.get("category", ""),
                        "city": vr.get("city", ""),
                        "tags": vr.get("tags", []),
                        "dimensions": vr.get("dimensions", {}),
                    }
                )

        return recommendations

    def _calculate_confidence(
        self, vector_results: list[dict], graph_results: list[dict]
    ) -> float:
        """Calculate confidence score for the response."""
        if not vector_results:
            return 0.3

        top_score = vector_results[0].get("score", 0) if vector_results else 0
        has_graph = 1.0 if graph_results else 0.0
        result_count = min(len(vector_results) / 5, 1.0)

        return min(0.5 * top_score + 0.3 * result_count + 0.2 * has_graph, 1.0)

    def _default_prompt(self) -> str:
        return (
            "You are The Curator, an expert advisor. Answer based ONLY on provided context. "
            "Never hallucinate entities. Be warm but professional."
        )
