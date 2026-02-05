"""GraphRAG integration for advanced knowledge graph reasoning."""

from app.db.neo4j.driver import get_neo4j_driver
from app.llm.gateway import LLMGateway


class GraphRAG:
    """GraphRAG for enhanced knowledge graph querying with LLM reasoning."""

    def __init__(self):
        self.driver = get_neo4j_driver()
        self.llm = LLMGateway()

    async def smart_query(self, question: str, query_type: str = "auto") -> dict:
        """Intelligent query that chooses between Cypher and LLM-enhanced search."""
        if query_type == "simple" or self._is_simple_query(question):
            return await self._cypher_query(question)
        elif query_type == "local" or self._mentions_specific_entity(question):
            return await self._local_search(question)
        else:
            return await self._global_search(question)

    async def _cypher_query(self, question: str) -> dict:
        """Generate and execute Cypher from natural language question."""
        cypher = await self._generate_cypher(question)

        async with self.driver.session() as session:
            result = await session.run(cypher)
            records = [dict(record) async for record in result]

        return {"type": "cypher", "results": records, "query": cypher}

    async def _local_search(self, question: str) -> dict:
        """Local search starting from specific entities mentioned in the question."""
        entities = await self._extract_entities(question)

        all_results = []
        for entity_name in entities:
            query = """
            MATCH (e)-[r*1..2]-(connected)
            WHERE e.name CONTAINS $name
            RETURN e, r, connected
            LIMIT 20
            """
            async with self.driver.session() as session:
                result = await session.run(query, name=entity_name)
                records = [dict(record) async for record in result]
                all_results.extend(records)

        summary = await self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize these graph results in context of the question.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nResults: {all_results[:10]}",
                },
            ]
        )

        return {"type": "local", "results": all_results, "summary": summary}

    async def _global_search(self, question: str) -> dict:
        """Global search for patterns and insights across the entire graph."""
        query = """
        MATCH (e:Entity)-[r]->(connected)
        WITH type(r) as rel_type, count(*) as count
        RETURN rel_type, count
        ORDER BY count DESC
        LIMIT 20
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            patterns = [dict(record) async for record in result]

        summary = await self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": "Based on the graph patterns, answer the user's question.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nGraph patterns: {patterns}",
                },
            ]
        )

        return {"type": "global", "patterns": patterns, "summary": summary}

    async def _generate_cypher(self, question: str) -> str:
        """Use LLM to generate Cypher query from natural language."""
        response = await self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a Cypher query for Neo4j. Node types: Entity, AbstractEntity, "
                        "Style, Curator. Relationship types: SELLS_BRAND, IS_HQ_OF, TRAINED_BY, "
                        "INSPIRED_BY, HAS_STYLE, REPRESENTS. Return ONLY the Cypher query."
                    ),
                },
                {"role": "user", "content": question},
            ],
            task_type="classification",
        )
        return response

    async def _extract_entities(self, question: str) -> list[str]:
        """Extract entity names from a question."""
        response = await self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract entity names (shops, brands, people, places) from the text. "
                        "Return as JSON array of strings."
                    ),
                },
                {"role": "user", "content": question},
            ],
            task_type="classification",
        )
        import json

        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return []

    def _is_simple_query(self, question: str) -> bool:
        simple_keywords = ["how many", "list all", "show me", "count"]
        return any(kw in question.lower() for kw in simple_keywords)

    def _mentions_specific_entity(self, question: str) -> bool:
        # Heuristic: questions with proper nouns likely mention specific entities
        words = question.split()
        return any(w[0].isupper() and len(w) > 2 for w in words[1:])
