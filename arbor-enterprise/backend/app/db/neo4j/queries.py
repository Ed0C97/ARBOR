"""Neo4j Cypher queries for knowledge graph operations."""

from app.db.neo4j.driver import get_neo4j_driver


class Neo4jQueries:
    """Knowledge graph query operations."""

    def __init__(self):
        self.driver = get_neo4j_driver()

    async def create_entity_node(
        self, entity_id: str, name: str, category: str, city: str | None = None
    ) -> None:
        """Create or update an entity node."""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name, e.category = $category, e.city = $city
        """
        async with self.driver.session() as session:
            await session.run(query, id=entity_id, name=name, category=category, city=city)

    async def create_abstract_entity_node(
        self, entity_id: str, name: str, category: str | None = None, origin: str | None = None
    ) -> None:
        """Create or update an abstract entity (brand/concept) node."""
        query = """
        MERGE (a:AbstractEntity {id: $id})
        SET a.name = $name, a.category = $category, a.origin = $origin
        """
        async with self.driver.session() as session:
            await session.run(query, id=entity_id, name=name, category=category, origin=origin)

    async def create_style_node(self, name: str, description: str = "") -> None:
        """Create or update a style node."""
        query = """
        MERGE (s:Style {name: $name})
        SET s.description = $description
        """
        async with self.driver.session() as session:
            await session.run(query, name=name, description=description)

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        """Create a relationship between two nodes."""
        props = properties or {}
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        async with self.driver.session() as session:
            await session.run(query, from_id=from_id, to_id=to_id, props=props)

    async def create_style_relationship(self, entity_id: str, style_name: str) -> None:
        """Link an entity to a style."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (s:Style {name: $style_name})
        MERGE (e)-[:HAS_STYLE]->(s)
        """
        async with self.driver.session() as session:
            await session.run(query, entity_id=entity_id, style_name=style_name)

    async def find_lineage(self, entity_name: str, depth: int = 3) -> list[dict]:
        """Find the training/mentorship lineage of an entity."""
        query = """
        MATCH path = (e:Entity {name: $name})-[:TRAINED_BY*1..3]->(m:Entity)
        RETURN e.name as entity, m.name as mentor,
               length(path) as distance
        ORDER BY distance
        """
        async with self.driver.session() as session:
            result = await session.run(query, name=entity_name)
            return [dict(record) async for record in result]

    async def find_related_by_style(self, entity_name: str) -> list[dict]:
        """Find entities that share a style with the given entity."""
        query = """
        MATCH (e:Entity {name: $name})-[:HAS_STYLE]->(s:Style)<-[:HAS_STYLE]-(related:Entity)
        WHERE related.name <> e.name
        RETURN related.id as id, related.name as name,
               related.category as category, related.city as city,
               s.name as shared_style
        """
        async with self.driver.session() as session:
            result = await session.run(query, name=entity_name)
            return [dict(record) async for record in result]

    async def find_brand_retailers(self, brand_name: str, city: str | None = None) -> list[dict]:
        """Find entities that sell a specific brand."""
        query = """
        MATCH (v:Entity)-[r:SELLS_BRAND|IS_HQ_OF]->(b:AbstractEntity {name: $brand})
        WHERE ($city IS NULL OR v.city = $city)
        RETURN v.id as id, v.name as name, v.city as city,
               type(r) as relationship_type
        """
        async with self.driver.session() as session:
            result = await session.run(query, brand=brand_name, city=city)
            return [dict(record) async for record in result]

    async def get_entity_graph(self, entity_id: str, depth: int = 2) -> list[dict]:
        """Get the full graph around an entity."""
        query = """
        MATCH path = (e:Entity {id: $entity_id})-[*1..2]-(connected)
        RETURN path
        LIMIT 50
        """
        async with self.driver.session() as session:
            result = await session.run(query, entity_id=entity_id)
            return [dict(record) async for record in result]

    async def get_full_graph(self, limit: int = 200) -> dict:
        """Get the entire knowledge graph (nodes and links) up to a limit."""
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, limit=limit)
            nodes = {}
            links = []

            async for record in result:
                n = record["n"]
                m = record["m"]
                r = record["r"]

                # Use elementId or id property
                n_id = n.get("id") or n.element_id

                if n_id not in nodes:
                    nodes[n_id] = dict(n)
                    nodes[n_id]["id"] = n_id
                    nodes[n_id]["labels"] = list(n.labels)

                # If there is a relationship
                if r is not None and m is not None:
                    m_id = m.get("id") or m.element_id

                    if m_id not in nodes:
                        nodes[m_id] = dict(m)
                        nodes[m_id]["id"] = m_id
                        nodes[m_id]["labels"] = list(m.labels)

                    links.append(
                        {"source": n_id, "target": m_id, "type": r.type, "properties": dict(r)}
                    )

            return {"nodes": list(nodes.values()), "links": links}
