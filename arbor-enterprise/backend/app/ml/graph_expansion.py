"""Graph auto-expansion module for ARBOR Enterprise.

Automatically discovers and creates new relationships between entities
in the Neo4j knowledge graph by applying multiple expansion strategies:

- Style similarity via vibe_dna cosine distance
- Geographic proximity via Haversine formula
- Category bridging (brand <-> venue retail links)
- Embedding-cluster neighbors via Qdrant
- LLM-inferred semantic relationships

Candidates above an auto-approve threshold are written directly to Neo4j;
lower-confidence candidates are queued for human review.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RelationshipCandidate:
    """A proposed relationship discovered by an expansion strategy.

    Attributes:
        source_id: Identifier of the source entity.
        target_id: Identifier of the target entity.
        rel_type: Relationship label (e.g. ``SIMILAR_VIBE``, ``NEAR``).
        confidence: Score in [0, 1] indicating how certain the discovery is.
        evidence: Human-readable explanation of why the relationship was proposed.
        discovered_by: Name of the strategy that produced this candidate.
    """

    source_id: str
    target_id: str
    rel_type: str
    confidence: float
    evidence: str
    discovered_by: str


# ---------------------------------------------------------------------------
# Expansion strategies
# ---------------------------------------------------------------------------


class ExpansionStrategy(str, Enum):
    """Available strategies for discovering new graph relationships."""

    STYLE_SIMILARITY = "style_similarity"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    CATEGORY_BRIDGE = "category_bridge"
    LLM_INFERENCE = "llm_inference"
    EMBEDDING_CLUSTER = "embedding_cluster"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector (must be same length as *a*).

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 when either vector has
        zero magnitude.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute the great-circle distance between two points on Earth.

    Uses the Haversine formula.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    earth_radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


# ---------------------------------------------------------------------------
# Core expander
# ---------------------------------------------------------------------------


class GraphExpander:
    """Discovers candidate relationships using multiple expansion strategies.

    Each ``expand_by_*`` method queries the Neo4j graph (and optionally
    Qdrant) for entities related to a given entity, scores them, and
    returns a list of :class:`RelationshipCandidate` objects.

    Usage::

        expander = GraphExpander()
        candidates = await expander.run_full_expansion("entity-123")
        await expander.apply_candidates(candidates)
    """

    def __init__(self) -> None:
        self._neo4j_driver = None
        self._qdrant_client = None

    # ------------------------------------------------------------------
    # Lazy connections
    # ------------------------------------------------------------------

    def _get_neo4j_driver(self):
        """Return a lazy-loaded Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase

                self._neo4j_driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
                logger.info("Neo4j driver initialised: %s", settings.neo4j_uri)
            except Exception as exc:
                logger.warning("Failed to connect to Neo4j: %s", exc)
        return self._neo4j_driver

    def _get_qdrant_client(self):
        """Return a lazy-loaded Qdrant client."""
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient

                self._qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    prefer_grpc=settings.qdrant_prefer_grpc,
                )
                logger.info("Qdrant client initialised: %s", settings.qdrant_url)
            except Exception as exc:
                logger.warning("Failed to connect to Qdrant: %s", exc)
        return self._qdrant_client

    # ------------------------------------------------------------------
    # Strategy: style similarity
    # ------------------------------------------------------------------

    async def expand_by_style_similarity(
        self,
        entity_id: str,
        threshold: float = 0.7,
    ) -> list[RelationshipCandidate]:
        """Find entities with similar ``vibe_dna`` vectors.

        Queries Neo4j for the source entity's vibe_dna, then compares it
        against every other entity with a vibe_dna property using cosine
        similarity.  Pairs above *threshold* produce ``SIMILAR_VIBE``
        candidates.

        Args:
            entity_id: The source entity to expand from.
            threshold: Minimum cosine similarity to suggest a relationship.

        Returns:
            List of relationship candidates.
        """
        candidates: list[RelationshipCandidate] = []
        driver = self._get_neo4j_driver()
        if driver is None:
            logger.warning("Neo4j unavailable — skipping style similarity expansion")
            return candidates

        try:
            with driver.session() as session:
                # Fetch the source entity's vibe_dna
                source_result = session.run(
                    "MATCH (e:Entity {id: $id}) RETURN e.vibe_dna AS vibe_dna",
                    id=entity_id,
                )
                source_record = source_result.single()
                if source_record is None or source_record["vibe_dna"] is None:
                    logger.debug("Entity %s has no vibe_dna — skipping", entity_id)
                    return candidates

                source_dna: list[float] = source_record["vibe_dna"]

                # Fetch all other entities with vibe_dna
                others_result = session.run(
                    "MATCH (e:Entity) WHERE e.id <> $id AND e.vibe_dna IS NOT NULL "
                    "RETURN e.id AS id, e.vibe_dna AS vibe_dna, e.name AS name",
                    id=entity_id,
                )

                for record in others_result:
                    sim = _cosine_similarity(source_dna, record["vibe_dna"])
                    if sim >= threshold:
                        candidates.append(
                            RelationshipCandidate(
                                source_id=entity_id,
                                target_id=record["id"],
                                rel_type="SIMILAR_VIBE",
                                confidence=round(sim, 4),
                                evidence=(
                                    f"Cosine similarity of vibe_dna vectors: {sim:.4f} "
                                    f"(threshold={threshold})"
                                ),
                                discovered_by=ExpansionStrategy.STYLE_SIMILARITY.value,
                            )
                        )

            logger.info(
                "Style similarity expansion for %s: %d candidates (threshold=%.2f)",
                entity_id,
                len(candidates),
                threshold,
            )
        except Exception as exc:
            logger.error("Style similarity expansion failed for %s: %s", entity_id, exc)

        return candidates

    # ------------------------------------------------------------------
    # Strategy: geographic proximity
    # ------------------------------------------------------------------

    async def expand_by_geographic_proximity(
        self,
        entity_id: str,
        radius_km: float = 2.0,
    ) -> list[RelationshipCandidate]:
        """Find entities within *radius_km* using the Haversine formula.

        Queries Neo4j for entities with latitude/longitude properties and
        calculates the great-circle distance from the source entity.
        Entities within the radius produce ``NEAR`` candidates.

        Args:
            entity_id: The source entity to expand from.
            radius_km: Search radius in kilometres.

        Returns:
            List of relationship candidates.
        """
        candidates: list[RelationshipCandidate] = []
        driver = self._get_neo4j_driver()
        if driver is None:
            logger.warning("Neo4j unavailable — skipping geographic proximity expansion")
            return candidates

        try:
            with driver.session() as session:
                # Fetch source coordinates
                source_result = session.run(
                    "MATCH (e:Entity {id: $id}) " "RETURN e.latitude AS lat, e.longitude AS lon",
                    id=entity_id,
                )
                source_record = source_result.single()
                if source_record is None:
                    return candidates

                src_lat = source_record["lat"]
                src_lon = source_record["lon"]
                if src_lat is None or src_lon is None:
                    logger.debug("Entity %s has no coordinates — skipping", entity_id)
                    return candidates

                # Fetch all other geolocated entities
                others_result = session.run(
                    "MATCH (e:Entity) WHERE e.id <> $id "
                    "AND e.latitude IS NOT NULL AND e.longitude IS NOT NULL "
                    "RETURN e.id AS id, e.latitude AS lat, e.longitude AS lon, "
                    "e.name AS name",
                    id=entity_id,
                )

                for record in others_result:
                    dist = _haversine_km(src_lat, src_lon, record["lat"], record["lon"])
                    if dist <= radius_km:
                        # Confidence is inversely proportional to distance
                        confidence = max(0.0, 1.0 - (dist / radius_km))
                        candidates.append(
                            RelationshipCandidate(
                                source_id=entity_id,
                                target_id=record["id"],
                                rel_type="NEAR",
                                confidence=round(confidence, 4),
                                evidence=(
                                    f"Geographic distance: {dist:.3f} km "
                                    f"(radius={radius_km} km)"
                                ),
                                discovered_by=ExpansionStrategy.GEOGRAPHIC_PROXIMITY.value,
                            )
                        )

            logger.info(
                "Geographic proximity expansion for %s: %d candidates (radius=%.1f km)",
                entity_id,
                len(candidates),
                radius_km,
            )
        except Exception as exc:
            logger.error("Geographic proximity expansion failed for %s: %s", entity_id, exc)

        return candidates

    # ------------------------------------------------------------------
    # Strategy: category bridge
    # ------------------------------------------------------------------

    async def expand_by_category_bridge(
        self,
        entity_id: str,
    ) -> list[RelationshipCandidate]:
        """Create brand-to-venue retail relationships.

        If the source entity is a brand with a ``retailer`` property that
        references existing venue entities, propose ``SELLS_AT`` /
        ``AVAILABLE_AT`` relationships between the brand and those venues.

        Args:
            entity_id: The source entity to expand from.

        Returns:
            List of relationship candidates.
        """
        candidates: list[RelationshipCandidate] = []
        driver = self._get_neo4j_driver()
        if driver is None:
            logger.warning("Neo4j unavailable — skipping category bridge expansion")
            return candidates

        try:
            with driver.session() as session:
                # Check if source entity has retailer/cross-reference data
                entity_types = settings.get_entity_types()

                brand_result = session.run(
                    "MATCH (b:Entity {id: $id}) "
                    "WHERE b.retailers IS NOT NULL "
                    "RETURN b.id AS id, b.retailers AS retailers, "
                    "b.name AS name, b.entity_type AS entity_type",
                    id=entity_id,
                )
                brand_record = brand_result.single()

                if brand_record is not None:
                    retailers = brand_record["retailers"]
                    if isinstance(retailers, str):
                        retailers = [r.strip() for r in retailers.split(",")]

                    source_type = brand_record["entity_type"]

                    # Match retailer names to entities of other types
                    for retailer_name in retailers:
                        venue_result = session.run(
                            "MATCH (v:Entity) "
                            "WHERE v.entity_type <> $source_type "
                            "AND toLower(v.name) CONTAINS toLower($name) "
                            "RETURN v.id AS id, v.name AS name",
                            name=retailer_name,
                            source_type=source_type,
                        )
                        for venue in venue_result:
                            # SELLS_AT: source -> target
                            candidates.append(
                                RelationshipCandidate(
                                    source_id=entity_id,
                                    target_id=venue["id"],
                                    rel_type="SELLS_AT",
                                    confidence=0.75,
                                    evidence=(
                                        f"{source_type} '{brand_record['name']}' lists "
                                        f"retailer '{retailer_name}' matching "
                                        f"entity '{venue['name']}'"
                                    ),
                                    discovered_by=ExpansionStrategy.CATEGORY_BRIDGE.value,
                                )
                            )
                            # AVAILABLE_AT: target -> source (reverse)
                            candidates.append(
                                RelationshipCandidate(
                                    source_id=venue["id"],
                                    target_id=entity_id,
                                    rel_type="AVAILABLE_AT",
                                    confidence=0.75,
                                    evidence=(
                                        f"Entity '{venue['name']}' carries {source_type} "
                                        f"'{brand_record['name']}' "
                                        f"(retailer field match)"
                                    ),
                                    discovered_by=ExpansionStrategy.CATEGORY_BRIDGE.value,
                                )
                            )

            logger.info(
                "Category bridge expansion for %s: %d candidates",
                entity_id,
                len(candidates),
            )
        except Exception as exc:
            logger.error("Category bridge expansion failed for %s: %s", entity_id, exc)

        return candidates

    # ------------------------------------------------------------------
    # Strategy: embedding cluster
    # ------------------------------------------------------------------

    async def expand_by_embedding_cluster(
        self,
        entity_id: str,
        top_k: int = 10,
        threshold: float = 0.8,
    ) -> list[RelationshipCandidate]:
        """Find nearest neighbours in embedding space via Qdrant.

        Retrieves the entity's embedding vector from Qdrant, searches for
        the ``top_k`` most similar vectors, and proposes
        ``EMBEDDING_NEIGHBOR`` relationships for those exceeding
        *threshold*.

        Args:
            entity_id: The source entity to expand from.
            top_k: Number of nearest neighbours to retrieve.
            threshold: Minimum similarity score to suggest a relationship.

        Returns:
            List of relationship candidates.
        """
        candidates: list[RelationshipCandidate] = []
        client = self._get_qdrant_client()
        if client is None:
            logger.warning("Qdrant unavailable — skipping embedding cluster expansion")
            return candidates

        try:
            # Retrieve the entity's stored vector
            points = client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[entity_id],
                with_vectors=True,
            )

            if not points or points[0].vector is None:
                logger.debug("Entity %s not found in Qdrant collection — skipping", entity_id)
                return candidates

            query_vector = points[0].vector

            # Search for nearest neighbours
            search_results = client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_vector,
                limit=top_k + 1,  # +1 to exclude self
                with_payload=True,
            )

            for hit in search_results:
                # Skip self-match
                hit_id = str(hit.id)
                if hit_id == entity_id:
                    continue

                if hit.score >= threshold:
                    candidates.append(
                        RelationshipCandidate(
                            source_id=entity_id,
                            target_id=hit_id,
                            rel_type="EMBEDDING_NEIGHBOR",
                            confidence=round(hit.score, 4),
                            evidence=(
                                f"Qdrant embedding similarity: {hit.score:.4f} "
                                f"(top_k={top_k}, threshold={threshold})"
                            ),
                            discovered_by=ExpansionStrategy.EMBEDDING_CLUSTER.value,
                        )
                    )

            logger.info(
                "Embedding cluster expansion for %s: %d candidates " "(top_k=%d, threshold=%.2f)",
                entity_id,
                len(candidates),
                top_k,
                threshold,
            )
        except Exception as exc:
            logger.error("Embedding cluster expansion failed for %s: %s", entity_id, exc)

        return candidates

    # ------------------------------------------------------------------
    # Full expansion
    # ------------------------------------------------------------------

    async def run_full_expansion(
        self,
        entity_id: str,
    ) -> list[RelationshipCandidate]:
        """Run all expansion strategies and deduplicate results.

        Executes every ``expand_by_*`` method, merges the results, and
        removes duplicates (keeping the candidate with the highest
        confidence for each ``(source_id, target_id, rel_type)`` triple).

        Args:
            entity_id: The entity to expand.

        Returns:
            Deduplicated list of relationship candidates across all strategies.
        """
        logger.info("Running full expansion for entity %s", entity_id)

        all_candidates: list[RelationshipCandidate] = []

        # Gather candidates from each strategy
        all_candidates.extend(await self.expand_by_style_similarity(entity_id))
        all_candidates.extend(await self.expand_by_geographic_proximity(entity_id))
        all_candidates.extend(await self.expand_by_category_bridge(entity_id))
        all_candidates.extend(await self.expand_by_embedding_cluster(entity_id))

        # Deduplicate: keep highest confidence per (source, target, rel_type)
        best: dict[tuple[str, str, str], RelationshipCandidate] = {}
        for candidate in all_candidates:
            key = (candidate.source_id, candidate.target_id, candidate.rel_type)
            existing = best.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                best[key] = candidate

        deduplicated = list(best.values())

        logger.info(
            "Full expansion for %s: %d raw -> %d deduplicated candidates",
            entity_id,
            len(all_candidates),
            len(deduplicated),
        )
        return deduplicated

    # ------------------------------------------------------------------
    # Applying candidates
    # ------------------------------------------------------------------

    async def apply_candidates(
        self,
        candidates: list[RelationshipCandidate],
        auto_approve_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Write high-confidence candidates to Neo4j; queue the rest.

        Candidates whose confidence meets or exceeds *auto_approve_threshold*
        are written directly to the graph as new relationships.  Those below
        the threshold are added to the review queue for human approval.

        Args:
            candidates: List of relationship candidates to process.
            auto_approve_threshold: Minimum confidence for automatic approval.

        Returns:
            Dict with counts of approved and queued candidates.
        """
        approved: list[RelationshipCandidate] = []
        queued: list[RelationshipCandidate] = []

        for candidate in candidates:
            if candidate.confidence >= auto_approve_threshold:
                approved.append(candidate)
            else:
                queued.append(candidate)

        # Write approved candidates to Neo4j
        driver = self._get_neo4j_driver()
        written = 0
        if driver is not None and approved:
            try:
                with driver.session() as session:
                    for candidate in approved:
                        session.run(
                            f"MATCH (a:Entity {{id: $source_id}}) "
                            f"MATCH (b:Entity {{id: $target_id}}) "
                            f"MERGE (a)-[r:{candidate.rel_type}]->(b) "
                            f"SET r.confidence = $confidence, "
                            f"    r.evidence = $evidence, "
                            f"    r.discovered_by = $discovered_by, "
                            f"    r.created_at = datetime()",
                            source_id=candidate.source_id,
                            target_id=candidate.target_id,
                            confidence=candidate.confidence,
                            evidence=candidate.evidence,
                            discovered_by=candidate.discovered_by,
                        )
                        written += 1
                logger.info(
                    "Applied %d candidates to Neo4j (auto-approved >= %.2f)",
                    written,
                    auto_approve_threshold,
                )
            except Exception as exc:
                logger.error("Failed to write candidates to Neo4j: %s", exc)

        # Queue remaining candidates for review
        if queued:
            scheduler = get_auto_expansion_scheduler()
            scheduler.review_queue.extend(queued)
            logger.info(
                "Queued %d candidates for human review (below %.2f threshold)",
                len(queued),
                auto_approve_threshold,
            )

        return {
            "approved": written,
            "queued": len(queued),
            "total": len(candidates),
            "auto_approve_threshold": auto_approve_threshold,
        }


# ---------------------------------------------------------------------------
# Auto-expansion scheduler (singleton)
# ---------------------------------------------------------------------------


class AutoExpansionScheduler:
    """Queues entities for batch graph expansion.

    Maintains a simple FIFO queue of entity IDs.  Call
    :meth:`schedule_expansion` to enqueue, then :meth:`run_batch` to
    process up to *batch_size* entities at once.

    Uses the singleton pattern — obtain via :func:`get_auto_expansion_scheduler`.

    Usage::

        scheduler = get_auto_expansion_scheduler()
        scheduler.schedule_expansion(["entity-1", "entity-2"])
        results = await scheduler.run_batch(batch_size=50)
    """

    def __init__(self) -> None:
        self._queue: list[str] = []
        self.review_queue: list[RelationshipCandidate] = []
        self._expander = GraphExpander()
        self._total_processed: int = 0

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule_expansion(self, entity_ids: list[str]) -> None:
        """Add entity IDs to the expansion queue.

        Duplicates are silently ignored to avoid redundant work.

        Args:
            entity_ids: List of entity identifiers to queue.
        """
        existing = set(self._queue)
        added = 0
        for eid in entity_ids:
            if eid not in existing:
                self._queue.append(eid)
                existing.add(eid)
                added += 1

        logger.info(
            "Scheduled %d entities for expansion (queue size: %d)",
            added,
            len(self._queue),
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def run_batch(
        self,
        batch_size: int = 50,
    ) -> list[RelationshipCandidate]:
        """Process up to *batch_size* queued entities.

        For each entity, runs :meth:`GraphExpander.run_full_expansion`
        and collects all candidates.  Processed entities are removed
        from the queue.

        Args:
            batch_size: Maximum number of entities to process in one batch.

        Returns:
            Aggregated list of relationship candidates from the batch.
        """
        batch = self._queue[:batch_size]
        self._queue = self._queue[batch_size:]

        if not batch:
            logger.debug("Expansion queue is empty — nothing to process")
            return []

        logger.info(
            "Processing expansion batch: %d entities (remaining: %d)",
            len(batch),
            len(self._queue),
        )

        all_candidates: list[RelationshipCandidate] = []
        for entity_id in batch:
            try:
                candidates = await self._expander.run_full_expansion(entity_id)
                all_candidates.extend(candidates)
            except Exception as exc:
                logger.error("Expansion failed for entity %s: %s", entity_id, exc)

        self._total_processed += len(batch)

        logger.info(
            "Batch complete: %d entities processed, %d candidates found " "(total processed: %d)",
            len(batch),
            len(all_candidates),
            self._total_processed,
        )
        return all_candidates

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return current scheduler statistics.

        Returns:
            Dict with queue size, review queue size, and total processed.
        """
        return {
            "queue_size": len(self._queue),
            "review_queue_size": len(self.review_queue),
            "total_processed": self._total_processed,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_scheduler: AutoExpansionScheduler | None = None


def get_auto_expansion_scheduler() -> AutoExpansionScheduler:
    """Return the singleton AutoExpansionScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AutoExpansionScheduler()
    return _scheduler
