"""Continuous Learning Feedback Loop — Layer 5.

Connects user feedback (clicks, saves, conversions) back to the enrichment
pipeline for continuous improvement:
1. Implicit feedback: click-through rates, dwell time, conversion signals
2. Explicit feedback: curator overrides, rating corrections
3. Drift detection: compares current scores against gold standard
4. Embedding refresh: re-generates embeddings when scores change significantly
"""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import (
    ArborEnrichment,
    ArborFeedback,
    ArborReviewQueue,
)
from app.ingestion.pipeline.gold_standard import GoldStandardManager

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """Manages continuous improvement of the enrichment pipeline."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.gold_standard = GoldStandardManager(session)

    # ─── Curator Override Processing ──────────────────────────────────

    async def apply_curator_override(
        self,
        entity_type: str,
        source_id: int,
        overridden_scores: dict[str, int],
        overridden_tags: list[str],
        reviewer: str,
        reviewer_notes: str = "",
        promote_to_gold: bool = False,
    ) -> dict:
        """Apply a curator's score override and optionally promote to gold standard.

        Args:
            entity_type: Configured entity type (e.g. "brand", "venue", "product")
            source_id: Source table ID
            overridden_scores: Curator's corrected dimension scores
            overridden_tags: Curator's corrected tags
            reviewer: Who reviewed it
            reviewer_notes: Explanation for the overrides
            promote_to_gold: If True, add this to the gold standard set
        """
        entity_id = f"{entity_type}_{source_id}"

        # Update the enrichment with curator scores
        enrichment = await self.session.execute(
            select(ArborEnrichment).where(
                ArborEnrichment.entity_type == entity_type,
                ArborEnrichment.source_id == source_id,
            )
        )
        enr = enrichment.scalar_one_or_none()
        if enr and enr.vibe_dna:
            vibe_dna = dict(enr.vibe_dna)
            # Merge curator overrides into existing scores
            dims = vibe_dna.get("dimensions", {})
            for dim, score in overridden_scores.items():
                dims[dim] = max(0, min(100, score))
            vibe_dna["dimensions"] = dims
            vibe_dna["tags"] = overridden_tags or vibe_dna.get("tags", [])
            vibe_dna["curator_reviewed"] = True
            vibe_dna["curator_override_at"] = datetime.now(UTC).isoformat()
            vibe_dna["curator"] = reviewer
            vibe_dna["needs_review"] = False

            enr.vibe_dna = vibe_dna
            enr.tags = overridden_tags or enr.tags
            enr.neo4j_synced = False  # Trigger resync
            await self.session.flush()

        # Update review queue item
        await self.session.execute(
            update(ArborReviewQueue)
            .where(
                ArborReviewQueue.entity_type == entity_type,
                ArborReviewQueue.source_id == source_id,
                ArborReviewQueue.status == "needs_review",
            )
            .values(
                status="approved",
                reviewer=reviewer,
                reviewed_at=datetime.now(UTC),
                reviewer_notes=reviewer_notes,
                overridden_scores=overridden_scores,
                overridden_tags=overridden_tags,
            )
        )

        # Optionally promote to gold standard
        if promote_to_gold:
            await self.gold_standard.add_gold_entity(
                entity_type=entity_type,
                source_id=source_id,
                scores=overridden_scores,
                tags=overridden_tags,
                curator_notes=reviewer_notes,
                curated_by=reviewer,
            )
            logger.info(f"Promoted {entity_id} to gold standard")

        await self.session.commit()

        return {
            "entity_id": entity_id,
            "overrides_applied": True,
            "promoted_to_gold": promote_to_gold,
        }

    # ─── Implicit Feedback Analysis ───────────────────────────────────

    async def analyze_implicit_feedback(self, since_hours: int = 24) -> dict:
        """Analyze user feedback to identify entities that may need re-scoring.

        Looks for patterns like:
        - Entities consistently dismissed (low reward) despite high scores
        - Entities getting high engagement despite low scores
        """
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)

        # Get entities with significant feedback
        result = await self.session.execute(
            select(
                ArborFeedback.entity_type,
                ArborFeedback.source_id,
                func.count().label("interaction_count"),
                func.avg(ArborFeedback.reward).label("avg_reward"),
            )
            .where(ArborFeedback.created_at >= cutoff)
            .group_by(ArborFeedback.entity_type, ArborFeedback.source_id)
            .having(func.count() >= 3)
        )
        feedback_stats = list(result)

        anomalies = []
        for row in feedback_stats:
            entity_id = f"{row.entity_type}_{row.source_id}"
            avg_reward = float(row.avg_reward) if row.avg_reward else 0.0

            # Check against current enrichment scores
            enr_result = await self.session.execute(
                select(ArborEnrichment).where(
                    ArborEnrichment.entity_type == row.entity_type,
                    ArborEnrichment.source_id == row.source_id,
                )
            )
            enr = enr_result.scalar_one_or_none()

            if enr and enr.vibe_dna:
                confidence = enr.vibe_dna.get("overall_confidence", 0.5)

                # High confidence but low engagement = possible over-scoring
                if confidence > 0.7 and avg_reward < 0.2:
                    anomalies.append(
                        {
                            "entity_id": entity_id,
                            "type": "over_scored",
                            "avg_reward": round(avg_reward, 3),
                            "confidence": round(confidence, 3),
                            "interactions": row.interaction_count,
                        }
                    )

                # Low confidence but high engagement = possible under-scoring
                if confidence < 0.4 and avg_reward > 0.6:
                    anomalies.append(
                        {
                            "entity_id": entity_id,
                            "type": "under_scored",
                            "avg_reward": round(avg_reward, 3),
                            "confidence": round(confidence, 3),
                            "interactions": row.interaction_count,
                        }
                    )

        return {
            "period_hours": since_hours,
            "entities_analyzed": len(feedback_stats),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
        }

    # ─── Drift Detection ─────────────────────────────────────────────

    async def detect_drift(self) -> dict:
        """Compare current enrichment scores against gold standard.

        Returns a drift report indicating whether recalibration is needed.
        """
        # Get all current enrichment scores
        result = await self.session.execute(select(ArborEnrichment))
        enrichments = list(result.scalars().all())

        current_scores = {}
        for enr in enrichments:
            entity_id = f"{enr.entity_type}_{enr.source_id}"
            if enr.vibe_dna and "dimensions" in enr.vibe_dna:
                current_scores[entity_id] = enr.vibe_dna["dimensions"]

        return await self.gold_standard.compute_drift(current_scores)

    # ─── Embedding Refresh ────────────────────────────────────────────

    async def get_entities_needing_refresh(self, score_change_threshold: int = 10) -> list[dict]:
        """Find entities whose scores changed significantly and need embedding refresh.

        Looks for enrichments where neo4j_synced is False (indicating a change).
        """
        result = await self.session.execute(
            select(ArborEnrichment)
            .where(ArborEnrichment.neo4j_synced == False)  # noqa: E712
            .limit(100)
        )
        enrichments = list(result.scalars().all())

        entities_to_refresh = []
        for enr in enrichments:
            entities_to_refresh.append(
                {
                    "entity_type": enr.entity_type,
                    "source_id": enr.source_id,
                    "entity_id": f"{enr.entity_type}_{enr.source_id}",
                }
            )

        return entities_to_refresh

    # ─── Review Queue Stats ───────────────────────────────────────────

    async def get_review_queue_stats(self) -> dict:
        """Get statistics about the review queue."""
        total = await self.session.execute(select(func.count()).select_from(ArborReviewQueue))
        pending = await self.session.execute(
            select(func.count())
            .select_from(ArborReviewQueue)
            .where(ArborReviewQueue.status == "needs_review")
        )
        approved = await self.session.execute(
            select(func.count())
            .select_from(ArborReviewQueue)
            .where(ArborReviewQueue.status == "approved")
        )

        return {
            "total": total.scalar_one(),
            "pending": pending.scalar_one(),
            "approved": approved.scalar_one(),
        }
