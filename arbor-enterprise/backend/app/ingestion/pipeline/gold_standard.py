"""Gold Standard management system.

Manages the curated reference entities used for calibrating the scoring engine.
Curators assign ground-truth Vibe DNA scores to selected entities, and these
are used as few-shot examples and calibration anchors.
"""

import logging
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres.models import (
    ArborGoldStandard,
    Brand,
    Venue,
)
from app.ingestion.pipeline.schemas import (
    DimensionName,
    GoldStandardEntity,
)

logger = logging.getLogger(__name__)


class GoldStandardRepository:
    """CRUD for the gold standard reference set."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, entity_type: str, source_id: int) -> ArborGoldStandard | None:
        result = await self.session.execute(
            select(ArborGoldStandard).where(
                ArborGoldStandard.entity_type == entity_type,
                ArborGoldStandard.source_id == source_id,
            )
        )
        return result.scalar_one_or_none()

    async def upsert(
        self, entity_type: str, source_id: int, **kwargs: Any
    ) -> ArborGoldStandard:
        existing = await self.get(entity_type, source_id)
        if existing:
            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.session.flush()
            return existing
        else:
            gs = ArborGoldStandard(
                entity_type=entity_type,
                source_id=source_id,
                **kwargs,
            )
            self.session.add(gs)
            await self.session.flush()
            return gs

    async def list_all(self, limit: int = 200) -> list[ArborGoldStandard]:
        result = await self.session.execute(
            select(ArborGoldStandard).limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_category(self, category: str, limit: int = 10) -> list[ArborGoldStandard]:
        """Find gold standard entities matching a category."""
        result = await self.session.execute(
            select(ArborGoldStandard).limit(limit)
        )
        all_gs = list(result.scalars().all())
        # Filter by matching category in the source entity
        matching = []
        for gs in all_gs:
            entity_cat = await self._get_entity_category(gs.entity_type, gs.source_id)
            if entity_cat and entity_cat.lower() == category.lower():
                matching.append(gs)
            if len(matching) >= limit:
                break
        return matching

    async def count(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(ArborGoldStandard)
        )
        return result.scalar_one()

    async def delete(self, entity_type: str, source_id: int) -> bool:
        gs = await self.get(entity_type, source_id)
        if gs:
            await self.session.delete(gs)
            await self.session.flush()
            return True
        return False

    async def _get_entity_category(self, entity_type: str, source_id: int) -> str | None:
        if entity_type == "brand":
            brand = await self.session.get(Brand, source_id)
            return brand.category if brand else None
        elif entity_type == "venue":
            venue = await self.session.get(Venue, source_id)
            return venue.category if venue else None
        return None


class GoldStandardManager:
    """Business logic for managing the gold standard reference set."""

    def __init__(self, session: AsyncSession):
        self.repo = GoldStandardRepository(session)
        self.session = session

    async def add_gold_entity(
        self,
        entity_type: str,
        source_id: int,
        scores: dict[str, int],
        tags: list[str],
        curator_notes: str = "",
        curated_by: str = "",
        fact_sheet_snapshot: dict | None = None,
    ) -> ArborGoldStandard:
        """Add or update a gold standard entity."""
        # Validate scores
        validated_scores = {}
        for dim in DimensionName.all_names():
            val = scores.get(dim, 50)
            validated_scores[dim] = max(0, min(100, int(val)))

        return await self.repo.upsert(
            entity_type=entity_type,
            source_id=source_id,
            ground_truth_scores=validated_scores,
            ground_truth_tags=tags,
            curator_notes=curator_notes,
            curated_by=curated_by,
            fact_sheet_snapshot=fact_sheet_snapshot or {},
        )

    async def get_few_shot_examples(
        self,
        category: str,
        max_examples: int = 5,
    ) -> list[GoldStandardEntity]:
        """Get gold standard examples for a category, for use as few-shot prompts.

        Strategy:
        1. First try to find same-category examples
        2. Fill remaining slots with diverse examples from other categories
        """
        same_category = await self.repo.list_by_category(category, limit=max_examples)

        entities = []
        for gs in same_category:
            name = await self._get_entity_name(gs.entity_type, gs.source_id)
            cat = await self._get_entity_category(gs.entity_type, gs.source_id)
            entities.append(
                GoldStandardEntity(
                    entity_id=f"{gs.entity_type}_{gs.source_id}",
                    entity_type=gs.entity_type,
                    source_id=gs.source_id,
                    name=name or "Unknown",
                    category=cat or category,
                    ground_truth_scores=gs.ground_truth_scores or {},
                    ground_truth_tags=gs.ground_truth_tags or [],
                    curator_notes=gs.curator_notes or "",
                    curated_by=gs.curated_by or "",
                )
            )

        # If not enough same-category, fill with others
        if len(entities) < max_examples:
            remaining = max_examples - len(entities)
            all_gs = await self.repo.list_all(limit=50)
            existing_ids = {e.entity_id for e in entities}
            for gs in all_gs:
                gs_id = f"{gs.entity_type}_{gs.source_id}"
                if gs_id not in existing_ids:
                    name = await self._get_entity_name(gs.entity_type, gs.source_id)
                    cat = await self._get_entity_category(gs.entity_type, gs.source_id)
                    entities.append(
                        GoldStandardEntity(
                            entity_id=gs_id,
                            entity_type=gs.entity_type,
                            source_id=gs.source_id,
                            name=name or "Unknown",
                            category=cat or "",
                            ground_truth_scores=gs.ground_truth_scores or {},
                            ground_truth_tags=gs.ground_truth_tags or [],
                            curator_notes=gs.curator_notes or "",
                            curated_by=gs.curated_by or "",
                        )
                    )
                    if len(entities) >= max_examples:
                        break

        return entities

    async def compute_drift(self, current_scores: dict[str, dict[str, int]]) -> dict:
        """Compare current pipeline scores against gold standard.

        Args:
            current_scores: Dict of entity_id -> {dimension: score}

        Returns:
            Drift report with per-dimension MAE and drift flags.
        """
        all_gs = await self.repo.list_all()
        if not all_gs:
            return {"gold_standard_count": 0, "needs_recalibration": False}

        dimension_errors: dict[str, list[float]] = {}
        matched = 0

        for gs in all_gs:
            gs_id = f"{gs.entity_type}_{gs.source_id}"
            if gs_id not in current_scores:
                continue
            matched += 1
            current = current_scores[gs_id]
            truth = gs.ground_truth_scores or {}

            for dim in DimensionName.all_names():
                if dim in truth and dim in current:
                    error = abs(truth[dim] - current[dim])
                    dimension_errors.setdefault(dim, []).append(error)

        per_dim_mae = {}
        drifted = []
        for dim, errors in dimension_errors.items():
            mae = sum(errors) / len(errors) if errors else 0
            per_dim_mae[dim] = round(mae, 2)
            if mae > 15:  # Threshold: 15 points avg error = drift
                drifted.append(dim)

        all_errors = [e for errs in dimension_errors.values() for e in errs]
        avg_mae = sum(all_errors) / len(all_errors) if all_errors else 0

        return {
            "gold_standard_count": len(all_gs),
            "matched_entities": matched,
            "avg_mae": round(avg_mae, 2),
            "per_dimension_mae": per_dim_mae,
            "drifted_dimensions": drifted,
            "needs_recalibration": len(drifted) > 0 or avg_mae > 12,
        }

    async def _get_entity_name(self, entity_type: str, source_id: int) -> str | None:
        if entity_type == "brand":
            brand = await self.session.get(Brand, source_id)
            return brand.name if brand else None
        elif entity_type == "venue":
            venue = await self.session.get(Venue, source_id)
            return venue.name if venue else None
        return None

    async def _get_entity_category(self, entity_type: str, source_id: int) -> str | None:
        if entity_type == "brand":
            brand = await self.session.get(Brand, source_id)
            return brand.category if brand else None
        elif entity_type == "venue":
            venue = await self.session.get(Venue, source_id)
            return venue.category if venue else None
        return None
