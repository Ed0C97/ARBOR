"""Calibrated Scoring Engine — Layer 3 of the enrichment pipeline.

Converts a FactSheet into ScoredVibeDNA using:
1. Dynamic few-shot prompts from the Gold Standard
2. Structured fact-to-score mapping via LLM
3. Cross-calibration normalization
4. Per-dimension confidence scoring
"""

import json
import logging
import statistics

from app.ingestion.pipeline.schemas import (
    DimensionName,
    DimensionScore,
    FactSheet,
    GoldStandardEntity,
    ScoredVibeDNA,
)
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


def _get_scoring_system_prompt() -> str:
    """Build the scoring system prompt from the active DomainConfig.

    Uses ``DomainConfig.build_scoring_prompt()`` which auto-generates
    the prompt from VibeDimension metadata so that dimensions are never
    hardcoded.

    Falls back to a minimal generic prompt if the domain registry is
    unavailable (e.g. during Temporal activity deserialization).
    """
    try:
        from app.core.domain_portability import get_domain_registry
        domain = get_domain_registry().get_active_domain()
        return domain.build_scoring_prompt()
    except Exception:
        logger.warning(
            "Could not load scoring prompt from DomainConfig; "
            "using minimal fallback prompt"
        )
        return (
            "You are the A.R.B.O.R. Calibrated Scoring Engine.\n\n"
            "Assign Vibe DNA dimensional scores (0-100) to an entity "
            "based on its fact sheet. Output JSON with 'dimensions', "
            "'tags', 'target_audience', and 'summary'."
        )


class CalibratedScoringEngine:
    """Score entities using fact sheets and gold standard calibration."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def score(
        self,
        fact_sheet: FactSheet,
        gold_examples: list[GoldStandardEntity] | None = None,
    ) -> ScoredVibeDNA:
        """Score an entity's FactSheet into ScoredVibeDNA.

        Args:
            fact_sheet: The comprehensive fact sheet for the entity.
            gold_examples: Optional gold standard examples for few-shot calibration.

        Returns:
            ScoredVibeDNA with per-dimension scores, confidence, and metadata.
        """
        # Build the prompt with optional few-shot examples
        messages = [{"role": "system", "content": _get_scoring_system_prompt()}]

        # Add few-shot examples from gold standard
        if gold_examples:
            examples_text = self._build_few_shot_block(gold_examples)
            messages.append(
                {
                    "role": "user",
                    "content": f"Here are calibration examples from expert curators:\n\n{examples_text}\n\nUse these as reference for consistent scoring.",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "I've internalized these calibration examples. I'll score the next entity consistently with these references.",
                }
            )

        # Add the entity to score
        fact_text = fact_sheet.to_scoring_text()
        messages.append(
            {
                "role": "user",
                "content": f"Score this entity:\n\n{fact_text}",
            }
        )

        try:
            response = await self.gateway.complete_json(
                messages=messages,
                task_type="complex",
            )
            raw_result = json.loads(response)
            return self._parse_scores(fact_sheet, raw_result, gold_examples)

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Scoring failed for {fact_sheet.entity_id}: {e}")
            return self._default_scores(fact_sheet)

    async def score_batch(
        self,
        fact_sheets: list[FactSheet],
        gold_examples: list[GoldStandardEntity] | None = None,
    ) -> list[ScoredVibeDNA]:
        """Score a batch and apply cross-calibration normalization."""
        # Score each individually
        scored = []
        for fs in fact_sheets:
            result = await self.score(fs, gold_examples)
            scored.append(result)

        # Apply cross-calibration if batch is large enough
        if len(scored) >= 5:
            scored = self._cross_calibrate(scored)

        return scored

    def _build_few_shot_block(self, examples: list[GoldStandardEntity]) -> str:
        """Build few-shot example text from gold standard entities."""
        blocks = []
        for i, ex in enumerate(examples, 1):
            scores_str = ", ".join(f"{k}: {v}" for k, v in sorted(ex.ground_truth_scores.items()))
            tags_str = ", ".join(ex.ground_truth_tags[:8])
            block = (
                f"--- Example {i}: {ex.name} ({ex.category}) ---\n"
                f"Scores: {scores_str}\n"
                f"Tags: {tags_str}\n"
            )
            if ex.curator_notes:
                block += f"Curator reasoning: {ex.curator_notes}\n"
            blocks.append(block)
        return "\n".join(blocks)

    def _parse_scores(
        self,
        fact_sheet: FactSheet,
        raw: dict,
        gold_examples: list[GoldStandardEntity] | None,
    ) -> ScoredVibeDNA:
        """Parse LLM scoring output into ScoredVibeDNA."""
        dimensions = []
        raw_dims = raw.get("dimensions", {})

        for dim_name in DimensionName.all_names():
            dim_data = raw_dims.get(dim_name, {})
            if isinstance(dim_data, dict):
                score = max(0, min(100, int(dim_data.get("score", 50))))
                confidence = max(0.0, min(1.0, float(dim_data.get("confidence", 0.5))))
            elif isinstance(dim_data, (int, float)):
                score = max(0, min(100, int(dim_data)))
                confidence = 0.5
            else:
                score = 50
                confidence = 0.2

            dimensions.append(
                DimensionScore(
                    dimension=dim_name,
                    score=score,
                    confidence=confidence,
                    source_scores={"llm_calibrated": score},
                )
            )

        # Determine if review is needed
        needs_review = False
        review_reasons = []

        low_confidence_dims = [d for d in dimensions if d.confidence < 0.4]
        if len(low_confidence_dims) >= 3:
            needs_review = True
            review_reasons.append(f"Low confidence on {len(low_confidence_dims)} dimensions")

        if len(fact_sheet.sources_used) < 2:
            needs_review = True
            review_reasons.append("Only 1 data source available")

        # Overall confidence
        confidences = [d.confidence for d in dimensions]
        overall_confidence = statistics.mean(confidences) if confidences else 0.0

        return ScoredVibeDNA(
            entity_id=fact_sheet.entity_id,
            dimensions=dimensions,
            tags=raw.get("tags", [])[:15],
            signature_items=fact_sheet.signature_items,
            target_audience=raw.get("target_audience", "General"),
            visual_style=fact_sheet.visual_style,
            summary=raw.get("summary", ""),
            overall_confidence=overall_confidence,
            sources_count=len(fact_sheet.sources_used),
            needs_review=needs_review,
            review_reasons=review_reasons,
            calibrated=bool(gold_examples),
            calibration_reference_ids=[ex.entity_id for ex in (gold_examples or [])],
        )

    def _cross_calibrate(self, scored_batch: list[ScoredVibeDNA]) -> list[ScoredVibeDNA]:
        """Apply cross-calibration normalization across a batch.

        Ensures scores are distributed reasonably:
        - No dimension is systematically biased high or low
        - Relative rankings are preserved
        """
        if len(scored_batch) < 3:
            return scored_batch

        for dim_name in DimensionName.all_names():
            scores = []
            for sv in scored_batch:
                dim = sv.get_dimension(dim_name)
                if dim:
                    scores.append(dim.score)

            if len(scores) < 3:
                continue

            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0

            # If batch mean is too extreme, gently pull toward center
            if mean_score > 80 or mean_score < 20:
                target_mean = 50
                adjustment = (target_mean - mean_score) * 0.3  # Gentle pull
                for sv in scored_batch:
                    dim = sv.get_dimension(dim_name)
                    if dim:
                        dim.score = max(0, min(100, int(dim.score + adjustment)))

            # If std is too low (all same score), spread them out slightly
            if std_score < 5 and len(scores) > 3:
                for i, sv in enumerate(scored_batch):
                    dim = sv.get_dimension(dim_name)
                    if dim:
                        # Add small jitter based on position to break ties
                        jitter = (i - len(scored_batch) / 2) * 2
                        dim.score = max(0, min(100, int(dim.score + jitter)))

        return scored_batch

    def _default_scores(self, fact_sheet: FactSheet) -> ScoredVibeDNA:
        """Return default scores when scoring fails."""
        dimensions = [
            DimensionScore(
                dimension=dim,
                score=50,
                confidence=0.1,
                source_scores={},
            )
            for dim in DimensionName.all_names()
        ]
        return ScoredVibeDNA(
            entity_id=fact_sheet.entity_id,
            dimensions=dimensions,
            overall_confidence=0.1,
            needs_review=True,
            review_reasons=["Scoring failed — using defaults"],
        )
