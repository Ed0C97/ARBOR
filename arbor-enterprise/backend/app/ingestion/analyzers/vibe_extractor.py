"""Review-based Vibe DNA extraction using LLM analysis."""

import json
import logging

from app.llm.gateway import get_llm_gateway
from app.llm.prompts import VIBE_EXTRACTOR, load_prompt

logger = logging.getLogger(__name__)


class VibeExtractor:
    """Extract Vibe DNA from textual reviews."""

    def __init__(self):
        self.gateway = get_llm_gateway()
        try:
            self.system_prompt = load_prompt(VIBE_EXTRACTOR)
        except FileNotFoundError:
            self.system_prompt = self._default_prompt()

    async def extract_vibe(self, reviews: list[str], entity_name: str) -> dict:
        """Extract Vibe DNA from a list of reviews."""
        if not reviews:
            return self._default_result()

        reviews_text = "\n---\n".join(reviews[:20])

        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Entity: {entity_name}\n\nReviews:\n{reviews_text}",
                    },
                ],
                task_type="extraction",
            )
            result = json.loads(response)
            return self._validate_result(result)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Vibe extraction failed for {entity_name}: {e}")
            return self._default_result()

    def _validate_result(self, result: dict) -> dict:
        """Validate and normalize the extracted vibe data."""
        dimensions = result.get("dimensions", {})
        for key, value in dimensions.items():
            dimensions[key] = max(0, min(100, int(value)))

        return {
            "dimensions": dimensions,
            "tags": result.get("tags", [])[:15],
            "signature_items": result.get("signature_items", [])[:5],
            "target_audience": result.get("target_audience", "General"),
            "summary": result.get("summary", ""),
        }

    def _default_result(self) -> dict:
        dim_ids = self._get_dimension_ids()
        return {
            "dimensions": {d: 50 for d in dim_ids},
            "tags": [],
            "signature_items": [],
            "target_audience": "General",
            "summary": "",
        }

    @staticmethod
    def _get_dimension_ids() -> list[str]:
        """Return dimension IDs from the active DomainConfig.

        Falls back to a minimal set if the domain registry is
        unavailable (e.g. during Temporal activity serialization).
        """
        try:
            from app.core.domain_portability import get_domain_registry
            domain = get_domain_registry().get_active_domain()
            return domain.dimension_ids
        except Exception:
            return [
                "formality", "craftsmanship", "price_value",
                "atmosphere", "service_quality", "exclusivity",
            ]

    def _default_prompt(self) -> str:
        dim_ids = self._get_dimension_ids()
        dims_str = ", ".join(dim_ids)
        return (
            "Analyze reviews and extract dimensional scores as JSON. "
            f"Dimensions: {dims_str} (all 0-100)."
        )
