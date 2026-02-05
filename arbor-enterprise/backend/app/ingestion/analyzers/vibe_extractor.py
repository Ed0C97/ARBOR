"""Review-based Vibe DNA extraction using LLM analysis."""

import json
import logging

from app.llm.gateway import get_llm_gateway
from app.llm.prompts import load_prompt, VIBE_EXTRACTOR

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
        return {
            "dimensions": {
                "formality": 50,
                "craftsmanship": 50,
                "price_value": 50,
                "atmosphere": 50,
                "service_quality": 50,
                "exclusivity": 50,
            },
            "tags": [],
            "signature_items": [],
            "target_audience": "General",
            "summary": "",
        }

    def _default_prompt(self) -> str:
        return (
            "Analyze reviews and extract dimensional scores as JSON. "
            "Dimensions: formality, craftsmanship, price_value, atmosphere, "
            "service_quality, exclusivity (all 0-100)."
        )
