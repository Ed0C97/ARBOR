"""Text Fact Analyzer — extracts objective facts from reviews and descriptions.

Unlike the old VibeExtractor which asked the LLM for scores directly,
this analyzer extracts *facts* (materials, price mentions, service observations)
and leaves scoring to the calibrated scoring engine in Layer 3.
"""

import json
import logging

from app.ingestion.pipeline.schemas import ExtractedFact, SourceType
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert retail/hospitality analyst. Your job is to extract
OBJECTIVE FACTS from reviews and descriptions. Do NOT assign scores or ratings.
Extract only what is directly stated or clearly implied.

Return a JSON object with these keys:
{
  "materials": ["Italian leather", "organic cotton", ...],
  "price_mentions": ["$250 for a jacket", "expensive but worth it", ...],
  "interior_elements": ["exposed brick", "marble countertops", ...],
  "service_observations": ["attentive staff", "long wait times", ...],
  "audience_indicators": ["young professionals", "tourists", ...],
  "brand_signals": ["independent boutique", "luxury chain", ...],
  "common_themes": ["quality craftsmanship", "cozy atmosphere", ...],
  "signature_items": ["handmade espresso", "truffle pasta", ...],
  "sentiment_summary": "generally positive with some complaints about pricing",
  "review_count_analyzed": 15
}

Rules:
- Only include facts explicitly mentioned in the text
- Use the exact words from reviews when possible
- Do NOT infer or hallucinate facts not present in the text
- Empty arrays are fine if no facts found for a category
"""


class TextFactAnalyzer:
    """Extract objective facts from text reviews and descriptions."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def analyze(
        self,
        reviews: list[str],
        description: str = "",
        entity_name: str = "",
        notes: str = "",
    ) -> dict:
        """Extract facts from text sources.

        Returns a dict of fact categories ready to populate a FactSheet.
        """
        if not reviews and not description:
            return self._empty_result()

        # Build the input text
        parts = []
        if description:
            parts.append(f"DESCRIPTION:\n{description}")
        if notes:
            parts.append(f"NOTES:\n{notes}")
        if reviews:
            reviews_text = "\n---\n".join(reviews[:25])
            parts.append(f"REVIEWS ({len(reviews)} total, showing up to 25):\n{reviews_text}")

        user_content = f"Entity: {entity_name}\n\n" + "\n\n".join(parts)

        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                task_type="extraction",
            )
            result = json.loads(response)
            return self._parse_result(result)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Text fact analysis failed for {entity_name}: {e}")
            return self._empty_result()

    def _parse_result(self, raw: dict) -> dict:
        """Parse LLM output into structured fact categories."""

        def to_facts(
            items: list, fact_type: str, source: SourceType = SourceType.GOOGLE_REVIEWS
        ) -> list[ExtractedFact]:
            return [
                ExtractedFact(fact_type=fact_type, value=str(v), source=source)
                for v in (items or [])
                if v
            ]

        return {
            "materials": to_facts(raw.get("materials", []), "material"),
            "price_points": to_facts(raw.get("price_mentions", []), "price_point"),
            "interior_elements": to_facts(raw.get("interior_elements", []), "interior_element"),
            "service_indicators": to_facts(
                raw.get("service_observations", []), "service_indicator"
            ),
            "audience_indicators": to_facts(
                raw.get("audience_indicators", []), "audience_indicator"
            ),
            "brand_signals": to_facts(raw.get("brand_signals", []), "brand_signal"),
            "common_themes": raw.get("common_themes", []),
            "signature_items": raw.get("signature_items", []),
            "sentiment_summary": raw.get("sentiment_summary", ""),
            "review_count_analyzed": raw.get("review_count_analyzed", 0),
        }

    def _empty_result(self) -> dict:
        return {
            "materials": [],
            "price_points": [],
            "interior_elements": [],
            "service_indicators": [],
            "audience_indicators": [],
            "brand_signals": [],
            "common_themes": [],
            "signature_items": [],
            "sentiment_summary": "",
            "review_count_analyzed": 0,
        }
