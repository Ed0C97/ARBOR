"""Price Analyzer — extracts and normalizes pricing data from multiple sources.

Combines explicit prices from menus/websites with price mentions
in reviews and the database's price_range field to produce a
consolidated pricing picture.
"""

import json
import logging
import re

from app.ingestion.pipeline.schemas import ExtractedFact, SourceType
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)

# Price tier mapping
PRICE_TIERS = {
    "$": {"label": "$", "min": 0, "max": 25, "description": "Budget-friendly"},
    "$$": {"label": "$$", "min": 25, "max": 60, "description": "Mid-range"},
    "$$$": {"label": "$$$", "min": 60, "max": 150, "description": "Upscale"},
    "$$$$": {"label": "$$$$", "min": 150, "max": 999, "description": "Luxury"},
}


class PriceAnalyzer:
    """Extract and normalize pricing information from all available sources."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def analyze(
        self,
        price_range_db: str | None = None,
        review_price_facts: list[ExtractedFact] | None = None,
        description: str = "",
        website_text: str = "",
        entity_name: str = "",
        category: str = "",
    ) -> dict:
        """Analyze pricing from all sources.

        Returns pricing facts and a normalized price tier.
        """
        price_signals: list[str] = []

        # 1. Database price_range field
        if price_range_db:
            normalized = self._normalize_db_price(price_range_db)
            if normalized:
                price_signals.append(f"Database lists price as: {normalized}")

        # 2. Price facts from reviews
        if review_price_facts:
            for fact in review_price_facts:
                price_signals.append(f"Review mentions: {fact.value}")

        # 3. Extract prices from description
        desc_prices = self._extract_prices_from_text(description)
        for p in desc_prices:
            price_signals.append(f"Description mentions: {p}")

        # 4. Extract from website text
        web_prices = self._extract_prices_from_text(website_text)
        for p in web_prices:
            price_signals.append(f"Website mentions: {p}")

        if not price_signals:
            return self._estimate_from_category(category)

        # Use LLM to synthesize a price tier from all signals
        return await self._synthesize_pricing(price_signals, entity_name, category)

    async def _synthesize_pricing(
        self, price_signals: list[str], entity_name: str, category: str
    ) -> dict:
        """Use LLM to synthesize pricing from multiple signals."""
        signals_text = "\n".join(f"- {s}" for s in price_signals)

        prompt = f"""Given these pricing signals for "{entity_name}" ({category}):

{signals_text}

Determine the pricing information. Return JSON:
{{
  "price_tier": "$" | "$$" | "$$$" | "$$$$",
  "avg_price_estimate": <number or null>,
  "price_confidence": <0.0-1.0>,
  "price_summary": "brief description of pricing"
}}

Rules:
- $: under $25 per item/visit
- $$: $25-60 per item/visit
- $$$: $60-150 per item/visit
- $$$$: $150+ per item/visit
- If signals are contradictory, use the majority signal
- price_confidence should be lower when signals disagree
"""

        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": "You are a pricing analyst. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                task_type="extraction",
            )
            result = json.loads(response)
            return {
                "price_tier": result.get("price_tier", "$$"),
                "avg_price": result.get("avg_price_estimate"),
                "price_confidence": result.get("price_confidence", 0.5),
                "price_summary": result.get("price_summary", ""),
                "price_facts": [
                    ExtractedFact(
                        fact_type="price_tier",
                        value=result.get("price_tier", "$$"),
                        source=SourceType.DATABASE,
                        confidence=result.get("price_confidence", 0.5),
                    )
                ],
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Price synthesis failed for {entity_name}: {e}")
            return self._estimate_from_category(category)

    def _normalize_db_price(self, price_range: str) -> str | None:
        """Normalize the database price_range field."""
        if not price_range:
            return None
        pr = price_range.strip().lower()
        # Handle common formats
        if pr in ("$", "$$", "$$$", "$$$$"):
            return pr
        if "cheap" in pr or "budget" in pr or "affordable" in pr:
            return "$"
        if "expensive" in pr or "luxury" in pr or "high-end" in pr:
            return "$$$$"
        if "mid" in pr or "moderate" in pr:
            return "$$"
        return price_range

    def _extract_prices_from_text(self, text: str) -> list[str]:
        """Extract price mentions from free text."""
        if not text:
            return []
        # Match currency patterns
        patterns = [
            r'\$\d+(?:\.\d{2})?',  # $25, $25.00
            r'€\d+(?:\.\d{2})?',   # €25
            r'£\d+(?:\.\d{2})?',   # £25
            r'\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)',  # 25 USD
        ]
        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            mentions.extend(matches)
        return mentions[:10]

    def _estimate_from_category(self, category: str) -> dict:
        """Estimate price tier from category alone (low confidence)."""
        luxury_categories = {"luxury", "designer", "fine dining", "jewelry", "haute couture"}
        budget_categories = {"fast food", "thrift", "street food", "market"}
        upscale_categories = {"cocktail bar", "gallery", "spa", "wine bar"}

        cat = category.lower() if category else ""
        if cat in luxury_categories:
            tier = "$$$$"
        elif cat in upscale_categories:
            tier = "$$$"
        elif cat in budget_categories:
            tier = "$"
        else:
            tier = "$$"

        return {
            "price_tier": tier,
            "avg_price": None,
            "price_confidence": 0.2,
            "price_summary": f"Estimated from category: {category}",
            "price_facts": [
                ExtractedFact(
                    fact_type="price_tier_estimated",
                    value=tier,
                    source=SourceType.DATABASE,
                    confidence=0.2,
                )
            ],
        }
