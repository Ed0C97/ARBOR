"""Context Analyzer — extracts location and contextual facts.

Analyzes the entity's neighborhood, city, and surroundings to provide
geographic and cultural context that influences the Vibe DNA scores.
"""

import json
import logging

from app.ingestion.pipeline.schemas import ExtractedFact, SourceType
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """Extract contextual facts from location and metadata."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def analyze(
        self,
        name: str,
        category: str,
        city: str | None = None,
        neighborhood: str | None = None,
        country: str | None = None,
        address: str | None = None,
        description: str | None = None,
        style: str | None = None,
        gender: str | None = None,
        specialty: str | None = None,
        rating: float | None = None,
        is_featured: bool = False,
    ) -> dict:
        """Analyze contextual data and return facts.

        Returns location context facts and additional metadata-derived facts.
        """
        location_facts: list[ExtractedFact] = []
        brand_facts: list[ExtractedFact] = []
        audience_facts: list[ExtractedFact] = []

        # Direct metadata facts
        if city:
            location_facts.append(
                ExtractedFact(fact_type="city", value=city, source=SourceType.DATABASE)
            )
        if neighborhood:
            location_facts.append(
                ExtractedFact(fact_type="neighborhood", value=neighborhood, source=SourceType.DATABASE)
            )
        if country:
            location_facts.append(
                ExtractedFact(fact_type="country", value=country, source=SourceType.DATABASE)
            )
        if address:
            location_facts.append(
                ExtractedFact(fact_type="address", value=address, source=SourceType.DATABASE)
            )

        if style:
            brand_facts.append(
                ExtractedFact(fact_type="style_tag", value=style, source=SourceType.DATABASE)
            )
        if gender:
            audience_facts.append(
                ExtractedFact(fact_type="gender_focus", value=gender, source=SourceType.DATABASE)
            )
        if specialty:
            brand_facts.append(
                ExtractedFact(fact_type="specialty", value=specialty, source=SourceType.DATABASE)
            )
        if rating is not None:
            brand_facts.append(
                ExtractedFact(
                    fact_type="rating",
                    value=f"{rating:.1f}/5",
                    source=SourceType.DATABASE,
                )
            )
        if is_featured:
            brand_facts.append(
                ExtractedFact(
                    fact_type="featured",
                    value="editorially featured",
                    source=SourceType.DATABASE,
                )
            )

        # Use LLM for neighborhood context enrichment if we have location
        if city or neighborhood:
            llm_context = await self._enrich_location_context(
                name, category, city, neighborhood, country
            )
            location_facts.extend(llm_context.get("location_facts", []))
            audience_facts.extend(llm_context.get("audience_facts", []))

        return {
            "location_context": location_facts,
            "brand_signals": brand_facts,
            "audience_indicators": audience_facts,
        }

    async def _enrich_location_context(
        self,
        name: str,
        category: str,
        city: str | None,
        neighborhood: str | None,
        country: str | None,
    ) -> dict:
        """Use LLM to infer neighborhood context."""
        location_str = ", ".join(filter(None, [neighborhood, city, country]))

        prompt = f"""For a {category} called "{name}" located in {location_str}:

What can we infer about the location context? Return JSON:
{{
  "neighborhood_type": "upscale" | "trendy" | "historic" | "commercial" | "residential" | "tourist",
  "foot_traffic": "high" | "medium" | "low",
  "typical_clientele": "locals" | "tourists" | "mixed" | "professionals",
  "area_character": "brief description of the area's character"
}}

Only use well-known facts about the area. If unsure, use "unknown".
"""
        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": "You are a location analyst. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                task_type="simple",
            )
            result = json.loads(response)

            location_facts = []
            audience_facts = []

            if result.get("neighborhood_type") and result["neighborhood_type"] != "unknown":
                location_facts.append(
                    ExtractedFact(
                        fact_type="neighborhood_type",
                        value=result["neighborhood_type"],
                        source=SourceType.DATABASE,
                        confidence=0.6,
                    )
                )
            if result.get("foot_traffic") and result["foot_traffic"] != "unknown":
                location_facts.append(
                    ExtractedFact(
                        fact_type="foot_traffic",
                        value=result["foot_traffic"],
                        source=SourceType.DATABASE,
                        confidence=0.5,
                    )
                )
            if result.get("typical_clientele") and result["typical_clientele"] != "unknown":
                audience_facts.append(
                    ExtractedFact(
                        fact_type="typical_clientele",
                        value=result["typical_clientele"],
                        source=SourceType.DATABASE,
                        confidence=0.5,
                    )
                )
            if result.get("area_character") and result["area_character"] != "unknown":
                location_facts.append(
                    ExtractedFact(
                        fact_type="area_character",
                        value=result["area_character"],
                        source=SourceType.DATABASE,
                        confidence=0.5,
                    )
                )

            return {"location_facts": location_facts, "audience_facts": audience_facts}

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Location context enrichment failed for {name}: {e}")
            return {"location_facts": [], "audience_facts": []}
