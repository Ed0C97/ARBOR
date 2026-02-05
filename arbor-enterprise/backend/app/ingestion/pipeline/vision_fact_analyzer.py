"""Vision Fact Analyzer — extracts objective visual facts from images.

Unlike the old VisionAnalyzer which asked for dimension scores directly,
this extracts *visual facts* (materials visible, lighting, layout, etc.)
and leaves scoring to the calibrated scoring engine.
"""

import base64
import json
import logging

import httpx

from app.ingestion.pipeline.schemas import ExtractedFact, SourceType
from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert visual analyst for retail and hospitality spaces.
Analyze the images and extract OBJECTIVE VISUAL FACTS. Do NOT assign scores.

Return a JSON object:
{
  "materials_visible": ["marble floors", "wooden shelving", "leather seats", ...],
  "lighting": "warm ambient" | "bright fluorescent" | "natural light" | "dim moody" | ...,
  "layout": "open plan" | "intimate booths" | "counter service" | "multi-room" | ...,
  "color_palette": ["earth tones", "monochrome", ...],
  "furniture_style": "mid-century modern" | "industrial" | "minimalist" | "rustic" | ...,
  "cleanliness": "immaculate" | "clean" | "lived-in" | "messy",
  "crowd_density": "empty" | "sparse" | "moderate" | "packed",
  "decor_elements": ["plants", "artwork", "neon signs", "vintage posters", ...],
  "exterior_features": ["street-facing", "courtyard", "rooftop", ...],
  "brand_presentation": "luxury boutique" | "casual shop" | "chain store" | ...,
  "visual_tags": ["minimalist", "cozy", "industrial", "bohemian", ...],
  "visual_summary": "A brief 1-2 sentence description of the visual identity"
}

Rules:
- Only describe what you can actually SEE in the images
- Use descriptive but factual language
- Do NOT guess or infer things not visible
- Empty strings/arrays are fine for categories not visible
"""


class VisionFactAnalyzer:
    """Extract objective visual facts from images."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def analyze(self, image_urls: list[str], source: SourceType = SourceType.GOOGLE_PHOTOS) -> dict:
        """Analyze images and return visual facts.

        Returns a dict of visual facts ready to merge into a FactSheet.
        """
        if not image_urls:
            return self._empty_result()

        images_b64 = await self._download_images(image_urls[:6])
        if not images_b64:
            return self._empty_result()

        content = [{"type": "text", "text": "Analyze these images of a venue or shop:"}]
        for img in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })

        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                task_type="complex",
            )
            result = json.loads(response)
            return self._parse_result(result, source)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Vision fact analysis failed: {e}")
            return self._empty_result()

    def _parse_result(self, raw: dict, source: SourceType) -> dict:
        """Parse LLM output into structured visual facts."""
        materials = [
            ExtractedFact(fact_type="material_visible", value=m, source=source)
            for m in (raw.get("materials_visible") or [])
        ]
        interior = []
        if raw.get("lighting"):
            interior.append(ExtractedFact(fact_type="lighting", value=raw["lighting"], source=source))
        if raw.get("layout"):
            interior.append(ExtractedFact(fact_type="layout", value=raw["layout"], source=source))
        if raw.get("furniture_style"):
            interior.append(ExtractedFact(fact_type="furniture_style", value=raw["furniture_style"], source=source))
        if raw.get("cleanliness"):
            interior.append(ExtractedFact(fact_type="cleanliness", value=raw["cleanliness"], source=source))
        for elem in (raw.get("decor_elements") or []):
            interior.append(ExtractedFact(fact_type="decor_element", value=elem, source=source))

        audience = []
        if raw.get("crowd_density"):
            audience.append(ExtractedFact(fact_type="crowd_density", value=raw["crowd_density"], source=source))

        brand = []
        if raw.get("brand_presentation"):
            brand.append(ExtractedFact(fact_type="brand_presentation", value=raw["brand_presentation"], source=source))

        return {
            "materials": materials,
            "interior_elements": interior,
            "audience_indicators": audience,
            "brand_signals": brand,
            "visual_tags": raw.get("visual_tags", []),
            "visual_summary": raw.get("visual_summary", ""),
            "visual_style": raw.get("furniture_style", ""),
            "color_palette": raw.get("color_palette", []),
        }

    async def _download_images(self, urls: list[str]) -> list[str]:
        """Download images and convert to base64."""
        images = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in urls:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        images.append(base64.b64encode(response.content).decode())
                except Exception as e:
                    logger.warning(f"Failed to download image {url}: {e}")
        return images

    def _empty_result(self) -> dict:
        return {
            "materials": [],
            "interior_elements": [],
            "audience_indicators": [],
            "brand_signals": [],
            "visual_tags": [],
            "visual_summary": "",
            "visual_style": "",
            "color_palette": [],
        }
