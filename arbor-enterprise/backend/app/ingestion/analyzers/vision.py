"""GPT-4o Vision analyzer for image-based Vibe extraction."""

import base64
import json
import logging

import httpx

from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analyze images to extract Vibe DNA and visual characteristics."""

    ANALYSIS_PROMPT = """Analyze these images of a venue/shop and extract:

1. DIMENSIONAL SCORES (0-100):
   - formality: 0=streetwear casual, 100=black tie formal
   - craftsmanship: 0=industrial mass production, 100=handmade artisan
   - atmosphere: 0=chaotic busy, 100=zen peaceful
   - exclusivity: 0=mainstream accessible, 100=hidden gem VIP
   - modernity: 0=vintage antique, 100=cutting edge contemporary

2. VISUAL TAGS: 5-10 English adjectives describing the aesthetic

3. STYLE CATEGORY: One primary style classification

OUTPUT FORMAT (JSON only):
{
    "dimensions": {"formality": X, "craftsmanship": X, "atmosphere": X, "exclusivity": X, "modernity": X},
    "tags": ["tag1", "tag2", ...],
    "style": "StyleName",
    "visual_summary": "One sentence describing the visual identity"
}"""

    def __init__(self):
        self.gateway = get_llm_gateway()

    async def analyze_images(self, image_urls: list[str]) -> dict:
        """Analyze images and return Vibe DNA."""
        if not image_urls:
            return self._default_result()

        # Download and encode images (max 5)
        images_b64 = await self._download_images(image_urls[:5])

        if not images_b64:
            return self._default_result()

        # Build multimodal message
        content = [{"type": "text", "text": self.ANALYSIS_PROMPT}]
        for img in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
            )

        try:
            response = await self.gateway.complete_json(
                messages=[
                    {"role": "system", "content": "You are an expert visual analyst."},
                    {"role": "user", "content": content},
                ],
                task_type="complex",
            )
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Vision analysis failed: {e}")
            return self._default_result()

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

    def _default_result(self) -> dict:
        """Return neutral defaults when analysis fails."""
        return {
            "dimensions": {
                "formality": 50,
                "craftsmanship": 50,
                "atmosphere": 50,
                "exclusivity": 50,
                "modernity": 50,
            },
            "tags": [],
            "style": "unknown",
            "visual_summary": "",
        }
