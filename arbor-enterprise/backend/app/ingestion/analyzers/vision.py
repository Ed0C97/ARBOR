"""GPT-4o Vision analyzer for image-based Vibe extraction."""

import base64
import json
import logging

import httpx

from app.llm.gateway import get_llm_gateway

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Analyze images to extract Vibe DNA and visual characteristics."""

    def __init__(self):
        self.gateway = get_llm_gateway()

    @staticmethod
    def _get_vibe_dimensions():
        """Return vibe dimensions from the active DomainConfig.

        Returns a tuple of (vibe_dimensions_list, dimension_ids_list).
        Falls back to a minimal set if the domain registry is unavailable.
        """
        try:
            from app.core.domain_portability import get_domain_registry
            domain = get_domain_registry().get_active_domain()
            return domain.vibe_dimensions, domain.dimension_ids
        except Exception:
            return None, [
                "formality", "craftsmanship", "atmosphere",
                "exclusivity", "modernity",
            ]

    def _build_analysis_prompt(self) -> str:
        """Build the vision analysis prompt from active domain dimensions."""
        vibe_dims, dim_ids = self._get_vibe_dimensions()

        if vibe_dims:
            # Rich format: use low_label / high_label from VibeDimension
            dim_lines = "\n".join(
                f"   - {d.id}: {d.low_label}, {d.high_label}"
                for d in vibe_dims
            )
            dims_json = ", ".join(f'"{d.id}": X' for d in vibe_dims)
        else:
            # Fallback: plain IDs with no scale descriptions
            dim_lines = "\n".join(f"   - {d}" for d in dim_ids)
            dims_json = ", ".join(f'"{d}": X' for d in dim_ids)

        return (
            "Analyze these images of a venue/shop and extract:\n\n"
            f"1. DIMENSIONAL SCORES (0-100):\n{dim_lines}\n\n"
            "2. VISUAL TAGS: 5-10 English adjectives describing the aesthetic\n\n"
            "3. STYLE CATEGORY: One primary style classification\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "{\n"
            f'    "dimensions": {{{dims_json}}},\n'
            '    "tags": ["tag1", "tag2", ...],\n'
            '    "style": "StyleName",\n'
            '    "visual_summary": "One sentence describing the visual identity"\n'
            "}"
        )

    async def analyze_images(self, image_urls: list[str]) -> dict:
        """Analyze images and return Vibe DNA."""
        if not image_urls:
            return self._default_result()

        # Download and encode images (max 5)
        images_b64 = await self._download_images(image_urls[:5])

        if not images_b64:
            return self._default_result()

        # Build multimodal message
        content = [{"type": "text", "text": self._build_analysis_prompt()}]
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
        _, dim_ids = self._get_vibe_dimensions()
        return {
            "dimensions": {d: 50 for d in dim_ids},
            "tags": [],
            "style": "unknown",
            "visual_summary": "",
        }
