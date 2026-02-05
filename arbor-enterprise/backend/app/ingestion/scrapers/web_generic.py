"""Generic web scraper for fallback data collection."""

import logging

import httpx

from app.ingestion.scrapers.base import BaseScraper, RawEntity

logger = logging.getLogger(__name__)


class WebGenericScraper(BaseScraper):
    """Generic web scraper for pages without specific API access."""

    async def scrape(self, query: str, location: str) -> list[RawEntity]:
        """Web scraping not implemented for generic - use specific scrapers."""
        logger.warning("Generic web scraper called - no batch scraping supported")
        return []

    def parse_entity(self, raw_data: dict) -> RawEntity:
        return RawEntity(
            name=raw_data.get("name", "Unknown"),
            address=raw_data.get("address"),
            website=raw_data.get("url"),
            source_type="web",
            raw_data=raw_data,
        )

    async def scrape_single(self, source_url: str) -> RawEntity | None:
        """Scrape basic info from a web page."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(source_url, follow_redirects=True)
                if response.status_code != 200:
                    return None

                html = response.text
                # Basic extraction from HTML
                title = self._extract_title(html)

                return RawEntity(
                    name=title or "Unknown",
                    website=source_url,
                    source_url=source_url,
                    source_type="web",
                    raw_data={"html_length": len(html)},
                )
        except Exception as e:
            logger.error(f"Web scrape failed for {source_url}: {e}")
            return None

    def _extract_title(self, html: str) -> str | None:
        """Extract title from HTML."""
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
