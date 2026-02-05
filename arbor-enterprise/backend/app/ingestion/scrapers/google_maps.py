"""Google Maps/Places API scraper."""

import logging

import httpx

from app.config import get_settings
from app.ingestion.scrapers.base import BaseScraper, RawEntity

logger = logging.getLogger(__name__)
settings = get_settings()


class GoogleMapsScraper(BaseScraper):
    """Scraper using Google Places API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.google_maps_api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    async def scrape(self, query: str, location: str) -> list[RawEntity]:
        """Search for entities via Google Places text search."""
        entities = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Text search
            response = await client.get(
                f"{self.base_url}/textsearch/json",
                params={
                    "query": f"{query} in {location}",
                    "key": self.api_key,
                },
            )
            data = response.json()

            if data.get("status") != "OK":
                logger.warning(f"Places API error: {data.get('status')}")
                return []

            for result in data.get("results", []):
                try:
                    details = await self._get_details(client, result["place_id"])
                    entity = self.parse_entity(details)
                    entities.append(entity)
                except Exception as e:
                    logger.error(f"Error parsing place {result.get('name')}: {e}")

        logger.info(f"Scraped {len(entities)} entities for '{query}' in {location}")
        return entities

    async def _get_details(self, client: httpx.AsyncClient, place_id: str) -> dict:
        """Fetch detailed info for a single place."""
        response = await client.get(
            f"{self.base_url}/details/json",
            params={
                "place_id": place_id,
                "fields": (
                    "name,formatted_address,formatted_phone_number,"
                    "website,reviews,photos,opening_hours,geometry,"
                    "rating,user_ratings_total,types,url"
                ),
                "key": self.api_key,
            },
        )
        return response.json().get("result", {})

    def parse_entity(self, raw_data: dict) -> RawEntity:
        """Parse Google Places data into RawEntity."""
        geometry = raw_data.get("geometry", {}).get("location", {})

        # Extract photo references
        photos = []
        for photo in raw_data.get("photos", [])[:5]:
            ref = photo.get("photo_reference")
            if ref:
                photos.append(
                    f"{self.base_url}/photo"
                    f"?maxwidth=800&photo_reference={ref}&key={self.api_key}"
                )

        return RawEntity(
            name=raw_data.get("name", ""),
            address=raw_data.get("formatted_address"),
            phone=raw_data.get("formatted_phone_number"),
            website=raw_data.get("website"),
            reviews=[r.get("text", "") for r in raw_data.get("reviews", []) if r.get("text")],
            images=photos,
            source_url=raw_data.get("url", ""),
            source_type="google_maps",
            latitude=geometry.get("lat"),
            longitude=geometry.get("lng"),
            raw_data={
                "rating": raw_data.get("rating"),
                "total_ratings": raw_data.get("user_ratings_total"),
                "types": raw_data.get("types", []),
                "opening_hours": raw_data.get("opening_hours", {}).get("weekday_text", []),
            },
        )

    async def scrape_single(self, source_url: str) -> RawEntity | None:
        """Scrape a single place from its Google Maps URL or place_id."""
        place_id = self._extract_place_id(source_url)
        if not place_id:
            return None

        async with httpx.AsyncClient(timeout=30.0) as client:
            details = await self._get_details(client, place_id)
            if details:
                return self.parse_entity(details)
        return None

    def _extract_place_id(self, url_or_id: str) -> str | None:
        """Extract place_id from URL or return as-is if already an ID."""
        if url_or_id.startswith("ChI"):
            return url_or_id
        # Try to extract from URL parameters
        if "place_id=" in url_or_id:
            parts = url_or_id.split("place_id=")
            if len(parts) > 1:
                return parts[1].split("&")[0]
        return url_or_id
