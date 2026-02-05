"""Multi-source Data Collector — Layer 1 of the enrichment pipeline.

Gathers data from all available sources for an entity:
- Database fields (brands/venues)
- Google Maps reviews and photos
- Instagram profile and posts
- Website scraping
- Google Street View (if lat/lng available)

Produces a CollectedSources object with all raw data.
"""

import logging

from app.ingestion.pipeline.schemas import (
    CollectedSources,
    SourceData,
    SourceType,
)

logger = logging.getLogger(__name__)


class MultiSourceCollector:
    """Collect data from all available sources for an entity."""

    async def collect(
        self,
        entity_type: str,
        source_id: int,
        name: str,
        category: str,
        city: str | None = None,
        # Database fields
        description: str | None = None,
        specialty: str | None = None,
        notes: str | None = None,
        website: str | None = None,
        instagram: str | None = None,
        style: str | None = None,
        gender: str | None = None,
        rating: float | None = None,
        price_range: str | None = None,
        address: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        neighborhood: str | None = None,
        country: str | None = None,
        maps_url: str | None = None,
        is_featured: bool = False,
    ) -> CollectedSources:
        """Collect data from all available sources.

        Returns a CollectedSources with data from every accessible source.
        """
        entity_id = f"{entity_type}_{source_id}"
        collected = CollectedSources(
            entity_id=entity_id,
            entity_type=entity_type,
            source_id=source_id,
            name=name,
            category=category,
            city=city,
        )

        # Source 1: Database fields (always available)
        db_data = SourceData(
            source_type=SourceType.DATABASE,
            raw_text="\n".join(filter(None, [description, specialty, notes])),
            structured_data={
                "name": name,
                "category": category,
                "city": city,
                "country": country,
                "address": address,
                "neighborhood": neighborhood,
                "website": website,
                "instagram": instagram,
                "style": style,
                "gender": gender,
                "rating": rating,
                "price_range": price_range,
                "latitude": latitude,
                "longitude": longitude,
                "maps_url": maps_url,
                "is_featured": is_featured,
                "description": description,
                "specialty": specialty,
                "notes": notes,
            },
        )
        collected.sources.append(db_data)

        # Source 2: Google Maps reviews and photos
        if maps_url:
            google_data = await self._collect_google_maps(maps_url, name)
            if google_data:
                collected.sources.append(google_data)

        # Source 3: Instagram
        if instagram:
            ig_data = await self._collect_instagram(instagram)
            if ig_data:
                collected.sources.append(ig_data)

        # Source 4: Website
        if website:
            web_data = await self._collect_website(website)
            if web_data:
                collected.sources.append(web_data)

        return collected

    async def _collect_google_maps(self, maps_url: str, name: str) -> SourceData | None:
        """Collect reviews and photos from Google Maps."""
        try:
            from app.ingestion.scrapers.google_maps import GoogleMapsScraper

            scraper = GoogleMapsScraper()
            raw = await scraper.scrape_single(maps_url)
            if raw is None:
                return None

            return SourceData(
                source_type=SourceType.GOOGLE_REVIEWS,
                raw_text="\n---\n".join(raw.reviews),
                images=raw.images,
                structured_data={
                    "reviews": raw.reviews,
                    "review_count": len(raw.reviews),
                    "rating": raw.raw_data.get("rating"),
                    "total_ratings": raw.raw_data.get("total_ratings"),
                },
            )
        except Exception as e:
            logger.warning(f"Google Maps collection failed for {name}: {e}")
            return None

    async def _collect_instagram(self, handle: str) -> SourceData | None:
        """Collect profile data and posts from Instagram."""
        try:
            from app.ingestion.scrapers.instagram import InstagramScraper

            scraper = InstagramScraper()
            raw = await scraper.scrape_profile(handle)
            if raw is None:
                return None

            return SourceData(
                source_type=SourceType.INSTAGRAM,
                raw_text="\n---\n".join(raw.reviews),  # captions as reviews
                images=raw.images,
                structured_data={
                    "username": handle,
                    "followers_count": raw.raw_data.get("followers_count"),
                    "media_count": raw.raw_data.get("media_count"),
                    "captions": raw.reviews,
                },
            )
        except Exception as e:
            logger.warning(f"Instagram collection failed for {handle}: {e}")
            return None

    async def _collect_website(self, url: str) -> SourceData | None:
        """Collect content from the entity's website."""
        try:
            from app.ingestion.scrapers.web_generic import WebGenericScraper

            scraper = WebGenericScraper()
            raw = await scraper.scrape_single(url)
            if raw is None:
                return None

            return SourceData(
                source_type=SourceType.WEBSITE,
                raw_text=raw.raw_data.get("text", ""),
                images=raw.images,
                structured_data={
                    "title": raw.raw_data.get("title"),
                    "meta_description": raw.raw_data.get("meta_description"),
                },
            )
        except Exception as e:
            logger.warning(f"Website collection failed for {url}: {e}")
            return None
