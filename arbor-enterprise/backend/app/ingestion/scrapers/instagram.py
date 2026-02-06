"""Instagram scraper for entity enrichment.

Uses the Instagram Basic Display / Graph API (or public profile scraping
as fallback) to extract profile data, recent posts, and follower counts
for entities that have an Instagram handle stored in the database.

NOTE: Requires a valid Instagram Graph API token for full functionality.
Without one, falls back to limited public profile metadata via oembed.
"""

import logging
import re

import httpx

from app.ingestion.scrapers.base import BaseScraper, RawEntity

logger = logging.getLogger(__name__)

# Instagram oEmbed endpoint (public, no auth required)
OEMBED_URL = "https://api.instagram.com/oembed"
# Instagram Graph API base
GRAPH_API_BASE = "https://graph.instagram.com"


class InstagramScraper(BaseScraper):
    """Scrape entity data from Instagram profiles.

    Two modes:
    1. **Graph API** (if token provided): Full profile + recent media + captions
    2. **oEmbed fallback** (no token): Basic profile name and thumbnail
    """

    def __init__(self, access_token: str | None = None):
        self.access_token = access_token
        self.timeout = 15.0

    async def scrape(self, query: str, location: str) -> list[RawEntity]:
        """Not applicable for Instagram — use scrape_profile instead."""
        return []

    async def scrape_profile(self, username: str) -> RawEntity | None:
        """Scrape an Instagram profile by username."""
        username = username.lstrip("@").strip()
        if not username:
            return None

        if self.access_token:
            return await self._scrape_via_graph_api(username)
        return await self._scrape_via_oembed(username)

    async def scrape_single(self, source_url: str) -> RawEntity | None:
        """Scrape from an Instagram URL or @handle."""
        username = self._extract_username(source_url)
        if not username:
            return None
        return await self.scrape_profile(username)

    def parse_entity(self, raw_data: dict) -> RawEntity:
        """Parse raw Instagram API data into RawEntity."""
        username = raw_data.get("username", "")
        return RawEntity(
            name=raw_data.get("name") or raw_data.get("full_name") or username,
            address=None,
            phone=None,
            website=raw_data.get("website"),
            reviews=[],
            images=raw_data.get("images", []),
            source_url=f"https://instagram.com/{username}",
            source_type="instagram",
            latitude=None,
            longitude=None,
            category=None,
            raw_data=raw_data,
        )

    # ----- Graph API (authenticated) -----

    async def _scrape_via_graph_api(self, username: str) -> RawEntity | None:
        """Use the Instagram Graph API for rich profile data."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: Get user profile
                profile_resp = await client.get(
                    f"{GRAPH_API_BASE}/me",
                    params={
                        "fields": "id,username,name,biography,profile_picture_url,"
                        "followers_count,media_count,website",
                        "access_token": self.access_token,
                    },
                )
                if profile_resp.status_code != 200:
                    logger.warning(f"Graph API profile fetch failed: {profile_resp.status_code}")
                    return await self._scrape_via_oembed(username)

                profile = profile_resp.json()

                # Step 2: Get recent media (captions as pseudo-reviews, images)
                media_resp = await client.get(
                    f"{GRAPH_API_BASE}/me/media",
                    params={
                        "fields": "id,caption,media_url,media_type,timestamp",
                        "limit": 20,
                        "access_token": self.access_token,
                    },
                )

                images = []
                captions = []
                if media_resp.status_code == 200:
                    media_data = media_resp.json().get("data", [])
                    for item in media_data:
                        if item.get("media_type") in ("IMAGE", "CAROUSEL_ALBUM"):
                            url = item.get("media_url")
                            if url:
                                images.append(url)
                        caption = item.get("caption")
                        if caption:
                            captions.append(caption)

                raw_data = {
                    **profile,
                    "images": images[:10],
                    "captions": captions,
                }

                entity = self.parse_entity(raw_data)
                entity.reviews = captions[:10]
                entity.images = images[:10]
                return entity

        except httpx.HTTPError as e:
            logger.error(f"Instagram Graph API error for {username}: {e}")
            return await self._scrape_via_oembed(username)

    # ----- oEmbed fallback (no auth) -----

    async def _scrape_via_oembed(self, username: str) -> RawEntity | None:
        """Fallback: use Instagram oEmbed for basic metadata."""
        profile_url = f"https://instagram.com/{username}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    OEMBED_URL,
                    params={"url": profile_url, "omitscript": "true"},
                )
                if resp.status_code != 200:
                    logger.warning(f"oEmbed failed for {username}: {resp.status_code}")
                    return None

                data = resp.json()
                thumbnail = data.get("thumbnail_url")

                raw_data = {
                    "username": username,
                    "name": data.get("author_name", username),
                    "full_name": data.get("author_name", username),
                    "website": None,
                    "images": [thumbnail] if thumbnail else [],
                }

                return self.parse_entity(raw_data)

        except httpx.HTTPError as e:
            logger.error(f"Instagram oEmbed error for {username}: {e}")
            return None

    # ----- Helpers -----

    @staticmethod
    def _extract_username(source: str) -> str | None:
        """Extract username from URL or @handle."""
        if not source:
            return None

        source = source.strip()

        # Handle @username format
        if source.startswith("@"):
            return source[1:]

        # Handle URL format
        match = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", source)
        if match:
            return match.group(1)

        # Treat as raw username if simple string
        if re.match(r"^[A-Za-z0-9_.]+$", source):
            return source

        return None
