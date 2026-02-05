"""Abstract base scraper for data ingestion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RawEntity:
    """Raw entity data from scraping."""

    name: str
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    reviews: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    source_url: str = ""
    source_type: str = ""
    latitude: float | None = None
    longitude: float | None = None
    category: str | None = None
    raw_data: dict = field(default_factory=dict)


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    @abstractmethod
    async def scrape(self, query: str, location: str) -> list[RawEntity]:
        """Scrape entities matching query in location."""
        ...

    @abstractmethod
    def parse_entity(self, raw_data: dict) -> RawEntity:
        """Parse raw API data into a RawEntity."""
        ...

    async def scrape_single(self, source_url: str) -> RawEntity | None:
        """Scrape a single entity from its URL. Override in subclasses."""
        return None
