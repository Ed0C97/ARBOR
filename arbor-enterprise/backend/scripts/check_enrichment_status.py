"""Check detailed enrichment status."""

import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.models import Brand, Venue, ArborEnrichment
from sqlalchemy import select, func

async def main():
    async with magazine_session_factory() as main_session, arbor_session_factory() as arbor_session:
        # Count brands and venues
        brand_count = await main_session.scalar(select(func.count()).select_from(Brand))
        venue_count = await main_session.scalar(select(func.count()).select_from(Venue))

        # Count enrichments
        enr_count = await arbor_session.scalar(select(func.count()).select_from(ArborEnrichment))

        # Count by type
        brand_enr = await arbor_session.scalar(
            select(func.count()).select_from(ArborEnrichment).where(ArborEnrichment.entity_type == "brand")
        )
        venue_enr = await arbor_session.scalar(
            select(func.count()).select_from(ArborEnrichment).where(ArborEnrichment.entity_type == "venue")
        )

        # Count synced to neo4j
        neo4j_synced = await arbor_session.scalar(
            select(func.count()).select_from(ArborEnrichment).where(ArborEnrichment.neo4j_synced == True)
        )

        print("\n=== ENRICHMENT STATUS ===")
        print(f"Total Brands in DB: {brand_count}")
        print(f"Total Venues in DB: {venue_count}")
        print(f"Total Entities: {brand_count + venue_count}")
        print()
        print(f"Brand Enrichments: {brand_enr} / {brand_count} ({brand_enr/brand_count*100 if brand_count else 0:.1f}%)")
        print(f"Venue Enrichments: {venue_enr} / {venue_count} ({venue_enr/venue_count*100 if venue_count else 0:.1f}%)")
        print(f"Total Enrichments: {enr_count}")
        print()
        print(f"Synced to Neo4j: {neo4j_synced} / {enr_count} ({neo4j_synced/enr_count*100 if enr_count else 0:.1f}%)")
        print(f"Need Sync: {enr_count - neo4j_synced}")

if __name__ == "__main__":
    asyncio.run(main())
