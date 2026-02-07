"""Check detailed enrichment status."""

import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.postgres.connection import magazine_session_factory, arbor_session_factory
from app.db.postgres.models import ArborEnrichment
from app.config import get_settings
from app.db.postgres.dynamic_model import create_dynamic_model
from app.db.postgres.connection import magazine_engine
from sqlalchemy import select, func

async def main():
    async with magazine_session_factory() as main_session, arbor_session_factory() as arbor_session:
        from sqlalchemy import MetaData
        settings = get_settings()
        configs = settings.get_entity_type_configs()
        _metadata = MetaData()

        by_type = {}
        for config in configs:
            table = await create_dynamic_model(config, magazine_engine, _metadata)
            count = await main_session.scalar(select(func.count()).select_from(table))
            by_type[config.entity_type] = count

        total_entities = sum(by_type.values())

        # Count enrichments
        enr_count = await arbor_session.scalar(select(func.count()).select_from(ArborEnrichment))

        # Count synced to neo4j
        neo4j_synced = await arbor_session.scalar(
            select(func.count()).select_from(ArborEnrichment).where(ArborEnrichment.neo4j_synced == True)
        )

        print("\n=== ENRICHMENT STATUS ===")
        for etype, count in by_type.items():
            enr_count_for_type = await arbor_session.scalar(
                select(func.count()).select_from(ArborEnrichment)
                .where(ArborEnrichment.entity_type == etype)
            )
            pct = enr_count_for_type / count * 100 if count else 0
            print(f"  {etype}: {enr_count_for_type} / {count} ({pct:.1f}%)")
        print(f"Total Entities: {total_entities}")
        print(f"Total Enrichments: {enr_count}")
        print()
        print(f"Synced to Neo4j: {neo4j_synced} / {enr_count} ({neo4j_synced/enr_count*100 if enr_count else 0:.1f}%)")
        print(f"Need Sync: {enr_count - neo4j_synced}")

if __name__ == "__main__":
    asyncio.run(main())
