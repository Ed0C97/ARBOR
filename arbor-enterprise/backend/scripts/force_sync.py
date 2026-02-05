
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.workflows.sync_workflow import (
    fetch_unsynced_enrichments,
    sync_entity_to_neo4j,
    sync_entity_to_qdrant,
    mark_enrichments_synced
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting manual sync...")
    
    # 1. Fetch unsynced
    logger.info("Fetching unsynced enrichments...")
    entities = await fetch_unsynced_enrichments(batch_size=100)
    logger.info(f"Found {len(entities)} entities to sync.")
    
    if not entities:
        logger.info("Nothing to sync.")
        return

    synced_ids = []
    
    for entity in entities:
        try:
            logger.info(f"Syncing {entity['name']} ({entity['composite_id']})...")
            
            # Sync to Neo4j (Critical for user visibility)
            await sync_entity_to_neo4j(entity)
            logger.info("  -> Neo4j OK")
            
            # Sync to Qdrant (Optional but good)
            try:
                await sync_entity_to_qdrant(entity)
                logger.info("  -> Qdrant OK")
            except Exception as e:
                logger.warning(f"  -> Qdrant Failed: {e}")

            synced_ids.append(entity["enrichment_id"])
            
        except Exception as e:
            logger.error(f"Failed to sync {entity['composite_id']}: {e}")

    # 2. Mark as synced
    if synced_ids:
        logger.info(f"Marking {len(synced_ids)} as synced...")
        count = await mark_enrichments_synced(synced_ids)
        logger.info(f"marked {count} rows.")

    logger.info("Sync complete.")

if __name__ == "__main__":
    asyncio.run(main())
