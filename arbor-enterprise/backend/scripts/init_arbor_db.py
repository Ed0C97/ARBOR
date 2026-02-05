"""Initialize the ARBOR database — creates all arbor_* tables.

Usage:
    cd backend
    python -m scripts.init_arbor_db

This script:
1. Connects to the arbor_db (configured via ARBOR_DATABASE_URL in .env)
2. Creates all ARBOR-owned tables:
   - arbor_enrichments
   - arbor_users
   - arbor_gold_standard
   - arbor_review_queue
   - arbor_feedback
3. Does NOT touch magazine_h182 (brands/venues stay on Render)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    from app.config import get_settings
    from app.db.postgres.connection import create_arbor_tables, arbor_engine

    settings = get_settings()

    print("=" * 60)
    print("A.R.B.O.R. Enterprise - Database Initialization")
    print("=" * 60)
    print()
    print(f"  ARBOR DB URL: {settings.arbor_database_url.split('@')[1] if '@' in settings.arbor_database_url else settings.arbor_database_url}")
    print(f"  SSL:          {settings.arbor_database_ssl}")
    print()

    try:
        print("Creating ARBOR tables...")
        await create_arbor_tables()
        print()
        print("  [OK] arbor_enrichments")
        print("  [OK] arbor_users")
        print("  [OK] arbor_gold_standard")
        print("  [OK] arbor_review_queue")
        print("  [OK] arbor_feedback")
        print()
        print("All tables created successfully!")
        print()
        print("Magazine DB (brands/venues) was NOT modified.")
        print()
    except Exception as e:
        print(f"\n  [FAIL] ERROR: {e}\n")
        print("Make sure:")
        print("  1. Docker is running: docker-compose up -d postgres-arbor")
        print("  2. ARBOR_DATABASE_URL in .env is correct")
        print("  3. The database 'arbor_db' exists")
        sys.exit(1)
    finally:
        await arbor_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
