"""Create ARBOR-owned tables (enrichments, users, feedback).

Does NOT touch the existing brands/venues tables — those are managed by
the Flask backend in the same magazine_h182 database.

Revision ID: 001
Revises: None
Create Date: 2026-02-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # arbor_enrichments — stores vibe_dna, tags, embedding refs
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS arbor_enrichments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_type VARCHAR(10) NOT NULL,
            source_id INTEGER NOT NULL,
            vibe_dna JSONB DEFAULT '{}',
            tags JSONB DEFAULT '[]',
            embedding_id VARCHAR(255),
            neo4j_synced BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_enrichment_entity UNIQUE (entity_type, source_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_enrichment_entity
        ON arbor_enrichments (entity_type, source_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_enrichment_vibe_dna
        ON arbor_enrichments USING GIN (vibe_dna)
    """)

    # ------------------------------------------------------------------
    # arbor_users — ARBOR platform users
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS arbor_users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255),
            password_hash VARCHAR(255),
            role VARCHAR(20) DEFAULT 'user',
            preferences JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT now(),
            last_login TIMESTAMPTZ
        )
    """)

    # ------------------------------------------------------------------
    # arbor_feedback — user interactions with entities
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS arbor_feedback (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES arbor_users(id) ON DELETE SET NULL,
            entity_type VARCHAR(10) NOT NULL,
            source_id INTEGER NOT NULL,
            query TEXT,
            action VARCHAR(20) NOT NULL,
            position INTEGER,
            reward FLOAT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_user
        ON arbor_feedback (user_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_entity
        ON arbor_feedback (entity_type, source_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS arbor_feedback")
    op.execute("DROP TABLE IF EXISTS arbor_users")
    op.execute("DROP TABLE IF EXISTS arbor_enrichments")
