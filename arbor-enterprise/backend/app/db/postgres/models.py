"""SQLAlchemy ORM models for A.R.B.O.R. Enterprise.

TWO separate declarative bases for TWO separate databases:
- MagazineBase → magazine_h182 on Render (READ-ONLY: brands, venues)
- ArborBase    → arbor_db (READ-WRITE: enrichments, users, feedback, gold standard)

Brand / Venue still inherit from the shared "Base" for backwards compatibility
with existing code that does `from app.db.postgres.models import Brand`.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ═══════════════════════════════════════════════════════════════════════════
# BASE CLASSES — one per database
# ═══════════════════════════════════════════════════════════════════════════


class MagazineBase(DeclarativeBase):
    """Base for tables in magazine_h182 (read-only)."""

    pass


class ArborBase(DeclarativeBase):
    """Base for tables in arbor_db (read-write)."""

    pass


# Alias for backwards compatibility
Base = MagazineBase


# ==========================================================================
# READ-ONLY MODELS — existing tables in magazine_h182
# These are managed by the Flask backend.  ARBOR only reads from them.
# ==========================================================================


class Brand(MagazineBase):
    """Maps to the existing 'brands' table (read-only).

    Column types match the ACTUAL magazine_h182 database schema exactly.
    """

    __tablename__ = "brands"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    country: Mapped[str | None] = mapped_column(String)
    website: Mapped[str | None] = mapped_column(String)
    instagram: Mapped[str | None] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String)
    phone: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    specialty: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime)  # no timezone in DB
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)  # no timezone in DB
    created_by: Mapped[int | None] = mapped_column(Integer)  # integer in DB
    gender: Mapped[str | None] = mapped_column(String)
    style: Mapped[str | None] = mapped_column(String)
    area: Mapped[str | None] = mapped_column(String)
    neighborhood: Mapped[str | None] = mapped_column(String)
    contact_person: Mapped[str | None] = mapped_column(String)
    visited: Mapped[str | None] = mapped_column(String)  # varchar in DB, not boolean
    rating: Mapped[str | None] = mapped_column(String)  # varchar in DB, not float
    retailer: Mapped[str | None] = mapped_column(String)
    venue: Mapped[str | None] = mapped_column(String)
    # Additional columns found in actual DB
    city: Mapped[str | None] = mapped_column(String)
    region: Mapped[str | None] = mapped_column(String)
    address: Mapped[str | None] = mapped_column(String)
    maps_url: Mapped[str | None] = mapped_column(String)
    latitude: Mapped[float | None] = mapped_column(Float)  # numeric in DB
    longitude: Mapped[float | None] = mapped_column(Float)  # numeric in DB

    def __repr__(self) -> str:
        return f"<Brand id={self.id} name={self.name!r}>"


class Venue(MagazineBase):
    """Maps to the existing 'venues' table (read-only).

    Column types match the ACTUAL magazine_h182 database schema exactly.
    """

    __tablename__ = "venues"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    city: Mapped[str | None] = mapped_column(String)
    region: Mapped[str | None] = mapped_column(String)
    country: Mapped[str | None] = mapped_column(String)
    address: Mapped[str | None] = mapped_column(Text)
    maps_url: Mapped[str | None] = mapped_column(Text)
    latitude: Mapped[float | None] = mapped_column(Float)  # numeric in DB
    longitude: Mapped[float | None] = mapped_column(Float)  # numeric in DB
    website: Mapped[str | None] = mapped_column(String)
    instagram: Mapped[str | None] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String)
    phone: Mapped[str | None] = mapped_column(String)
    contact_person: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    gender: Mapped[str | None] = mapped_column(String)
    style: Mapped[str | None] = mapped_column(String)
    verified: Mapped[str | None] = mapped_column(String)  # varchar in DB, not boolean
    rating: Mapped[str | None] = mapped_column(String)  # varchar in DB, not float
    price_range: Mapped[str | None] = mapped_column(String)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime)  # no timezone in DB
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)  # no timezone in DB
    created_by: Mapped[int | None] = mapped_column(Integer)  # integer in DB
    brand: Mapped[str | None] = mapped_column(String)

    def __repr__(self) -> str:
        return f"<Venue id={self.id} name={self.name!r}>"


# ==========================================================================
# ARBOR-OWNED MODELS — tables in arbor_db (separate database)
# ==========================================================================


class ArborEnrichment(ArborBase):
    """Stores ARBOR-generated enrichment data (vibe_dna, tags, embeddings).

    Links to brands/venues via entity_type + source_id.
    """

    __tablename__ = "arbor_enrichments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to source: entity_type = "brand" | "venue", source_id = brands.id or venues.id
    entity_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Enrichment data
    vibe_dna: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    embedding_id: Mapped[str | None] = mapped_column(String(255))
    neo4j_synced: Mapped[bool] = mapped_column(Boolean, default=False)

    # Audit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("entity_type", "source_id", name="uq_enrichment_entity"),
        Index("idx_enrichment_entity", "entity_type", "source_id"),
        Index("idx_enrichment_vibe_dna", "vibe_dna", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<ArborEnrichment {self.entity_type}_{self.source_id}>"


class ArborUser(ArborBase):
    """ARBOR platform users (separate from Flask backend users)."""

    __tablename__ = "arbor_users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")
    preferences: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    feedback: Mapped[list["ArborFeedback"]] = relationship(back_populates="user")


class ArborGoldStandard(ArborBase):
    """Curator-assigned ground-truth scores for calibration entities.

    These serve as few-shot examples and calibration anchors for
    the scoring engine.
    """

    __tablename__ = "arbor_gold_standard"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to source entity
    entity_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Ground truth
    ground_truth_scores: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    ground_truth_tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    fact_sheet_snapshot: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    curator_notes: Mapped[str | None] = mapped_column(Text)
    curated_by: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("entity_type", "source_id", name="uq_gold_standard_entity"),
        Index("idx_gold_entity", "entity_type", "source_id"),
    )

    def __repr__(self) -> str:
        return f"<ArborGoldStandard {self.entity_type}_{self.source_id}>"


class ArborReviewQueue(ArborBase):
    """Enrichments flagged for curator review.

    Items are added when confidence is low, sources disagree,
    or the entity is high-priority.
    """

    __tablename__ = "arbor_review_queue"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to source entity
    entity_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Review data
    reasons: Mapped[list | None] = mapped_column(JSONB, default=list)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    scored_vibe_snapshot: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    fact_sheet_snapshot: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Review state
    status: Mapped[str] = mapped_column(String(20), default="needs_review")
    reviewer: Mapped[str | None] = mapped_column(String(255))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_notes: Mapped[str | None] = mapped_column(Text)
    overridden_scores: Mapped[dict | None] = mapped_column(JSONB)
    overridden_tags: Mapped[list | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_review_queue_status", "status"),
        Index("idx_review_queue_priority", "priority"),
        Index("idx_review_queue_entity", "entity_type", "source_id"),
    )

    def __repr__(self) -> str:
        return f"<ArborReviewQueue {self.entity_type}_{self.source_id} [{self.status}]>"


class ArborFeedback(ArborBase):
    """User feedback on discovery results.

    Links to brands/venues via entity_type + source_id (no FK to avoid
    cross-table constraints on the read-only source tables).
    """

    __tablename__ = "arbor_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("arbor_users.id", ondelete="SET NULL")
    )

    # Link to source entity
    entity_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Feedback data
    query: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # click, save, convert
    position: Mapped[int | None] = mapped_column(Integer)
    reward: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["ArborUser | None"] = relationship(back_populates="feedback")

    __table_args__ = (
        Index("idx_feedback_user", "user_id"),
        Index("idx_feedback_entity", "entity_type", "source_id"),
    )
