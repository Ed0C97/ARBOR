"""Immutable Audit Log for compliance and governance.

TIER 13 - Point 69: Immutable Audit Log (Append-Only)

Provides cryptographic audit trail for all entity modifications.
Database user has no UPDATE/DELETE permissions on audit tables.

Features:
- Append-only logging (no updates/deletes)
- JSON diff of previous vs new state
- Actor tracking (user/system)
- Timestamp with timezone
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class AuditAction(str, Enum):
    """Types of auditable actions."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SOFT_DELETE = "soft_delete"
    RESTORE = "restore"
    ACCESS = "access"
    EXPORT = "export"
    ANONYMIZE = "anonymize"


class AuditLog(Base):
    """Immutable audit log table.

    TIER 13 - Point 69: Append-only audit trail.

    IMPORTANT: The database user for this table should only have
    INSERT and SELECT permissions. No UPDATE or DELETE.

    CREATE USER audit_writer WITH PASSWORD '...';
    GRANT INSERT, SELECT ON audit_log TO audit_writer;
    """

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # When
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Who
    actor_id = Column(String(255), nullable=False, index=True)  # user ID or "system"
    actor_type = Column(String(50), nullable=False)  # "user", "admin", "system", "api"
    actor_ip = Column(String(45), nullable=True)  # IPv4 or IPv6

    # What
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)  # "entity", "user", etc.
    resource_id = Column(String(255), nullable=False, index=True)

    # State changes
    previous_state = Column(JSONB, nullable=True)
    new_state = Column(JSONB, nullable=True)
    change_summary = Column(Text, nullable=True)  # Human-readable summary

    # Integrity
    checksum = Column(String(64), nullable=False)  # SHA-256 of content

    # Context
    request_id = Column(String(100), nullable=True)  # Correlation ID
    metadata = Column(JSONB, nullable=True)  # Additional context


@dataclass
class AuditEntry:
    """Data class for creating audit entries."""

    actor_id: str
    actor_type: str
    action: AuditAction
    resource_type: str
    resource_id: str
    previous_state: dict | None = None
    new_state: dict | None = None
    actor_ip: str | None = None
    request_id: str | None = None
    metadata: dict | None = None


class AuditLogger:
    """Service for writing audit log entries.

    TIER 13 - Point 69: Immutable audit logging.

    Usage:
        audit = AuditLogger(session)
        await audit.log(
            actor_id="user_123",
            actor_type="user",
            action=AuditAction.UPDATE,
            resource_type="entity",
            resource_id="venue_42",
            previous_state={"name": "Old Name"},
            new_state={"name": "New Name"},
        )
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(
        self,
        actor_id: str,
        actor_type: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        previous_state: dict | None = None,
        new_state: dict | None = None,
        actor_ip: str | None = None,
        request_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Write an audit log entry.

        Returns:
            The ID of the created audit log entry.
        """
        # Generate human-readable change summary
        change_summary = self._generate_summary(action, resource_type, previous_state, new_state)

        # Calculate checksum for integrity verification
        checksum = self._calculate_checksum(
            actor_id=actor_id,
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            previous_state=previous_state,
            new_state=new_state,
        )

        entry = AuditLog(
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            previous_state=previous_state,
            new_state=new_state,
            change_summary=change_summary,
            checksum=checksum,
            actor_ip=actor_ip,
            request_id=request_id,
            metadata=metadata,
        )

        self.session.add(entry)
        await self.session.flush()

        logger.info(f"Audit: {actor_id} {action.value} {resource_type}/{resource_id}")

        return entry.id

    async def log_entry(self, entry: AuditEntry) -> int:
        """Write an audit log entry from AuditEntry dataclass."""
        return await self.log(
            actor_id=entry.actor_id,
            actor_type=entry.actor_type,
            action=entry.action,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            previous_state=entry.previous_state,
            new_state=entry.new_state,
            actor_ip=entry.actor_ip,
            request_id=entry.request_id,
            metadata=entry.metadata,
        )

    def _generate_summary(
        self,
        action: AuditAction,
        resource_type: str,
        previous_state: dict | None,
        new_state: dict | None,
    ) -> str:
        """Generate human-readable change summary."""
        if action == AuditAction.CREATE:
            return f"Created {resource_type}"

        if action == AuditAction.DELETE:
            return f"Deleted {resource_type}"

        if action == AuditAction.SOFT_DELETE:
            return f"Soft-deleted {resource_type}"

        if action == AuditAction.UPDATE and previous_state and new_state:
            # Find changed fields
            changes = []
            all_keys = set(previous_state.keys()) | set(new_state.keys())

            for key in all_keys:
                old_val = previous_state.get(key)
                new_val = new_state.get(key)
                if old_val != new_val:
                    changes.append(key)

            if changes:
                return f"Updated {resource_type}: {', '.join(changes[:5])}"

        return f"{action.value.title()} {resource_type}"

    def _calculate_checksum(self, **kwargs) -> str:
        """Calculate SHA-256 checksum for integrity verification."""
        # Serialize consistently
        content = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


async def get_audit_history(
    session: AsyncSession,
    resource_type: str,
    resource_id: str,
    limit: int = 100,
) -> list[dict]:
    """Get audit history for a specific resource.

    Returns chronological list of all changes to the resource.
    """
    from sqlalchemy import select

    stmt = (
        select(AuditLog)
        .where(AuditLog.resource_type == resource_type)
        .where(AuditLog.resource_id == resource_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    entries = result.scalars().all()

    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "actor_id": e.actor_id,
            "action": e.action.value,
            "change_summary": e.change_summary,
            "previous_state": e.previous_state,
            "new_state": e.new_state,
        }
        for e in entries
    ]


async def verify_audit_integrity(session: AsyncSession, entry_id: int) -> bool:
    """Verify that an audit log entry has not been tampered with."""
    from sqlalchemy import select

    stmt = select(AuditLog).where(AuditLog.id == entry_id)
    result = await session.execute(stmt)
    entry = result.scalar_one_or_none()

    if not entry:
        return False

    # Recalculate checksum
    audit_logger = AuditLogger(session)
    expected_checksum = audit_logger._calculate_checksum(
        actor_id=entry.actor_id,
        action=entry.action.value,
        resource_type=entry.resource_type,
        resource_id=entry.resource_id,
        previous_state=entry.previous_state,
        new_state=entry.new_state,
    )

    return entry.checksum == expected_checksum
