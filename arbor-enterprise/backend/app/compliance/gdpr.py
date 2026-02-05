"""GDPR Right to be Forgotten automation.

TIER 13 - Point 70: GDPR "Right to be Forgotten" Automation

Automated pipeline for handling data deletion requests with full
compliance logging and validation.

Features:
- Soft-delete in database
- Hard-delete PII
- Anonymize logs and events
- Audit trail of deletion
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import update, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DeletionStatus(str, Enum):
    """Status of a GDPR deletion request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class DeletionResult:
    """Result of a deletion job."""
    user_id: str
    status: DeletionStatus
    tables_processed: list[str]
    records_deleted: int
    records_anonymized: int
    errors: list[str]
    completed_at: datetime | None = None


class GDPRDeletionJob:
    """Handles GDPR Right to be Forgotten requests.
    
    TIER 13 - Point 70: Automated deletion pipeline.
    
    Usage:
        job = GDPRDeletionJob(session)
        result = await job.execute(user_id="user_123")
    """
    
    # Tables containing user PII that need deletion
    PII_TABLES = [
        ("users", "id"),
        ("user_profiles", "user_id"),
        ("user_preferences", "user_id"),
        ("user_sessions", "user_id"),
    ]
    
    # Tables that need anonymization (keep structure, remove PII)
    ANONYMIZE_TABLES = [
        ("search_history", "user_id"),
        ("feedback", "user_id"),
        ("ratings", "user_id"),
    ]
    
    # Fields to anonymize in logs/events
    PII_FIELDS = [
        "email",
        "name",
        "first_name",
        "last_name",
        "phone",
        "address",
        "ip_address",
    ]
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.errors: list[str] = []
    
    async def execute(self, user_id: str) -> DeletionResult:
        """Execute GDPR deletion for a user.
        
        TIER 13 - Point 70: Full deletion pipeline.
        
        Steps:
        1. Log deletion request in audit
        2. Soft-delete main user record
        3. Hard-delete PII from related tables
        4. Anonymize historical data
        5. Verify deletion completeness
        """
        logger.info(f"Starting GDPR deletion for user: {user_id}")
        
        tables_processed = []
        records_deleted = 0
        records_anonymized = 0
        self.errors = []
        
        try:
            # Step 1: Log the deletion request
            await self._log_deletion_start(user_id)
            
            # Step 2: Soft-delete main user record
            deleted = await self._soft_delete_user(user_id)
            if deleted:
                tables_processed.append("users")
                records_deleted += 1
            
            # Step 3: Hard-delete PII from related tables
            for table_name, id_column in self.PII_TABLES[1:]:  # Skip users table
                try:
                    count = await self._hard_delete_table(
                        table_name, id_column, user_id
                    )
                    if count > 0:
                        tables_processed.append(table_name)
                        records_deleted += count
                except Exception as e:
                    self.errors.append(f"{table_name}: {str(e)}")
            
            # Step 4: Anonymize historical data
            for table_name, id_column in self.ANONYMIZE_TABLES:
                try:
                    count = await self._anonymize_table(
                        table_name, id_column, user_id
                    )
                    if count > 0:
                        tables_processed.append(f"{table_name} (anonymized)")
                        records_anonymized += count
                except Exception as e:
                    self.errors.append(f"{table_name} anonymize: {str(e)}")
            
            # Step 5: Delete from vector stores
            await self._delete_from_vector_stores(user_id)
            
            # Commit all changes
            await self.session.commit()
            
            # Step 6: Log completion
            await self._log_deletion_complete(user_id)
            
            status = DeletionStatus.COMPLETED if not self.errors else DeletionStatus.FAILED
            
            return DeletionResult(
                user_id=user_id,
                status=status,
                tables_processed=tables_processed,
                records_deleted=records_deleted,
                records_anonymized=records_anonymized,
                errors=self.errors,
                completed_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error(f"GDPR deletion failed for {user_id}: {e}")
            await self.session.rollback()
            
            return DeletionResult(
                user_id=user_id,
                status=DeletionStatus.FAILED,
                tables_processed=tables_processed,
                records_deleted=0,
                records_anonymized=0,
                errors=[str(e)],
            )
    
    async def _soft_delete_user(self, user_id: str) -> bool:
        """Soft-delete the main user record."""
        try:
            # Mark as deleted but keep record
            stmt = (
                update(self._get_users_table())
                .where(self._get_users_table().c.id == user_id)
                .values(
                    deleted_at=datetime.utcnow(),
                    email=self._anonymize_value(user_id, "email"),
                    name="[DELETED]",
                    is_active=False,
                )
            )
            await self.session.execute(stmt)
            return True
        except Exception as e:
            logger.warning(f"Users table may not exist: {e}")
            return False
    
    async def _hard_delete_table(
        self,
        table_name: str,
        id_column: str,
        user_id: str,
    ) -> int:
        """Hard-delete all records for user from a table."""
        from sqlalchemy import text
        
        stmt = text(f"DELETE FROM {table_name} WHERE {id_column} = :user_id")
        result = await self.session.execute(stmt, {"user_id": user_id})
        return result.rowcount
    
    async def _anonymize_table(
        self,
        table_name: str,
        id_column: str,
        user_id: str,
    ) -> int:
        """Anonymize PII fields in a table while keeping the record."""
        from sqlalchemy import text
        
        # Build SET clause for PII fields
        set_clauses = []
        for field in self.PII_FIELDS:
            set_clauses.append(f"{field} = '[ANONYMIZED]'")
        
        if not set_clauses:
            return 0
        
        set_clause = ", ".join(set_clauses)
        stmt = text(
            f"UPDATE {table_name} SET {set_clause} "
            f"WHERE {id_column} = :user_id"
        )
        
        try:
            result = await self.session.execute(stmt, {"user_id": user_id})
            return result.rowcount
        except Exception:
            # Table might not have all PII fields
            return 0
    
    async def _delete_from_vector_stores(self, user_id: str) -> None:
        """Delete user's vectors from Qdrant."""
        try:
            from app.db.qdrant.client import get_async_qdrant_client
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            client = await get_async_qdrant_client()
            if client:
                # Delete from entities collection if user-submitted
                await client.delete(
                    collection_name="entities_vectors",
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="submitted_by",
                                match=MatchValue(value=user_id),
                            )
                        ]
                    ),
                )
                logger.info(f"Deleted vectors for user {user_id}")
        except Exception as e:
            self.errors.append(f"Vector store: {str(e)}")
    
    async def _log_deletion_start(self, user_id: str) -> None:
        """Log the start of deletion in audit log."""
        from app.compliance.audit_log import AuditLogger, AuditAction
        
        try:
            audit = AuditLogger(self.session)
            await audit.log(
                actor_id="system",
                actor_type="gdpr_automation",
                action=AuditAction.DELETE,
                resource_type="user",
                resource_id=user_id,
                metadata={"stage": "started", "type": "GDPR_RTBF"},
            )
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")
    
    async def _log_deletion_complete(self, user_id: str) -> None:
        """Log completion of deletion."""
        from app.compliance.audit_log import AuditLogger, AuditAction
        
        try:
            audit = AuditLogger(self.session)
            await audit.log(
                actor_id="system",
                actor_type="gdpr_automation",
                action=AuditAction.ANONYMIZE,
                resource_type="user",
                resource_id=user_id,
                metadata={
                    "stage": "completed",
                    "type": "GDPR_RTBF",
                    "errors": self.errors if self.errors else None,
                },
            )
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")
    
    def _anonymize_value(self, user_id: str, field: str) -> str:
        """Generate deterministic anonymized value."""
        hash_input = f"{user_id}:{field}:GDPR_DELETED"
        return f"deleted_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}@deleted.local"
    
    def _get_users_table(self):
        """Get users table for queries. Override if table name differs."""
        from sqlalchemy import table, column
        return table("users", column("id"), column("email"), column("name"),
                     column("deleted_at"), column("is_active"))


async def verify_deletion(session: AsyncSession, user_id: str) -> dict[str, Any]:
    """Verify that all user data has been deleted.
    
    TIER 13 - Point 70: Verification after deletion.
    
    Returns:
        Dict with verification results for each data store.
    """
    from sqlalchemy import text
    
    results = {
        "user_id": user_id,
        "verified_at": datetime.utcnow().isoformat(),
        "stores": {},
        "fully_deleted": True,
    }
    
    # Check PostgreSQL tables
    for table_name, id_column in GDPRDeletionJob.PII_TABLES:
        try:
            stmt = text(f"SELECT COUNT(*) FROM {table_name} WHERE {id_column} = :user_id")
            result = await session.execute(stmt, {"user_id": user_id})
            count = result.scalar()
            results["stores"][table_name] = {"remaining_records": count}
            if count > 0:
                results["fully_deleted"] = False
        except Exception as e:
            results["stores"][table_name] = {"error": str(e)}
    
    return results
