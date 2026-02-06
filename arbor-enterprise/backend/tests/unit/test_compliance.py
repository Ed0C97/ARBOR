"""Unit tests for the compliance module — audit log, data retention, and GDPR."""

import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.compliance.audit_log import (
    AuditAction,
    AuditEntry,
    AuditLog,
    AuditLogger,
    get_audit_history,
    verify_audit_integrity,
)
from app.compliance.data_retention import (
    CleanupResult,
    DataRetentionEnforcer,
    RetentionPeriod,
    RetentionPolicy,
    DEFAULT_POLICIES,
    cleanup_redis_cache,
    cleanup_qdrant_cache,
)
from app.compliance.gdpr import (
    DeletionResult,
    DeletionStatus,
    GDPRDeletionJob,
    verify_deletion,
)


# ==========================================================================
# Audit Log Tests
# ==========================================================================


class TestAuditAction:
    """Tests for the AuditAction enum."""

    def test_action_values(self):
        assert AuditAction.CREATE.value == "create"
        assert AuditAction.UPDATE.value == "update"
        assert AuditAction.DELETE.value == "delete"
        assert AuditAction.SOFT_DELETE.value == "soft_delete"
        assert AuditAction.RESTORE.value == "restore"
        assert AuditAction.ACCESS.value == "access"
        assert AuditAction.EXPORT.value == "export"
        assert AuditAction.ANONYMIZE.value == "anonymize"

    def test_action_is_string_enum(self):
        assert isinstance(AuditAction.CREATE, str)
        assert AuditAction.CREATE == "create"


class TestAuditEntry:
    """Tests for the AuditEntry dataclass."""

    def test_required_fields(self):
        entry = AuditEntry(
            actor_id="user_1",
            actor_type="user",
            action=AuditAction.CREATE,
            resource_type="entity",
            resource_id="venue_42",
        )
        assert entry.actor_id == "user_1"
        assert entry.actor_type == "user"
        assert entry.action == AuditAction.CREATE
        assert entry.resource_type == "entity"
        assert entry.resource_id == "venue_42"

    def test_optional_fields_default_to_none(self):
        entry = AuditEntry(
            actor_id="sys",
            actor_type="system",
            action=AuditAction.DELETE,
            resource_type="user",
            resource_id="u_1",
        )
        assert entry.previous_state is None
        assert entry.new_state is None
        assert entry.actor_ip is None
        assert entry.request_id is None
        assert entry.metadata is None

    def test_optional_fields_set(self):
        entry = AuditEntry(
            actor_id="admin_1",
            actor_type="admin",
            action=AuditAction.UPDATE,
            resource_type="entity",
            resource_id="venue_10",
            previous_state={"name": "Old"},
            new_state={"name": "New"},
            actor_ip="192.168.1.1",
            request_id="req-abc",
            metadata={"source": "api"},
        )
        assert entry.previous_state == {"name": "Old"}
        assert entry.new_state == {"name": "New"}
        assert entry.actor_ip == "192.168.1.1"
        assert entry.request_id == "req-abc"
        assert entry.metadata == {"source": "api"}


class TestAuditLogger:
    """Tests for the AuditLogger service."""

    def setup_method(self):
        self.session = AsyncMock()
        self.logger = AuditLogger(self.session)

    async def test_log_creates_entry_and_flushes(self):
        mock_entry = MagicMock()
        mock_entry.id = 42
        # Make add capture the argument so we can set id
        def capture_add(obj):
            obj.id = 42
        self.session.add = MagicMock(side_effect=capture_add)

        entry_id = await self.logger.log(
            actor_id="user_1",
            actor_type="user",
            action=AuditAction.CREATE,
            resource_type="entity",
            resource_id="venue_1",
        )
        self.session.add.assert_called_once()
        self.session.flush.assert_awaited_once()
        assert entry_id == 42

    async def test_log_with_state_changes(self):
        def capture_add(obj):
            obj.id = 10
        self.session.add = MagicMock(side_effect=capture_add)

        await self.logger.log(
            actor_id="admin",
            actor_type="admin",
            action=AuditAction.UPDATE,
            resource_type="entity",
            resource_id="venue_5",
            previous_state={"name": "Old Name"},
            new_state={"name": "New Name"},
        )

        added_obj = self.session.add.call_args[0][0]
        assert isinstance(added_obj, AuditLog)
        assert added_obj.action == AuditAction.UPDATE
        assert added_obj.previous_state == {"name": "Old Name"}
        assert added_obj.new_state == {"name": "New Name"}

    async def test_log_entry_delegates_to_log(self):
        def capture_add(obj):
            obj.id = 99
        self.session.add = MagicMock(side_effect=capture_add)

        entry = AuditEntry(
            actor_id="sys",
            actor_type="system",
            action=AuditAction.EXPORT,
            resource_type="user",
            resource_id="u_7",
            metadata={"format": "csv"},
        )
        entry_id = await self.logger.log_entry(entry)
        assert entry_id == 99

        added_obj = self.session.add.call_args[0][0]
        assert added_obj.actor_id == "sys"
        assert added_obj.action == AuditAction.EXPORT
        assert added_obj.metadata == {"format": "csv"}

    def test_generate_summary_create(self):
        summary = self.logger._generate_summary(AuditAction.CREATE, "entity", None, None)
        assert summary == "Created entity"

    def test_generate_summary_delete(self):
        summary = self.logger._generate_summary(AuditAction.DELETE, "user", None, None)
        assert summary == "Deleted user"

    def test_generate_summary_soft_delete(self):
        summary = self.logger._generate_summary(AuditAction.SOFT_DELETE, "user", None, None)
        assert summary == "Soft-deleted user"

    def test_generate_summary_update_with_changes(self):
        summary = self.logger._generate_summary(
            AuditAction.UPDATE,
            "entity",
            {"name": "Old", "address": "123 St"},
            {"name": "New", "address": "123 St"},
        )
        assert "Updated entity" in summary
        assert "name" in summary

    def test_generate_summary_update_no_states(self):
        summary = self.logger._generate_summary(AuditAction.UPDATE, "entity", None, None)
        assert "Update" in summary

    def test_generate_summary_access_fallback(self):
        summary = self.logger._generate_summary(AuditAction.ACCESS, "entity", None, None)
        assert "Access" in summary
        assert "entity" in summary

    def test_calculate_checksum_deterministic(self):
        cs1 = self.logger._calculate_checksum(actor_id="u1", action="create")
        cs2 = self.logger._calculate_checksum(actor_id="u1", action="create")
        assert cs1 == cs2

    def test_calculate_checksum_differs_for_different_inputs(self):
        cs1 = self.logger._calculate_checksum(actor_id="u1", action="create")
        cs2 = self.logger._calculate_checksum(actor_id="u2", action="create")
        assert cs1 != cs2

    def test_calculate_checksum_is_sha256(self):
        cs = self.logger._calculate_checksum(actor_id="x", action="y")
        assert len(cs) == 64  # SHA-256 hex string length


class TestGetAuditHistory:
    """Tests for the get_audit_history function."""

    async def test_returns_formatted_entries(self):
        session = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.id = 1
        mock_entry.timestamp = datetime(2024, 1, 15, 12, 0, 0)
        mock_entry.actor_id = "user_1"
        mock_entry.action = AuditAction.CREATE
        mock_entry.change_summary = "Created entity"
        mock_entry.previous_state = None
        mock_entry.new_state = {"name": "Test"}

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_entry]
        session.execute = AsyncMock(return_value=mock_result)

        history = await get_audit_history(session, "entity", "venue_1")
        assert len(history) == 1
        assert history[0]["id"] == 1
        assert history[0]["actor_id"] == "user_1"
        assert history[0]["action"] == "create"
        assert history[0]["new_state"] == {"name": "Test"}

    async def test_empty_history(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        history = await get_audit_history(session, "entity", "nonexistent_id")
        assert history == []


class TestVerifyAuditIntegrity:
    """Tests for the verify_audit_integrity function."""

    async def test_valid_entry_returns_true(self):
        session = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.actor_id = "user_1"
        mock_entry.action = AuditAction.CREATE
        mock_entry.resource_type = "entity"
        mock_entry.resource_id = "venue_1"
        mock_entry.previous_state = None
        mock_entry.new_state = {"name": "Test"}

        # Calculate the expected checksum
        content = json.dumps({
            "actor_id": "user_1",
            "action": "create",
            "resource_type": "entity",
            "resource_id": "venue_1",
            "previous_state": None,
            "new_state": {"name": "Test"},
        }, sort_keys=True, default=str)
        mock_entry.checksum = hashlib.sha256(content.encode()).hexdigest()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_entry
        session.execute = AsyncMock(return_value=mock_result)

        is_valid = await verify_audit_integrity(session, 1)
        assert is_valid is True

    async def test_tampered_entry_returns_false(self):
        session = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.actor_id = "user_1"
        mock_entry.action = AuditAction.CREATE
        mock_entry.resource_type = "entity"
        mock_entry.resource_id = "venue_1"
        mock_entry.previous_state = None
        mock_entry.new_state = {"name": "Test"}
        mock_entry.checksum = "tampered_checksum_that_wont_match"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_entry
        session.execute = AsyncMock(return_value=mock_result)

        is_valid = await verify_audit_integrity(session, 1)
        assert is_valid is False

    async def test_missing_entry_returns_false(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        is_valid = await verify_audit_integrity(session, 999)
        assert is_valid is False


# ==========================================================================
# Data Retention Tests
# ==========================================================================


class TestRetentionPeriod:
    """Tests for the RetentionPeriod enum."""

    def test_standard_periods(self):
        assert RetentionPeriod.DAYS_7.value == 7
        assert RetentionPeriod.DAYS_30.value == 30
        assert RetentionPeriod.DAYS_90.value == 90
        assert RetentionPeriod.DAYS_365.value == 365
        assert RetentionPeriod.DAYS_730.value == 730


class TestRetentionPolicy:
    """Tests for the RetentionPolicy dataclass."""

    def test_create_policy(self):
        policy = RetentionPolicy(
            name="test_logs",
            table_name="test_table",
            timestamp_column="created_at",
            retention_days=30,
            description="Test logs older than 30 days",
        )
        assert policy.name == "test_logs"
        assert policy.table_name == "test_table"
        assert policy.retention_days == 30
        assert policy.soft_delete is False

    def test_soft_delete_policy(self):
        policy = RetentionPolicy(
            name="soft",
            table_name="soft_table",
            timestamp_column="ts",
            retention_days=90,
            description="Soft-delete policy",
            soft_delete=True,
        )
        assert policy.soft_delete is True


class TestDefaultPolicies:
    """Tests for the default retention policies."""

    def test_default_policies_exist(self):
        assert len(DEFAULT_POLICIES) == 6

    def test_default_policy_names(self):
        names = {p.name for p in DEFAULT_POLICIES}
        expected = {
            "search_logs",
            "api_access_logs",
            "session_data",
            "temporary_uploads",
            "semantic_cache",
            "dlq_messages",
        }
        assert names == expected

    def test_session_data_retention_is_7_days(self):
        session_policy = next(p for p in DEFAULT_POLICIES if p.name == "session_data")
        assert session_policy.retention_days == 7

    def test_search_logs_retention_is_90_days(self):
        search_policy = next(p for p in DEFAULT_POLICIES if p.name == "search_logs")
        assert search_policy.retention_days == 90


class TestDataRetentionEnforcer:
    """Tests for the DataRetentionEnforcer class."""

    def setup_method(self):
        self.session = AsyncMock()

    async def test_init_uses_default_policies(self):
        enforcer = DataRetentionEnforcer(self.session)
        assert enforcer.policies == DEFAULT_POLICIES

    async def test_init_with_custom_policies(self):
        custom = [
            RetentionPolicy(
                name="custom",
                table_name="custom_table",
                timestamp_column="ts",
                retention_days=14,
                description="Custom policy",
            )
        ]
        enforcer = DataRetentionEnforcer(self.session, policies=custom)
        assert len(enforcer.policies) == 1
        assert enforcer.policies[0].name == "custom"

    async def test_run_cleanup_dry_run(self):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        self.session.execute = AsyncMock(return_value=mock_result)

        policies = [
            RetentionPolicy(
                name="test",
                table_name="test_table",
                timestamp_column="created_at",
                retention_days=30,
                description="Test",
            )
        ]
        enforcer = DataRetentionEnforcer(self.session, policies=policies)
        results = await enforcer.run_cleanup(dry_run=True)

        assert len(results) == 1
        assert results[0].policy_name == "test"
        assert results[0].records_deleted == 5
        # Dry run should not call commit
        self.session.commit.assert_not_awaited()

    async def test_run_cleanup_handles_exceptions(self):
        self.session.execute = AsyncMock(side_effect=Exception("DB error"))

        policies = [
            RetentionPolicy(
                name="fail_test",
                table_name="bad_table",
                timestamp_column="ts",
                retention_days=7,
                description="Will fail",
            )
        ]
        enforcer = DataRetentionEnforcer(self.session, policies=policies)
        results = await enforcer.run_cleanup()

        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "DB error" in results[0].errors[0]

    async def test_add_policy(self):
        enforcer = DataRetentionEnforcer(self.session, policies=[])
        new_policy = RetentionPolicy(
            name="new_policy",
            table_name="new_table",
            timestamp_column="ts",
            retention_days=60,
            description="New policy",
        )
        await enforcer.add_policy(new_policy)
        assert len(enforcer.policies) == 1
        assert enforcer.policies[0].name == "new_policy"

    async def test_get_retention_stats(self):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        self.session.execute = AsyncMock(return_value=mock_result)

        policies = [
            RetentionPolicy(
                name="stats_test",
                table_name="stats_table",
                timestamp_column="ts",
                retention_days=30,
                description="Stats test",
            )
        ]
        enforcer = DataRetentionEnforcer(self.session, policies=policies)
        stats = await enforcer.get_retention_stats()

        assert "timestamp" in stats
        assert len(stats["policies"]) == 1
        assert stats["policies"][0]["name"] == "stats_test"
        assert stats["policies"][0]["pending_deletion"] == 42

    async def test_get_retention_stats_handles_error(self):
        self.session.execute = AsyncMock(side_effect=Exception("Table not found"))

        policies = [
            RetentionPolicy(
                name="err_test",
                table_name="missing_table",
                timestamp_column="ts",
                retention_days=30,
                description="Error test",
            )
        ]
        enforcer = DataRetentionEnforcer(self.session, policies=policies)
        stats = await enforcer.get_retention_stats()

        assert "error" in stats["policies"][0]


class TestCleanupRedisCache:
    """Tests for the cleanup_redis_cache function."""

    @patch("app.compliance.data_retention.get_redis_client")
    async def test_cleanup_deletes_keys_without_ttl(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        # Simulate scan returning keys then finishing
        mock_client.scan = AsyncMock(
            side_effect=[(0, [b"llm_cache:key1", b"llm_cache:key2"])]
        )
        mock_client.ttl = AsyncMock(side_effect=[-1, 3600])
        mock_client.delete = AsyncMock()

        deleted = await cleanup_redis_cache()
        assert deleted == 1  # Only key1 had ttl == -1
        mock_client.delete.assert_awaited_once()

    @patch("app.compliance.data_retention.get_redis_client")
    async def test_cleanup_returns_zero_when_no_client(self, mock_get_client):
        mock_get_client.return_value = None
        deleted = await cleanup_redis_cache()
        assert deleted == 0


class TestCleanupQdrantCache:
    """Tests for the cleanup_qdrant_cache function."""

    @patch("app.compliance.data_retention.get_async_qdrant_client")
    async def test_cleanup_succeeds(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.delete = AsyncMock()

        result = await cleanup_qdrant_cache()
        assert result == 1
        mock_client.delete.assert_awaited_once()

    @patch("app.compliance.data_retention.get_async_qdrant_client")
    async def test_cleanup_returns_zero_on_error(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.delete = AsyncMock(side_effect=Exception("connection failed"))

        result = await cleanup_qdrant_cache()
        assert result == 0

    @patch("app.compliance.data_retention.get_async_qdrant_client")
    async def test_cleanup_returns_zero_when_no_client(self, mock_get_client):
        mock_get_client.return_value = None
        result = await cleanup_qdrant_cache()
        assert result == 0


# ==========================================================================
# GDPR Tests
# ==========================================================================


class TestDeletionStatus:
    """Tests for the DeletionStatus enum."""

    def test_status_values(self):
        assert DeletionStatus.PENDING.value == "pending"
        assert DeletionStatus.IN_PROGRESS.value == "in_progress"
        assert DeletionStatus.COMPLETED.value == "completed"
        assert DeletionStatus.FAILED.value == "failed"
        assert DeletionStatus.VERIFIED.value == "verified"


class TestDeletionResult:
    """Tests for the DeletionResult dataclass."""

    def test_create_result(self):
        result = DeletionResult(
            user_id="user_1",
            status=DeletionStatus.COMPLETED,
            tables_processed=["users", "user_profiles"],
            records_deleted=5,
            records_anonymized=3,
            errors=[],
            completed_at=datetime(2024, 6, 15),
        )
        assert result.user_id == "user_1"
        assert result.status == DeletionStatus.COMPLETED
        assert len(result.tables_processed) == 2
        assert result.records_deleted == 5
        assert result.records_anonymized == 3
        assert result.errors == []

    def test_default_completed_at_is_none(self):
        result = DeletionResult(
            user_id="u",
            status=DeletionStatus.PENDING,
            tables_processed=[],
            records_deleted=0,
            records_anonymized=0,
            errors=[],
        )
        assert result.completed_at is None


class TestGDPRDeletionJob:
    """Tests for the GDPRDeletionJob class."""

    def setup_method(self):
        self.session = AsyncMock()

    def test_pii_tables_defined(self):
        assert len(GDPRDeletionJob.PII_TABLES) >= 4
        table_names = [t[0] for t in GDPRDeletionJob.PII_TABLES]
        assert "users" in table_names
        assert "user_profiles" in table_names

    def test_anonymize_tables_defined(self):
        assert len(GDPRDeletionJob.ANONYMIZE_TABLES) >= 3
        table_names = [t[0] for t in GDPRDeletionJob.ANONYMIZE_TABLES]
        assert "search_history" in table_names

    def test_pii_fields_include_common_fields(self):
        assert "email" in GDPRDeletionJob.PII_FIELDS
        assert "name" in GDPRDeletionJob.PII_FIELDS
        assert "phone" in GDPRDeletionJob.PII_FIELDS
        assert "ip_address" in GDPRDeletionJob.PII_FIELDS

    def test_anonymize_value_is_deterministic(self):
        job = GDPRDeletionJob(self.session)
        val1 = job._anonymize_value("user_1", "email")
        val2 = job._anonymize_value("user_1", "email")
        assert val1 == val2

    def test_anonymize_value_differs_by_user(self):
        job = GDPRDeletionJob(self.session)
        val1 = job._anonymize_value("user_1", "email")
        val2 = job._anonymize_value("user_2", "email")
        assert val1 != val2

    def test_anonymize_value_format(self):
        job = GDPRDeletionJob(self.session)
        val = job._anonymize_value("user_1", "email")
        assert val.startswith("deleted_")
        assert val.endswith("@deleted.local")

    @patch("app.compliance.gdpr.AuditLogger")
    async def test_execute_successful_deletion(self, mock_audit_cls):
        mock_audit = AsyncMock()
        mock_audit_cls.return_value = mock_audit

        # Mock all internal methods
        job = GDPRDeletionJob(self.session)
        job._log_deletion_start = AsyncMock()
        job._soft_delete_user = AsyncMock(return_value=True)
        job._hard_delete_table = AsyncMock(return_value=2)
        job._anonymize_table = AsyncMock(return_value=3)
        job._delete_from_vector_stores = AsyncMock()
        job._log_deletion_complete = AsyncMock()

        result = await job.execute("user_123")

        assert result.user_id == "user_123"
        assert result.status == DeletionStatus.COMPLETED
        assert result.records_deleted > 0
        assert result.records_anonymized > 0
        assert result.completed_at is not None
        self.session.commit.assert_awaited_once()

    @patch("app.compliance.gdpr.AuditLogger")
    async def test_execute_rollback_on_exception(self, mock_audit_cls):
        mock_audit = AsyncMock()
        mock_audit_cls.return_value = mock_audit

        job = GDPRDeletionJob(self.session)
        job._log_deletion_start = AsyncMock(side_effect=Exception("fatal error"))

        result = await job.execute("user_fail")

        assert result.status == DeletionStatus.FAILED
        assert result.records_deleted == 0
        assert "fatal error" in result.errors[0]
        self.session.rollback.assert_awaited_once()


class TestVerifyDeletion:
    """Tests for the verify_deletion function."""

    async def test_verify_returns_structure(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)

        result = await verify_deletion(session, "user_1")
        assert result["user_id"] == "user_1"
        assert "verified_at" in result
        assert "fully_deleted" in result
        assert "stores" in result

    async def test_verify_detects_remaining_data(self):
        session = AsyncMock()
        # First call returns count > 0, rest return 0
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            mock_r = MagicMock()
            if call_count == 0:
                mock_r.scalar.return_value = 3
            else:
                mock_r.scalar.return_value = 0
            call_count += 1
            return mock_r
        session.execute = AsyncMock(side_effect=side_effect)

        result = await verify_deletion(session, "user_1")
        assert result["fully_deleted"] is False
