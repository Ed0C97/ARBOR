"""Unit tests for Webhook Event System, Agentic Workflows, and GraphQL Schema."""

import pytest
import hmac
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

from app.events.webhooks import (
    WebhookEvent,
    WebhookSubscription,
    WebhookManager,
    verify_webhook_signature,
    _compute_signature,
)
from app.agents.agentic_workflows import (
    WorkflowType,
    QueryDecomposer,
    DeepResearchAgent,
    EntityComparisonAgent,
)


# ===========================================================================
# Webhook Event Types
# ===========================================================================


class TestWebhookEvent:
    """Tests for the WebhookEvent enum."""

    def test_event_types_exist(self):
        """All 6 webhook event types are accessible as enum members."""
        expected_names = [
            "ENTITY_CREATED",
            "ENTITY_UPDATED",
            "ENTITY_ENRICHED",
            "SEARCH_PERFORMED",
            "ANOMALY_DETECTED",
            "ENRICHMENT_COMPLETED",
        ]
        for name in expected_names:
            assert hasattr(WebhookEvent, name), f"WebhookEvent.{name} should exist"

    def test_event_values_are_strings(self):
        """Every WebhookEvent value is a dotted string (e.g. 'entity.created')."""
        for event in WebhookEvent:
            assert isinstance(event.value, str), f"{event.name} value should be a string"
            assert "." in event.value, (
                f"{event.name} value '{event.value}' should be a dotted string"
            )


# ===========================================================================
# Webhook Subscription Dataclass
# ===========================================================================


class TestWebhookSubscription:
    """Tests for the WebhookSubscription dataclass."""

    def test_subscription_creation(self):
        """A subscription can be created with the correct fields."""
        sub = WebhookSubscription(
            id="sub-1",
            tenant_id="tenant-a",
            url="https://example.com/hook",
            events=[WebhookEvent.ENTITY_CREATED],
            secret="s3cret",
        )
        assert sub.id == "sub-1"
        assert sub.tenant_id == "tenant-a"
        assert sub.url == "https://example.com/hook"
        assert sub.events == [WebhookEvent.ENTITY_CREATED]
        assert sub.secret == "s3cret"

    def test_default_max_retries(self):
        """max_retries defaults to 3."""
        sub = WebhookSubscription(
            id="sub-2",
            tenant_id="tenant-b",
            url="https://example.com/hook",
            events=[WebhookEvent.ENTITY_UPDATED],
            secret="key",
        )
        assert sub.max_retries == 3

    def test_default_active(self):
        """is_active defaults to True."""
        sub = WebhookSubscription(
            id="sub-3",
            tenant_id="tenant-c",
            url="https://example.com/hook",
            events=[WebhookEvent.ANOMALY_DETECTED],
            secret="key",
        )
        assert sub.is_active is True


# ===========================================================================
# Webhook Signature Verification
# ===========================================================================


class TestWebhookSignature:
    """Tests for HMAC-SHA256 webhook signature verification."""

    def test_verify_valid_signature(self):
        """A correctly computed HMAC signature is accepted."""
        payload = b'{"event": "entity.created"}'
        secret = "webhook-secret"
        signature = hmac.new(
            key=secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        assert verify_webhook_signature(payload, signature, secret) is True

    def test_verify_invalid_signature(self):
        """A wrong signature is rejected."""
        payload = b'{"event": "entity.created"}'
        secret = "webhook-secret"
        assert verify_webhook_signature(payload, "badsignature", secret) is False

    def test_verify_different_secret(self):
        """A signature computed with a different secret is rejected."""
        payload = b'{"event": "entity.created"}'
        sig_with_secret_a = hmac.new(
            key=b"secret-a",
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        assert verify_webhook_signature(payload, sig_with_secret_a, "secret-b") is False


# ===========================================================================
# Webhook Manager
# ===========================================================================


class TestWebhookManager:
    """Tests for the WebhookManager registry and dispatch."""

    def setup_method(self):
        self.manager = WebhookManager()

    def test_register_subscription(self):
        """register() returns a subscription with a generated id."""
        sub = self.manager.register(
            tenant_id="t1",
            url="https://hook.example.com/a",
            events=[WebhookEvent.ENTITY_CREATED],
            secret="sec",
        )
        assert sub.id is not None
        assert len(sub.id) > 0
        assert sub.tenant_id == "t1"
        assert sub.url == "https://hook.example.com/a"

    def test_unregister_subscription(self):
        """unregister() returns True on first call, False on second."""
        sub = self.manager.register(
            tenant_id="t1",
            url="https://hook.example.com/b",
            events=[WebhookEvent.ENTITY_UPDATED],
            secret="sec",
        )
        assert self.manager.unregister(sub.id) is True
        assert self.manager.unregister(sub.id) is False

    def test_list_subscriptions_by_tenant(self):
        """list_subscriptions() returns only the matching tenant's subscriptions."""
        self.manager.register(
            tenant_id="alpha",
            url="https://hook.example.com/1",
            events=[WebhookEvent.ENTITY_CREATED],
            secret="s1",
        )
        self.manager.register(
            tenant_id="beta",
            url="https://hook.example.com/2",
            events=[WebhookEvent.ENTITY_UPDATED],
            secret="s2",
        )
        self.manager.register(
            tenant_id="alpha",
            url="https://hook.example.com/3",
            events=[WebhookEvent.ANOMALY_DETECTED],
            secret="s3",
        )

        alpha_subs = self.manager.list_subscriptions("alpha")
        assert len(alpha_subs) == 2
        assert all(s.tenant_id == "alpha" for s in alpha_subs)

        beta_subs = self.manager.list_subscriptions("beta")
        assert len(beta_subs) == 1
        assert beta_subs[0].tenant_id == "beta"

    @patch("app.events.webhooks.asyncio.create_task")
    def test_dispatch_creates_tasks(self, mock_create_task):
        """dispatch() creates an asyncio task for each matching subscription."""
        self.manager.register(
            tenant_id="t1",
            url="https://hook.example.com/x",
            events=[WebhookEvent.ENTITY_CREATED, WebhookEvent.ENTITY_UPDATED],
            secret="sec",
        )
        self.manager.register(
            tenant_id="t2",
            url="https://hook.example.com/y",
            events=[WebhookEvent.ENTITY_CREATED],
            secret="sec2",
        )

        self.manager.dispatch(
            WebhookEvent.ENTITY_CREATED,
            {"entity_id": "e1"},
        )

        assert mock_create_task.call_count == 2

    def test_delivery_stats(self):
        """get_delivery_stats() returns a dict with expected keys."""
        sub = self.manager.register(
            tenant_id="t1",
            url="https://hook.example.com/z",
            events=[WebhookEvent.ENTITY_CREATED],
            secret="sec",
        )
        stats = self.manager.get_delivery_stats(sub.id)
        expected_keys = {
            "subscription_id",
            "is_active",
            "failure_count",
            "last_success_at",
            "last_failure_at",
            "registered_events",
        }
        assert expected_keys.issubset(stats.keys()), (
            f"Missing keys: {expected_keys - stats.keys()}"
        )


# ===========================================================================
# Workflow Type Enum
# ===========================================================================


class TestWorkflowType:
    """Tests for the WorkflowType enum."""

    def test_workflow_types_exist(self):
        """All 5 workflow types are accessible as enum members."""
        expected_names = [
            "DEEP_RESEARCH",
            "TREND_ANALYSIS",
            "ENTITY_COMPARISON",
            "MARKET_MAPPING",
            "STYLE_EXPLORATION",
        ]
        for name in expected_names:
            assert hasattr(WorkflowType, name), f"WorkflowType.{name} should exist"

    def test_workflow_type_values(self):
        """Every WorkflowType value is a lowercase string."""
        for wt in WorkflowType:
            assert isinstance(wt.value, str), f"{wt.name} value should be a string"
            assert wt.value == wt.value.lower(), (
                f"{wt.name} value '{wt.value}' should be lowercase"
            )


# ===========================================================================
# Query Decomposer
# ===========================================================================


class TestQueryDecomposer:
    """Tests for QueryDecomposer."""

    def setup_method(self):
        self.decomposer = QueryDecomposer()

    def test_decompose_deep_research(self):
        """DEEP_RESEARCH decomposition produces multiple sub-queries."""
        result = self.decomposer.decompose(
            "luxury fashion brands in Paris",
            WorkflowType.DEEP_RESEARCH,
        )
        assert len(result) > 1, "Deep research should produce multiple sub-queries"

    def test_decompose_entity_comparison(self):
        """A comparison query containing 'vs' is split on the separator."""
        result = self.decomposer.decompose(
            "Gucci vs Prada",
            WorkflowType.ENTITY_COMPARISON,
        )
        assert len(result) >= 2, "Comparison should split on 'vs'"
        # The individual entity names should appear in the sub-queries
        combined = " ".join(result).lower()
        assert "gucci" in combined
        assert "prada" in combined

    def test_decompose_trend_analysis(self):
        """TREND_ANALYSIS decomposition produces trend-specific sub-queries."""
        result = self.decomposer.decompose(
            "streetwear trends",
            WorkflowType.TREND_ANALYSIS,
        )
        assert len(result) > 1, "Trend analysis should produce multiple sub-queries"
        combined = " ".join(result).lower()
        assert "trending" in combined or "emerging" in combined or "popular" in combined

    def test_decompose_returns_list(self):
        """decompose() always returns a list of strings."""
        for wt in WorkflowType:
            result = self.decomposer.decompose("test query", wt)
            assert isinstance(result, list), f"Result for {wt.name} should be a list"
            for item in result:
                assert isinstance(item, str), (
                    f"Each sub-query for {wt.name} should be a string"
                )


# ===========================================================================
# Deep Research Agent
# ===========================================================================


class TestDeepResearchAgent:
    """Tests for DeepResearchAgent._should_continue logic."""

    def setup_method(self):
        self.agent = DeepResearchAgent(default_max_iterations=3)

    def test_should_continue_false_when_complete(self):
        """_should_continue returns False when is_complete is True."""
        state = {
            "is_complete": True,
            "iteration": 0,
            "max_iterations": 3,
            "sub_queries": ["q1"],
        }
        assert self.agent._should_continue(state) is False

    def test_should_continue_false_at_max_iterations(self):
        """_should_continue returns False when iteration reaches max_iterations."""
        state = {
            "is_complete": False,
            "iteration": 3,
            "max_iterations": 3,
            "sub_queries": ["q1"],
        }
        assert self.agent._should_continue(state) is False


# ===========================================================================
# Entity Comparison Agent
# ===========================================================================


class TestEntityComparisonAgent:
    """Tests for EntityComparisonAgent._build_comparison_matrix."""

    def setup_method(self):
        self.agent = EntityComparisonAgent()

    def test_build_comparison_matrix_empty(self):
        """An empty entity list produces an empty matrix."""
        matrix = self.agent._build_comparison_matrix([])
        assert matrix["entities"] == []
        assert matrix["dimensions"] is not None
        for key in matrix["dimensions"]:
            assert matrix["dimensions"][key] == []

    def test_build_comparison_matrix_single(self):
        """A single entity produces a matrix with one entry per dimension."""
        entities = [
            {
                "name": "TestBrand",
                "id": "b1",
                "category": "Fashion",
                "city": "Milan",
                "price_range": "$$$$",
                "vibe_dna": {"minimalist": 0.8},
                "tags": ["luxury", "italian"],
            }
        ]
        matrix = self.agent._build_comparison_matrix(entities)
        assert len(matrix["entities"]) == 1
        assert matrix["entities"][0]["name"] == "TestBrand"
        assert len(matrix["dimensions"]["category"]) == 1
        assert matrix["dimensions"]["category"][0] == "Fashion"
