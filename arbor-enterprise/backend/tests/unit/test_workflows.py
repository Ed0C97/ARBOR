"""Unit tests for Temporal.io workflows and activities."""

import inspect
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio import activity, workflow

from app.workflows.activities import (
    scrape_entity,
    analyze_with_vision,
    extract_vibe,
    generate_embedding,
    save_to_qdrant,
    save_to_neo4j,
)
from app.workflows.enrichment_activities import (
    run_full_enrichment,
    collect_entity_data,
    run_fact_analyzers,
    score_with_calibration,
    validate_and_persist,
    get_unenriched_entities,
    generate_and_sync_embedding,
    _entity_to_kwargs,
    _compute_priority,
)
from app.workflows.ingestion_workflow import (
    EntityIngestionWorkflow,
    BatchIngestionWorkflow,
)
from app.workflows.enrichment_workflow import (
    EntityEnrichmentWorkflow,
    SteppedEnrichmentWorkflow,
    BatchEnrichmentWorkflow,
)


# ==========================================================================
# Activities — Existence and Signatures
# ==========================================================================


class TestActivityDefinitions:
    """Tests that all activities are properly decorated and have correct signatures."""

    def test_scrape_entity_is_activity(self):
        # temporalio marks activities; check it is async and has the right params
        assert inspect.iscoroutinefunction(scrape_entity)

    def test_scrape_entity_signature(self):
        sig = inspect.signature(scrape_entity)
        params = list(sig.parameters.keys())
        assert "source_url" in params
        assert "scraper_type" in params

    def test_scrape_entity_default_scraper_type(self):
        sig = inspect.signature(scrape_entity)
        assert sig.parameters["scraper_type"].default == "google_maps"

    def test_analyze_with_vision_is_async(self):
        assert inspect.iscoroutinefunction(analyze_with_vision)

    def test_analyze_with_vision_signature(self):
        sig = inspect.signature(analyze_with_vision)
        params = list(sig.parameters.keys())
        assert "images" in params

    def test_extract_vibe_is_async(self):
        assert inspect.iscoroutinefunction(extract_vibe)

    def test_extract_vibe_signature(self):
        sig = inspect.signature(extract_vibe)
        params = list(sig.parameters.keys())
        assert "reviews" in params
        assert "entity_name" in params

    def test_generate_embedding_is_async(self):
        assert inspect.iscoroutinefunction(generate_embedding)

    def test_generate_embedding_signature(self):
        sig = inspect.signature(generate_embedding)
        params = list(sig.parameters.keys())
        assert "text" in params

    def test_save_to_qdrant_is_async(self):
        assert inspect.iscoroutinefunction(save_to_qdrant)

    def test_save_to_qdrant_signature(self):
        sig = inspect.signature(save_to_qdrant)
        params = list(sig.parameters.keys())
        assert "entity_id" in params
        assert "vector" in params
        assert "payload" in params

    def test_save_to_neo4j_is_async(self):
        assert inspect.iscoroutinefunction(save_to_neo4j)

    def test_save_to_neo4j_signature(self):
        sig = inspect.signature(save_to_neo4j)
        params = list(sig.parameters.keys())
        assert "entity_id" in params
        assert "name" in params
        assert "category" in params
        assert "city" in params


# ==========================================================================
# Enrichment Activities — Existence and Signatures
# ==========================================================================


class TestEnrichmentActivityDefinitions:
    """Tests that enrichment activities are properly defined."""

    def test_run_full_enrichment_is_async(self):
        assert inspect.iscoroutinefunction(run_full_enrichment)

    def test_run_full_enrichment_signature(self):
        sig = inspect.signature(run_full_enrichment)
        params = list(sig.parameters.keys())
        assert "entity_type" in params
        assert "source_id" in params

    def test_collect_entity_data_is_async(self):
        assert inspect.iscoroutinefunction(collect_entity_data)

    def test_run_fact_analyzers_is_async(self):
        assert inspect.iscoroutinefunction(run_fact_analyzers)

    def test_score_with_calibration_is_async(self):
        assert inspect.iscoroutinefunction(score_with_calibration)

    def test_score_with_calibration_signature(self):
        sig = inspect.signature(score_with_calibration)
        params = list(sig.parameters.keys())
        assert "fact_sheet_data" in params
        assert "entity_type" in params
        assert "source_id" in params

    def test_validate_and_persist_is_async(self):
        assert inspect.iscoroutinefunction(validate_and_persist)

    def test_validate_and_persist_signature(self):
        sig = inspect.signature(validate_and_persist)
        params = list(sig.parameters.keys())
        assert "scored_data" in params
        assert "fact_sheet_data" in params
        assert "entity_type" in params
        assert "source_id" in params

    def test_get_unenriched_entities_is_async(self):
        assert inspect.iscoroutinefunction(get_unenriched_entities)

    def test_get_unenriched_entities_defaults(self):
        sig = inspect.signature(get_unenriched_entities)
        assert sig.parameters["max_entities"].default == 100

    def test_generate_and_sync_embedding_is_async(self):
        assert inspect.iscoroutinefunction(generate_and_sync_embedding)


# ==========================================================================
# Ingestion Workflow Definitions
# ==========================================================================


class TestEntityIngestionWorkflow:
    """Tests for the EntityIngestionWorkflow class."""

    def test_workflow_class_exists(self):
        assert EntityIngestionWorkflow is not None

    def test_workflow_has_run_method(self):
        assert hasattr(EntityIngestionWorkflow, "run")

    def test_run_method_is_async(self):
        assert inspect.iscoroutinefunction(EntityIngestionWorkflow.run)

    def test_run_signature(self):
        sig = inspect.signature(EntityIngestionWorkflow.run)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "source_url" in params
        assert "category" in params

    def test_run_default_category_is_empty(self):
        sig = inspect.signature(EntityIngestionWorkflow.run)
        assert sig.parameters["category"].default == ""


class TestBatchIngestionWorkflow:
    """Tests for the BatchIngestionWorkflow class."""

    def test_workflow_class_exists(self):
        assert BatchIngestionWorkflow is not None

    def test_workflow_has_run_method(self):
        assert hasattr(BatchIngestionWorkflow, "run")

    def test_run_method_is_async(self):
        assert inspect.iscoroutinefunction(BatchIngestionWorkflow.run)

    def test_run_signature(self):
        sig = inspect.signature(BatchIngestionWorkflow.run)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "source_urls" in params
        assert "category" in params

    def test_run_returns_list_annotation(self):
        sig = inspect.signature(BatchIngestionWorkflow.run)
        assert sig.return_annotation == list[str]


# ==========================================================================
# Enrichment Workflow Definitions
# ==========================================================================


class TestEntityEnrichmentWorkflow:
    """Tests for the EntityEnrichmentWorkflow class."""

    def test_workflow_class_exists(self):
        assert EntityEnrichmentWorkflow is not None

    def test_workflow_has_run_method(self):
        assert hasattr(EntityEnrichmentWorkflow, "run")

    def test_run_method_is_async(self):
        assert inspect.iscoroutinefunction(EntityEnrichmentWorkflow.run)

    def test_run_signature(self):
        sig = inspect.signature(EntityEnrichmentWorkflow.run)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "entity_type" in params
        assert "source_id" in params

    def test_run_returns_dict_annotation(self):
        sig = inspect.signature(EntityEnrichmentWorkflow.run)
        assert sig.return_annotation == dict


class TestSteppedEnrichmentWorkflow:
    """Tests for the SteppedEnrichmentWorkflow class."""

    def test_workflow_class_exists(self):
        assert SteppedEnrichmentWorkflow is not None

    def test_workflow_has_run_method(self):
        assert hasattr(SteppedEnrichmentWorkflow, "run")

    def test_run_method_is_async(self):
        assert inspect.iscoroutinefunction(SteppedEnrichmentWorkflow.run)

    def test_run_signature(self):
        sig = inspect.signature(SteppedEnrichmentWorkflow.run)
        params = list(sig.parameters.keys())
        assert "entity_type" in params
        assert "source_id" in params


class TestBatchEnrichmentWorkflow:
    """Tests for the BatchEnrichmentWorkflow class."""

    def test_workflow_class_exists(self):
        assert BatchEnrichmentWorkflow is not None

    def test_workflow_has_run_method(self):
        assert hasattr(BatchEnrichmentWorkflow, "run")

    def test_run_method_is_async(self):
        assert inspect.iscoroutinefunction(BatchEnrichmentWorkflow.run)

    def test_run_signature(self):
        sig = inspect.signature(BatchEnrichmentWorkflow.run)
        params = list(sig.parameters.keys())
        assert "max_entities" in params
        assert "entity_type" in params

    def test_run_max_entities_default(self):
        sig = inspect.signature(BatchEnrichmentWorkflow.run)
        assert sig.parameters["max_entities"].default == 100


# ==========================================================================
# Helper Function Tests
# ==========================================================================


class TestEntityToKwargs:
    """Tests for the _entity_to_kwargs helper."""

    def test_brand_kwargs(self):
        entity = MagicMock()
        entity.name = "TestBrand"
        entity.category = "fashion"
        entity.description = "A test brand"
        entity.specialty = "shoes"
        entity.notes = "Popular"
        entity.website = "https://test.com"
        entity.instagram = "@test"
        entity.style = "modern"
        entity.gender = "unisex"
        entity.rating = 4.5
        entity.is_featured = True
        entity.country = "Italy"
        entity.area = "Milan"
        entity.neighborhood = "Brera"

        result = _entity_to_kwargs(entity, "brand", 1)

        assert result["entity_type"] == "brand"
        assert result["source_id"] == 1
        assert result["name"] == "TestBrand"
        assert result["category"] == "fashion"
        assert result["city"] == "Milan"  # area mapped to city for brands
        assert result["neighborhood"] == "Brera"

    def test_venue_kwargs(self):
        entity = MagicMock()
        entity.name = "TestVenue"
        entity.category = "restaurant"
        entity.description = "A test venue"
        entity.specialty = None
        entity.notes = None
        entity.website = None
        entity.instagram = None
        entity.style = None
        entity.gender = None
        entity.rating = None
        entity.is_featured = False
        entity.country = "Italy"
        entity.city = "Rome"
        entity.address = "Via Roma 1"
        entity.latitude = 41.9
        entity.longitude = 12.5
        entity.region = "Centro"
        entity.maps_url = "https://maps.google.com/test"
        entity.price_range = "$$$"

        result = _entity_to_kwargs(entity, "venue", 2)

        assert result["entity_type"] == "venue"
        assert result["source_id"] == 2
        assert result["city"] == "Rome"
        assert result["address"] == "Via Roma 1"
        assert result["latitude"] == 41.9
        assert result["maps_url"] == "https://maps.google.com/test"
        assert result["price_range"] == "$$$"


class TestComputePriority:
    """Tests for the _compute_priority helper."""

    def test_low_confidence_high_priority(self):
        scored = {"overall_confidence": 0.2, "review_reasons": []}
        priority = _compute_priority(scored)
        assert priority >= 3.0

    def test_medium_confidence_moderate_priority(self):
        scored = {"overall_confidence": 0.4, "review_reasons": []}
        priority = _compute_priority(scored)
        assert priority >= 1.5
        assert priority < 3.0

    def test_high_confidence_low_priority(self):
        scored = {"overall_confidence": 0.8, "review_reasons": []}
        priority = _compute_priority(scored)
        assert priority == 0.0

    def test_review_reasons_increase_priority(self):
        scored_no_reasons = {"overall_confidence": 0.8, "review_reasons": []}
        scored_with_reasons = {
            "overall_confidence": 0.8,
            "review_reasons": ["low_data", "no_images", "conflicting"],
        }
        p1 = _compute_priority(scored_no_reasons)
        p2 = _compute_priority(scored_with_reasons)
        assert p2 > p1

    def test_combined_low_confidence_and_reasons(self):
        scored = {
            "overall_confidence": 0.2,
            "review_reasons": ["r1", "r2"],
        }
        priority = _compute_priority(scored)
        # 3.0 (low confidence) + 2*0.5 (reasons) = 4.0
        assert priority == 4.0
