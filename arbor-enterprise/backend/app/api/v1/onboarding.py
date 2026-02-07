"""Domain Onboarding API — guided domain profile generation service.

Provides a structured, multi-step API for generating and activating
domain profiles.  This is the ARBOR-provided service that replaces
client-side profile generation, ensuring near-zero error rates.

Flow:
    1. GET  /onboarding/verticals          → list available verticals
    2. GET  /onboarding/aspects/{vertical} → get quality aspects for a vertical
    3. GET  /onboarding/tones              → list advisor tone options
    4. POST /onboarding/generate           → generate a profile from intake
    5. GET  /onboarding/drafts             → list all drafts
    6. GET  /onboarding/drafts/{id}        → get a specific draft
    7. POST /onboarding/validate           → validate a profile (standalone)
    8. POST /onboarding/activate/{id}      → activate a draft
    9. GET  /onboarding/export/{id}        → download draft as JSON file
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.domain_profile_service import (
    DomainIntake,
    DomainProfileService,
    DomainVertical,
    EntityTypeSpec,
    QualityAspect,
    get_profile_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QualityAspectRequest(BaseModel):
    """A quality aspect with importance rating from the client."""

    aspect_id: str
    importance: int = Field(ge=1, le=5, default=3)
    what_makes_it_great: str = ""
    what_makes_it_poor: str = ""


class EntityTypeRequest(BaseModel):
    """Entity type specification from the client."""

    name: str
    description: str = ""
    example_entities: list[str] = []


class GenerateProfileRequest(BaseModel):
    """Full intake form for domain profile generation."""

    domain_name: str
    vertical: str = "other"
    geographic_focus: str = ""
    language: str = "en"

    target_audience_description: str = ""
    audience_expertise_level: str = "mixed"

    entity_types: list[EntityTypeRequest] = []
    quality_aspects: list[QualityAspectRequest] = []

    advisor_tone: str = "warm_expert"
    known_categories: list[str] = []

    sample_best_entities: list[str] = []
    sample_average_entities: list[str] = []


class DraftSummaryResponse(BaseModel):
    """Summary of a profile draft."""

    draft_id: str
    domain_name: str = ""
    quality_score: float = 0.0
    is_valid: bool = False
    generation_rounds: int = 0
    created_at: str = ""


class DraftDetailResponse(BaseModel):
    """Full detail of a profile draft."""

    draft_id: str
    profile: dict[str, Any] = {}
    quality_score: float = 0.0
    is_valid: bool = False
    generation_rounds: int = 0
    created_at: str = ""
    validation: dict[str, Any] | None = None
    intake_snapshot: dict[str, Any] = {}


class ValidationResponse(BaseModel):
    """Validation result for a profile."""

    is_valid: bool = False
    score: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []


class ActivateResponse(BaseModel):
    """Response after activating a profile."""

    domain_id: str
    name: str
    status: str = "activated"
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/verticals")
async def list_verticals():
    """List all available domain verticals.

    Step 1 of the onboarding flow. The client selects the closest
    vertical match to get pre-populated quality aspects.
    """
    service = get_profile_service()
    return {"verticals": service.get_available_verticals()}


@router.get("/aspects/{vertical}")
async def get_quality_aspects(vertical: str):
    """Get pre-defined quality aspects for a domain vertical.

    Step 2 of the onboarding flow. The client receives a list of
    quality aspects to rate (importance 1-5) and optionally describe.

    Args:
        vertical: The domain vertical ID (e.g., "food_dining").
    """
    try:
        vert_enum = DomainVertical(vertical)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vertical '{vertical}'. Use GET /onboarding/verticals for valid options.",
        )

    service = get_profile_service()
    aspects = service.get_quality_aspects(vert_enum)
    return {"vertical": vertical, "aspects": aspects}


@router.get("/tones")
async def list_tone_options():
    """List available advisor tone options.

    Step 3 of the onboarding flow. The client selects the
    communication style for the discovery advisor.
    """
    service = get_profile_service()
    return {"tones": service.get_tone_options()}


@router.post("/generate", response_model=DraftDetailResponse)
async def generate_profile(body: GenerateProfileRequest):
    """Generate a domain profile from structured intake data.

    This is the main generation endpoint. Takes the complete intake
    form and produces a validated, auto-corrected domain profile.

    The profile goes through multiple validation and auto-fix rounds
    to ensure near-zero error rates.

    Returns the draft with profile, validation results, and quality score.
    """
    service = get_profile_service()

    # Convert request to DomainIntake
    try:
        vertical = DomainVertical(body.vertical)
    except ValueError:
        vertical = DomainVertical.OTHER

    intake = DomainIntake(
        domain_name=body.domain_name,
        vertical=vertical,
        geographic_focus=body.geographic_focus,
        language=body.language,
        target_audience_description=body.target_audience_description,
        audience_expertise_level=body.audience_expertise_level,
        entity_types=[
            EntityTypeSpec(
                name=et.name,
                description=et.description,
                example_entities=et.example_entities,
            )
            for et in body.entity_types
        ],
        quality_aspects=[
            QualityAspect(
                aspect_id=qa.aspect_id,
                importance=qa.importance,
                what_makes_it_great=qa.what_makes_it_great,
                what_makes_it_poor=qa.what_makes_it_poor,
            )
            for qa in body.quality_aspects
        ],
        advisor_tone=body.advisor_tone,
        known_categories=body.known_categories,
        sample_best_entities=body.sample_best_entities,
        sample_average_entities=body.sample_average_entities,
    )

    try:
        draft = await service.generate_profile(intake)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return DraftDetailResponse(
        draft_id=draft.draft_id,
        profile=draft.profile,
        quality_score=draft.quality_score,
        is_valid=draft.is_valid,
        generation_rounds=draft.generation_rounds,
        created_at=draft.created_at,
        validation=draft.validation.to_dict() if draft.validation else None,
        intake_snapshot=draft.intake_snapshot,
    )


@router.get("/drafts")
async def list_drafts():
    """List all generated profile drafts."""
    service = get_profile_service()
    return {"drafts": service.list_drafts()}


@router.get("/drafts/{draft_id}", response_model=DraftDetailResponse)
async def get_draft(draft_id: str):
    """Get full details of a specific draft."""
    service = get_profile_service()
    draft = service.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    return DraftDetailResponse(
        draft_id=draft.draft_id,
        profile=draft.profile,
        quality_score=draft.quality_score,
        is_valid=draft.is_valid,
        generation_rounds=draft.generation_rounds,
        created_at=draft.created_at,
        validation=draft.validation.to_dict() if draft.validation else None,
        intake_snapshot=draft.intake_snapshot,
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_profile(profile: dict[str, Any]):
    """Validate a profile JSON (standalone, without generation).

    Useful for validating manually edited or externally provided profiles.
    """
    service = get_profile_service()
    result = service.validate_profile(profile)

    return ValidationResponse(**result.to_dict())


@router.post("/activate/{draft_id}", response_model=ActivateResponse)
async def activate_draft(draft_id: str):
    """Activate a draft profile, making it the active domain configuration.

    The draft must be valid and meet the minimum quality score threshold.
    Once activated, the scoring engine, search, and discovery modules
    will use this profile's dimensions, categories, and persona.
    """
    service = get_profile_service()
    draft = service.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    try:
        profile = service.activate_profile(draft)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ActivateResponse(
        domain_id=profile.get("domain_id", ""),
        name=profile.get("name", ""),
        status="activated",
        quality_score=draft.quality_score,
    )


@router.get("/export/{draft_id}")
async def export_draft(draft_id: str):
    """Export a draft profile as a downloadable JSON.

    Returns the profile JSON that can be saved as DOMAIN_PROFILE_CONFIG_FILE.
    """
    service = get_profile_service()
    draft = service.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    return JSONResponse(
        content=draft.profile,
        headers={
            "Content-Disposition": (
                f'attachment; filename="domain_profile_{draft.profile.get("domain_id", "draft")}.json"'
            ),
        },
    )
