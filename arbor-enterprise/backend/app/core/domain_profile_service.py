"""Domain Profile Service for A.R.B.O.R. Enterprise.

Server-side domain profile generation service that ARBOR provides to clients
during onboarding.  Unlike the CLI-based ``domain_profile_generator.py``
(which relies on free-text user input), this service uses a structured,
multi-step conversational flow with validation and auto-correction to
produce near-error-free domain profiles.

Architecture:
    1. **Structured intake** — collects domain info via typed fields, not
       free-form text, eliminating user description errors.
    2. **Multi-pass generation** — generates a profile, validates it,
       auto-fixes issues, and re-validates (up to N rounds).
    3. **Semantic validation** — ensures dimensions are distinct, well-
       calibrated, and domain-appropriate.
    4. **Draft → Preview → Activate** — profiles go through a review
       pipeline before they affect scoring.

Usage:
    service = DomainProfileService()

    # Step 1: Client provides structured domain info
    draft = await service.generate_profile(intake)

    # Step 2: Review the draft
    validation = service.validate_profile(draft)

    # Step 3: Auto-fix and finalize
    if not validation.is_valid:
        draft = await service.auto_fix_profile(draft, validation)

    # Step 4: Activate
    service.activate_profile(draft)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.profile_validation import ProfileValidator, ValidationResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Structured intake — replaces free-text input
# ═══════════════════════════════════════════════════════════════════════════


class DomainVertical(str, Enum):
    """Pre-defined domain verticals for guided onboarding.

    The client selects the closest match. This gives the LLM strong priors
    about what dimensions and categories are relevant, dramatically reducing
    generation errors.
    """

    FOOD_DINING = "food_dining"
    FASHION_RETAIL = "fashion_retail"
    HOSPITALITY_HOTELS = "hospitality_hotels"
    REAL_ESTATE = "real_estate"
    HEALTH_WELLNESS = "health_wellness"
    BEAUTY_COSMETICS = "beauty_cosmetics"
    ENTERTAINMENT_EVENTS = "entertainment_events"
    EDUCATION_TRAINING = "education_training"
    PROFESSIONAL_SERVICES = "professional_services"
    AUTOMOTIVE = "automotive"
    TRAVEL_TOURISM = "travel_tourism"
    ART_CULTURE = "art_culture"
    SPORTS_FITNESS = "sports_fitness"
    TECHNOLOGY_SAAS = "technology_saas"
    OTHER = "other"


@dataclass
class EntityTypeSpec:
    """Specification for a single entity type in the client's domain."""

    name: str  # e.g., "restaurant", "hotel"
    description: str = ""  # Brief explanation
    example_entities: list[str] = field(default_factory=list)
    # 3-5 real examples from their DB, e.g., ["Osteria Francescana", "Cracco"]


@dataclass
class QualityAspect:
    """A quality aspect the client considers important.

    Instead of asking "what dimensions matter?" (vague), we present
    a checklist of common quality aspects per vertical and let the
    client rate importance (1-5) and optionally describe extremes.
    """

    aspect_id: str  # e.g., "product_quality", "service_level"
    importance: int = 3  # 1 (not important) to 5 (critical)
    what_makes_it_great: str = ""  # Client's description of the ideal
    what_makes_it_poor: str = ""  # Client's description of the worst


@dataclass
class DomainIntake:
    """Structured domain information collected from the client.

    This replaces the free-text L1/L2 questions with typed, validated
    fields.  The structure eliminates ambiguity and reduces errors.
    """

    # --- Basic info (required) ---
    domain_name: str  # e.g., "Fine Dining Milano"
    vertical: DomainVertical = DomainVertical.OTHER
    geographic_focus: str = ""  # e.g., "Milano, Lombardia"
    language: str = "en"  # ISO 639-1

    # --- Audience ---
    target_audience_description: str = ""
    # e.g., "Food bloggers, wealthy diners, tourists"
    audience_expertise_level: str = "mixed"
    # "novice", "intermediate", "expert", "mixed"

    # --- Entity types ---
    entity_types: list[EntityTypeSpec] = field(default_factory=list)

    # --- Quality aspects (the key innovation) ---
    quality_aspects: list[QualityAspect] = field(default_factory=list)
    # Pre-populated per vertical, client rates importance + describes extremes

    # --- Tone & style ---
    advisor_tone: str = "warm_expert"
    # "formal_expert", "warm_expert", "casual_friend", "enthusiastic_guide"

    # --- Optional: existing categories ---
    known_categories: list[str] = field(default_factory=list)
    # If they already have categories in their DB

    # --- Optional: sample entities for calibration ---
    sample_best_entities: list[str] = field(default_factory=list)
    # Names of their "best" entities (for high-end calibration)
    sample_average_entities: list[str] = field(default_factory=list)
    # Names of "average" entities (for mid-range calibration)


# ═══════════════════════════════════════════════════════════════════════════
# Vertical-specific quality aspect templates
# ═══════════════════════════════════════════════════════════════════════════

# These provide strong priors. The client selects a vertical, gets a
# pre-populated list of quality aspects to rate. This is WAY more
# precise than "tell me what matters in your domain".

VERTICAL_QUALITY_ASPECTS: dict[DomainVertical, list[dict[str, str]]] = {
    DomainVertical.FOOD_DINING: [
        {"id": "culinary_mastery", "label": "Culinary technique & creativity"},
        {"id": "ingredient_quality", "label": "Quality & sourcing of ingredients"},
        {"id": "ambiance", "label": "Atmosphere, design & setting"},
        {"id": "service_excellence", "label": "Service quality & professionalism"},
        {"id": "beverage_program", "label": "Wine/cocktail selection & quality"},
        {"id": "price_value", "label": "Price-to-value ratio"},
        {"id": "exclusivity", "label": "Exclusivity & uniqueness of experience"},
        {"id": "consistency", "label": "Consistency of quality over time"},
    ],
    DomainVertical.FASHION_RETAIL: [
        {"id": "design_quality", "label": "Design originality & aesthetic"},
        {"id": "craftsmanship", "label": "Material quality & craftsmanship"},
        {"id": "brand_identity", "label": "Brand identity & storytelling"},
        {"id": "shopping_experience", "label": "In-store/online shopping experience"},
        {"id": "price_positioning", "label": "Price positioning & perceived value"},
        {"id": "sustainability", "label": "Sustainability & ethical practices"},
        {"id": "trend_relevance", "label": "Trend awareness & innovation"},
        {"id": "customer_service", "label": "Customer service & after-sales"},
    ],
    DomainVertical.HOSPITALITY_HOTELS: [
        {"id": "room_quality", "label": "Room quality, comfort & amenities"},
        {"id": "location", "label": "Location & accessibility"},
        {"id": "service_level", "label": "Service quality & attentiveness"},
        {"id": "dining_options", "label": "On-site dining & F&B quality"},
        {"id": "facilities", "label": "Facilities (spa, pool, gym, etc.)"},
        {"id": "design_atmosphere", "label": "Design, architecture & atmosphere"},
        {"id": "value_proposition", "label": "Value for money"},
        {"id": "uniqueness", "label": "Uniqueness & character"},
    ],
    DomainVertical.REAL_ESTATE: [
        {"id": "location_quality", "label": "Location & neighborhood desirability"},
        {"id": "build_quality", "label": "Construction quality & materials"},
        {"id": "design_architecture", "label": "Architectural design & layout"},
        {"id": "amenities", "label": "Building/unit amenities"},
        {"id": "investment_potential", "label": "Investment potential & ROI"},
        {"id": "price_positioning", "label": "Price relative to market"},
        {"id": "energy_efficiency", "label": "Energy efficiency & sustainability"},
        {"id": "connectivity", "label": "Transport links & accessibility"},
    ],
    DomainVertical.HEALTH_WELLNESS: [
        {"id": "practitioner_quality", "label": "Practitioner expertise & credentials"},
        {"id": "treatment_range", "label": "Range & depth of treatments/services"},
        {"id": "facility_quality", "label": "Facility quality & cleanliness"},
        {"id": "results_effectiveness", "label": "Treatment effectiveness & results"},
        {"id": "patient_experience", "label": "Patient/client experience & care"},
        {"id": "innovation", "label": "Use of modern techniques & technology"},
        {"id": "accessibility", "label": "Accessibility & availability"},
        {"id": "price_value", "label": "Price-to-value ratio"},
    ],
    DomainVertical.BEAUTY_COSMETICS: [
        {"id": "product_quality", "label": "Product quality & formulation"},
        {"id": "brand_prestige", "label": "Brand prestige & reputation"},
        {"id": "ingredient_sourcing", "label": "Ingredient quality & transparency"},
        {"id": "innovation", "label": "Innovation & technology"},
        {"id": "sustainability", "label": "Sustainability & eco-consciousness"},
        {"id": "user_experience", "label": "User experience & packaging"},
        {"id": "effectiveness", "label": "Effectiveness & results"},
        {"id": "price_positioning", "label": "Price positioning"},
    ],
}

# Default quality aspects for verticals not explicitly listed
_DEFAULT_QUALITY_ASPECTS = [
    {"id": "core_quality", "label": "Core product/service quality"},
    {"id": "experience", "label": "Overall customer experience"},
    {"id": "price_value", "label": "Price-to-value ratio"},
    {"id": "uniqueness", "label": "Uniqueness & differentiation"},
    {"id": "service_level", "label": "Service & support quality"},
    {"id": "reputation", "label": "Reputation & trustworthiness"},
    {"id": "accessibility", "label": "Accessibility & convenience"},
    {"id": "innovation", "label": "Innovation & modernity"},
]


def get_quality_aspects_for_vertical(vertical: DomainVertical) -> list[dict[str, str]]:
    """Return the pre-defined quality aspects for a given vertical.

    The client receives this list and rates each aspect's importance.
    """
    return VERTICAL_QUALITY_ASPECTS.get(vertical, _DEFAULT_QUALITY_ASPECTS)


# ═══════════════════════════════════════════════════════════════════════════
# Tone templates
# ═══════════════════════════════════════════════════════════════════════════

TONE_TEMPLATES: dict[str, str] = {
    "formal_expert": (
        "You speak with measured authority and precision. "
        "Your tone is professional and data-driven, like a respected industry analyst."
    ),
    "warm_expert": (
        "You are warm but authoritative, like a knowledgeable friend "
        "who genuinely cares about finding the perfect match."
    ),
    "casual_friend": (
        "You are relaxed and approachable, like a friend who happens to know "
        "everything about this domain and loves sharing recommendations."
    ),
    "enthusiastic_guide": (
        "You are passionate and enthusiastic, showing genuine excitement "
        "when you find a great match. Your energy is infectious."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Profile draft — intermediate representation before activation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ProfileDraft:
    """A generated domain profile that hasn't been activated yet.

    Tracks generation metadata for audit and debugging.
    """

    draft_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    profile: dict[str, Any] = field(default_factory=dict)
    validation: ValidationResult | None = None
    generation_rounds: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    intake_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validation is not None and self.validation.is_valid

    @property
    def quality_score(self) -> float:
        return self.validation.score if self.validation else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Domain Profile Service
# ═══════════════════════════════════════════════════════════════════════════


class DomainProfileService:
    """Server-side domain profile generation and management service.

    This is the core service that ARBOR provides to clients during
    onboarding.  It replaces the CLI-based generator with a validated,
    multi-pass pipeline that produces near-error-free profiles.

    Usage:
        service = DomainProfileService()

        # Get quality aspects for the client's vertical
        aspects = service.get_quality_aspects(DomainVertical.FOOD_DINING)

        # Client fills in the intake form
        intake = DomainIntake(
            domain_name="Fine Dining Milano",
            vertical=DomainVertical.FOOD_DINING,
            language="it",
            ...
        )

        # Generate + validate + auto-fix
        draft = await service.generate_profile(intake)

        # Activate when ready
        service.activate_profile(draft)
    """

    MAX_AUTO_FIX_ROUNDS = 3
    MIN_QUALITY_SCORE = 70.0  # Minimum score to allow activation

    def __init__(self) -> None:
        self._validator = ProfileValidator()
        self._drafts: dict[str, ProfileDraft] = {}

    # --- Public API -------------------------------------------------------

    def get_quality_aspects(self, vertical: DomainVertical) -> list[dict[str, str]]:
        """Get the pre-defined quality aspects for a vertical.

        Returns a list of aspects the client should rate during intake.
        """
        return get_quality_aspects_for_vertical(vertical)

    def get_available_verticals(self) -> list[dict[str, str]]:
        """Return all available domain verticals with labels."""
        return [
            {"id": v.value, "label": v.name.replace("_", " ").title()}
            for v in DomainVertical
        ]

    def get_tone_options(self) -> list[dict[str, str]]:
        """Return available advisor tone options."""
        return [
            {"id": tone_id, "description": desc}
            for tone_id, desc in TONE_TEMPLATES.items()
        ]

    async def generate_profile(
        self,
        intake: DomainIntake,
        max_rounds: int | None = None,
    ) -> ProfileDraft:
        """Generate a domain profile from structured intake data.

        Runs a multi-pass pipeline:
          1. Generate initial profile via LLM
          2. Validate
          3. If issues found, auto-fix and re-validate (up to N rounds)

        Args:
            intake: Structured domain information from the client.
            max_rounds: Override for max auto-fix rounds.

        Returns:
            A ProfileDraft with the generated profile and validation results.
        """
        rounds = max_rounds or self.MAX_AUTO_FIX_ROUNDS

        # Step 1: Generate initial profile
        logger.info(f"Generating domain profile for '{intake.domain_name}'...")
        profile_dict = await self._generate_via_llm(intake)

        draft = ProfileDraft(
            profile=profile_dict,
            generation_rounds=1,
            intake_snapshot=self._intake_to_dict(intake),
        )

        # Step 2: Validate
        validation = self._validator.validate(profile_dict)
        draft.validation = validation

        logger.info(
            f"Round 1 validation: score={validation.score:.0f}, "
            f"errors={len(validation.errors)}, warnings={len(validation.warnings)}"
        )

        # Step 3: Multi-pass auto-fix loop
        current_round = 1
        while validation.has_errors and current_round < rounds:
            current_round += 1
            logger.info(f"Auto-fix round {current_round}/{rounds}...")

            fix_instructions = validation.auto_fix_instructions
            if not fix_instructions:
                logger.warning("No auto-fixable issues found, stopping auto-fix")
                break

            profile_dict = await self._auto_fix_via_llm(
                profile_dict, fix_instructions, intake
            )
            validation = self._validator.validate(profile_dict)

            draft.profile = profile_dict
            draft.validation = validation
            draft.generation_rounds = current_round

            logger.info(
                f"Round {current_round} validation: score={validation.score:.0f}, "
                f"errors={len(validation.errors)}, warnings={len(validation.warnings)}"
            )

        # Store draft
        self._drafts[draft.draft_id] = draft

        logger.info(
            f"Profile generation complete: draft_id={draft.draft_id}, "
            f"score={draft.quality_score:.0f}, valid={draft.is_valid}, "
            f"rounds={draft.generation_rounds}"
        )

        return draft

    def validate_profile(self, profile_dict: dict[str, Any]) -> ValidationResult:
        """Validate a profile dict without generating it.

        Useful for validating externally provided or manually edited profiles.
        """
        return self._validator.validate(profile_dict)

    def get_draft(self, draft_id: str) -> ProfileDraft | None:
        """Retrieve a stored draft by ID."""
        return self._drafts.get(draft_id)

    def list_drafts(self) -> list[dict[str, Any]]:
        """List all stored drafts with summary info."""
        return [
            {
                "draft_id": d.draft_id,
                "domain_name": d.profile.get("name", "?"),
                "quality_score": d.quality_score,
                "is_valid": d.is_valid,
                "generation_rounds": d.generation_rounds,
                "created_at": d.created_at,
            }
            for d in self._drafts.values()
        ]

    def activate_profile(self, draft: ProfileDraft) -> dict[str, Any]:
        """Activate a draft profile, making it the active domain config.

        Args:
            draft: The draft to activate. Must be valid.

        Returns:
            The activated profile dict.

        Raises:
            ValueError: If the draft is not valid or below minimum quality.
        """
        if not draft.is_valid:
            raise ValueError(
                f"Cannot activate invalid profile (draft_id={draft.draft_id}). "
                f"Fix validation errors first."
            )

        if draft.quality_score < self.MIN_QUALITY_SCORE:
            raise ValueError(
                f"Profile quality score ({draft.quality_score:.0f}) is below "
                f"minimum threshold ({self.MIN_QUALITY_SCORE:.0f}). "
                f"Improve the profile before activating."
            )

        # Import and register
        from app.core.domain_portability import DomainExporter, get_domain_registry

        exporter = DomainExporter()
        config = exporter.import_domain_config(draft.profile)

        registry = get_domain_registry()
        registry.register_domain(config)
        registry.set_active_domain(config.domain_id)

        logger.info(
            f"Profile activated: domain_id='{config.domain_id}', "
            f"name='{config.name}', score={draft.quality_score:.0f}"
        )

        return draft.profile

    # --- LLM generation (internal) ----------------------------------------

    async def _generate_via_llm(self, intake: DomainIntake) -> dict[str, Any]:
        """Generate the initial profile using the LLM.

        The prompt is carefully structured using the intake data to
        minimize LLM "hallucination" and transcription errors.
        """
        system_prompt = self._build_generation_system_prompt()
        user_prompt = self._build_generation_user_prompt(intake)

        return await self._call_llm(system_prompt, user_prompt)

    async def _auto_fix_via_llm(
        self,
        current_profile: dict[str, Any],
        fix_instructions: list[str],
        intake: DomainIntake,
    ) -> dict[str, Any]:
        """Use the LLM to fix validation issues in the profile."""
        system_prompt = (
            "You are the A.R.B.O.R. Domain Profile Architect.\n\n"
            "You will receive a domain profile JSON that has validation issues. "
            "Fix ONLY the issues listed below. Do NOT change anything else.\n\n"
            "Output ONLY the corrected JSON. No explanation, no markdown."
        )

        fixes_text = "\n".join(f"- {fix}" for fix in fix_instructions)
        user_prompt = (
            f"=== CURRENT PROFILE ===\n"
            f"{json.dumps(current_profile, indent=2, ensure_ascii=False)}\n\n"
            f"=== ISSUES TO FIX ===\n"
            f"{fixes_text}\n\n"
            f"=== DOMAIN CONTEXT ===\n"
            f"Domain: {intake.domain_name}\n"
            f"Vertical: {intake.vertical.value}\n"
            f"Language: {intake.language}\n\n"
            f"Output the corrected JSON only."
        )

        return await self._call_llm(system_prompt, user_prompt)

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Call the LLM and parse the JSON response."""
        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            response = await asyncio.to_thread(
                model.generate_content,
                user_prompt,
            )

            raw = response.text.strip()
            return json.loads(raw)

        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            raise RuntimeError(f"Profile generation failed: {exc}") from exc

    # --- Prompt construction (internal) -----------------------------------

    def _build_generation_system_prompt(self) -> str:
        """Build the system prompt for profile generation."""
        return (
            "You are the A.R.B.O.R. Domain Profile Architect.\n\n"
            "Your task is to generate a PERFECT domain profile JSON based on "
            "structured client data. The profile will directly control how an "
            "AI scoring engine evaluates entities in this domain.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. ALL dimension IDs must be lowercase snake_case (English)\n"
            "2. ALL user-facing text must be in the specified language\n"
            "3. Each dimension must have concrete, vivid low/high extremes\n"
            "4. Examples must reference real-world archetypes the audience knows\n"
            "5. Weights must reflect the stated quality aspect importances\n"
            "6. Categories must be comprehensive but not excessive (5-25)\n"
            "7. The discovery persona must match the requested tone\n"
            "8. Search keywords must be domain-specific and natural\n\n"
            "DIMENSION WEIGHT MAPPING:\n"
            "- Importance 5 (critical) → weight 1.8-2.0\n"
            "- Importance 4 (very important) → weight 1.3-1.5\n"
            "- Importance 3 (important) → weight 1.0\n"
            "- Importance 2 (somewhat important) → weight 0.7-0.8\n"
            "- Importance 1 (not important) → weight 0.5\n\n"
            "OUTPUT FORMAT: Valid JSON only, no markdown, no explanation.\n"
            "Required keys: domain_id, name, description, language, "
            "target_audience, vibe_dimensions, categories, "
            "scoring_prompt_template (empty string), "
            "search_prompt_template, discovery_persona, "
            "search_context_keywords"
        )

    def _build_generation_user_prompt(self, intake: DomainIntake) -> str:
        """Build the user prompt from structured intake data."""

        # Format quality aspects
        aspects_text = ""
        if intake.quality_aspects:
            aspects_lines = []
            for qa in intake.quality_aspects:
                line = f"- {qa.aspect_id}: importance={qa.importance}/5"
                if qa.what_makes_it_great:
                    line += f"\n  GREAT: {qa.what_makes_it_great}"
                if qa.what_makes_it_poor:
                    line += f"\n  POOR: {qa.what_makes_it_poor}"
                aspects_lines.append(line)
            aspects_text = "\n".join(aspects_lines)
        else:
            aspects_text = "(no specific quality aspects provided)"

        # Format entity types
        entity_types_text = ""
        if intake.entity_types:
            for et in intake.entity_types:
                entity_types_text += f"- {et.name}: {et.description}\n"
                if et.example_entities:
                    entity_types_text += (
                        f"  Examples: {', '.join(et.example_entities)}\n"
                    )
        else:
            entity_types_text = "(not specified)"

        # Format calibration entities
        calibration_text = ""
        if intake.sample_best_entities:
            calibration_text += (
                f"Best entities (score ~90-100): "
                f"{', '.join(intake.sample_best_entities)}\n"
            )
        if intake.sample_average_entities:
            calibration_text += (
                f"Average entities (score ~40-60): "
                f"{', '.join(intake.sample_average_entities)}\n"
            )
        if not calibration_text:
            calibration_text = "(none provided)"

        # Tone description
        tone_desc = TONE_TEMPLATES.get(
            intake.advisor_tone,
            TONE_TEMPLATES["warm_expert"],
        )

        return (
            f"=== DOMAIN SPECIFICATION ===\n"
            f"Domain name: {intake.domain_name}\n"
            f"Industry vertical: {intake.vertical.value}\n"
            f"Geographic focus: {intake.geographic_focus or 'global'}\n"
            f"Language for ALL user-facing text: {intake.language}\n\n"
            f"=== TARGET AUDIENCE ===\n"
            f"Description: {intake.target_audience_description}\n"
            f"Expertise level: {intake.audience_expertise_level}\n\n"
            f"=== ENTITY TYPES ===\n"
            f"{entity_types_text}\n"
            f"=== QUALITY ASPECTS (rated by client) ===\n"
            f"These define what the scoring dimensions should measure.\n"
            f"Map each aspect with importance >= 2 to a vibe dimension.\n"
            f"Aspects with importance 1 may be excluded.\n\n"
            f"{aspects_text}\n\n"
            f"=== CALIBRATION ENTITIES ===\n"
            f"Use these as examples in low_examples / high_examples:\n"
            f"{calibration_text}\n"
            f"=== ADVISOR TONE ===\n"
            f"{tone_desc}\n\n"
            f"=== KNOWN CATEGORIES ===\n"
            f"{', '.join(intake.known_categories) if intake.known_categories else '(generate appropriate categories for this domain)'}\n\n"
            f"Generate the complete domain profile JSON now."
        )

    # --- Utility ----------------------------------------------------------

    @staticmethod
    def _intake_to_dict(intake: DomainIntake) -> dict[str, Any]:
        """Convert intake to a serializable dict for audit."""
        return {
            "domain_name": intake.domain_name,
            "vertical": intake.vertical.value,
            "geographic_focus": intake.geographic_focus,
            "language": intake.language,
            "target_audience_description": intake.target_audience_description,
            "audience_expertise_level": intake.audience_expertise_level,
            "entity_types": [
                {"name": et.name, "description": et.description}
                for et in intake.entity_types
            ],
            "quality_aspects": [
                {
                    "aspect_id": qa.aspect_id,
                    "importance": qa.importance,
                    "what_makes_it_great": qa.what_makes_it_great,
                    "what_makes_it_poor": qa.what_makes_it_poor,
                }
                for qa in intake.quality_aspects
            ],
            "advisor_tone": intake.advisor_tone,
            "known_categories": intake.known_categories,
            "sample_best_entities": intake.sample_best_entities,
            "sample_average_entities": intake.sample_average_entities,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_service: DomainProfileService | None = None


def get_profile_service() -> DomainProfileService:
    """Return the singleton DomainProfileService."""
    global _service
    if _service is None:
        _service = DomainProfileService()
    return _service
