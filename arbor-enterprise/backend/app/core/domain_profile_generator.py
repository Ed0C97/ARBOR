"""Domain Profile Generator CLI for A.R.B.O.R. Enterprise.

CLI wrapper around ``DomainProfileService`` for generating domain profiles
from the command line.  Internally uses the same validated, multi-pass
generation pipeline as the API endpoints.

Two modes:
  --guided    (default) Structured intake with vertical selection and
              quality aspect rating.  Uses ``DomainProfileService``.
  --legacy    Free-text L1/L2 questionnaire (original 3-phase approach).
              Kept for backward compatibility.

Usage:
    # Recommended: guided mode (structured, validated)
    python -m app.core.domain_profile_generator

    # Legacy: free-text mode
    python -m app.core.domain_profile_generator --legacy

    # With DB calibration (either mode):
    python -m app.core.domain_profile_generator --with-db-sample
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Guided mode — structured intake via DomainProfileService
# ═══════════════════════════════════════════════════════════════════════════


async def run_guided_mode(
    output: str = "domain_profile.json",
    with_db_sample: bool = False,
) -> None:
    """Run the guided domain profile generation with structured intake.

    Uses the same ``DomainProfileService`` as the API, but collects
    input interactively from the CLI.
    """
    from app.core.domain_profile_service import (
        DomainIntake,
        DomainProfileService,
        DomainVertical,
        EntityTypeSpec,
        QualityAspect,
        get_quality_aspects_for_vertical,
    )

    service = DomainProfileService()

    print("\n" + "=" * 60)
    print("  A.R.B.O.R. Domain Profile Generator (Guided Mode)")
    print("  Structured intake with validation")
    print("=" * 60)

    # --- Step 1: Basic info ---
    print("\n--- Step 1/5: Domain Info ---\n")

    domain_name = input("Domain name (e.g., 'Fine Dining Milano'): ").strip()

    # Show verticals
    verticals = service.get_available_verticals()
    print("\nSelect your industry vertical:")
    for i, v in enumerate(verticals, 1):
        print(f"  {i}. {v['label']}")
    vert_idx = input(f"\nChoice (1-{len(verticals)}): ").strip()
    try:
        vertical = DomainVertical(verticals[int(vert_idx) - 1]["id"])
    except (ValueError, IndexError):
        vertical = DomainVertical.OTHER

    geo_focus = input("Geographic focus (e.g., 'Milano, Lombardia'): ").strip()
    language = input("Language for user-facing text (ISO code, e.g., 'it'): ").strip() or "en"

    # --- Step 2: Audience ---
    print("\n--- Step 2/5: Target Audience ---\n")

    audience_desc = input("Describe your target audience: ").strip()
    print("\nExpertise level:")
    print("  1. Novice   2. Intermediate   3. Expert   4. Mixed")
    expertise_choice = input("Choice (1-4): ").strip()
    expertise_map = {"1": "novice", "2": "intermediate", "3": "expert", "4": "mixed"}
    expertise = expertise_map.get(expertise_choice, "mixed")

    # --- Step 3: Entity types ---
    print("\n--- Step 3/5: Entity Types ---\n")

    entity_count_raw = input("How many entity types? (1-3): ").strip()
    try:
        entity_count = max(1, min(3, int(entity_count_raw)))
    except ValueError:
        entity_count = 1

    entity_types: list[EntityTypeSpec] = []
    for i in range(entity_count):
        print(f"\n  Entity type {i + 1}/{entity_count}:")
        et_name = input("    Name (e.g., 'restaurant'): ").strip()
        et_desc = input("    Description: ").strip()
        et_examples_raw = input("    Example entities (comma-separated): ").strip()
        et_examples = [e.strip() for e in et_examples_raw.split(",") if e.strip()]
        entity_types.append(
            EntityTypeSpec(name=et_name, description=et_desc, example_entities=et_examples)
        )

    # --- Step 4: Quality aspects ---
    print("\n--- Step 4/5: Quality Aspects ---\n")
    print("Rate each aspect from 1 (not important) to 5 (critical).")
    print("Optionally describe what 'great' and 'poor' look like.\n")

    raw_aspects = get_quality_aspects_for_vertical(vertical)
    quality_aspects: list[QualityAspect] = []

    for asp in raw_aspects:
        print(f"  {asp['label']}:")
        imp_raw = input(f"    Importance (1-5, default 3): ").strip()
        try:
            importance = max(1, min(5, int(imp_raw)))
        except ValueError:
            importance = 3

        great = ""
        poor = ""
        if importance >= 3:
            great = input(f"    What makes it GREAT? (optional): ").strip()
            poor = input(f"    What makes it POOR? (optional): ").strip()

        quality_aspects.append(
            QualityAspect(
                aspect_id=asp["id"],
                importance=importance,
                what_makes_it_great=great,
                what_makes_it_poor=poor,
            )
        )

    # --- Step 5: Tone ---
    print("\n--- Step 5/5: Advisor Tone ---\n")

    tones = service.get_tone_options()
    for i, t in enumerate(tones, 1):
        print(f"  {i}. {t['id']}: {t['description']}")
    tone_idx = input(f"\nChoice (1-{len(tones)}, default 2): ").strip()
    try:
        tone = tones[int(tone_idx) - 1]["id"]
    except (ValueError, IndexError):
        tone = "warm_expert"

    # --- Calibration entities (optional) ---
    print("\n--- Calibration (optional) ---\n")
    best_raw = input("Names of your BEST entities (comma-separated, or Enter to skip): ").strip()
    best_entities = [e.strip() for e in best_raw.split(",") if e.strip()] if best_raw else []

    avg_raw = input("Names of AVERAGE entities (comma-separated, or Enter to skip): ").strip()
    avg_entities = [e.strip() for e in avg_raw.split(",") if e.strip()] if avg_raw else []

    # --- Build intake ---
    intake = DomainIntake(
        domain_name=domain_name,
        vertical=vertical,
        geographic_focus=geo_focus,
        language=language,
        target_audience_description=audience_desc,
        audience_expertise_level=expertise,
        entity_types=entity_types,
        quality_aspects=quality_aspects,
        advisor_tone=tone,
        sample_best_entities=best_entities,
        sample_average_entities=avg_entities,
    )

    # --- Generate ---
    print("\n" + "=" * 60)
    print("  Generating your domain profile...")
    print("  (multi-pass validation + auto-correction)")
    print("=" * 60 + "\n")

    draft = await service.generate_profile(intake)

    # --- Show results ---
    print(f"\n  Quality Score: {draft.quality_score:.0f}/100")
    print(f"  Valid: {draft.is_valid}")
    print(f"  Rounds: {draft.generation_rounds}")

    if draft.validation:
        if draft.validation.errors:
            print(f"\n  Errors ({len(draft.validation.errors)}):")
            for err in draft.validation.errors:
                print(f"    - [{err.field}] {err.message}")
        if draft.validation.warnings:
            print(f"\n  Warnings ({len(draft.validation.warnings)}):")
            for warn in draft.validation.warnings:
                print(f"    - [{warn.field}] {warn.message}")

    # --- Write output ---
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(draft.profile, f, indent=2, ensure_ascii=False)

    print(f"\n  Profile saved to: {output_path}")
    print(f"  Set DOMAIN_PROFILE_CONFIG_FILE={output_path} in your .env")

    # Show dimension summary
    dims = draft.profile.get("vibe_dimensions", [])
    if dims:
        print(f"\n  Dimensions ({len(dims)}):")
        for dim in dims:
            if isinstance(dim, dict):
                print(
                    f"    - {dim.get('id', '?')} ({dim.get('label', '?')}): "
                    f"weight={dim.get('weight', 1.0)}"
                )

    cats = draft.profile.get("categories", [])
    print(f"  Categories: {len(cats)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Legacy mode — free-text L1/L2/L3 (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

# Legacy data structures kept for backward compatibility
@dataclass
class L1Answers:
    """Responses to the universal (Level 1) questions."""
    domain_name: str = ""
    domain_description: str = ""
    language: str = "en"
    target_audience: str = ""
    entity_types: list[dict[str, str]] = field(default_factory=list)


@dataclass
class L2Question:
    """A single adaptive question generated by the LLM."""
    question: str
    answer_type: str = "text"
    choices: list[str] = field(default_factory=list)
    purpose: str = ""


@dataclass
class L2Answers:
    """Responses to the adaptive (Level 2) questions."""
    questions_and_answers: list[dict[str, str]] = field(default_factory=list)


@dataclass
class L3DataSample:
    """Statistics derived from real database records (Level 3)."""
    categories_found: list[str] = field(default_factory=list)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    numeric_distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    text_field_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_records_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class DomainProfileOutput:
    """The final standardised domain profile."""
    domain_id: str
    name: str
    description: str
    language: str = "en"
    target_audience: str = ""
    vibe_dimensions: list[dict[str, Any]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    scoring_prompt: str = ""
    search_prompt: str = ""
    discovery_persona: str = ""
    search_context_keywords: list[str] = field(default_factory=list)


L1_QUESTIONS = [
    {"key": "domain_name", "prompt": "What is your domain / industry?\n> "},
    {"key": "domain_description", "prompt": "Describe your domain in 1-2 sentences:\n> "},
    {"key": "language", "prompt": "Language (ISO code, e.g., en, it, fr):\n> ", "default": "en"},
    {"key": "target_audience", "prompt": "Who are the end users?\n> "},
    {"key": "entity_type_count", "prompt": "How many entity types?\n> ", "type": "int", "default": "1"},
]


def collect_l1(interactive: bool = True) -> L1Answers:
    """Collect Level 1 answers interactively."""
    answers = L1Answers()
    if not interactive:
        return answers

    print("\n" + "=" * 60)
    print("  A.R.B.O.R. Domain Profile Generator — Phase 1/3 (Legacy)")
    print("=" * 60 + "\n")

    for q in L1_QUESTIONS:
        default = q.get("default", "")
        raw = input(q["prompt"]).strip()
        if not raw and default:
            raw = default

        if q["key"] == "domain_name":
            answers.domain_name = raw
        elif q["key"] == "domain_description":
            answers.domain_description = raw
        elif q["key"] == "language":
            answers.language = raw or "en"
        elif q["key"] == "target_audience":
            answers.target_audience = raw
        elif q["key"] == "entity_type_count":
            try:
                count = int(raw)
            except ValueError:
                count = 1
            for i in range(count):
                print(f"\n--- Entity type {i + 1}/{count} ---")
                name = input("  Short name: ").strip()
                desc = input("  Brief description: ").strip()
                answers.entity_types.append({"name": name, "description": desc})
    return answers


async def generate_l2_questions(l1: L1Answers) -> list[L2Question]:
    """Use the LLM to generate adaptive questions based on L1 answers."""
    try:
        import google.generativeai as genai

        entity_types_str = ", ".join(
            f"{et['name']} ({et['description']})" for et in l1.entity_types
        ) or "not specified"

        user_msg = (
            f"Domain: {l1.domain_name}\nDescription: {l1.domain_description}\n"
            f"Audience: {l1.target_audience}\nLanguage: {l1.language}\n"
            f"Entity types: {entity_types_str}\n\n"
            f"Generate 5-8 targeted questions for building a scoring profile."
        )

        system = (
            "You are a domain analyst for A.R.B.O.R. Generate questions to build "
            "a Vibe DNA scoring system. Output ONLY valid JSON array of question objects: "
            '[{"question":"...","answer_type":"text","choices":[],"purpose":"..."}]'
        )

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )

        response = await asyncio.to_thread(
            model.generate_content,
            user_msg,
        )
        data = json.loads(response.text.strip())
        return [
            L2Question(
                question=q["question"],
                answer_type=q.get("answer_type", "text"),
                choices=q.get("choices", []),
                purpose=q.get("purpose", ""),
            )
            for q in data
        ]
    except Exception as exc:
        logger.warning(f"L2 generation failed: {exc}")
        return [
            L2Question(question="What qualities distinguish the best from the average?", purpose="core dims"),
            L2Question(question="Describe the ideal entity.", purpose="high calibration"),
            L2Question(question="Describe the worst entity.", purpose="low calibration"),
            L2Question(question="How important is price? (1-5)", answer_type="scale_1_5", purpose="price weight"),
            L2Question(question="What tone should the advisor use?", purpose="persona"),
        ]


def collect_l2(questions: list[L2Question], interactive: bool = True) -> L2Answers:
    """Collect Level 2 answers interactively."""
    answers = L2Answers()
    if not interactive:
        return answers

    print("\n" + "=" * 60)
    print("  Phase 2/3 — Domain-Specific Questions (Legacy)")
    print("=" * 60 + "\n")

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q.question}")
        if q.answer_type in ("choice", "multi_choice") and q.choices:
            for j, c in enumerate(q.choices, 1):
                print(f"  {j}. {c}")
        raw = input("> ").strip()
        answers.questions_and_answers.append(
            {"question": q.question, "answer": raw, "purpose": q.purpose}
        )
    return answers


async def collect_l3_from_db(sample_size: int = 50) -> L3DataSample:
    """Sample real records from the configured source database."""
    sample = L3DataSample()
    try:
        from app.config import get_settings
        from app.db.postgres.connection import magazine_engine
        from app.db.postgres.dynamic_model import create_dynamic_model
        from sqlalchemy import MetaData, func, select
        from sqlalchemy.ext.asyncio import AsyncSession

        settings = get_settings()
        configs = settings.get_entity_type_configs()
        if not magazine_engine:
            return sample

        metadata = MetaData()
        for cfg in configs:
            table = await create_dynamic_model(cfg, magazine_engine, metadata)
            if table is None:
                continue
            async with AsyncSession(magazine_engine) as session:
                total = (await session.execute(select(func.count()).select_from(table))).scalar_one()
                sample.total_records_by_type[cfg.entity_type] = total

                rows = (await session.execute(select(table).limit(sample_size))).fetchall()
                for row in rows:
                    record = {col.name: getattr(row, col.name, None) for col in table.columns}
                    sample.sample_records.append(record)

                cat_col = cfg.all_mappings.get("category")
                if cat_col and hasattr(table.c, cat_col):
                    cat_rows = (await session.execute(select(getattr(table.c, cat_col)).distinct())).fetchall()
                    for r in cat_rows:
                        val = r[0]
                        if val and str(val).strip() and val not in sample.categories_found:
                            sample.categories_found.append(str(val).strip())
    except Exception as exc:
        logger.warning(f"L3 sampling failed: {exc}")
    return sample


async def generate_profile_legacy(
    l1: L1Answers, l2: L2Answers, l3: L3DataSample | None = None,
) -> DomainProfileOutput:
    """Legacy: synthesize a profile from L1/L2/L3 answers.

    NOTE: This path does NOT validate/auto-fix. For production use,
    prefer the guided mode via ``DomainProfileService``.
    """
    from app.core.profile_validation import ProfileValidator
    import google.generativeai as genai

    entity_types_str = ", ".join(
        f"{et['name']} ({et['description']})" for et in l1.entity_types
    ) or "not specified"

    l2_text = "\n".join(
        f"Q: {qa['question']}\nA: {qa['answer']}" for qa in l2.questions_and_answers
    ) or "No answers."

    categories = ", ".join(l3.categories_found) if l3 and l3.categories_found else "None"
    record_counts = json.dumps(l3.total_records_by_type) if l3 else "{}"
    sample_records = json.dumps(l3.sample_records[:5], default=str) if l3 else "[]"

    system = (
        "You are the A.R.B.O.R. Domain Profile Architect. Generate a perfect "
        "domain profile JSON. Output ONLY valid JSON with keys: domain_id, name, "
        "description, language, target_audience, vibe_dimensions (5-8 with id, "
        "label, description, low_label, high_label, low_examples, high_examples, "
        "weight), categories, scoring_prompt_template (empty string), "
        "search_prompt_template, discovery_persona, search_context_keywords."
    )

    user_msg = (
        f"Domain: {l1.domain_name}\nDescription: {l1.domain_description}\n"
        f"Language: {l1.language}\nAudience: {l1.target_audience}\n"
        f"Entity types: {entity_types_str}\n\n"
        f"User answers:\n{l2_text}\n\n"
        f"DB categories: {categories}\nRecord counts: {record_counts}\n"
        f"Sample records: {sample_records}"
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )

    response = await asyncio.to_thread(
        model.generate_content,
        user_msg,
    )

    data = json.loads(response.text.strip())

    # Validate the generated profile
    validator = ProfileValidator()
    result = validator.validate(data)
    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    - [{w.field}] {w.message}")
    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"    - [{e.field}] {e.message}")
    print(f"  Quality score: {result.score:.0f}/100")

    return DomainProfileOutput(
        domain_id=data.get("domain_id", l1.domain_name.lower().replace(" ", "_")),
        name=data.get("name", l1.domain_name),
        description=data.get("description", l1.domain_description),
        language=data.get("language", l1.language),
        target_audience=data.get("target_audience", l1.target_audience),
        vibe_dimensions=data.get("vibe_dimensions", []),
        categories=data.get("categories", []),
        scoring_prompt=data.get("scoring_prompt_template", ""),
        search_prompt=data.get("search_prompt_template", ""),
        discovery_persona=data.get("discovery_persona", ""),
        search_context_keywords=data.get("search_context_keywords", []),
    )


def write_profile(profile: DomainProfileOutput, output_path: Path) -> None:
    """Write the domain profile to a JSON file."""
    output = {
        "domain_id": profile.domain_id,
        "name": profile.name,
        "description": profile.description,
        "language": profile.language,
        "target_audience": profile.target_audience,
        "vibe_dimensions": profile.vibe_dimensions,
        "categories": profile.categories,
        "scoring_prompt_template": profile.scoring_prompt,
        "search_prompt_template": profile.search_prompt,
        "discovery_persona": profile.discovery_persona,
        "search_context_keywords": profile.search_context_keywords,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Profile saved to: {output_path}")
    print(f"  Set DOMAIN_PROFILE_CONFIG_FILE={output_path} in your .env")


async def run_legacy_mode(
    with_db_sample: bool = False, output: str = "domain_profile.json"
) -> None:
    """Run the legacy free-text 3-phase pipeline."""
    print("\n" + "=" * 60)
    print("  A.R.B.O.R. Domain Profile Generator (Legacy Mode)")
    print("  Free-text questionnaire")
    print("=" * 60)

    l1 = collect_l1(interactive=True)

    print("\n  Generating questions...")
    l2_questions = await generate_l2_questions(l1)
    l2 = collect_l2(l2_questions, interactive=True)

    l3: L3DataSample | None = None
    if with_db_sample:
        print("\n  Sampling database...")
        l3 = await collect_l3_from_db()

    print("\n  Generating profile...")
    profile = await generate_profile_legacy(l1, l2, l3)

    output_path = Path(output)
    write_profile(profile, output_path)

    print(f"\n  Domain: {profile.name}")
    print(f"  Dimensions: {len(profile.vibe_dimensions)}")
    for dim in profile.vibe_dimensions:
        if isinstance(dim, dict):
            print(f"    - {dim['id']} ({dim.get('label', '')})")
    print(f"  Categories: {len(profile.categories)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


async def main(
    legacy: bool = False,
    with_db_sample: bool = False,
    output: str = "domain_profile.json",
) -> None:
    """Run the domain profile generator."""
    if legacy:
        await run_legacy_mode(with_db_sample=with_db_sample, output=output)
    else:
        await run_guided_mode(output=output, with_db_sample=with_db_sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A.R.B.O.R. Domain Profile Generator")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy free-text mode instead of guided structured mode",
    )
    parser.add_argument(
        "--with-db-sample",
        action="store_true",
        help="Sample real records from the source database for calibration",
    )
    parser.add_argument(
        "--output", "-o",
        default="domain_profile.json",
        help="Output file path (default: domain_profile.json)",
    )
    args = parser.parse_args()

    asyncio.run(main(legacy=args.legacy, with_db_sample=args.with_db_sample, output=args.output))
