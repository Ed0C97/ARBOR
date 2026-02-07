"""Quick test for DomainProfileService."""

import asyncio
import os
from pathlib import Path

# Load .env from project root
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

# Configure genai with API key if present
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    print(f"  Using GOOGLE_API_KEY: {api_key[:8]}...")


async def test_profile_generation():
    """Test the full profile generation pipeline."""
    from app.core.domain_profile_service import (
        DomainIntake,
        DomainProfileService,
        DomainVertical,
        EntityTypeSpec,
        QualityAspect,
    )

    print("=" * 60)
    print("  Testing DomainProfileService")
    print("=" * 60)

    service = DomainProfileService()

    # Step 1: Show available verticals
    print("\n1. Available verticals:")
    for v in service.get_available_verticals()[:5]:
        print(f"   - {v['id']}: {v['label']}")

    # Step 2: Get quality aspects for food_dining
    print("\n2. Quality aspects for FOOD_DINING:")
    aspects = service.get_quality_aspects(DomainVertical.FOOD_DINING)
    for a in aspects[:4]:
        print(f"   - {a['id']}: {a['label']}")

    # Step 3: Build a test intake
    print("\n3. Building test intake...")
    intake = DomainIntake(
        domain_name="Ristoranti Milano Test",
        vertical=DomainVertical.FOOD_DINING,
        geographic_focus="Milano, Italia",
        language="it",
        target_audience_description="Food blogger, turisti, milanesi gourmet",
        audience_expertise_level="mixed",
        entity_types=[
            EntityTypeSpec(
                name="ristorante",
                description="Ristoranti di alta cucina e trattorie",
                example_entities=["Cracco", "Osteria Francescana", "Da Giacomo"],
            )
        ],
        quality_aspects=[
            QualityAspect(
                aspect_id="culinary_mastery",
                importance=5,
                what_makes_it_great="Tecniche impeccabili, creatività, piatti memorabili",
                what_makes_it_poor="Esecuzione sciatta, sapori piatti",
            ),
            QualityAspect(
                aspect_id="ingredient_quality",
                importance=5,
                what_makes_it_great="Prodotti di prima scelta, filiera corta",
                what_makes_it_poor="Ingredienti industriali, surgelati",
            ),
            QualityAspect(
                aspect_id="ambiance",
                importance=4,
                what_makes_it_great="Design curato, atmosfera unica",
                what_makes_it_poor="Arredamento anonimo, rumoroso",
            ),
            QualityAspect(aspect_id="service_excellence", importance=4),
            QualityAspect(aspect_id="price_value", importance=3),
        ],
        advisor_tone="warm_expert",
        sample_best_entities=["Cracco", "Seta"],
        sample_average_entities=["Trattoria tipica"],
    )

    # Step 4: Generate profile
    print("\n4. Generating profile (multi-pass validation)...")
    print("   This calls the LLM and may take 10-30 seconds...\n")

    try:
        draft = await service.generate_profile(intake)

        print(f"   Draft ID: {draft.draft_id}")
        print(f"   Quality Score: {draft.quality_score:.0f}/100")
        print(f"   Valid: {draft.is_valid}")
        print(f"   Rounds: {draft.generation_rounds}")

        if draft.validation:
            if draft.validation.errors:
                print(f"\n   Errors ({len(draft.validation.errors)}):")
                for e in draft.validation.errors[:3]:
                    print(f"     - [{e.field}] {e.message}")
            if draft.validation.warnings:
                print(f"\n   Warnings ({len(draft.validation.warnings)}):")
                for w in draft.validation.warnings[:3]:
                    print(f"     - [{w.field}] {w.message}")

        # Show generated dimensions
        dims = draft.profile.get("vibe_dimensions", [])
        if dims:
            print(f"\n   Generated {len(dims)} dimensions:")
            for d in dims[:5]:
                if isinstance(d, dict):
                    print(f"     - {d.get('id', '?')}: {d.get('label', '?')} (weight={d.get('weight', 1.0)})")

        print("\n" + "=" * 60)
        print("  TEST PASSED ✓" if draft.is_valid else "  TEST COMPLETED (with issues)")
        print("=" * 60)

    except Exception as exc:
        print(f"\n   ERROR: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_profile_generation())
