"""Prompt loader for A.R.B.O.R. system prompts."""

import os
from functools import lru_cache
from pathlib import Path

# Base path for prompt files
PROMPTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "config" / "prompts"


@lru_cache(maxsize=20)
def load_prompt(name: str) -> str:
    """Load a prompt template from the config/prompts directory.

    Args:
        name: Prompt filename without extension (e.g., 'curator_persona')

    Returns:
        The prompt text content.
    """
    filepath = PROMPTS_DIR / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    return filepath.read_text(encoding="utf-8").strip()


def load_all_prompts() -> dict[str, str]:
    """Load all available prompts into a dictionary."""
    prompts = {}
    if PROMPTS_DIR.exists():
        for filepath in PROMPTS_DIR.glob("*.txt"):
            name = filepath.stem
            prompts[name] = filepath.read_text(encoding="utf-8").strip()
    return prompts


# Pre-load commonly used prompts
CURATOR_PERSONA = "curator_persona"
VIBE_EXTRACTOR = "vibe_extractor"
INTENT_CLASSIFIER = "intent_classifier"
CYPHER_GENERATOR = "cypher_generator"
