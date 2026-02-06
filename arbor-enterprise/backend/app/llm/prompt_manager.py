"""Prompt Version Management System.

TIER 10 - Point 54: Prompt Version Management (YAML Store)

Stores prompts in version-controlled YAML files with environment-based
rollout capabilities.

Features:
- YAML-based prompt storage
- Version tracking with checksums
- Environment-based rollout (canary → staging → production)
- A/B testing support
- Hot-reload without restart
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Directory containing prompt YAML files
PROMPTS_DIR = Path(__file__).parent / "prompts" / "versions"


@dataclass
class PromptVersion:
    """A versioned prompt template."""

    name: str
    version: str
    template: str
    description: str = ""
    environment: str = "development"  # development, staging, production
    checksum: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.variables:
            self.variables = self._extract_variables()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of template content."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def _extract_variables(self) -> list[str]:
        """Extract variable placeholders from template."""
        import re

        # Match {variable_name} patterns
        pattern = r"\{(\w+)\}"
        return list(set(re.findall(pattern, self.template)))

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")


@dataclass
class PromptRegistry:
    """Registry of all prompt versions.

    TIER 10 - Point 54: YAML-based prompt management.

    Usage:
        registry = PromptRegistry.load()
        prompt = registry.get("discovery_system", environment="production")
        rendered = prompt.render(category="restaurant", city="Milan")
    """

    prompts: dict[str, dict[str, PromptVersion]] = field(default_factory=dict)

    @classmethod
    def load(cls, prompts_dir: Path | None = None) -> "PromptRegistry":
        """Load all prompts from YAML files."""
        prompts_dir = prompts_dir or PROMPTS_DIR
        registry = cls()

        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return registry

        for yaml_file in prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data or "prompts" not in data:
                    continue

                for prompt_data in data["prompts"]:
                    version = PromptVersion(
                        name=prompt_data["name"],
                        version=prompt_data.get("version", "1.0.0"),
                        template=prompt_data["template"],
                        description=prompt_data.get("description", ""),
                        environment=prompt_data.get("environment", "development"),
                        variables=prompt_data.get("variables", []),
                        metadata=prompt_data.get("metadata", {}),
                    )

                    if version.name not in registry.prompts:
                        registry.prompts[version.name] = {}

                    registry.prompts[version.name][version.environment] = version

            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {len(registry.prompts)} prompt templates")
        return registry

    def get(
        self,
        name: str,
        environment: str | None = None,
        fallback_to_dev: bool = True,
    ) -> PromptVersion | None:
        """Get a prompt by name and environment.

        Args:
            name: Prompt name
            environment: Target environment (default: from APP_ENV)
            fallback_to_dev: If True, fall back to development if not found

        Returns:
            PromptVersion or None if not found
        """
        environment = environment or os.getenv("APP_ENV", "development")

        if name not in self.prompts:
            return None

        versions = self.prompts[name]

        # Try exact environment match
        if environment in versions:
            return versions[environment]

        # Fallback hierarchy: production → staging → development
        if fallback_to_dev:
            for env in ["staging", "development"]:
                if env in versions:
                    logger.debug(f"Prompt {name}: falling back to {env}")
                    return versions[env]

        return None

    def list_prompts(self) -> list[dict[str, Any]]:
        """List all available prompts with metadata."""
        result = []
        for name, versions in self.prompts.items():
            for env, prompt in versions.items():
                result.append(
                    {
                        "name": name,
                        "version": prompt.version,
                        "environment": env,
                        "checksum": prompt.checksum,
                        "variables": prompt.variables,
                    }
                )
        return result


# Singleton registry
_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get singleton PromptRegistry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry.load()
    return _registry


def reload_prompts() -> PromptRegistry:
    """Hot-reload prompts from disk."""
    global _registry
    _registry = PromptRegistry.load()
    logger.info("Prompts reloaded")
    return _registry


def get_prompt(
    name: str,
    environment: str | None = None,
    **render_kwargs,
) -> str:
    """Convenience function to get and render a prompt.

    Usage:
        prompt = get_prompt(
            "discovery_system",
            category="restaurant",
            user_query="romantic dinner spot",
        )
    """
    registry = get_prompt_registry()
    version = registry.get(name, environment)

    if not version:
        raise ValueError(f"Prompt not found: {name}")

    if render_kwargs:
        return version.render(**render_kwargs)

    return version.template


# Default prompts (fallback if YAML not found)
DEFAULT_PROMPTS = {
    "discovery_system": PromptVersion(
        name="discovery_system",
        version="1.0.0",
        template="""You are ARBOR, an AI assistant for discovering {category} experiences.

Your task is to analyze the user's query and provide personalized recommendations.

User Query: {user_query}
Location: {location}

Provide recommendations that match the user's intent and preferences.""",
        environment="development",
        description="Main discovery system prompt",
    ),
    "intent_classifier": PromptVersion(
        name="intent_classifier",
        version="1.0.0",
        template="""Classify the following query into one of these intents:
- DISCOVERY: Looking for recommendations
- INFORMATION: Asking about specific places
- COMPARISON: Comparing options
- NAVIGATION: Getting directions or contact info
- OTHER: Unrelated to lifestyle discovery

Query: {query}

Respond with only the intent name.""",
        environment="development",
        description="Query intent classification",
    ),
}


def _ensure_default_prompts():
    """Ensure default prompts are available."""
    registry = get_prompt_registry()

    for name, prompt in DEFAULT_PROMPTS.items():
        if name not in registry.prompts:
            registry.prompts[name] = {prompt.environment: prompt}


# Initialize defaults on import
_ensure_default_prompts()
