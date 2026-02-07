"""Domain Portability System for A.R.B.O.R. Enterprise.

Enables ARBOR to operate across multiple discovery domains (fashion, restaurants,
hotels, art galleries, etc.) by abstracting domain-specific configuration into
swappable DomainConfig objects.

Each domain defines:
- Entity schema (required/optional fields)
- Vibe dimensions (the axes of the "Vibe DNA" radar)
- Valid categories
- LLM prompt templates for scoring and search
- A discovery persona for the curator agent

Usage:
    registry = get_domain_registry()
    registry.set_active_domain("my_domain")
    config = registry.get_active_domain()

    adapter = DomainAdapter()
    prompt = adapter.get_scoring_prompt(config)
"""

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Domain Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VibeDimension:
    """A single axis of the Vibe DNA scoring radar.

    Each dimension defines what is being measured and how to interpret the
    0-100 scale, enabling the LLM scorer to produce calibrated, domain-
    specific scores regardless of the industry vertical.

    Attributes:
        id: Snake-case identifier used as the key in vibe_dna dicts
            (e.g., "culinary_mastery").
        label: Human-readable display name in the target language
            (e.g., "Maestria Culinaria").
        description: One-sentence explanation of what this dimension
            captures.
        low_label: What a score of 0 means (e.g., "Fast food, no technique").
        high_label: What a score of 100 means (e.g., "Michelin-level mastery").
        low_examples: Concrete examples of entities scoring near 0.
        high_examples: Concrete examples of entities scoring near 100.
        weight: Relative importance multiplier (0.5 - 2.0, default 1.0).
            Higher weight = this dimension counts more in similarity
            calculations and ranking.
    """

    id: str
    label: str
    description: str = ""
    low_label: str = "0 = low"
    high_label: str = "100 = high"
    low_examples: str = ""
    high_examples: str = ""
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VibeDimension":
        """Create from a JSON-parsed dictionary."""
        return cls(
            id=data["id"],
            label=data.get("label", data["id"].replace("_", " ").title()),
            description=data.get("description", ""),
            low_label=data.get("low_label", "0 = low"),
            high_label=data.get("high_label", "100 = high"),
            low_examples=data.get("low_examples", ""),
            high_examples=data.get("high_examples", ""),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class DomainConfig:
    """Complete configuration for a single discovery domain.

    Encapsulates everything ARBOR needs to operate within a domain:
    entity structure, vibe dimensions, prompt templates, and persona.

    Attributes:
        domain_id: Unique identifier for the domain (e.g., "default", "hospitality").
        name: Human-readable name (e.g., "Hospitality & Hotels").
        description: Short description of the domain scope.
        language: ISO 639-1 language code for user-facing output (e.g., "en", "it").
        target_audience: Who uses the discovery system (e.g., "food bloggers").
        entity_schema: Defines required and optional fields for entities
            in this domain. Expected format:
            {
                "required": ["name", "category", ...],
                "optional": ["website", "instagram", ...],
                "field_types": {"name": "str", "latitude": "float", ...}
            }
        vibe_dimensions: Ordered list of VibeDimension objects defining the
            scoring axes. Each carries its own scale descriptions and weight.
        categories: Valid category labels for entities in this domain.
        scoring_prompt_template: System prompt for the calibrated scoring
            engine.  Auto-generated from vibe_dimensions if left empty.
        search_prompt_template: System prompt for semantic search and query
            understanding.
        discovery_persona: Curator agent persona description used as the
            system prompt when synthesizing discovery results.
        search_context_keywords: Domain-specific keywords for NLP routing.
    """

    domain_id: str
    name: str
    description: str
    language: str = "en"
    target_audience: str = ""

    entity_schema: dict[str, Any] = field(default_factory=dict)
    vibe_dimensions: list[VibeDimension] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    scoring_prompt_template: str = ""
    search_prompt_template: str = ""
    discovery_persona: str = ""
    search_context_keywords: list[str] = field(default_factory=list)

    # --- Convenience helpers -------------------------------------------------

    @property
    def dimension_ids(self) -> list[str]:
        """Return ordered list of dimension ID strings (backward-compat)."""
        return [d.id for d in self.vibe_dimensions]

    def build_scoring_prompt(self) -> str:
        """Auto-generate a scoring prompt from the vibe_dimensions metadata.

        If ``scoring_prompt_template`` is already set, returns it as-is.
        Otherwise, constructs a complete prompt from the rich dimension
        descriptors so the LLM understands the exact 0-100 scale for this
        domain.
        """
        if self.scoring_prompt_template:
            return self.scoring_prompt_template

        dims_block = "\n".join(
            f"- {d.id} ({d.label}): {d.low_label} … {d.high_label}"
            + (f"\n  Examples low: {d.low_examples}" if d.low_examples else "")
            + (f"\n  Examples high: {d.high_examples}" if d.high_examples else "")
            for d in self.vibe_dimensions
        )
        dims_json = "\n".join(
            f'    "{d.id}": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}}'
            + ("," if i < len(self.vibe_dimensions) - 1 else "")
            for i, d in enumerate(self.vibe_dimensions)
        )

        return (
            "You are the A.R.B.O.R. Calibrated Scoring Engine.\n\n"
            "Your task is to assign Vibe DNA dimensional scores to an entity "
            "based on its fact sheet. You have been calibrated with reference "
            "examples from expert curators.\n\n"
            f"DIMENSIONS (all 0-100):\n{dims_block}\n\n"
            "RULES:\n"
            "1. Score ONLY based on facts provided — never infer facts not in the sheet\n"
            "2. If insufficient data for a dimension, score 50 and mark confidence as low\n"
            "3. Your scores must be consistent with the calibration examples\n"
            "4. Explain your reasoning for each score in 1 sentence\n"
            "5. Assign 5-10 descriptive tags\n"
            "6. Identify the target audience in 1-2 words\n"
            "7. Write a 1-sentence summary of the entity's vibe\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "{{\n"
            '  "dimensions": {{\n'
            f"{dims_json}\n"
            "  }},\n"
            '  "tags": ["tag1", "tag2", ...],\n'
            '  "target_audience": "...",\n'
            '  "summary": "..."\n'
            "}}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Default domain builder — reads entity schema from active configuration
# ═══════════════════════════════════════════════════════════════════════════

# Type hint for float-like fields detected from EntityTypeConfig mappings.
# When the source column name contains these keywords the field type is
# assumed to be "float"; everything else defaults to "str".
_FLOAT_FIELD_HINTS = {"latitude", "longitude", "priority", "rating"}


def _entity_schema_from_config() -> dict[str, Any]:
    """Derive entity_schema dynamically from the active EntityTypeConfig list.

    Merges required and optional mappings across all configured entity types
    so that the DomainConfig has a complete picture of available fields.
    """
    from app.config import get_settings

    settings = get_settings()
    configs = settings.get_entity_type_configs()

    required: set[str] = set()
    optional: set[str] = set()
    field_types: dict[str, str] = {}

    for cfg in configs:
        for arbor_field in cfg.required_mappings:
            required.add(arbor_field)
            field_types.setdefault(
                arbor_field,
                "float" if arbor_field in _FLOAT_FIELD_HINTS else "str",
            )
        for arbor_field in cfg.optional_mappings:
            if arbor_field not in required:
                optional.add(arbor_field)
            field_types.setdefault(
                arbor_field,
                "float" if arbor_field in _FLOAT_FIELD_HINTS else "str",
            )

    return {
        "required": sorted(required),
        "optional": sorted(optional - required),
        "field_types": field_types,
    }


def _build_default_domain() -> DomainConfig:
    """Construct a domain-neutral default config from the active schema.

    Entity schema is derived dynamically from EntityTypeConfig.
    Vibe dimensions use universal axes that apply reasonably to any
    discovery domain.  For optimal results, generate a domain-specific
    profile using ``DomainProfileGenerator``.
    """
    # Universal dimensions — intentionally abstract so they map to any domain.
    # A proper domain profile (generated or hand-crafted) replaces these.
    default_dimensions = [
        VibeDimension(
            id="quality",
            label="Quality",
            description="Overall quality of the entity's core offering",
            low_label="0 = poor quality, mass-market, no attention to detail",
            high_label="100 = exceptional quality, best-in-class, meticulous detail",
        ),
        VibeDimension(
            id="price_positioning",
            label="Price Positioning",
            description="Where the entity sits on the price spectrum",
            low_label="0 = budget, lowest price tier",
            high_label="100 = ultra-premium, highest price tier",
        ),
        VibeDimension(
            id="experience",
            label="Experience",
            description="The overall experience and atmosphere provided",
            low_label="0 = purely functional, no experiential value",
            high_label="100 = immersive, memorable, transformative experience",
        ),
        VibeDimension(
            id="uniqueness",
            label="Uniqueness",
            description="How distinctive and differentiated the entity is",
            low_label="0 = generic, interchangeable with competitors",
            high_label="100 = one-of-a-kind, impossible to replicate",
        ),
        VibeDimension(
            id="accessibility",
            label="Accessibility",
            description="How easy it is to discover, reach, and engage with",
            low_label="0 = hidden, invite-only, hard to access",
            high_label="100 = highly visible, easy to find and use",
        ),
    ]

    return DomainConfig(
        domain_id="default",
        name="Default Domain",
        description=(
            "Auto-configured domain with universal scoring dimensions. "
            "For optimal results, generate a domain-specific profile using "
            "the Domain Profile Generator."
        ),
        entity_schema=_entity_schema_from_config(),
        vibe_dimensions=default_dimensions,
        categories=[],
        # scoring_prompt_template left empty — DomainConfig.build_scoring_prompt()
        # auto-generates it from the VibeDimension metadata above.
        scoring_prompt_template="",
        search_prompt_template=(
            "You are the A.R.B.O.R. Search Intelligence.\n\n"
            "Interpret the user's natural-language query and extract:\n"
            "1. Intent: what the user is looking for (recommendation, comparison, exploration)\n"
            "2. Entities: specific entities mentioned by name\n"
            "3. Attributes: preferences, constraints, and desired characteristics\n"
            "4. Constraints: must-have vs nice-to-have filters\n\n"
            "Answer based ONLY on the provided context."
        ),
        discovery_persona=(
            "You are The Curator — ARBOR's expert discovery advisor.\n\n"
            "Your personality:\n"
            "- Warm but authoritative, like a knowledgeable friend\n"
            "- You speak with genuine passion about the entities you recommend\n"
            "- You never fabricate entities; you only discuss what is in the provided context\n"
            "- You explain WHY an entity suits the user, not just WHAT it is\n"
            "- You draw connections between entities (shared traits, complementary qualities)\n"
            "- You are honest about limitations in your data"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Domain Registry (singleton)
# ═══════════════════════════════════════════════════════════════════════════


class DomainRegistry:
    """In-memory registry of domain configurations.

    Maintains a mapping of domain_id -> DomainConfig and tracks which
    domain is currently active.  Accessed via the ``get_domain_registry()``
    singleton factory.

    Usage:
        registry = get_domain_registry()
        registry.register_domain(my_config)
        registry.set_active_domain("my_domain")
        config = registry.get_active_domain()
    """

    def __init__(self) -> None:
        self._domains: dict[str, DomainConfig] = {}
        self._active_domain_id: str | None = None

        # Try loading a custom domain profile from ENV/file first
        from app.config import get_settings

        profile_data = get_settings().get_domain_profile()
        if profile_data:
            exporter = DomainExporter()
            custom = exporter.import_domain_config(profile_data)
            self.register_domain(custom)
            self._active_domain_id = custom.domain_id
            logger.info(
                f"DomainRegistry initialized with custom profile "
                f"'{custom.domain_id}' as active"
            )
        else:
            # Fall back to universal defaults derived from schema config
            default = _build_default_domain()
            self.register_domain(default)
            self._active_domain_id = default.domain_id
            logger.info("DomainRegistry initialized with default domain as active")

    def register_domain(self, config: DomainConfig) -> None:
        """Register a new domain configuration.

        Args:
            config: The domain configuration to register.

        Raises:
            ValueError: If ``config.domain_id`` is empty.
        """
        if not config.domain_id:
            raise ValueError("domain_id must be a non-empty string")

        overwriting = config.domain_id in self._domains
        self._domains[config.domain_id] = config

        action = "updated" if overwriting else "registered"
        logger.info(
            f"Domain '{config.domain_id}' ({config.name}) {action} "
            f"({len(config.vibe_dimensions)} dimensions, "
            f"{len(config.categories)} categories)"
        )

    def get_domain(self, domain_id: str) -> DomainConfig:
        """Retrieve a domain configuration by ID.

        Args:
            domain_id: The unique domain identifier.

        Returns:
            The matching DomainConfig.

        Raises:
            KeyError: If no domain with this ID is registered.
        """
        if domain_id not in self._domains:
            available = ", ".join(sorted(self._domains.keys())) or "(none)"
            raise KeyError(f"Domain '{domain_id}' not found. Available domains: {available}")
        return self._domains[domain_id]

    def list_domains(self) -> list[DomainConfig]:
        """Return all registered domain configurations.

        Returns:
            A list of all DomainConfig objects, sorted by domain_id.
        """
        return [self._domains[k] for k in sorted(self._domains.keys())]

    def get_active_domain(self) -> DomainConfig:
        """Return the currently active domain configuration.

        Returns:
            The active DomainConfig.

        Raises:
            RuntimeError: If no active domain has been set.
        """
        if self._active_domain_id is None:
            raise RuntimeError("No active domain set. Call set_active_domain() first.")
        return self.get_domain(self._active_domain_id)

    def set_active_domain(self, domain_id: str) -> None:
        """Set the active domain by ID.

        Args:
            domain_id: The domain to activate.

        Raises:
            KeyError: If no domain with this ID is registered.
        """
        # Validate the domain exists before switching
        self.get_domain(domain_id)
        previous = self._active_domain_id
        self._active_domain_id = domain_id
        logger.info(f"Active domain changed: '{previous}' -> '{domain_id}'")


# Singleton instance
_registry: DomainRegistry | None = None


def get_domain_registry() -> DomainRegistry:
    """Return the singleton DomainRegistry instance.

    Creates the registry on first call, pre-loaded with the default domain
    derived from the active SOURCE_SCHEMA_CONFIG.

    Returns:
        The global DomainRegistry.
    """
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


# ═══════════════════════════════════════════════════════════════════════════
# Domain Adapter
# ═══════════════════════════════════════════════════════════════════════════


class DomainAdapter:
    """Adapts data and prompts between domains.

    Provides utilities for mapping raw entity data to a domain schema,
    translating vibe dimensions between domains, generating domain-
    specific prompts, and validating entity data.

    Usage:
        adapter = DomainAdapter()
        entity = adapter.adapt_entity(raw_data, my_config)
        prompt = adapter.get_scoring_prompt(my_config)
    """

    def adapt_entity(self, raw_data: dict[str, Any], domain: DomainConfig) -> dict[str, Any]:
        """Map raw data to the domain's entity schema.

        Extracts fields defined in the domain schema from ``raw_data``,
        applies type coercion where possible, and returns a clean dict.

        Args:
            raw_data: Unstructured entity data (e.g., from a scraper).
            domain: The target domain configuration.

        Returns:
            A dict containing only the fields recognised by the domain
            schema, with values coerced to declared types.
        """
        schema = domain.entity_schema
        all_fields = set(schema.get("required", []) + schema.get("optional", []))
        field_types = schema.get("field_types", {})

        adapted: dict[str, Any] = {}
        for field_name in all_fields:
            if field_name not in raw_data:
                continue
            value = raw_data[field_name]
            if value is None:
                adapted[field_name] = None
                continue

            # Coerce to declared type if specified
            expected_type = field_types.get(field_name)
            adapted[field_name] = self._coerce_value(value, expected_type, field_name)

        return adapted

    def adapt_vibe_dimensions(
        self,
        vibe_dna: dict[str, int],
        from_domain: DomainConfig,
        to_domain: DomainConfig,
    ) -> dict[str, int]:
        """Translate vibe dimensions from one domain to another.

        Dimensions that exist in both domains are carried over as-is.
        Dimensions unique to the target domain receive a neutral default
        score of 50.  Dimensions that exist only in the source domain
        are dropped.

        Args:
            vibe_dna: Dimension name -> score (0-100) mapping.
            from_domain: The source domain configuration.
            to_domain: The target domain configuration.

        Returns:
            A new dict with dimensions aligned to ``to_domain``.
        """
        translated: dict[str, int] = {}
        source_dim_ids = set(from_domain.dimension_ids)

        for dim in to_domain.vibe_dimensions:
            if dim.id in vibe_dna and dim.id in source_dim_ids:
                # Shared dimension — carry over the score
                translated[dim.id] = max(0, min(100, vibe_dna[dim.id]))
            else:
                # Target-only dimension — neutral default
                translated[dim.id] = 50
                logger.debug(
                    f"Dimension '{dim.id}' not in source domain "
                    f"'{from_domain.domain_id}'; defaulting to 50"
                )

        dropped = source_dim_ids - set(to_domain.dimension_ids)
        if dropped:
            logger.debug(
                f"Dropped dimensions not in target domain "
                f"'{to_domain.domain_id}': {sorted(dropped)}"
            )

        return translated

    def get_scoring_prompt(self, domain: DomainConfig) -> str:
        """Return the scoring system prompt for a domain.

        Uses ``domain.build_scoring_prompt()`` which auto-generates the
        prompt from VibeDimension metadata if no explicit template is set.

        Args:
            domain: The domain configuration.

        Returns:
            The scoring prompt template string.
        """
        return domain.build_scoring_prompt()

    def get_discovery_persona(self, domain: DomainConfig) -> str:
        """Return the curator/discovery persona for a domain.

        Args:
            domain: The domain configuration.

        Returns:
            The persona description string.
        """
        if not domain.discovery_persona:
            logger.warning(
                f"Domain '{domain.domain_id}' has no discovery persona; " "using generic fallback"
            )
            return (
                f"You are The Curator -- an expert advisor for the "
                f"{domain.name} domain. Answer based ONLY on provided "
                f"context. Never fabricate entities. Be warm but "
                f"professional."
            )
        return domain.discovery_persona

    def validate_entity(
        self, entity_data: dict[str, Any], domain: DomainConfig
    ) -> tuple[bool, list[str]]:
        """Validate entity data against a domain's entity schema.

        Checks that all required fields are present and non-empty, and
        that field values match their declared types where possible.

        Args:
            entity_data: The entity data dict to validate.
            domain: The domain configuration with the target schema.

        Returns:
            A tuple of (is_valid, errors) where ``errors`` is a list of
            human-readable validation failure descriptions.
        """
        schema = domain.entity_schema
        errors: list[str] = []

        # Check required fields
        for req_field in schema.get("required", []):
            value = entity_data.get(req_field)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"Missing required field: '{req_field}'")

        # Type checking for present fields
        field_types = schema.get("field_types", {})
        for field_name, value in entity_data.items():
            if value is None:
                continue
            expected_type = field_types.get(field_name)
            if expected_type and not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field_name}' expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )

        # Validate category if categories are defined
        if domain.categories and "category" in entity_data:
            cat = entity_data["category"]
            if cat and cat not in domain.categories:
                errors.append(
                    f"Invalid category '{cat}'. " f"Valid categories: {domain.categories}"
                )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.debug(
                f"Entity validation failed for domain '{domain.domain_id}': "
                f"{len(errors)} error(s)"
            )
        return is_valid, errors

    # --- Private helpers ---------------------------------------------------

    @staticmethod
    def _coerce_value(value: Any, expected_type: str | None, field_name: str) -> Any:
        """Attempt to coerce a value to the expected type.

        Returns the original value if coercion fails or no type is specified.
        """
        if expected_type is None:
            return value

        try:
            if expected_type == "str":
                return str(value)
            elif expected_type == "int":
                return int(value)
            elif expected_type == "float":
                return float(value)
            elif expected_type == "bool":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
        except (ValueError, TypeError):
            logger.debug(
                f"Could not coerce field '{field_name}' value " f"{value!r} to {expected_type}"
            )
        return value

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check whether a value matches the expected type string."""
        type_map: dict[str, type | tuple[type, ...]] = {
            "str": str,
            "int": (int,),
            "float": (int, float),
            "bool": bool,
        }
        allowed = type_map.get(expected_type)
        if allowed is None:
            # Unknown type spec -- pass validation
            return True
        return isinstance(value, allowed)

    @staticmethod
    def _generic_scoring_prompt(domain: DomainConfig) -> str:
        """Build a minimal scoring prompt from the domain's dimensions."""
        dims_block = "\n".join(
            f"- {dim.id} ({dim.label}): {dim.low_label} … {dim.high_label}"
            for dim in domain.vibe_dimensions
        )
        return (
            f"You are a calibrated scoring engine for the {domain.name} domain.\n\n"
            f"Assign scores (0-100) for each dimension based on the provided fact sheet.\n\n"
            f"DIMENSIONS:\n{dims_block}\n\n"
            f"Output JSON with 'dimensions', 'tags', 'target_audience', and 'summary'."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Domain Exporter / Importer
# ═══════════════════════════════════════════════════════════════════════════


class DomainExporter:
    """Serialize and deserialize DomainConfig objects for persistence.

    Enables exporting domain configurations to JSON-compatible dicts
    (for storage, API responses, or file-based exchange) and importing
    them back.

    Usage:
        exporter = DomainExporter()
        data = exporter.export_domain_config("default")
        json.dumps(data)  # ready for storage

        config = exporter.import_domain_config(data)
        registry.register_domain(config)
    """

    def export_domain_config(self, domain_id: str) -> dict[str, Any]:
        """Serialize a registered domain config to a JSON-compatible dict.

        Args:
            domain_id: The ID of the domain to export.

        Returns:
            A plain dict representation of the DomainConfig, safe for
            ``json.dumps``.

        Raises:
            KeyError: If no domain with this ID is registered.
        """
        registry = get_domain_registry()
        config = registry.get_domain(domain_id)
        exported = asdict(config)
        logger.info(f"Exported domain config '{domain_id}' " f"({len(json.dumps(exported))} bytes)")
        return exported

    def import_domain_config(self, config_dict: dict[str, Any]) -> DomainConfig:
        """Deserialize a dict into a DomainConfig instance.

        Performs basic validation to ensure required fields are present.

        Args:
            config_dict: A dict previously produced by ``export_domain_config``
                or constructed manually.

        Returns:
            A new DomainConfig instance.

        Raises:
            ValueError: If required keys are missing from ``config_dict``.
        """
        # Work on a deep copy to avoid mutating the caller's data
        data = copy.deepcopy(config_dict)

        required_keys = {"domain_id", "name", "description"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"Cannot import domain config: missing required keys " f"{sorted(missing)}"
            )

        # Parse vibe_dimensions — accept both rich dicts and plain strings
        raw_dims = data.get("vibe_dimensions", [])
        parsed_dims: list[VibeDimension] = []
        for item in raw_dims:
            if isinstance(item, dict):
                parsed_dims.append(VibeDimension.from_dict(item))
            elif isinstance(item, str):
                # Legacy format: just a dimension name string
                parsed_dims.append(VibeDimension(id=item, label=item.replace("_", " ").title()))
            elif isinstance(item, VibeDimension):
                parsed_dims.append(item)

        config = DomainConfig(
            domain_id=data["domain_id"],
            name=data["name"],
            description=data["description"],
            language=data.get("language", "en"),
            target_audience=data.get("target_audience", ""),
            entity_schema=data.get("entity_schema", {}),
            vibe_dimensions=parsed_dims,
            categories=data.get("categories", []),
            scoring_prompt_template=data.get("scoring_prompt_template", ""),
            search_prompt_template=data.get("search_prompt_template", ""),
            discovery_persona=data.get("discovery_persona", ""),
            search_context_keywords=data.get("search_context_keywords", []),
        )

        logger.info(f"Imported domain config '{config.domain_id}' ({config.name})")
        return config
