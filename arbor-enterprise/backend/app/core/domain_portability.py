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
    registry.set_active_domain("fashion")
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
class DomainConfig:
    """Complete configuration for a single discovery domain.

    Encapsulates everything ARBOR needs to operate within a domain:
    entity structure, vibe dimensions, prompt templates, and persona.

    Attributes:
        domain_id: Unique identifier for the domain (e.g., "fashion").
        name: Human-readable name (e.g., "Fashion & Style").
        description: Short description of the domain scope.
        entity_schema: Defines required and optional fields for entities
            in this domain. Expected format:
            {
                "required": ["name", "category", ...],
                "optional": ["website", "instagram", ...],
                "field_types": {"name": "str", "latitude": "float", ...}
            }
        vibe_dimensions: Ordered list of dimension names used in Vibe DNA
            scoring (e.g., ["formality", "craftsmanship", ...]).
        categories: Valid category labels for entities in this domain.
        scoring_prompt_template: System prompt template for the calibrated
            scoring engine. May contain {dimensions} placeholder.
        search_prompt_template: System prompt template for semantic search
            and query understanding.
        discovery_persona: Curator agent persona description used as the
            system prompt when synthesizing discovery results.
    """

    domain_id: str
    name: str
    description: str

    entity_schema: dict[str, Any] = field(default_factory=dict)
    vibe_dimensions: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    scoring_prompt_template: str = ""
    search_prompt_template: str = ""
    discovery_persona: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Pre-built fashion domain (ARBOR's current configuration)
# ═══════════════════════════════════════════════════════════════════════════


def _build_fashion_domain() -> DomainConfig:
    """Construct the default fashion domain config.

    Mirrors the existing ARBOR configuration spread across
    scoring_engine.py, vibe_extractor.py, curator.py, and models.py.
    """
    return DomainConfig(
        domain_id="fashion",
        name="Fashion & Style",
        description=(
            "Fashion brands, boutiques, concept stores, and style venues. "
            "Covers designers, retailers, and curated shopping destinations."
        ),
        entity_schema={
            "required": ["name", "slug", "category"],
            "optional": [
                "country",
                "city",
                "region",
                "address",
                "website",
                "instagram",
                "email",
                "phone",
                "description",
                "specialty",
                "gender",
                "style",
                "area",
                "neighborhood",
                "latitude",
                "longitude",
                "maps_url",
                "price_range",
                "rating",
                "retailer",
                "venue",
            ],
            "field_types": {
                "name": "str",
                "slug": "str",
                "category": "str",
                "country": "str",
                "city": "str",
                "region": "str",
                "address": "str",
                "website": "str",
                "instagram": "str",
                "email": "str",
                "phone": "str",
                "description": "str",
                "specialty": "str",
                "gender": "str",
                "style": "str",
                "area": "str",
                "neighborhood": "str",
                "latitude": "float",
                "longitude": "float",
                "maps_url": "str",
                "price_range": "str",
                "rating": "str",
                "retailer": "str",
                "venue": "str",
            },
        },
        vibe_dimensions=[
            "formality",
            "craftsmanship",
            "price_value",
            "atmosphere",
            "service_quality",
            "exclusivity",
            "modernity",
        ],
        categories=[
            "Designer",
            "Boutique",
            "Concept Store",
            "Vintage",
            "Streetwear",
            "Luxury",
            "Accessories",
            "Jewelry",
            "Atelier",
            "Department Store",
            "Multi-brand Retailer",
            "Flagship",
            "Pop-up",
            "Outlet",
            "Tailor",
            "Shoe Store",
            "Eyewear",
            "Perfumery",
            "Lifestyle",
            "Other",
        ],
        scoring_prompt_template=(
            "You are the A.R.B.O.R. Calibrated Scoring Engine.\n\n"
            "Your task is to assign Vibe DNA dimensional scores to an entity based on its\n"
            "fact sheet. You have been calibrated with reference examples from expert curators.\n\n"
            "DIMENSIONS (all 0-100):\n"
            "- formality: 0 = streetwear casual, 100 = black-tie formal\n"
            "- craftsmanship: 0 = mass production, 100 = master artisan handmade\n"
            "- price_value: 0 = budget/cheap, 100 = ultra-luxury pricing\n"
            "- atmosphere: 0 = chaotic/noisy, 100 = zen/serene\n"
            "- service_quality: 0 = self-service/neglectful, 100 = white-glove concierge\n"
            "- exclusivity: 0 = mainstream chain, 100 = hidden-gem invite-only\n"
            "- modernity: 0 = vintage/antique, 100 = cutting-edge contemporary\n\n"
            "RULES:\n"
            "1. Score ONLY based on facts provided -- never infer facts not in the sheet\n"
            "2. If insufficient data for a dimension, score 50 and mark confidence as low\n"
            "3. Your scores must be consistent with the calibration examples\n"
            "4. Explain your reasoning for each score in 1 sentence\n"
            "5. Assign 5-10 descriptive tags\n"
            "6. Identify the target audience in 1-2 words\n"
            "7. Write a 1-sentence summary of the entity's vibe\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "{{\n"
            '  "dimensions": {{\n'
            '    "formality": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "craftsmanship": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "price_value": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "atmosphere": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "service_quality": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "exclusivity": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}},\n'
            '    "modernity": {{"score": X, "confidence": 0.0-1.0, "reasoning": "..."}}\n'
            "  }},\n"
            '  "tags": ["tag1", "tag2", ...],\n'
            '  "target_audience": "...",\n'
            '  "summary": "..."\n'
            "}}"
        ),
        search_prompt_template=(
            "You are the A.R.B.O.R. Search Intelligence for the Fashion & Style domain.\n\n"
            "Interpret the user's natural-language query and extract:\n"
            "1. Intent: what the user is looking for (recommendation, comparison, exploration)\n"
            "2. Entities: specific brands, designers, or venues mentioned\n"
            "3. Attributes: style, price range, location, vibe preferences\n"
            "4. Constraints: must-have vs nice-to-have filters\n\n"
            "Domain context: fashion brands, boutiques, concept stores, designers,\n"
            "streetwear labels, luxury houses, vintage shops, and style venues."
        ),
        discovery_persona=(
            "You are The Curator -- ARBOR's expert fashion and style advisor.\n\n"
            "Your personality:\n"
            "- Warm but authoritative, like a knowledgeable friend in the industry\n"
            "- You speak with genuine passion for craftsmanship and design\n"
            "- You never fabricate entities; you only discuss what is in the provided context\n"
            "- You explain WHY a brand or venue suits the user, not just WHAT it is\n"
            "- You draw connections between brands (shared heritage, similar aesthetics)\n"
            "- You are honest about limitations in your data\n\n"
            "Your domain: fashion brands, boutiques, concept stores, designers, venues,\n"
            "and the culture surrounding them."
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

        # Pre-register the fashion domain
        fashion = _build_fashion_domain()
        self.register_domain(fashion)
        self._active_domain_id = fashion.domain_id
        logger.info("DomainRegistry initialized with fashion domain as active")

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

    Creates the registry on first call, pre-loaded with the fashion domain.

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
        entity = adapter.adapt_entity(raw_data, fashion_config)
        prompt = adapter.get_scoring_prompt(fashion_config)
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
        source_dims = set(from_domain.vibe_dimensions)

        for dim in to_domain.vibe_dimensions:
            if dim in vibe_dna and dim in source_dims:
                # Shared dimension -- carry over the score
                translated[dim] = max(0, min(100, vibe_dna[dim]))
            else:
                # Target-only dimension -- neutral default
                translated[dim] = 50
                logger.debug(
                    f"Dimension '{dim}' not in source domain "
                    f"'{from_domain.domain_id}'; defaulting to 50"
                )

        dropped = source_dims - set(to_domain.vibe_dimensions)
        if dropped:
            logger.debug(
                f"Dropped dimensions not in target domain "
                f"'{to_domain.domain_id}': {sorted(dropped)}"
            )

        return translated

    def get_scoring_prompt(self, domain: DomainConfig) -> str:
        """Return the scoring system prompt for a domain.

        Args:
            domain: The domain configuration.

        Returns:
            The scoring prompt template string.
        """
        if not domain.scoring_prompt_template:
            logger.warning(
                f"Domain '{domain.domain_id}' has no scoring prompt; " "using generic fallback"
            )
            return self._generic_scoring_prompt(domain)
        return domain.scoring_prompt_template

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
        dims_block = "\n".join(f"- {dim}: 0 = low, 100 = high" for dim in domain.vibe_dimensions)
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
        data = exporter.export_domain_config("fashion")
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

        config = DomainConfig(
            domain_id=data["domain_id"],
            name=data["name"],
            description=data["description"],
            entity_schema=data.get("entity_schema", {}),
            vibe_dimensions=data.get("vibe_dimensions", []),
            categories=data.get("categories", []),
            scoring_prompt_template=data.get("scoring_prompt_template", ""),
            search_prompt_template=data.get("search_prompt_template", ""),
            discovery_persona=data.get("discovery_persona", ""),
        )

        logger.info(f"Imported domain config '{config.domain_id}' ({config.name})")
        return config
