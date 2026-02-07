"""Domain Profile Validation Engine for A.R.B.O.R. Enterprise.

Validates generated domain profiles both structurally (JSON schema) and
semantically (dimension quality, coherence, coverage).  Used by the
``DomainProfileService`` in the multi-pass generation loop to ensure
near-zero error rates before a profile is activated.

Validation layers:
  1. Structural — required keys, types, value ranges
  2. Dimensional — dimension quality, distinctness, weight distribution
  3. Semantic — LLM-based coherence check (optional, expensive)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Validation result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationIssue:
    """A single validation issue found in a domain profile."""

    severity: str  # "error" (blocks activation), "warning" (fixable), "info"
    field: str  # Which part of the profile is affected
    message: str  # Human-readable description
    auto_fixable: bool = False  # Can the system fix this automatically?
    fix_instruction: str = ""  # Instruction for the LLM to fix it


@dataclass
class ValidationResult:
    """Aggregate result of validating a domain profile."""

    is_valid: bool = True
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    score: float = 0.0  # 0-100 quality score

    @property
    def all_issues(self) -> list[ValidationIssue]:
        return self.errors + self.warnings + self.info

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def auto_fix_instructions(self) -> list[str]:
        """Collect fix instructions for all auto-fixable issues."""
        return [
            issue.fix_instruction
            for issue in self.all_issues
            if issue.auto_fixable and issue.fix_instruction
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "score": round(self.score, 1),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [
                {"field": e.field, "message": e.message, "auto_fixable": e.auto_fixable}
                for e in self.errors
            ],
            "warnings": [
                {"field": w.field, "message": w.message, "auto_fixable": w.auto_fixable}
                for w in self.warnings
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Validator
# ═══════════════════════════════════════════════════════════════════════════

# Valid dimension ID pattern: lowercase snake_case
_DIM_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,48}$")

# Minimum meaningful length for text fields
_MIN_LABEL_LEN = 3
_MIN_DESCRIPTION_LEN = 10
_MIN_EXAMPLE_LEN = 10
_MIN_PERSONA_LEN = 50
_MIN_SEARCH_PROMPT_LEN = 30


class ProfileValidator:
    """Validates a domain profile dict for structural and semantic correctness.

    Usage:
        validator = ProfileValidator()
        result = validator.validate(profile_dict)
        if not result.is_valid:
            # Fix issues and retry
            ...
    """

    def validate(self, profile: dict[str, Any]) -> ValidationResult:
        """Run all validation layers on a profile dict.

        Args:
            profile: The raw profile dict (as parsed from JSON).

        Returns:
            A ValidationResult with all issues found.
        """
        result = ValidationResult()

        self._check_structure(profile, result)
        if result.has_errors:
            # Don't bother with deeper checks if structure is broken
            result.is_valid = False
            result.score = 0.0
            return result

        self._check_dimensions(profile, result)
        self._check_categories(profile, result)
        self._check_prompts(profile, result)
        self._check_keywords(profile, result)
        self._check_metadata(profile, result)

        # Calculate score
        result.score = self._calculate_score(profile, result)
        result.is_valid = not result.has_errors

        return result

    # --- Layer 1: Structural validation -----------------------------------

    def _check_structure(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Verify required keys, types, and basic structure."""

        # Required top-level keys
        required_keys = {
            "domain_id": str,
            "name": str,
            "description": str,
            "vibe_dimensions": list,
            "categories": list,
        }

        for key, expected_type in required_keys.items():
            if key not in profile:
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=key,
                        message=f"Missing required key '{key}'",
                        auto_fixable=False,
                    )
                )
            elif not isinstance(profile[key], expected_type):
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=key,
                        message=(
                            f"'{key}' must be {expected_type.__name__}, "
                            f"got {type(profile[key]).__name__}"
                        ),
                        auto_fixable=False,
                    )
                )

        # domain_id format
        domain_id = profile.get("domain_id", "")
        if domain_id and not _DIM_ID_PATTERN.match(domain_id):
            result.errors.append(
                ValidationIssue(
                    severity="error",
                    field="domain_id",
                    message=(
                        f"domain_id '{domain_id}' must be lowercase snake_case "
                        f"(letters, digits, underscores)"
                    ),
                    auto_fixable=True,
                    fix_instruction=(
                        f"Fix the domain_id: convert '{domain_id}' to valid "
                        f"lowercase snake_case (e.g., 'fine_dining_milano')"
                    ),
                )
            )

        # Language code
        language = profile.get("language", "")
        if language and (len(language) < 2 or len(language) > 5):
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="language",
                    message=f"Language '{language}' doesn't look like a valid ISO code",
                    auto_fixable=True,
                    fix_instruction=f"Fix the language field: use a 2-letter ISO 639-1 code",
                )
            )

        # Vibe dimensions count
        dims = profile.get("vibe_dimensions", [])
        if isinstance(dims, list):
            if len(dims) < 3:
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field="vibe_dimensions",
                        message=f"Too few dimensions ({len(dims)}). Minimum is 3.",
                        auto_fixable=True,
                        fix_instruction=(
                            "Add more vibe dimensions. A good profile has 5-8 "
                            "dimensions that cover different aspects of quality "
                            "in this domain."
                        ),
                    )
                )
            elif len(dims) > 12:
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        field="vibe_dimensions",
                        message=(
                            f"Too many dimensions ({len(dims)}). "
                            f"Consider merging related ones. Ideal: 5-8."
                        ),
                        auto_fixable=True,
                        fix_instruction=(
                            f"Reduce dimensions from {len(dims)} to 5-8 by "
                            f"merging closely related ones."
                        ),
                    )
                )

    # --- Layer 2: Dimension quality validation ----------------------------

    def _check_dimensions(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Validate individual dimension quality and cross-dimension properties."""
        dims = profile.get("vibe_dimensions", [])
        if not isinstance(dims, list):
            return

        seen_ids: set[str] = set()
        total_weight = 0.0

        for i, dim in enumerate(dims):
            if not isinstance(dim, dict):
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=f"vibe_dimensions[{i}]",
                        message=f"Dimension at index {i} must be a dict, got {type(dim).__name__}",
                    )
                )
                continue

            dim_id = dim.get("id", "")
            prefix = f"vibe_dimensions[{i}] ({dim_id or '?'})"

            # ID validation
            if not dim_id:
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=prefix,
                        message="Dimension missing 'id'",
                    )
                )
            elif not _DIM_ID_PATTERN.match(dim_id):
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=prefix,
                        message=f"Dimension id '{dim_id}' must be lowercase snake_case",
                        auto_fixable=True,
                        fix_instruction=(
                            f"Fix dimension id '{dim_id}': convert to valid "
                            f"lowercase snake_case"
                        ),
                    )
                )

            # Duplicate ID check
            if dim_id in seen_ids:
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=prefix,
                        message=f"Duplicate dimension id '{dim_id}'",
                        auto_fixable=True,
                        fix_instruction=(
                            f"Remove or rename duplicate dimension '{dim_id}'. "
                            f"Each dimension must have a unique id."
                        ),
                    )
                )
            seen_ids.add(dim_id)

            # Label validation
            label = dim.get("label", "")
            if not label or len(str(label)) < _MIN_LABEL_LEN:
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=f"{prefix}.label",
                        message="Dimension missing or too short 'label'",
                        auto_fixable=True,
                        fix_instruction=(
                            f"Add a meaningful label for dimension '{dim_id}' "
                            f"(at least {_MIN_LABEL_LEN} characters)"
                        ),
                    )
                )

            # Description validation
            desc = dim.get("description", "")
            if not desc or len(str(desc)) < _MIN_DESCRIPTION_LEN:
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        field=f"{prefix}.description",
                        message="Dimension description is missing or too short",
                        auto_fixable=True,
                        fix_instruction=(
                            f"Add a clear description for dimension '{dim_id}' "
                            f"explaining what it measures (at least {_MIN_DESCRIPTION_LEN} chars)"
                        ),
                    )
                )

            # Low/high label validation
            for endpoint in ("low_label", "high_label"):
                val = dim.get(endpoint, "")
                if not val or len(str(val)) < _MIN_LABEL_LEN:
                    result.warnings.append(
                        ValidationIssue(
                            severity="warning",
                            field=f"{prefix}.{endpoint}",
                            message=f"Dimension '{endpoint}' is missing or too short",
                            auto_fixable=True,
                            fix_instruction=(
                                f"Add a descriptive '{endpoint}' for dimension '{dim_id}' "
                                f"that concretely describes what a score of "
                                f"{'0' if endpoint == 'low_label' else '100'} means"
                            ),
                        )
                    )

            # Examples validation (warning only — nice to have)
            for example_field in ("low_examples", "high_examples"):
                val = dim.get(example_field, "")
                if not val or len(str(val)) < _MIN_EXAMPLE_LEN:
                    result.info.append(
                        ValidationIssue(
                            severity="info",
                            field=f"{prefix}.{example_field}",
                            message=f"Consider adding '{example_field}' for better calibration",
                        )
                    )

            # Weight validation
            weight = dim.get("weight", 1.0)
            try:
                weight = float(weight)
                if weight < 0.1 or weight > 5.0:
                    result.warnings.append(
                        ValidationIssue(
                            severity="warning",
                            field=f"{prefix}.weight",
                            message=f"Weight {weight} is outside recommended range (0.5-2.0)",
                            auto_fixable=True,
                            fix_instruction=(
                                f"Adjust weight for dimension '{dim_id}' to be "
                                f"between 0.5 and 2.0"
                            ),
                        )
                    )
                total_weight += weight
            except (TypeError, ValueError):
                result.errors.append(
                    ValidationIssue(
                        severity="error",
                        field=f"{prefix}.weight",
                        message=f"Weight '{weight}' is not a valid number",
                        auto_fixable=True,
                        fix_instruction=f"Set weight for dimension '{dim_id}' to 1.0",
                    )
                )

        # Cross-dimension checks
        if len(seen_ids) >= 3:
            avg_weight = total_weight / len(seen_ids) if seen_ids else 1.0
            if avg_weight > 2.0:
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        field="vibe_dimensions",
                        message=(
                            f"Average weight ({avg_weight:.1f}) is very high. "
                            f"Weights should average around 1.0."
                        ),
                        auto_fixable=True,
                        fix_instruction="Normalize dimension weights so they average around 1.0",
                    )
                )

    # --- Layer 3: Categories validation -----------------------------------

    def _check_categories(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Validate category list."""
        categories = profile.get("categories", [])
        if not isinstance(categories, list):
            return

        if len(categories) < 2:
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="categories",
                    message=f"Only {len(categories)} categories defined. Consider adding more.",
                    auto_fixable=True,
                    fix_instruction=(
                        "Add more categories that represent the main types "
                        "of entities in this domain. Aim for 5-20 categories."
                    ),
                )
            )
        elif len(categories) > 50:
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="categories",
                    message=f"Too many categories ({len(categories)}). Consider consolidating.",
                    auto_fixable=True,
                    fix_instruction=(
                        f"Reduce categories from {len(categories)} to under 30 "
                        f"by merging similar ones."
                    ),
                )
            )

        # Check for duplicates
        seen = set()
        for cat in categories:
            normalized = str(cat).strip().lower()
            if normalized in seen:
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        field="categories",
                        message=f"Duplicate category: '{cat}'",
                        auto_fixable=True,
                        fix_instruction=f"Remove the duplicate category '{cat}'",
                    )
                )
            seen.add(normalized)

        # Check for empty strings
        if any(not str(c).strip() for c in categories):
            result.errors.append(
                ValidationIssue(
                    severity="error",
                    field="categories",
                    message="Categories list contains empty strings",
                    auto_fixable=True,
                    fix_instruction="Remove empty strings from the categories list",
                )
            )

    # --- Layer 4: Prompts and persona validation --------------------------

    def _check_prompts(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Validate discovery persona and search prompt."""

        persona = profile.get("discovery_persona", "")
        if not persona or len(str(persona)) < _MIN_PERSONA_LEN:
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="discovery_persona",
                    message="Discovery persona is missing or too short",
                    auto_fixable=True,
                    fix_instruction=(
                        "Write a discovery persona (4-6 lines) that describes "
                        "the AI advisor's personality, tone, and expertise for "
                        "this domain."
                    ),
                )
            )

        search_prompt = profile.get("search_prompt_template", profile.get("search_prompt", ""))
        if not search_prompt or len(str(search_prompt)) < _MIN_SEARCH_PROMPT_LEN:
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="search_prompt_template",
                    message="Search prompt template is missing or too short",
                    auto_fixable=True,
                    fix_instruction=(
                        "Write a search prompt that instructs the AI how to "
                        "interpret user queries in this domain."
                    ),
                )
            )

    # --- Layer 5: Keywords validation -------------------------------------

    def _check_keywords(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Validate search context keywords."""
        keywords = profile.get("search_context_keywords", [])

        if not keywords or len(keywords) < 5:
            result.warnings.append(
                ValidationIssue(
                    severity="warning",
                    field="search_context_keywords",
                    message=f"Only {len(keywords)} keywords. Recommend 10-20.",
                    auto_fixable=True,
                    fix_instruction=(
                        "Add more domain-specific search keywords (10-20) "
                        "that users might use when searching for entities."
                    ),
                )
            )

    # --- Layer 6: Metadata validation -------------------------------------

    def _check_metadata(self, profile: dict[str, Any], result: ValidationResult) -> None:
        """Validate target audience and other metadata."""
        audience = profile.get("target_audience", "")
        if not audience or len(str(audience)) < 5:
            result.info.append(
                ValidationIssue(
                    severity="info",
                    field="target_audience",
                    message="Consider specifying the target audience for better results",
                )
            )

    # --- Score calculation -------------------------------------------------

    def _calculate_score(self, profile: dict[str, Any], result: ValidationResult) -> float:
        """Calculate a quality score (0-100) for the profile."""
        score = 100.0

        # Deductions
        score -= len(result.errors) * 20  # Each error = -20
        score -= len(result.warnings) * 5  # Each warning = -5
        score -= len(result.info) * 1  # Each info = -1

        # Bonuses for completeness
        dims = profile.get("vibe_dimensions", [])
        if isinstance(dims, list) and 5 <= len(dims) <= 8:
            score += 5  # Ideal dimension count

        # Check dimension completeness
        complete_dims = 0
        for dim in dims if isinstance(dims, list) else []:
            if isinstance(dim, dict):
                has_all = all(
                    dim.get(f) and len(str(dim.get(f, ""))) >= 5
                    for f in ("id", "label", "description", "low_label", "high_label")
                )
                if has_all:
                    complete_dims += 1

        if isinstance(dims, list) and dims:
            completeness_ratio = complete_dims / len(dims)
            score += completeness_ratio * 10  # Up to +10 for complete dims

        # Bonus for examples
        dims_with_examples = 0
        for dim in dims if isinstance(dims, list) else []:
            if isinstance(dim, dict):
                if dim.get("low_examples") and dim.get("high_examples"):
                    dims_with_examples += 1
        if isinstance(dims, list) and dims:
            example_ratio = dims_with_examples / len(dims)
            score += example_ratio * 5  # Up to +5 for examples

        # Bonus for keywords
        keywords = profile.get("search_context_keywords", [])
        if 10 <= len(keywords) <= 25:
            score += 3

        # Bonus for categories
        categories = profile.get("categories", [])
        if 5 <= len(categories) <= 25:
            score += 3

        return max(0.0, min(100.0, score))
