"""Feature Store and Model Registry for the ARBOR Enterprise ML pipeline.

Centralises feature engineering, storage, lineage tracking, and model
lifecycle management so that every downstream consumer (personalization,
drift detection, A/B testing, competitive intelligence) operates on a
single, versioned set of features.

Feature Store
-------------
* Register features with optional compute functions and dependency graphs.
* Compute, store, and retrieve feature vectors per entity.
* Built-in features derived from the ARBOR enrichment schema (vibe_dna,
  category, tags, embeddings, social links).

Model Registry
--------------
* Register models with typed metadata.
* Version models with metrics, parameters, and feature requirements.
* Promote through staging -> canary -> production lifecycle.
* Roll back to the previous production version with one call.

Both subsystems expose a module-level singleton accessor
(:func:`get_feature_store`, :func:`get_model_registry`) following the
same pattern used by the personalization engine and A/B testing service.

Usage::

    store = get_feature_store()
    vector = store.compute_features("entity_42", entity_data)

    registry = get_model_registry()
    registry.register_model("vibe_scorer", "gradient_boost", "Scores vibe dimensions")
    version = registry.log_version(
        "vibe_scorer",
        metrics={"rmse": 0.12},
        parameters={"n_estimators": 200},
        feature_requirements=["vibe_formality", "vibe_craftsmanship"],
    )
    registry.promote("vibe_scorer", version.version, "production")
"""

import copy
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# ARBOR vibe dimensions — loaded dynamically from active domain config
# ---------------------------------------------------------------------------


def _get_vibe_dimensions() -> list[str]:
    """Return dimension IDs from the active DomainConfig."""
    try:
        from app.core.domain_portability import get_domain_registry
        return get_domain_registry().get_active_domain().dimension_ids
    except Exception:
        # Fallback for unit tests or early import before registry is ready
        return ["quality", "price_positioning", "experience", "uniqueness", "accessibility"]


VIBE_DIMENSIONS: list[str] = _get_vibe_dimensions()


# ============================================================================
# Feature Store - data classes
# ============================================================================


@dataclass
class FeatureDefinition:
    """Schema and compute metadata for a single named feature.

    Attributes:
        name: Unique feature identifier (e.g. ``"vibe_formality"``).
        dtype: Logical type hint - one of ``"float"``, ``"int"``, ``"str"``,
            ``"list"``, ``"dict"``.
        description: Human-readable explanation of the feature.
        source: Provenance tag - ``"computed"`` (derived via compute_fn),
            ``"raw"`` (taken directly from entity data), or ``"derived"``
            (post-processed from other features).
        version: Monotonically increasing schema version.
        compute_fn: Optional callable ``(entity_data: dict) -> Any`` that
            produces the feature value from raw entity data.
        dependencies: Names of other features that must be computed first.
        created_at: UTC timestamp of initial registration.
        updated_at: UTC timestamp of the most recent metadata change.
        owner: Team or individual responsible for this feature.
    """

    name: str
    dtype: str = "float"
    description: str = ""
    source: str = "computed"
    version: int = 1
    compute_fn: Callable[..., Any] | None = None
    dependencies: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    owner: str = "arbor-ml"


@dataclass
class FeatureVector:
    """A computed snapshot of feature values for a single entity.

    Attributes:
        entity_id: The entity these features belong to.
        features: Mapping of feature name to its computed value.
        computed_at: UTC timestamp when the vector was computed.
        feature_version: Mapping of feature name to the definition version
            that was used during computation.
    """

    entity_id: str
    features: dict[str, Any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    feature_version: dict[str, int] = field(default_factory=dict)


# ============================================================================
# Built-in compute functions
# ============================================================================


def _compute_vibe_dimension(dimension: str) -> Callable[[dict[str, Any]], float]:
    """Return a compute function that extracts a single vibe dimension score.

    The returned callable looks up ``entity_data["vibe_dna"][dimension]``
    and normalises to [0.0, 1.0] (scores stored as 0-100 integers in the
    enrichment layer, or as 0.0-1.0 floats from the competitive-intelligence
    module).

    Args:
        dimension: Vibe dimension key (e.g. ``"formality"``).

    Returns:
        A callable ``(entity_data) -> float``.
    """

    def _extract(entity_data: dict[str, Any]) -> float:
        vibe_dna: dict[str, Any] = entity_data.get("vibe_dna", {})
        raw = vibe_dna.get(dimension, 0)
        # If the value is a dict with a "score" key (enrichment format),
        # unwrap it first.
        if isinstance(raw, dict):
            raw = raw.get("score", 0)
        score = float(raw)
        # Normalise 0-100 scale to 0-1 if needed
        if score > 1.0:
            score = score / 100.0
        return round(max(0.0, min(1.0, score)), 4)

    return _extract


def _compute_category_encoded(entity_data: dict[str, Any]) -> dict[str, int]:
    """One-hot encode the entity category.

    Returns a dict whose keys are ``"cat_<category>"`` with value 1 for the
    entity's category, making it trivial to merge into a flat feature vector.
    """
    category: str = entity_data.get("category", "")
    if not category:
        return {}
    safe_key = category.lower().replace(" ", "_").replace("-", "_")
    return {f"cat_{safe_key}": 1}


def _compute_name_embedding_norm(entity_data: dict[str, Any]) -> float:
    """Compute the L2 (Euclidean) norm of the entity's embedding vector.

    Falls back to 0.0 when no embedding is available.
    """
    embedding: list[float] | None = entity_data.get("embedding")
    if not embedding:
        return 0.0
    return round(math.sqrt(sum(x * x for x in embedding)), 6)


def _compute_tag_count(entity_data: dict[str, Any]) -> int:
    """Return the number of tags associated with the entity."""
    tags = entity_data.get("tags", [])
    if isinstance(tags, list):
        return len(tags)
    return 0


def _compute_has_website(entity_data: dict[str, Any]) -> int:
    """Return 1 if the entity has a website URL, 0 otherwise."""
    website = entity_data.get("website", entity_data.get("url", ""))
    return 1 if website else 0


def _compute_has_instagram(entity_data: dict[str, Any]) -> int:
    """Return 1 if the entity has an Instagram handle or link."""
    instagram = entity_data.get("instagram", entity_data.get("instagram_url", ""))
    return 1 if instagram else 0


# ============================================================================
# Feature Store
# ============================================================================


class FeatureStore:
    """In-memory feature registry and vector store.

    On initialisation the store pre-registers the built-in ARBOR features
    so they are immediately available for any downstream consumer.

    Thread-safety note: the current implementation is **not** thread-safe.
    In production, wrap mutation methods with a lock or migrate storage to
    Redis / a relational table.
    """

    def __init__(self) -> None:
        # feature_name -> FeatureDefinition
        self._definitions: dict[str, FeatureDefinition] = {}

        # entity_id -> FeatureVector
        self._vectors: dict[str, FeatureVector] = {}

        self._register_builtin_features()
        logger.info(
            "FeatureStore initialised with %d built-in features",
            len(self._definitions),
        )

    # ------------------------------------------------------------------
    # Built-in feature registration
    # ------------------------------------------------------------------

    def _register_builtin_features(self) -> None:
        """Register ARBOR's standard feature set."""

        # Vibe-DNA dimension features
        for dim in VIBE_DIMENSIONS:
            self.register_feature(
                FeatureDefinition(
                    name=f"vibe_{dim}",
                    dtype="float",
                    description=f"Normalised {dim} score extracted from vibe_dna (0-1).",
                    source="computed",
                    version=1,
                    compute_fn=_compute_vibe_dimension(dim),
                    dependencies=[],
                    owner="arbor-ml",
                )
            )

        # Category one-hot encoding
        self.register_feature(
            FeatureDefinition(
                name="category_encoded",
                dtype="dict",
                description="One-hot encoded entity category as a dict of cat_<name> -> 1.",
                source="computed",
                version=1,
                compute_fn=_compute_category_encoded,
                dependencies=[],
                owner="arbor-ml",
            )
        )

        # Embedding L2 norm
        self.register_feature(
            FeatureDefinition(
                name="name_embedding_norm",
                dtype="float",
                description="L2 norm of the entity embedding vector.",
                source="computed",
                version=1,
                compute_fn=_compute_name_embedding_norm,
                dependencies=[],
                owner="arbor-ml",
            )
        )

        # Tag count
        self.register_feature(
            FeatureDefinition(
                name="tag_count",
                dtype="int",
                description="Number of tags associated with the entity.",
                source="computed",
                version=1,
                compute_fn=_compute_tag_count,
                dependencies=[],
                owner="arbor-ml",
            )
        )

        # Boolean presence indicators
        self.register_feature(
            FeatureDefinition(
                name="has_website",
                dtype="int",
                description="1 if entity has a website URL, 0 otherwise.",
                source="computed",
                version=1,
                compute_fn=_compute_has_website,
                dependencies=[],
                owner="arbor-ml",
            )
        )

        self.register_feature(
            FeatureDefinition(
                name="has_instagram",
                dtype="int",
                description="1 if entity has an Instagram presence, 0 otherwise.",
                source="computed",
                version=1,
                compute_fn=_compute_has_instagram,
                dependencies=[],
                owner="arbor-ml",
            )
        )

    # ------------------------------------------------------------------
    # Feature definition management
    # ------------------------------------------------------------------

    def register_feature(self, definition: FeatureDefinition) -> None:
        """Register or update a feature definition.

        If a feature with the same name already exists, the version is bumped
        and ``updated_at`` is refreshed.

        Args:
            definition: The feature definition to register.
        """
        existing = self._definitions.get(definition.name)
        if existing is not None:
            definition.version = existing.version + 1
            definition.created_at = existing.created_at
            logger.info(
                "Feature '%s' updated to version %d",
                definition.name,
                definition.version,
            )
        else:
            logger.debug("Feature '%s' registered (v%d)", definition.name, definition.version)

        definition.updated_at = datetime.now(UTC)
        self._definitions[definition.name] = definition

    def get_feature_definition(self, name: str) -> FeatureDefinition:
        """Return the definition for *name*.

        Args:
            name: Feature identifier.

        Returns:
            The corresponding :class:`FeatureDefinition`.

        Raises:
            KeyError: If no feature with that name is registered.
        """
        if name not in self._definitions:
            raise KeyError(f"Feature '{name}' is not registered in the store.")
        return self._definitions[name]

    def list_features(self) -> list[dict[str, Any]]:
        """Return summary metadata for every registered feature.

        Returns:
            A list of dicts, each containing ``name``, ``dtype``, ``source``,
            ``version``, ``description``, and ``owner``.
        """
        return [
            {
                "name": d.name,
                "dtype": d.dtype,
                "source": d.source,
                "version": d.version,
                "description": d.description,
                "owner": d.owner,
                "has_compute_fn": d.compute_fn is not None,
                "dependencies": list(d.dependencies),
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
            }
            for d in self._definitions.values()
        ]

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_features(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        feature_names: list[str] | None = None,
    ) -> FeatureVector:
        """Compute feature values for a single entity.

        Features are computed in dependency order: if feature A depends on
        feature B, B will be computed (and its result injected into the
        entity data) before A.  Only features that have a ``compute_fn``
        are computed; features without one are silently skipped.

        The computed :class:`FeatureVector` is automatically persisted via
        :meth:`store_features`.

        Args:
            entity_id: Identifier of the entity.
            entity_data: Raw entity payload (enrichment, metadata, etc.).
            feature_names: Optional subset of features to compute.  When
                ``None``, all registered features are computed.

        Returns:
            A :class:`FeatureVector` with the computed values.
        """
        targets = feature_names or list(self._definitions.keys())
        ordered = self._resolve_dependency_order(targets)

        features: dict[str, Any] = {}
        versions: dict[str, int] = {}

        # Build a mutable copy of entity_data so that derived features can
        # reference earlier results via the "computed_features" key.
        working_data = copy.deepcopy(entity_data)
        working_data.setdefault("_computed", {})

        for name in ordered:
            defn = self._definitions.get(name)
            if defn is None or defn.compute_fn is None:
                continue
            try:
                value = defn.compute_fn(working_data)
                features[name] = value
                versions[name] = defn.version
                # Make the result available to downstream dependents
                working_data["_computed"][name] = value
            except Exception:
                logger.exception("Failed to compute feature '%s' for entity %s", name, entity_id)
                features[name] = None
                versions[name] = defn.version

        now = datetime.now(UTC)
        vector = FeatureVector(
            entity_id=entity_id,
            features=features,
            computed_at=now,
            feature_version=versions,
        )

        self.store_features(entity_id, vector)

        logger.debug(
            "Computed %d features for entity %s",
            len(features),
            entity_id,
        )
        return vector

    def _resolve_dependency_order(self, feature_names: list[str]) -> list[str]:
        """Topologically sort *feature_names* so dependencies come first.

        Uses a simple iterative approach (Kahn-style) that is sufficient
        for the small dependency graphs used in practice.

        Args:
            feature_names: Features that need to be computed.

        Returns:
            A list of feature names in safe computation order.
        """
        # Collect the full transitive closure of dependencies
        needed: set[str] = set()
        stack = list(feature_names)
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            defn = self._definitions.get(name)
            if defn is not None:
                for dep in defn.dependencies:
                    if dep not in needed:
                        stack.append(dep)

        # Build in-degree map scoped to the needed set
        in_degree: dict[str, int] = {n: 0 for n in needed}
        for n in needed:
            defn = self._definitions.get(n)
            if defn is not None:
                for dep in defn.dependencies:
                    if dep in needed:
                        in_degree[n] = in_degree.get(n, 0) + 1

        # Kahn's algorithm
        queue = [n for n, d in in_degree.items() if d == 0]
        ordered: list[str] = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            # For every feature that depends on *node*, decrement its in-degree
            for n in needed:
                defn = self._definitions.get(n)
                if defn is not None and node in defn.dependencies:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)

        # Append anything that was missed (e.g. due to a cycle) at the end
        for n in needed:
            if n not in ordered:
                ordered.append(n)

        return ordered

    # ------------------------------------------------------------------
    # Feature storage and retrieval
    # ------------------------------------------------------------------

    def store_features(self, entity_id: str, vector: FeatureVector) -> None:
        """Persist a feature vector for the given entity.

        If features already exist for the entity, new values are merged in
        (existing values are overwritten per feature name).

        Args:
            entity_id: Entity identifier.
            vector: The :class:`FeatureVector` to store.
        """
        existing = self._vectors.get(entity_id)
        if existing is not None:
            existing.features.update(vector.features)
            existing.feature_version.update(vector.feature_version)
            existing.computed_at = vector.computed_at
        else:
            self._vectors[entity_id] = vector

        logger.debug(
            "Stored %d features for entity %s",
            len(vector.features),
            entity_id,
        )

    def get_features(
        self,
        entity_id: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve stored feature values for an entity.

        Args:
            entity_id: Entity identifier.
            feature_names: Optional subset of features to return.  When
                ``None``, all stored features are returned.

        Returns:
            A dict of feature name -> value.  Returns an empty dict when
            no features have been stored for the entity.
        """
        vector = self._vectors.get(entity_id)
        if vector is None:
            return {}

        if feature_names is None:
            return dict(vector.features)

        return {k: vector.features[k] for k in feature_names if k in vector.features}

    # ------------------------------------------------------------------
    # Lineage and observability
    # ------------------------------------------------------------------

    def get_feature_lineage(self, feature_name: str) -> dict[str, Any]:
        """Return the full dependency tree for a feature.

        The tree is represented as a nested dict::

            {
                "name": "derived_feature",
                "version": 2,
                "source": "derived",
                "dependencies": [
                    {"name": "base_feature", "version": 1, ...},
                ]
            }

        Args:
            feature_name: Root feature to trace.

        Returns:
            A nested dict describing the lineage.

        Raises:
            KeyError: If the feature is not registered.
        """
        defn = self.get_feature_definition(feature_name)
        return self._build_lineage_tree(defn, visited=set())

    def _build_lineage_tree(
        self,
        defn: FeatureDefinition,
        visited: set[str],
    ) -> dict[str, Any]:
        """Recursively build the lineage tree, guarding against cycles."""
        node: dict[str, Any] = {
            "name": defn.name,
            "version": defn.version,
            "dtype": defn.dtype,
            "source": defn.source,
            "owner": defn.owner,
            "dependencies": [],
        }

        if defn.name in visited:
            node["_cycle"] = True
            return node

        visited.add(defn.name)

        for dep_name in defn.dependencies:
            dep_defn = self._definitions.get(dep_name)
            if dep_defn is not None:
                child = self._build_lineage_tree(dep_defn, visited)
                node["dependencies"].append(child)
            else:
                node["dependencies"].append({"name": dep_name, "_unregistered": True})

        return node

    def get_feature_freshness(self, entity_id: str) -> dict[str, Any]:
        """Report how stale the stored features are for an entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            A dict containing ``entity_id``, ``computed_at`` ISO string,
            ``age_seconds``, per-feature version info, and a boolean
            ``is_stale`` flag (True when older than 1 hour).  Returns
            a minimal dict with ``entity_id`` and ``stored=False`` when
            no features exist.
        """
        vector = self._vectors.get(entity_id)
        if vector is None:
            return {"entity_id": entity_id, "stored": False}

        now = datetime.now(UTC)
        age = (now - vector.computed_at).total_seconds()

        return {
            "entity_id": entity_id,
            "stored": True,
            "computed_at": vector.computed_at.isoformat(),
            "age_seconds": round(age, 2),
            "is_stale": age > 3600.0,
            "feature_versions": dict(vector.feature_version),
            "feature_count": len(vector.features),
        }


# ============================================================================
# Model Registry - data classes
# ============================================================================

# Valid lifecycle transitions
_VALID_PROMOTIONS: dict[str, list[str]] = {
    "staging": ["canary", "production"],
    "canary": ["production", "archived"],
    "production": ["archived"],
    "archived": [],
}


@dataclass
class ModelVersion:
    """An immutable record of a specific model version.

    Attributes:
        model_id: Logical model name (e.g. ``"vibe_scorer"``).
        version: Positive integer version number.
        model_type: Algorithm family (e.g. ``"gradient_boost"``).
        description: Free-text description of this version.
        metrics: Evaluation metrics captured during training / validation.
        parameters: Hyperparameters or configuration used for training.
        artifact_path: Optional filesystem or object-store path to the
            serialised model artifact.
        status: Lifecycle status - one of ``"staging"``, ``"production"``,
            ``"archived"``, ``"canary"``.
        created_at: UTC timestamp of version creation.
        promoted_at: UTC timestamp of the last promotion event, or ``None``.
        created_by: Author or pipeline that created this version.
        feature_requirements: Ordered list of feature names this model
            expects as input.
    """

    model_id: str
    version: int
    model_type: str = ""
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    status: str = "staging"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    created_by: str = "arbor-ml"
    feature_requirements: list[str] = field(default_factory=list)


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """In-memory registry for model metadata and lifecycle management.

    Supports a simple promotion flow::

        staging  -->  canary  -->  production  -->  archived

    At any time there is at most one ``production`` and one ``canary``
    version per model.  Promoting a new version to ``production``
    automatically archives the current production version.
    """

    def __init__(self) -> None:
        # model_id -> {"model_type": ..., "description": ..., "versions": [...]}
        self._models: dict[str, dict[str, Any]] = {}
        logger.info("ModelRegistry initialised (in-memory store)")

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        model_type: str,
        description: str = "",
    ) -> str:
        """Register a new logical model.

        If the model already exists, this is a no-op that logs a warning
        and returns the existing ``model_id``.

        Args:
            model_id: Unique model identifier.
            model_type: Algorithm family string.
            description: Human-readable description.

        Returns:
            The ``model_id``.
        """
        if model_id in self._models:
            logger.warning("Model '%s' already registered - skipping.", model_id)
            return model_id

        self._models[model_id] = {
            "model_type": model_type,
            "description": description,
            "created_at": datetime.now(UTC).isoformat(),
            "versions": [],
        }
        logger.info("Registered model '%s' (type=%s)", model_id, model_type)
        return model_id

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def log_version(
        self,
        model_id: str,
        metrics: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        feature_requirements: list[str] | None = None,
        description: str = "",
        artifact_path: str | None = None,
        created_by: str = "arbor-ml",
    ) -> ModelVersion:
        """Create and log a new version for an already-registered model.

        The version number is assigned automatically (incrementing from 1).
        New versions always start in ``"staging"`` status.

        Args:
            model_id: Identifier of the registered model.
            metrics: Evaluation metrics for this version.
            parameters: Training hyperparameters / configuration.
            feature_requirements: Feature names this version depends on.
            description: Free-text note about this version.
            artifact_path: Path to serialised model artefact.
            created_by: Author or CI pipeline name.

        Returns:
            The newly created :class:`ModelVersion`.

        Raises:
            KeyError: If the model has not been registered.
        """
        if model_id not in self._models:
            raise KeyError(
                f"Model '{model_id}' is not registered. " f"Call register_model() first."
            )

        model_meta = self._models[model_id]
        versions: list[ModelVersion] = model_meta["versions"]
        next_version = len(versions) + 1

        mv = ModelVersion(
            model_id=model_id,
            version=next_version,
            model_type=model_meta["model_type"],
            description=description,
            metrics=metrics or {},
            parameters=parameters or {},
            artifact_path=artifact_path,
            status="staging",
            created_by=created_by,
            feature_requirements=feature_requirements or [],
        )

        versions.append(mv)
        logger.info(
            "Logged version %d for model '%s' (status=staging)",
            next_version,
            model_id,
        )
        return mv

    # ------------------------------------------------------------------
    # Promotion and rollback
    # ------------------------------------------------------------------

    def promote(self, model_id: str, version: int, to_status: str) -> None:
        """Promote a model version to a new lifecycle status.

        When promoting to ``"production"``, the current production version
        (if any) is automatically moved to ``"archived"``.  Similarly,
        promoting to ``"canary"`` archives the current canary version.

        Args:
            model_id: Model identifier.
            version: Version number to promote.
            to_status: Target status (``"canary"``, ``"production"``,
                ``"archived"``).

        Raises:
            KeyError: If the model or version does not exist.
            ValueError: If the transition is invalid.
        """
        mv = self._get_version(model_id, version)
        allowed = _VALID_PROMOTIONS.get(mv.status, [])
        if to_status not in allowed:
            raise ValueError(
                f"Cannot promote model '{model_id}' v{version} from "
                f"'{mv.status}' to '{to_status}'. "
                f"Allowed transitions: {allowed}"
            )

        # Archive the incumbent in the target slot
        if to_status == "production":
            current_prod = self.get_production_model(model_id)
            if current_prod is not None and current_prod.version != version:
                current_prod.status = "archived"
                current_prod.promoted_at = datetime.now(UTC)
                logger.info(
                    "Archived previous production version %d of model '%s'",
                    current_prod.version,
                    model_id,
                )

        if to_status == "canary":
            current_canary = self.get_canary_model(model_id)
            if current_canary is not None and current_canary.version != version:
                current_canary.status = "archived"
                current_canary.promoted_at = datetime.now(UTC)

        mv.status = to_status
        mv.promoted_at = datetime.now(UTC)

        logger.info(
            "Promoted model '%s' v%d to '%s'",
            model_id,
            version,
            to_status,
        )

    def rollback(self, model_id: str) -> ModelVersion | None:
        """Revert to the previous production version of a model.

        The current production version is archived, and the most recent
        archived version that was previously in production is restored.

        Args:
            model_id: Model identifier.

        Returns:
            The restored :class:`ModelVersion`, or ``None`` if there is
            no previous version to roll back to.

        Raises:
            KeyError: If the model is not registered.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' is not registered.")

        current_prod = self.get_production_model(model_id)
        versions: list[ModelVersion] = self._models[model_id]["versions"]

        # Find the most recent archived version (candidate for rollback)
        candidates = [
            mv
            for mv in reversed(versions)
            if mv.status == "archived" and mv.promoted_at is not None
        ]
        if not candidates:
            logger.warning("No previous version to roll back to for model '%s'", model_id)
            return None

        previous = candidates[0]

        # Archive the current production version
        if current_prod is not None:
            current_prod.status = "archived"
            current_prod.promoted_at = datetime.now(UTC)

        # Restore the previous version
        previous.status = "production"
        previous.promoted_at = datetime.now(UTC)

        logger.info(
            "Rolled back model '%s': v%d -> v%d",
            model_id,
            current_prod.version if current_prod else 0,
            previous.version,
        )
        return previous

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_production_model(self, model_id: str) -> ModelVersion | None:
        """Return the current production version of a model, or ``None``.

        Args:
            model_id: Model identifier.
        """
        return self._find_by_status(model_id, "production")

    def get_canary_model(self, model_id: str) -> ModelVersion | None:
        """Return the current canary version of a model, or ``None``.

        Args:
            model_id: Model identifier.
        """
        return self._find_by_status(model_id, "canary")

    def compare_versions(
        self,
        model_id: str,
        v1: int,
        v2: int,
    ) -> dict[str, Any]:
        """Compare two versions of the same model side by side.

        Returns a dict with keys ``v1``, ``v2``, ``metric_comparison``, and
        ``parameter_diff``.

        Args:
            model_id: Model identifier.
            v1: First version number.
            v2: Second version number.

        Returns:
            Comparison dict.

        Raises:
            KeyError: If the model or either version does not exist.
        """
        mv1 = self._get_version(model_id, v1)
        mv2 = self._get_version(model_id, v2)

        # Metric-by-metric comparison
        all_metrics = set(mv1.metrics.keys()) | set(mv2.metrics.keys())
        metric_comparison: dict[str, dict[str, Any]] = {}
        for metric in sorted(all_metrics):
            val1 = mv1.metrics.get(metric)
            val2 = mv2.metrics.get(metric)
            entry: dict[str, Any] = {"v1": val1, "v2": val2}
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                entry["diff"] = round(diff, 6)
                entry["pct_change"] = round(diff / val1 * 100, 2) if val1 != 0 else None
            metric_comparison[metric] = entry

        # Parameter diff
        all_params = set(mv1.parameters.keys()) | set(mv2.parameters.keys())
        parameter_diff: dict[str, dict[str, Any]] = {}
        for param in sorted(all_params):
            p1 = mv1.parameters.get(param)
            p2 = mv2.parameters.get(param)
            if p1 != p2:
                parameter_diff[param] = {"v1": p1, "v2": p2}

        return {
            "model_id": model_id,
            "v1": {
                "version": mv1.version,
                "status": mv1.status,
                "created_at": mv1.created_at.isoformat(),
                "description": mv1.description,
            },
            "v2": {
                "version": mv2.version,
                "status": mv2.status,
                "created_at": mv2.created_at.isoformat(),
                "description": mv2.description,
            },
            "metric_comparison": metric_comparison,
            "parameter_diff": parameter_diff,
            "feature_requirements": {
                "v1": mv1.feature_requirements,
                "v2": mv2.feature_requirements,
                "added": [f for f in mv2.feature_requirements if f not in mv1.feature_requirements],
                "removed": [
                    f for f in mv1.feature_requirements if f not in mv2.feature_requirements
                ],
            },
        }

    def list_models(self) -> list[dict[str, Any]]:
        """Return summary metadata for all registered models.

        Returns:
            A list of dicts with ``model_id``, ``model_type``,
            ``description``, ``version_count``, and ``production_version``.
        """
        result: list[dict[str, Any]] = []
        for model_id, meta in self._models.items():
            versions: list[ModelVersion] = meta["versions"]
            prod = self.get_production_model(model_id)
            result.append(
                {
                    "model_id": model_id,
                    "model_type": meta["model_type"],
                    "description": meta["description"],
                    "version_count": len(versions),
                    "production_version": prod.version if prod else None,
                    "created_at": meta["created_at"],
                }
            )
        return result

    def get_model_history(self, model_id: str) -> list[ModelVersion]:
        """Return every version of a model, oldest first.

        Args:
            model_id: Model identifier.

        Returns:
            List of :class:`ModelVersion` instances.

        Raises:
            KeyError: If the model is not registered.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' is not registered.")
        return list(self._models[model_id]["versions"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version(self, model_id: str, version: int) -> ModelVersion:
        """Look up a specific model version, raising on miss."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' is not registered.")

        versions: list[ModelVersion] = self._models[model_id]["versions"]
        for mv in versions:
            if mv.version == version:
                return mv

        raise KeyError(
            f"Version {version} not found for model '{model_id}'. "
            f"Available versions: {[mv.version for mv in versions]}"
        )

    def _find_by_status(
        self,
        model_id: str,
        status: str,
    ) -> ModelVersion | None:
        """Return the most recent version with the given status."""
        if model_id not in self._models:
            return None

        versions: list[ModelVersion] = self._models[model_id]["versions"]
        for mv in reversed(versions):
            if mv.status == status:
                return mv
        return None


# ============================================================================
# Singleton accessors
# ============================================================================

_feature_store: FeatureStore | None = None
_model_registry: ModelRegistry | None = None


def get_feature_store() -> FeatureStore:
    """Return the singleton :class:`FeatureStore` instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def get_model_registry() -> ModelRegistry:
    """Return the singleton :class:`ModelRegistry` instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
