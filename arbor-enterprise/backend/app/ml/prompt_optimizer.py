"""Automated Prompt Optimization (DSPy-inspired) for ARBOR Enterprise.

Treats prompts as optimizable programs.  Each prompt is defined by a
:class:`PromptSignature` that describes its inputs, outputs, and the metric
used to evaluate quality.  Multiple :class:`PromptVariant` instances compete
for each signature; the optimizer evaluates them against held-out test cases,
mutates instructions, selects few-shot examples, and promotes the best
performing variant to active deployment.

Key ideas borrowed from DSPy:
- Signatures as declarative I/O contracts for prompts.
- Bootstrapped few-shot selection from successful historical runs.
- Instruction optimization via deterministic string mutations (no LLM needed).

Usage::

    optimizer = get_prompt_optimizer()
    best = optimizer.get_best_variant("intent_classification")
    result = optimizer.evaluate_variant(best.variant_id, test_cases, my_evaluator)
    optimizer.promote_variant(result.variant_id)
"""

import copy
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptSignature:
    """Declarative contract for a prompt's inputs, outputs, and evaluation.

    Attributes:
        name: Unique identifier for this signature (e.g. ``intent_classification``).
        input_fields: List of field names the prompt expects as input.
        output_fields: List of field names the prompt is expected to produce.
        description: Human-readable description of the prompt's purpose.
        evaluation_metric: Name of the metric used to score variants
            (e.g. ``accuracy``, ``f1``, ``bleu``).
    """

    name: str
    input_fields: list[str]
    output_fields: list[str]
    description: str
    evaluation_metric: str


@dataclass
class PromptVariant:
    """A concrete instantiation of a prompt signature with a template,
    instruction, and optional few-shot examples.

    Attributes:
        variant_id: UUID string uniquely identifying this variant.
        signature_name: The :class:`PromptSignature` this variant belongs to.
        template: The prompt template string (may contain ``{field}`` placeholders).
        few_shot_examples: Optional list of example dicts for in-context learning.
        instruction: System / meta-instruction prepended to the prompt.
        version: Monotonically increasing version counter within a signature.
        score: Running average evaluation score.
        eval_count: Number of evaluations that contributed to *score*.
        created_at: UTC timestamp of creation.
        is_active: Whether this variant is the currently deployed one for its
            signature.
    """

    variant_id: str
    signature_name: str
    template: str
    few_shot_examples: list[dict] = field(default_factory=list)
    instruction: str = ""
    version: int = 1
    score: float = 0.0
    eval_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = False


@dataclass
class EvaluationResult:
    """Outcome of evaluating a single :class:`PromptVariant` on a set of test cases.

    Attributes:
        variant_id: The variant that was evaluated.
        metric_name: Name of the evaluation metric (mirrors the signature's
            ``evaluation_metric``).
        score: Aggregate score across all test cases.
        sample_count: Number of test cases used.
        details: Arbitrary extra information (per-case scores, error counts, etc.).
        evaluated_at: UTC timestamp of the evaluation.
    """

    variant_id: str
    metric_name: str
    score: float
    sample_count: int
    details: dict = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# InstructionMutator
# ---------------------------------------------------------------------------


class InstructionMutator:
    """Deterministic string-level mutations for prompt instructions.

    All mutations are pure string manipulation -- no LLM calls required.
    Each strategy transforms the instruction in a reproducible way so that
    the optimizer can explore the instruction-design space cheaply.
    """

    # Mapping of strategy name to method
    _STRATEGIES = (
        "elaborate",
        "simplify",
        "formalize",
        "conversational",
        "constrain",
    )

    # ------------------------------------------------------------------
    # Elaboration helpers
    # ------------------------------------------------------------------

    _ELABORATION_PREFIXES = [
        "Think carefully and ",
        "Step by step, ",
        "Be thorough and ",
        "Consider all aspects and ",
        "Analyze in depth and ",
    ]

    _ELABORATION_SUFFIXES = [
        " Provide detailed reasoning.",
        " Explain your thought process.",
        " Be as specific as possible.",
        " Include supporting evidence where available.",
        " Consider edge cases.",
    ]

    # ------------------------------------------------------------------
    # Formalization helpers
    # ------------------------------------------------------------------

    _FORMAL_SUBSTITUTIONS: list[tuple[str, str]] = [
        (r"\bfigure out\b", "determine"),
        (r"\bget\b", "obtain"),
        (r"\bgive\b", "provide"),
        (r"\bshow\b", "demonstrate"),
        (r"\btell\b", "indicate"),
        (r"\bhelp\b", "assist"),
        (r"\buse\b", "utilize"),
        (r"\bcheck\b", "verify"),
        (r"\bstart\b", "initiate"),
        (r"\bmake sure\b", "ensure"),
        (r"\bfind\b", "identify"),
        (r"\bpick\b", "select"),
        (r"\bthink about\b", "consider"),
        (r"\blook at\b", "examine"),
        (r"\btry\b", "attempt"),
    ]

    # ------------------------------------------------------------------
    # Conversational helpers
    # ------------------------------------------------------------------

    _CASUAL_SUBSTITUTIONS: list[tuple[str, str]] = [
        (r"\bdetermine\b", "figure out"),
        (r"\bobtain\b", "get"),
        (r"\bprovide\b", "give"),
        (r"\bdemonstrate\b", "show"),
        (r"\bindicate\b", "tell"),
        (r"\bassist\b", "help"),
        (r"\butilize\b", "use"),
        (r"\bverify\b", "check"),
        (r"\binitiate\b", "start"),
        (r"\bensure\b", "make sure"),
        (r"\bidentify\b", "find"),
        (r"\bselect\b", "pick"),
        (r"\bconsider\b", "think about"),
        (r"\bexamine\b", "look at"),
        (r"\battempt\b", "try"),
    ]

    # ------------------------------------------------------------------
    # Constraint templates
    # ------------------------------------------------------------------

    _CONSTRAINT_SUFFIXES = [
        " Respond with valid JSON only.",
        " Return your answer as a numbered list.",
        " Keep your response under 200 words.",
        " Format output as key: value pairs.",
        " Output exactly one answer per line.",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mutate(self, instruction: str, strategy: str) -> str:
        """Apply a named mutation strategy to *instruction*.

        Args:
            instruction: The base instruction string to mutate.
            strategy: One of ``elaborate``, ``simplify``, ``formalize``,
                ``conversational``, or ``constrain``.

        Returns:
            A mutated copy of the instruction.

        Raises:
            ValueError: If *strategy* is not recognised.
        """
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown mutation strategy '{strategy}'. "
                f"Choose from: {', '.join(self._STRATEGIES)}"
            )

        handler = getattr(self, f"_mutate_{strategy}")
        result: str = handler(instruction)
        return result.strip()

    def generate_candidates(self, base_instruction: str, n: int = 5) -> list[str]:
        """Generate *n* mutated candidates from *base_instruction*.

        Each candidate is produced by applying a randomly chosen strategy.
        The original instruction is always included as the first candidate so
        that the optimizer can compare mutations against the baseline.

        Args:
            base_instruction: The instruction to mutate.
            n: Total number of candidates to return (including the original).

        Returns:
            A list of *n* instruction strings.
        """
        candidates: list[str] = [base_instruction]
        strategies = list(self._STRATEGIES)

        for _ in range(n - 1):
            strategy = random.choice(strategies)
            mutated = self.mutate(base_instruction, strategy)
            # Avoid exact duplicates
            if mutated not in candidates:
                candidates.append(mutated)
            else:
                # Try a different strategy to get a unique variant
                for fallback_strategy in strategies:
                    alt = self.mutate(base_instruction, fallback_strategy)
                    if alt not in candidates:
                        candidates.append(alt)
                        break
                else:
                    # All strategies produced duplicates; keep the original
                    candidates.append(mutated)

        return candidates[:n]

    # ------------------------------------------------------------------
    # Private mutation implementations
    # ------------------------------------------------------------------

    def _mutate_elaborate(self, instruction: str) -> str:
        """Add more detail and specificity to the instruction."""
        prefix = random.choice(self._ELABORATION_PREFIXES)
        suffix = random.choice(self._ELABORATION_SUFFIXES)

        # Lower-case the first character of the original if we prepend
        elaborated = instruction
        if elaborated and elaborated[0].isupper():
            elaborated = elaborated[0].lower() + elaborated[1:]

        return f"{prefix}{elaborated}{suffix}"

    def _mutate_simplify(self, instruction: str) -> str:
        """Remove filler words and make the instruction more concise."""
        filler_patterns = [
            r"\bplease\b\s*",
            r"\bkindly\b\s*",
            r"\bjust\b\s*",
            r"\bsimply\b\s*",
            r"\bbasically\b\s*",
            r"\bactually\b\s*",
            r"\breally\b\s*",
            r"\bvery\b\s*",
            r"\bquite\b\s*",
            r"\bperhaps\b\s*",
            r"\bmaybe\b\s*",
            r"\bin order to\b",
            r"\bfor the purpose of\b",
            r"\bat this point in time\b",
            r"\bdue to the fact that\b",
        ]

        simplified = instruction
        for pattern in filler_patterns:
            simplified = re.sub(pattern, "", simplified, flags=re.IGNORECASE)

        # Collapse multiple spaces
        simplified = re.sub(r"\s{2,}", " ", simplified)

        # Replace wordy phrases with concise equivalents
        wordy_replacements = [
            (r"\bin order to\b", "to"),
            (r"\bfor the purpose of\b", "for"),
            (r"\bat this point in time\b", "now"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bin the event that\b", "if"),
            (r"\bwith regard to\b", "about"),
        ]
        for pattern, replacement in wordy_replacements:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

        return simplified.strip()

    def _mutate_formalize(self, instruction: str) -> str:
        """Replace informal language with technical / formal equivalents."""
        formalized = instruction
        for pattern, replacement in self._FORMAL_SUBSTITUTIONS:
            formalized = re.sub(pattern, replacement, formalized, flags=re.IGNORECASE)
        return formalized

    def _mutate_conversational(self, instruction: str) -> str:
        """Replace formal language with casual equivalents."""
        casual = instruction
        for pattern, replacement in self._CASUAL_SUBSTITUTIONS:
            casual = re.sub(pattern, replacement, casual, flags=re.IGNORECASE)
        return casual

    def _mutate_constrain(self, instruction: str) -> str:
        """Append an output format constraint to the instruction."""
        constraint = random.choice(self._CONSTRAINT_SUFFIXES)
        # Avoid duplicating a constraint that already exists
        if constraint.strip() in instruction:
            # Try another constraint
            for alt in self._CONSTRAINT_SUFFIXES:
                if alt.strip() not in instruction:
                    constraint = alt
                    break
        return f"{instruction}{constraint}"


# ---------------------------------------------------------------------------
# FewShotSelector
# ---------------------------------------------------------------------------


@dataclass
class _FewShotExample:
    """Internal representation of a stored few-shot example."""

    signature_name: str
    input_data: dict
    output_data: dict
    score: float
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FewShotSelector:
    """Manages a pool of few-shot examples and selects subsets for prompts.

    Supports three selection strategies:

    * **best** -- top-*k* examples by score.
    * **diverse** -- greedy maximal marginal relevance to maximise diversity.
    * **recent** -- most recently added *k* examples.
    """

    def __init__(self) -> None:
        self._pool: list[_FewShotExample] = []
        logger.info("FewShotSelector initialised (in-memory pool)")

    # ------------------------------------------------------------------
    # Adding examples
    # ------------------------------------------------------------------

    def add_example(
        self,
        signature_name: str,
        input_data: dict,
        output_data: dict,
        score: float,
    ) -> None:
        """Add a single example to the pool.

        Args:
            signature_name: The signature this example belongs to.
            input_data: The input dict used for the prompt.
            output_data: The expected / actual output dict.
            score: Quality score of this example (higher is better).
        """
        example = _FewShotExample(
            signature_name=signature_name,
            input_data=input_data,
            output_data=output_data,
            score=score,
        )
        self._pool.append(example)

        logger.debug(
            "Added few-shot example: signature=%s score=%.4f pool_size=%d",
            signature_name,
            score,
            len(self._pool),
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        signature_name: str,
        k: int = 3,
        strategy: str = "diverse",
    ) -> list[dict]:
        """Select *k* few-shot examples for *signature_name*.

        Args:
            signature_name: Only examples belonging to this signature are
                considered.
            k: Number of examples to return.
            strategy: Selection strategy -- ``best``, ``diverse``, or ``recent``.

        Returns:
            A list of dicts with keys ``input`` and ``output`` drawn from the
            pool, or fewer if the pool is smaller than *k*.

        Raises:
            ValueError: If *strategy* is not recognised.
        """
        candidates = [
            ex for ex in self._pool if ex.signature_name == signature_name
        ]

        if not candidates:
            return []

        if strategy == "best":
            selected = self._select_best(candidates, k)
        elif strategy == "diverse":
            selected = self._select_diverse(candidates, k)
        elif strategy == "recent":
            selected = self._select_recent(candidates, k)
        else:
            raise ValueError(
                f"Unknown selection strategy '{strategy}'. "
                f"Choose from: best, diverse, recent"
            )

        return [
            {"input": ex.input_data, "output": ex.output_data}
            for ex in selected
        ]

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------

    def bootstrap_from_runs(
        self,
        signature_name: str,
        runs: list[dict],
        min_score: float = 0.7,
    ) -> int:
        """Auto-populate the example pool from successful historical runs.

        Each run dict is expected to have keys ``input``, ``output``, and
        ``score``.  Only runs whose score meets or exceeds *min_score* are
        added.

        Args:
            signature_name: The signature the runs belong to.
            runs: List of run dicts.
            min_score: Minimum score threshold for inclusion.

        Returns:
            Number of examples added.
        """
        added = 0
        for run in runs:
            run_score = float(run.get("score", 0.0))
            if run_score >= min_score:
                self.add_example(
                    signature_name=signature_name,
                    input_data=run.get("input", {}),
                    output_data=run.get("output", {}),
                    score=run_score,
                )
                added += 1

        logger.info(
            "Bootstrapped %d examples from %d runs for signature=%s (min_score=%.2f)",
            added,
            len(runs),
            signature_name,
            min_score,
        )
        return added

    # ------------------------------------------------------------------
    # Private selection implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _select_best(
        candidates: list[_FewShotExample], k: int
    ) -> list[_FewShotExample]:
        """Return the top-*k* examples by score (descending)."""
        sorted_candidates = sorted(candidates, key=lambda ex: ex.score, reverse=True)
        return sorted_candidates[:k]

    @staticmethod
    def _select_recent(
        candidates: list[_FewShotExample], k: int
    ) -> list[_FewShotExample]:
        """Return the most recently added *k* examples."""
        sorted_candidates = sorted(
            candidates, key=lambda ex: ex.added_at, reverse=True
        )
        return sorted_candidates[:k]

    @staticmethod
    def _select_diverse(
        candidates: list[_FewShotExample], k: int
    ) -> list[_FewShotExample]:
        """Greedy maximal marginal relevance for diversity.

        The algorithm starts by selecting the highest-scored example, then
        iteratively picks the candidate that is most different from the
        already-selected set.  Difference is approximated by comparing the
        set of keys in each example's input/output dicts (Jaccard distance).
        """
        if len(candidates) <= k:
            return list(candidates)

        def _keys_set(ex: _FewShotExample) -> set[str]:
            """Flatten input + output dict keys and string values into a set."""
            tokens: set[str] = set()
            for d in (ex.input_data, ex.output_data):
                for key, value in d.items():
                    tokens.add(key)
                    if isinstance(value, str):
                        tokens.update(value.lower().split())
            return tokens

        # Pre-compute token sets for all candidates
        token_sets = {id(ex): _keys_set(ex) for ex in candidates}

        # Start with the best-scored candidate
        remaining = list(candidates)
        remaining.sort(key=lambda ex: ex.score, reverse=True)
        selected: list[_FewShotExample] = [remaining.pop(0)]

        while len(selected) < k and remaining:
            best_candidate = None
            best_mmr = -1.0

            selected_union: set[str] = set()
            for s in selected:
                selected_union |= token_sets[id(s)]

            for candidate in remaining:
                cand_tokens = token_sets[id(candidate)]
                # Jaccard distance from selected set
                if selected_union or cand_tokens:
                    intersection = len(cand_tokens & selected_union)
                    union = len(cand_tokens | selected_union)
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0

                diversity = 1.0 - similarity
                # MMR: balance quality and diversity (lambda = 0.5)
                mmr = 0.5 * candidate.score + 0.5 * diversity

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_candidate = candidate

            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------


class PromptOptimizer:
    """Central orchestrator for prompt registration, evaluation, and optimization.

    Maintains an in-memory registry of :class:`PromptSignature` definitions,
    their :class:`PromptVariant` candidates, and a full evaluation history.
    Provides methods to evaluate variants against test cases, mutate
    instructions, select few-shot examples, and promote the best variant.
    """

    def __init__(self) -> None:
        self._signatures: dict[str, PromptSignature] = {}
        self._variants: dict[str, PromptVariant] = {}  # variant_id -> variant
        self._evaluation_history: list[EvaluationResult] = []
        self._mutator = InstructionMutator()
        self._few_shot_selector = FewShotSelector()
        self._version_counters: dict[str, int] = {}  # signature_name -> next version

        # Register built-in signatures
        self._register_default_signatures()

        logger.info("PromptOptimizer initialised with %d default signatures",
                     len(self._signatures))

    # ------------------------------------------------------------------
    # Signature management
    # ------------------------------------------------------------------

    def register_signature(self, signature: PromptSignature) -> None:
        """Register a prompt signature.

        If a signature with the same name already exists, it is overwritten
        and a warning is logged.

        Args:
            signature: The :class:`PromptSignature` to register.
        """
        if signature.name in self._signatures:
            logger.warning(
                "Overwriting existing signature: %s", signature.name
            )
        self._signatures[signature.name] = signature
        self._version_counters.setdefault(signature.name, 1)

        logger.info(
            "Registered signature: name=%s inputs=%s outputs=%s metric=%s",
            signature.name,
            signature.input_fields,
            signature.output_fields,
            signature.evaluation_metric,
        )

    # ------------------------------------------------------------------
    # Variant management
    # ------------------------------------------------------------------

    def add_variant(
        self,
        signature_name: str,
        template: str,
        instruction: str,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> PromptVariant:
        """Create and register a new variant for *signature_name*.

        Args:
            signature_name: Must reference a previously registered signature.
            template: The prompt template string.
            instruction: Meta-instruction prepended to the prompt.
            few_shot_examples: Optional list of example dicts.

        Returns:
            The newly created :class:`PromptVariant`.

        Raises:
            KeyError: If *signature_name* is not registered.
        """
        if signature_name not in self._signatures:
            raise KeyError(
                f"Signature '{signature_name}' is not registered. "
                f"Call register_signature() first."
            )

        version = self._version_counters.get(signature_name, 1)
        self._version_counters[signature_name] = version + 1

        variant = PromptVariant(
            variant_id=str(uuid.uuid4()),
            signature_name=signature_name,
            template=template,
            few_shot_examples=few_shot_examples or [],
            instruction=instruction,
            version=version,
        )
        self._variants[variant.variant_id] = variant

        # If this is the first variant for the signature, auto-activate it
        existing_active = [
            v for v in self._variants.values()
            if v.signature_name == signature_name and v.is_active
        ]
        if not existing_active:
            variant.is_active = True

        logger.info(
            "Added variant: id=%s signature=%s version=%d active=%s",
            variant.variant_id,
            signature_name,
            version,
            variant.is_active,
        )
        return variant

    def get_best_variant(self, signature_name: str) -> Optional[PromptVariant]:
        """Return the highest-scoring active variant for *signature_name*.

        Only variants with at least one evaluation are considered.  If no
        evaluated variants exist, the currently active variant is returned.
        If no variants exist at all, ``None`` is returned.

        Args:
            signature_name: The signature to look up.

        Returns:
            The best :class:`PromptVariant`, or ``None``.
        """
        candidates = [
            v for v in self._variants.values()
            if v.signature_name == signature_name
        ]

        if not candidates:
            return None

        evaluated = [v for v in candidates if v.eval_count > 0]
        if evaluated:
            return max(evaluated, key=lambda v: v.score)

        # Fall back to the active variant
        active = [v for v in candidates if v.is_active]
        return active[0] if active else candidates[0]

    def get_active_variant(self, signature_name: str) -> Optional[PromptVariant]:
        """Return the currently deployed (active) variant for *signature_name*.

        Args:
            signature_name: The signature to look up.

        Returns:
            The active :class:`PromptVariant`, or ``None`` if no variant is
            active for that signature.
        """
        for variant in self._variants.values():
            if variant.signature_name == signature_name and variant.is_active:
                return variant
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_variant(
        self,
        variant_id: str,
        test_cases: list[dict],
        evaluator_fn: Callable[[PromptVariant, dict], float],
    ) -> EvaluationResult:
        """Evaluate a variant against a set of test cases.

        For each test case the *evaluator_fn* is invoked with the variant and
        the test case dict; it must return a float score.  The aggregate score
        is the mean across all test cases.  The variant's running average
        ``score`` is updated incrementally.

        Args:
            variant_id: Must reference an existing variant.
            test_cases: List of test-case dicts understood by the evaluator.
            evaluator_fn: ``(variant, test_case) -> float`` scoring function.

        Returns:
            An :class:`EvaluationResult` summarising the evaluation.

        Raises:
            KeyError: If *variant_id* is not found.
        """
        if variant_id not in self._variants:
            raise KeyError(f"Variant '{variant_id}' not found.")

        variant = self._variants[variant_id]
        signature = self._signatures.get(variant.signature_name)
        metric_name = signature.evaluation_metric if signature else "unknown"

        scores: list[float] = []
        per_case: list[dict] = []

        for idx, test_case in enumerate(test_cases):
            try:
                case_score = float(evaluator_fn(variant, test_case))
            except Exception:
                logger.exception(
                    "Evaluator failed on test case %d for variant=%s",
                    idx,
                    variant_id,
                )
                case_score = 0.0

            scores.append(case_score)
            per_case.append({"index": idx, "score": case_score})

        aggregate_score = sum(scores) / len(scores) if scores else 0.0

        # Update variant running average: weighted by eval_count
        old_total = variant.score * variant.eval_count
        variant.eval_count += len(scores)
        variant.score = (
            (old_total + sum(scores)) / variant.eval_count
            if variant.eval_count > 0
            else 0.0
        )

        result = EvaluationResult(
            variant_id=variant_id,
            metric_name=metric_name,
            score=aggregate_score,
            sample_count=len(test_cases),
            details={"per_case": per_case, "aggregate": aggregate_score},
        )
        self._evaluation_history.append(result)

        logger.info(
            "Evaluated variant=%s metric=%s score=%.4f samples=%d running_avg=%.4f",
            variant_id,
            metric_name,
            aggregate_score,
            len(test_cases),
            variant.score,
        )
        return result

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        signature_name: str,
        n_candidates: int = 5,
    ) -> PromptVariant:
        """Generate and evaluate candidate variants, returning the best.

        Optimization proceeds in three steps:

        1. **Instruction mutation** -- the current active variant's instruction
           is mutated into *n_candidates* alternatives using
           :class:`InstructionMutator`.
        2. **Few-shot selection** -- the best available few-shot examples are
           selected from the pool via :class:`FewShotSelector`.
        3. **Candidate creation** -- each mutated instruction is paired with
           the selected examples and registered as a new variant.

        The caller is responsible for later calling :meth:`evaluate_variant`
        on each candidate with real test cases and then :meth:`promote_variant`
        on the winner.

        Args:
            signature_name: Must reference a registered signature.
            n_candidates: Number of instruction variants to generate.

        Returns:
            The best new candidate :class:`PromptVariant` (by default, the
            first one; call :meth:`evaluate_variant` to establish scores).

        Raises:
            KeyError: If *signature_name* is not registered.
            RuntimeError: If no existing variant is available to seed from.
        """
        if signature_name not in self._signatures:
            raise KeyError(f"Signature '{signature_name}' is not registered.")

        # Seed from the best or active variant
        seed = self.get_best_variant(signature_name)
        if seed is None:
            raise RuntimeError(
                f"No existing variant for signature '{signature_name}' to seed from. "
                f"Call add_variant() first."
            )

        # 1. Generate instruction mutations
        instruction_candidates = self._mutator.generate_candidates(
            seed.instruction, n=n_candidates
        )

        # 2. Select best few-shot examples
        few_shots = self._few_shot_selector.select(
            signature_name, k=3, strategy="diverse"
        )

        # 3. Create candidate variants
        new_variants: list[PromptVariant] = []
        for instruction in instruction_candidates:
            variant = self.add_variant(
                signature_name=signature_name,
                template=seed.template,
                instruction=instruction,
                few_shot_examples=few_shots if few_shots else copy.deepcopy(seed.few_shot_examples),
            )
            new_variants.append(variant)

        logger.info(
            "Optimisation generated %d candidates for signature=%s",
            len(new_variants),
            signature_name,
        )

        # Return the first candidate (all start at score 0.0 until evaluated)
        return new_variants[0] if new_variants else seed

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_variant(self, variant_id: str) -> None:
        """Set a variant as the active deployment for its signature.

        Deactivates any previously active variant for the same signature.

        Args:
            variant_id: Must reference an existing variant.

        Raises:
            KeyError: If *variant_id* is not found.
        """
        if variant_id not in self._variants:
            raise KeyError(f"Variant '{variant_id}' not found.")

        variant = self._variants[variant_id]

        # Deactivate all variants for this signature
        for v in self._variants.values():
            if v.signature_name == variant.signature_name:
                v.is_active = False

        variant.is_active = True

        logger.info(
            "Promoted variant=%s for signature=%s (score=%.4f eval_count=%d)",
            variant_id,
            variant.signature_name,
            variant.score,
            variant.eval_count,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_optimization_history(self, signature_name: str) -> list[dict]:
        """Return the evaluation history for all variants of *signature_name*.

        Args:
            signature_name: The signature whose history to retrieve.

        Returns:
            A list of dicts, each containing the variant id, version, score,
            evaluation count, active status, and all evaluation results.
        """
        variants = [
            v for v in self._variants.values()
            if v.signature_name == signature_name
        ]
        variants.sort(key=lambda v: v.version)

        history: list[dict] = []
        for variant in variants:
            evals = [
                {
                    "metric_name": er.metric_name,
                    "score": er.score,
                    "sample_count": er.sample_count,
                    "evaluated_at": er.evaluated_at.isoformat(),
                    "details": er.details,
                }
                for er in self._evaluation_history
                if er.variant_id == variant.variant_id
            ]

            history.append({
                "variant_id": variant.variant_id,
                "version": variant.version,
                "instruction": variant.instruction,
                "score": variant.score,
                "eval_count": variant.eval_count,
                "is_active": variant.is_active,
                "created_at": variant.created_at.isoformat(),
                "evaluations": evals,
            })

        return history

    # ------------------------------------------------------------------
    # Few-shot selector proxy
    # ------------------------------------------------------------------

    @property
    def few_shot_selector(self) -> FewShotSelector:
        """Expose the internal :class:`FewShotSelector` for direct use."""
        return self._few_shot_selector

    @property
    def mutator(self) -> InstructionMutator:
        """Expose the internal :class:`InstructionMutator` for direct use."""
        return self._mutator

    # ------------------------------------------------------------------
    # Default signatures
    # ------------------------------------------------------------------

    def _register_default_signatures(self) -> None:
        """Register the pre-built ARBOR prompt signatures."""

        self.register_signature(PromptSignature(
            name="intent_classification",
            input_fields=["user_query", "conversation_history"],
            output_fields=["intent", "confidence", "entities"],
            description=(
                "Classify the user's query into one of the supported intents "
                "(discover, compare, detail, navigate, general) and extract "
                "mentioned entities."
            ),
            evaluation_metric="accuracy",
        ))

        self.register_signature(PromptSignature(
            name="entity_scoring",
            input_fields=["entity_data", "scoring_criteria"],
            output_fields=["vibe_scores", "reasoning"],
            description=(
                "Score an entity across multiple vibe dimensions (atmosphere, "
                "design, cultural_relevance, exclusivity, innovation) on a "
                "0-10 scale with supporting reasoning."
            ),
            evaluation_metric="mae",
        ))

        self.register_signature(PromptSignature(
            name="discovery_synthesis",
            input_fields=["user_query", "search_results", "user_context"],
            output_fields=["response", "recommendations", "follow_ups"],
            description=(
                "Synthesize search results into a coherent, conversational "
                "discovery response with tailored recommendations and "
                "follow-up questions."
            ),
            evaluation_metric="relevance",
        ))

        self.register_signature(PromptSignature(
            name="fact_extraction",
            input_fields=["text", "entity_type"],
            output_fields=["facts", "confidence_scores"],
            description=(
                "Extract structured facts (attributes, relationships, events) "
                "from unstructured text about a given entity type."
            ),
            evaluation_metric="f1",
        ))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_optimizer: Optional[PromptOptimizer] = None


def get_prompt_optimizer() -> PromptOptimizer:
    """Return the singleton PromptOptimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()
    return _optimizer
