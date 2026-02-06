"""Knowledge Distillation Pipeline for ARBOR Enterprise.

Transfers reasoning capability from expensive *teacher* models (e.g.
Gemini Pro) into lightweight *student* rule-sets that can serve the
majority of requests at near-zero cost.  An :class:`EscalationRouter`
decides at query time whether the student can handle the request or
whether the teacher must be consulted.

Architecture::

    ┌──────────────┐     training data      ┌──────────────┐
    │  Teacher LLM ├────────────────────────►│   Student    │
    │  (Gemini)    │   (input, output,       │  (rules /    │
    └──────┬───────┘    confidence)          │   lookups)   │
           │                                 └──────┬───────┘
           │                                        │
           └────────┐          ┌────────────────────┘
                    ▼          ▼
              ┌──────────────────────┐
              │  EscalationRouter    │
              │  simple → student    │
              │  complex → teacher   │
              └──────────────────────┘

Usage::

    pipeline = get_distillation_pipeline()
    dataset  = await pipeline.generate_training_data(inputs)
    model    = pipeline.train_student(dataset)
    metrics  = await pipeline.evaluate(test_inputs)
"""

import copy
import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TeacherConfig:
    """Configuration for the teacher (large) model."""

    model_name: str = "gemini-3-pro-preview"
    provider: str = "google"
    temperature: float = 0.3
    max_tokens: int = 2048
    cost_per_call: float = 0.005


@dataclass
class StudentConfig:
    """Configuration for the student (lightweight) model.

    The student is not a neural network – it is a set of deterministic
    rules and lookup tables distilled from the teacher's outputs.
    """

    model_name: str = "arbor-student-v1"
    model_type: str = "scorer"          # "classifier" | "scorer" | "ranker"
    input_features: list[str] = field(default_factory=lambda: ["query", "category"])
    output_type: str = "score"          # "label" | "score" | "ranking"
    training_examples: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Distillation Example & Dataset
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DistillationExample:
    """A single (input, teacher_output) training pair.

    Optionally stores the student's prediction for later evaluation.
    """

    input_data: dict = field(default_factory=dict)
    teacher_output: dict = field(default_factory=dict)
    teacher_confidence: float = 0.0
    student_output: dict | None = None
    student_score: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "input_data": self.input_data,
            "teacher_output": self.teacher_output,
            "teacher_confidence": self.teacher_confidence,
            "student_output": self.student_output,
            "student_score": self.student_score,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DistillationExample":
        return cls(
            input_data=data["input_data"],
            teacher_output=data["teacher_output"],
            teacher_confidence=data.get("teacher_confidence", 0.0),
            student_output=data.get("student_output"),
            student_score=data.get("student_score"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )


class DistillationDataset:
    """In-memory collection of :class:`DistillationExample` instances.

    Supports confidence-filtered training set extraction and stratified
    sampling for balanced evaluation.
    """

    def __init__(self) -> None:
        self._examples: list[DistillationExample] = []

    # -- Mutators ----------------------------------------------------------

    def add(self, example: DistillationExample) -> None:
        """Append *example* to the dataset."""
        self._examples.append(example)

    # -- Queries -----------------------------------------------------------

    def get_training_set(self, min_confidence: float = 0.7) -> list[DistillationExample]:
        """Return examples whose teacher confidence meets the threshold."""
        return [
            ex for ex in self._examples
            if ex.teacher_confidence >= min_confidence
        ]

    def get_stats(self) -> dict:
        """Summary statistics for the dataset."""
        if not self._examples:
            return {"size": 0, "avg_confidence": 0.0, "coverage": 0.0}

        confidences = [ex.teacher_confidence for ex in self._examples]
        high_confidence_count = sum(1 for c in confidences if c >= 0.7)

        return {
            "size": len(self._examples),
            "avg_confidence": sum(confidences) / len(confidences),
            "coverage": high_confidence_count / len(self._examples),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }

    def sample(
        self, n: int, strategy: str = "stratified"
    ) -> list[DistillationExample]:
        """Sample *n* examples from the dataset.

        Parameters
        ----------
        n:
            Number of examples to return (capped at dataset size).
        strategy:
            ``"stratified"`` – bin by confidence quartile and draw
            equally from each bin.  ``"random"`` – uniform random sample.
        """
        if not self._examples:
            return []

        n = min(n, len(self._examples))

        if strategy == "random" or len(self._examples) <= n:
            return random.sample(self._examples, n)

        # Stratified: bin by confidence quartile
        bins: dict[int, list[DistillationExample]] = defaultdict(list)
        for ex in self._examples:
            quartile = min(int(ex.teacher_confidence * 4), 3)
            bins[quartile].append(ex)

        sampled: list[DistillationExample] = []
        per_bin = max(1, n // max(len(bins), 1))

        for quartile in sorted(bins.keys()):
            bin_examples = bins[quartile]
            take = min(per_bin, len(bin_examples))
            sampled.extend(random.sample(bin_examples, take))

        # Fill remainder from full pool if needed
        while len(sampled) < n:
            candidate = random.choice(self._examples)
            if candidate not in sampled:
                sampled.append(candidate)

        return sampled[:n]

    def __len__(self) -> int:
        return len(self._examples)

    def __repr__(self) -> str:
        return f"DistillationDataset(size={len(self._examples)})"


# ═══════════════════════════════════════════════════════════════════════════
# Distillation Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class DistillationPipeline:
    """Orchestrates teacher inference, student training, and evaluation.

    The *student model* is not a neural network – it is a set of
    deterministic rules, lookup tables, and averaged heuristics distilled
    from the teacher's outputs.  This makes it fast, interpretable, and
    deployable without GPU infrastructure.
    """

    def __init__(
        self,
        teacher_config: TeacherConfig | None = None,
        student_config: StudentConfig | None = None,
    ) -> None:
        self.teacher_config = teacher_config or TeacherConfig()
        self.student_config = student_config or StudentConfig()

        # The trained student model (populated by train_student)
        self._student_model: dict[str, Any] = {}
        self._training_stats: dict[str, Any] = {}
        self._is_trained: bool = False

        logger.info(
            "DistillationPipeline initialised: teacher=%s student=%s (%s)",
            self.teacher_config.model_name,
            self.student_config.model_name,
            self.student_config.model_type,
        )

    # -- Teacher Data Generation -------------------------------------------

    async def generate_training_data(
        self,
        inputs: list[dict],
        batch_size: int = 10,
    ) -> DistillationDataset:
        """Run the teacher model on *inputs* and collect outputs.

        Calls the real LLM gateway to get teacher predictions. Falls back
        to deterministic heuristics if the gateway is unavailable.

        Parameters
        ----------
        inputs:
            List of input dicts, each containing at least the features
            declared in ``student_config.input_features``.
        batch_size:
            Number of inputs to process per batch (for logging / rate
            limiting purposes).
        """
        dataset = DistillationDataset()
        total_cost = 0.0

        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]

            for input_data in batch:
                teacher_output, confidence = await self._call_teacher(input_data)
                example = DistillationExample(
                    input_data=copy.deepcopy(input_data),
                    teacher_output=teacher_output,
                    teacher_confidence=confidence,
                )
                dataset.add(example)
                total_cost += self.teacher_config.cost_per_call

            logger.debug(
                "Generated %d / %d training examples (cost=$%.4f)",
                min(batch_start + batch_size, len(inputs)),
                len(inputs),
                total_cost,
            )

        logger.info(
            "Training data generation complete: %d examples, cost=$%.4f",
            len(dataset),
            total_cost,
        )
        return dataset

    async def _call_teacher(self, input_data: dict) -> tuple[dict, float]:
        """Call the real teacher LLM to generate a training label.

        Uses the LLM gateway to ask the teacher model to score/classify
        the input.  Falls back to ``_fallback_teacher()`` if the LLM is
        unavailable.

        Returns:
            (teacher_output_dict, confidence_float)
        """
        import json as _json

        query = str(input_data.get("query", ""))
        category = str(input_data.get("category", "general"))
        model_type = self.student_config.model_type

        try:
            from app.llm.gateway import get_llm_gateway
            gateway = get_llm_gateway()

            if model_type == "classifier":
                prompt = (
                    f"Classify this query into 'positive' (relevant to curated discovery) "
                    f"or 'negative' (irrelevant). Query: '{query}', Category: '{category}'. "
                    f"Respond with JSON: {{\"label\": \"positive\" or \"negative\", \"score\": 0.0-1.0, \"confidence\": 0.0-1.0}}"
                )
            elif model_type == "ranker":
                prompt = (
                    f"Rate the relevance of this query for the category on a scale of 1 (best) "
                    f"to 10 (worst). Query: '{query}', Category: '{category}'. "
                    f"Respond with JSON: {{\"rank\": 1-10, \"score\": 0.0-1.0, \"confidence\": 0.0-1.0}}"
                )
            else:
                prompt = (
                    f"Score the quality and relevance of this query for curated discovery "
                    f"(0.0=irrelevant, 1.0=perfect). Query: '{query}', Category: '{category}'. "
                    f"Respond with JSON: {{\"score\": 0.0-1.0, \"confidence\": 0.0-1.0}}"
                )

            response = await gateway.complete(
                messages=[
                    {"role": "system", "content": "You are a scoring model. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                task_type="simple",
                temperature=self.teacher_config.temperature,
            )

            # Parse the JSON response
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = _json.loads(cleaned)

            confidence = float(parsed.pop("confidence", 0.8))
            confidence = max(0.1, min(1.0, confidence))

            return parsed, round(confidence, 4)

        except Exception as exc:
            logger.debug("Teacher LLM call failed, using fallback: %s", exc)
            return self._fallback_teacher(input_data)

    def _fallback_teacher(self, input_data: dict) -> tuple[dict, float]:
        """Deterministic fallback when the LLM is unavailable.

        Produces heuristic outputs based on input features so that
        the pipeline can function without API keys.
        """
        query = str(input_data.get("query", ""))
        category = str(input_data.get("category", "general"))

        # Deterministic "score" based on query length and category hash
        query_signal = len(query) / max(len(query) + 50, 1)
        category_signal = (hash(category) % 100) / 100.0
        score = 0.4 * query_signal + 0.6 * category_signal
        score = max(0.0, min(1.0, score))

        # Confidence is higher for shorter, well-categorised queries
        confidence = 0.5 + 0.5 * (1.0 - query_signal)
        confidence = max(0.1, min(1.0, confidence))

        if self.student_config.model_type == "classifier":
            label = "positive" if score > 0.5 else "negative"
            return {"label": label, "score": round(score, 4)}, round(confidence, 4)
        elif self.student_config.model_type == "ranker":
            rank = max(1, int((1.0 - score) * 10))
            return {"rank": rank, "score": round(score, 4)}, round(confidence, 4)
        else:
            # scorer
            return {"score": round(score, 4)}, round(confidence, 4)

    # -- Student Training --------------------------------------------------

    def train_student(self, dataset: DistillationDataset) -> dict:
        """Build the student model from teacher-labelled data.

        The student is a collection of deterministic rules and lookup
        tables – no gradient descent involved:

        * **scorer**: average score per category, global mean fallback.
        * **classifier**: majority label per category.
        * **ranker**: average rank per category, sorted for tie-breaking.
        """
        training_set = dataset.get_training_set(min_confidence=0.7)
        if not training_set:
            logger.warning("No high-confidence examples; using full dataset")
            training_set = dataset.get_training_set(min_confidence=0.0)

        if not training_set:
            logger.error("Cannot train student: empty dataset")
            return {"error": "empty_dataset"}

        model: dict[str, Any] = {
            "model_type": self.student_config.model_type,
            "model_name": self.student_config.model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_size": len(training_set),
        }

        if self.student_config.model_type == "scorer":
            model.update(self._train_scorer(training_set))
        elif self.student_config.model_type == "classifier":
            model.update(self._train_classifier(training_set))
        elif self.student_config.model_type == "ranker":
            model.update(self._train_ranker(training_set))
        else:
            logger.warning("Unknown model_type %r; defaulting to scorer", self.student_config.model_type)
            model.update(self._train_scorer(training_set))

        self._student_model = model
        self._is_trained = True

        self._training_stats = {
            "training_size": len(training_set),
            "dataset_size": len(dataset),
            "model_type": self.student_config.model_type,
            "trained_at": model["trained_at"],
        }

        logger.info(
            "Student model trained: type=%s examples=%d",
            self.student_config.model_type,
            len(training_set),
        )
        return model

    def _train_scorer(self, examples: list[DistillationExample]) -> dict:
        """Build a lookup-table scorer: average score per category."""
        category_scores: dict[str, list[float]] = defaultdict(list)
        all_scores: list[float] = []

        for ex in examples:
            category = str(ex.input_data.get("category", "general"))
            score = float(ex.teacher_output.get("score", 0.5))
            category_scores[category].append(score)
            all_scores.append(score)

        global_mean = sum(all_scores) / len(all_scores) if all_scores else 0.5
        category_means = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        return {
            "global_mean": round(global_mean, 6),
            "category_means": {k: round(v, 6) for k, v in category_means.items()},
            "category_counts": {k: len(v) for k, v in category_scores.items()},
        }

    def _train_classifier(self, examples: list[DistillationExample]) -> dict:
        """Build a lookup-table classifier: majority label per category."""
        category_labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        global_labels: dict[str, int] = defaultdict(int)

        for ex in examples:
            category = str(ex.input_data.get("category", "general"))
            label = str(ex.teacher_output.get("label", "unknown"))
            category_labels[category][label] += 1
            global_labels[label] += 1

        # Majority label per category
        category_majority: dict[str, str] = {}
        for cat, label_counts in category_labels.items():
            category_majority[cat] = max(label_counts, key=label_counts.get)  # type: ignore[arg-type]

        global_majority = max(global_labels, key=global_labels.get) if global_labels else "unknown"  # type: ignore[arg-type]

        return {
            "global_majority": global_majority,
            "category_majority": category_majority,
            "label_distribution": dict(global_labels),
        }

    def _train_ranker(self, examples: list[DistillationExample]) -> dict:
        """Build a lookup-table ranker: average rank per category."""
        category_ranks: dict[str, list[int]] = defaultdict(list)
        all_ranks: list[int] = []

        for ex in examples:
            category = str(ex.input_data.get("category", "general"))
            rank = int(ex.teacher_output.get("rank", 5))
            category_ranks[category].append(rank)
            all_ranks.append(rank)

        global_avg_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 5.0
        category_avg_ranks = {
            cat: sum(ranks) / len(ranks)
            for cat, ranks in category_ranks.items()
        }

        return {
            "global_avg_rank": round(global_avg_rank, 4),
            "category_avg_ranks": {k: round(v, 4) for k, v in category_avg_ranks.items()},
            "category_counts": {k: len(v) for k, v in category_ranks.items()},
        }

    # -- Evaluation --------------------------------------------------------

    async def evaluate(self, test_inputs: list[dict]) -> dict:
        """Compare student vs teacher on *test_inputs*.

        Returns agreement rate, average score difference, and per-input
        breakdowns.
        """
        if not self._is_trained:
            logger.warning("Student not yet trained – evaluation will be inaccurate")

        agreements = 0
        score_diffs: list[float] = []
        details: list[dict] = []

        for input_data in test_inputs:
            teacher_out, teacher_conf = await self._call_teacher(input_data)
            student_out = self._predict_student(input_data)

            # Compute agreement depending on model type
            agreed = False
            diff = 0.0

            if self.student_config.model_type == "classifier":
                agreed = teacher_out.get("label") == student_out.get("label")
            elif self.student_config.model_type == "ranker":
                t_rank = teacher_out.get("rank", 5)
                s_rank = student_out.get("rank", 5)
                diff = abs(t_rank - s_rank)
                agreed = diff <= 1
            else:
                t_score = teacher_out.get("score", 0.5)
                s_score = student_out.get("score", 0.5)
                diff = abs(t_score - s_score)
                agreed = diff < 0.1

            if agreed:
                agreements += 1
            score_diffs.append(diff)

            details.append({
                "input": input_data,
                "teacher": teacher_out,
                "student": student_out,
                "agreed": agreed,
                "diff": round(diff, 6),
            })

        total = len(test_inputs) or 1
        avg_diff = sum(score_diffs) / total if score_diffs else 0.0

        metrics = {
            "total_evaluated": len(test_inputs),
            "agreement_rate": round(agreements / total, 4),
            "avg_diff": round(avg_diff, 6),
            "agreements": agreements,
            "disagreements": len(test_inputs) - agreements,
            "details": details[:50],  # Cap detail output
        }

        logger.info(
            "Evaluation complete: agreement=%.2f%% avg_diff=%.4f (%d inputs)",
            metrics["agreement_rate"] * 100,
            metrics["avg_diff"],
            len(test_inputs),
        )
        return metrics

    def _predict_student(self, input_data: dict) -> dict:
        """Run the student model (lookup tables) on a single input."""
        if not self._student_model:
            return {"score": 0.5}

        category = str(input_data.get("category", "general"))
        model_type = self._student_model.get("model_type", "scorer")

        if model_type == "scorer":
            category_means = self._student_model.get("category_means", {})
            score = category_means.get(
                category, self._student_model.get("global_mean", 0.5)
            )
            return {"score": round(score, 4)}

        elif model_type == "classifier":
            category_majority = self._student_model.get("category_majority", {})
            label = category_majority.get(
                category, self._student_model.get("global_majority", "unknown")
            )
            return {"label": label}

        elif model_type == "ranker":
            category_ranks = self._student_model.get("category_avg_ranks", {})
            rank = category_ranks.get(
                category, self._student_model.get("global_avg_rank", 5.0)
            )
            return {"rank": int(round(rank))}

        return {"score": 0.5}

    # -- Escalation --------------------------------------------------------

    def should_escalate(self, input_data: dict, student_output: dict) -> bool:
        """Decide whether to escalate this input to the teacher.

        Escalation occurs when the student's output suggests low
        confidence or when the input falls outside the student's known
        categories.
        """
        if not self._is_trained:
            return True

        category = str(input_data.get("category", "general"))
        model_type = self._student_model.get("model_type", "scorer")

        # Escalate if category was never seen during training
        if model_type == "scorer":
            known = self._student_model.get("category_means", {})
            if category not in known:
                return True
            # Escalate if score is near the decision boundary (ambiguous)
            score = student_output.get("score", 0.5)
            if 0.35 < score < 0.65:
                return True

        elif model_type == "classifier":
            known = self._student_model.get("category_majority", {})
            if category not in known:
                return True

        elif model_type == "ranker":
            known = self._student_model.get("category_avg_ranks", {})
            if category not in known:
                return True
            # Few training examples for this category → escalate
            counts = self._student_model.get("category_counts", {})
            if counts.get(category, 0) < 3:
                return True

        return False

    # -- Accessors ---------------------------------------------------------

    def get_student_model(self) -> dict:
        """Return the trained student model (rules and lookup tables)."""
        return copy.deepcopy(self._student_model)

    def __repr__(self) -> str:
        return (
            f"DistillationPipeline(teacher={self.teacher_config.model_name!r}, "
            f"student={self.student_config.model_name!r}, "
            f"trained={self._is_trained})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Escalation Router
# ═══════════════════════════════════════════════════════════════════════════


class EscalationRouter:
    """Routes incoming requests to either the student or the teacher.

    Simple / well-known inputs go to the student for fast, free
    inference.  Complex or novel inputs are escalated to the teacher.
    Routing outcomes are tracked so the split can be monitored.
    """

    def __init__(self, pipeline: DistillationPipeline | None = None) -> None:
        self._pipeline = pipeline
        self._routing_log: list[dict] = []
        self._outcome_log: list[dict] = []

    # -- Routing -----------------------------------------------------------

    def route(self, input_data: dict) -> str:
        """Decide routing for *input_data*.

        Returns
        -------
        str
            ``"student"`` if the student can handle the request,
            ``"teacher"`` otherwise.
        """
        # If no pipeline or untrained student, always escalate
        if self._pipeline is None or not self._pipeline._is_trained:
            decision = "teacher"
            self._record_route(input_data, decision, reason="no_trained_student")
            return decision

        # Quick complexity heuristic
        query = str(input_data.get("query", ""))
        category = str(input_data.get("category", ""))

        is_complex = (
            len(query) > 200
            or query.count(" AND ") > 1
            or query.count(" OR ") > 1
            or "compare" in query.lower()
            or "versus" in query.lower()
            or "explain why" in query.lower()
        )

        if is_complex:
            decision = "teacher"
            self._record_route(input_data, decision, reason="complex_query")
            return decision

        # Ask the pipeline's escalation logic
        student_out = self._pipeline._predict_student(input_data)
        if self._pipeline.should_escalate(input_data, student_out):
            decision = "teacher"
            self._record_route(input_data, decision, reason="student_uncertain")
            return decision

        decision = "student"
        self._record_route(input_data, decision, reason="student_confident")
        return decision

    def _record_route(self, input_data: dict, decision: str, reason: str) -> None:
        """Log a routing decision for observability."""
        self._routing_log.append({
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "reason": reason,
            "category": input_data.get("category", "unknown"),
        })

    # -- Outcome Tracking --------------------------------------------------

    def record_outcome(self, route: str, quality: float) -> None:
        """Record the quality of a routed prediction.

        Parameters
        ----------
        route:
            ``"student"`` or ``"teacher"``.
        quality:
            A 0-1 quality score for the response.
        """
        self._outcome_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": route,
            "quality": quality,
        })

    # -- Statistics --------------------------------------------------------

    def get_routing_stats(self) -> dict:
        """Aggregate routing and outcome statistics."""
        total = len(self._routing_log)
        student_count = sum(1 for r in self._routing_log if r["decision"] == "student")
        teacher_count = total - student_count

        # Reason breakdown
        reason_counts: dict[str, int] = defaultdict(int)
        for entry in self._routing_log:
            reason_counts[entry["reason"]] += 1

        # Outcome quality per route
        student_quality: list[float] = []
        teacher_quality: list[float] = []
        for outcome in self._outcome_log:
            if outcome["route"] == "student":
                student_quality.append(outcome["quality"])
            else:
                teacher_quality.append(outcome["quality"])

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "total_routed": total,
            "student_count": student_count,
            "teacher_count": teacher_count,
            "student_pct": round(student_count / max(total, 1), 4),
            "teacher_pct": round(teacher_count / max(total, 1), 4),
            "reason_breakdown": dict(reason_counts),
            "student_avg_quality": round(_avg(student_quality), 4),
            "teacher_avg_quality": round(_avg(teacher_quality), 4),
            "total_outcomes": len(self._outcome_log),
        }

    def __repr__(self) -> str:
        stats = self.get_routing_stats()
        return (
            f"EscalationRouter(routed={stats['total_routed']}, "
            f"student={stats['student_pct']:.0%}, "
            f"teacher={stats['teacher_pct']:.0%})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level Singletons
# ═══════════════════════════════════════════════════════════════════════════

_distillation_pipeline: DistillationPipeline | None = None
_escalation_router: EscalationRouter | None = None


def get_distillation_pipeline() -> DistillationPipeline:
    """Return the module-level :class:`DistillationPipeline` singleton."""
    global _distillation_pipeline
    if _distillation_pipeline is None:
        _distillation_pipeline = DistillationPipeline()
        logger.info("Global DistillationPipeline created")
    return _distillation_pipeline


def get_escalation_router() -> EscalationRouter:
    """Return the module-level :class:`EscalationRouter` singleton.

    Automatically wires itself to the global distillation pipeline.
    """
    global _escalation_router
    if _escalation_router is None:
        pipeline = get_distillation_pipeline()
        _escalation_router = EscalationRouter(pipeline=pipeline)
        logger.info("Global EscalationRouter created")
    return _escalation_router
