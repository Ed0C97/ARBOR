"""ML module initialization."""

from app.ml.drift_detection import (
    DriftMonitor,
    DriftResult,
    DriftSeverity,
    DriftType,
    EmbeddingDriftDetector,
    QueryDriftDetector,
)
from app.ml.ab_testing import (
    ABTestingService,
    Experiment,
    ExperimentStatus,
    Variant,
    get_ab_service,
)
from app.ml.rag_evaluation import (
    EvaluationReport,
    EvaluationResult,
    RAGEvaluator,
    RAGExample,
    run_golden_evaluation,
)

__all__ = [
    # Drift Detection
    "DriftMonitor",
    "DriftResult",
    "DriftSeverity",
    "DriftType",
    "EmbeddingDriftDetector",
    "QueryDriftDetector",
    # A/B Testing
    "ABTestingService",
    "Experiment",
    "ExperimentStatus",
    "Variant",
    "get_ab_service",
    # RAG Evaluation
    "EvaluationReport",
    "EvaluationResult",
    "RAGEvaluator",
    "RAGExample",
    "run_golden_evaluation",
]
