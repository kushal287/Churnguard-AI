"""Evaluation, benchmarking, and visualization modules for ChurnGuard AI."""
from src.evaluation.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    specificity_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    compute_financial_utility,
    find_optimal_threshold,
    evaluate_all,
)
from src.evaluation.benchmark import ModelBenchmark

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "specificity_score",
    "f1_score",
    "roc_curve",
    "roc_auc_score",
    "precision_recall_curve",
    "average_precision_score",
    "confusion_matrix",
    "compute_financial_utility",
    "find_optimal_threshold",
    "evaluate_all",
    "ModelBenchmark",
]
