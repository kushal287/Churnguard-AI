"""
Benchmarking Suite for ChurnGuard AI.
Provides fair, scientific, and un-fabricated comparisons between
Custom NumPy Logistic Regression and Scikit-Learn Logistic Regression.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from config.config import BENCHMARK_RESULTS_PATH, RANDOM_SEED
from src.evaluation.metrics import evaluate_all
from src.models.custom_logistic_regression import CustomLogisticRegression

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Rigorous side-by-side benchmark comparison harness."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.custom_model: Optional[CustomLogisticRegression] = None
        self.sklearn_model: Optional[SklearnLogisticRegression] = None
        self.benchmark_results_: Dict[str, Any] = {}

    def run_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        custom_hyperparams: Optional[Dict[str, Any]] = None,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Execute fair benchmark across identical data partitions and evaluate all metrics.
        """
        params = custom_hyperparams or {
            "learning_rate": 0.05,
            "max_iter": 1500,
            "l2_lambda": 0.01,
            "momentum": 0.9,
            "batch_size": 64,
            "early_stopping": True,
            "patience": 50,
            "use_class_weights": True,
            "random_state": self.random_state,
        }

        # ----------------------------------------------------
        # 1. Train Custom NumPy Logistic Regression
        # ----------------------------------------------------
        logger.info("Training Custom NumPy Logistic Regression...")
        self.custom_model = CustomLogisticRegression(**params)
        
        t0_custom_train = time.perf_counter()
        self.custom_model.fit(X_train, y_train, X_val, y_val)
        t_custom_train_ms = (time.perf_counter() - t0_custom_train) * 1000.0

        # Inference timing (1000 repeated forward passes)
        t0_custom_infer = time.perf_counter()
        for _ in range(100):
            _ = self.custom_model.predict_proba(X_test)
        t_custom_infer_ms_per_k = ((time.perf_counter() - t0_custom_infer) / 100.0) * (1000.0 / len(X_test)) * 1000.0

        # Custom Predictions
        custom_val_proba = self.custom_model.predict_proba(X_val)[:, 1]
        custom_test_proba = self.custom_model.predict_proba(X_test)[:, 1]
        custom_train_proba = self.custom_model.predict_proba(X_train)[:, 1]

        custom_train_eval = evaluate_all(y_train, custom_train_proba, threshold=threshold)
        custom_val_eval = evaluate_all(y_val, custom_val_proba, threshold=threshold)
        custom_test_eval = evaluate_all(y_test, custom_test_proba, threshold=threshold)

        # ----------------------------------------------------
        # 2. Train Scikit-Learn Logistic Regression (Benchmark)
        # ----------------------------------------------------
        logger.info("Training Scikit-Learn Logistic Regression Baseline...")
        # Map L2 lambda to Sklearn C: C = 1 / (l2_lambda * m) or C = 1 / l2_lambda
        # Sklearn objective: min C * sum(loss) + 0.5 * ||w||^2
        # Custom objective: min 1/m * sum(loss) + (l2_lambda / (2m)) * ||w||^2
        # Multiplying Custom by m: min sum(loss) + (l2_lambda / 2) * ||w||^2
        # Multiplying Sklearn by 1/C: min sum(loss) + (1 / (2C)) * ||w||^2
        # Therefore: C = 1.0 / l2_lambda
        sklearn_c = 1.0 / max(params.get("l2_lambda", 0.01), 1e-5)
        class_weight_setting = "balanced" if params.get("use_class_weights", True) else None

        self.sklearn_model = SklearnLogisticRegression(
            C=sklearn_c,
            solver="lbfgs",
            max_iter=params.get("max_iter", 1500),
            class_weight=class_weight_setting,
            random_state=self.random_state,
        )

        t0_sk_train = time.perf_counter()
        self.sklearn_model.fit(X_train, y_train)
        t_sk_train_ms = (time.perf_counter() - t0_sk_train) * 1000.0

        t0_sk_infer = time.perf_counter()
        for _ in range(100):
            _ = self.sklearn_model.predict_proba(X_test)
        t_sk_infer_ms_per_k = ((time.perf_counter() - t0_sk_infer) / 100.0) * (1000.0 / len(X_test)) * 1000.0

        # Sklearn Predictions
        sk_train_proba = self.sklearn_model.predict_proba(X_train)[:, 1]
        sk_val_proba = self.sklearn_model.predict_proba(X_val)[:, 1]
        sk_test_proba = self.sklearn_model.predict_proba(X_test)[:, 1]

        sk_train_eval = evaluate_all(y_train, sk_train_proba, threshold=threshold)
        sk_val_eval = evaluate_all(y_val, sk_val_proba, threshold=threshold)
        sk_test_eval = evaluate_all(y_test, sk_test_proba, threshold=threshold)

        # ----------------------------------------------------
        # 3. Mathematical Parameter Equivalence Analysis
        # ----------------------------------------------------
        w_custom = self.custom_model.weights
        w_sklearn = self.sklearn_model.coef_.flatten()
        b_custom = self.custom_model.bias
        b_sklearn = float(self.sklearn_model.intercept_[0])

        # Pearson correlation between weight vectors
        corr_matrix = np.corrcoef(w_custom, w_sklearn)
        weight_correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Cosine similarity
        norm_c = np.linalg.norm(w_custom)
        norm_s = np.linalg.norm(w_sklearn)
        cosine_similarity = float(np.dot(w_custom, w_sklearn) / (norm_c * norm_s + 1e-12))

        # Mean absolute difference
        weight_mad = float(np.mean(np.abs(w_custom - w_sklearn)))
        bias_diff = float(abs(b_custom - b_sklearn))

        # Prediction probability correlation on test set
        proba_corr = float(np.corrcoef(custom_test_proba, sk_test_proba)[0, 1])

        # ----------------------------------------------------
        # 4. Compile Comprehensive Benchmark Results
        # ----------------------------------------------------
        self.benchmark_results_ = {
            "metadata": {
                "dataset": "IBM Telco Customer Churn",
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "features_count": int(X_train.shape[1]),
                "decision_threshold": threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "custom_numpy_model": {
                "training_time_ms": round(t_custom_train_ms, 2),
                "inference_latency_ms_per_1k": round(t_custom_infer_ms_per_k, 3),
                "epochs_trained": len(self.custom_model.train_loss_history_),
                "best_epoch": self.custom_model.best_epoch_,
                "train_metrics": custom_train_eval,
                "val_metrics": custom_val_eval,
                "test_metrics": custom_test_eval,
            },
            "sklearn_benchmark_model": {
                "training_time_ms": round(t_sk_train_ms, 2),
                "inference_latency_ms_per_1k": round(t_sk_infer_ms_per_k, 3),
                "solver": "lbfgs",
                "train_metrics": sk_train_eval,
                "val_metrics": sk_val_eval,
                "test_metrics": sk_test_eval,
            },
            "mathematical_fidelity": {
                "weight_pearson_correlation": round(weight_correlation, 4),
                "weight_cosine_similarity": round(cosine_similarity, 4),
                "weight_mean_absolute_difference": round(weight_mad, 4),
                "bias_difference": round(bias_diff, 4),
                "prediction_probability_correlation": round(proba_corr, 4),
            },
        }

        logger.info(
            f"Benchmark completed successfully! "
            f"Weight Correlation: {weight_correlation:.4f}, Prob Correlation: {proba_corr:.4f}, "
            f"Custom Test ROC-AUC: {custom_test_eval['roc_auc']:.4f} vs Sklearn: {sk_test_eval['roc_auc']:.4f}"
        )
        return self.benchmark_results_

    def save_results(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persist benchmark telemetry to disk."""
        path = Path(filepath) if filepath else BENCHMARK_RESULTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.benchmark_results_, f, indent=2)
        logger.info(f"Saved benchmark results to {path}")
        return path
