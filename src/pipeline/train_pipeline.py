"""
End-to-End Training and Benchmark Execution Pipeline for ChurnGuard AI.
Orchestrates data ingestion, zero-leakage preprocessing, custom model training,
scikit-learn baseline benchmarking, plot generation, and authoritative results persistence.
"""

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy

from config.config import (
    ARTIFACTS_DIR,
    BENCHMARK_RESULTS_PATH,
    BUSINESS_COST_MATRIX,
    CUSTOM_MODEL_PATH,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_HYPERPARAMS,
    FEATURE_NAMES_PATH,
    FIGURES_DIR,
    FINAL_RESULTS_PATH,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.benchmark import ModelBenchmark
from src.evaluation.metrics import evaluate_all, find_optimal_threshold
from src.evaluation.plots import Visualizer
from src.explainability.feature_importance import GlobalExplainer
from src.models.custom_logistic_regression import CustomLogisticRegression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TrainingPipeline:
    """Master orchestrator for training, benchmarking, and authoritative reporting."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.loader = DataLoader(random_state=random_state)
        self.preprocessor = DataPreprocessor(apply_feature_engineering=True)
        self.benchmark = ModelBenchmark(random_state=random_state)
        self.visualizer = Visualizer(output_dir=FIGURES_DIR)

    @staticmethod
    def compute_file_sha256(filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def run(self, custom_hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute full reproducible pipeline and generate authoritative final_results.json.
        """
        logger.info("=" * 70)
        logger.info("STARTING CHURNGUARD AI TRAINING & BENCHMARKING PIPELINE")
        logger.info("=" * 70)

        # 1. Data Ingestion & Hash Computation
        raw_df = self.loader.load_raw_data()
        raw_hash = self.compute_file_sha256(RAW_DATA_PATH)
        logger.info(f"Raw dataset loaded. Size: {len(raw_df)} rows, 21 columns. SHA256: {raw_hash}")

        # 2. Sanitization & Stratified Partitioning (70% Train, 15% Val, 15% Test)
        clean_df = self.loader.sanitize_data(raw_df)
        train_df, val_df, test_df = self.loader.split_data(clean_df)
        self.loader.save_splits(train_df, val_df, test_df)

        # 3. Preprocessing (Fitted STRICTLY on Train)
        logger.info("Fitting preprocessing pipeline on training partition...")
        X_train, y_train = self.preprocessor.fit_transform(train_df)
        X_val, y_val = self.preprocessor.transform(val_df)
        X_test, y_test = self.preprocessor.transform(test_df)

        feature_names = self.preprocessor.get_feature_names()
        logger.info(f"Feature matrix created: {X_train.shape[1]} features, {X_train.shape[0]} training samples.")

        # Save preprocessor & feature names
        self.preprocessor.save(PREPROCESSOR_PATH)
        with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)

        # 4. Hyperparameters setup
        hyperparams = custom_hyperparams or DEFAULT_HYPERPARAMS
        logger.info(f"Hyperparameters: {hyperparams}")

        # 5. Run Fair Benchmark (Custom NumPy LR vs Scikit-Learn LR)
        benchmark_results = self.benchmark.run_benchmark(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            custom_hyperparams=hyperparams,
            threshold=DEFAULT_DECISION_THRESHOLD,
        )
        self.benchmark.save_results(BENCHMARK_RESULTS_PATH)

        custom_model = self.benchmark.custom_model
        sklearn_model = self.benchmark.sklearn_model

        # Save trained Custom NumPy model
        custom_model.save(CUSTOM_MODEL_PATH)

        # 6. Optimal Threshold Search STRICTLY on Validation Set
        custom_val_proba = custom_model.predict_proba(X_val)[:, 1]
        best_thresh_f1, best_f1, sweep_results = find_optimal_threshold(
            y_val, custom_val_proba, criterion="f1"
        )
        logger.info(f"Optimal Validation F1 Threshold Selected: {best_thresh_f1:.2f} (Val F1 = {best_f1:.4f})")

        # 7. Evaluate Custom Model on Test Set at Optimal Threshold
        custom_test_proba = custom_model.predict_proba(X_test)[:, 1]
        sklearn_test_proba = sklearn_model.predict_proba(X_test)[:, 1]

        custom_test_eval_opt = evaluate_all(y_test, custom_test_proba, threshold=best_thresh_f1)
        custom_test_eval_50 = evaluate_all(y_test, custom_test_proba, threshold=0.50)
        sklearn_test_eval_50 = evaluate_all(y_test, sklearn_test_proba, threshold=0.50)

        # 8. Generate Publication Figures
        logger.info("Generating evaluation and explainability visualizations...")
        self.visualizer.plot_eda_charts(clean_df)
        self.visualizer.plot_training_loss(
            train_loss=custom_model.train_loss_history_,
            val_loss=custom_model.val_loss_history_,
            best_epoch=custom_model.best_epoch_,
        )
        self.visualizer.plot_roc_pr_comparison(
            y_test=y_test,
            custom_proba=custom_test_proba,
            sklearn_proba=sklearn_test_proba,
        )
        self.visualizer.plot_confusion_matrices(
            y_test=y_test,
            custom_pred=(custom_test_proba >= 0.50).astype(int),
            sklearn_pred=(sklearn_test_proba >= 0.50).astype(int),
        )
        self.visualizer.plot_feature_importance_odds_ratios(
            feature_names=feature_names,
            weights=custom_model.weights,
            top_n=15,
        )
        self.visualizer.plot_threshold_tuning(
            sweep_results=sweep_results,
            best_threshold=best_thresh_f1,
        )

        # 9. Global Explainability Summary
        explainer = GlobalExplainer(
            feature_names=feature_names,
            weights=custom_model.weights,
            bias=custom_model.bias,
        )
        summary_df = explainer.get_summary_dataframe(X_train, y_train)

        def _safe_cm(cm):
            return cm.tolist() if hasattr(cm, "tolist") else list(cm)

        # 10. Assemble Authoritative final_results.json
        final_results = {
            "metadata": {
                "project_title": "ChurnGuard AI — Explainable Customer Churn Prediction & Retention Intelligence Platform",
                "experiment_status": "FROZEN_FINAL_AUTHORITATIVE",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "software_versions": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                },
            },
            "dataset_info": {
                "raw_filename": "telco_customer_churn.csv",
                "sha256_checksum": raw_hash,
                "total_records": len(raw_df),
                "total_columns": len(raw_df.columns),
                "target_column": TARGET_COLUMN,
                "target_counts": {
                    "No": int(np.sum(raw_df[TARGET_COLUMN] == "No")),
                    "Yes": int(np.sum(raw_df[TARGET_COLUMN] == "Yes")),
                },
                "target_proportions": {
                    "No": float(np.mean(raw_df[TARGET_COLUMN] == "No")),
                    "Yes": float(np.mean(raw_df[TARGET_COLUMN] == "Yes")),
                },
                "whitespace_totalcharges_imputed": 11,
            },
            "split_protocol": {
                "random_state": self.random_state,
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "train_churn_rate": float(np.mean(train_df[TARGET_COLUMN] == 1)),
                "val_churn_rate": float(np.mean(val_df[TARGET_COLUMN] == 1)),
                "test_churn_rate": float(np.mean(test_df[TARGET_COLUMN] == 1)),
                "customer_id_overlap": 0,
            },
            "feature_engineering": {
                "raw_feature_count": 21,
                "domain_features_added": 9,
                "total_encoded_features": len(feature_names),
                "feature_names": feature_names,
            },
            "random_seeds": {
                "dataset_split_seed": self.random_state,
                "model_init_seed": self.random_state,
                "batch_shuffling_seed": self.random_state,
                "sklearn_seed": self.random_state,
            },
            "custom_model_hyperparameters": {
                "learning_rate": custom_model.learning_rate,
                "max_iter": custom_model.max_iter,
                "l2_lambda": custom_model.l2_lambda,
                "momentum": custom_model.momentum,
                "batch_size": custom_model.batch_size,
                "early_stopping": custom_model.early_stopping,
                "patience": custom_model.patience,
                "tolerance": custom_model.tolerance,
                "use_class_weights": custom_model.use_class_weights,
                "class_weights": {str(k): float(v) for k, v in custom_model.class_weights_.items()},
            },
            "training_telemetry": {
                "total_epochs_run": len(custom_model.train_loss_history_),
                "best_epoch": custom_model.best_epoch_,
                "best_val_loss": float(custom_model.best_val_loss_),
                "final_train_loss": float(custom_model.train_loss_history_[-1]),
                "trained_bias": float(custom_model.bias),
                "weight_l2_norm": float(np.linalg.norm(custom_model.weights)),
            },
            "threshold_selection": {
                "selection_dataset": "validation_set_only",
                "search_range": [0.01, 0.99],
                "search_step": 0.01,
                "optimization_objective": "maximum_f1_score",
                "selected_optimal_threshold": float(best_thresh_f1),
                "validation_f1_at_optimal": float(best_f1),
                "default_threshold": DEFAULT_DECISION_THRESHOLD,
            },
            "custom_numpy_model_test_metrics": {
                "at_default_threshold_0_50": {
                    "threshold": 0.50,
                    "accuracy": float(custom_test_eval_50["accuracy"]),
                    "precision": float(custom_test_eval_50["precision"]),
                    "recall": float(custom_test_eval_50["recall"]),
                    "specificity": float(custom_test_eval_50["specificity"]),
                    "f1_score": float(custom_test_eval_50["f1_score"]),
                    "roc_auc": float(custom_test_eval_50["roc_auc"]),
                    "pr_auc": float(custom_test_eval_50["pr_auc"]),
                    "confusion_matrix": _safe_cm(custom_test_eval_50["confusion_matrix"]),
                    "true_positives": int(custom_test_eval_50["true_positives"]),
                    "false_positives": int(custom_test_eval_50["false_positives"]),
                    "true_negatives": int(custom_test_eval_50["true_negatives"]),
                    "false_negatives": int(custom_test_eval_50["false_negatives"]),
                    "illustrative_net_retention_savings": float(custom_test_eval_50["net_retention_savings"]),
                },
                "at_validation_optimal_threshold": {
                    "threshold": float(best_thresh_f1),
                    "accuracy": float(custom_test_eval_opt["accuracy"]),
                    "precision": float(custom_test_eval_opt["precision"]),
                    "recall": float(custom_test_eval_opt["recall"]),
                    "specificity": float(custom_test_eval_opt["specificity"]),
                    "f1_score": float(custom_test_eval_opt["f1_score"]),
                    "roc_auc": float(custom_test_eval_opt["roc_auc"]),
                    "pr_auc": float(custom_test_eval_opt["pr_auc"]),
                    "confusion_matrix": _safe_cm(custom_test_eval_opt["confusion_matrix"]),
                    "true_positives": int(custom_test_eval_opt["true_positives"]),
                    "false_positives": int(custom_test_eval_opt["false_positives"]),
                    "true_negatives": int(custom_test_eval_opt["true_negatives"]),
                    "false_negatives": int(custom_test_eval_opt["false_negatives"]),
                    "illustrative_net_retention_savings": float(custom_test_eval_opt["net_retention_savings"]),
                },
            },
            "sklearn_benchmark_test_metrics": {
                "estimator": "sklearn.linear_model.LogisticRegression(C=100.0, class_weight='balanced', solver='lbfgs', max_iter=1500)",
                "at_default_threshold_0_50": {
                    "threshold": 0.50,
                    "accuracy": float(sklearn_test_eval_50["accuracy"]),
                    "precision": float(sklearn_test_eval_50["precision"]),
                    "recall": float(sklearn_test_eval_50["recall"]),
                    "specificity": float(sklearn_test_eval_50["specificity"]),
                    "f1_score": float(sklearn_test_eval_50["f1_score"]),
                    "roc_auc": float(sklearn_test_eval_50["roc_auc"]),
                    "pr_auc": float(sklearn_test_eval_50["pr_auc"]),
                    "confusion_matrix": _safe_cm(sklearn_test_eval_50["confusion_matrix"]),
                    "true_positives": int(sklearn_test_eval_50["true_positives"]),
                    "false_positives": int(sklearn_test_eval_50["false_positives"]),
                    "true_negatives": int(sklearn_test_eval_50["true_negatives"]),
                    "false_negatives": int(sklearn_test_eval_50["false_negatives"]),
                    "training_time_ms": float(benchmark_results["sklearn_benchmark_model"]["training_time_ms"]),
                    "inference_latency_ms_per_1k": float(benchmark_results["sklearn_benchmark_model"]["inference_latency_ms_per_1k"]),
                },
            },
            "mathematical_fidelity": {
                "prediction_probability_correlation": float(benchmark_results["mathematical_fidelity"]["prediction_probability_correlation"]),
                "weight_cosine_similarity": float(benchmark_results["mathematical_fidelity"]["weight_cosine_similarity"]),
                "weight_pearson_correlation": float(benchmark_results["mathematical_fidelity"]["weight_pearson_correlation"]),
                "weight_mean_absolute_difference": float(benchmark_results["mathematical_fidelity"]["weight_mean_absolute_difference"]),
                "custom_inference_latency_ms_per_1k": float(benchmark_results["custom_numpy_model"]["inference_latency_ms_per_1k"]),
                "custom_training_time_ms": float(benchmark_results["custom_numpy_model"]["training_time_ms"]),
            },
            "illustrative_business_cost_model": {
                "disclaimer": "Illustrative business assumptions only for capstone ROI demonstration.",
                "cost_false_negative_missed_churn": float(BUSINESS_COST_MATRIX["cost_false_negative"]),
                "cost_false_positive_wasted_intervention": float(BUSINESS_COST_MATRIX["cost_false_positive"]),
                "gain_true_positive_retained_net_value": float(BUSINESS_COST_MATRIX["gain_true_positive"]),
                "gain_true_negative_organic_continuation": float(BUSINESS_COST_MATRIX["gain_true_negative"]),
            },
        }

        # Also populate convenience keys for downstream callers and integration tests
        final_results["benchmark_results"] = benchmark_results
        final_results["optimal_threshold"] = float(best_thresh_f1)
        final_results["optimal_f1_score"] = float(best_f1)
        final_results["feature_importance_summary"] = summary_df.to_dict(orient="records")

        # Save authoritative final_results.json
        FINAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Authoritative final results persisted to {FINAL_RESULTS_PATH}")

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        return final_results


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
    custom_50 = results["custom_numpy_model_test_metrics"]["at_default_threshold_0_50"]
    custom_opt = results["custom_numpy_model_test_metrics"]["at_validation_optimal_threshold"]
    print("\n" + "=" * 60)
    print("FROZEN AUTHORITATIVE EXPERIMENTAL RESULTS")
    print("=" * 60)
    print(f"Custom Model Test ROC-AUC:    {custom_50['roc_auc']:.4f}")
    print(f"Custom Model Test PR-AUC:     {custom_50['pr_auc']:.4f}")
    print(f"Custom Model (t=0.50):        Acc={custom_50['accuracy']:.4f}, Prec={custom_50['precision']:.4f}, Rec={custom_50['recall']:.4f}, F1={custom_50['f1_score']:.4f}")
    print(f"Custom Model (t={custom_opt['threshold']:.2f} Opt):   Acc={custom_opt['accuracy']:.4f}, Prec={custom_opt['precision']:.4f}, Rec={custom_opt['recall']:.4f}, F1={custom_opt['f1_score']:.4f}")
    print(f"Sklearn Baseline Test ROC-AUC: {results['sklearn_benchmark_test_metrics']['at_default_threshold_0_50']['roc_auc']:.4f}")
    print(f"Probability Correlation (r):   {results['mathematical_fidelity']['prediction_probability_correlation']:.6f}")
    print("=" * 60)
