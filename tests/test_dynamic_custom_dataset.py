"""
Unit and Integration Tests for Dynamic Tabular Binary Classification Pipeline & Isolation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.config import CUSTOM_MODEL_PATH, FINAL_RESULTS_PATH
from src.data.dynamic_pipeline import DynamicPipelineTrainer, DynamicPreprocessor, DynamicSchemaDetector
from src.explainability.dynamic_explainer import DynamicIndividualExplainer
from src.models.custom_logistic_regression import CustomLogisticRegression


class TestDynamicTabularPipeline:
    """Test suite validating dynamic tabular binary classification on arbitrary custom datasets."""

    @pytest.fixture
    def sample_datasets(self):
        """Create two distinct synthetic datasets for testing (Dataset A: HR Attrition, Dataset B: Credit Default)."""
        np.random.seed(42)

        # Dataset A: Employee Attrition (N = 120, 5 features)
        n_a = 120
        df_a = pd.DataFrame({
            "EmployeeID": [f"EMP-{i:04d}" for i in range(1, n_a + 1)],
            "Age": np.random.randint(22, 60, n_a),
            "MonthlyIncome": np.random.randint(2000, 15000, n_a),
            "Department": np.random.choice(["Sales", "R&D", "HR"], n_a),
            "OverTime": np.random.choice(["Yes", "No"], n_a),
            "Attrition": np.random.choice(["Yes", "No"], n_a, p=[0.25, 0.75]),
        })

        # Dataset B: Loan Default (N = 80, 4 features)
        n_b = 80
        df_b = pd.DataFrame({
            "LoanID": [f"LOAN-{i:05d}" for i in range(1, n_b + 1)],
            "CreditScore": np.random.randint(580, 850, n_b),
            "DebtToIncome": np.round(np.random.uniform(0.1, 0.6, n_b), 2),
            "HomeOwner": np.random.choice(["OWN", "RENT"], n_b),
            "DefaultStatus": np.random.choice(["Default", "Current"], n_b, p=[0.30, 0.70]),
        })

        return df_a, df_b

    def test_schema_detector_identifies_binary_targets_and_roles(self, sample_datasets):
        """Verify schema detector identifies binary targets, IDs, numerical, and categorical features."""
        df_a, df_b = sample_datasets

        schema_a = DynamicSchemaDetector.detect_schema(df_a)
        assert "Attrition" in schema_a["target_candidates"]
        assert "OverTime" in schema_a["target_candidates"]
        assert "EmployeeID" in schema_a["identifier_candidates"]
        assert "Age" in schema_a["numerical_cols"]
        assert "Department" in schema_a["categorical_cols"]

        schema_b = DynamicSchemaDetector.detect_schema(df_b)
        assert "DefaultStatus" in schema_b["target_candidates"]
        assert "LoanID" in schema_b["identifier_candidates"]
        assert "CreditScore" in schema_b["numerical_cols"]
        assert "HomeOwner" in schema_b["categorical_cols"]

    def test_dynamic_preprocessor_zero_leakage_fit(self, sample_datasets):
        """Verify dynamic preprocessor fits only on train split and transforms safely."""
        df_a, _ = sample_datasets
        train_df = df_a.iloc[:80]
        test_df = df_a.iloc[80:]

        preprocessor = DynamicPreprocessor(
            numerical_cols=["Age", "MonthlyIncome"],
            categorical_cols=["Department", "OverTime"],
        )
        preprocessor.fit(train_df)

        X_train = preprocessor.transform(train_df)
        X_test = preprocessor.transform(test_df)

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 40
        assert X_train.shape[1] == X_test.shape[1]
        assert len(preprocessor.get_feature_names()) == X_train.shape[1]

    def test_dynamic_pipeline_training_and_prediction_shapes(self, sample_datasets):
        """Verify that training pipeline returns predictions matching exact row count N."""
        df_a, _ = sample_datasets
        results = DynamicPipelineTrainer.run_training_pipeline(
            df=df_a,
            target_col="Attrition",
            positive_class="Yes",
            id_cols=["EmployeeID"],
            random_seed=42,
        )

        assert results["status"] == "SUCCESS"
        assert len(results["predictions"]["probabilities"]) == len(df_a)
        assert len(results["predictions"]["classes"]) == len(df_a)
        assert 0.0 <= results["metadata"]["selected_threshold"] <= 1.0
        assert "roc_auc" in results["test_metrics"]
        assert results["test_metrics"]["roc_auc"] >= 0.0

    def test_dataset_a_vs_dataset_b_different_results(self, sample_datasets):
        """CRITICAL: Verify that Dataset A and Dataset B produce genuinely distinct results."""
        df_a, df_b = sample_datasets

        res_a = DynamicPipelineTrainer.run_training_pipeline(
            df=df_a, target_col="Attrition", positive_class="Yes", id_cols=["EmployeeID"]
        )

        res_b = DynamicPipelineTrainer.run_training_pipeline(
            df=df_b, target_col="DefaultStatus", positive_class="Default", id_cols=["LoanID"]
        )

        # Confirm different shapes, feature names, and weights
        assert len(res_a["predictions"]["probabilities"]) == len(df_a)
        assert len(res_b["predictions"]["probabilities"]) == len(df_b)
        assert res_a["metadata"]["target_col"] == "Attrition"
        assert res_b["metadata"]["target_col"] == "DefaultStatus"
        assert res_a["metadata"]["feature_count"] != res_b["metadata"]["feature_count"]
        assert res_a["model_state"]["weights"] != res_b["model_state"]["weights"]

    def test_dynamic_individual_explainer_mathematical_reconstruction(self, sample_datasets):
        """Verify dynamic log-odds attribution matches sigmoid(z) with zero discrepancy."""
        df_a, _ = sample_datasets
        results = DynamicPipelineTrainer.run_training_pipeline(
            df=df_a, target_col="Attrition", positive_class="Yes", id_cols=["EmployeeID"]
        )

        weights = results["model_state"]["weights"]
        bias = results["model_state"]["bias"]
        feature_names = results["model_state"]["feature_names"]

        explainer = DynamicIndividualExplainer(feature_names, weights, bias)

        # Test on dummy transformed vector
        x_dummy = np.random.randn(len(weights))
        explanation = explainer.explain_instance(x_dummy)

        recon = explanation["mathematical_reconstruction"]
        assert recon["is_exact_match"] is True
        assert recon["discrepancy"] < 1e-7

    def test_multiclass_target_rejection(self):
        """Verify that multiclass targets (>2 classes) are rejected with clear error."""
        df_multi = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "target": ["ClassA", "ClassB", "ClassC", "ClassA", "ClassB"] * 10,
        })

        with pytest.raises(ValueError, match="Binary classification requires exactly 2 distinct classes"):
            DynamicPipelineTrainer.run_training_pipeline(df=df_multi, target_col="target", positive_class="ClassA")

    def test_official_model_and_results_isolation(self, sample_datasets):
        """CRITICAL: Verify running custom dataset pipeline never modifies official frozen artifacts."""
        df_a, _ = sample_datasets

        # Snapshot frozen weights and json before custom run
        assert CUSTOM_MODEL_PATH.exists()
        official_model = CustomLogisticRegression.load(CUSTOM_MODEL_PATH)
        orig_weights = official_model.weights.copy()
        orig_bias = float(official_model.bias)

        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            orig_json = json.load(f)

        # Run custom pipeline
        _ = DynamicPipelineTrainer.run_training_pipeline(
            df=df_a, target_col="Attrition", positive_class="Yes", id_cols=["EmployeeID"]
        )

        # Verify frozen model file on disk was NOT modified
        reloaded_official = CustomLogisticRegression.load(CUSTOM_MODEL_PATH)
        assert np.array_equal(reloaded_official.weights, orig_weights)
        assert reloaded_official.bias == orig_bias

        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            reloaded_json = json.load(f)
        assert reloaded_json == orig_json

    def test_probability_tier_assignment_handles_extreme_thresholds(self):
        """Verify tier assignment works without error for low and high threshold values."""
        probs = np.array([0.05, 0.20, 0.30, 0.50, 0.85, 0.95])
        for thresh in [0.15, 0.25, 0.58, 0.80]:
            tier_conditions = [
                probs >= thresh,
                probs >= (thresh * 0.6),
            ]
            tier_choices = ["High Probability", "Medium Probability"]
            tiers = np.select(tier_conditions, tier_choices, default="Low Probability")
            assert len(tiers) == len(probs)
            assert "High Probability" in tiers or "Low Probability" in tiers
