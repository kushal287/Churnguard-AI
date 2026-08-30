"""
Unit & Integration Tests for User Dataset Upload Workflow & Invariance Guarantees.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.config import CUSTOM_MODEL_PATH, FINAL_RESULTS_PATH, PREPROCESSOR_PATH
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DatasetValidator
from src.explainability.retention_playbook import RetentionPlaybook
from src.models.custom_logistic_regression import CustomLogisticRegression


class TestUserDatasetWorkflow:
    """Test suite validating schema validation, inference on user data, and model invariance."""

    @pytest.fixture
    def system_artifacts(self):
        """Load frozen model and preprocessor."""
        assert CUSTOM_MODEL_PATH.exists()
        assert PREPROCESSOR_PATH.exists()
        model = CustomLogisticRegression.load(CUSTOM_MODEL_PATH)
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        playbook = RetentionPlaybook(threshold=0.58)
        return model, preprocessor, playbook

    def test_valid_labeled_csv_validation(self):
        """Test validation on a valid labeled dataset."""
        df_valid = DatasetValidator.generate_example_dataset(n_samples=10)
        assert "Churn" in df_valid.columns
        report = DatasetValidator.validate_dataset(df_valid)

        assert report["is_compatible"] is True
        assert report["is_labeled"] is True
        assert len(report["errors"]) == 0
        assert report["total_rows"] == 10

    def test_valid_unlabeled_csv_validation(self):
        """Test validation on a valid unlabeled dataset (prediction-only mode)."""
        df_unlabeled = DatasetValidator.generate_example_dataset(n_samples=10).drop(columns=["Churn"])
        report = DatasetValidator.validate_dataset(df_unlabeled)

        assert report["is_compatible"] is True
        assert report["is_labeled"] is False
        assert len(report["errors"]) == 0

    def test_missing_required_column(self):
        """Test rejection when a required feature column is missing."""
        df_bad = DatasetValidator.generate_example_dataset(n_samples=5).drop(columns=["Contract"])
        report = DatasetValidator.validate_dataset(df_bad)

        assert report["is_compatible"] is False
        assert any("Contract" in err for err in report["errors"])

    def test_extra_columns_allowed(self):
        """Test that extra non-standard columns do not crash validation and are recorded in statistics."""
        df_extra = DatasetValidator.generate_example_dataset(n_samples=5)
        df_extra["ExtraSurveyScore"] = [9, 8, 7, 6, 5]
        df_extra["ZipCode"] = ["90210", "90211", "90212", "90213", "90214"]

        report = DatasetValidator.validate_dataset(df_extra)
        assert report["is_compatible"] is True
        assert report["statistics"]["extra_columns_count"] == 2

    def test_wrong_numeric_type(self):
        """Test rejection when numeric columns contain unparseable strings."""
        df_bad = DatasetValidator.generate_example_dataset(n_samples=5)
        df_bad["MonthlyCharges"] = df_bad["MonthlyCharges"].astype(object)
        df_bad.loc[0, "MonthlyCharges"] = "INVALID_PRICE_ABC"

        report = DatasetValidator.validate_dataset(df_bad)
        assert report["is_compatible"] is False
        assert any("MonthlyCharges" in err for err in report["errors"])

    def test_impossible_negative_numeric_value(self):
        """Test rejection when numeric columns contain negative numbers."""
        df_bad = DatasetValidator.generate_example_dataset(n_samples=5)
        df_bad.loc[0, "tenure"] = -5

        report = DatasetValidator.validate_dataset(df_bad)
        assert report["is_compatible"] is False
        assert any("negative" in err.lower() for err in report["errors"])

    def test_duplicate_customer_ids_warning(self):
        """Test warning generated on duplicate customer IDs."""
        df_dupe = DatasetValidator.generate_example_dataset(n_samples=5)
        df_dupe.loc[1, "customerID"] = df_dupe.loc[0, "customerID"]

        report = DatasetValidator.validate_dataset(df_dupe)
        assert report["is_compatible"] is True  # Compatible with warning
        assert any("duplicate" in w.lower() for w in report["warnings"])

    def test_invalid_churn_target_value(self):
        """Test rejection when Churn target contains unrecognized values."""
        df_bad = DatasetValidator.generate_example_dataset(n_samples=5)
        df_bad.loc[0, "Churn"] = "Maybe"

        report = DatasetValidator.validate_dataset(df_bad)
        assert report["is_compatible"] is False
        assert any("Churn" in err for err in report["errors"])

    def test_empty_csv_rejection(self):
        """Test rejection when an empty DataFrame is supplied."""
        df_empty = pd.DataFrame()
        report = DatasetValidator.validate_dataset(df_empty)

        assert report["is_compatible"] is False
        assert any("empty" in err.lower() for err in report["errors"])

    def test_unknown_categorical_value_handling(self, system_artifacts):
        """Test that unknown categories generate warnings and transform safely without failing."""
        model, preprocessor, _ = system_artifacts
        df_unknown = DatasetValidator.generate_example_dataset(n_samples=5)
        df_unknown.loc[0, "InternetService"] = "SatelliteQuantum5G"  # Unseen category

        report = DatasetValidator.validate_dataset(df_unknown)
        assert report["is_compatible"] is True
        assert any("unrecognized" in w.lower() for w in report["warnings"])

        # Ensure transform runs safely without exception
        X_mat, _ = preprocessor.transform(df_unknown)
        proba = model.predict_proba(X_mat)
        assert proba.shape == (5, 2)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_csv_template_generation_integrity(self):
        """Verify that the generated CSV template matches expected columns and data formats."""
        template_df = DatasetValidator.generate_csv_template()
        assert len(template_df) == 2
        for col in DatasetValidator.REQUIRED_FEATURES:
            assert col in template_df.columns
        assert "customerID" in template_df.columns
        assert "Churn" in template_df.columns

    def test_model_and_preprocessor_invariance_guarantee(self, system_artifacts):
        """CRITICAL: Verify that scoring user data NEVER alters preprocessor scalers or model weights."""
        model, preprocessor, _ = system_artifacts

        # Snapshot parameters before user inference
        orig_weights = model.weights.copy()
        orig_bias = float(model.bias)
        orig_scaler_mean = preprocessor.scaler.mean_.copy()
        orig_scaler_scale = preprocessor.scaler.scale_.copy()

        # Score random user cohort
        user_df = DatasetValidator.generate_example_dataset(n_samples=50)
        X_mat, _ = preprocessor.transform(user_df)
        _ = model.predict_proba(X_mat)

        # Assert bitwise identical parameters after scoring
        assert np.array_equal(model.weights, orig_weights), "Model weights were modified during user inference!"
        assert model.bias == orig_bias, "Model bias was modified during user inference!"
        assert np.array_equal(preprocessor.scaler.mean_, orig_scaler_mean), "Preprocessor scaler mean was refitted!"
        assert np.array_equal(preprocessor.scaler.scale_, orig_scaler_scale), "Preprocessor scaler scale was refitted!"

    def test_official_final_results_json_remains_authoritative(self):
        """Verify that final_results.json remains untouched and valid."""
        assert FINAL_RESULTS_PATH.exists()
        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        assert res["metadata"]["experiment_status"] == "FROZEN_FINAL_AUTHORITATIVE"
        c50 = res["custom_numpy_model_test_metrics"]["at_default_threshold_0_50"]
        assert c50["roc_auc"] == 0.8451590417140551
        assert c50["illustrative_net_retention_savings"] == 64350.0
