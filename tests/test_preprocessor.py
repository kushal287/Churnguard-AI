"""
Tests for Data Loader and Data Preprocessor (Zero-Leakage & Robustness).
Includes regression tests for batch scoring, missing categories, unknown categories, and type coercion.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


class TestPreprocessorAndLoader:
    """Validate preprocessing isolation, missing value fixes, unknown category safety, and transformation shapes."""

    @pytest.fixture
    def sample_data(self):
        """Create representative raw mock dataframe with known quirks."""
        data = {
            "customerID": [f"ID_{i}" for i in range(10)],
            "gender": ["Male", "Female"] * 5,
            "SeniorCitizen": ["0", "1"] * 5,
            "Partner": ["Yes", "No"] * 5,
            "Dependents": ["No", "Yes"] * 5,
            "tenure": [0, 12, 24, 36, 48, 60, 72, 1, 6, 18],
            "PhoneService": ["Yes", "No"] * 5,
            "MultipleLines": ["No phone service", "Yes"] * 5,
            "InternetService": ["DSL", "Fiber optic"] * 5,
            "OnlineSecurity": ["Yes", "No"] * 5,
            "OnlineBackup": ["No", "Yes"] * 5,
            "DeviceProtection": ["Yes", "No"] * 5,
            "TechSupport": ["No", "Yes"] * 5,
            "StreamingTV": ["Yes", "No"] * 5,
            "StreamingMovies": ["No", "Yes"] * 5,
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"] * 2,
            "PaperlessBilling": ["Yes", "No"] * 5,
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"] * 2,
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 19.50, 45.20, 80.00],
            "TotalCharges": [" ", "683.40", "1292.40", "1522.80", "3393.60", "5979.00", "6415.20", "19.50", "271.20", "1440.00"],
            "Churn": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        }
        return pd.DataFrame(data)

    def test_totalcharges_whitespace_conversion(self, sample_data):
        """Verify loader converts blank TotalCharges for tenure=0 to 0.0 float."""
        loader = DataLoader()
        clean_df = loader.sanitize_data(sample_data)

        assert clean_df["TotalCharges"].dtype == np.float64
        # tenure == 0 row should have TotalCharges == 0.0
        tenure_0_tc = clean_df.loc[clean_df["tenure"] == 0, "TotalCharges"].values[0]
        assert tenure_0_tc == 0.0

    def test_zero_leakage_scaler_fit(self, sample_data):
        """Verify transforming new data does NOT alter training statistics."""
        loader = DataLoader()
        clean_df = loader.sanitize_data(sample_data)

        train_df = clean_df.iloc[:6].copy()
        test_df = clean_df.iloc[6:].copy()

        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(train_df)

        train_means = preprocessor.scaler.mean_.copy()
        train_stds = preprocessor.scaler.scale_.copy()

        # Transform test data
        _ = preprocessor.transform(test_df)

        # Ensure scaler parameters were unchanged
        assert np.allclose(preprocessor.scaler.mean_, train_means), "Scaler mean was modified by transform!"
        assert np.allclose(preprocessor.scaler.scale_, train_stds), "Scaler std was modified by transform!"

    def test_single_record_transform_shape(self, sample_data):
        """Verify single customer dictionary transform matches feature dimension."""
        loader = DataLoader()
        clean_df = loader.sanitize_data(sample_data)

        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        X_train, _ = preprocessor.fit_transform(clean_df)

        single_rec = clean_df.iloc[0].to_dict()
        X_single = preprocessor.transform_single_record(single_rec)

        assert X_single.shape == (1, X_train.shape[1]), f"Expected (1, {X_train.shape[1]}), got {X_single.shape}"
        assert not np.isnan(X_single).any(), "Single record transform produced NaNs"

    # =========================================================================
    # REGRESSION TESTS FOR BATCH SCORING & TYPE SAFETY (TASK 4)
    # =========================================================================

    def test_batch_dataframe_with_csv_integer_types(self, sample_data):
        """
        Regression Test for original Batch Scoring failure:
        SeniorCitizen in raw CSV is int64 (0/1). When passed directly to preprocessor.transform(),
        it must coerce SeniorCitizen to string without raising TypeError: ufunc 'isnan'.
        """
        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(sample_data)

        # Create raw batch with int64 SeniorCitizen and unparsed TotalCharges string
        raw_batch = sample_data.copy()
        raw_batch["SeniorCitizen"] = raw_batch["SeniorCitizen"].astype(int)  # int64 dtype

        X_batch, _ = preprocessor.transform(raw_batch)
        assert X_batch.shape == (len(sample_data), len(preprocessor.get_feature_names()))
        assert not np.isnan(X_batch).any()

    def test_batch_dataframe_with_missing_categorical_values(self, sample_data):
        """
        Verify that batch inputs with NaN, None, or empty string categoricals
        are gracefully handled as 'Missing' and do not crash OneHotEncoder.
        """
        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(sample_data)

        batch_with_nulls = sample_data.copy()
        batch_with_nulls.loc[0, "PaymentMethod"] = np.nan
        batch_with_nulls.loc[1, "InternetService"] = None
        batch_with_nulls.loc[2, "Contract"] = ""
        batch_with_nulls.loc[3, "SeniorCitizen"] = np.nan

        X_batch, _ = preprocessor.transform(batch_with_nulls)
        assert X_batch.shape == (len(sample_data), len(preprocessor.get_feature_names()))
        assert not np.isnan(X_batch).any()

    def test_batch_dataframe_with_unknown_categorical_values(self, sample_data):
        """
        Verify that unseen categories (e.g. Contract='Ten Year', PaymentMethod='Bitcoin')
        are safely ignored and encoded as 0s by handle_unknown='ignore'.
        """
        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(sample_data)

        batch_unknown = sample_data.copy()
        batch_unknown.loc[0, "Contract"] = "Five Year Ultra Long"
        batch_unknown.loc[1, "PaymentMethod"] = "Cryptocurrency Wallet"
        batch_unknown.loc[2, "InternetService"] = "Satellite Quantum"

        X_batch, _ = preprocessor.transform(batch_unknown)
        assert X_batch.shape == (len(sample_data), len(preprocessor.get_feature_names()))
        assert not np.isnan(X_batch).any()

    def test_batch_dataframe_with_missing_numerical_values(self, sample_data):
        """
        Verify that missing numerical values in inference batches are imputed with training medians.
        """
        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(sample_data)

        batch_num_nulls = sample_data.copy()
        batch_num_nulls.loc[0, "MonthlyCharges"] = np.nan
        batch_num_nulls.loc[1, "TotalCharges"] = np.nan
        batch_num_nulls.loc[2, "tenure"] = np.nan

        X_batch, _ = preprocessor.transform(batch_num_nulls)
        assert X_batch.shape == (len(sample_data), len(preprocessor.get_feature_names()))
        assert not np.isnan(X_batch).any()

    def test_single_and_batch_equivalence(self, sample_data):
        """
        Verify that a single customer processed via transform_single_record()
        produces the exact same numerical vector as when processed in a batch via transform().
        """
        preprocessor = DataPreprocessor(apply_feature_engineering=True)
        preprocessor.fit(sample_data)

        rec = sample_data.iloc[2].to_dict()
        X_single = preprocessor.transform_single_record(rec)
        X_batch, _ = preprocessor.transform(pd.DataFrame([rec]))

        assert np.array_equal(X_single, X_batch), "Single customer transform does not equal 1-row batch transform!"
