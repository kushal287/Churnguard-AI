"""
Preprocessing and Scaling Pipeline for ChurnGuard AI.
Guarantees strict zero-data-leakage transformation and robust categorical type safety.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
)
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Stateful data preprocessor that isolates feature scaling and one-hot encoding.
    Ensures zero data leakage by fitting strictly on training data.
    Guarantees robust type safety across batch CSVs, missing categoricals, and single-instance scoring.
    """

    def __init__(self, apply_feature_engineering: bool = True):
        self.apply_feature_engineering = apply_feature_engineering
        self.feature_engineer = FeatureEngineer() if apply_feature_engineering else None

        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None

        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_imputer_values_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _determine_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical column lists after feature engineering."""
        exclude = {ID_COLUMN, TARGET_COLUMN}
        
        # Categorical columns
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if self.apply_feature_engineering and "tenure_cohort" in df.columns:
            if "tenure_cohort" not in cat_cols:
                cat_cols.append("tenure_cohort")

        # Numerical columns
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        if self.apply_feature_engineering:
            eng_num = [
                "monthly_to_total_ratio",
                "monthly_charge_discrepancy",
                "total_services_count",
                "protection_services_count",
                "streaming_services_count",
                "has_internet",
                "is_solo_senior",
                "high_risk_fiber_m2m"
            ]
            for c in eng_num:
                if c in df.columns and c not in num_cols:
                    num_cols.append(c)

        return num_cols, cat_cols

    def _prepare_dataframe(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Sanitize and normalize raw column dtypes before feature engineering and transformation.
        Guarantees uniform string types for categoricals and valid floats for numericals.
        """
        data = df.copy()

        # 1. Base Numerical Sanitization
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
            if "tenure" in data.columns:
                data.loc[data["TotalCharges"].isna() & (data["tenure"] == 0), "TotalCharges"] = 0.0
            data["TotalCharges"] = data["TotalCharges"].fillna(0.0).astype(float)

        if "tenure" in data.columns:
            data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").fillna(0).astype(int)

        if "MonthlyCharges" in data.columns:
            data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)

        # 2. Base Categorical Normalization (Cast SeniorCitizen and other categoricals to string)
        if "SeniorCitizen" in data.columns:
            data["SeniorCitizen"] = data["SeniorCitizen"].fillna("0").astype(str).str.strip()

        for c in CATEGORICAL_FEATURES:
            if c in data.columns and c != "SeniorCitizen":
                data[c] = data[c].fillna("Missing").astype(str).str.strip()

        # 3. Domain Feature Engineering
        if self.apply_feature_engineering and self.feature_engineer:
            data = self.feature_engineer.transform(data)

        # 4. Standardize all categorical columns (including engineered tenure_cohort) to uniform strings
        for c in self.cat_cols:
            if c in data.columns:
                data[c] = data[c].fillna("Missing").astype(str).str.strip()

        # 5. Standardize all numerical columns and handle missing values deterministically
        for c in self.num_cols:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")
                if is_training:
                    median_val = float(data[c].median()) if not data[c].isna().all() else 0.0
                    self.num_imputer_values_[c] = median_val
                    data[c] = data[c].fillna(median_val).astype(float)
                else:
                    impute_val = self.num_imputer_values_.get(c, 0.0)
                    data[c] = data[c].fillna(impute_val).astype(float)

        return data

    def fit(self, train_df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit transformers strictly on the training partition.
        """
        # 1. Prepare normalized DataFrame and determine active column lists
        df = self._prepare_dataframe(train_df, is_training=True)
        self.num_cols, self.cat_cols = self._determine_columns(df)

        # 2. Fit numerical standard scaler
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_cols])

        # 3. Fit categorical OneHotEncoder with handle_unknown='ignore'
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        self.encoder.fit(df[self.cat_cols])

        # 4. Extract encoded feature names
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = self.num_cols + cat_feature_names
        self.is_fitted = True

        logger.info(
            f"Preprocessor fitted successfully. Total features: {len(self.feature_names_)} "
            f"({len(self.num_cols)} numerical, {len(cat_feature_names)} one-hot categorical)."
        )
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform any input DataFrame into standardized NumPy feature matrix X and target y.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform()!")

        data = self._prepare_dataframe(df, is_training=False)

        # Ensure all required columns exist
        missing_num = [c for c in self.num_cols if c not in data.columns]
        missing_cat = [c for c in self.cat_cols if c not in data.columns]
        if missing_num or missing_cat:
            raise ValueError(f"Missing columns for transform: Num={missing_num}, Cat={missing_cat}")

        # Scale numericals
        X_num = self.scaler.transform(data[self.num_cols])

        # Encode categoricals
        X_cat = self.encoder.transform(data[self.cat_cols])

        # Concatenate horizontally
        X = np.hstack([X_num, X_cat])

        # Extract target if present
        y = None
        if TARGET_COLUMN in data.columns:
            # Map binary target robustly if string
            if data[TARGET_COLUMN].dtype == object or isinstance(data[TARGET_COLUMN].iloc[0], str):
                y_series = data[TARGET_COLUMN].apply(lambda v: 1 if str(v).strip().lower() in ["yes", "1", "true"] else 0)
            else:
                y_series = data[TARGET_COLUMN]
            y = y_series.values.astype(np.float64)

        return X, y

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convenience method to fit on train_df and return transformed matrices."""
        return self.fit(train_df).transform(train_df)

    def transform_single_record(self, record_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single customer dictionary / series into a (1, d) NumPy feature vector.
        """
        df_single = pd.DataFrame([record_dict])
        X, _ = self.transform(df_single)
        return X

    def get_feature_names(self) -> List[str]:
        """Return the final list of ordered feature names."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")
        return self.feature_names_

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Serialize preprocessor pipeline to disk."""
        path = Path(filepath) if filepath else PREPROCESSOR_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")
        return path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "DataPreprocessor":
        """Deserialize preprocessor pipeline from disk."""
        path = Path(filepath) if filepath else PREPROCESSOR_PATH
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {path}")
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
