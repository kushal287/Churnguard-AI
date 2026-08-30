"""
Data Loader and Ingestion Module for ChurnGuard AI.
Handles dataset loading, validation, sanitization, and stratified partitioning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_RATIO,
    TRAIN_DATA_PATH,
    TRAIN_RATIO,
    VAL_DATA_PATH,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataLoader:
    """Robust data loader for customer churn dataset."""

    def __init__(self, raw_data_path: Optional[Path] = None, random_state: int = RANDOM_SEED):
        self.raw_data_path = Path(raw_data_path) if raw_data_path else RAW_DATA_PATH
        self.random_state = random_state

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV dataset with integrity checks."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded raw dataset with shape: {df.shape}")
        self._validate_raw_schema(df)
        return df

    def _validate_raw_schema(self, df: pd.DataFrame) -> None:
        """Validate expected columns and minimal sanity constraints."""
        required_cols = [ID_COLUMN, TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        if df[ID_COLUMN].duplicated().any():
            dup_count = df[ID_COLUMN].duplicated().sum()
            raise ValueError(f"Found {dup_count} duplicate customer IDs in raw data!")

    def sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize raw dataset:
        1. Fix whitespace TotalCharges for tenure=0 to 0.0
        2. Convert TotalCharges to float64
        3. Convert SeniorCitizen to categorical/string for unified encoding
        4. Convert Churn target 'Yes'/'No' to binary 1/0 integers
        """
        clean_df = df.copy()

        # Handle whitespace or null in TotalCharges using pd.to_numeric
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce")
        # Tenure 0 customers logically have 0 TotalCharges
        clean_df.loc[clean_df["TotalCharges"].isna() & (clean_df["tenure"] == 0), "TotalCharges"] = 0.0
        # If any other NaN remains, impute with MonthlyCharges * tenure
        if clean_df["TotalCharges"].isna().any():
            clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(
                clean_df["MonthlyCharges"] * clean_df["tenure"]
            )

        clean_df["TotalCharges"] = clean_df["TotalCharges"].astype(float)
        clean_df["tenure"] = clean_df["tenure"].astype(int)
        clean_df["MonthlyCharges"] = clean_df["MonthlyCharges"].astype(float)
        # Standardize SeniorCitizen to string category
        clean_df["SeniorCitizen"] = clean_df["SeniorCitizen"].astype(str)

        # Binary encode target robustly across all pandas dtypes
        if TARGET_COLUMN in clean_df.columns:
            clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].apply(
                lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0
            ).astype(int)

        logger.info("Data sanitization completed successfully.")
        return clean_df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified 3-way split: Train (70%), Validation (15%), Test (15%).
        Guarantees exact stratified representation of churn rate across all splits.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=self.random_state,
            stratify=df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None,
        )

        # Second split: Val vs Test
        val_rel_ratio = val_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_rel_ratio),
            random_state=self.random_state,
            stratify=temp_df[TARGET_COLUMN] if TARGET_COLUMN in temp_df.columns else None,
        )

        logger.info(
            f"Dataset Split: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
            f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
            f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
        )

        # Validate churn stratification
        train_churn_rate = train_df[TARGET_COLUMN].mean()
        val_churn_rate = val_df[TARGET_COLUMN].mean()
        test_churn_rate = test_df[TARGET_COLUMN].mean()
        logger.info(
            f"Churn Rates - Train: {train_churn_rate:.3f}, Val: {val_churn_rate:.3f}, Test: {test_churn_rate:.3f}"
        )

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Persist processed splits to CSV files."""
        out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved dataset splits to {out_dir}")
        return {"train": train_path, "val": val_path, "test": test_path}

    def load_splits(
        self, input_dir: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load persisted train, validation, and test splits."""
        inp_dir = Path(input_dir) if input_dir else PROCESSED_DATA_DIR
        train_df = pd.read_csv(inp_dir / "train.csv")
        val_df = pd.read_csv(inp_dir / "val.csv")
        test_df = pd.read_csv(inp_dir / "test.csv")
        return train_df, val_df, test_df
