"""
Dynamic Tabular Machine Learning Pipeline for Custom User Datasets (Mode 2).
Supports general tabular binary classification datasets (Churn, Fraud, Attrition, Default, Medical, etc.)
using the project's from-scratch NumPy Logistic Regression classifier.
Enforces zero data leakage: preprocessing is fitted strictly on the training partition.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.models.custom_logistic_regression import CustomLogisticRegression


class DynamicSchemaDetector:
    """Inspects arbitrary tabular datasets and classifies columns into semantic ML roles."""

    @staticmethod
    def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a DataFrame and detect:
        - Target candidates (columns with exactly 2 unique non-null values)
        - Potential identifier columns
        - Numerical features
        - Categorical features
        - Unsupported columns (e.g. constant, high-cardinality free text)
        """
        if df is None or df.empty:
            return {
                "total_rows": 0,
                "total_cols": 0,
                "target_candidates": [],
                "identifier_candidates": [],
                "numerical_cols": [],
                "categorical_cols": [],
                "unsupported_cols": [],
            }

        total_rows = len(df)
        target_candidates = []
        identifier_candidates = []
        numerical_cols = []
        categorical_cols = []
        unsupported_cols = []

        common_id_names = {
            "id", "customerid", "employeeid", "loanid", "cust_id", "user_id", "userid",
            "uuid", "account_id", "guid", "member_id", "client_id", "transaction_id"
        }

        for col in df.columns:
            series = df[col]
            n_unique = series.nunique(dropna=True)
            col_lower = str(col).lower().strip()

            # 1. Check for identifier columns
            if (
                col_lower in common_id_names
                or col_lower.endswith("_id")
                or col_lower.endswith("id")
                or col_lower.startswith("id_")
                or (n_unique == total_rows and not pd.api.types.is_numeric_dtype(series))
            ):
                identifier_candidates.append(col)
                continue

            # 2. Check for binary target candidates (exactly 2 unique non-null values)
            if n_unique == 2:
                target_candidates.append(col)

            # 3. Classify remaining features
            if pd.api.types.is_numeric_dtype(series) and n_unique > 2:
                numerical_cols.append(col)
            elif n_unique > 1 and n_unique <= min(100, int(total_rows * 0.7)):
                categorical_cols.append(col)
            elif n_unique == 1:
                unsupported_cols.append((col, "Single constant value across all rows"))
            elif n_unique > 100 and series.dtype == "object":
                unsupported_cols.append((col, f"High-cardinality text column ({n_unique} unique values)"))
            else:
                categorical_cols.append(col)

        return {
            "total_rows": total_rows,
            "total_cols": len(df.columns),
            "target_candidates": target_candidates,
            "identifier_candidates": identifier_candidates,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "unsupported_cols": unsupported_cols,
        }


class DynamicPreprocessor:
    """
    Dynamically fits imputers, scalers, and encoders strictly on the training partition
    to eliminate data leakage for arbitrary tabular datasets.
    """

    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols)
        self.num_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "DynamicPreprocessor":
        """Fit imputation statistics, scaler, and one-hot encoder on training split."""
        # 1. Numerical Imputation & Scaling
        if self.numerical_cols:
            num_df = df_train[self.numerical_cols].copy()
            for col in self.numerical_cols:
                med = float(pd.to_numeric(num_df[col], errors="coerce").median())
                self.num_medians[col] = 0.0 if np.isnan(med) else med
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(self.num_medians[col])
            self.scaler.fit(num_df.values)

        # 2. Categorical Imputation & Encoding
        if self.categorical_cols:
            cat_df = df_train[self.categorical_cols].astype(str).copy()
            for col in self.categorical_cols:
                mode_vals = cat_df[col].mode()
                mode_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
                self.cat_modes[col] = mode_val
                cat_df[col] = cat_df[col].replace({"nan": mode_val, "None": mode_val, "": mode_val})
            self.encoder.fit(cat_df)

        # 3. Assemble Feature Names
        names = []
        if self.numerical_cols:
            names.extend(self.numerical_cols)
        if self.categorical_cols:
            enc_names = self.encoder.get_feature_names_out(self.categorical_cols)
            names.extend(list(enc_names))

        self.feature_names = names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a dataset using the fitted training statistics."""
        if not self.is_fitted:
            raise ValueError("DynamicPreprocessor must be fitted on training data before transformation.")

        blocks = []

        # 1. Transform Numerical
        if self.numerical_cols:
            num_df = df[self.numerical_cols].copy()
            for col in self.numerical_cols:
                med = self.num_medians.get(col, 0.0)
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(med)
            scaled_num = self.scaler.transform(num_df.values)
            blocks.append(scaled_num)

        # 2. Transform Categorical
        if self.categorical_cols:
            cat_df = df[self.categorical_cols].astype(str).copy()
            for col in self.categorical_cols:
                mode_val = self.cat_modes.get(col, "Missing")
                cat_df[col] = cat_df[col].replace({"nan": mode_val, "None": mode_val, "": mode_val})
            encoded_cat = self.encoder.transform(cat_df)
            blocks.append(encoded_cat)

        if not blocks:
            raise ValueError("No valid numerical or categorical features found to transform.")

        return np.hstack(blocks)

    def get_feature_names(self) -> List[str]:
        """Return the transformed feature names."""
        return self.feature_names


class DynamicPipelineTrainer:
    """Executes end-to-end training and evaluation for custom tabular binary classification datasets."""

    @staticmethod
    def run_training_pipeline(
        df: pd.DataFrame,
        target_col: str,
        positive_class: Any,
        id_cols: Optional[List[str]] = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_seed: int = 42,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full dynamic workflow on the uploaded dataset:
        1. Validate target and extract binary label vector y in {0, 1}
        2. Isolate feature matrix (excluding target and IDs)
        3. Stratified 3-way split (Train / Val / Test)
        4. Fit DynamicPreprocessor strictly on Train
        5. Train scratch CustomLogisticRegression on Train with balanced class weights
        6. Optimize decision threshold t* on Val
        7. Evaluate performance metrics on Test
        8. Predict on all uploaded rows N
        9. Compute global feature importance
        """
        id_cols = id_cols or []

        # 1. Validate Target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        clean_target = df[target_col].dropna()
        unique_vals = clean_target.unique()
        if len(unique_vals) != 2:
            raise ValueError(
                f"Target column '{target_col}' contains {len(unique_vals)} unique values ({list(unique_vals)[:5]}). "
                f"Binary classification requires exactly 2 distinct classes."
            )

        # Convert target to binary integer 0/1
        pos_str = str(positive_class).strip().lower()
        y_all = (df[target_col].astype(str).str.strip().str.lower() == pos_str).astype(int).values
        n_pos = int(np.sum(y_all))
        n_neg = int(len(y_all) - n_pos)
        pos_rate = float(n_pos / len(y_all))

        # Check minimum dataset size
        if len(df) < 15 or n_pos < 2 or n_neg < 2:
            raise ValueError(
                f"Dataset contains {len(df)} rows with {n_pos} positive and {n_neg} negative cases. "
                f"A minimum of 15 total rows and at least 2 instances per class is required for ML training."
            )

        # 2. Extract Feature Subset (Exclude target and ID columns)
        excluded = set(id_cols + [target_col])
        candidate_features = [c for c in df.columns if c not in excluded]

        if not candidate_features:
            raise ValueError("No feature columns remaining after excluding target and identifier columns.")

        num_cols = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
        cat_cols = [c for c in candidate_features if c not in num_cols]

        # 3. Stratified 3-way Split (Train, Val, Test) with small dataset safety
        effective_test_size = min(test_size, max(0.10, 2.0 / len(df))) if len(df) < 40 else test_size
        effective_val_size = min(val_size, max(0.10, 2.0 / len(df))) if len(df) < 40 else val_size

        indices = np.arange(len(df))
        try:
            train_val_idx, test_idx, y_train_val, y_test = train_test_split(
                indices, y_all, test_size=effective_test_size, random_state=random_seed, stratify=y_all
            )
            val_rel_size = effective_val_size / (1.0 - effective_test_size)
            train_idx, val_idx, y_train, y_val = train_test_split(
                train_val_idx, y_train_val, test_size=val_rel_size, random_state=random_seed, stratify=y_train_val
            )
        except ValueError:
            # Fallback without stratification if singletons present
            train_val_idx, test_idx, y_train_val, y_test = train_test_split(
                indices, y_all, test_size=effective_test_size, random_state=random_seed
            )
            val_rel_size = effective_val_size / (1.0 - effective_test_size)
            train_idx, val_idx, y_train, y_val = train_test_split(
                train_val_idx, y_train_val, test_size=val_rel_size, random_state=random_seed
            )

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        df_test = df.iloc[test_idx]

        # 4. Fit Preprocessor STRICTLY on Training Split
        preprocessor = DynamicPreprocessor(numerical_cols=num_cols, categorical_cols=cat_cols)
        preprocessor.fit(df_train)

        X_train = preprocessor.transform(df_train)
        X_val = preprocessor.transform(df_val)
        X_test = preprocessor.transform(df_test)
        X_all = preprocessor.transform(df)

        feature_names = preprocessor.get_feature_names()

        # 5. Train Custom NumPy Logistic Regression
        default_hp = {
            "learning_rate": 0.05,
            "max_iter": 1000,
            "l2_lambda": 0.01,
            "momentum": 0.9,
            "batch_size": min(64, len(X_train)),
            "early_stopping": True,
            "patience": 40,
            "use_class_weights": True,
            "fit_intercept": True,
        }
        if hyperparams:
            default_hp.update(hyperparams)

        model = CustomLogisticRegression(
            learning_rate=default_hp["learning_rate"],
            max_iter=default_hp["max_iter"],
            l2_lambda=default_hp["l2_lambda"],
            momentum=default_hp["momentum"],
            batch_size=default_hp["batch_size"],
            early_stopping=default_hp["early_stopping"],
            patience=default_hp["patience"],
            use_class_weights=default_hp["use_class_weights"],
            fit_intercept=default_hp["fit_intercept"],
            random_state=random_seed,
        )

        # Execute training
        train_history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # 6. Optimize Decision Threshold on Validation Split
        val_probs = model.predict_proba(X_val)[:, 1]
        best_t = 0.50
        best_val_f1 = 0.0

        for t in np.linspace(0.10, 0.90, 81):
            y_pred_t = (val_probs >= t).astype(int)
            f1_t = f1_score(y_val, y_pred_t)
            if f1_t > best_val_f1:
                best_val_f1 = f1_t
                best_t = float(t)

        selected_threshold = round(best_t, 2)

        # 7. Evaluate on Independent Test Split
        test_probs = model.predict_proba(X_test)[:, 1]
        y_test_pred_50 = (test_probs >= 0.50).astype(int)
        y_test_pred_opt = (test_probs >= selected_threshold).astype(int)

        # Calculate metrics at default 0.50
        acc_50 = accuracy_score(y_test, y_test_pred_50)
        rec_50 = recall_score(y_test, y_test_pred_50)
        prec_50 = precision_score(y_test, y_test_pred_50)
        f1_50 = f1_score(y_test, y_test_pred_50)

        # Calculate metrics at selected optimal threshold
        acc_opt = accuracy_score(y_test, y_test_pred_opt)
        rec_opt = recall_score(y_test, y_test_pred_opt)
        prec_opt = precision_score(y_test, y_test_pred_opt)
        f1_opt = f1_score(y_test, y_test_pred_opt)

        try:
            auc_test = roc_auc_score(y_test, test_probs)
            pr_auc_test = average_precision_score(y_test, test_probs)
        except Exception:
            auc_test, pr_auc_test = 0.5, 0.0

        cm_50 = confusion_matrix(y_test, y_test_pred_50)
        cm_opt = confusion_matrix(y_test, y_test_pred_opt)

        # 8. Generate Predictions for ALL Uploaded Rows (N)
        all_probs = model.predict_proba(X_all)[:, 1]
        all_preds = (all_probs >= selected_threshold).astype(int)

        # 9. Compute Global Feature Importance (Odds Ratios)
        weights = model.weights
        odds_ratios = np.exp(weights)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Weight (Log-Odds)": np.round(weights, 4),
            "Odds Ratio": np.round(odds_ratios, 4),
            "Absolute Magnitude": np.abs(weights),
        }).sort_values("Absolute Magnitude", ascending=False)

        top_positive = importance_df[importance_df["Weight (Log-Odds)"] > 0].head(5)
        top_negative = importance_df[importance_df["Weight (Log-Odds)"] < 0].head(5)

        return {
            "status": "SUCCESS",
            "metadata": {
                "total_rows": len(df),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "feature_count": len(feature_names),
                "numerical_feature_count": len(num_cols),
                "categorical_feature_count": len(cat_cols),
                "target_col": target_col,
                "positive_class": positive_class,
                "positive_rate": pos_rate,
                "selected_threshold": selected_threshold,
                "val_f1_at_threshold": best_val_f1,
                "iterations_trained": len(model.train_loss_history_),
            },
            "test_metrics": {
                "at_default_0_50": {
                    "accuracy": acc_50,
                    "precision": prec_50,
                    "recall": rec_50,
                    "f1": f1_50,
                    "confusion_matrix": cm_50.tolist(),
                },
                "at_optimal_threshold": {
                    "threshold": selected_threshold,
                    "accuracy": acc_opt,
                    "precision": prec_opt,
                    "recall": rec_opt,
                    "f1": f1_opt,
                    "confusion_matrix": cm_opt.tolist(),
                },
                "roc_auc": auc_test,
                "pr_auc": pr_auc_test,
            },
            "predictions": {
                "probabilities": all_probs,
                "classes": all_preds,
            },
            "feature_importance": {
                "top_positive": top_positive.to_dict(orient="records"),
                "top_negative": top_negative.to_dict(orient="records"),
                "all_features": importance_df.to_dict(orient="records"),
            },
            "model_state": {
                "weights": model.weights.tolist(),
                "bias": float(model.bias),
                "feature_names": feature_names,
            },
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
        }
