"""
Dataset Schema Validator & Health Check Engine for ChurnGuard AI.
Validates uploaded user datasets against the required Telco Customer Churn schema,
providing comprehensive health reports, repair instructions, template generation,
and safe handling of labeled vs. unlabeled datasets without modifying frozen model artifacts.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


class DatasetValidator:
    """Validates user-uploaded datasets for compatibility with the ChurnGuard AI inference pipeline."""

    REQUIRED_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ALL_EXPECTED_COLUMNS: List[str] = [ID_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    ACCEPTED_CATEGORIES: Dict[str, List[str]] = {
        "gender": ["Female", "Male"],
        "SeniorCitizen": ["0", "1", 0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }

    @classmethod
    def get_schema_specification(cls) -> Dict[str, Any]:
        """Return the structured specification of all expected columns and accepted values."""
        return {
            "identifier": {
                "name": ID_COLUMN,
                "type": "String / Alphanumeric",
                "description": "Unique subscriber account identifier",
                "example": "7590-VHVEG",
                "required": True,
            },
            "numerical_features": [
                {
                    "name": "tenure",
                    "type": "Integer / Float",
                    "description": "Number of months subscribed",
                    "valid_range": ">= 0",
                    "example": "12",
                },
                {
                    "name": "MonthlyCharges",
                    "type": "Float",
                    "description": "Current monthly recurring bill ($)",
                    "valid_range": "> 0.0",
                    "example": "70.35",
                },
                {
                    "name": "TotalCharges",
                    "type": "Float / String",
                    "description": "Total cumulative charges ($). Whitespace allowed for tenure=0.",
                    "valid_range": ">= 0.0",
                    "example": "840.20",
                },
            ],
            "categorical_features": [
                {
                    "name": col,
                    "type": "Categorical String",
                    "accepted_values": [str(v) for v in vals],
                    "example": str(vals[0]),
                }
                for col, vals in cls.ACCEPTED_CATEGORIES.items()
            ],
            "target": {
                "name": TARGET_COLUMN,
                "type": "Categorical String",
                "accepted_values": ["Yes", "No"],
                "description": "Historical ground truth churn indicator (Optional for prediction-only mode).",
                "example": "No",
                "required": False,
            },
        }

    @classmethod
    def generate_csv_template(cls) -> pd.DataFrame:
        """
        Generate a downloadable CSV template containing all expected columns
        and two clearly labeled illustrative rows.
        """
        template_data = {
            "customerID": ["SAMPLE-CUST-001", "SAMPLE-CUST-002"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 2],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["Yes", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Bank transfer (automatic)", "Electronic check"],
            "MonthlyCharges": [55.85, 98.50],
            "TotalCharges": [670.20, 197.00],
            "Churn": ["No", "Yes"],
        }
        return pd.DataFrame(template_data)

    @classmethod
    def generate_example_dataset(cls, n_samples: int = 15) -> pd.DataFrame:
        """Generate a realistic demonstration batch dataset for immediate testing."""
        np.random.seed(42)
        customers = []
        for i in range(1, n_samples + 1):
            is_high_risk = i % 3 == 0
            cust_id = f"DEMO-USER-{i:04d}"
            gender = "Female" if i % 2 == 0 else "Male"
            senior = 1 if i % 4 == 0 else 0
            partner = "Yes" if i % 2 == 0 else "No"
            dependents = "Yes" if i % 3 == 1 else "No"
            
            if is_high_risk:
                tenure = int(np.random.randint(1, 6))
                contract = "Month-to-month"
                internet = "Fiber optic"
                tech_support = "No"
                online_sec = "No"
                payment = "Electronic check"
                monthly = round(float(np.random.uniform(85.0, 105.0)), 2)
                total = round(float(monthly * tenure), 2)
                churn = "Yes" if i % 2 == 0 else "No"
            else:
                tenure = int(np.random.randint(18, 65))
                contract = "Two year" if i % 2 == 0 else "One year"
                internet = "DSL" if i % 2 == 0 else "Fiber optic"
                tech_support = "Yes"
                online_sec = "Yes"
                payment = "Credit card (automatic)" if i % 2 == 0 else "Bank transfer (automatic)"
                monthly = round(float(np.random.uniform(45.0, 75.0)), 2)
                total = round(float(monthly * tenure), 2)
                churn = "No"

            customers.append({
                "customerID": cust_id,
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No" if i % 2 == 0 else "Yes",
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "OnlineBackup": "Yes" if not is_high_risk else "No",
                "DeviceProtection": "Yes" if not is_high_risk else "No",
                "TechSupport": tech_support,
                "StreamingTV": "Yes" if is_high_risk else "No",
                "StreamingMovies": "Yes" if is_high_risk else "No",
                "Contract": contract,
                "PaperlessBilling": "Yes",
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "Churn": churn,
            })
        return pd.DataFrame(customers)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute comprehensive pre-flight health checks on an uploaded DataFrame.
        Returns a structured validation diagnostic report.
        """
        report: Dict[str, Any] = {
            "is_compatible": True,
            "is_labeled": False,
            "status_title": "Dataset Compatible",
            "total_rows": len(df),
            "total_cols": len(df.columns) if df is not None else 0,
            "errors": [],
            "warnings": [],
            "info": [],
            "statistics": {},
        }

        # 1. Empty Check
        if df is None or df.empty or len(df) == 0:
            report["is_compatible"] = False
            report["status_title"] = "Dataset Incompatible (Empty File)"
            report["errors"].append("The uploaded CSV file is empty. Please upload a CSV with at least one customer row.")
            return report

        # 2. Check Size Limits
        if len(df) > 50000:
            report["is_compatible"] = False
            report["status_title"] = "Dataset Incompatible (File Too Large)"
            report["errors"].append(
                f"Dataset contains {len(df):,} rows, exceeding the safe interactive batch limit of 50,000 rows."
            )
            return report

        # 3. Check Required Feature Columns
        df_cols = [str(c).strip() for c in df.columns]
        missing_features = [f for f in cls.REQUIRED_FEATURES if f not in df_cols]
        extra_cols = [c for c in df_cols if c not in cls.ALL_EXPECTED_COLUMNS and c != TARGET_COLUMN]
        found_features = [f for f in cls.REQUIRED_FEATURES if f in df_cols]

        report["statistics"]["required_features_found"] = len(found_features)
        report["statistics"]["required_features_total"] = len(cls.REQUIRED_FEATURES)
        report["statistics"]["extra_columns_count"] = len(extra_cols)
        report["statistics"]["extra_columns"] = extra_cols

        if missing_features:
            report["is_compatible"] = False
            report["status_title"] = "Dataset Incompatible (Missing Required Columns)"
            report["errors"].append(
                f"Missing {len(missing_features)} required feature column(s): {', '.join(missing_features)}. "
                f"Please verify column headers match the ChurnGuard Telco schema."
            )

        # 4. Check Customer Identifier
        if ID_COLUMN not in df_cols:
            report["warnings"].append(
                f"Column '{ID_COLUMN}' was not found. Synthetic customer IDs (e.g. CUST-0001) will be auto-generated for reporting."
            )
        else:
            n_dupes = int(df[ID_COLUMN].duplicated().sum())
            if n_dupes > 0:
                report["warnings"].append(
                    f"Found {n_dupes} duplicate customer ID(s) in '{ID_COLUMN}'. IDs should uniquely identify each account."
                )

        # 5. Check Target Column (Scenario A vs Scenario B)
        if TARGET_COLUMN in df_cols:
            report["is_labeled"] = True
            unique_targets = set(df[TARGET_COLUMN].dropna().astype(str).str.strip().unique())
            valid_targets = {"Yes", "No", "1", "0", "True", "False"}
            invalid_targets = unique_targets - valid_targets
            if invalid_targets:
                report["is_compatible"] = False
                report["status_title"] = "Dataset Incompatible (Invalid Churn Target Values)"
                report["errors"].append(
                    f"Target column '{TARGET_COLUMN}' contains unrecognized values: {invalid_targets}. "
                    f"Expected values are 'Yes' / 'No' (or 1 / 0)."
                )
            else:
                churn_count = int(np.sum(df[TARGET_COLUMN].astype(str).str.strip().isin(["Yes", "1", "True"])))
                report["statistics"]["target_churn_count"] = churn_count
                report["statistics"]["target_churn_rate"] = float(churn_count / len(df))
                report["info"].append(
                    f"Ground-truth '{TARGET_COLUMN}' detected: {churn_count:,} churners ({churn_count/len(df):.1%}). "
                    f"Model performance metrics and confusion matrix will be calculated."
                )
        else:
            report["is_labeled"] = False
            report["info"].append(
                f"No ground-truth '{TARGET_COLUMN}' column found. Operating in Prediction & Retention Prioritization Mode."
            )

        # 6. Check Numerical Columns Data Types & Impossible Values
        for num_col in NUMERICAL_FEATURES:
            if num_col not in df_cols:
                continue

            # Special case for TotalCharges which might contain whitespace strings
            if num_col == "TotalCharges":
                s_str = df[num_col].astype(str).str.strip()
                whitespace_count = int((s_str == "").sum())
                if whitespace_count > 0:
                    report["info"].append(
                        f"Found {whitespace_count} blank/whitespace TotalCharges entry(ies). "
                        f"These will be safely imputed to 0.0 (consistent with tenure=0 subscribers)."
                    )
                # Attempt conversion
                numeric_series = pd.to_numeric(s_str.replace("", "0"), errors="coerce")
            else:
                numeric_series = pd.to_numeric(df[num_col], errors="coerce")

            invalid_nums = int(numeric_series.isna().sum())
            if invalid_nums > 0:
                report["is_compatible"] = False
                report["status_title"] = "Dataset Incompatible (Invalid Numeric Values)"
                report["errors"].append(
                    f"Column '{num_col}' contains {invalid_nums} non-numeric value(s) that could not be parsed."
                )

            # Check impossible negative values
            neg_count = int((numeric_series.dropna() < 0).sum())
            if neg_count > 0:
                report["is_compatible"] = False
                report["status_title"] = "Dataset Incompatible (Negative Numeric Values)"
                report["errors"].append(
                    f"Column '{num_col}' contains {neg_count} negative value(s). Charges and tenure cannot be negative."
                )

        # 7. Check Categorical Values
        unknown_cat_summary: Dict[str, List[str]] = {}
        for cat_col, valid_vals in cls.ACCEPTED_CATEGORIES.items():
            if cat_col not in df_cols:
                continue
            str_valid = {str(v).lower() for v in valid_vals}
            actual_vals = set(df[cat_col].dropna().astype(str).str.strip().unique())
            unknowns = [v for v in actual_vals if str(v).lower() not in str_valid and v != ""]
            if unknowns:
                unknown_cat_summary[cat_col] = unknowns

        if unknown_cat_summary:
            report["statistics"]["unknown_categories"] = unknown_cat_summary
            details = [f"{col}: {vals}" for col, vals in unknown_cat_summary.items()]
            report["warnings"].append(
                f"Found unrecognized categorical values in: {'; '.join(details)}. "
                f"During transformation, unknown categories will be safely one-hot encoded as neutral zeros."
            )

        return report
