"""
Global Feature Importance and Odds Ratio Analysis for ChurnGuard AI.
Translates mathematical logistic regression weights into business-interpretable odds ratios
with rigorous non-causal statistical interpretations for both standardized numerical
and one-hot encoded categorical features.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


class GlobalExplainer:
    """Computes global feature importance, odds ratios, and non-causal statistical interpretations."""

    # Reference categories for one-hot encoded features (due to drop='first' in OneHotEncoder)
    REFERENCE_CATEGORIES: Dict[str, str] = {
        "gender_Male": "Female",
        "SeniorCitizen_1": "Non-Senior (SeniorCitizen=0)",
        "Partner_Yes": "No Partner",
        "Dependents_Yes": "No Dependents",
        "PhoneService_Yes": "No Phone Service",
        "MultipleLines_No phone service": "No Multiple Lines (Single line)",
        "MultipleLines_Yes": "Single Phone Line",
        "InternetService_Fiber optic": "DSL Internet",
        "InternetService_No": "DSL Internet",
        "OnlineSecurity_No internet service": "No Online Security (with Internet)",
        "OnlineSecurity_Yes": "No Online Security (with Internet)",
        "OnlineBackup_No internet service": "No Online Backup (with Internet)",
        "OnlineBackup_Yes": "No Online Backup (with Internet)",
        "DeviceProtection_No internet service": "No Device Protection (with Internet)",
        "DeviceProtection_Yes": "No Device Protection (with Internet)",
        "TechSupport_No internet service": "No Tech Support (with Internet)",
        "TechSupport_Yes": "No Tech Support (with Internet)",
        "StreamingTV_No internet service": "No Streaming TV (with Internet)",
        "StreamingTV_Yes": "No Streaming TV (with Internet)",
        "StreamingMovies_No internet service": "No Streaming Movies (with Internet)",
        "StreamingMovies_Yes": "No Streaming Movies (with Internet)",
        "Contract_One year": "Month-to-month Contract",
        "Contract_Two year": "Month-to-month Contract",
        "PaperlessBilling_Yes": "Paper Billing (PaperlessBilling=No)",
        "PaymentMethod_Credit card (automatic)": "Bank transfer (automatic)",
        "PaymentMethod_Electronic check": "Bank transfer (automatic)",
        "PaymentMethod_Mailed check": "Bank transfer (automatic)",
        "tenure_cohort_13-24m": "New Subscriber (0-12m cohort)",
        "tenure_cohort_25-48m": "New Subscriber (0-12m cohort)",
        "tenure_cohort_49-72m": "New Subscriber (0-12m cohort)",
        "has_internet": "No Internet Service",
        "is_solo_senior": "Non-Senior or Senior with family",
        "high_risk_fiber_m2m": "Non-Fiber or Non-Month-to-Month Contract",
    }

    # Standardized numerical features
    STANDARDIZED_NUMERICAL_FEATURES: List[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "monthly_to_total_ratio",
        "monthly_charge_discrepancy",
        "total_services_count",
        "protection_services_count",
        "streaming_services_count",
    ]

    def __init__(self, feature_names: List[str], weights: np.ndarray, bias: float):
        self.feature_names = feature_names
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def _get_feature_type(self, feature_name: str) -> str:
        """Determine whether a feature is a standardized numerical or categorical indicator."""
        if feature_name in self.STANDARDIZED_NUMERICAL_FEATURES:
            return "Standardized Numerical"
        return "Categorical / Binary Indicator"

    def _generate_interpretation_text(self, feature_name: str, weight: float, odds_ratio: float) -> str:
        """
        Generate precise, non-causal statistical interpretations.
        Distinguishes standardized numerical variables (per 1 std-dev increase)
        from one-hot indicator variables (relative to the reference category).
        """
        feat_type = self._get_feature_type(feature_name)
        pct_change = (odds_ratio - 1.0) * 100.0

        if feat_type == "Standardized Numerical":
            if weight > 0:
                return (
                    f"A 1-standard-deviation increase in {feature_name} is associated with approximately "
                    f"{pct_change:+.1f}% higher modeled churn odds (OR = {odds_ratio:.4f}), "
                    f"holding other model inputs constant."
                )
            else:
                return (
                    f"A 1-standard-deviation increase in {feature_name} is associated with approximately "
                    f"{abs(pct_change):.1f}% lower modeled churn odds (OR = {odds_ratio:.4f}), "
                    f"holding other model inputs constant."
                )
        else:
            ref_cat = self.REFERENCE_CATEGORIES.get(feature_name, "Reference Group")
            if weight > 0:
                return (
                    f"Presence of '{feature_name}' is associated with approximately "
                    f"{pct_change:+.1f}% higher modeled churn odds (OR = {odds_ratio:.4f}) "
                    f"relative to {ref_cat}, holding other model inputs constant."
                )
            else:
                return (
                    f"Presence of '{feature_name}' is associated with approximately "
                    f"{abs(pct_change):.1f}% lower modeled churn odds (OR = {odds_ratio:.4f}) "
                    f"relative to {ref_cat}, holding other model inputs constant."
                )

    def get_summary_dataframe(
        self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Construct a comprehensive DataFrame containing:
        - Weight (Log-Odds)
        - Odds Ratio: exp(w)
        - % Change in Odds: (exp(w) - 1) * 100
        - Impact Direction: 'Risk Driver' vs 'Retention Anchor'
        - Feature Type: Standardized Numerical vs Categorical
        - Reference Category
        - Human-Readable Non-Causal Explanation
        - Standard Error, Wald Z-Score, and P-Value (if X_train provided)
        """
        odds_ratios = np.exp(self.weights)
        pct_change = (odds_ratios - 1.0) * 100.0
        directions = [
            "Risk Driver (Increases Churn Odds)" if w > 0 else "Retention Anchor (Decreases Churn Odds)"
            for w in self.weights
        ]
        feat_types = [self._get_feature_type(f) for f in self.feature_names]
        ref_cats = [self.REFERENCE_CATEGORIES.get(f, "N/A (Standardized)") for f in self.feature_names]
        interpretations = [
            self._generate_interpretation_text(f, w, r)
            for f, w, r in zip(self.feature_names, self.weights, odds_ratios)
        ]

        data: Dict[str, Any] = {
            "Feature": self.feature_names,
            "Feature Type": feat_types,
            "Reference Category": ref_cats,
            "Weight (Log-Odds)": np.round(self.weights, 4),
            "Odds Ratio": np.round(odds_ratios, 4),
            "% Change in Odds": np.round(pct_change, 2),
            "Impact Direction": directions,
            "Statistical Interpretation": interpretations,
            "Absolute Importance": np.round(np.abs(self.weights), 4),
        }

        # Calculate standard errors and Wald test statistics if X_train provided
        if X_train is not None and y_train is not None:
            try:
                from src.models.custom_logistic_regression import CustomLogisticRegression
                z = np.dot(X_train, self.weights) + self.bias
                p = CustomLogisticRegression.sigmoid(z)
                v = p * (1.0 - p)
                
                # Weighted Fisher Information Matrix: H = X^T * diag(v) * X + lambda * I
                H = np.dot(X_train.T * v, X_train) + 1e-4 * np.eye(len(self.weights))
                cov = np.linalg.pinv(H)
                se = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
                z_scores = self.weights / se
                p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))

                data["Std Error"] = np.round(se, 4)
                data["Wald Z-Score"] = np.round(z_scores, 2)
                data["P-Value"] = np.round(p_values, 5)
            except Exception:
                pass

        df = pd.DataFrame(data).sort_values("Absolute Importance", ascending=False)
        return df

    def get_top_risk_drivers(self, top_n: int = 5) -> pd.DataFrame:
        """Top features that most aggressively increase modeled churn odds."""
        df = self.get_summary_dataframe()
        return df[df["Weight (Log-Odds)"] > 0].head(top_n)

    def get_top_retention_anchors(self, top_n: int = 5) -> pd.DataFrame:
        """Top features that most strongly decrease modeled churn odds."""
        df = self.get_summary_dataframe()
        return df[df["Weight (Log-Odds)"] < 0].head(top_n)
