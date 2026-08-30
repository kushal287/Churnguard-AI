"""
Individual Customer Prediction Explainer for ChurnGuard AI.
Decomposes single-customer churn probability into additive feature log-odds contributions,
providing exact mathematical waterfall reconstruction and non-causal attribution.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.custom_logistic_regression import CustomLogisticRegression


class IndividualExplainer:
    """Explains predictions for individual customer instances via linear log-odds decomposition."""

    def __init__(self, feature_names: List[str], weights: np.ndarray, bias: float):
        self.feature_names = feature_names
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def explain_instance(
        self,
        x_vector: np.ndarray,
        top_n: int = 5,
        raw_customer_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deconstruct a single customer feature vector into exact linear contributions:
        z = b + sum_j (w_j * x_j)
        p = sigmoid(z)

        Returns complete waterfall telemetry, risk drivers, and protective factors.
        """
        x = np.asarray(x_vector, dtype=float).flatten()
        if len(x) != len(self.weights):
            raise ValueError(
                f"Feature vector length ({len(x)}) does not match model weights ({len(self.weights)})"
            )

        # Element-wise additive contribution to log-odds: c_j = w_j * x_j
        contributions = self.weights * x
        total_logit = self.bias + float(np.sum(contributions))
        churn_probability = float(CustomLogisticRegression.sigmoid(np.array([total_logit]))[0])
        base_probability = float(CustomLogisticRegression.sigmoid(np.array([self.bias]))[0])

        # Mathematical verification of reconstruction
        reconstructed_prob = 1.0 / (1.0 + np.exp(-total_logit))
        discrepancy = abs(churn_probability - reconstructed_prob)

        # Build feature contributions table
        contrib_records = []
        for i, (feat, w_val, x_val, c_val) in enumerate(zip(self.feature_names, self.weights, x, contributions)):
            direction = "Increases Churn Risk (Pushing UP)" if c_val > 0 else (
                "Decreases Churn Risk (Pushing DOWN)" if c_val < 0 else "Neutral / Inactive"
            )
            contrib_records.append({
                "Feature": feat,
                "TransformedValue": float(x_val),
                "Weight": float(w_val),
                "LogOddsContribution": float(c_val),
                "AbsContribution": abs(float(c_val)),
                "Direction": direction,
            })

        df_contrib = pd.DataFrame(contrib_records)

        # Top positive contributors (Risk Drivers: c_j > 0)
        risk_drivers = (
            df_contrib[df_contrib["LogOddsContribution"] > 0.001]
            .sort_values("AbsContribution", ascending=False)
            .head(top_n)
        )

        # Top negative contributors (Protective Factors: c_j < 0)
        protective_factors = (
            df_contrib[df_contrib["LogOddsContribution"] < -0.001]
            .sort_values("AbsContribution", ascending=False)
            .head(top_n)
        )

        # Construct Step-by-Step Cumulative Waterfall
        # Start with Baseline Intercept
        waterfall_steps = []
        waterfall_steps.append({
            "step": "1. Baseline Intercept (b)",
            "feature": "Model Intercept (Prior Bias)",
            "feature_value": "N/A",
            "weight": round(self.bias, 4),
            "log_odds_delta": round(self.bias, 4),
            "cumulative_log_odds": round(self.bias, 4),
            "implied_churn_probability": round(base_probability, 4),
            "direction": "Baseline Prior",
            "type": "base",
        })

        # Add top active contributing features sorted by absolute impact
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        cum_logit = self.bias
        step_num = 2

        for idx in sorted_indices:
            c = contributions[idx]
            if abs(c) < 1e-4:
                continue
            cum_logit += c
            p_cum = float(CustomLogisticRegression.sigmoid(np.array([cum_logit]))[0])
            waterfall_steps.append({
                "step": f"{step_num}. {self.feature_names[idx]}",
                "feature": self.feature_names[idx],
                "feature_value": round(float(x[idx]), 3),
                "weight": round(float(self.weights[idx]), 4),
                "log_odds_delta": round(float(c), 4),
                "cumulative_log_odds": round(float(cum_logit), 4),
                "implied_churn_probability": round(p_cum, 4),
                "direction": "Pushing UP (+)" if c > 0 else "Pushing DOWN (-)",
                "type": "risk" if c > 0 else "protective",
            })
            step_num += 1
            if step_num > top_n + 1:
                break

        # Final Reconstruction Step
        waterfall_steps.append({
            "step": "FINAL MODEL LOGIT & PROBABILITY",
            "feature": "Total Customer Score",
            "feature_value": "All Inputs Combined",
            "weight": 1.0,
            "log_odds_delta": round(float(np.sum(contributions)), 4),
            "cumulative_log_odds": round(total_logit, 4),
            "implied_churn_probability": round(churn_probability, 4),
            "direction": "Final Churn Prediction",
            "type": "total",
        })

        return {
            "churn_probability": round(churn_probability, 4),
            "retention_probability": round(1.0 - churn_probability, 4),
            "base_probability": round(base_probability, 4),
            "bias_weight": round(self.bias, 4),
            "total_log_odds": round(total_logit, 4),
            "mathematical_reconstruction": {
                "direct_probability": float(churn_probability),
                "reconstructed_probability": float(reconstructed_prob),
                "discrepancy": float(discrepancy),
                "formula": "p = 1 / (1 + exp(-(b + sum(w_j * x_j))))",
            },
            "risk_drivers": risk_drivers[
                ["Feature", "TransformedValue", "Weight", "LogOddsContribution"]
            ].to_dict(orient="records"),
            "protective_factors": protective_factors[
                ["Feature", "TransformedValue", "Weight", "LogOddsContribution"]
            ].to_dict(orient="records"),
            "retention_anchors": protective_factors[
                ["Feature", "TransformedValue", "Weight", "LogOddsContribution"]
            ].to_dict(orient="records"),
            "waterfall_steps": waterfall_steps,
        }
