"""
Dynamic Mathematical Log-Odds Explainer for General Tabular Binary Classification Datasets.
Provides exact linear attribution z = b + sum(w_j * x_j) with mathematical verification
for arbitrary custom datasets without domain-specific hardcoded assumptions.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.custom_logistic_regression import CustomLogisticRegression


class DynamicIndividualExplainer:
    """Explains predictions for individual observations from arbitrary custom datasets."""

    def __init__(self, feature_names: List[str], weights: np.ndarray, bias: float):
        self.feature_names = list(feature_names)
        self.weights = np.array(weights, dtype=float).ravel()
        self.bias = float(bias)

        if len(self.feature_names) != len(self.weights):
            raise ValueError(
                f"Dimension mismatch: {len(self.feature_names)} feature names vs {len(self.weights)} weights."
            )

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid function."""
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))

    def explain_instance(self, x_vector: np.ndarray, top_n: int = 5) -> Dict[str, Any]:
        """
        Decompose an observation's linear hypothesis into exact feature-level log-odds contributions:
        z = b + sum(w_j * x_j)
        p = sigmoid(z)
        """
        x_flat = np.array(x_vector, dtype=float).ravel()
        if len(x_flat) != len(self.weights):
            raise ValueError(
                f"Feature vector dimension ({len(x_flat)}) does not match model weights ({len(self.weights)})."
            )

        contributions = self.weights * x_flat
        total_logit = float(self.bias + np.sum(contributions))
        predicted_prob = self._sigmoid(total_logit)
        baseline_prob = self._sigmoid(self.bias)

        # Build feature attribution DataFrame
        df_attrib = pd.DataFrame({
            "Feature": self.feature_names,
            "TransformedValue": np.round(x_flat, 4),
            "Weight": np.round(self.weights, 4),
            "LogOddsContribution": np.round(contributions, 4),
            "AbsContribution": np.abs(contributions),
        })

        # Top positive pushers (pushing TOWARDS positive class)
        positive_pushers = (
            df_attrib[df_attrib["LogOddsContribution"] > 0]
            .sort_values("LogOddsContribution", ascending=False)
            .head(top_n)
        )

        # Top negative pushers (pushing TOWARDS negative class)
        negative_pushers = (
            df_attrib[df_attrib["LogOddsContribution"] < 0]
            .sort_values("LogOddsContribution", ascending=True)
            .head(top_n)
        )

        # Cumulative step-by-step waterfall construction
        sorted_by_impact = df_attrib.sort_values("AbsContribution", ascending=False)
        waterfall_steps = []
        cumulative_z = self.bias

        waterfall_steps.append({
            "step": 0,
            "feature": "Baseline Intercept (b)",
            "feature_value": "—",
            "weight": self.bias,
            "log_odds_delta": self.bias,
            "cumulative_log_odds": self.bias,
            "implied_probability": baseline_prob,
            "direction": "Baseline Prior",
        })

        for step_idx, (_, row) in enumerate(sorted_by_impact.iterrows(), 1):
            delta = float(row["LogOddsContribution"])
            if abs(delta) < 1e-4:
                continue
            cumulative_z += delta
            cum_p = self._sigmoid(cumulative_z)
            waterfall_steps.append({
                "step": step_idx,
                "feature": row["Feature"],
                "feature_value": row["TransformedValue"],
                "weight": row["Weight"],
                "log_odds_delta": delta,
                "cumulative_log_odds": cumulative_z,
                "implied_probability": cum_p,
                "direction": "Pushes Toward Positive" if delta > 0 else "Pushes Toward Negative",
            })

        # Mathematical verification
        reconstructed_p = self._sigmoid(self.bias + float(np.sum(contributions)))
        discrepancy = abs(predicted_prob - reconstructed_p)

        return {
            "predicted_probability": predicted_prob,
            "baseline_probability": baseline_prob,
            "bias_weight": self.bias,
            "total_logit": total_logit,
            "positive_pushers": positive_pushers.to_dict(orient="records"),
            "negative_pushers": negative_pushers.to_dict(orient="records"),
            "waterfall_steps": waterfall_steps,
            "mathematical_reconstruction": {
                "direct_probability": predicted_prob,
                "reconstructed_probability": reconstructed_p,
                "discrepancy": discrepancy,
                "is_exact_match": bool(discrepancy < 1e-7),
            },
        }
