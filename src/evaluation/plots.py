"""
Visualization and Plotting Suite for ChurnGuard AI.
Generates publication-quality charts for model evaluation, explainability, and EDA.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import FIGURES_DIR
from src.evaluation.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class Visualizer:
    """Generates and persists evaluation and explainability figures."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_loss(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        best_epoch: Optional[int] = None,
        save_name: str = "training_loss_curves.png",
    ) -> Path:
        """Plot loss curves across training epochs."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label="Training Loss (BCE)", color="#1E88E5", linewidth=2.2)

        if val_loss and len(val_loss) > 0:
            ax.plot(epochs[: len(val_loss)], val_loss, label="Validation Loss (BCE)", color="#E53935", linewidth=2.0, linestyle="--")

        if best_epoch is not None and best_epoch <= len(train_loss):
            best_val = val_loss[best_epoch - 1] if val_loss and best_epoch <= len(val_loss) else train_loss[best_epoch - 1]
            ax.scatter([best_epoch], [best_val], color="#43A047", s=100, zorder=5, label=f"Best Model (Epoch {best_epoch})")
            ax.axvline(x=best_epoch, color="#43A047", linestyle=":", alpha=0.6)

        ax.set_title("Custom Logistic Regression Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch / Iteration", fontsize=11, fontweight="medium")
        ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, fontweight="medium")
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_roc_pr_comparison(
        self,
        y_test: np.ndarray,
        custom_proba: np.ndarray,
        sklearn_proba: np.ndarray,
        save_name: str = "roc_pr_curves_comparison.png",
    ) -> Path:
        """Side-by-side ROC and Precision-Recall comparison curves."""
        from src.evaluation.metrics import average_precision_score, roc_auc_score

        # Compute ROC curves
        fpr_c, tpr_c, _ = roc_curve(y_test, custom_proba)
        fpr_s, tpr_s, _ = roc_curve(y_test, sklearn_proba)
        auc_c = roc_auc_score(y_test, custom_proba)
        auc_s = roc_auc_score(y_test, sklearn_proba)

        # Compute PR curves
        prec_c, rec_c, _ = precision_recall_curve(y_test, custom_proba)
        prec_s, rec_s, _ = precision_recall_curve(y_test, sklearn_proba)
        ap_c = average_precision_score(y_test, custom_proba)
        ap_s = average_precision_score(y_test, sklearn_proba)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # 1. ROC Curve
        ax1.plot(fpr_c, tpr_c, label=f"Custom NumPy LR (AUC = {auc_c:.4f})", color="#1E88E5", linewidth=2.2)
        ax1.plot(fpr_s, tpr_s, label=f"Scikit-Learn LR (AUC = {auc_s:.4f})", color="#FB8C00", linewidth=1.8, linestyle="--")
        ax1.plot([0, 1], [0, 1], color="#9E9E9E", linestyle=":", label="Random Chance (AUC = 0.50)")
        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax1.legend(loc="lower right", frameon=True, facecolor="white")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. PR Curve
        base_rate = np.mean(y_test)
        ax2.plot(rec_c, prec_c, label=f"Custom NumPy LR (AP = {ap_c:.4f})", color="#1E88E5", linewidth=2.2)
        ax2.plot(rec_s, prec_s, label=f"Scikit-Learn LR (AP = {ap_s:.4f})", color="#FB8C00", linewidth=1.8, linestyle="--")
        ax2.axhline(y=base_rate, color="#9E9E9E", linestyle=":", label=f"No-Skill Baseline ({base_rate:.2f})")
        ax2.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax2.set_ylabel("Precision (PPV)", fontsize=11)
        ax2.legend(loc="lower left", frameon=True, facecolor="white")
        ax2.grid(True, linestyle="--", alpha=0.5)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_confusion_matrices(
        self,
        y_test: np.ndarray,
        custom_pred: np.ndarray,
        sklearn_pred: np.ndarray,
        save_name: str = "confusion_matrices.png",
    ) -> Path:
        """Side-by-side heatmaps of confusion matrices."""
        cm_c = confusion_matrix(y_test, custom_pred)
        cm_s = confusion_matrix(y_test, sklearn_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

        # Custom LR Matrix
        sns.heatmap(
            cm_c,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax1,
            xticklabels=["Stay (0)", "Churn (1)"],
            yticklabels=["Stay (0)", "Churn (1)"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax1.set_title("Custom NumPy Logistic Regression", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel("Predicted Label", fontsize=11)
        ax1.set_ylabel("True Label", fontsize=11)

        # Sklearn LR Matrix
        sns.heatmap(
            cm_s,
            annot=True,
            fmt="d",
            cmap="Oranges",
            cbar=False,
            ax=ax2,
            xticklabels=["Stay (0)", "Churn (1)"],
            yticklabels=["Stay (0)", "Churn (1)"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax2.set_title("Scikit-Learn Logistic Regression", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel("Predicted Label", fontsize=11)
        ax2.set_ylabel("True Label", fontsize=11)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_feature_importance_odds_ratios(
        self,
        feature_names: List[str],
        weights: np.ndarray,
        top_n: int = 12,
        save_name: str = "odds_ratio_feature_importance.png",
    ) -> Path:
        """
        Plot top risk drivers (Odds Ratio > 1) and retention anchors (Odds Ratio < 1).
        """
        odds_ratios = np.exp(weights)
        pct_change = (odds_ratios - 1.0) * 100.0

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Weight": weights,
            "OddsRatio": odds_ratios,
            "PctChange": pct_change,
            "AbsWeight": np.abs(weights),
        }).sort_values("AbsWeight", ascending=False)

        top_df = df_imp.head(top_n).sort_values("Weight", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        colors = ["#E53935" if w > 0 else "#43A047" for w in top_df["Weight"]]

        bars = ax.barh(top_df["Feature"], top_df["Weight"], color=colors, height=0.65, edgecolor="none")

        # Annotate with Odds Ratio and % Impact
        for bar, or_val, pct in zip(bars, top_df["OddsRatio"], top_df["PctChange"]):
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.03 if width >= 0 else -0.03
            sign = "+" if pct >= 0 else ""
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"OR: {or_val:.2f} ({sign}{pct:.1f}%)",
                va="center",
                ha=align,
                fontsize=8.5,
                fontweight="medium",
                color="#333333",
            )

        ax.axvline(0, color="#333333", linewidth=1.0, linestyle="-")
        ax.set_title(f"Top {top_n} Churn Drivers vs Retention Anchors (Standardized Log-Odds)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Logistic Regression Weight (Log-Odds)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Custom legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E53935", label="Increases Churn Risk (OR > 1)"),
            Patch(facecolor="#43A047", label="Decreases Churn Risk / Retention Anchor (OR < 1)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white")

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_threshold_tuning(
        self,
        sweep_results: List[Dict[str, float]],
        best_threshold: float,
        save_name: str = "threshold_tuning_curve.png",
    ) -> Path:
        """Plot Precision, Recall, F1, and Net Savings vs Decision Threshold."""
        df_sw = pd.DataFrame(sweep_results)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

        # Statistical Metrics
        ax1.plot(df_sw["threshold"], df_sw["f1_score"], label="F1-Score", color="#8E24AA", linewidth=2.2)
        ax1.plot(df_sw["threshold"], df_sw["precision"], label="Precision", color="#1E88E5", linewidth=1.8, linestyle="--")
        ax1.plot(df_sw["threshold"], df_sw["recall"], label="Recall", color="#E53935", linewidth=1.8, linestyle="--")
        ax1.plot(df_sw["threshold"], df_sw["accuracy"], label="Accuracy", color="#43A047", linewidth=1.5, linestyle=":")
        ax1.axvline(x=best_threshold, color="#8E24AA", linestyle="-.", alpha=0.8, label=f"Optimal F1 Threshold ({best_threshold:.2f})")

        ax1.set_title("Decision Threshold Sensitivity: Classification Metrics", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Metric Value", fontsize=11)
        ax1.legend(loc="lower center", ncol=5, frameon=True, facecolor="white", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Financial Net Savings
        ax2.plot(df_sw["threshold"], df_sw["net_savings"], label="Net Retention Savings ($)", color="#00897B", linewidth=2.2)
        ax2.axvline(x=best_threshold, color="#8E24AA", linestyle="-.", alpha=0.8)
        ax2.set_title("Business Retention Value vs Decision Threshold", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Decision Threshold (Probability Cutoff)", fontsize=11)
        ax2.set_ylabel("Net Savings (USD $)", fontsize=11)
        ax2.legend(loc="lower right", frameon=True, facecolor="white")
        ax2.grid(True, linestyle="--", alpha=0.5)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_eda_charts(self, df_clean: pd.DataFrame) -> Tuple[Path, Path]:
        """Generate exploratory data analysis visualizations."""
        # 1. Churn by Contract, Internet, and Tenure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

        # Contract vs Churn
        contract_churn = df_clean.groupby("Contract")["Churn"].mean() * 100
        axes[0].bar(contract_churn.index, contract_churn.values, color=["#E53935", "#1E88E5", "#43A047"])
        axes[0].set_title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Churn Rate (%)", fontsize=10)
        axes[0].set_ylim(0, 50)
        for i, v in enumerate(contract_churn.values):
            axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # Internet Service vs Churn
        if "InternetService" in df_clean.columns:
            internet_churn = df_clean.groupby("InternetService")["Churn"].mean() * 100
            axes[1].bar(internet_churn.index, internet_churn.values, color=["#FB8C00", "#E53935", "#43A047"])
            axes[1].set_title("Churn Rate by Internet Service", fontsize=12, fontweight="bold")
            axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
            axes[1].set_ylim(0, 50)
            for i, v in enumerate(internet_churn.values):
                axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # Tenure Distribution by Churn
        sns.kdeplot(data=df_clean, x="tenure", hue="Churn", common_norm=False, fill=True, ax=axes[2], palette=["#1E88E5", "#E53935"])
        axes[2].set_title("Tenure Density: Retained (0) vs Churned (1)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Tenure (Months)", fontsize=10)

        path_eda_dist = self.output_dir / "eda_churn_distribution.png"
        plt.tight_layout()
        plt.savefig(path_eda_dist)
        plt.close(fig)

        # 2. Correlation Heatmap for Numerical Features
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        valid_num = [c for c in num_cols if c in df_clean.columns]
        corr_matrix = df_clean[valid_num].corr()

        fig_corr, ax_corr = plt.subplots(figsize=(7, 6), dpi=300)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, cbar=True, square=True)
        ax_corr.set_title("Numerical Features Correlation Matrix", fontsize=13, fontweight="bold", pad=12)

        path_eda_corr = self.output_dir / "eda_correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(path_eda_corr)
        plt.close(fig_corr)

        return path_eda_dist, path_eda_corr
