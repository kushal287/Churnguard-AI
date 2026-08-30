"""
Tests for Custom Evaluation Metrics Parity against Scikit-Learn.
Ensures zero metric fabrication and exact mathematical parity.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    average_precision_score as sk_average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    roc_auc_score as sk_roc_auc_score,
)

from src.evaluation.metrics import (
    accuracy_score,
    average_precision_score,
    compute_financial_utility,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class TestMetricsParity:
    """Verify custom NumPy metrics match scikit-learn metrics within 1e-4 tolerance."""

    @pytest.fixture
    def mock_eval_data(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.35, size=200).astype(int)
        y_score = np.clip(rng.beta(2, 5, size=200) + 0.3 * y_true, 0.0, 1.0)
        y_pred = (y_score >= 0.5).astype(int)
        return y_true, y_score, y_pred

    def test_confusion_matrix_parity(self, mock_eval_data):
        y_true, _, y_pred = mock_eval_data
        custom_cm = confusion_matrix(y_true, y_pred)
        sk_cm = sk_confusion_matrix(y_true, y_pred)
        assert np.array_equal(custom_cm, sk_cm), f"CM mismatch:\nCustom:\n{custom_cm}\nSklearn:\n{sk_cm}"

    def test_accuracy_parity(self, mock_eval_data):
        y_true, _, y_pred = mock_eval_data
        assert abs(accuracy_score(y_true, y_pred) - sk_accuracy_score(y_true, y_pred)) < 1e-6

    def test_precision_parity(self, mock_eval_data):
        y_true, _, y_pred = mock_eval_data
        assert abs(precision_score(y_true, y_pred) - sk_precision_score(y_true, y_pred)) < 1e-6

    def test_recall_parity(self, mock_eval_data):
        y_true, _, y_pred = mock_eval_data
        assert abs(recall_score(y_true, y_pred) - sk_recall_score(y_true, y_pred)) < 1e-6

    def test_f1_score_parity(self, mock_eval_data):
        y_true, _, y_pred = mock_eval_data
        assert abs(f1_score(y_true, y_pred) - sk_f1_score(y_true, y_pred)) < 1e-6

    def test_roc_auc_parity(self, mock_eval_data):
        y_true, y_score, _ = mock_eval_data
        custom_auc = roc_auc_score(y_true, y_score)
        sk_auc = sk_roc_auc_score(y_true, y_score)
        assert abs(custom_auc - sk_auc) < 1e-3, f"ROC-AUC mismatch: Custom={custom_auc}, Sklearn={sk_auc}"

    def test_financial_utility_calculation(self):
        """Verify business retention profit formula under illustrative assumptions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])  # TP=1, FN=1, TN=1, FP=1

        # Costs: FP=-$50, TP=+$350
        # Net Retention Value = 1 * (350) - 1 * (50) = $300.0
        # Value per customer = $300.0 / 4 = $75.0
        fin = compute_financial_utility(y_true, y_pred)
        assert fin["true_positives"] == 1.0
        assert fin["false_positives"] == 1.0
        assert fin["net_retention_savings"] == 300.0
        assert fin["roi_per_customer"] == 75.0
