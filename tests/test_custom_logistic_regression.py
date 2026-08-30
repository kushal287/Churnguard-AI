"""
Tests for Custom Logistic Regression Model Training, Convergence, Bounds, and Serialization.
"""

from pathlib import Path
import numpy as np
import pytest
from src.models.custom_logistic_regression import CustomLogisticRegression


class TestCustomLogisticRegression:
    """Test model training behaviors and interface contracts."""

    def test_convergence_on_synthetic_data(self):
        """Verify model achieves near-zero training loss on separable data."""
        rng = np.random.RandomState(42)
        X0 = rng.randn(100, 3) - 2.0
        X1 = rng.randn(100, 3) + 2.0
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100, dtype=float)

        model = CustomLogisticRegression(
            learning_rate=0.1,
            max_iter=300,
            l2_lambda=0.001,
            batch_size=32,
            early_stopping=False,
            random_state=42
        )
        model.fit(X, y)

        assert model.is_fitted
        assert len(model.train_loss_history_) > 0
        assert model.train_loss_history_[-1] < 0.15, "Model failed to converge on separable data"

        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc >= 0.98, f"Expected accuracy >= 0.98, got {acc}"

    def test_predict_proba_shapes_and_bounds(self):
        """Verify probability outputs are strictly in [0, 1] and sum to 1."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4)
        y = rng.binomial(1, 0.5, size=50).astype(float)

        model = CustomLogisticRegression(max_iter=100, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (50, 2), f"Expected shape (50, 2), got {proba.shape}"
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0), "Probabilities out of bounds"
        assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities do not sum to 1.0"

    def test_l2_regularization_shrinks_weights(self):
        """Verify higher L2 regularization lambda leads to smaller weight norm."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.binomial(1, 0.5, size=200).astype(float)

        model_low_reg = CustomLogisticRegression(l2_lambda=0.0001, max_iter=200, random_state=42)
        model_low_reg.fit(X, y)

        model_high_reg = CustomLogisticRegression(l2_lambda=1.0, max_iter=200, random_state=42)
        model_high_reg.fit(X, y)

        norm_low = np.linalg.norm(model_low_reg.weights)
        norm_high = np.linalg.norm(model_high_reg.weights)

        assert norm_high < norm_low, f"High reg norm ({norm_high}) should be less than low reg norm ({norm_low})"

    def test_class_weights_improves_minority_recall(self):
        """Verify balanced class weighting increases sensitivity / recall on imbalanced dataset."""
        rng = np.random.RandomState(42)
        # Create heavily imbalanced dataset: 90% negative, 10% positive
        X_neg = rng.randn(450, 4)
        X_pos = rng.randn(50, 4) + 0.8
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * 450 + [1] * 50, dtype=float)

        model_unweighted = CustomLogisticRegression(
            use_class_weights=False, learning_rate=0.05, max_iter=300, random_state=42
        )
        model_unweighted.fit(X, y)
        pred_unweighted = model_unweighted.predict(X, threshold=0.5)
        recall_unweighted = np.sum((pred_unweighted == 1) & (y == 1)) / 50.0

        model_weighted = CustomLogisticRegression(
            use_class_weights=True, learning_rate=0.05, max_iter=300, random_state=42
        )
        model_weighted.fit(X, y)
        pred_weighted = model_weighted.predict(X, threshold=0.5)
        recall_weighted = np.sum((pred_weighted == 1) & (y == 1)) / 50.0

        assert recall_weighted >= recall_unweighted, (
            f"Weighted recall ({recall_weighted}) should be >= unweighted recall ({recall_unweighted})"
        )

    def test_model_serialization_and_recovery(self, tmp_path):
        """Verify save() and load() restore bitwise identical predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        y = rng.binomial(1, 0.4, size=60).astype(float)

        model = CustomLogisticRegression(max_iter=150, random_state=42)
        model.fit(X, y)
        original_proba = model.predict_proba(X)

        save_file = tmp_path / "test_model.npz"
        model.save(save_file)

        loaded_model = CustomLogisticRegression.load(save_file)
        loaded_proba = loaded_model.predict_proba(X)

        assert np.allclose(original_proba, loaded_proba, atol=1e-12), "Loaded predictions do not match original"
        assert np.allclose(model.weights, loaded_model.weights, atol=1e-12), "Weights mismatch"
        assert abs(model.bias - loaded_model.bias) < 1e-12, "Bias mismatch"
