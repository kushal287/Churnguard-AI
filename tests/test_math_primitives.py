"""
Tests for Core Mathematical Primitives (Sigmoid, Loss, Analytical Gradients).
"""

import numpy as np
import pytest
from src.models.custom_logistic_regression import CustomLogisticRegression


class TestMathPrimitives:
    """Rigorous tests for numerical stability and gradient correctness."""

    def test_sigmoid_standard_points(self):
        """Verify sigmoid at canonical points: z=0 -> 0.5, z=large -> 1, z=-large -> 0."""
        z = np.array([-100.0, -2.0, 0.0, 2.0, 100.0])
        sig = CustomLogisticRegression.sigmoid(z)

        assert abs(sig[2] - 0.5) < 1e-7, "Sigmoid(0) must equal 0.5"
        assert abs(sig[0] - 0.0) < 1e-7, "Sigmoid(-100) must approach 0.0"
        assert abs(sig[4] - 1.0) < 1e-7, "Sigmoid(100) must approach 1.0"
        assert np.all(sig >= 0.0) and np.all(sig <= 1.0), "Sigmoid output must be in [0, 1]"

    def test_sigmoid_numerical_overflow_protection(self):
        """Verify that extreme inputs (e.g. z = +/- 1000) do not produce NaN or Inf."""
        z_extreme = np.array([-1000.0, -500.0, 500.0, 1000.0])
        sig = CustomLogisticRegression.sigmoid(z_extreme)

        assert not np.isnan(sig).any(), "Sigmoid produced NaN on extreme values!"
        assert not np.isinf(sig).any(), "Sigmoid produced Inf on extreme values!"
        assert sig[0] < 1e-15, "Sigmoid(-1000) must be virtually 0"
        assert abs(sig[-1] - 1.0) < 1e-15, "Sigmoid(1000) must be virtually 1"

    def test_sigmoid_monotonicity(self):
        """Verify sigmoid is strictly monotonically increasing."""
        z = np.linspace(-10, 10, 100)
        sig = CustomLogisticRegression.sigmoid(z)
        assert np.all(np.diff(sig) > 0), "Sigmoid must be strictly monotonically increasing"

    def test_loss_computation_bounds(self):
        """Verify binary cross-entropy loss properties."""
        model = CustomLogisticRegression(l2_lambda=0.0)
        y_true = np.array([1, 0, 1, 0], dtype=float)

        # Perfect predictions: loss should be close to 0
        p_perfect = np.array([0.9999, 0.0001, 0.9999, 0.0001])
        loss_perfect = model.compute_loss(y_true, p_perfect)
        assert loss_perfect < 1e-3, "Loss on near-perfect predictions should be near 0"

        # Worst predictions: loss should be large
        p_worst = np.array([0.0001, 0.9999, 0.0001, 0.9999])
        loss_worst = model.compute_loss(y_true, p_worst)
        assert loss_worst > 5.0, "Loss on terrible predictions should be high"

    def test_l2_regularization_penalty_effect(self):
        """Verify that L2 regularization penalty adds strictly positive penalty for non-zero weights."""
        model_no_reg = CustomLogisticRegression(l2_lambda=0.0)
        model_with_reg = CustomLogisticRegression(l2_lambda=1.0)

        y_true = np.array([1, 0], dtype=float)
        p = np.array([0.7, 0.3])
        w = np.array([2.0, -3.0])

        loss_no_reg = model_no_reg.compute_loss(y_true, p, weights=w)
        loss_with_reg = model_with_reg.compute_loss(y_true, p, weights=w)

        expected_l2 = (1.0 / (2.0 * len(y_true))) * np.sum(w ** 2)
        assert abs((loss_with_reg - loss_no_reg) - expected_l2) < 1e-7

    def test_analytical_vs_numerical_gradient_checking(self):
        """
        Verify that analytical gradient equals numerical two-sided finite difference:
        grad_num = [J(theta + eps) - J(theta - eps)] / (2 * eps)
        Relative error must be < 1e-5.
        """
        rng = np.random.RandomState(42)
        m, d = 30, 6
        X = rng.randn(m, d)
        y = rng.binomial(1, 0.5, size=m).astype(float)
        w = rng.randn(d)
        b = 0.45

        model = CustomLogisticRegression(l2_lambda=0.05, use_class_weights=False, fit_intercept=True)

        # Analytical gradients
        z = np.dot(X, w) + b
        p = model.sigmoid(z)
        grad_w_ana, grad_b_ana = model.compute_gradients(X, y, p, w)

        # Numerical gradient checking for weights
        eps = 1e-6
        grad_w_num = np.zeros(d)
        for j in range(d):
            w_plus = w.copy(); w_plus[j] += eps
            w_minus = w.copy(); w_minus[j] -= eps
            p_plus = model.sigmoid(np.dot(X, w_plus) + b)
            p_minus = model.sigmoid(np.dot(X, w_minus) + b)
            loss_plus = model.compute_loss(y, p_plus, weights=w_plus)
            loss_minus = model.compute_loss(y, p_minus, weights=w_minus)
            grad_w_num[j] = (loss_plus - loss_minus) / (2.0 * eps)

        rel_error_w = np.linalg.norm(grad_w_ana - grad_w_num) / (
            np.linalg.norm(grad_w_ana) + np.linalg.norm(grad_w_num)
        )
        assert rel_error_w < 1e-5, f"Weight gradient error too high: {rel_error_w}"

        # Numerical gradient checking for bias
        p_b_plus = model.sigmoid(np.dot(X, w) + (b + eps))
        p_b_minus = model.sigmoid(np.dot(X, w) + (b - eps))
        loss_b_plus = model.compute_loss(y, p_b_plus, weights=w)
        loss_b_minus = model.compute_loss(y, p_b_minus, weights=w)
        grad_b_num = (loss_b_plus - loss_b_minus) / (2.0 * eps)

        rel_error_b = abs(grad_b_ana - grad_b_num) / (abs(grad_b_ana) + abs(grad_b_num) + 1e-12)
        assert rel_error_b < 1e-5, f"Bias gradient error too high: {rel_error_b}"
