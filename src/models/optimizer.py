"""
Gradient Descent Optimization Routines for Pure NumPy Logistic Regression.
Supports Batch, Mini-batch, and Momentum Gradient Descent.
"""

from typing import Generator, Optional, Tuple
import numpy as np


class MiniBatchOptimizer:
    """Handles mini-batch generation and momentum state updates."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        momentum: float = 0.9,
        batch_size: Optional[int] = 64,
        lr_decay: float = 0.0,
        random_state: int = 42
    ):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.v_w: Optional[np.ndarray] = None
        self.v_b: float = 0.0

    def reset_state(self, n_features: int) -> None:
        """Reset velocity vectors for fresh training run."""
        self.v_w = np.zeros(n_features, dtype=np.float64)
        self.v_b = 0.0
        self.learning_rate = self.initial_lr
        self.rng = np.random.RandomState(self.random_state)

    def get_batches(
        self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        Yield shuffled mini-batches or the entire batch if batch_size is None or >= m.
        """
        m = X.shape[0]
        if self.batch_size is None or self.batch_size >= m:
            yield X, y, sample_weights
            return

        indices = self.rng.permutation(m)
        for start_idx in range(0, m, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            w_batch = sample_weights[batch_idx] if sample_weights is not None else None
            yield X_batch, y_batch, w_batch

    def update_parameters(
        self,
        w: np.ndarray,
        b: float,
        grad_w: np.ndarray,
        grad_b: float,
        epoch: int
    ) -> Tuple[np.ndarray, float]:
        """
        Perform momentum parameter update:
        v = beta * v + (1 - beta) * grad
        param = param - lr * v
        """
        if self.v_w is None:
            self.v_w = np.zeros_like(w)

        # Decay learning rate if configured: lr = lr_0 / (1 + decay * epoch)
        current_lr = self.initial_lr / (1.0 + self.lr_decay * epoch)

        if self.momentum > 0.0:
            self.v_w = self.momentum * self.v_w + (1.0 - self.momentum) * grad_w
            self.v_b = self.momentum * self.v_b + (1.0 - self.momentum) * grad_b
            step_w = self.v_w
            step_b = self.v_b
        else:
            step_w = grad_w
            step_b = grad_b

        w_new = w - current_lr * step_w
        b_new = b - current_lr * step_b

        return w_new, float(b_new)
