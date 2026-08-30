"""
Custom Logistic Regression Implementation from Scratch in Pure NumPy.
Primary Machine Learning Model for ChurnGuard AI.

Includes:
- Numerically stable Sigmoid activation
- Weighted Binary Cross-Entropy (Log-Loss)
- Analytical Gradient Computation
- Mini-batch & Momentum Gradient Descent
- L2 Regularization (Weight Decay)
- Class Imbalance Mitigation (Balanced Class Weights)
- Early Stopping & Convergence Telemetry
- Serialization & Model Recovery
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.models.optimizer import MiniBatchOptimizer

logger = logging.getLogger(__name__)


class CustomLogisticRegression:
    """
    Pure NumPy Logistic Regression Classifier.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1500,
        l2_lambda: float = 0.01,
        momentum: float = 0.9,
        batch_size: Optional[int] = 64,
        early_stopping: bool = True,
        patience: int = 50,
        tolerance: float = 1e-6,
        use_class_weights: bool = True,
        fit_intercept: bool = True,
        lr_decay: float = 0.0,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.tolerance = tolerance
        self.use_class_weights = use_class_weights
        self.fit_intercept = fit_intercept
        self.lr_decay = lr_decay
        self.random_state = random_state

        # Model Parameters
        self.weights: Optional[np.ndarray] = None  # Shape: (d,)
        self.bias: float = 0.0
        self.class_weights_: Dict[int, float] = {0: 1.0, 1: 1.0}
        self.n_features_in_: int = 0
        self.classes_: np.ndarray = np.array([0, 1])

        # Training history
        self.train_loss_history_: List[float] = []
        self.val_loss_history_: List[float] = []
        self.best_epoch_: int = 0
        self.best_val_loss_: float = float("inf")
        self.best_weights_: Optional[np.ndarray] = None
        self.best_bias_: float = 0.0
        self.is_fitted: bool = False

        self.optimizer = MiniBatchOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            batch_size=self.batch_size,
            lr_decay=self.lr_decay,
            random_state=self.random_state,
        )

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid activation function.
        Prevents overflow for large positive or negative inputs:
        - For z >= 0: 1 / (1 + exp(-z))
        - For z < 0:  exp(z) / (1 + exp(z))
        """
        # Clip z to prevent numerical overflow in exponential
        z_clipped = np.clip(z, -500.0, 500.0)
        
        # Branching implementation for absolute float stability
        pos_mask = z_clipped >= 0
        neg_mask = ~pos_mask
        
        result = np.empty_like(z_clipped, dtype=np.float64)
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z_clipped[pos_mask]))
        exp_neg = np.exp(z_clipped[neg_mask])
        result[neg_mask] = exp_neg / (1.0 + exp_neg)
        
        return result

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute inverse class frequency weights (balanced):
        w_k = N / (2 * N_k)
        """
        n_samples = len(y)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        if n_pos == 0 or n_neg == 0:
            return {0: 1.0, 1: 1.0}

        w0 = n_samples / (2.0 * n_neg)
        w1 = n_samples / (2.0 * n_pos)
        return {0: float(w0), 1: float(w1)}

    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Assign sample weights based on binary label and computed class weights."""
        if not self.use_class_weights:
            return np.ones_like(y, dtype=np.float64)
        sample_weights = np.where(y == 1, self.class_weights_[1], self.class_weights_[0])
        return sample_weights.astype(np.float64)

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        weights: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Weighted Binary Cross-Entropy Loss with L2 Regularization Penalty.
        
        J(w, b) = - 1/m * sum [ v_i * (y_i * ln(p_i + eps) + (1 - y_i) * ln(1 - p_i + eps)) ]
                  + (lambda / (2 * m)) * ||w||_2^2
        """
        m = len(y_true)
        if m == 0:
            return 0.0

        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1.0 - eps)

        if sample_weights is not None:
            bce_per_sample = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            loss = np.sum(sample_weights * bce_per_sample) / m
        else:
            loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

        # Add L2 Regularization term: (lambda / (2 * m)) * sum(w^2)
        if self.l2_lambda > 0.0 and weights is not None:
            l2_penalty = (self.l2_lambda / (2.0 * m)) * np.sum(weights ** 2)
            loss += l2_penalty

        return float(loss)

    def compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        weights: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute analytical gradients with class weights and L2 regularization:
        
        dJ/dw = 1/m * X^T (v * (y_hat - y)) + (lambda / m) * w
        dJ/db = 1/m * sum(v * (y_hat - y))
        """
        m = X.shape[0]
        residuals = y_pred_proba - y_true  # (m,)

        if sample_weights is not None:
            weighted_residuals = residuals * sample_weights
        else:
            weighted_residuals = residuals

        # Gradient w.r.t weights
        grad_w = (1.0 / m) * np.dot(X.T, weighted_residuals)

        # Add L2 Regularization gradient: (lambda / m) * w
        if self.l2_lambda > 0.0:
            grad_w += (self.l2_lambda / m) * weights

        # Gradient w.r.t bias
        if self.fit_intercept:
            grad_b = float((1.0 / m) * np.sum(weighted_residuals))
        else:
            grad_b = 0.0

        return grad_w, grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CustomLogisticRegression":
        """
        Train Custom Logistic Regression model using Gradient Descent.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        m, d = X.shape
        self.n_features_in_ = d

        # Compute class weights if enabled
        if self.use_class_weights:
            self.class_weights_ = self._compute_class_weights(y)
        else:
            self.class_weights_ = {0: 1.0, 1: 1.0}

        sample_weights_train = self._get_sample_weights(y)
        sample_weights_val = self._get_sample_weights(y_val) if y_val is not None else None

        # Initialize parameters using Xavier/Glorot normal initialization
        rng = np.random.RandomState(self.random_state)
        limit = np.sqrt(2.0 / (d + 1))
        self.weights = rng.uniform(-limit, limit, size=d)
        self.bias = 0.0

        self.optimizer.reset_state(d)
        self.train_loss_history_ = []
        self.val_loss_history_ = []

        self.best_val_loss_ = float("inf")
        self.best_epoch_ = 0
        self.best_weights_ = self.weights.copy()
        self.best_bias_ = self.bias

        no_improvement_count = 0

        for epoch in range(1, self.max_iter + 1):
            # Mini-batch gradient descent passes
            for X_batch, y_batch, w_batch in self.optimizer.get_batches(X, y, sample_weights_train):
                # Forward pass on batch
                z_batch = np.dot(X_batch, self.weights) + self.bias
                p_batch = self.sigmoid(z_batch)

                # Compute gradients on batch
                grad_w, grad_b = self.compute_gradients(
                    X_batch, y_batch, p_batch, self.weights, w_batch
                )

                # Update parameters with optimizer
                self.weights, self.bias = self.optimizer.update_parameters(
                    self.weights, self.bias, grad_w, grad_b, epoch
                )

            # Compute full-epoch train loss
            z_train_full = np.dot(X, self.weights) + self.bias
            p_train_full = self.sigmoid(z_train_full)
            train_loss = self.compute_loss(y, p_train_full, self.weights, sample_weights_train)
            self.train_loss_history_.append(train_loss)

            # Validation loss evaluation
            if X_val is not None and y_val is not None:
                z_val = np.dot(X_val, self.weights) + self.bias
                p_val = self.sigmoid(z_val)
                val_loss = self.compute_loss(y_val, p_val, self.weights, sample_weights_val)
                self.val_loss_history_.append(val_loss)

                # Check validation improvement
                if val_loss < self.best_val_loss_ - self.tolerance:
                    self.best_val_loss_ = val_loss
                    self.best_epoch_ = epoch
                    self.best_weights_ = self.weights.copy()
                    self.best_bias_ = self.bias
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping trigger
                if self.early_stopping and no_improvement_count >= self.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best validation loss {self.best_val_loss_:.5f} at epoch {self.best_epoch_}."
                    )
                    # Restore best parameters
                    self.weights = self.best_weights_.copy()
                    self.bias = self.best_bias_
                    break
            else:
                # If no validation set, track best training loss
                if train_loss < self.best_val_loss_:
                    self.best_val_loss_ = train_loss
                    self.best_epoch_ = epoch
                    self.best_weights_ = self.weights.copy()
                    self.best_bias_ = self.bias

        # Finalize model state
        if self.early_stopping and self.best_weights_ is not None:
            self.weights = self.best_weights_.copy()
            self.bias = self.best_bias_

        self.is_fitted = True
        logger.info(
            f"Training completed. Epochs: {len(self.train_loss_history_)}, "
            f"Final Train Loss: {self.train_loss_history_[-1]:.5f}, "
            f"Best Loss: {self.best_val_loss_:.5f} (Epoch {self.best_epoch_})"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction probabilities for binary classes [P(y=0), P(y=1)].
        
        Returns:
            np.ndarray of shape (m, 2)
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before calling predict_proba()!")

        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        p1 = self.sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict discrete binary class labels based on a decision threshold.
        """
        proba = self.predict_proba(X)
        p1 = proba[:, 1]
        return (p1 >= threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "l2_lambda": self.l2_lambda,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "use_class_weights": self.use_class_weights,
            "fit_intercept": self.fit_intercept,
            "lr_decay": self.lr_decay,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomLogisticRegression":
        """Set hyperparameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save trained model parameters and state to .npz file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model!")

        from config.config import CUSTOM_MODEL_PATH
        path = Path(filepath) if filepath else CUSTOM_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        params = self.get_params()
        history = {
            "train_loss": self.train_loss_history_,
            "val_loss": self.val_loss_history_,
            "best_epoch": self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
        }

        np.savez_compressed(
            path,
            weights=self.weights,
            bias=np.array([self.bias]),
            classes=self.classes_,
            class_weights_0=np.array([self.class_weights_[0]]),
            class_weights_1=np.array([self.class_weights_[1]]),
            n_features_in=np.array([self.n_features_in_]),
            params_json=np.array([json.dumps(params)]),
            history_json=np.array([json.dumps(history)]),
        )

        logger.info(f"Model parameters successfully saved to {path}")
        return path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "CustomLogisticRegression":
        """Load trained model from .npz file."""
        from config.config import CUSTOM_MODEL_PATH
        path = Path(filepath) if filepath else CUSTOM_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        data = np.load(path, allow_pickle=True)
        params = json.loads(str(data["params_json"][0]))
        history = json.loads(str(data["history_json"][0]))

        model = cls(**params)
        model.weights = data["weights"]
        model.bias = float(data["bias"][0])
        model.classes_ = data["classes"]
        model.class_weights_ = {
            0: float(data["class_weights_0"][0]),
            1: float(data["class_weights_1"][0]),
        }
        model.n_features_in_ = int(data["n_features_in"][0])
        model.train_loss_history_ = history["train_loss"]
        model.val_loss_history_ = history["val_loss"]
        model.best_epoch_ = history["best_epoch"]
        model.best_val_loss_ = history["best_val_loss"]
        model.is_fitted = True

        logger.info(f"Model loaded successfully from {path}")
        return model
