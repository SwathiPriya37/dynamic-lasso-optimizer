"""Custom Dynamic LASSO optimizer using proximal gradient descent."""

from __future__ import annotations

import numpy as np


class DynamicLasso:
    """L1-regularized linear regression with dynamic soft-thresholding.

    Objective (at iteration t):
        (1 / 2n) * ||Xw - y||^2 + lambda_t * ||w||_1

    where lambda_t = lambda_0 / sqrt(t).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lambda_0: float = 0.1,
        max_iter: int = 5000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        verbose: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if lambda_0 < 0:
            raise ValueError("lambda_0 must be non-negative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.learning_rate = learning_rate
        self.lambda_0 = lambda_0
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.objective_history_: list[float] = []
        self.lambda_history_: list[float] = []
        self.sparsity_history_: list[int] = []
        self.n_iter_: int = 0

    @staticmethod
    def _soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
        """Apply element-wise soft thresholding."""
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    @staticmethod
    def _objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lambda_t: float) -> float:
        """Compute optimization objective value."""
        residual = X @ w - y
        data_term = 0.5 * np.mean(residual ** 2)
        l1_term = lambda_t * np.sum(np.abs(w))
        return float(data_term + l1_term)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DynamicLasso":
        """Fit model parameters using proximal gradient updates."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.")

        n_samples, n_features = X.shape

        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_opt = X - x_mean
            y_opt = y - y_mean
        else:
            x_mean = np.zeros(n_features)
            y_mean = 0.0
            X_opt = X
            y_opt = y

        w = np.zeros(n_features, dtype=float)
        self.objective_history_ = []
        self.lambda_history_ = []
        self.sparsity_history_ = []

        for iteration in range(1, self.max_iter + 1):
            residual = X_opt @ w - y_opt
            grad = (X_opt.T @ residual) / n_samples

            lambda_t = self.lambda_0 / np.sqrt(iteration)
            z = w - self.learning_rate * grad
            threshold = lambda_t
            w_new = self._soft_threshold(z, threshold)

            objective_value = self._objective(X_opt, y_opt, w_new, lambda_t)
            non_zero = int(np.count_nonzero(np.abs(w_new) > 1e-8))

            self.objective_history_.append(objective_value)
            self.lambda_history_.append(float(lambda_t))
            self.sparsity_history_.append(non_zero)

            if self.verbose and iteration % 500 == 0:
                print(
                    f"Iter={iteration:5d} | Obj={objective_value:.6f} "
                    f"| Lambda_t={lambda_t:.6f} | NonZero={non_zero}"
                )

            if np.linalg.norm(w_new - w, ord=2) < self.tol:
                w = w_new
                self.n_iter_ = iteration
                break

            w = w_new
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w
        self.intercept_ = float(y_mean - np.dot(x_mean, w)) if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for given features."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit first.")

        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def count_selected_features(self, tolerance: float = 1e-8) -> int:
        """Count non-zero coefficients using a tolerance threshold."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit first.")
        return int(np.count_nonzero(np.abs(self.coef_) > tolerance))
