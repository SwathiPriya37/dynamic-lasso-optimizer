"""Evaluation metrics for regression and sparsity."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error


def count_nonzero_weights(weights: np.ndarray, tolerance: float = 1e-8) -> int:
    """Count how many coefficients are effectively non-zero."""
    return int(np.count_nonzero(np.abs(weights) > tolerance))


def evaluate_model(
    model_name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tolerance: float = 1e-8,
) -> dict[str, float | int | str]:
    """Evaluate a regression model using MSE, RMSE, and sparsity."""
    y_true = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(model.predict(X_test), dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    if hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"), dtype=float).reshape(-1)
        selected_features = count_nonzero_weights(coefficients, tolerance=tolerance)
    else:
        selected_features = -1

    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "Selected Features": selected_features,
    }
