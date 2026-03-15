"""Baseline models for feature selection and regression comparison."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, Ridge


def train_ridge_model(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 10.0) -> Ridge:
    """Train a Ridge regression model."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_lasso_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 20000,
    random_state: int = 42,
) -> Lasso:
    """Train a standard sklearn LASSO model."""
    model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float = 10.0,
    lasso_alpha: float = 0.1,
    lasso_max_iter: int = 20000,
) -> dict[str, object]:
    """Train baseline Ridge and LASSO models."""
    ridge_model = train_ridge_model(X_train, y_train, alpha=ridge_alpha)
    lasso_model = train_lasso_model(
        X_train,
        y_train,
        alpha=lasso_alpha,
        max_iter=lasso_max_iter,
    )
    return {
        "Ridge Regression": ridge_model,
        "LASSO Regression": lasso_model,
    }
