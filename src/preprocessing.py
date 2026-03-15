"""Preprocessing utilities for the House Prices dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class HousePricePreprocessor:
    """Fit and apply preprocessing for mixed numerical/categorical features."""

    def __init__(self) -> None:
        self._preprocessor: ColumnTransformer | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame) -> "HousePricePreprocessor":
        """Fit imputers, encoders, and scalers on training features."""
        if X.empty:
            raise ValueError("Input features are empty.")

        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _build_one_hot_encoder()),
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )
        self._preprocessor.fit(X)
        self._feature_names = [str(name) for name in self._preprocessor.get_feature_names_out()]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted preprocessing pipeline."""
        if self._preprocessor is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit first.")

        transformed = self._preprocessor.transform(X)
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit preprocessing and transform in one call."""
        return self.fit(X).transform(X)

    def get_feature_names(self) -> list[str]:
        """Return transformed feature names after fitting."""
        if not self._feature_names:
            raise RuntimeError("Feature names are unavailable before fitting.")
        return self._feature_names.copy()
