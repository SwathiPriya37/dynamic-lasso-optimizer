"""Training helpers for model development."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test subsets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
