"""Data loading utilities for the House Prices dataset."""

from pathlib import Path

import pandas as pd


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    """Load the Kaggle House Prices training dataset.

    Args:
        data_path: Path to the train.csv file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the target column is missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    data = pd.read_csv(data_path)
    if "SalePrice" not in data.columns:
        raise ValueError("Expected target column 'SalePrice' in training data.")

    return data


def split_features_target(data: pd.DataFrame, target_col: str = "SalePrice") -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target.

    Args:
        data: Input DataFrame.
        target_col: Name of target column.

    Returns:
        Tuple of (X, y).
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y
