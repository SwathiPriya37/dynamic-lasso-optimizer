"""Entry point for Dynamic Soft-Thresholding project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline_models import train_baseline_models
from src.data_loader import load_training_data, split_features_target
from src.dynamic_lasso_optimizer import DynamicLasso
from src.evaluation import evaluate_model
from src.preprocessing import HousePricePreprocessor
from src.training import split_train_test

RANDOM_STATE = 42
TEST_SIZE = 0.2


def plot_dynamic_lasso_curves(model: DynamicLasso, output_dir: Path) -> None:
    """Plot convergence and sparsity curves for Dynamic LASSO."""
    iterations = np.arange(1, len(model.objective_history_) + 1)
    if iterations.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, model.objective_history_, color="#0b5f8c", linewidth=2)
    ax.set_title("Dynamic LASSO Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_lasso_convergence.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, model.sparsity_history_, color="#af3a36", linewidth=2)
    ax.set_title("Dynamic LASSO Sparsity Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Non-Zero Coefficients")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_lasso_sparsity.png", dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar plots comparing RMSE and sparsity across models."""
    ordered = results_df.sort_values("RMSE").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ordered["Model"], ordered["RMSE"], color=["#386fa4", "#59a5d8", "#1f2d3d"])
    ax.set_title("Model Comparison: RMSE")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "model_rmse_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        ordered["Model"],
        ordered["Selected Features"],
        color=["#77b255", "#4f772d", "#2a9d8f"],
    )
    ax.set_title("Model Comparison: Selected Features")
    ax.set_ylabel("Count of Non-Zero Coefficients")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "model_sparsity_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run loading, preprocessing, training, evaluation, and visualization."""
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "train.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_training_data(data_path)
    X_df, y_series = split_features_target(data, target_col="SalePrice")

    print("Creating train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(
        X_df,
        y_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Preprocessing features...")
    preprocessor = HousePricePreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names()

    print("Training baseline models...")
    baseline_models = train_baseline_models(
        X_train,
        y_train.to_numpy(dtype=float),
        ridge_alpha=10.0,
        lasso_alpha=0.1,
        lasso_max_iter=20000,
    )

    print("Training Dynamic LASSO optimizer...")
    dynamic_lasso = DynamicLasso(
        learning_rate=0.01,
        lambda_0=10.0,
        max_iter=5000,
        tol=1e-6,
        fit_intercept=True,
        verbose=False,
    )
    dynamic_lasso.fit(X_train, y_train.to_numpy(dtype=float))

    models: dict[str, object] = {
        **baseline_models,
        "Dynamic LASSO": dynamic_lasso,
    }

    print("Evaluating models...")
    metrics = []
    for model_name, model in models.items():
        metrics.append(
            evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test.to_numpy(dtype=float),
            )
        )

    results_df = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
    print("\nEvaluation Results")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    metrics_path = results_dir / "model_comparison_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    selected_mask = np.abs(dynamic_lasso.coef_) > 1e-8
    selected_features = np.array(feature_names)[selected_mask]
    selected_features_path = results_dir / "dynamic_lasso_selected_features.csv"
    pd.DataFrame({"feature": selected_features}).to_csv(selected_features_path, index=False)

    plot_dynamic_lasso_curves(dynamic_lasso, results_dir)
    plot_model_comparison(results_df, results_dir)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved Dynamic LASSO selected features to: {selected_features_path}")
    print(f"Generated plots in: {results_dir}")


if __name__ == "__main__":
    main()
