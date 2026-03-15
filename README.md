# Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression

This project implements a complete machine learning workflow for a Numerical Optimization course using the Kaggle House Prices dataset.

The main contribution is a custom optimizer, Dynamic LASSO, based on proximal gradient descent with a dynamic threshold:

- Objective:
  - (1 / 2n) * ||Xw - y||^2 + lambda * ||w||_1
- Dynamic regularization schedule:
  - lambda_t = lambda_0 / sqrt(t)
- Soft-thresholding proximal step:
  - w = sign(z) * max(|z| - threshold, 0)

## Project Structure

```text
dynamic-lasso-optimizer/
|-- data/
|   |-- train.csv
|   |-- test.csv
|   |-- sample_submission.csv
|   `-- data_description.txt
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- baseline_models.py
|   |-- dynamic_lasso_optimizer.py
|   |-- training.py
|   `-- evaluation.py
|-- results/
|-- main.py
|-- requirements.txt
`-- README.md
```

## What Is Implemented

1. Data loading
	- Loads `data/train.csv`
	- Splits into feature matrix and target (`SalePrice`)

2. Preprocessing pipeline
	- Missing value handling:
	  - Numeric: median imputation
	  - Categorical: most-frequent imputation
	- Categorical encoding: one-hot encoding
	- Feature scaling: standardization on numeric columns

3. Baseline models
	- Ridge Regression (sklearn)
	- LASSO Regression (sklearn)

4. Custom model: DynamicLasso
	- Proximal gradient descent optimizer
	- Dynamic lambda schedule: lambda_t = lambda_0 / sqrt(t)
	- Iteration-wise objective, lambda, and sparsity tracking
	- Predict and selected-feature counting utilities

5. Evaluation
	- Mean Squared Error (MSE)
	- Root Mean Squared Error (RMSE)
	- Number of selected features (non-zero coefficients)

6. Visualization
	- Dynamic LASSO convergence curve (objective vs iteration)
	- Dynamic LASSO sparsity curve (non-zero coefficients vs iteration)
	- RMSE comparison bar chart across models
	- Sparsity comparison bar chart across models

## Setup

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

From the project root:

```bash
python main.py
```

## Outputs

The following files are generated in `results/` after running:

- `model_comparison_metrics.csv`
- `dynamic_lasso_selected_features.csv`
- `dynamic_lasso_convergence.png`
- `dynamic_lasso_sparsity.png`
- `model_rmse_comparison.png`
- `model_sparsity_comparison.png`

## Notes on the Dynamic Proximal Update

At each iteration t:

1. Compute gradient of smooth loss term:
	- grad = (X^T (Xw - y)) / n
2. Gradient step:
	- z = w - eta * grad
3. Dynamic lambda:
	- lambda_t = lambda_0 / sqrt(t)
4. Proximal soft-thresholding:
	- w_new = S_(eta * lambda_t)(z)

This gradually relaxes shrinkage over time, typically producing high sparsity early and better fine-tuning later.

## Course Context

This repository is designed for university submission in a Numerical Optimization course and emphasizes:

- Modular, readable code
- Reproducible train/test evaluation
- Algorithmic transparency for the custom optimizer
- Baseline comparison against standard regularized linear models