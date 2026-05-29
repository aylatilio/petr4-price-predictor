"""
Evaluation metrics module.

Provides standard regression metrics for time-series forecasting:
  - RMSE  (Root Mean Squared Error)
  - MAE   (Mean Absolute Error)
  - MAPE  (Mean Absolute Percentage Error)
  - R²    (Coefficient of determination)

All functions work on NumPy arrays and are scaler-agnostic — pass
inverse-transformed (real BRL) values for interpretable results.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Penalises large errors more heavily than MAE.
    Units are the same as the target variable (BRL).

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Robust to outliers. Units are the same as the target (BRL).

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Expresses error relative to the true value (percentage). A small
    epsilon avoids division by zero when y_true contains zeros.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
        epsilon: Small constant to prevent division by zero. Default 1e-8.

    Returns:
        MAPE as a percentage (e.g. 3.2 means 3.2%).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    1.0 is perfect; 0.0 means the model explains none of the variance;
    negative values mean it performs worse than a constant-mean baseline.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        R² value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Evaluation",
) -> dict:
    """Compute RMSE, MAE, MAPE and R² and log them.

    Args:
        y_true: Ground-truth values (real BRL prices recommended).
        y_pred: Model predictions (same scale as y_true).
        label: Label used in log messages (e.g. 'Test', 'Walk-forward').

    Returns:
        Dict with keys 'rmse', 'mae', 'mape', 'r2'.
    """
    metrics = {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }
    logger.info(
        "[%s] RMSE=%.4f | MAE=%.4f | MAPE=%.2f%% | R²=%.4f",
        label,
        metrics["rmse"],
        metrics["mae"],
        metrics["mape"],
        metrics["r2"],
    )
    return metrics
