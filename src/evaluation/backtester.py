"""
Walk-forward backtesting module.

Implements a walk-forward validation strategy:
  - Start with an initial training window.
  - Train the model on that window.
  - Predict `forecast_horizon` days ahead.
  - Advance the window by `step_size` days.
  - Repeat until the end of the dataset.

This mimics real-world deployment where the model is periodically retrained
on the most recent data and evaluated on genuinely unseen future data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessor import build_sequences, fit_and_save_scaler
from src.evaluation.metrics import compute_all_metrics
from src.models.lstm_model import build_lstm_model

logger = logging.getLogger(__name__)


def run_walk_forward_validation(
    df_full: pd.DataFrame,
    feature_columns: list[str],
    target_col_index: int,
    config: dict,
    artifacts_dir: str,
) -> dict:
    """Run walk-forward validation on the full feature-engineered DataFrame.

    For each step:
      1. Slice the training window.
      2. Fit a fresh scaler on that window.
      3. Build sequences.
      4. Train the model.
      5. Predict the next `forecast_horizon` steps.
      6. Record actuals and predictions.

    Args:
        df_full: Full feature-engineered DataFrame (output of add_technical_indicators).
        feature_columns: List of column names used as model features.
        target_col_index: Index of the target column ('Close') in feature_columns.
        config: Full project config dict.
        artifacts_dir: Directory for temporary scaler artefacts during backtesting.

    Returns:
        Dict with keys:
          - 'dates':       DatetimeIndex of prediction dates.
          - 'actuals':     Array of true close prices.
          - 'predictions': Array of predicted close prices.
          - 'metrics':     Dict of RMSE, MAE, MAPE, R².
    """
    wf_cfg = config["evaluation"]["walk_forward"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    sequence_length = config["features"]["sequence_length"]
    n_features = len(feature_columns)

    data_array = df_full[feature_columns].values
    dates = df_full.index
    n_total = len(data_array)

    initial_train_size = int(n_total * wf_cfg["initial_train_size"])
    step_size = wf_cfg["step_size"]
    forecast_horizon = wf_cfg["forecast_horizon"]

    all_actuals: list[float] = []
    all_predictions: list[float] = []
    all_dates: list = []

    train_end = initial_train_size
    fold = 0

    logger.info(
        "Walk-forward validation: initial_train=%d | step=%d | horizon=%d",
        initial_train_size,
        step_size,
        forecast_horizon,
    )

    while train_end + forecast_horizon <= n_total:
        fold += 1
        test_end = min(train_end + forecast_horizon, n_total)

        # --- Fit scaler on current training window ---
        train_window = data_array[:train_end]
        scaler_path = str(Path(artifacts_dir) / f"wf_scaler_fold_{fold}.pkl")
        scaler = fit_and_save_scaler(train_window, scaler_path)

        # Scale full window including test horizon (for sequence continuity)
        scaled = scaler.transform(data_array[: test_end + sequence_length])

        # Build sequences
        X_train_seq, y_train_seq = build_sequences(
            scaled[:train_end], sequence_length, target_col_index
        )

        if len(X_train_seq) == 0:
            logger.warning(
                "Fold %d: not enough data to build sequences. Skipping.", fold
            )
            train_end += step_size
            continue

        # Train a fresh model for this fold
        input_shape = (sequence_length, n_features)
        model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=model_cfg["lstm_units"],
            dense_units=model_cfg["dense_units"],
            dropout_rate=model_cfg["dropout_rate"],
            recurrent_dropout=model_cfg["recurrent_dropout"],
            learning_rate=train_cfg["learning_rate"],
        )

        # Use fewer epochs during backtesting to keep runtime manageable
        model.fit(
            X_train_seq,
            y_train_seq,
            epochs=min(train_cfg["epochs"], 30),
            batch_size=train_cfg["batch_size"],
            validation_split=train_cfg["validation_split"],
            shuffle=False,
            verbose=0,
        )

        # Predict `forecast_horizon` steps
        for h in range(forecast_horizon):
            pred_idx = train_end + h
            if pred_idx >= n_total:
                break

            seq_start = pred_idx - sequence_length
            if seq_start < 0:
                continue

            sequence = scaled[seq_start:pred_idx].reshape(
                1, sequence_length, n_features
            )
            scaled_pred = model.predict(sequence, verbose=0)[0][0]

            # Inverse transform
            dummy = np.zeros((1, n_features))
            dummy[0, target_col_index] = scaled_pred
            price = scaler.inverse_transform(dummy)[0, target_col_index]

            actual_price = data_array[pred_idx, target_col_index]

            all_predictions.append(float(price))
            all_actuals.append(float(actual_price))
            all_dates.append(dates[pred_idx])

        logger.info(
            "Fold %d | train_end=%d (%s) | pred range: %s → %s",
            fold,
            train_end,
            dates[train_end - 1].date() if train_end < n_total else "end",
            dates[train_end].date() if train_end < n_total else "—",
            dates[test_end - 1].date() if test_end <= n_total else "—",
        )

        train_end += step_size

    actuals_arr = np.array(all_actuals)
    preds_arr = np.array(all_predictions)
    metrics = compute_all_metrics(actuals_arr, preds_arr, label="Walk-forward")

    return {
        "dates": pd.DatetimeIndex(all_dates),
        "actuals": actuals_arr,
        "predictions": preds_arr,
        "metrics": metrics,
    }
