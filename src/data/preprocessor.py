"""
Data preprocessing module.

Handles the full preprocessing pipeline:
  1. Feature engineering (technical indicators)
  2. Scaling with MinMaxScaler (fit on train, apply to test)
  3. Sequence building — converts the flat DataFrame into overlapping
     windows of shape (samples, sequence_length, features) for LSTM input.

Scaler persistence and loading are also handled here so that the exact
same transformation can be applied at inference time.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.feature_engineering import add_technical_indicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------


def fit_and_save_scaler(
    data: np.ndarray,
    scaler_path: str,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> MinMaxScaler:
    """Fit a MinMaxScaler on *training* data and persist it to disk.

    Args:
        data: 2-D array of shape (n_samples, n_features).
        scaler_path: File path where the scaler will be saved (.pkl).
        feature_range: Desired output range. Default is (0, 1).

    Returns:
        Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(data)

    path = Path(scaler_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Scaler fitted and saved to %s", scaler_path)
    return scaler


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Load a previously fitted scaler from disk.

    Args:
        scaler_path: Path to the .pkl scaler file.

    Returns:
        Loaded MinMaxScaler instance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(path, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Scaler loaded from %s", scaler_path)
    return scaler


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------


def build_sequences(
    scaled_data: np.ndarray,
    sequence_length: int,
    target_col_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping look-back windows (X) and next-step targets (y).

    Each sample X[i] contains `sequence_length` consecutive time steps of all
    features; the corresponding y[i] is the *scaled* target value at step i+1.

    Args:
        scaled_data: 2-D array of shape (n_timesteps, n_features), already scaled.
        sequence_length: Number of past time steps to use as input.
        target_col_index: Column index of the target variable (Close price).

    Returns:
        Tuple (X, y) where:
          - X has shape (n_samples, sequence_length, n_features)
          - y has shape (n_samples,)
    """
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i, :])
        y.append(scaled_data[i, target_col_index])

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(
    raw_df: pd.DataFrame,
    config: dict,
    artifacts_dir: str,
    processed_dir: str,
    is_training: bool = True,
) -> dict:
    """Execute the full preprocessing pipeline.

    Steps:
      1. Add technical indicators.
      2. Split into train / test according to config test_start_date.
      3. Scale features (fit on train only; transform both splits).
      4. Build sequences for LSTM.
      5. Persist artefacts (scaler, feature column list, processed arrays).

    Args:
        raw_df: Raw OHLCV DataFrame (output of collector).
        config: Full project config dict (loaded from config.yaml).
        artifacts_dir: Directory to save scaler and feature columns.
        processed_dir: Directory to save numpy arrays.
        is_training: When True, fit the scaler. When False, load it.

    Returns:
        Dictionary with keys:
          - 'X_train', 'y_train': Training sequences and targets.
          - 'X_test',  'y_test':  Test sequences and targets.
          - 'scaler':             Fitted MinMaxScaler.
          - 'feature_columns':   List of column names used as features.
          - 'target_col_index':  Integer index of the target column.
          - 'df_full':           DataFrame after feature engineering.
          - 'test_start_index':  Integer position where test set begins.
    """
    feat_cfg = config["features"]
    ind_cfg = feat_cfg["technical_indicators"]
    sequence_length = feat_cfg["sequence_length"]
    target_column = feat_cfg["target_column"]
    test_start_date = config["data"]["test_start_date"]
    scaler_path = str(Path(artifacts_dir) / config["paths"]["scaler_filename"])
    feature_columns_path = str(
        Path(artifacts_dir) / config["paths"]["feature_columns_filename"]
    )

    # 1. Feature engineering
    df = add_technical_indicators(raw_df, ind_cfg)

    # Determine feature columns (all numeric columns in the DataFrame)
    feature_columns = [
        c for c in df.columns if df[c].dtype in [np.float64, np.float32, float]
    ]
    target_col_index = feature_columns.index(target_column)
    logger.info(
        "Using %d features. Target '%s' is at index %d.",
        len(feature_columns),
        target_column,
        target_col_index,
    )

    # 2. Train / test split by date
    df_train = df[df.index < test_start_date]
    df_test = df[df.index >= test_start_date]
    logger.info(
        "Train: %d rows (%s → %s) | Test: %d rows (%s → %s)",
        len(df_train),
        df_train.index.min().date(),
        df_train.index.max().date(),
        len(df_test),
        df_test.index.min().date(),
        df_test.index.max().date(),
    )

    train_array = df_train[feature_columns].values
    # For sequences that span the train/test boundary we include the tail of
    # the training set in the test array.
    combined_array = df[feature_columns].values
    test_start_pos = len(df_train)

    # 3. Scaling
    if is_training:
        scaler = fit_and_save_scaler(train_array, scaler_path)
        # Persist feature column list
        path = Path(feature_columns_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(feature_columns, f)
        logger.info("Feature columns saved to %s", feature_columns_path)
    else:
        scaler = load_scaler(scaler_path)
        with open(feature_columns_path) as f:
            feature_columns = json.load(f)
        target_col_index = feature_columns.index(target_column)

    scaled_full = scaler.transform(combined_array)

    # 4. Build sequences
    X_all, y_all = build_sequences(scaled_full, sequence_length, target_col_index)

    # Sequences whose last step falls before test_start_pos belong to train
    # (each sequence[i] corresponds to row i + sequence_length in the full df)
    split_idx = test_start_pos - sequence_length
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]

    logger.info("Sequences — Train: %s | Test: %s", X_train.shape, X_test.shape)

    # 5. Persist arrays
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    np.save(proc_path / "X_train.npy", X_train)
    np.save(proc_path / "y_train.npy", y_train)
    np.save(proc_path / "X_test.npy", X_test)
    np.save(proc_path / "y_test.npy", y_test)
    logger.info("Processed arrays saved to %s", processed_dir)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "target_col_index": target_col_index,
        "df_full": df,
        "test_start_index": test_start_pos,
    }


def prepare_inference_sequence(
    recent_df: pd.DataFrame,
    config: dict,
    artifacts_dir: str,
) -> np.ndarray:
    """Prepare a single inference sequence from the most recent market data.

    Applies feature engineering and scaling using persisted artefacts,
    then returns the last `sequence_length` rows as an LSTM-ready array.

    Args:
        recent_df: DataFrame with at least `sequence_length` rows of OHLCV data.
        config: Full project config dict.
        artifacts_dir: Directory containing scaler and feature_columns artefacts.

    Returns:
        Array of shape (1, sequence_length, n_features) ready for model.predict().
    """
    feat_cfg = config["features"]
    ind_cfg = feat_cfg["technical_indicators"]
    sequence_length = feat_cfg["sequence_length"]
    scaler_path = str(Path(artifacts_dir) / config["paths"]["scaler_filename"])
    feature_columns_path = str(
        Path(artifacts_dir) / config["paths"]["feature_columns_filename"]
    )

    df = add_technical_indicators(recent_df, ind_cfg)

    with open(feature_columns_path) as f:
        feature_columns = json.load(f)

    scaler = load_scaler(scaler_path)

    if len(df) < sequence_length:
        raise ValueError(
            f"Not enough rows after feature engineering. "
            f"Need at least {sequence_length}, got {len(df)}."
        )

    window = df[feature_columns].tail(sequence_length).values
    scaled = scaler.transform(window)
    return scaled.reshape(1, sequence_length, len(feature_columns))
