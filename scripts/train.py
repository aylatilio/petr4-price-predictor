"""
CLI script: run the full training pipeline.

Steps:
  1. Load raw data from disk (must run collect_data.py first).
  2. Run preprocessing pipeline (feature engineering + scaling + sequences).
  3. Train the LSTM model.
  4. Evaluate on the held-out test set and print metrics.

Usage:
    python scripts/train.py
    python scripts/train.py --config config.yaml
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collector import load_raw_data
from src.data.preprocessor import preprocess_pipeline
from src.evaluation.metrics import compute_all_metrics
from src.monitoring.logger import setup_logging
from src.training.trainer import train


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Path to the project configuration file.",
)
def main(config_path: str) -> None:
    """Preprocess data, train the LSTM model, and evaluate on the test set."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level=config["monitoring"]["log_level"])
    logger = logging.getLogger(__name__)

    ticker = config["data"]["ticker"]
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]

    # --- 1. Load raw data ---------------------------------------------------
    logger.info("Loading raw data for %s", ticker)
    raw_df = load_raw_data(ticker=ticker, raw_dir=raw_dir)

    # --- 2. Preprocessing ---------------------------------------------------
    logger.info("Running preprocessing pipeline")
    result = preprocess_pipeline(
        raw_df=raw_df,
        config=config,
        artifacts_dir=artifacts_dir,
        processed_dir=processed_dir,
        is_training=True,
    )

    X_train = result["X_train"]
    y_train = result["y_train"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    scaler = result["scaler"]
    feature_columns = result["feature_columns"]
    target_col_index = result["target_col_index"]
    n_features = len(feature_columns)

    # --- 3. Train -----------------------------------------------------------
    logger.info("Training model. X_train shape: %s", X_train.shape)
    model, history = train(X_train=X_train, y_train=y_train, config=config)

    # --- 4. Evaluate on test set -------------------------------------------
    logger.info("Evaluating on test set. X_test shape: %s", X_test.shape)

    scaled_preds = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform predictions and actuals to real BRL prices
    dummy = np.zeros((len(scaled_preds), n_features))
    dummy[:, target_col_index] = scaled_preds
    y_pred_brl = scaler.inverse_transform(dummy)[:, target_col_index]

    dummy_actual = np.zeros((len(y_test), n_features))
    dummy_actual[:, target_col_index] = y_test
    y_true_brl = scaler.inverse_transform(dummy_actual)[:, target_col_index]

    metrics = compute_all_metrics(y_true_brl, y_pred_brl, label="Test")

    click.echo("\n=== Test Set Metrics ===")
    click.echo(f"  RMSE : {metrics['rmse']:.4f} BRL")
    click.echo(f"  MAE  : {metrics['mae']:.4f} BRL")
    click.echo(f"  MAPE : {metrics['mape']:.2f}%")
    click.echo(f"  R²   : {metrics['r2']:.4f}")
    click.echo("========================\n")

    logger.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
