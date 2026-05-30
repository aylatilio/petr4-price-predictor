"""
CLI script: run walk-forward backtesting and generate a results report.

Saves a CSV of actuals vs predictions and prints aggregate metrics.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --output data/predictions/backtest_results.csv
"""

import logging
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collector import load_raw_data
from src.data.feature_engineering import add_technical_indicators
from src.evaluation.backtester import run_walk_forward_validation
from src.monitoring.logger import setup_logging


@click.command()
@click.option(
    "--output",
    default="data/predictions/backtest_results.csv",
    show_default=True,
    help="Path to save the backtest results CSV.",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Path to the project configuration file.",
)
def main(output: str, config_path: str) -> None:
    """Run walk-forward validation and save results."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level=config["monitoring"]["log_level"])
    logger = logging.getLogger(__name__)

    ticker = config["data"]["ticker"]
    raw_dir = config["data"]["raw_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]

    # Load raw data and compute features
    raw_df = load_raw_data(ticker=ticker, raw_dir=raw_dir)
    df_full = add_technical_indicators(
        raw_df, config["features"]["technical_indicators"]
    )

    feature_columns = [c for c in df_full.columns if df_full[c].dtype in [float]]
    target_col_index = feature_columns.index(config["features"]["target_column"])

    logger.info("Starting walk-forward validation on %d rows", len(df_full))

    results = run_walk_forward_validation(
        df_full=df_full,
        feature_columns=feature_columns,
        target_col_index=target_col_index,
        config=config,
        artifacts_dir=artifacts_dir,
    )

    # Save results
    results_df = pd.DataFrame(
        {
            "date": results["dates"],
            "actual_close": results["actuals"],
            "predicted_close": results["predictions"],
        }
    )
    results_df["abs_error"] = abs(
        results_df["actual_close"] - results_df["predicted_close"]
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Backtest results saved to %s", output_path)

    metrics = results["metrics"]
    click.echo("\n=== Walk-forward Validation Metrics ===")
    click.echo(f"  RMSE : {metrics['rmse']:.4f} BRL")
    click.echo(f"  MAE  : {metrics['mae']:.4f} BRL")
    click.echo(f"  MAPE : {metrics['mape']:.2f}%")
    click.echo(f"  R²   : {metrics['r2']:.4f}")
    click.echo(f"  Folds: {len(results_df)} predictions")
    click.echo("=======================================\n")


if __name__ == "__main__":
    main()
