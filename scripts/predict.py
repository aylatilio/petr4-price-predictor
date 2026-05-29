"""
CLI script: run a single next-day prediction using the trained model.

Fetches the latest PETR4 market data, builds the inference sequence,
and prints the predicted closing price.

Usage:
    python scripts/predict.py
    python scripts/predict.py --days-back 200
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collector import download_stock_data
from src.data.preprocessor import load_scaler, prepare_inference_sequence
from src.models.lstm_model import load_model, predict_next_close
from src.monitoring.logger import setup_logging

import json


@click.command()
@click.option(
    "--days-back",
    default=200,
    show_default=True,
    help="How many calendar days back to fetch data (must cover warm-up + sequence_length).",
)
@click.option(
    "--use-cached",
    is_flag=True,
    default=False,
    help="Load data from the local raw CSV instead of fetching from Yahoo Finance.",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Path to the project configuration file.",
)
def main(days_back: int, use_cached: bool, config_path: str) -> None:
    """Fetch latest PETR4 data and predict next-day closing price."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level=config["monitoring"]["log_level"])
    logger = logging.getLogger(__name__)

    ticker = config["data"]["ticker"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    model_path = str(
        Path(config["paths"]["model_dir"]) / config["paths"]["model_filename"]
    )
    scaler_path = str(
        Path(artifacts_dir) / config["paths"]["scaler_filename"]
    )
    feature_columns_path = str(
        Path(artifacts_dir) / config["paths"]["feature_columns_filename"]
    )

    # Load data: from disk cache or live download
    if use_cached:
        import pandas as pd
        csv_path = Path(config["data"]["raw_dir"]) / f"{ticker.replace('.', '_')}_raw.csv"
        logger.info("Loading cached data from %s", csv_path)
        recent_df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        logger.info("Fetching %s from %s to %s", ticker, start_date, end_date)
        recent_df = download_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)

    last_known_close = float(recent_df["Close"].iloc[-1])

    # Load artefacts and model
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    with open(feature_columns_path) as f:
        feature_columns = json.load(f)
    target_col_index = feature_columns.index("Close")
    n_features = len(feature_columns)

    # Build inference sequence
    sequence = prepare_inference_sequence(
        recent_df=recent_df,
        config=config,
        artifacts_dir=artifacts_dir,
    )

    # Predict
    predicted_price = predict_next_close(
        model=model,
        sequence=sequence,
        scaler=scaler,
        target_col_index=target_col_index,
        n_features=n_features,
    )

    # Determine the prediction date (next business day)
    today_weekday = datetime.today().weekday()
    delta_days = 3 if today_weekday == 4 else 1  # Friday → Monday
    prediction_date = (datetime.today() + timedelta(days=delta_days)).date()

    click.echo("\n" + "=" * 45)
    click.echo(f"  Ticker           : {ticker}")
    click.echo(f"  Last known close : R$ {last_known_close:.2f}")
    click.echo(f"  Predicted close  : R$ {predicted_price:.2f}")
    click.echo(f"  Prediction date  : {prediction_date}")
    click.echo("=" * 45 + "\n")


if __name__ == "__main__":
    main()
