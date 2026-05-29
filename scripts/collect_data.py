"""
CLI script: collect historical PETR4 data from Yahoo Finance.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --ticker PETR4.SA --start 2014-01-01
"""

import logging
import sys
from pathlib import Path

import click
import yaml

# Ensure the project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collector import download_stock_data
from src.monitoring.logger import setup_logging


@click.command()
@click.option(
    "--ticker",
    default=None,
    help="Yahoo Finance ticker symbol. Defaults to value in config.yaml.",
)
@click.option(
    "--start",
    default=None,
    help="Start date in YYYY-MM-DD format. Defaults to value in config.yaml.",
)
@click.option(
    "--end",
    default=None,
    help="End date in YYYY-MM-DD format. Defaults to today.",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="Path to the project configuration file.",
)
def main(ticker: str | None, start: str | None, end: str | None, config_path: str) -> None:
    """Download historical OHLCV data and save it to data/raw/."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level=config["monitoring"]["log_level"])
    logger = logging.getLogger(__name__)

    resolved_ticker = ticker or config["data"]["ticker"]
    resolved_start = start or config["data"]["start_date"]
    resolved_end = end or config["data"].get("end_date")
    raw_dir = config["data"]["raw_dir"]

    logger.info(
        "Collecting data — ticker=%s | start=%s | end=%s",
        resolved_ticker,
        resolved_start,
        resolved_end or "today",
    )

    df = download_stock_data(
        ticker=resolved_ticker,
        start_date=resolved_start,
        end_date=resolved_end,
        output_dir=raw_dir,
    )

    logger.info("Collection complete. Shape: %s", df.shape)
    click.echo(df.tail())


if __name__ == "__main__":
    main()
