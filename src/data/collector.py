"""
Data collection module.

Downloads historical OHLCV data for a given ticker from Yahoo Finance via
yfinance and persists it as a CSV in the raw data directory.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'PETR4.SA').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format. Defaults to today.
        output_dir: Directory where the CSV will be saved.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date.

    Raises:
        ValueError: If the downloaded DataFrame is empty (ticker not found or
                    no data for the given period).
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    logger.info("Downloading %s from %s to %s", ticker, start_date, end_date)

    raw_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if raw_df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
            "Check that the ticker is valid and the date range is correct."
        )

    # Flatten MultiIndex columns produced by yfinance when a single ticker is passed
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.droplevel(1)

    # Keep only the standard OHLCV columns
    columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]
    df = raw_df[columns_to_keep].copy()
    df.index.name = "Date"

    # Persist to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{ticker.replace('.', '_')}_raw.csv"
    df.to_csv(csv_path)

    logger.info(
        "Saved %d rows to %s (%.1f MB)",
        len(df),
        csv_path,
        csv_path.stat().st_size / 1024 / 1024,
    )

    return df


def load_raw_data(ticker: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load previously downloaded raw data from disk.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'PETR4.SA').
        raw_dir: Directory containing raw CSV files.

    Returns:
        DataFrame with Date as index.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(raw_dir) / f"{ticker.replace('.', '_')}_raw.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {csv_path}. "
            "Run the collect_data script first."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df
