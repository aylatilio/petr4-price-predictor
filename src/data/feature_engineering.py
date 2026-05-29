"""
Feature engineering module.

Computes technical indicators (SMA, RSI, MACD, Bollinger Bands) and
appends them as additional columns to the raw OHLCV DataFrame.

All indicator parameters are driven by config.yaml so that changes are
centralised and reproducible.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Return Simple Moving Average for a price series.

    Args:
        series: Numeric price series (e.g. Close prices).
        window: Look-back window in days.

    Returns:
        SMA series with the same index as the input.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return Relative Strength Index (RSI) for a price series.

    Uses Wilder's smoothing (exponential weighted moving average with
    alpha = 1 / period), which matches most charting platforms.

    Args:
        series: Closing price series.
        period: Look-back period. Default is 14.

    Returns:
        RSI series bounded in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's EMA (equivalent to ewm with com=period-1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return MACD line, signal line, and histogram.

    Args:
        series: Closing price series.
        fast: Fast EMA window. Default is 12.
        slow: Slow EMA window. Default is 26.
        signal: Signal line EMA window. Default is 9.

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series.
    """
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger Bands (middle, upper, lower).

    Args:
        series: Closing price series.
        window: Rolling window for mean/std. Default is 20.
        num_std: Number of standard deviations for band width. Default is 2.

    Returns:
        Tuple of (middle_band, upper_band, lower_band) Series.
    """
    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def add_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute and append all configured technical indicators to the DataFrame.

    Args:
        df: Raw OHLCV DataFrame with at least a 'Close' column.
        config: The 'features.technical_indicators' section from config.yaml.

    Returns:
        DataFrame with original columns plus computed indicator columns.
        Rows where any indicator is NaN (warm-up period) are dropped.
    """
    df = df.copy()
    close = df["Close"]

    # --- Simple Moving Averages ---
    for window in config.get("moving_averages", []):
        col_name = f"SMA_{window}"
        df[col_name] = compute_sma(close, window)
        logger.debug("Computed %s", col_name)

    # --- RSI ---
    rsi_period = config.get("rsi_period", 14)
    df["RSI"] = compute_rsi(close, period=rsi_period)
    logger.debug("Computed RSI(%d)", rsi_period)

    # --- MACD ---
    macd_cfg = config.get("macd", {})
    macd_line, signal_line, histogram = compute_macd(
        close,
        fast=macd_cfg.get("fast", 12),
        slow=macd_cfg.get("slow", 26),
        signal=macd_cfg.get("signal", 9),
    )
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = histogram
    logger.debug("Computed MACD")

    # --- Bollinger Bands ---
    bb_cfg = config.get("bollinger_bands", {})
    bb_middle, bb_upper, bb_lower = compute_bollinger_bands(
        close,
        window=bb_cfg.get("window", 20),
        num_std=bb_cfg.get("num_std", 2),
    )
    df["BB_middle"] = bb_middle
    df["BB_upper"] = bb_upper
    df["BB_lower"] = bb_lower
    df["BB_width"] = bb_upper - bb_lower  # band width as a volatility proxy
    logger.debug("Computed Bollinger Bands")

    # Drop warm-up rows where any indicator has NaN
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    logger.info(
        "Technical indicators added. Dropped %d warm-up rows, %d rows remaining.",
        rows_dropped,
        len(df),
    )

    return df
