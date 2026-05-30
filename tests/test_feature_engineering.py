"""
Unit tests for the feature engineering module.

Validates indicator computations, NaN drop behaviour, and
that the output shape matches expectations.
"""

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    add_technical_indicators,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    compute_sma,
)


def _make_price_series(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series for testing."""
    rng = np.random.default_rng(seed)
    prices = 30.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.Series(prices, name="Close")


def _make_ohlcv_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    closes = 30.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": closes * rng.uniform(0.98, 1.00, n),
            "High": closes * rng.uniform(1.00, 1.02, n),
            "Low": closes * rng.uniform(0.97, 1.00, n),
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


class TestComputeSma:
    def test_output_length_matches_input(self):
        s = _make_price_series(50)
        sma = compute_sma(s, window=10)
        assert len(sma) == len(s)

    def test_first_valid_index(self):
        s = _make_price_series(30)
        sma = compute_sma(s, window=5)
        # First 4 values should be NaN
        assert sma.iloc[:4].isna().all()
        assert pd.notna(sma.iloc[4])

    def test_constant_series_sma_equals_constant(self):
        s = pd.Series([10.0] * 20)
        sma = compute_sma(s, window=5)
        assert sma.dropna().eq(10.0).all()


class TestComputeRsi:
    def test_rsi_bounded_between_0_and_100(self):
        s = _make_price_series(100)
        rsi = compute_rsi(s, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_has_correct_nan_count(self):
        s = _make_price_series(50)
        rsi = compute_rsi(s, period=14)
        # First 14 values should be NaN due to min_periods
        assert rsi.iloc[:14].isna().all()


class TestComputeMacd:
    def test_returns_three_series(self):
        s = _make_price_series(100)
        macd, signal, hist = compute_macd(s)
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_histogram_equals_macd_minus_signal(self):
        s = _make_price_series(100)
        macd, signal, hist = compute_macd(s)
        pd.testing.assert_series_equal(hist, macd - signal, check_names=False)


class TestComputeBollingerBands:
    def test_upper_always_above_lower(self):
        s = _make_price_series(100)
        middle, upper, lower = compute_bollinger_bands(s, window=20)
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()

    def test_middle_band_is_sma(self):
        s = _make_price_series(100)
        middle, _, _ = compute_bollinger_bands(s, window=20)
        sma = compute_sma(s, window=20)
        pd.testing.assert_series_equal(middle, sma, check_names=False)


class TestAddTechnicalIndicators:
    def get_default_config(self) -> dict:
        return {
            "moving_averages": [7, 14],
            "rsi_period": 14,
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"window": 20, "num_std": 2},
        }

    def test_output_has_no_nan(self):
        df = _make_ohlcv_df(200)
        result = add_technical_indicators(df, self.get_default_config())
        assert not result.isnull().any().any()

    def test_output_has_more_columns_than_input(self):
        df = _make_ohlcv_df(200)
        result = add_technical_indicators(df, self.get_default_config())
        assert result.shape[1] > df.shape[1]

    def test_close_column_preserved(self):
        df = _make_ohlcv_df(200)
        result = add_technical_indicators(df, self.get_default_config())
        assert "Close" in result.columns

    def test_expected_indicator_columns_present(self):
        df = _make_ohlcv_df(200)
        result = add_technical_indicators(df, self.get_default_config())
        expected = ["SMA_7", "SMA_14", "RSI", "MACD", "BB_upper", "BB_lower"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"
