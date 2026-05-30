"""
Unit tests for the evaluation metrics module.

Tests cover expected values, edge cases (zeros, perfect predictions),
and type/shape contracts.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
)


class TestComputeRmse:
    def test_perfect_prediction_returns_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        # Manually: errors = [0.5, 0.5, 0, 1] → mse = (0.25+0.25+0+1)/4 = 0.375
        assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.375), rel=1e-5)

    def test_returns_float(self):
        assert isinstance(compute_rmse(np.array([1.0]), np.array([2.0])), float)


class TestComputeMae:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # |1-2|+|2-2|+|3-2| / 3 = (1+0+1)/3 = 0.667
        assert compute_mae(y_true, y_pred) == pytest.approx(2 / 3, rel=1e-5)


class TestComputeMape:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_mape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_percentage(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |100-110|/100 + |200-180|/200 / 2 = (0.1 + 0.1) / 2 = 0.1 → 10%
        assert compute_mape(y_true, y_pred) == pytest.approx(10.0, rel=1e-3)

    def test_does_not_divide_by_zero(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 1.0])
        result = compute_mape(y_true, y_pred)
        assert np.isfinite(result)


class TestComputeR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r2(y, y) == pytest.approx(1.0)

    def test_constant_target_returns_zero(self):
        # ss_tot = 0 → R² forced to 0
        y_const = np.array([5.0, 5.0, 5.0])
        assert compute_r2(y_const, y_const) == 0.0

    def test_worse_than_baseline_is_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        # Predictions are way off
        y_pred = np.array([100.0, 200.0, 300.0])
        assert compute_r2(y_true, y_pred) < 0


class TestComputeAllMetrics:
    def test_returns_expected_keys(self):
        y = np.array([10.0, 20.0, 30.0])
        metrics = compute_all_metrics(y, y)
        assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2"}

    def test_perfect_prediction_values(self):
        y = np.array([15.0, 25.0, 35.0])
        metrics = compute_all_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
