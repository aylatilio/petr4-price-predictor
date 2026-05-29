"""
Monitoring and logging module.

Sets up structured JSON logging and exposes Prometheus metrics for:
  - Total predictions served
  - Prediction latency (histogram)
  - Model error rate (MAPE gauge)
  - Drift alerts (counter)

The module is imported once at API startup; all other modules use the
standard `logging` library and inherit this configuration.
"""

import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, Summary

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure root logger with both console and rotating file handlers.

    Uses pythonjsonlogger for structured JSON output in file logs,
    making them easier to parse with log aggregation tools.

    Args:
        log_dir: Directory where log files will be written.
        log_level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # File handler (JSON-structured)
    log_file = Path(log_dir) / "api.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)

    try:
        from pythonjsonlogger import jsonlogger  # type: ignore

        json_fmt = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        file_handler.setFormatter(json_fmt)
    except ImportError:
        # Fallback to plain text if pythonjsonlogger is not installed
        file_handler.setFormatter(console_fmt)

    # Avoid duplicate handlers if setup_logging is called more than once
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "petr4_predictions_total",
    "Total number of prediction requests served.",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "petr4_prediction_errors_total",
    "Total number of failed prediction requests.",
)

PREDICTION_LATENCY = Histogram(
    "petr4_prediction_latency_seconds",
    "End-to-end latency of the /predict endpoint.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

MODEL_MAPE_GAUGE = Gauge(
    "petr4_model_mape",
    "Last recorded MAPE on recent predictions (percentage).",
)

DRIFT_ALERTS_TOTAL = Counter(
    "petr4_drift_alerts_total",
    "Number of times a prediction exceeded the drift threshold.",
)

DATA_FETCH_LATENCY = Histogram(
    "petr4_data_fetch_latency_seconds",
    "Time to fetch and preprocess market data for inference.",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0],
)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator that measures and records prediction endpoint latency.

    Args:
        func: Async or sync function to wrap.

    Returns:
        Wrapped function with Prometheus timing instrumentation.
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            PREDICTIONS_TOTAL.inc()
            return result
        except Exception:
            PREDICTION_ERRORS_TOTAL.inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            PREDICTION_LATENCY.observe(elapsed)

    return async_wrapper


# ---------------------------------------------------------------------------
# Drift detection helper
# ---------------------------------------------------------------------------


def check_drift(predicted_price: float, last_known_close: float, threshold_pct: float) -> bool:
    """Check whether a prediction deviates excessively from the last known close.

    Args:
        predicted_price: Model's predicted closing price.
        last_known_close: Most recent real closing price.
        threshold_pct: Maximum acceptable percentage deviation before alert.

    Returns:
        True if drift is detected (alert triggered), False otherwise.
    """
    if last_known_close == 0:
        return False

    deviation_pct = abs(predicted_price - last_known_close) / last_known_close * 100

    if deviation_pct > threshold_pct:
        DRIFT_ALERTS_TOTAL.inc()
        logging.getLogger(__name__).warning(
            "Drift detected: predicted=%.4f, last_close=%.4f, deviation=%.2f%% (threshold=%.2f%%)",
            predicted_price,
            last_known_close,
            deviation_pct,
            threshold_pct,
        )
        return True

    return False
