"""
FastAPI application entry point.

Exposes the following endpoints:
  POST /predict      — Predict next-day PETR4 closing price.
  GET  /health       — Liveness/readiness check.
  GET  /model/info   — Metadata about the loaded model.
  GET  /metrics      — Prometheus metrics (text exposition format).

The model and artefacts are loaded once at startup and stored in the
application state to avoid repeated disk I/O on every request.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.config import get_settings
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictRequest,
)
from src.data.collector import download_stock_data
from src.data.preprocessor import prepare_inference_sequence
from src.models.lstm_model import load_model, predict_next_close
from src.monitoring.logger import (
    DATA_FETCH_LATENCY,
    PREDICTION_LATENCY,
    check_drift,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artefacts at startup; release on shutdown."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting PETR4 Price Predictor API v%s", settings.api_version)

    # Load config.yaml for full pipeline config (feature engineering, etc.)
    with open("config.yaml") as f:
        full_config = yaml.safe_load(f)

    # Load model
    try:
        model = load_model(settings.model_path)
        logger.info("Model loaded successfully from %s", settings.model_path)
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
        model = None

    # Load feature columns
    feature_columns = []
    try:
        with open(settings.feature_columns_path) as f:
            feature_columns = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "feature_columns.json not found at %s", settings.feature_columns_path
        )

    target_col_index = (
        feature_columns.index("Close") if "Close" in feature_columns else 0
    )

    # Store everything in app state
    app.state.model = model
    app.state.full_config = full_config
    app.state.feature_columns = feature_columns
    app.state.target_col_index = target_col_index
    app.state.model_loaded = model is not None

    yield  # Application runs here

    logger.info("Shutting down PETR4 Price Predictor API.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request_body: PredictRequest, http_request: Request
) -> PredictionResponse:
    """Predict the next-day closing price for PETR4.

    When `use_latest_market_data=True` (default), the API fetches the most
    recent OHLCV data from Yahoo Finance. Alternatively, pass `custom_data`
    with at least `sequence_length` (default 60) rows.

    Returns the predicted price, last known close, and drift status.
    """
    wall_start = time.perf_counter()

    if not getattr(http_request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first.",
        )

    model = getattr(http_request.app.state, "model", None)
    full_config = getattr(http_request.app.state, "full_config", {})
    feature_columns = getattr(http_request.app.state, "feature_columns", [])
    target_col_index = getattr(http_request.app.state, "target_col_index", 0)
    n_features = len(feature_columns)
    sequence_length = settings.sequence_length

    # ----- 1. Fetch or parse input data -------------------------------------
    fetch_start = time.perf_counter()

    if request_body.use_latest_market_data:
        # Fetch enough rows to cover warm-up (max indicator window = 50 SMA) + sequence
        lookback_days = sequence_length + 100
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=lookback_days * 2)).strftime(
            "%Y-%m-%d"
        )
        try:
            recent_df = download_stock_data(
                ticker=settings.ticker,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            logger.error("Failed to fetch market data: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"Could not fetch market data from Yahoo Finance: {exc}",
            )
    else:
        if not request_body.custom_data:
            raise HTTPException(
                status_code=422,
                detail="custom_data is required when use_latest_market_data=False.",
            )
        recent_df = pd.DataFrame(request_body.custom_data)
        recent_df["Date"] = pd.to_datetime(recent_df["Date"])
        recent_df = recent_df.set_index("Date").sort_index()

    DATA_FETCH_LATENCY.observe(time.perf_counter() - fetch_start)

    last_known_close = float(recent_df["Close"].iloc[-1])

    # ----- 2. Preprocess + build inference sequence -------------------------
    try:
        sequence = prepare_inference_sequence(
            recent_df=recent_df,
            config=full_config,
            artifacts_dir=settings.artifacts_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))

    # ----- 3. Predict -------------------------------------------------------
    from src.data.preprocessor import load_scaler

    scaler = load_scaler(settings.scaler_path)
    predicted_price = predict_next_close(
        model=model,
        sequence=sequence,
        scaler=scaler,
        target_col_index=target_col_index,
        n_features=n_features,
    )

    # ----- 4. Drift detection -----------------------------------------------
    drift_detected = check_drift(
        predicted_price=predicted_price,
        last_known_close=last_known_close,
        threshold_pct=settings.drift_threshold_pct,
    )

    # ----- 5. Persist prediction to disk ------------------------------------
    pred_dir = Path(settings.predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_date = (datetime.today() + timedelta(days=1)).date()
    if datetime.today().weekday() == 4:  # Friday → Monday
        prediction_date = (datetime.today() + timedelta(days=3)).date()

    pred_record = {
        "timestamp": datetime.now().isoformat(),
        "ticker": settings.ticker,
        "prediction_date": str(prediction_date),
        "last_known_close": last_known_close,
        "predicted_close": predicted_price,
        "drift_detected": drift_detected,
    }

    import json as _json

    pred_file = pred_dir / "predictions_log.jsonl"
    with open(pred_file, "a") as f:
        f.write(_json.dumps(pred_record) + "\n")

    latency_ms = (time.perf_counter() - wall_start) * 1000
    PREDICTION_LATENCY.observe(latency_ms / 1000)

    logger.info(
        "Prediction served: ticker=%s | predicted=%.4f | last_close=%.4f | "
        "drift=%s | latency=%.1f ms",
        settings.ticker,
        predicted_price,
        last_known_close,
        drift_detected,
        latency_ms,
    )

    return PredictionResponse(
        ticker=settings.ticker,
        predicted_close=round(predicted_price, 4),
        last_known_close=round(last_known_close, 4),
        prediction_date=prediction_date,
        model_version=settings.api_version,
        drift_detected=drift_detected,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(request: Request) -> HealthResponse:
    """Liveness and readiness check.

    Returns 200 OK when the model is loaded and ready to serve predictions.
    Returns 200 with status='degraded' if the model failed to load.
    """
    model_loaded = getattr(request.app.state, "model_loaded", False)
    status_code = "ok" if model_loaded else "degraded"

    return HealthResponse(
        status=status_code,
        model_loaded=model_loaded,
        timestamp=datetime.now(UTC),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    feature_columns = getattr(request.app.state, "feature_columns", [])

    return ModelInfoResponse(
        ticker=settings.ticker,
        architecture="LSTM",
        sequence_length=settings.sequence_length,
        n_features=len(feature_columns),
        feature_columns=feature_columns,
        model_version=settings.api_version,
        model_path=settings.model_path,
    )


@app.get("/metrics", tags=["Operations"])
async def metrics() -> Response:
    """Expose Prometheus metrics in the standard text exposition format.

    Prometheus should scrape this endpoint at regular intervals.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
