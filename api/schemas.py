"""
Pydantic request/response schemas for the PETR4 Predictor API.

Strict typing ensures that FastAPI auto-generates accurate OpenAPI docs
and that invalid payloads are rejected before they reach the model.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    The caller may either provide raw OHLCV data directly (for testing)
    or let the API fetch the latest market data automatically.
    """

    use_latest_market_data: bool = Field(
        default=True,
        description=(
            "When True (default), the API fetches the most recent PETR4 data "
            "from Yahoo Finance automatically. Set to False to provide custom data."
        ),
    )
    custom_data: list[dict] | None = Field(
        default=None,
        description=(
            "List of OHLCV dicts when use_latest_market_data=False. "
            "Each dict must have keys: Date (YYYY-MM-DD), Open, High, Low, Close, Volume. "
            "Must contain at least `sequence_length` rows (default 60)."
        ),
    )

    @field_validator("custom_data")
    @classmethod
    def validate_custom_data(cls, v):
        """Ensure each row in custom_data has the required OHLCV keys."""
        if v is None:
            return v
        required_keys = {"Date", "Open", "High", "Low", "Close", "Volume"}
        for i, row in enumerate(v):
            missing = required_keys - set(row.keys())
            if missing:
                raise ValueError(f"Row {i} is missing required keys: {missing}")
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Response body for POST /predict."""

    model_config = {"protected_namespaces": ()}

    ticker: str = Field(description="Ticker symbol predicted (e.g. 'PETR4.SA').")
    predicted_close: float = Field(
        description="Predicted next-day closing price in BRL (R$)."
    )
    last_known_close: float = Field(
        description="Most recent actual closing price used as input context."
    )
    prediction_date: date = Field(
        description="The trading day for which the prediction applies."
    )
    model_version: str = Field(description="Identifier of the model used.")
    drift_detected: bool = Field(
        default=False,
        description="True if the prediction deviates by more than the configured threshold.",
    )
    latency_ms: float = Field(
        description="End-to-end prediction latency in milliseconds."
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = {"protected_namespaces": ()}

    status: Literal["ok", "degraded", "error"] = "ok"
    model_loaded: bool
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Response body for GET /model/info."""

    model_config = {"protected_namespaces": ()}

    ticker: str
    architecture: str
    sequence_length: int
    n_features: int
    feature_columns: list[str]
    model_version: str
    model_path: str
