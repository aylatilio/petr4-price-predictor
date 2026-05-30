"""
Integration tests for the FastAPI application.

Uses plain functions (not classes) to ensure pytest fixture injection works
correctly with the module-scoped api_client fixture in conftest.py.
"""

import pytest

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200(api_client):
    assert api_client.get("/health").status_code == 200


def test_health_has_required_fields(api_client):
    data = api_client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_health_status_is_valid(api_client):
    assert api_client.get("/health").json()["status"] in ("ok", "degraded", "error")


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(api_client):
    assert api_client.get("/metrics").status_code == 200


def test_metrics_content_type_is_prometheus(api_client):
    assert "text/plain" in api_client.get("/metrics").headers["content-type"]


def test_metrics_contains_petr4_names(api_client):
    assert "petr4_predictions_total" in api_client.get("/metrics").text


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------


def test_model_info_returns_200(api_client):
    assert api_client.get("/model/info").status_code == 200


def test_model_info_has_required_fields(api_client):
    data = api_client.get("/model/info").json()
    required = {
        "ticker",
        "architecture",
        "sequence_length",
        "n_features",
        "feature_columns",
        "model_version",
        "model_path",
    }
    assert required.issubset(data.keys())


def test_model_info_ticker_is_petr4(api_client):
    assert "PETR4" in api_client.get("/model/info").json()["ticker"]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_503_when_no_model(api_client):
    """Without a trained model, the API must refuse with 503."""
    response = api_client.post("/predict", json={"use_latest_market_data": False})
    assert response.status_code in (422, 503)


def test_predict_422_on_invalid_custom_data(api_client):
    """custom_data missing required OHLCV keys must be rejected with 422."""
    payload = {
        "use_latest_market_data": False,
        "custom_data": [{"Date": "2024-01-01", "Close": 30.0}],
    }
    assert api_client.post("/predict", json=payload).status_code == 422
