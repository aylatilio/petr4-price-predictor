# PETR4 Price Predictor

LSTM-based model to predict the next-day closing price of **PETR4 (Petrobras)**, built as the final project for the FIAP Machine Learning Engineering post-graduation programme (Phase 5).

---

## Overview

| Item | Decision |
|---|---|
| **Ticker** | PETR4.SA (Petrobras ON, B3) |
| **Algorithm** | Stacked LSTM (2 layers) |
| **Input** | Last 60 days of OHLCV + technical indicators |
| **Output** | Predicted next-day closing price (BRL) |
| **Evaluation** | RMSE, MAE, MAPE, R² + walk-forward validation |
| **Explainability** | SHAP global importance |
| **Baseline** | Prophet |
| **API** | FastAPI + Uvicorn |
| **Monitoring** | Prometheus + Grafana |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |

---

## Project Structure

```
petr4-price-predictor/
├── data/
│   ├── raw/                 # Raw OHLCV CSV from Yahoo Finance
│   ├── processed/           # Scaled numpy arrays (X_train, X_test …)
│   └── predictions/         # Prediction log (JSONL) + backtest CSV
├── models/
│   ├── trained/             # Trained model (.keras)
│   └── artifacts/           # Scaler (.pkl), feature columns (.json)
├── src/
│   ├── data/                # collector.py, preprocessor.py, feature_engineering.py
│   ├── models/              # lstm_model.py
│   ├── training/            # trainer.py
│   ├── evaluation/          # metrics.py, backtester.py
│   └── monitoring/          # logger.py (Prometheus metrics + structured logging)
├── api/
│   ├── app.py               # FastAPI application
│   ├── config.py            # Settings (pydantic-settings + config.yaml)
│   ├── schemas.py           # Pydantic request/response models
│   └── requirements-api.txt # Slim dependency list for Docker
├── scripts/
│   ├── collect_data.py      # Download historical data
│   ├── train.py             # Full training pipeline
│   ├── predict.py           # Single inference from CLI
│   └── backtest.py          # Walk-forward validation
├── tests/                   # pytest unit + integration tests
├── docs/                    # prometheus.yml, Grafana provisioning
├── notebooks/               # Exploratory analysis
├── .github/workflows/ci.yml # GitHub Actions CI pipeline
├── config.yaml              # Central configuration
├── Dockerfile               # Multi-stage API image
└── docker-compose.yml       # API + Prometheus + Grafana stack
```

---

## Quickstart (WSL / Ubuntu)

### 1. Clone and set up the environment

```bash
git clone https://github.com/aylatilio/petr4-price-predictor.git
cd petr4-price-predictor

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect historical data

```bash
python scripts/collect_data.py
```

Downloads PETR4.SA from Yahoo Finance (2014–today) to `data/raw/`.

### 3. Train the model

```bash
python scripts/train.py
```

Preprocesses data, trains the LSTM, and prints test-set metrics. Saves model to `models/trained/lstm_petr4.keras`.

### 4. Run a single prediction

```bash
python scripts/predict.py
```

If you encouter YFRateLimitError('Too Many Requests. Rate limited. Try after a while.'), try using your own data/raw/ csv.

```bash
python scripts/predict.py --use-cached
```

Fetches latest market data and prints the next-day predicted close.

### 5. Run walk-forward backtesting

```bash
python scripts/backtest.py
```

Saves `data/predictions/backtest_results.csv` with actuals vs predictions.

---

## Running with Docker

### Option A — single container (API only)

```bash
# Build
docker build -t petr4-price-predictor:1.0.0 .

# Run (mount trained model and output dirs)
docker run -d \
  --name petr4_api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/predictions:/app/data/predictions \
  -v $(pwd)/logs:/app/logs \
  petr4-price-predictor:1.0.0

# Stop and remove
docker stop petr4_api && docker rm petr4_api
```

### Option B — full stack (API + Prometheus + Grafana)

```bash
docker compose up --build
```

This starts:
- **API** → http://localhost:8000
- **Prometheus** → http://localhost:9090
- **Grafana** → http://localhost:3000 (admin / admin)

> **Note:** Mount your trained model before starting: the `models/` directory is mounted read-only into the API container. Run `scripts/train.py` locally first.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict next-day closing price |
| `GET` | `/health` | Liveness / readiness check |
| `GET` | `/model/info` | Model metadata |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |

#### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"use_latest_market_data": true}'
```

```json
{
  "ticker": "PETR4.SA",
  "predicted_close": 38.42,
  "last_known_close": 37.85,
  "prediction_date": "2026-05-30",
  "model_version": "1.0.0",
  "drift_detected": false,
  "latency_ms": 312.5
}
```

---

## MLOps Strategy

### 1. Data pipeline
Raw data is downloaded via `yfinance`, validated, and transformed through a reproducible feature engineering pipeline. The scaler is fitted once on training data and persisted — never re-fitted on new data.

### 2. Model versioning & serialisation
The model is saved in the native Keras format (`.keras`). The scaler (`.pkl`) and feature column list (`.json`) are stored alongside it in `models/artifacts/`. Together, these three artefacts fully define the model's inference contract.

### 3. Evaluation & backtesting
Walk-forward validation simulates real-world deployment: the model is retrained periodically on the latest data and evaluated on strictly future observations. Metrics logged: RMSE, MAE, MAPE, R².

### 4. API & containerisation
FastAPI exposes a clean REST interface. The Docker image uses a multi-stage build (builder + runtime stages); final compressed size is ~837 MB.

### 5. Monitoring
Prometheus scrapes `/metrics` every 10 s. Tracked signals:
- `petr4_predictions_total` — request throughput
- `petr4_prediction_latency_seconds` — P50/P95/P99 latency
- `petr4_drift_alerts_total` — how often predictions deviate by >15% from last close
- `petr4_model_mape` — rolling MAPE gauge

### 6. CI/CD
GitHub Actions runs lint → test → Docker build on every push to `main`. Coverage is reported to Codecov.

### Future improvements (v2.0)
- Ensemble: LSTM + Prophet + XGBoost with weighted averaging.
- MLflow for experiment tracking and model registry.
- Automated retraining trigger when MAPE drifts above threshold.
- Feature store for shared indicator computation.

---

## Tests

```bash
PYTHONPATH=. pytest tests/ -v --cov=src --cov=api
```

---

## Configuration

All hyperparameters, paths, and API settings live in `config.yaml`. Environment variables (set in `.env` or passed to Docker) override YAML values — see `.env.example`.
