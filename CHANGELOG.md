# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [1.0.0] — 2026-05-29

### Added
- Full project scaffold: `data/`, `models/`, `src/`, `api/`, `scripts/`, `tests/`, `docs/`
- `src/data/collector.py` — yfinance download with CSV persistence
- `src/data/feature_engineering.py` — SMA, RSI, MACD, Bollinger Bands
- `src/data/preprocessor.py` — MinMaxScaler pipeline, sequence building, artefact persistence
- `src/models/lstm_model.py` — Stacked LSTM architecture (Keras), save/load helpers
- `src/training/trainer.py` — Training pipeline with EarlyStopping, ReduceLROnPlateau
- `src/evaluation/metrics.py` — RMSE, MAE, MAPE, R²
- `src/evaluation/backtester.py` — Walk-forward validation
- `src/monitoring/logger.py` — Prometheus metrics + structured JSON logging
- `api/app.py` — FastAPI with `/predict`, `/health`, `/model/info`, `/metrics`
- `api/schemas.py` — Pydantic v2 request/response models
- `api/config.py` — pydantic-settings configuration with YAML + env override
- `scripts/collect_data.py`, `train.py`, `predict.py`, `backtest.py` — CLI pipeline
- `tests/test_metrics.py`, `test_feature_engineering.py`, `test_api.py` — pytest suite
- `Dockerfile` — Multi-stage build (builder + runtime)
- `docker-compose.yml` — API + Prometheus + Grafana stack
- `.github/workflows/ci.yml` — lint → test → Docker build CI pipeline
- `config.yaml` — Central configuration for all pipeline components
- `README.md`, `CHANGELOG.md`, `.gitignore`, `.env.example`
