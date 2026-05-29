"""
API configuration module.

Loads settings from environment variables (with fallback to config.yaml)
using pydantic-settings. Environment variables take precedence over the
YAML file, which makes container-level overrides clean and explicit.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


def _load_yaml_config(yaml_path: str = "config.yaml") -> dict:
    """Load the root config.yaml file.

    Args:
        yaml_path: Path to the YAML config file.

    Returns:
        Parsed config dict, or empty dict if the file is not found.
    """
    path = Path(yaml_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


_yaml = _load_yaml_config()
_api_cfg = _yaml.get("api", {})
_paths_cfg = _yaml.get("paths", {})
_mon_cfg = _yaml.get("monitoring", {})
_data_cfg = _yaml.get("data", {})
_feat_cfg = _yaml.get("features", {})
_model_cfg = _yaml.get("model", {})


class Settings(BaseSettings):
    """Application settings, overridable via environment variables.

    Environment variable names are the field names uppercased (default
    pydantic-settings behaviour). E.g. MODEL_PATH=... overrides model_path.
    """

    # Server
    api_host: str = _api_cfg.get("host", "0.0.0.0")
    api_port: int = _api_cfg.get("port", 8000)
    log_level: str = os.getenv("LOG_LEVEL", _mon_cfg.get("log_level", "INFO"))

    # Model
    model_path: str = str(
        Path(_paths_cfg.get("model_dir", "models/trained"))
        / _paths_cfg.get("model_filename", "lstm_petr4.keras")
    )
    scaler_path: str = str(
        Path(_paths_cfg.get("artifacts_dir", "models/artifacts"))
        / _paths_cfg.get("scaler_filename", "scaler.pkl")
    )
    feature_columns_path: str = str(
        Path(_paths_cfg.get("artifacts_dir", "models/artifacts"))
        / _paths_cfg.get("feature_columns_filename", "feature_columns.json")
    )
    artifacts_dir: str = _paths_cfg.get("artifacts_dir", "models/artifacts")

    # Data
    ticker: str = _data_cfg.get("ticker", "PETR4.SA")
    sequence_length: int = _feat_cfg.get("sequence_length", 60)

    # Monitoring
    drift_threshold_pct: float = (
        _yaml.get("monitoring", {})
        .get("drift_detection", {})
        .get("alert_threshold_pct", 15.0)
    )
    predictions_dir: str = _data_cfg.get("predictions_dir", "data/predictions")

    # API metadata
    api_title: str = _api_cfg.get("title", "PETR4 Price Predictor API")
    api_description: str = _api_cfg.get("description", "")
    api_version: str = _api_cfg.get("version", "1.0.0")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Using lru_cache avoids re-parsing the .env file on every call.

    Returns:
        Cached Settings instance.
    """
    return Settings()
