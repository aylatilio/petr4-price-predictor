"""Root-level conftest: TF stub + shared API fixture."""
import sys
from types import ModuleType
from unittest.mock import MagicMock
import pytest


def _make_keras_stub():
    keras = ModuleType("keras")
    keras.layers = MagicMock()
    keras.models = MagicMock()
    keras.optimizers = MagicMock()
    keras.callbacks = MagicMock()
    for sub in ("layers", "models", "optimizers", "callbacks"):
        mod = ModuleType(f"keras.{sub}")
        mod.__dict__.update(vars(getattr(keras, sub)))
        sys.modules[f"keras.{sub}"] = mod
    return keras


if "tensorflow" not in sys.modules:
    tf_stub = ModuleType("tensorflow")
    tf_stub.keras = MagicMock()
    tf_stub.random = MagicMock()
    tf_stub.keras.Model = MagicMock
    tf_stub.keras.models = MagicMock()
    tf_stub.keras.models.load_model = MagicMock(return_value=MagicMock())
    sys.modules["tensorflow"] = tf_stub
    sys.modules["keras"] = _make_keras_stub()


@pytest.fixture(scope="module")
def api_client():
    """TestClient with lifespan active (app.state populated)."""
    from fastapi.testclient import TestClient
    from api.app import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
