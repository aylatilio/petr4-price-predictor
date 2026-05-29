"""
LSTM model architecture.

Defines a stacked LSTM network for single-step time-series forecasting.
The architecture is fully parametric — every hyperparameter is read from
config.yaml so that experiments only require a config change.

Architecture overview:
  Input → LSTM (units[0]) → Dropout → LSTM (units[1]) → Dropout
        → Dense (dense_units[0]) → Dense (1)  [regression output]
"""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers

logger = logging.getLogger(__name__)


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units: list[int],
    dense_units: list[int],
    dropout_rate: float,
    recurrent_dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    """Build and compile a stacked LSTM regression model.

    Args:
        input_shape: Tuple (sequence_length, n_features).
        lstm_units: List of unit counts for each LSTM layer.
        dense_units: List of unit counts for each Dense layer before the output.
        dropout_rate: Fraction of units to drop after each LSTM layer.
        recurrent_dropout: Fraction of recurrent connections to drop inside LSTM.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Model ready for training.
    """
    if len(lstm_units) < 1:
        raise ValueError("lstm_units must contain at least one element.")

    model = models.Sequential(name="petr4_lstm_predictor")
    model.add(layers.Input(shape=input_shape, name="input"))

    # Build stacked LSTM layers
    for i, units in enumerate(lstm_units):
        is_last_lstm = i == len(lstm_units) - 1
        model.add(
            layers.LSTM(
                units=units,
                return_sequences=not is_last_lstm,  # True for all but the last
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{i + 1}",
            )
        )
        model.add(layers.Dropout(rate=dropout_rate, name=f"dropout_{i + 1}"))

    # Optional dense hidden layers
    for j, units in enumerate(dense_units):
        model.add(layers.Dense(units=units, activation="relu", name=f"dense_{j + 1}"))

    # Single regression output
    model.add(layers.Dense(units=1, activation="linear", name="output"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info("Model built. Total parameters: %d", model.count_params())
    model.summary(print_fn=logger.info)

    return model


def save_model(model: tf.keras.Model, model_path: str) -> None:
    """Persist a trained Keras model to disk in the native Keras format.

    Args:
        model: Trained Keras model.
        model_path: Destination file path (e.g. 'models/trained/lstm_petr4.keras').
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("Model saved to %s", model_path)


def load_model(model_path: str) -> tf.keras.Model:
    """Load a Keras model from disk.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded Keras Model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train the model first using scripts/train.py."
        )

    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def predict_next_close(
    model: tf.keras.Model,
    sequence: np.ndarray,
    scaler,
    target_col_index: int,
    n_features: int,
) -> float:
    """Predict the next closing price and inverse-transform back to BRL.

    The scaler was fitted on all features, so we reconstruct a full-width
    dummy row to call inverse_transform correctly.

    Args:
        model: Loaded Keras LSTM model.
        sequence: Array of shape (1, sequence_length, n_features).
        scaler: Fitted MinMaxScaler (loaded from artifacts).
        target_col_index: Index of the 'Close' column in the feature list.
        n_features: Total number of features (needed for inverse transform).

    Returns:
        Predicted next-day closing price in BRL (R$).
    """
    scaled_prediction = model.predict(sequence, verbose=0)[0][0]

    # Reconstruct a full-feature row filled with zeros except for the target
    dummy_row = np.zeros((1, n_features))
    dummy_row[0, target_col_index] = scaled_prediction
    price = scaler.inverse_transform(dummy_row)[0, target_col_index]

    return float(price)
