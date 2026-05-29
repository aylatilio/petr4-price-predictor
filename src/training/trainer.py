"""
Model training pipeline.

Orchestrates the end-to-end training flow:
  1. Load preprocessed sequences.
  2. Build the LSTM model from config.
  3. Train with early stopping and model checkpoint.
  4. Persist the trained model and training history.
"""

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.models.lstm_model import build_lstm_model, save_model

logger = logging.getLogger(__name__)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> tuple[tf.keras.Model, dict]:
    """Train the LSTM model.

    Args:
        X_train: Training sequences of shape (n_samples, sequence_length, n_features).
        y_train: Training targets of shape (n_samples,).
        config: Full project config dict.

    Returns:
        Tuple of (trained Keras model, training history dict).
    """
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    # Reproducibility
    tf.random.set_seed(train_cfg["random_seed"])
    np.random.seed(train_cfg["random_seed"])

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info("Building model with input_shape=%s", input_shape)

    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=model_cfg["lstm_units"],
        dense_units=model_cfg["dense_units"],
        dropout_rate=model_cfg["dropout_rate"],
        recurrent_dropout=model_cfg["recurrent_dropout"],
        learning_rate=train_cfg["learning_rate"],
    )

    # Callbacks
    model_dir = Path(paths_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(model_dir / "best_checkpoint.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, val_split=%.0f%%",
        train_cfg["epochs"],
        train_cfg["batch_size"],
        train_cfg["validation_split"] * 100,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_split=train_cfg["validation_split"],
        shuffle=train_cfg["shuffle"],  # must remain False for time series
        callbacks=callbacks,
        verbose=1,
    )

    # Persist trained model
    final_model_path = str(model_dir / paths_cfg["model_filename"])
    save_model(model, final_model_path)

    # Persist training history
    history_dict = history.history
    artifacts_dir = Path(paths_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifacts_dir / paths_cfg["training_history_filename"]
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1
    best_val_loss = min(history_dict["val_loss"])
    logger.info(
        "Training complete. Best epoch: %d | Best val_loss: %.6f",
        best_epoch,
        best_val_loss,
    )

    return model, history_dict
