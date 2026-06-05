"""Keras callback builders for checkpoints and scheduling."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_callbacks(config: dict[str, Any]) -> list[Any]:
    """Build training callbacks from config.

    Args:
        config: Parsed project configuration.

    Returns:
        List of Keras callback instances.

    """
    import tensorflow as tf

    train_cfg = config["train"]
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"]).expanduser()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / train_cfg["checkpoint_name"]
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))
    reduce_lr_patience = int(train_cfg.get("reduce_lr_patience", 2))

    callbacks: list[Any] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks
