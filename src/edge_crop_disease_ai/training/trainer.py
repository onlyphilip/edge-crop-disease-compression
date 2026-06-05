"""Training workflow entry point."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser


def _set_global_seed(seed: int) -> None:
    """Set project-wide random seeds for reproducibility."""
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_optimizer(config: dict[str, Any]) -> Any:
    """Build the configured optimizer for training."""
    import tensorflow as tf

    train_cfg = config["train"]
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")


def _serialize_history(history: Any) -> dict[str, list[float]]:
    """Convert Keras history values into JSON-safe floats."""
    serialized: dict[str, list[float]] = {}
    for key, values in history.history.items():
        serialized[key] = [float(value) for value in values]
    return serialized


def _save_training_artifacts(
    config: dict[str, Any],
    dataset_bundle: dict[str, Any],
    history: Any,
) -> None:
    """Persist training metadata for later evaluation and export."""
    from edge_crop_disease_ai.utils.io import save_json

    results_dir = Path(config["paths"]["results_dir"]).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        split_name: len(split_samples)
        for split_name, split_samples in dataset_bundle["splits"].items()
    }
    artifacts = {
        "model_name": config["model"]["name"],
        "num_classes": dataset_bundle["num_classes"],
        "class_names": dataset_bundle["class_names"],
        "input_shape": list(dataset_bundle["input_shape"]),
        "split_counts": split_counts,
        "history": _serialize_history(history),
    }
    save_json(artifacts, results_dir / "training_history.json")


def train(config: dict[str, Any]) -> Any:
    """Run end-to-end model training.

    Args:
        config: Parsed project configuration.

    Returns:
        Training artifacts such as history, checkpoints, or model handles.

    """
    from edge_crop_disease_ai.config import ensure_output_dirs
    from edge_crop_disease_ai.data.dataset import build_tf_datasets
    from edge_crop_disease_ai.models.factory import build_model
    from edge_crop_disease_ai.training.callbacks import build_callbacks

    ensure_output_dirs(config)

    seed = int(config.get("project", {}).get("seed", 42))
    _set_global_seed(seed)

    dataset_bundle = build_tf_datasets(config)
    train_ds = dataset_bundle["datasets"]["train"]
    val_ds = dataset_bundle["datasets"]["val"]

    model = build_model(config, dataset_bundle["num_classes"])
    model.compile(
        optimizer=_build_optimizer(config),
        loss=config["train"].get("loss", "sparse_categorical_crossentropy"),
        metrics=list(config["train"].get("metrics", ["accuracy"])),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["train"]["epochs"]),
        callbacks=build_callbacks(config),
        verbose=1,
    )

    _save_training_artifacts(config, dataset_bundle, history)
    return {
        "model": model,
        "history": history,
        "dataset_bundle": dataset_bundle,
    }


def main() -> None:
    """CLI entry point for training.

    """
    parser = build_common_parser("Train a plant disease classifier.")
    args = parser.parse_args()
    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    artifacts = train(config)

    checkpoint_path = Path(config["paths"]["checkpoints_dir"]).expanduser() / config["train"]["checkpoint_name"]
    print("Training completed.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Classes: {artifacts['dataset_bundle']['num_classes']}")
