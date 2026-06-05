"""Model factory for selecting deployable classifiers."""

from __future__ import annotations

from typing import Any


def build_model(config: dict[str, Any], num_classes: int) -> Any:
    """Build a classifier selected by project config.

    Args:
        config: Parsed project configuration.
        num_classes: Number of target classes.

    Returns:
        Keras model instance.

    """
    model_name = str(config["model"]["name"]).lower()

    if model_name == "mobilenetv2":
        from edge_crop_disease_ai.models.mobilenetv2 import build_mobilenetv2_classifier

        return build_mobilenetv2_classifier(config, num_classes)

    raise ValueError(f"Unsupported model name: {config['model']['name']}")
