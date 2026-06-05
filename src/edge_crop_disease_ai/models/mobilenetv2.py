"""MobileNetV2 model builder for edge deployment."""

from __future__ import annotations

from typing import Any


def _resolve_input_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    """Resolve model input shape from config."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    input_shape = model_cfg.get("input_shape")
    if input_shape:
        return tuple(int(dim) for dim in input_shape)
    return (
        int(data_cfg["image_size"]),
        int(data_cfg["image_size"]),
        int(data_cfg.get("channels", 3)),
    )


def build_mobilenetv2_classifier(config: dict[str, Any], num_classes: int) -> Any:
    """Build the primary MobileNetV2 classifier.

    Args:
        config: Parsed project configuration.
        num_classes: Number of target classes.

    Returns:
        Uncompiled or compiled Keras model, depending on final design.

    """
    import tensorflow as tf

    model_cfg = config["model"]
    input_shape = _resolve_input_shape(config)
    dropout_rate = float(model_cfg.get("dropout_rate", 0.2))
    weights = model_cfg.get("weights", "imagenet")
    train_base = bool(model_cfg.get("train_base", False))

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_rescale")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = train_base

    x = base_model(x, training=train_base)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="plant_disease_mobilenetv2")
    return model
