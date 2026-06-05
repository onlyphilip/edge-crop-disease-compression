"""Single-image inference using a Keras model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser


def predict_with_keras(config: dict[str, Any], image_path: str) -> dict[str, Any]:
    """Run single-image prediction with a saved Keras model.

    Args:
        config: Parsed project configuration.
        image_path: Path to an image file.

    Returns:
        Prediction summary containing top classes and confidence scores.

    """
    import tensorflow as tf
    from edge_crop_disease_ai.inference.preprocess import (
        build_topk_predictions,
        load_and_preprocess_image,
        load_class_names,
    )

    model_path = Path(config["export"]["keras_model_path"]).expanduser()
    if not model_path.exists():
        fallback_path = (
            Path(config["paths"]["checkpoints_dir"]).expanduser()
            / config["train"]["checkpoint_name"]
        )
        model_path = fallback_path

    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    class_names = load_class_names(config)
    input_tensor = load_and_preprocess_image(image_path, config)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predictions = build_topk_predictions(
        probabilities=probabilities,
        class_names=class_names,
        top_k=int(config["inference"].get("top_k", 3)),
    )

    return {
        "backend": "keras",
        "image_path": str(Path(image_path).expanduser()),
        "model_path": str(model_path),
        "predictions": predictions,
        "predicted_class": predictions[0]["class_name"],
        "predicted_index": predictions[0]["class_index"],
    }


def main() -> None:
    """CLI entry point for Keras single-image inference.

    """
    parser = build_common_parser("Run single-image inference with Keras.")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    args = parser.parse_args()

    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    result = predict_with_keras(config, args.image)

    print("Keras inference completed.")
    print(f"Predicted class: {result['predicted_class']}")
    for prediction in result["predictions"]:
        print(
            f"- {prediction['class_name']}: "
            f"{prediction['confidence']:.4f}"
        )
