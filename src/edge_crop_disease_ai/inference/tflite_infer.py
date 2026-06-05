"""Single-image inference using a TFLite model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser


def _quantize_input(input_tensor: Any, input_details: dict[str, Any]) -> Any:
    """Quantize a float input tensor to the dtype required by the TFLite model."""
    import numpy as np

    dtype = input_details["dtype"]
    if dtype == np.float32:
        return input_tensor.astype(np.float32)

    scale, zero_point = input_details["quantization"]
    if not scale:
        raise ValueError("Input quantization scale is zero; cannot quantize input.")

    quantized = np.round(input_tensor / scale + zero_point)
    return np.clip(quantized, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def _dequantize_output(output_tensor: Any, output_details: dict[str, Any]) -> Any:
    """Dequantize TFLite model outputs when required."""
    import numpy as np

    dtype = output_details["dtype"]
    if dtype == np.float32:
        return output_tensor.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    if not scale:
        return output_tensor.astype(np.float32)
    return (output_tensor.astype(np.float32) - zero_point) * scale


def predict_with_tflite(config: dict[str, Any], image_path: str, model_path: str) -> dict[str, Any]:
    """Run single-image prediction with a TFLite model.

    Args:
        config: Parsed project configuration.
        image_path: Path to an image file.
        model_path: Path to a TFLite model.

    Returns:
        Prediction summary containing top classes and confidence scores.

    """
    import tensorflow as tf
    from edge_crop_disease_ai.inference.preprocess import (
        build_topk_predictions,
        load_and_preprocess_image,
        load_class_names,
    )

    tflite_model_path = Path(model_path).expanduser()
    if not tflite_model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")

    class_names = load_class_names(config)
    float_input = load_and_preprocess_image(image_path, config)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    model_input = _quantize_input(float_input, input_details)
    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details["index"])
    probabilities = _dequantize_output(raw_output, output_details)[0]
    predictions = build_topk_predictions(
        probabilities=probabilities,
        class_names=class_names,
        top_k=int(config["inference"].get("top_k", 3)),
    )

    return {
        "backend": "tflite",
        "image_path": str(Path(image_path).expanduser()),
        "model_path": str(tflite_model_path),
        "predictions": predictions,
        "predicted_class": predictions[0]["class_name"],
        "predicted_index": predictions[0]["class_index"],
    }


def main() -> None:
    """CLI entry point for TFLite single-image inference.

    """
    parser = build_common_parser("Run single-image inference with TFLite.")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model.")
    args = parser.parse_args()

    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    result = predict_with_tflite(config, args.image, args.model)

    print("TFLite inference completed.")
    print(f"Predicted class: {result['predicted_class']}")
    for prediction in result["predictions"]:
        print(
            f"- {prediction['class_name']}: "
            f"{prediction['confidence']:.4f}"
        )
