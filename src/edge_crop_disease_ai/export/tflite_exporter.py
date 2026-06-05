"""Keras-to-TFLite export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser
from edge_crop_disease_ai.utils.io import save_json


def _resolve_keras_model_path(config: dict[str, Any]) -> Path:
    """Resolve the source Keras model path for TFLite export."""
    export_cfg = config["export"]
    candidate = Path(export_cfg["keras_model_path"]).expanduser()
    if candidate.exists():
        return candidate

    fallback = (
        Path(config["paths"]["checkpoints_dir"]).expanduser()
        / config["train"]["checkpoint_name"]
    )
    return fallback


def _convert_fp32(model: Any) -> bytes:
    """Convert a Keras model to FP32 TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def _convert_fp16(model: Any) -> bytes:
    """Convert a Keras model to FP16-quantized TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def _convert_int8(model: Any, representative_dataset_factory: Any) -> bytes:
    """Convert a Keras model to fully quantized INT8 TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_factory
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export_tflite_models(config: dict[str, Any]) -> dict[str, str]:
    """Export Keras checkpoints to multiple TFLite precision variants.

    Args:
        config: Parsed project configuration.

    Returns:
        Mapping from export type to output file path.

    """
    import tensorflow as tf

    from edge_crop_disease_ai.config import ensure_output_dirs
    from edge_crop_disease_ai.data.dataset import build_representative_dataset

    ensure_output_dirs(config)

    model_path = _resolve_keras_model_path(config)
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    export_cfg = config["export"]
    export_dir = Path(export_cfg["export_dir"]).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)

    representative_dataset_factory = build_representative_dataset(config)
    exported_files: dict[str, str] = {}

    fp32_path = export_dir / export_cfg["tflite_name_fp32"]
    fp32_path.write_bytes(_convert_fp32(model))
    exported_files["fp32"] = str(fp32_path)

    fp16_path = export_dir / export_cfg["tflite_name_fp16"]
    fp16_path.write_bytes(_convert_fp16(model))
    exported_files["fp16"] = str(fp16_path)

    int8_path = export_dir / export_cfg["tflite_name_int8"]
    int8_path.write_bytes(_convert_int8(model, representative_dataset_factory))
    exported_files["int8"] = str(int8_path)

    summary = {
        "source_model": str(model_path),
        "exports": {
            export_type: {
                "path": file_path,
                "size_bytes": Path(file_path).stat().st_size,
            }
            for export_type, file_path in exported_files.items()
        },
    }
    save_json(summary, export_dir / "export_summary.json")

    return exported_files


def main() -> None:
    """CLI entry point for TFLite export.

    """
    parser = build_common_parser("Export a Keras model to TFLite variants.")
    args = parser.parse_args()

    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    exported_files = export_tflite_models(config)

    print("TFLite export completed.")
    for export_type, file_path in exported_files.items():
        print(f"{export_type}: {file_path}")
