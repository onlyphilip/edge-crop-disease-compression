"""Edge deployment benchmarking utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser
from edge_crop_disease_ai.evaluation.edge_metrics import (
    EdgeModelSpec,
    NOT_AVAILABLE,
    array_nbytes,
    benchmark_inference,
)
from edge_crop_disease_ai.utils.io import save_json


SUMMARY_COLUMNS = [
    "model_name",
    "backend",
    "model_path",
    "model_size_mb",
    "accuracy",
    "macro_f1",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "fps",
    "ram_before_mb",
    "ram_after_mb",
    "ram_delta_mb",
    "peak_memory_mb",
    "cpu_percent_avg",
    "energy_kwh",
    "emissions_kg",
    "input_bytes",
    "output_bytes",
    "status",
]


def _edge_output_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["results_dir"]).expanduser() / "edge_benchmark"


def _resolve_sample_image(config: dict[str, Any]) -> str:
    """Pick a deterministic sample image from the test split for benchmarking."""
    from edge_crop_disease_ai.data.split import create_data_splits, discover_imagefolder_samples

    samples = discover_imagefolder_samples(
        data_dir=config["paths"]["data_dir"],
        allowed_extensions=config["data"]["file_extensions"],
    )
    splits = create_data_splits(
        samples=samples,
        val_ratio=float(config["data"]["val_ratio"]),
        test_ratio=float(config["data"]["test_ratio"]),
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    if not splits["test"]:
        raise ValueError("Test split is empty; cannot benchmark inference.")
    return splits["test"][0]["image_path"]


def _resolve_keras_model_path(config: dict[str, Any]) -> Path:
    candidate = Path(config["export"]["keras_model_path"]).expanduser()
    if candidate.exists():
        return candidate
    return Path(config["paths"]["checkpoints_dir"]).expanduser() / config["train"]["checkpoint_name"]


def _model_specs(config: dict[str, Any]) -> list[EdgeModelSpec]:
    export_cfg = config["export"]
    export_dir = Path(export_cfg["export_dir"]).expanduser()
    optional_cfg = config.get("edge_models", {})
    specs = [
        EdgeModelSpec(
            model_name="Keras original",
            backend="keras",
            model_path=_resolve_keras_model_path(config),
            variant="keras_original",
            purpose="Baseline training model",
        ),
        EdgeModelSpec(
            model_name="TFLite FP32",
            backend="tflite",
            model_path=export_dir / export_cfg["tflite_name_fp32"],
            variant="tflite_fp32",
            purpose="Edge-compatible baseline",
        ),
        EdgeModelSpec(
            model_name="TFLite FP16",
            backend="tflite",
            model_path=export_dir / export_cfg["tflite_name_fp16"],
            variant="tflite_fp16",
            purpose="Reduced precision model",
        ),
        EdgeModelSpec(
            model_name="TFLite INT8",
            backend="tflite",
            model_path=export_dir / export_cfg["tflite_name_int8"],
            variant="tflite_int8",
            purpose="Aggressively quantized edge model",
        ),
    ]

    optional_variants = [
        ("pruned", "Pruned model", "Sparse architecture, optional"),
        ("distilled", "Distilled model", "Smaller student network, optional"),
        ("combined", "Combined optimization", "Real deployment configuration, optional"),
    ]
    for key, name, purpose in optional_variants:
        raw_path = optional_cfg.get(f"{key}_model_path") or optional_cfg.get(key)
        if raw_path:
            path = Path(raw_path).expanduser()
            backend = "tflite" if path.suffix.lower() == ".tflite" else "keras"
            specs.append(
                EdgeModelSpec(
                    model_name=name,
                    backend=backend,
                    model_path=path,
                    variant=key,
                    purpose=purpose,
                )
            )
        else:
            specs.append(
                EdgeModelSpec(
                    model_name=name,
                    backend="optional",
                    model_path=Path(""),
                    variant=key,
                    purpose=purpose,
                    available=False,
                )
            )
    return specs


def _load_quality_metrics(config: dict[str, Any]) -> tuple[float | None, float | None]:
    path = Path(config["paths"]["results_dir"]).expanduser() / "metrics" / "evaluation_summary.json"
    if not path.exists():
        return None, None
    try:
        import json

        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics", {})
        return metrics.get("accuracy"), metrics.get("f1_macro")
    except Exception:
        return None, None


def _build_keras_runner(config: dict[str, Any], model_path: Path, image_path: str) -> tuple[Any, Any, Any]:
    import tensorflow as tf

    from edge_crop_disease_ai.inference.preprocess import load_and_preprocess_image

    model = tf.keras.models.load_model(model_path)
    input_tensor = load_and_preprocess_image(image_path, config)

    def infer() -> Any:
        return model.predict(input_tensor, verbose=0)

    return infer, input_tensor, infer()


def _build_tflite_runner(config: dict[str, Any], model_path: Path, image_path: str) -> tuple[Any, Any, Any]:
    import tensorflow as tf

    from edge_crop_disease_ai.inference.preprocess import load_and_preprocess_image
    from edge_crop_disease_ai.inference.tflite_infer import _dequantize_output, _quantize_input

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    float_input = load_and_preprocess_image(image_path, config)
    model_input = _quantize_input(float_input, input_details)

    def infer() -> Any:
        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()
        return _dequantize_output(interpreter.get_tensor(output_details["index"]), output_details)

    return infer, model_input, infer()


def _unavailable_result(spec: EdgeModelSpec, reason: str) -> dict[str, Any]:
    return {
        "model_name": spec.model_name,
        "backend": spec.backend,
        "model_path": str(spec.model_path),
        "model_size_mb": None,
        "accuracy": None,
        "macro_f1": None,
        "avg_latency_ms": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "fps": None,
        "ram_before_mb": NOT_AVAILABLE,
        "ram_after_mb": NOT_AVAILABLE,
        "ram_delta_mb": NOT_AVAILABLE,
        "peak_memory_mb": NOT_AVAILABLE,
        "cpu_percent_avg": NOT_AVAILABLE,
        "energy_kwh": NOT_AVAILABLE,
        "emissions_kg": NOT_AVAILABLE,
        "input_bytes": NOT_AVAILABLE,
        "output_bytes": NOT_AVAILABLE,
        "status": reason,
        "latencies_ms": [],
        "purpose": spec.purpose,
        "variant": spec.variant,
    }


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def benchmark_edge_models(config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark all configured edge model variants."""
    from edge_crop_disease_ai.config import ensure_output_dirs

    ensure_output_dirs(config)
    output_dir = _edge_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_sample_image(config)
    warmup_runs = int(config["benchmark"].get("warmup_runs", 10))
    benchmark_runs = int(config["benchmark"].get("benchmark_runs", 100))
    measure_peak_memory = bool(config["benchmark"].get("measure_peak_memory", False))
    measure_energy = bool(config["benchmark"].get("measure_energy", True))
    accuracy, macro_f1 = _load_quality_metrics(config)

    raw_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in _model_specs(config):
        if not spec.available:
            result = _unavailable_result(spec, "optional_not_configured")
        elif not spec.model_path.exists():
            result = _unavailable_result(spec, "model_not_found")
        else:
            try:
                if spec.backend == "keras":
                    infer, model_input, output_sample = _build_keras_runner(config, spec.model_path, image_path)
                elif spec.backend == "tflite":
                    infer, model_input, output_sample = _build_tflite_runner(config, spec.model_path, image_path)
                else:
                    result = _unavailable_result(spec, "unsupported_backend")
                    raw_results.append(result)
                    summary_rows.append(result)
                    continue

                result = benchmark_inference(
                    model_name=spec.model_name,
                    backend=spec.backend,
                    model_path=spec.model_path,
                    inference_fn=infer,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                    input_bytes=array_nbytes(model_input),
                    output_bytes_fn=array_nbytes,
                    energy_output_dir=output_dir,
                    measure_peak_memory=measure_peak_memory,
                    measure_energy=measure_energy,
                )
                result["output_bytes"] = array_nbytes(output_sample)
            except Exception as exc:
                result = _unavailable_result(spec, "failed")
                result["error_message"] = str(exc)

        result["variant"] = spec.variant
        result["purpose"] = spec.purpose
        result["accuracy"] = accuracy
        result["macro_f1"] = macro_f1
        raw_results.append(result)
        summary_rows.append({column: result.get(column) for column in SUMMARY_COLUMNS})

    payload = {
        "image_path": image_path,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "results": raw_results,
    }
    save_json(payload, output_dir / "edge_metrics_raw.json")
    _write_summary_csv(summary_rows, output_dir / "edge_metrics_summary.csv")
    return payload


def benchmark_latency(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible benchmark entry returning the first successful result."""
    payload = benchmark_edge_models(config)
    for result in payload["results"]:
        if result.get("status") == "ok":
            return {
                "backend": result["backend"],
                "model_path": result["model_path"],
                "image_path": payload["image_path"],
                "warmup_runs": payload["warmup_runs"],
                "benchmark_runs": payload["benchmark_runs"],
                "latency_ms": {
                    "mean": result["avg_latency_ms"],
                    "median": result["p50_latency_ms"],
                    "p95": result["p95_latency_ms"],
                    "min": result.get("min_latency_ms"),
                    "max": result.get("max_latency_ms"),
                },
                "fps": result["fps"],
            }
    raise RuntimeError("No model variant completed benchmarking successfully.")


def main() -> None:
    """CLI entry point for edge deployment benchmarking."""
    parser = build_common_parser("Benchmark Keras and TFLite edge deployment metrics.")
    args = parser.parse_args()

    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    payload = benchmark_edge_models(config)

    print("Edge benchmark completed.")
    print(f"Raw metrics: {_edge_output_dir(config) / 'edge_metrics_raw.json'}")
    print(f"Summary CSV: {_edge_output_dir(config) / 'edge_metrics_summary.csv'}")
    for result in payload["results"]:
        latency = result.get("avg_latency_ms")
        fps = result.get("fps")
        latency_text = f"{latency:.3f} ms" if isinstance(latency, (int, float)) else "n/a"
        fps_text = f"{fps:.3f}" if isinstance(fps, (int, float)) else "n/a"
        print(f"- {result['model_name']} [{result['backend']}]: {result['status']}, latency={latency_text}, fps={fps_text}")
