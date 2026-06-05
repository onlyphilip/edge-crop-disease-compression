"""Combine quality, export, and edge benchmark metrics into report tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_crop_disease_ai.cli import build_common_parser
from edge_crop_disease_ai.config import load_config


COMPARISON_COLUMNS = [
    "Model Variant",
    "Backend",
    "Purpose",
    "Accuracy",
    "Macro-F1",
    "Model Size MB",
    "Avg Latency ms",
    "FPS",
    "RAM Delta MB",
    "Energy / Emissions if available",
    "Deployment Suitability",
]


MODEL_PURPOSES = {
    "Keras original": "Baseline training model",
    "TFLite FP32": "Edge-compatible baseline",
    "TFLite FP16": "Reduced precision model",
    "TFLite INT8": "Aggressively quantized edge model",
    "Pruned model": "Sparse architecture, optional",
    "Distilled model": "Smaller student network, optional",
    "Combined optimization": "Real deployment configuration, optional",
}


def _results_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["results_dir"]).expanduser()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    if value in (None, "", "not_available"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return "not_available"
    return f"{number:.{digits}f}"


def _energy_text(row: dict[str, Any]) -> str:
    energy = _to_float(row.get("energy_kwh"))
    emissions = _to_float(row.get("emissions_kg"))
    parts = []
    if energy is not None:
        parts.append(f"{energy:.8f} kWh")
    if emissions is not None:
        parts.append(f"{emissions:.8f} kg CO2e")
    return " / ".join(parts) if parts else "not_available"


def _deployment_suitability(row: dict[str, Any]) -> str:
    status = row.get("status")
    if status != "ok":
        return "Not evaluated"
    backend = row.get("backend", "")
    latency = _to_float(row.get("avg_latency_ms"))
    size = _to_float(row.get("model_size_mb"))
    ram_delta = _to_float(row.get("ram_delta_mb"))

    if backend == "keras":
        return "Baseline only; heavier runtime dependency"
    if latency is not None and latency <= 50 and size is not None and size <= 10:
        return "Strong edge candidate"
    if latency is not None and latency <= 100 and (ram_delta is None or ram_delta <= 100):
        return "Suitable for CPU edge inference"
    return "Usable with device-specific validation"


def _export_size_lookup(export_summary: dict[str, Any]) -> dict[str, float]:
    exports = export_summary.get("exports", {})
    lookup = {}
    for key, value in exports.items():
        path = value.get("path")
        size_bytes = value.get("size_bytes")
        if path and size_bytes:
            lookup[str(Path(path))] = float(size_bytes) / (1024.0 * 1024.0)
            lookup[key] = float(size_bytes) / (1024.0 * 1024.0)
    return lookup


def build_comparison(config: dict[str, Any]) -> list[dict[str, str]]:
    results_dir = _results_dir(config)
    eval_summary = _read_json(results_dir / "metrics" / "evaluation_summary.json")
    export_summary = _read_json(results_dir / "export" / "export_summary.json")
    benchmark_rows = _read_csv(results_dir / "edge_benchmark" / "edge_metrics_summary.csv")
    export_sizes = _export_size_lookup(export_summary)

    metrics = eval_summary.get("metrics", {})
    default_accuracy = metrics.get("accuracy")
    default_macro_f1 = metrics.get("f1_macro")

    comparison_rows = []
    for row in benchmark_rows:
        model_name = row.get("model_name", "Unknown")
        model_path = row.get("model_path", "")
        size = row.get("model_size_mb") or export_sizes.get(model_path)
        comparison_rows.append(
            {
                "Model Variant": model_name,
                "Backend": row.get("backend", "not_available"),
                "Purpose": MODEL_PURPOSES.get(model_name, "Optional model variant"),
                "Accuracy": _format_number(row.get("accuracy") or default_accuracy),
                "Macro-F1": _format_number(row.get("macro_f1") or default_macro_f1),
                "Model Size MB": _format_number(size),
                "Avg Latency ms": _format_number(row.get("avg_latency_ms"), digits=3),
                "FPS": _format_number(row.get("fps"), digits=3),
                "RAM Delta MB": _format_number(row.get("ram_delta_mb"), digits=3),
                "Energy / Emissions if available": _energy_text(row),
                "Deployment Suitability": _deployment_suitability(row),
            }
        )
    return comparison_rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Edge Model Comparison",
        "",
        "| " + " | ".join(COMPARISON_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COMPARISON_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in COMPARISON_COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = build_common_parser("Compare model quality and edge deployment metrics.")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = _results_dir(config) / "edge_benchmark"
    rows = build_comparison(config)
    write_csv(rows, output_dir / "model_comparison.csv")
    write_markdown(rows, output_dir / "model_comparison.md")
    print(f"Wrote {output_dir / 'model_comparison.csv'}")
    print(f"Wrote {output_dir / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
