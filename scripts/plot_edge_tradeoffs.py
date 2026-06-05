"""Plot EdgeAI accuracy, latency, size, FPS, and memory tradeoffs."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_crop_disease_ai.cli import build_common_parser
from edge_crop_disease_ai.config import load_config


ALIASES = {
    "label": ("Model Variant", "model_name"),
    "accuracy": ("Accuracy", "accuracy"),
    "model_size_mb": ("Model Size MB", "model_size_mb"),
    "avg_latency_ms": ("Avg Latency ms", "avg_latency_ms"),
    "fps": ("FPS", "fps"),
    "ram_delta_mb": ("RAM Delta MB", "ram_delta_mb"),
    "peak_memory_mb": ("Peak Memory MB", "peak_memory_mb"),
    "status": ("status",),
}

SHORT_LABELS = {
    "Keras original": "Keras",
    "TFLite FP32": "FP32",
    "TFLite FP16": "FP16",
    "TFLite INT8": "INT8",
}


def _to_float(value: Any) -> float | None:
    if value in (None, "", "not_available"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value(row: dict[str, Any], key: str) -> Any:
    for name in ALIASES[key]:
        if name in row:
            return row.get(name)
    return None


def _short_label(label: Any) -> str:
    text = str(label)
    return SHORT_LABELS.get(text, text)


def _load_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    results_dir = Path(config["paths"]["results_dir"]).expanduser()
    benchmark_path = results_dir / "edge_benchmark" / "edge_metrics_summary.csv"
    comparison_path = results_dir / "edge_benchmark" / "model_comparison.csv"
    path = benchmark_path if benchmark_path.exists() else comparison_path
    if not path.exists():
        raise FileNotFoundError(
            "No edge metrics found. Run scripts/benchmark.py and scripts/compare_edge_models.py first."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if _value(row, "status") in (None, "", "ok")]


def _points(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[str], list[float], list[float]]:
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        label = _value(row, "label")
        x_value = _to_float(_value(row, x_key))
        y_value = _to_float(_value(row, y_key))
        if label and x_value is not None and y_value is not None:
            labels.append(_short_label(label))
            xs.append(x_value)
            ys.append(y_value)
    return labels, xs, ys


def _bars(rows: list[dict[str, Any]], value_key: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        label = _value(row, "label")
        value = _to_float(_value(row, value_key))
        if label and value is not None:
            labels.append(_short_label(label))
            values.append(value)
    return labels, values


def _expand_limits(values: list[float], padding_ratio: float = 0.12) -> tuple[float, float] | None:
    if not values:
        return None
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.01, 0.01)
        return low - pad, high + pad
    pad = (high - low) * padding_ratio
    return low - pad, high + pad


def _no_data(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def _annotate_points(ax: Any, labels: list[str], xs: list[float], ys: list[float]) -> None:
    offsets = [(10, 12), (10, -22), (-38, 12), (-38, -22), (10, 28)]
    for index, (label, x_value, y_value) in enumerate(zip(labels, xs, ys, strict=False)):
        offset = offsets[index % len(offsets)]
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.6},
        )


def _save_scatter(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels, xs, ys = _points(rows, x_key, y_key)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if labels:
        ax.scatter(xs, ys, s=90, alpha=0.85)
        _annotate_points(ax, labels, xs, ys)
        x_limits = _expand_limits(xs)
        y_limits = _expand_limits(ys)
        if x_limits:
            ax.set_xlim(*x_limits)
        if y_limits:
            ax.set_ylim(*y_limits)
    else:
        _no_data(ax, "No numeric data available for this plot")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_bar(
    *,
    rows: list[dict[str, Any]],
    value_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    fallback_key: str | None = None,
    fallback_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels, values = _bars(rows, value_key)
    label = y_label
    if not values and fallback_key is not None:
        labels, values = _bars(rows, fallback_key)
        label = fallback_label or y_label

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if values:
        bars = ax.bar(labels, values, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        upper = max(values) * 1.18 if max(values) > 0 else 1.0
        ax.set_ylim(0, upper)
    else:
        _no_data(ax, "No memory metric available on this machine")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_common_parser("Plot EdgeAI model tradeoffs.")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = _load_rows(config)
    figures_dir = Path(config["paths"]["results_dir"]).expanduser() / "edge_benchmark" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_scatter(
        rows=rows,
        x_key="avg_latency_ms",
        y_key="accuracy",
        x_label="Average Latency (ms)",
        y_label="Accuracy",
        title="Accuracy vs Latency",
        output_path=figures_dir / "accuracy_vs_latency.png",
    )
    _save_scatter(
        rows=rows,
        x_key="model_size_mb",
        y_key="accuracy",
        x_label="Model Size (MB)",
        y_label="Accuracy",
        title="Model Size vs Accuracy",
        output_path=figures_dir / "model_size_vs_accuracy.png",
    )
    _save_scatter(
        rows=rows,
        x_key="model_size_mb",
        y_key="avg_latency_ms",
        x_label="Model Size (MB)",
        y_label="Average Latency (ms)",
        title="Latency vs Model Size",
        output_path=figures_dir / "latency_vs_model_size.png",
    )
    _save_bar(
        rows=rows,
        value_key="fps",
        y_label="FPS",
        title="FPS Comparison",
        output_path=figures_dir / "fps_comparison.png",
    )
    _save_bar(
        rows=rows,
        value_key="ram_delta_mb",
        y_label="RAM Delta (MB)",
        title="Memory Comparison",
        output_path=figures_dir / "memory_comparison.png",
        fallback_key="model_size_mb",
        fallback_label="Model Size (MB, RAM metric unavailable)",
    )
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
