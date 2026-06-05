"""Evaluation workflow entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_crop_disease_ai.cli import build_common_parser
from edge_crop_disease_ai.utils.io import save_json


def _resolve_model_path(config: dict[str, Any]) -> Path:
    """Resolve the Keras checkpoint path used for evaluation."""
    export_model_path = config.get("export", {}).get("keras_model_path")
    if export_model_path:
        candidate = Path(export_model_path).expanduser()
        if candidate.exists():
            return candidate

    checkpoint_path = (
        Path(config["paths"]["checkpoints_dir"]).expanduser()
        / config["train"]["checkpoint_name"]
    )
    return checkpoint_path


def _save_confusion_matrix_plot(
    matrix: Any,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render and save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(10, len(class_names) * 0.6)
    figure_height = max(8, len(class_names) * 0.5)

    plt.figure(figsize=(figure_width, figure_height))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a trained model on the test split.

    Args:
        config: Parsed project configuration.

    Returns:
        Evaluation summary with metrics and artifact paths.

    """
    import numpy as np
    import tensorflow as tf

    from edge_crop_disease_ai.config import ensure_output_dirs
    from edge_crop_disease_ai.data.dataset import build_tf_datasets
    from edge_crop_disease_ai.evaluation.metrics import (
        compute_classification_metrics,
        compute_confusion_matrix,
    )

    ensure_output_dirs(config)
    dataset_bundle = build_tf_datasets(config)
    test_dataset = dataset_bundle["datasets"]["test"]
    test_samples = dataset_bundle["splits"]["test"]
    class_names = dataset_bundle["class_names"]

    model_path = _resolve_model_path(config)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1).tolist()
    y_true = [sample["class_index"] for sample in test_samples]

    metrics_summary = compute_classification_metrics(y_true, y_pred, class_names)
    normalized_cm = compute_confusion_matrix(
        y_true,
        y_pred,
        normalize=bool(config["evaluation"].get("normalize_confusion_matrix", True)),
    )
    raw_cm = compute_confusion_matrix(y_true, y_pred, normalize=False)

    metrics_dir = Path(config["paths"]["results_dir"]).expanduser() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    confusion_plot_path = metrics_dir / "confusion_matrix.png"
    _save_confusion_matrix_plot(normalized_cm, class_names, confusion_plot_path)

    evaluation_summary = {
        "model_path": str(model_path),
        "num_test_samples": len(test_samples),
        "class_names": class_names,
        "metrics": metrics_summary,
        "confusion_matrix_normalized": np.asarray(normalized_cm).tolist(),
        "confusion_matrix_raw": np.asarray(raw_cm).tolist(),
        "artifacts": {
            "summary_json": str(metrics_dir / "evaluation_summary.json"),
            "classification_report_json": str(metrics_dir / "classification_report.json"),
            "confusion_matrix_plot": str(confusion_plot_path),
        },
    }

    save_json(evaluation_summary, metrics_dir / "evaluation_summary.json")
    save_json(
        metrics_summary["classification_report"],
        metrics_dir / "classification_report.json",
    )
    return evaluation_summary


def main() -> None:
    """CLI entry point for evaluation.

    """
    parser = build_common_parser("Evaluate a trained classifier.")
    args = parser.parse_args()

    from edge_crop_disease_ai.config import load_config

    config = load_config(args.config)
    summary = evaluate(config)

    metrics = summary["metrics"]
    print("Evaluation completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Macro-F1: {metrics['f1_macro']:.4f}")
    print(f"Artifacts: {summary['artifacts']}")
