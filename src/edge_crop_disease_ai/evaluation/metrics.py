"""Metric computation helpers for classification tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    """Compute classification metrics for model evaluation.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Ordered class label names.

    Returns:
        Dictionary containing accuracy, precision, recall, macro-F1, and related outputs.

    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_scores = {
        class_name: {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1_score": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        }
        for class_name in class_names
    }

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": macro_f1,
        "per_class": per_class_scores,
        "classification_report": report,
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1_score": float(report["macro avg"]["f1-score"]),
            "support": int(report["macro avg"]["support"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1_score": float(report["weighted avg"]["f1-score"]),
            "support": int(report["weighted avg"]["support"]),
        },
        "per_class_f1": {
            class_name: float(score)
            for class_name, score in zip(class_names, per_class_f1.tolist())
        },
    }


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], normalize: bool = True) -> Any:
    """Compute a confusion matrix for classification results.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        normalize: Whether to normalize rows in the resulting matrix.

    Returns:
        Confusion matrix object or array.

    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=labels)

    if not normalize:
        return matrix

    matrix = matrix.astype(np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums
