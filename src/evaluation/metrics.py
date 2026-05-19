"""Classification metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.load import LABELS


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Compute aggregate, per-class, and confusion-matrix metrics."""
    if class_names is None:
        class_names = LABELS
    num_classes = len(class_names)
    label_ids = list(range(num_classes))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {
        class_names[class_id]: {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in label_ids
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": _safe_weighted_average(f1, support),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_ids)),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
    }


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() == 0:
        return 0.0
    return float(np.average(values, weights=weights))
