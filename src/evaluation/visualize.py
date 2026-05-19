"""Experiment visualization helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wafer-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wafer-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from src.utils.io import ensure_dir


def plot_confusion_matrix(
    cm,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot an annotated confusion-matrix heatmap."""
    matrix = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums != 0,
        )
        fmt = ".2f"
        colorbar_label = "Recall-normalized count"
    else:
        fmt = "d"
        colorbar_label = "Count"
        matrix = matrix.astype(int)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": colorbar_label},
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _save_if_requested(fig, save_path)
    return fig


def plot_per_class_f1_bars(
    metrics: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-class F1 bars sorted from weakest to strongest."""
    rows = [
        {
            "class_name": class_name,
            "f1": class_metrics["f1"],
            "support": class_metrics["support"],
        }
        for class_name, class_metrics in metrics["per_class"].items()
    ]
    data = pd.DataFrame(rows).sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(data=data, x="f1", y="class_name", ax=ax, color="#4C78A8")
    for patch, (_, row) in zip(ax.patches, data.iterrows()):
        ax.text(
            min(patch.get_width() + 0.02, 1.02),
            patch.get_y() + patch.get_height() / 2,
            f"n={int(row['support'])}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1.08)
    ax.set_title("Per-Class F1")
    ax.set_xlabel("F1")
    ax.set_ylabel("Class")
    _save_if_requested(fig, save_path)
    return fig


def plot_pr_curves(
    y_true,
    y_proba,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one precision-recall curve per class."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    labels = list(range(len(class_names)))
    y_binary = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for class_id, class_name in enumerate(class_names):
        if y_binary[:, class_id].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(
            y_binary[:, class_id],
            y_proba[:, class_id],
        )
        ap = average_precision_score(y_binary[:, class_id], y_proba[:, class_id])
        ax.plot(recall, precision, linewidth=1.4, label=f"{class_name} AP={ap:.2f}")

    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    _save_if_requested(fig, save_path)
    return fig


def plot_training_curves(
    history: dict | list[dict[str, Any]] | pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train/val loss and validation macro-F1."""
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    if "train_loss" in history_df:
        axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    if "val_loss" in history_df:
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "val_macro_f1" in history_df:
        axes[1].plot(history_df["epoch"], history_df["val_macro_f1"])
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1)

    _save_if_requested(fig, save_path)
    return fig


def _save_if_requested(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    output_path = Path(save_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
