"""Generate Grad-CAM and error-case figures for the best CNN experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wafer-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wafer-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import nn

from src.data.load import LABELS, LABEL_TO_ID
from src.evaluation.gradcam import generate_gradcam
from src.models.cnn_scratch import SimpleCNN
from src.models.transfer import build_transfer_model
from src.utils.io import ensure_dir

CNN_EXPERIMENTS = [
    "exp_A_cnn_scratch",
    "exp_B_transfer",
    "exp_C_class_weight",
    "exp_D_focal_loss",
    "exp_E_augmentation",
]
WAFER_CMAP = ListedColormap(["#141414", "#D8D8D8", "#D62728"])


@dataclass(frozen=True)
class Example:
    """Selected test example for visualization."""

    index: int
    true_id: int
    pred_id: int
    confidence: float


@dataclass(frozen=True)
class AnalysisContext:
    """Loaded model, predictions, and test arrays for explainability figures."""

    exp_name: str
    metrics: dict[str, Any]
    model: nn.Module
    target_layer: nn.Module
    x_test: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figure-dir", default="results/figures/experiments")
    parser.add_argument(
        "--device",
        default=os.environ.get("WAFER_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_dir = ensure_dir(_project_path(args.figure_dir))
    context = _load_analysis_context(args)

    _save_gradcam_figure(context, figure_dir)
    _save_confusion_pair_figures(context, figure_dir)
    _save_label_noise_figure(context, figure_dir)


def _load_analysis_context(args: argparse.Namespace) -> AnalysisContext:
    results_dir = _project_path(args.results_dir)
    exp_name, metrics = _select_best_cnn_experiment(results_dir)
    print(
        f"best_cnn={exp_name} macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )

    exp_dir = results_dir / exp_name
    checkpoint = _load_checkpoint(exp_dir / "best_model.pt")
    config = checkpoint["config"]
    encoding = config.get("data", {}).get("encoding", "onehot")
    x_test, y_test = _load_test_arrays(encoding)
    y_true, y_pred, y_proba = _load_predictions(exp_dir / "test_predictions.npz")
    _validate_prediction_shapes(x_test, y_test, y_true, y_pred, y_proba)

    device = _resolve_device(args.device)
    print(f"device={device.type}", flush=True)
    model = _load_model(checkpoint, config, x_test.shape[1], device)
    target_layer = _find_target_layer(model)

    return AnalysisContext(
        exp_name=exp_name,
        metrics=metrics,
        model=model,
        target_layer=target_layer,
        x_test=x_test,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        device=device,
    )


def _save_gradcam_figure(context: AnalysisContext, figure_dir: Path) -> None:
    gradcam_path = figure_dir / "gradcam_per_class.png"
    _plot_gradcam_per_class(
        model=context.model,
        target_layer=context.target_layer,
        x_test=context.x_test,
        y_true=context.y_true,
        y_pred=context.y_pred,
        y_proba=context.y_proba,
        device=context.device,
        save_path=gradcam_path,
    )
    print(f"saved {gradcam_path}", flush=True)


def _save_confusion_pair_figures(context: AnalysisContext, figure_dir: Path) -> None:
    for true_id, pred_id, _count in _top_confusion_pairs(context.metrics, top_k=2):
        pair_path = figure_dir / (
            f"confusion_pair_{_safe_name(LABELS[true_id])}"
            f"_vs_{_safe_name(LABELS[pred_id])}.png"
        )
        _plot_confusion_pair(
            x_test=context.x_test,
            y_true=context.y_true,
            y_pred=context.y_pred,
            y_proba=context.y_proba,
            true_id=true_id,
            pred_id=pred_id,
            save_path=pair_path,
        )
        print(f"saved {pair_path}", flush=True)


def _save_label_noise_figure(context: AnalysisContext, figure_dir: Path) -> None:
    mislabeled_path = figure_dir / "likely_mislabeled_none.png"
    _plot_likely_mislabeled_none(
        x_test=context.x_test,
        y_true=context.y_true,
        y_pred=context.y_pred,
        y_proba=context.y_proba,
        save_path=mislabeled_path,
    )
    print(f"saved {mislabeled_path}", flush=True)


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _select_best_cnn_experiment(results_dir: Path) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for exp_name in CNN_EXPERIMENTS:
        exp_dir = results_dir / exp_name
        metric_path = exp_dir / "metrics.json"
        checkpoint_path = exp_dir / "best_model.pt"
        prediction_path = exp_dir / "test_predictions.npz"
        if not metric_path.exists():
            continue
        missing = [
            str(path)
            for path in (checkpoint_path, prediction_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{exp_name} has metrics.json but is missing: {missing}"
            )
        with metric_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if "macro_f1" not in metrics:
            raise KeyError(f"{metric_path} is missing `macro_f1`.")
        candidates.append((exp_name, metrics))

    if not candidates:
        raise FileNotFoundError(
            "No completed CNN experiments found under results/exp_A through exp_E."
        )
    return max(candidates, key=lambda item: float(item[1]["macro_f1"]))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint is missing config: {path}")
    return checkpoint


def _load_test_arrays(encoding: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROJECT_ROOT / f"data/processed/{encoding}_test.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed test split: {path}")
    with np.load(path) as data:
        return data["X"], data["y"].astype(np.int64)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            data["y_true"].astype(np.int64),
            data["y_pred"].astype(np.int64),
            data["y_proba"].astype(np.float32),
        )


def _validate_prediction_shapes(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    n_samples = len(y_true)
    if x_test.shape[0] != n_samples or y_test.shape[0] != n_samples:
        raise ValueError("Processed test arrays and predictions have different sizes.")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")
    if y_proba.shape != (n_samples, len(LABELS)):
        raise ValueError(
            f"Expected y_proba shape {(n_samples, len(LABELS))}, got {y_proba.shape}."
        )
    if not np.array_equal(y_test, y_true):
        raise ValueError("Processed test labels do not match saved predictions.")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _load_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    model_config = dict(config.get("model", {}))
    model_type = model_config.get("type", "cnn_scratch")
    configured_channels = model_config.get("in_channels")
    in_channels = int(configured_channels or in_channels)

    if model_type == "cnn_scratch":
        model = SimpleCNN(in_channels=in_channels, num_classes=len(LABELS))
    elif model_type == "transfer":
        model = build_transfer_model(
            backbone=model_config.get("backbone", "mobilenetv3_small_100"),
            num_classes=len(LABELS),
            pretrained=False,
            in_channels=in_channels,
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        )
    else:
        raise ValueError(f"Unsupported CNN model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _find_target_layer(model: nn.Module) -> nn.Module:
    if isinstance(model, SimpleCNN):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "blocks"):
        return model.blocks[-1]

    conv_layers = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")
    return conv_layers[-1]


def _plot_gradcam_per_class(
    model: nn.Module,
    target_layer: nn.Module,
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    device: torch.device,
    save_path: Path,
) -> None:
    selections = _build_gradcam_selections(y_true, y_pred, y_proba)
    fig, axes = plt.subplots(len(LABELS), 6, figsize=(14, 18))
    column_titles = ["C1", "C2", "C3", "W1", "W2", "W3"]
    _draw_gradcam_grid(
        axes=axes,
        selections=selections,
        model=model,
        target_layer=target_layer,
        x_test=x_test,
        column_titles=column_titles,
        device=device,
    )
    _format_gradcam_grid(fig, save_path)


def _build_gradcam_selections(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[int, list[Example | None]]:
    confidence = _prediction_confidence(y_pred, y_proba)
    return {
        class_id: _select_correct_and_wrong(
            class_id,
            y_true,
            y_pred,
            confidence,
        )
        for class_id in range(len(LABELS))
    }


def _draw_gradcam_grid(
    axes: np.ndarray,
    selections: dict[int, list[Example | None]],
    model: nn.Module,
    target_layer: nn.Module,
    x_test: np.ndarray,
    column_titles: list[str],
    device: torch.device,
) -> None:
    for class_id, class_name in enumerate(LABELS):
        examples = selections[class_id]
        for column, example in enumerate(examples):
            ax = axes[class_id, column]
            if example is None:
                ax.axis("off")
                continue
            _draw_gradcam_cell(
                ax=ax,
                model=model,
                target_layer=target_layer,
                x=x_test[example.index],
                example=example,
                column_label=column_titles[column],
                device=device,
            )
        axes[class_id, 0].set_ylabel(class_name, fontsize=9)


def _format_gradcam_grid(fig: plt.Figure, save_path: Path) -> None:
    fig.suptitle("Grad-CAM: Most Confident Correct and Wrong Predictions", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.02)
    fig.subplots_adjust(hspace=0.52, wspace=0.16)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _select_correct_and_wrong(
    class_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> list[Example | None]:
    correct_mask = (y_true == class_id) & (y_pred == class_id)
    wrong_mask = (y_true == class_id) & (y_pred != class_id)
    correct = _top_examples(correct_mask, y_true, y_pred, confidence, top_k=3)
    wrong = _top_examples(wrong_mask, y_true, y_pred, confidence, top_k=3)
    return _pad_examples(correct, 3) + _pad_examples(wrong, 3)


def _top_examples(
    mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    top_k: int,
) -> list[Example]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    ranked = indices[np.argsort(confidence[indices])[::-1]][:top_k]
    return [
        Example(
            index=int(index),
            true_id=int(y_true[index]),
            pred_id=int(y_pred[index]),
            confidence=float(confidence[index]),
        )
        for index in ranked
    ]


def _pad_examples(examples: list[Example], size: int) -> list[Example | None]:
    return examples + [None] * max(size - len(examples), 0)


def _draw_gradcam_cell(
    ax: plt.Axes,
    model: nn.Module,
    target_layer: nn.Module,
    x: np.ndarray,
    example: Example,
    column_label: str,
    device: torch.device,
) -> None:
    input_tensor = torch.from_numpy(x[None]).to(device=device, dtype=torch.float32)
    heatmap = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=example.pred_id,
        target_layer=target_layer,
    )
    _draw_wafer(ax, x)
    ax.imshow(heatmap, cmap="magma", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(
        f"{column_label} P:{LABELS[example.pred_id]}\nconf={example.confidence:.2f}",
        fontsize=7,
    )


def _top_confusion_pairs(metrics: dict[str, Any], top_k: int) -> list[tuple[int, int, int]]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)

    pairs: list[tuple[int, int, int]] = []
    flat_indices = np.argsort(matrix.ravel())[::-1]
    for flat_index in flat_indices:
        count = int(matrix.ravel()[flat_index])
        if count <= 0:
            break
        true_id, pred_id = np.unravel_index(flat_index, matrix.shape)
        pairs.append((int(true_id), int(pred_id), count))
        if len(pairs) >= top_k:
            break

    if len(pairs) < top_k:
        raise ValueError("Fewer than two non-zero confusion pairs were found.")
    return pairs


def _plot_confusion_pair(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    true_id: int,
    pred_id: int,
    save_path: Path,
) -> None:
    confidence = _prediction_confidence(y_pred, y_proba)
    mask = (y_true == true_id) & (y_pred == pred_id)
    examples = _pad_examples(
        _top_examples(mask, y_true, y_pred, confidence, top_k=10),
        10,
    )

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for ax, example in zip(axes.ravel(), examples):
        if example is None:
            ax.axis("off")
            continue
        _draw_wafer(ax, x_test[example.index])
        ax.set_title(
            f"T:{LABELS[example.true_id]} P:{LABELS[example.pred_id]}\n"
            f"conf={example.confidence:.2f}",
            fontsize=8,
        )
    fig.suptitle(
        f"Confusion Pair: {LABELS[true_id]} misclassified as {LABELS[pred_id]}",
        y=0.97,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.04)
    fig.subplots_adjust(hspace=0.42, wspace=0.14)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_likely_mislabeled_none(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
) -> None:
    none_id = LABEL_TO_ID["none"]
    confidence = _prediction_confidence(y_pred, y_proba)
    mask = (y_true == none_id) & (y_pred != none_id) & (confidence > 0.95)
    candidate_count = int(mask.sum())
    examples = _pad_examples(
        _top_examples(mask, y_true, y_pred, confidence, top_k=16),
        16,
    )

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, example in zip(axes.ravel(), examples):
        if example is None:
            ax.axis("off")
            continue
        _draw_wafer(ax, x_test[example.index])
        ax.set_title(
            f"T:none P:{LABELS[example.pred_id]}\nconf={example.confidence:.2f}",
            fontsize=8,
        )
    title = "Likely Mislabeled `none`: Confident Defect Predictions"
    fig.suptitle(f"{title} ({candidate_count} found, conf > 0.95)", y=0.98)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.03)
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prediction_confidence(y_pred: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    return y_proba[np.arange(len(y_pred)), y_pred]


def _draw_wafer(ax: plt.Axes, x: np.ndarray) -> None:
    image = _wafer_image(x)
    ax.imshow(image, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _wafer_image(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected wafer tensor shape (C, H, W), got {x.shape}.")
    if x.shape[0] == 3:
        return x.argmax(axis=0)
    if x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Unsupported channel count: {x.shape[0]}")


def _safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


if __name__ == "__main__":
    main()
