"""Generic experiment training loop."""

from __future__ import annotations

import json
import gc
import logging
import os
import pickle
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "wafer-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "wafer-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.load import LABELS, get_labeled_subset, load_wm811k
from src.evaluation.metrics import compute_all_metrics
from src.data.split import stratified_split
from src.features.handcrafted import extract_all_features
from src.models.cnn_scratch import SimpleCNN
from src.models.svm_baseline import train_svm
from src.models.transfer import build_transfer_model
from src.training.augmentation import get_train_transforms
from src.training.losses import FocalLoss, compute_class_weights
from src.training.sampler import build_weighted_sampler
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 9
LOGGER = logging.getLogger(__name__)


class WaferArrayDataset(Dataset):
    """Torch dataset backed by preprocessed NumPy arrays."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        transform: Any | None = None,
    ) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[index]


def run_experiment(config: dict) -> dict:
    """Run an experiment from a config dictionary."""
    set_seed(int(config.get("seed", 42)))
    exp_name = config["exp_name"]
    output_dir = ensure_dir(PROJECT_ROOT / "results" / exp_name)

    if config.get("model", {}).get("type") == "svm":
        return _run_svm_experiment(config, output_dir)

    device = _get_device()
    LOGGER.info("Using device: %s", device.type)
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    encoding = data_config.get("encoding", "onehot")

    train_x, train_y = _load_processed_split(encoding, "train")
    val_x, val_y = _load_processed_split(encoding, "val")

    train_loader = _build_train_loader(config, train_x, train_y)
    val_loader = _build_eval_loader(config, val_x, val_y)

    model = _build_model(config, in_channels=train_x.shape[1]).to(device)
    criterion = _build_loss(config, train_y).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 0.001)),
        weight_decay=1e-4,
    )
    n_epochs = int(training_config.get("n_epochs", 50))
    batch_log_interval = int(
        os.environ.get("WAFER_BATCH_LOG_INTERVAL", training_config.get("batch_log_interval", 0))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
    )
    scaler = _make_grad_scaler(device)

    history: list[dict[str, float | int]] = []
    best_macro_f1 = -np.inf
    best_epoch = 0
    patience = int(training_config.get("patience", 10))
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            batch_log_interval,
        )
        val_loss, val_true, val_pred, _ = _evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        val_macro_f1 = float(
            f1_score(val_true, val_pred, average="macro", zero_division=0)
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(model, config, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    del train_loader, val_loader, train_x, train_y, val_x, val_y
    gc.collect()

    test_x, test_y = _load_processed_split(encoding, "test")
    test_loader = _build_eval_loader(config, test_x, test_y)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, y_true, y_pred, y_proba = _evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    _save_training_curves(history, output_dir / "training_curves.png")
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

    results = compute_all_metrics(y_true, y_pred, y_proba, LABELS)
    results.update(
        {
            "exp_name": exp_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_macro_f1),
            "test_loss": float(test_loss),
            "n_epochs_ran": len(history),
            "device": device.type,
        }
    )
    save_json(results, output_dir / "metrics.json")
    return results


def _run_svm_experiment(config: dict, output_dir: Path) -> dict:
    data_config = config.get("data", {})
    raw_path = data_config.get("raw_path", "data/raw/LSWMD.pkl")
    df = get_labeled_subset(load_wm811k(raw_path)).reset_index(drop=True)
    splits = _load_or_create_splits(df, int(config.get("seed", 42)))

    train_x = _extract_features(df, splits["train"])
    val_x = _extract_features(df, splits["val"])
    test_x = _extract_features(df, splits["test"])
    train_y = df.iloc[splits["train"]]["label_id"].to_numpy(dtype=np.int64)
    val_y = df.iloc[splits["val"]]["label_id"].to_numpy(dtype=np.int64)
    test_y = df.iloc[splits["test"]]["label_id"].to_numpy(dtype=np.int64)

    model = train_svm(train_x, train_y, val_x, val_y)
    bundle_path = PROJECT_ROOT / "results/exp_F_svm/svm_model.pkl"
    with bundle_path.open("rb") as file:
        bundle = pickle.load(file)
    scaler = bundle["scaler"]

    test_x_scaled = scaler.transform(test_x)
    y_pred = model.predict(test_x_scaled)
    y_score = model.decision_function(test_x_scaled)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true=test_y,
        y_pred=y_pred,
        y_proba=y_score,
    )
    _save_svm_placeholder_curves(output_dir / "training_curves.png")
    _save_checkpoint_payload({"model_type": "svm"}, output_dir / "best_model.pt")

    results = compute_all_metrics(test_y, y_pred, y_score, LABELS)
    results.update(
        {
            "exp_name": config["exp_name"],
            "best_epoch": 1,
            "best_val_macro_f1": float(bundle["val_accuracy"]),
            "n_epochs_ran": 1,
            "device": "cpu",
            "note": "SVM y_proba stores decision_function scores.",
        }
    )
    save_json(results, output_dir / "metrics.json")
    pd.DataFrame([results]).to_csv(output_dir / "metrics.csv", index=False)
    return results


def _load_processed_split(
    encoding: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = PROJECT_ROOT / f"data/processed/{encoding}_{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    with np.load(path) as data:
        return data["X"].astype(np.float16), data["y"]


def _build_train_loader(
    config: dict,
    x: np.ndarray,
    y: np.ndarray,
) -> DataLoader:
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    sampler_config = config.get("sampler", {})
    dataset = WaferArrayDataset(
        x,
        y,
        transform=get_train_transforms(data_config.get("use_augmentation", False)),
    )
    sampler = None
    shuffle = True
    if sampler_config.get("type", "standard") == "weighted":
        sampler = build_weighted_sampler(y)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 128)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _build_eval_loader(config: dict, x: np.ndarray, y: np.ndarray) -> DataLoader:
    training_config = config.get("training", {})
    dataset = WaferArrayDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _build_model(config: dict, in_channels: int) -> nn.Module:
    model_config = config.get("model", {})
    model_type = model_config.get("type", "cnn_scratch")
    configured_channels = model_config.get("in_channels")
    in_channels = int(configured_channels or in_channels)

    if model_type == "cnn_scratch":
        return SimpleCNN(in_channels=in_channels, num_classes=NUM_CLASSES)
    if model_type == "transfer":
        return build_transfer_model(
            backbone=model_config.get("backbone", "mobilenetv3_small_100"),
            num_classes=NUM_CLASSES,
            pretrained=bool(model_config.get("pretrained", True)),
            in_channels=in_channels,
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _build_loss(config: dict, labels: np.ndarray) -> nn.Module:
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "ce")
    weights = None
    if loss_type == "weighted_ce":
        weights = compute_class_weights(labels, NUM_CLASSES)
        return nn.CrossEntropyLoss(weight=weights)
    if loss_type == "focal":
        if loss_config.get("use_alpha", False):
            weights = compute_class_weights(labels, NUM_CLASSES)
        return FocalLoss(
            gamma=float(loss_config.get("focal_gamma", 2.0)),
            alpha=weights,
        )
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    batch_log_interval: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_index, (x, y) in enumerate(loader, start=1):
        non_blocking = device.type == "cuda"
        x = x.to(device, dtype=torch.float32, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        if batch_log_interval and batch_index % batch_log_interval == 0:
            print(
                f"train_batch={batch_index} loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_parts = []
    y_pred_parts = []
    y_proba_parts = []

    for x, y in loader:
        non_blocking = device.type == "cuda"
        x = x.to(device, dtype=torch.float32, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)
        proba = torch.softmax(logits, dim=1)

        batch_size = x.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true_parts.append(y.cpu().numpy())
        y_pred_parts.append(proba.argmax(dim=1).cpu().numpy())
        y_proba_parts.append(proba.cpu().numpy())

    return (
        total_loss / max(total_samples, 1),
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(y_proba_parts),
    )


def _save_checkpoint(model: nn.Module, config: dict, path: Path) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def _save_checkpoint_payload(payload: dict, path: Path) -> None:
    torch.save(payload, path)


def _save_training_curves(history: list[dict[str, float | int]], path: Path) -> None:
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_macro_f1"], color="#4C78A8")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_svm_placeholder_curves(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, "SVM baseline has no epoch-wise curve.", ha="center")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_device() -> torch.device:
    requested = os.environ.get("WAFER_DEVICE", "auto").lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            "WAFER_DEVICE must be one of: auto, cuda, mps, cpu "
            f"(got {requested!r})"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("WAFER_DEVICE=cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise RuntimeError("WAFER_DEVICE=mps requested, but MPS is unavailable.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_grad_scaler(device: torch.device) -> Any:
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _load_or_create_splits(df: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    split_dir = PROJECT_ROOT / "data/splits"
    paths = {name: split_dir / f"{name}_indices.csv" for name in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return {
            name: pd.read_csv(path)["index"].astype(int).tolist()
            for name, path in paths.items()
        }
    return stratified_split(df, seed=seed)


def _extract_features(df: pd.DataFrame, indices: list[int]) -> np.ndarray:
    features = [
        extract_all_features(wafer)
        for wafer in tqdm(df.iloc[indices]["waferMap"], desc="SVM features")
    ]
    return np.vstack(features).astype(np.float32)
