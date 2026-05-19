"""Stratified split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import LABELS, PROJECT_ROOT
from src.utils.io import ensure_dir


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label_id",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Stratified train/val/test split.

    Returns integer row positions into the provided DataFrame and saves split
    index files plus a per-class split summary under `data/splits/`.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")
    if len(ratios) != 3:
        raise ValueError("ratios must be a train/val/test tuple.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.4f}.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be positive.")

    train_ratio, val_ratio, test_ratio = ratios
    positions = np.arange(len(df))
    labels = df[label_col].to_numpy()

    train_idx, temp_idx = train_test_split(
        positions,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = labels[temp_idx]
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction,
        random_state=seed,
        stratify=temp_labels,
    )

    splits = {
        "train": sorted(int(index) for index in train_idx),
        "val": sorted(int(index) for index in val_idx),
        "test": sorted(int(index) for index in test_idx),
    }
    _save_split_indices(splits)
    _save_split_summary(df, splits, label_col)
    return splits


def _save_split_indices(splits: dict[str, list[int]]) -> None:
    split_dir = ensure_dir(PROJECT_ROOT / "data/splits")
    for split_name, indices in splits.items():
        pd.DataFrame({"index": indices}).to_csv(
            split_dir / f"{split_name}_indices.csv",
            index=False,
        )


def _save_split_summary(
    df: pd.DataFrame,
    splits: dict[str, list[int]],
    label_col: str,
) -> None:
    rows: list[dict[str, int | str]] = []
    label_lookup = _label_lookup(df, label_col)

    for label_id in sorted(df[label_col].unique()):
        row: dict[str, int | str] = {
            "label_id": int(label_id),
            "label_str": label_lookup.get(int(label_id), LABELS[int(label_id)]),
        }
        total = 0
        for split_name, indices in splits.items():
            count = int((df.iloc[indices][label_col] == label_id).sum())
            row[split_name] = count
            total += count
        row["total"] = total
        rows.append(row)

    output_path = Path(PROJECT_ROOT / "data/splits/split_summary.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _label_lookup(df: pd.DataFrame, label_col: str) -> dict[int, str]:
    if "label_str" not in df.columns:
        return {}
    lookup = (
        df[[label_col, "label_str"]]
        .drop_duplicates()
        .set_index(label_col)["label_str"]
        .to_dict()
    )
    return {int(label_id): str(label_str) for label_id, label_str in lookup.items()}
