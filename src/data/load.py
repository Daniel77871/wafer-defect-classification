"""Load and summarize the WM-811K wafer-map dataset."""

from __future__ import annotations

import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import save_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS: list[str] = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_wm811k(path: str = "data/raw/LSWMD.pkl") -> pd.DataFrame:
    """Load the raw WM-811K DataFrame."""
    _install_legacy_pandas_pickle_aliases()
    dataset_path = _project_path(path)
    try:
        return pd.read_pickle(dataset_path)
    except UnicodeDecodeError:
        with dataset_path.open("rb") as file:
            loaded = pickle.load(file, encoding="latin1")
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(loaded)!r}.")
        return loaded


def _install_legacy_pandas_pickle_aliases() -> None:
    """Support old WM-811K pickles created with pre-1.0 pandas module paths."""
    import pandas.core.indexes as core_indexes
    import pandas.core.indexes.base as core_indexes_base

    sys.modules.setdefault("pandas.indexes", core_indexes)
    sys.modules.setdefault("pandas.indexes.base", core_indexes_base)


def normalize_failure_type(value: Any) -> str:
    """Flatten nested failureType values to a string.

    Empty arrays, missing values, and unlabeled rows return an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        for item in value.ravel():
            normalized = normalize_failure_type(item)
            if normalized:
                return normalized
        return ""
    if isinstance(value, (list, tuple, set)):
        if not value:
            return ""
        for item in value:
            normalized = normalize_failure_type(item)
            if normalized:
                return normalized
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return ""
    return text


def get_labeled_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with valid labels and deterministic label ids.

    The label mapping is saved to `data/processed/label_mapping.json`.
    """
    labels = df["failureType"].map(normalize_failure_type)
    valid_mask = labels.isin(LABELS)

    labeled = df.loc[valid_mask].copy()
    labeled["label_str"] = labels.loc[valid_mask].to_numpy()
    labeled["label_id"] = labeled["label_str"].map(LABEL_TO_ID).astype(int)

    mapping_path = PROJECT_ROOT / "data/processed/label_mapping.json"
    save_json(LABEL_TO_ID, mapping_path)
    return labeled


def compute_wafer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute shape and defect-density statistics for each wafer map."""
    records: list[dict[str, float | int]] = []

    for index, wafer_map in df["waferMap"].items():
        array = np.asarray(wafer_map)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D waferMap at index {index}, got {array.ndim}D.")

        height, width = array.shape
        total_dies = int(np.count_nonzero(array > 0))
        defective_dies = int(np.count_nonzero(array == 2))
        defect_density = defective_dies / total_dies if total_dies else 0.0

        records.append(
            {
                "height": int(height),
                "width": int(width),
                "total_dies": total_dies,
                "defective_dies": defective_dies,
                "defect_density": float(defect_density),
            }
        )

    return pd.DataFrame(records, index=df.index)
