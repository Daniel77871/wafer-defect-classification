"""Resize and encode wafer maps for model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.transform import resize

TARGET_SIZE = (64, 64)
ENCODINGS = {"gray", "onehot", "mask"}


def resize_wafer_map(
    wafer: np.ndarray,
    size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Resize a wafer map with nearest-neighbor interpolation."""
    resized = resize(
        np.asarray(wafer),
        size,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return resized.astype(np.int8)


def encode_one_hot(wafer_resized: np.ndarray) -> np.ndarray:
    """Convert a resized wafer map into a 3-channel one-hot tensor."""
    wafer_int = np.asarray(wafer_resized, dtype=np.int8)
    return np.stack(
        [(wafer_int == value).astype(np.float32) for value in range(3)],
        axis=0,
    )


def build_arrays(
    df: pd.DataFrame,
    indices: list[int],
    encoding: str = "onehot",
) -> tuple[np.ndarray, np.ndarray]:
    """Build model arrays for selected row positions.

    Returns X with shape `(N, C, H, W)` and y with shape `(N,)`.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {sorted(ENCODINGS)}, got {encoding}.")
    if "label_id" not in df.columns:
        raise KeyError("DataFrame must contain a `label_id` column.")

    selected = df.iloc[indices]
    channels = 3 if encoding == "onehot" else 1
    x = np.empty(
        (len(selected), channels, TARGET_SIZE[0], TARGET_SIZE[1]),
        dtype=np.float32,
    )
    y = selected["label_id"].to_numpy(dtype=np.int64)

    for output_index, wafer in enumerate(selected["waferMap"]):
        resized = resize_wafer_map(wafer)
        if encoding == "onehot":
            x[output_index] = encode_one_hot(resized)
        elif encoding == "gray":
            x[output_index, 0] = resized.astype(np.float32) / 2.0
        elif encoding == "mask":
            x[output_index, 0] = (resized == 2).astype(np.float32)

    return x, y
