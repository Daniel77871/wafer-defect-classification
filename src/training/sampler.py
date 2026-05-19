"""Sampling helpers for imbalanced classes."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Build an inverse-frequency weighted sampler."""
    labels_int = labels.astype(np.int64)
    counts = np.bincount(labels_int)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot build sampler; missing classes: {missing}")

    class_weights = 1.0 / counts
    sample_weights = class_weights[labels_int]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels_int),
        replacement=True,
    )
