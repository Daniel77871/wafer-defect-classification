"""Reproducibility helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random sources."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch is not installed; skipped torch seeding.")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
