"""Loss helpers for model training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to mean 1."""
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute weights; missing classes: {missing}")

    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
