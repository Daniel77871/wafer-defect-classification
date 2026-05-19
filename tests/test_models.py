"""Model and handcrafted-feature smoke tests."""

from __future__ import annotations

import numpy as np
import torch

from src.features.handcrafted import extract_geometric_features
from src.models.cnn_scratch import SimpleCNN


def test_geometric_features_on_empty_wafer() -> None:
    """No defects -> regionprops returns empty list -> must not crash."""
    wafer = np.zeros((64, 64), dtype=np.int8)
    g = extract_geometric_features(wafer)
    assert g.shape == (7,)
    assert not np.any(np.isnan(g))
    assert np.allclose(g, 0)


def test_simple_cnn_1ch_with_loss_backward() -> None:
    """End-to-end: forward + loss + backward, single-channel input."""
    model = SimpleCNN(in_channels=1, num_classes=9)
    x = torch.randn(4, 1, 64, 64)
    y = torch.tensor([0, 1, 2, 8])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
    assert has_grad, "no gradients flowed"
