"""Grad-CAM utilities for CNN wafer-map classifiers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
) -> np.ndarray:
    """Return a Grad-CAM heatmap in ``[0, 1]`` with shape ``(H, W)``.

    ``input_tensor`` must contain exactly one sample with shape ``(1, C, H, W)``.
    The caller chooses ``target_layer`` so this function works for both the
    scratch CNN and timm backbones such as MobileNetV3.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must have shape (1, C, H, W).")

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    forward_handle = target_layer.register_forward_hook(forward_hook)

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    try:
        logits = model(input_tensor)
        if target_class < 0 or target_class >= logits.shape[1]:
            raise ValueError(
                f"target_class must be in [0, {logits.shape[1] - 1}], "
                f"got {target_class}."
            )
        logits[0, target_class].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam[0, 0].detach().cpu().numpy()
        heatmap -= float(heatmap.min())
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap /= max_value
        return heatmap.astype(np.float32)
    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
