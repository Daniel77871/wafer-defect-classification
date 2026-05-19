"""Training-time augmentation for discrete wafer maps."""

from __future__ import annotations

import random

import torch
from torchvision import transforms


class RandomDiscreteRotation:
    """Rotate tensors by one of the physically meaningful right angles."""

    def __init__(self, angles: tuple[int, ...] = (90, 180, 270)) -> None:
        self.angles = angles

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        angle = random.choice(self.angles)
        return torch.rot90(tensor, k=angle // 90, dims=(-2, -1))


def get_train_transforms(use_augmentation: bool):
    """Return identity or discrete wafer-map augmentations."""
    if not use_augmentation:
        return transforms.Lambda(lambda tensor: tensor)

    return transforms.Compose(
        [
            RandomDiscreteRotation(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
