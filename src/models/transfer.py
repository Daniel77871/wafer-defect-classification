"""Transfer-learning model definitions."""

from __future__ import annotations

import math

import timm
import torch
from torch import nn

SUPPORTED_BACKBONES = {"resnet50", "mobilenetv3_small_100", "efficientnet_b0"}


def build_transfer_model(
    backbone: str = "mobilenetv3_small_100",
    num_classes: int = 9,
    pretrained: bool = True,
    in_channels: int = 3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a timm transfer model for wafer maps."""
    if backbone not in SUPPORTED_BACKBONES:
        supported = ", ".join(sorted(SUPPORTED_BACKBONES))
        raise ValueError(f"Unsupported backbone `{backbone}`. Choose from: {supported}.")
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )
    if in_channels != 3:
        _replace_first_conv(model, in_channels)

    if freeze_backbone:
        _freeze_backbone_except_classifier(model)

    return model


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    conv_name = model.default_cfg.get("first_conv")
    if not conv_name:
        raise ValueError("Could not locate first convolution from timm default_cfg.")

    parent, attribute = _resolve_parent_module(model, conv_name)
    old_conv = getattr(parent, attribute)
    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError(f"`{conv_name}` is not an nn.Conv2d layer.")
    if old_conv.groups != 1:
        raise ValueError("Grouped first convolutions are not supported.")

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.copy_(_adapt_first_conv_weight(old_conv.weight, in_channels))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    setattr(parent, attribute, new_conv)


def _adapt_first_conv_weight(
    weight: torch.Tensor,
    in_channels: int,
) -> torch.Tensor:
    old_channels = weight.shape[1]
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    repeat_count = math.ceil(in_channels / old_channels)
    adapted = weight.repeat(1, repeat_count, 1, 1)[:, :in_channels, :, :]
    return adapted * (old_channels / in_channels)


def _resolve_parent_module(
    model: nn.Module,
    dotted_name: str,
) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _freeze_backbone_except_classifier(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for parameter in classifier.parameters():
            parameter.requires_grad = True
