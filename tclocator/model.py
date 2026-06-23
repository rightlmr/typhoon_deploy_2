"""U-Net keypoint model for full-field TC center localization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""

        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip features, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply upsampling and skip fusion."""

        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TCLocatorUNet(nn.Module):
    """CenterNet-style full-field U-Net with heatmap and offset heads."""

    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 32,
        pad_multiple: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.pad_multiple = pad_multiple

        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)

        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2, kernel_size=1),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        """Initialize heads for sparse-positive heatmaps."""

        final_heatmap = self.heatmap_head[-1]
        if isinstance(final_heatmap, nn.Conv2d):
            nn.init.constant_(final_heatmap.bias, -2.19)
        final_offset = self.offset_head[-1]
        if isinstance(final_offset, nn.Conv2d):
            nn.init.zeros_(final_offset.bias)

    def freeze_encoder(self) -> None:
        """Freeze encoder and bottleneck parameters for AIFS fine-tuning."""

        for module in [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck]:
            for param in module.parameters():
                param.requires_grad = False

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Pad input tensor to the configured multiple."""

        height, width = x.shape[-2:]
        pad_h = (self.pad_multiple - height % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - width % self.pad_multiple) % self.pad_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (height, width)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the locator and return ``heatmap`` and ``offset`` tensors."""

        x, original_shape = self._pad(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        heatmap = torch.sigmoid(self.heatmap_head(d1))
        offset = self.offset_head(d1)
        height, width = original_shape
        return {
            "heatmap": heatmap[..., :height, :width],
            "offset": offset[..., :height, :width],
        }

    def checkpoint_payload(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a serializable checkpoint payload."""

        return {
            "model_state": self.state_dict(),
            "in_channels": self.in_channels,
            "pad_multiple": self.pad_multiple,
            "config": config or {},
        }


def build_model_from_config(config: dict[str, Any]) -> TCLocatorUNet:
    """Construct a model from project config."""

    channels = config.get("channels")
    if not channels:
        raise ValueError("config.channels must be a non-empty list")
    model_cfg = config.get("model", {})
    return TCLocatorUNet(
        len(channels),
        base_channels=int(model_cfg.get("base_channels", 32)),
        pad_multiple=int(model_cfg.get("pad_multiple", 32)),
    )

