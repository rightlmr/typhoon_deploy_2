"""CenterNet-style heatmap and offset losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def centernet_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalty-reduced focal loss used for Gaussian keypoint heatmaps."""

    if target.ndim == 3:
        target = target.unsqueeze(1)
    pred = pred.clamp(min=eps, max=1.0 - eps)
    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)
    neg_weights = torch.pow(1.0 - target, beta)

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask
    num_pos = pos_mask.float().sum()
    if num_pos > 0:
        return (pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)
    return neg_loss.mean()


def offset_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 offset loss evaluated only at positive center pixels."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    denom = mask.sum().clamp(min=1.0)
    return loss / denom


@dataclass(frozen=True)
class LossConfig:
    """Loss hyperparameters."""

    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    lambda_offset: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "LossConfig":
        """Build from a config mapping."""

        loss_cfg = config.get("loss", {})
        return cls(
            focal_alpha=float(loss_cfg.get("focal_alpha", 2.0)),
            focal_beta=float(loss_cfg.get("focal_beta", 4.0)),
            lambda_offset=float(loss_cfg.get("lambda_offset", 1.0)),
        )


class TCLocatorLoss(torch.nn.Module):
    """Combined heatmap focal loss and masked offset L1 loss."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Calculate total loss and components."""

        heatmap_loss = centernet_focal_loss(
            outputs["heatmap"],
            batch["heatmap"],
            alpha=self.config.focal_alpha,
            beta=self.config.focal_beta,
        )
        offset_loss = offset_l1_loss(outputs["offset"], batch["offset"], batch["mask"])
        total = heatmap_loss + self.config.lambda_offset * offset_loss
        return {"loss": total, "heatmap_loss": heatmap_loss, "offset_loss": offset_loss}

