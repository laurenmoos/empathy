"""A CNN whose complete production input is a belief history and validity mask."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import WorldPrediction


class InverseWorldCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks = []
        for before, after in ((12, 32), (32, 64), (64, 64)):
            blocks.extend(
                [
                    nn.Conv2d(before, after, 3, stride=2, padding=1),
                    nn.GroupNorm(8, after),
                    nn.ReLU(),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.hidden = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU()
        )
        self.activity_head = nn.Linear(128, 5)
        self.parameter_head = nn.Linear(128, 7)

    def forward(self, beliefs: torch.Tensor, history_mask: torch.Tensor) -> WorldPrediction:
        if (
            beliefs.ndim != 5
            or beliefs.shape[0] < 1
            or beliefs.shape[1:3] != (4, 3)
            or min(beliefs.shape[-2:]) < 8
            or not beliefs.is_floating_point()
        ):
            raise ValueError("beliefs must be floating [N,4,3,H,W], H,W>=8")
        if history_mask.dtype != torch.bool or history_mask.shape != beliefs.shape[:2]:
            raise ValueError("history mask must be boolean [N,4]")
        if not history_mask.any(dim=1).all() or (history_mask[:, :-1] & ~history_mask[:, 1:]).any():
            raise ValueError("history mask must be a nonempty suffix")
        values = torch.where(history_mask[:, :, None, None, None], beliefs, 0.0)
        if not torch.isfinite(values).all():
            raise ValueError("valid beliefs must be finite")
        if ((values[:, :, :2] < 0) | (values[:, :, :2] > 1)).any():
            raise ValueError("pre/post beliefs must lie in [0,1]")
        if not torch.allclose(
            values[:, :, 2], values[:, :, 1] - values[:, :, 0], atol=1e-6, rtol=0
        ):
            raise ValueError("belief difference must equal post minus pre")
        features = self.hidden(self.encoder(values.flatten(1, 2)))
        return WorldPrediction(
            self.activity_head(features), torch.sigmoid(self.parameter_head(features))
        )
