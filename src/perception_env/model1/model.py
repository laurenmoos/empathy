"""Small convolutional/recurrent trajectory mixture model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .types import TrajectoryMixture


class FrameEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        incoming = 1
        for outgoing in channels:
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                ]
            )
            incoming = outgoing
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_size = channels[-1]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(frames)).flatten(1)


class ClipStateEncoder(nn.Module):
    def __init__(self, frame_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(frame_size + 1, hidden_size, batch_first=True)

    def forward(
        self, frame_features: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("clip mask must be boolean [batch, time]")
        lengths = mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("every clip needs at least one valid frame")
        expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("clip masks must be right padded")
        sequence = torch.cat((frame_features, times.unsqueeze(-1)), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class TrajectoryMixtureHead(nn.Module):
    def __init__(self, hidden_size: int, modes: int, prediction_steps: int) -> None:
        super().__init__()
        self.modes = modes
        self.prediction_steps = prediction_steps
        self.positions = nn.Linear(hidden_size, modes * prediction_steps * 2)
        self.logits = nn.Linear(hidden_size, modes)

    def forward(self, encoding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.sigmoid(self.positions(encoding)).reshape(
            encoding.shape[0], self.modes, self.prediction_steps, 2
        )
        return positions, self.logits(encoding)


class Model1(nn.Module):
    def __init__(self, config: ModelConfig, *, prediction_steps: int) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(config.frame_channels)
        self.clip_encoder = ClipStateEncoder(
            self.frame_encoder.output_size, config.hidden_size
        )
        self.state_head = nn.Linear(config.hidden_size, 4)
        self.trajectory_head = TrajectoryMixtureHead(
            config.hidden_size, config.modes, prediction_steps
        )

    def forward(
        self, frames: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> TrajectoryMixture:
        if frames.ndim != 5 or frames.shape[2] != 1:
            raise ValueError("frames must have shape [batch, time, 1, height, width]")
        batch, steps = frames.shape[:2]
        if times.shape != (batch, steps) or mask.shape != (batch, steps):
            raise ValueError("times and mask must match the clip batch/time shape")
        features = self.frame_encoder(frames.reshape(batch * steps, *frames.shape[2:]))
        features = features.reshape(batch, steps, -1)
        encoding = self.clip_encoder(features, times, mask)
        positions, logits = self.trajectory_head(encoding)
        prediction = TrajectoryMixture(
            positions=positions,
            logits=logits,
            entry_state=torch.sigmoid(self.state_head(encoding)),
        )
        prediction.validate()
        return prediction
