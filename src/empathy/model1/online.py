"""Continuous hidden-path beliefs from short video clips."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OnlineSeeingConfig:
    channels: tuple[int, ...] = (8, 16, 24)
    representation_size: int = 48
    world_size: int = 12
    sample_count: int = 32


class SharedVideoEncoder3D(nn.Module):
    """Small masked 3D CNN for the visible approach clip."""

    def __init__(self, channels: tuple[int, ...], *, output_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        incoming = 1
        for outgoing in channels:
            layers.extend(
                (
                    nn.Conv3d(incoming, outgoing, kernel_size=3, stride=(1, 2, 2), padding=1),
                    nn.GELU(),
                )
            )
            incoming = outgoing
        self.blocks = nn.Sequential(*layers)
        self.projection = nn.Linear(incoming, output_size)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 1:
            raise ValueError("frames must have shape [batch, time, 1, height, width]")
        if mask.dtype != torch.bool or mask.shape != frames.shape[:2]:
            raise ValueError("mask must be boolean [batch, time]")
        if torch.any(mask.sum(dim=1) < 1):
            raise ValueError("every clip needs at least one visible frame")
        safe = frames.masked_fill(~mask[:, :, None, None, None], 0.0)
        features = self.blocks(safe.permute(0, 2, 1, 3, 4))
        temporal = mask[:, None, :, None, None].to(features.dtype)
        pooled_space = features.mean(dim=(-1, -2))
        pooled = (pooled_space * temporal.squeeze(-1).squeeze(-1)).sum(dim=-1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).to(features.dtype)
        return self.projection(pooled)


@dataclass(frozen=True)
class LatentStateBelief:
    """Distribution over the recurrent state that starts an open-loop rollout."""

    mean: torch.Tensor
    log_scale: torch.Tensor

    def sample(
        self, count: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape[0],
            count,
            self.mean.shape[-1],
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean[:, None] + noise * self.log_scale.clamp(-4.0, 1.0).exp()[:, None]


class StepwiseWorldModel(nn.Module):
    """Video-conditioned latent dynamics that predicts one physical step at a time."""

    def __init__(
        self,
        config: OnlineSeeingConfig,
        *,
        hidden_size: int = 48,
        acceleration_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.acceleration_scale = acceleration_scale
        self.video_encoder = SharedVideoEncoder3D(
            config.channels, output_size=config.representation_size
        )
        self.initializer = nn.Sequential(
            nn.Linear(config.representation_size + 4 + config.world_size, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size * 2),
        )
        self.transition = nn.GRUCell(4 + config.world_size, hidden_size)
        self.acceleration = nn.Linear(hidden_size, 2)

    def initial_belief(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        entry_state: torch.Tensor,
        world_latent: torch.Tensor,
    ) -> LatentStateBelief:
        video = self.video_encoder(frames, mask)
        parameters = self.initializer(torch.cat((video, entry_state, world_latent), dim=-1))
        mean, log_scale = parameters.chunk(2, dim=-1)
        return LatentStateBelief(mean, log_scale.clamp(-4.0, 0.5))

    def rollout(
        self,
        latent_samples: torch.Tensor,
        entry_state: torch.Tensor,
        world_latent: torch.Tensor,
        *,
        steps: int,
        dt: float,
    ) -> torch.Tensor:
        batch, samples, hidden_size = latent_samples.shape
        hidden = latent_samples.reshape(batch * samples, hidden_size)
        state = entry_state[:, None].expand(-1, samples, -1).clone()
        world = world_latent[:, None].expand(-1, samples, -1)
        positions: list[torch.Tensor] = []
        for _ in range(steps):
            transition_input = torch.cat((state, world), dim=-1).reshape(batch * samples, -1)
            hidden = self.transition(transition_input, hidden)
            acceleration = self.acceleration_scale * torch.tanh(
                self.acceleration(hidden).reshape(batch, samples, 2)
            )
            velocity = state[..., 2:] + acceleration * dt
            position = state[..., :2] + velocity * dt
            state = torch.cat((position, velocity), dim=-1)
            positions.append(position)
        return torch.stack(positions, dim=2)
