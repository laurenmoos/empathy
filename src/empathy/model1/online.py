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
    path_steps: int = 16


class SharedVideoEncoder3D(nn.Module):
    """Small masked 3D CNN shared by clips before and after occlusion."""

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
class HiddenPathBelief:
    """Diagonal Gaussian in unconstrained cubic-Bezier coordinates."""

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

    def decode(
        self,
        samples: torch.Tensor,
        entry_state: torch.Tensor,
        *,
        occluder: tuple[float, float],
        steps: int,
    ) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 6:
            raise ValueError("samples must have shape [batch, samples, 6]")
        p0, p1, p2, p3, _ = self._geometry(samples, entry_state, occluder)
        t = torch.linspace(0.0, 1.0, steps, device=samples.device, dtype=samples.dtype)
        t = t.view(1, 1, steps, 1)
        omt = 1.0 - t
        return (
            omt**3 * p0[:, :, None]
            + 3.0 * omt**2 * t * p1[:, :, None]
            + 3.0 * omt * t**2 * p2[:, :, None]
            + t**3 * p3[:, :, None]
        )

    @staticmethod
    def _geometry(
        samples: torch.Tensor,
        entry_state: torch.Tensor,
        occluder: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = samples.sigmoid()
        left, right = occluder
        width = right - left
        p0 = entry_state[:, None, :2].expand(-1, samples.shape[1], -1)
        p1 = torch.stack(
            (left + width * (0.15 + 0.35 * values[..., 0]), values[..., 1]), dim=-1
        )
        p2 = torch.stack(
            (left + width * (0.55 + 0.35 * values[..., 2]), values[..., 3]), dim=-1
        )
        p3 = torch.stack((torch.full_like(values[..., 4], right), values[..., 4]), dim=-1)
        duration = 0.2 + 1.8 * values[..., 5]
        return p0, p1, p2, p3, duration

    def implied_exit_state(
        self,
        samples: torch.Tensor,
        entry_state: torch.Tensor,
        *,
        occluder: tuple[float, float],
    ) -> torch.Tensor:
        """Return exit position and physical velocity, independent of draw resolution."""
        _, _, p2, p3, duration = self._geometry(samples, entry_state, occluder)
        velocity = 3.0 * (p3 - p2) / duration[..., None]
        return torch.cat((p3, velocity), dim=-1)


class OnlineSeeingModel(nn.Module):
    def __init__(self, config: OnlineSeeingConfig) -> None:
        super().__init__()
        self.config = config
        self.video_encoder = SharedVideoEncoder3D(
            config.channels, output_size=config.representation_size
        )
        self.prior = nn.Sequential(
            nn.Linear(config.representation_size + 4 + config.world_size, 64),
            nn.GELU(),
            nn.Linear(64, 12),
        )

    def encode_post(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.video_encoder(frames, mask)

    def pre_belief(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        entry_state: torch.Tensor,
        world_latent: torch.Tensor,
    ) -> HiddenPathBelief:
        encoded = self.video_encoder(frames, mask)
        parameters = self.prior(torch.cat((encoded, entry_state, world_latent), dim=-1))
        mean, log_scale = parameters.chunk(2, dim=-1)
        return HiddenPathBelief(mean=mean, log_scale=log_scale.clamp(-4.0, 1.0))
