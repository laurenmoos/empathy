"""Observable-only continual correction for the Seeing model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .online import HiddenPathBelief, OnlineSeeingModel


@dataclass(frozen=True)
class ObservedSeeingEpisode:
    pre_frames: torch.Tensor
    pre_mask: torch.Tensor
    post_frames: torch.Tensor
    post_mask: torch.Tensor
    entry_state: torch.Tensor
    exit_state: torch.Tensor
    departure_positions: torch.Tensor


@dataclass(frozen=True)
class FrozenPreBelief:
    belief: HiddenPathBelief
    samples: torch.Tensor
    paths: torch.Tensor


@dataclass(frozen=True)
class OnlineSeeingMetrics:
    loss: float
    geometric_loss: float
    contrastive_loss: float
    exit_error: float
    belief_spread: float
    effective_samples: float
    updates: int


@dataclass(frozen=True)
class OnlineSeeingResult:
    prior: FrozenPreBelief
    paths: torch.Tensor
    posterior_weights: torch.Tensor
    metrics: OnlineSeeingMetrics


class OnlineSeeingRuntime:
    """Persistent learner for repeated throws within one fixed world."""

    def __init__(
        self,
        model: OnlineSeeingModel,
        *,
        seed: int = 0,
        fast_lr: float = 0.02,
        slow_lr: float = 0.0005,
        replay_capacity: int = 16,
        occluder: tuple[float, float] = (0.38, 0.58),
    ) -> None:
        self.model = model
        self.world_latent = nn.Parameter(torch.zeros(1, model.config.world_size))
        self.fast_optimizer = torch.optim.Adam((self.world_latent,), lr=fast_lr)
        self.slow_optimizer = torch.optim.Adam(model.parameters(), lr=slow_lr)
        self.replay: deque[ObservedSeeingEpisode] = deque(maxlen=replay_capacity)
        self.generator = torch.Generator().manual_seed(seed)
        self.occluder = occluder
        self.updates = 0

    def _world(self, batch: int) -> torch.Tensor:
        return self.world_latent.expand(batch, -1)

    def _belief(self, episode: ObservedSeeingEpisode) -> HiddenPathBelief:
        return self.model.pre_belief(
            episode.pre_frames,
            episode.pre_mask,
            episode.entry_state,
            self._world(episode.pre_frames.shape[0]),
        )

    def begin(self, episode: ObservedSeeingEpisode) -> FrozenPreBelief:
        with torch.no_grad():
            belief = self._belief(episode)
            samples = belief.sample(self.model.config.sample_count, generator=self.generator)
            paths = belief.decode(
                samples,
                episode.entry_state,
                occluder=self.occluder,
                steps=self.model.config.path_steps,
            )
            frozen = HiddenPathBelief(
                belief.mean.detach().clone(), belief.log_scale.detach().clone()
            )
            return FrozenPreBelief(frozen, samples.detach().clone(), paths.detach().clone())

    @staticmethod
    def _sample_error(paths: torch.Tensor, episode: ObservedSeeingEpisode) -> torch.Tensor:
        endpoint = paths[:, :, -1]
        exit_error = (endpoint - episode.exit_state[:, None, :2]).square().sum(dim=-1)
        velocity = paths[:, :, -1] - paths[:, :, -2]
        observed_velocity = episode.exit_state[:, None, 2:4] * 0.08
        velocity_error = (velocity - observed_velocity).square().sum(dim=-1)
        count = episode.departure_positions.shape[1]
        steps = torch.arange(1, count + 1, device=paths.device, dtype=paths.dtype)
        implied = endpoint[:, :, None] + velocity[:, :, None] * steps[None, None, :, None]
        departure_error = (
            implied - episode.departure_positions[:, None]
        ).square().sum(dim=-1).mean(dim=-1)
        return exit_error + 0.35 * velocity_error + 0.5 * departure_error

    def pair_score(
        self, pre_episode: ObservedSeeingEpisode, post_episode: ObservedSeeingEpisode
    ) -> float:
        with torch.no_grad():
            pre = self.model.video_encoder(pre_episode.pre_frames, pre_episode.pre_mask)
            post = self.model.encode_post(post_episode.post_frames, post_episode.post_mask)
            return float(F.cosine_similarity(pre, post).mean())

    def _contrastive_loss(self, episode: ObservedSeeingEpisode) -> torch.Tensor:
        pre = self.model.video_encoder(episode.pre_frames, episode.pre_mask)
        post = self.model.encode_post(episode.post_frames, episode.post_mask)
        matched = F.cosine_similarity(pre, post).mean()
        if not self.replay:
            return 1.0 - matched
        swapped_episode = self.replay[-1]
        swapped = self.model.encode_post(
            swapped_episode.post_frames, swapped_episode.post_mask
        )
        mismatched = F.cosine_similarity(pre, swapped).mean()
        return 1.0 - matched + F.relu(mismatched - matched + 0.1)

    def complete(
        self,
        frozen: FrozenPreBelief,
        episode: ObservedSeeingEpisode,
        *,
        train: bool = True,
    ) -> OnlineSeeingResult:
        frozen_error = self._sample_error(frozen.paths, episode)
        weights = torch.softmax(-frozen_error / 0.01, dim=1)
        geometric_value = float(
            (weights * frozen_error).sum(dim=1).mean().detach()
        )
        contrastive_value = 0.0
        total_value = geometric_value
        if train:
            self.fast_optimizer.zero_grad()
            self.slow_optimizer.zero_grad()
            belief = self._belief(episode)
            # Reuse the pre-exit noise so the update explains the frozen prediction.
            scale = belief.log_scale.clamp(-4.0, 1.0).exp()[:, None]
            noise = (frozen.samples - frozen.belief.mean[:, None]) / (
                frozen.belief.log_scale.exp()[:, None] + 1e-8
            )
            samples = belief.mean[:, None] + noise * scale
            paths = belief.decode(
                samples,
                episode.entry_state,
                occluder=self.occluder,
                steps=self.model.config.path_steps,
            )
            error = self._sample_error(paths, episode)
            geometric = -(torch.logsumexp(-error / 0.01, dim=1) - torch.log(
                torch.tensor(error.shape[1], dtype=error.dtype, device=error.device)
            )).mean() * 0.01
            contrastive = self._contrastive_loss(episode)
            loss = geometric + 0.05 * contrastive
            loss.backward()
            self.fast_optimizer.step()
            self.slow_optimizer.step()
            self.updates += 1
            geometric_value = float(geometric.detach())
            contrastive_value = float(contrastive.detach())
            total_value = float(loss.detach())
            self.replay.append(episode)
        weighted_endpoint = (weights[..., None] * frozen.paths[:, :, -1]).sum(dim=1)
        exit_error = torch.linalg.vector_norm(
            weighted_endpoint - episode.exit_state[:, :2], dim=-1
        ).mean()
        spread = frozen.paths[:, :, -1, 1].std(dim=1).mean()
        effective = (1.0 / weights.square().sum(dim=1)).mean()
        metrics = OnlineSeeingMetrics(
            loss=total_value,
            geometric_loss=geometric_value,
            contrastive_loss=contrastive_value,
            exit_error=float(exit_error),
            belief_spread=float(spread),
            effective_samples=float(effective),
            updates=self.updates,
        )
        return OnlineSeeingResult(frozen, frozen.paths, weights.detach(), metrics)
