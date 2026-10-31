"""Masked objectives for multimodal paths and belief assimilation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import LossConfig
from .types import BeliefPrediction, Model1Batch, TrajectoryMixture


@dataclass(frozen=True)
class Model1LossTerms:
    total: torch.Tensor
    trajectory: torch.Tensor
    mode: torch.Tensor
    state: torch.Tensor
    pre_belief: torch.Tensor
    post_belief: torch.Tensor


def _trajectory_terms(
    prediction: TrajectoryMixture,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction.validate()
    if targets.shape != (
        prediction.positions.shape[0],
        prediction.positions.shape[2],
        2,
    ):
        raise ValueError("target positions do not match prediction shape")
    if mask.shape != targets.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("target mask must be boolean [batch, steps]")
    counts = mask.sum(dim=1)
    if torch.any(counts < 1):
        raise ValueError("every example needs at least one valid target step")
    distances = torch.linalg.vector_norm(
        prediction.positions - targets[:, None], dim=-1
    )
    average = (
        distances * mask[:, None].to(distances.dtype)
    ).sum(dim=-1) / counts[:, None].to(distances.dtype)
    best_values, best_indices = average.min(dim=1)
    mode_loss = nn.functional.cross_entropy(prediction.logits, best_indices.detach())
    return best_values.mean(), mode_loss


def _belief_bce(prediction: BeliefPrediction, target: torch.Tensor) -> torch.Tensor:
    prediction.validate()
    if prediction.collapsed.shape != target.shape:
        raise ValueError("belief target shape does not match prediction")
    if not torch.isfinite(target).all():
        raise ValueError("belief target must be finite")
    values = prediction.collapsed.clamp(1.0e-6, 1.0 - 1.0e-6)
    return nn.functional.binary_cross_entropy(values, target.to(values.dtype))


def model1_loss(
    pre_prediction: TrajectoryMixture,
    post_prediction: TrajectoryMixture,
    pre_belief: BeliefPrediction,
    post_belief: BeliefPrediction,
    batch: Model1Batch,
    config: LossConfig,
) -> Model1LossTerms:
    pre_trajectory, pre_mode = _trajectory_terms(
        pre_prediction, batch.target_positions, batch.target_mask
    )
    post_trajectory, post_mode = _trajectory_terms(
        post_prediction, batch.target_positions, batch.target_mask
    )
    trajectory = config.trajectory_weight * (pre_trajectory + post_trajectory) / 2
    mode = config.mode_weight * (pre_mode + post_mode) / 2
    state_raw = (
        nn.functional.mse_loss(pre_prediction.entry_state, batch.entry_state)
        + nn.functional.mse_loss(post_prediction.entry_state, batch.entry_state)
    ) / 2
    state = config.state_weight * state_raw
    pre_belief_loss = config.pre_belief_weight * _belief_bce(
        pre_belief, batch.oracle_pre_belief
    )
    post_belief_loss = config.post_belief_weight * _belief_bce(
        post_belief, batch.actual_path_belief
    )
    total = trajectory + mode + state + pre_belief_loss + post_belief_loss
    if not torch.isfinite(total):
        raise ValueError("loss must be finite")
    return Model1LossTerms(
        total=total,
        trajectory=trajectory,
        mode=mode,
        state=state,
        pre_belief=pre_belief_loss,
        post_belief=post_belief_loss,
    )
