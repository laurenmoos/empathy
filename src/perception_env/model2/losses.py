"""Activity supervision and masked, identifiable parameter regression."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from perception_env.physics.world import WorldSchema

from .contracts import SchemaLayout, WorldPrediction


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    activity: torch.Tensor
    parameters: torch.Tensor
    parameter_count: int


def world_code_loss(
    prediction: WorldPrediction,
    targets: torch.Tensor,
    parameter_identifiable: torch.Tensor,
    *,
    activity_weight: float = 1.0,
    parameter_weight: float = 1.0,
) -> LossResult:
    if (
        targets.ndim != 2
        or targets.shape[1] != 12
        or len(targets) == 0
        or prediction.activity_logits.shape != (len(targets), 5)
        or prediction.normalized_parameters.shape != (len(targets), 7)
        or parameter_identifiable.shape != (len(targets), 7)
        or parameter_identifiable.dtype != torch.bool
    ):
        raise ValueError("invalid loss tensor shapes or identifiability dtype")
    if not all(
        torch.isfinite(value).all()
        for value in (targets, prediction.activity_logits, prediction.normalized_parameters)
    ):
        raise ValueError("loss inputs must be finite")
    if any(not math.isfinite(v) or v < 0 for v in (activity_weight, parameter_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    layout = SchemaLayout.from_schema(WorldSchema.v1())
    activity = targets[:, layout.activity_indices]
    if not ((activity == 0) | (activity == 1)).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must be normalized with binary activity")
    mask = activity[:, layout.parameter_to_slot].bool() & parameter_identifiable
    count = int(mask.sum().item())
    activity_loss = F.binary_cross_entropy_with_logits(prediction.activity_logits, activity)
    squared_error = (prediction.normalized_parameters - targets[:, layout.parameter_indices]) ** 2
    parameter_loss = (squared_error * mask).sum() / max(count, 1)
    return LossResult(
        activity_weight * activity_loss + parameter_weight * parameter_loss,
        activity_loss,
        parameter_loss,
        count,
    )
