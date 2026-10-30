"""Differentiable rasterization and the strict belief-only downstream bridge."""

from __future__ import annotations

import numpy as np
import torch

from perception_env.beliefs import BeliefEvent

from .types import BeliefPrediction, TrajectoryMixture


def _bilinear_splats(positions: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("raster dimensions must be positive")
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [batch, modes, steps, 2]")
    batch, modes, steps = positions.shape[:3]
    margin_x = 1.0 / max(width - 1, 1)
    margin_y = 1.0 / max(height - 1, 1)
    valid = (
        (positions[..., 0] >= -margin_x)
        & (positions[..., 0] <= 1.0 + margin_x)
        & (positions[..., 1] >= -margin_y)
        & (positions[..., 1] <= 1.0 + margin_y)
    )
    x = positions[..., 0].clamp(0.0, 1.0) * (width - 1)
    y = (1.0 - positions[..., 1].clamp(0.0, 1.0)) * (height - 1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    dx = x - x0.to(x.dtype)
    dy = y - y0.to(y.dtype)
    neighbors = (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    )
    flat_count = batch * modes * steps
    raster = positions.new_zeros((flat_count, height * width))
    for column, row, mass in neighbors:
        indices = (row * width + column).reshape(flat_count, 1)
        values = (mass * valid.to(mass.dtype)).reshape(flat_count, 1)
        raster = raster.scatter_add(1, indices, values)
    return raster.reshape(batch, modes, steps, height, width)


def rasterize_mixture(
    prediction: TrajectoryMixture, *, height: int, width: int
) -> BeliefPrediction:
    prediction.validate()
    per_mode = _bilinear_splats(prediction.positions, height, width)
    weights = prediction.weights.to(per_mode.dtype)
    time_belief = (per_mode * weights[:, :, None, None, None]).sum(dim=1)
    mode_union = 1.0 - torch.prod(1.0 - per_mode.clamp(0.0, 1.0), dim=2)
    collapsed = (mode_union * weights[:, :, None, None]).sum(dim=1).clamp(0.0, 1.0)
    result = BeliefPrediction(time_belief=time_belief, collapsed=collapsed)
    result.validate()
    return result


def rasterize_actual_path(
    positions: torch.Tensor,
    mask: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("actual positions must have shape [batch, steps, 2]")
    if mask.shape != positions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("actual path mask must be boolean [batch, steps]")
    splats = _bilinear_splats(positions[:, None], height, width)[:, 0]
    splats = splats * mask[:, :, None, None].to(splats.dtype)
    return (1.0 - torch.prod(1.0 - splats.clamp(0.0, 1.0), dim=1)).clamp(0.0, 1.0)


def belief_event_from_predictions(
    pre: BeliefPrediction,
    post: BeliefPrediction,
    *,
    event_step: int,
    batch_index: int = 0,
) -> BeliefEvent:
    pre.validate()
    post.validate()
    if pre.collapsed.shape != post.collapsed.shape:
        raise ValueError("pre and post belief batches must have the same shape")
    if not 0 <= batch_index < pre.collapsed.shape[0]:
        raise IndexError(batch_index)

    def array(value: torch.Tensor) -> np.ndarray:
        return value[batch_index].detach().to(device="cpu", dtype=torch.float32).numpy().copy()

    return BeliefEvent(event_step, array(pre.collapsed), array(post.collapsed))
