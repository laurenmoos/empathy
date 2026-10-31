"""Pure trajectory, calibration, and belief metrics for Model 1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .types import TrajectoryMixture


@dataclass(frozen=True)
class TrajectoryMetrics:
    min_ade: torch.Tensor
    min_fde: torch.Tensor
    top1_ade: torch.Tensor


@dataclass(frozen=True)
class CalibrationReport:
    ece: torch.Tensor
    bin_confidence: tuple[float | None, ...]
    bin_accuracy: tuple[float | None, ...]
    bin_count: tuple[int, ...]
    credible_coverage: dict[str, float]


def _distances(
    prediction: TrajectoryMixture, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction.validate()
    if targets.shape != (
        prediction.positions.shape[0],
        prediction.positions.shape[2],
        2,
    ):
        raise ValueError("targets do not match trajectory prediction")
    if mask.shape != targets.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean [batch, steps]")
    counts = mask.sum(dim=1)
    if torch.any(counts < 1):
        raise ValueError("every row needs a valid target")
    distances = torch.linalg.vector_norm(
        prediction.positions - targets[:, None], dim=-1
    )
    return distances, counts


def displacement_metrics(
    prediction: TrajectoryMixture, targets: torch.Tensor, mask: torch.Tensor
) -> TrajectoryMetrics:
    distances, counts = _distances(prediction, targets, mask)
    ade = (distances * mask[:, None]).sum(dim=-1) / counts[:, None]
    last = counts - 1
    fde = distances.gather(
        2, last[:, None, None].expand(-1, distances.shape[1], 1)
    ).squeeze(-1)
    top = prediction.weights.argmax(dim=1)
    return TrajectoryMetrics(
        min_ade=ade.min(dim=1).values,
        min_fde=fde.min(dim=1).values,
        top1_ade=ade.gather(1, top[:, None]).squeeze(1),
    )


def mixture_nll(
    prediction: TrajectoryMixture,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float = 0.03,
) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    distances, counts = _distances(prediction, targets, mask)
    log_kernel = -(distances**2) / (2 * sigma**2) - math.log(2 * math.pi * sigma**2)
    log_weights = torch.log_softmax(prediction.logits.float(), dim=1)[:, :, None]
    per_step = -torch.logsumexp(log_weights + log_kernel, dim=1)
    return (per_step * mask).sum(dim=1) / counts


def calibration_report(
    prediction: TrajectoryMixture,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    radius: float,
    bins: int,
) -> CalibrationReport:
    if not math.isfinite(radius) or radius <= 0 or bins < 1:
        raise ValueError("radius and bins must be positive")
    distances, counts = _distances(prediction, targets, mask)
    ade = (distances * mask[:, None]).sum(dim=-1) / counts[:, None]
    weights = prediction.weights
    confidence, top = weights.max(dim=1)
    correct = (ade.gather(1, top[:, None]).squeeze(1) <= radius).float()
    assignments = torch.clamp((confidence * bins).long(), max=bins - 1)
    bin_confidence: list[float | None] = []
    bin_accuracy: list[float | None] = []
    bin_count: list[int] = []
    ece = confidence.new_zeros(())
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        bin_count.append(count)
        if count == 0:
            bin_confidence.append(None)
            bin_accuracy.append(None)
            continue
        mean_confidence = confidence[selected].mean()
        mean_accuracy = correct[selected].mean()
        ece = ece + selected.float().mean() * torch.abs(mean_confidence - mean_accuracy)
        bin_confidence.append(float(mean_confidence))
        bin_accuracy.append(float(mean_accuracy))
    sorted_weights, order = weights.sort(dim=1, descending=True)
    sorted_ade = ade.gather(1, order)
    cumulative_before = sorted_weights.cumsum(dim=1) - sorted_weights
    coverage: dict[str, float] = {}
    for level in (0.5, 0.8, 0.9):
        included = cumulative_before < level
        success = ((sorted_ade <= radius) & included).any(dim=1).float().mean()
        coverage[f"{level:.1f}"] = float(success)
    return CalibrationReport(
        ece=ece,
        bin_confidence=tuple(bin_confidence),
        bin_accuracy=tuple(bin_accuracy),
        bin_count=tuple(bin_count),
        credible_coverage=coverage,
    )


def belief_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("belief tensors must have the same image shape")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("belief tensors must be finite")
    if predicted.ndim == 2:
        predicted, target = predicted[None], target[None]
    dimensions = tuple(range(1, predicted.ndim))
    intersection = torch.minimum(predicted, target).sum(dim=dimensions)
    union = torch.maximum(predicted, target).sum(dim=dimensions)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    bce = F.binary_cross_entropy(
        predicted.clamp(1.0e-6, 1.0 - 1.0e-6), target, reduction="none"
    ).mean(dim=dimensions)
    return {"soft_iou": iou, "bce": bce}
