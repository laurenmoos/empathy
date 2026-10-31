"""Validated tensors exchanged by Model 1 components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Model1Batch:
    pre_frames: torch.Tensor
    pre_times: torch.Tensor
    pre_mask: torch.Tensor
    post_frames: torch.Tensor
    post_times: torch.Tensor
    post_mask: torch.Tensor
    entry_state: torch.Tensor
    target_positions: torch.Tensor
    target_mask: torch.Tensor
    oracle_pre_belief: torch.Tensor
    actual_path_belief: torch.Tensor
    example_ids: tuple[str, ...]
    family_levels: tuple[str, ...]

    def to(self, device: torch.device | str) -> Model1Batch:
        values = {
            name: getattr(self, name).to(device)
            for name in (
                "pre_frames",
                "pre_times",
                "pre_mask",
                "post_frames",
                "post_times",
                "post_mask",
                "entry_state",
                "target_positions",
                "target_mask",
                "oracle_pre_belief",
                "actual_path_belief",
            )
        }
        return Model1Batch(**values, example_ids=self.example_ids, family_levels=self.family_levels)


@dataclass(frozen=True)
class TrajectoryMixture:
    positions: torch.Tensor
    logits: torch.Tensor
    entry_state: torch.Tensor

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.float(), dim=-1)

    def validate(self) -> None:
        if self.positions.ndim != 4 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape [batch, modes, steps, 2]")
        if self.logits.shape != self.positions.shape[:2]:
            raise ValueError("logits must have one value per trajectory mode")
        if self.entry_state.shape != (self.positions.shape[0], 4):
            raise ValueError("entry_state must have shape [batch, 4]")
        if not all(
            torch.isfinite(value).all()
            for value in (self.positions, self.logits, self.entry_state)
        ):
            raise ValueError("trajectory prediction must contain finite values")


@dataclass(frozen=True)
class Model1Trace:
    prediction: TrajectoryMixture
    activation: torch.Tensor

    def validate(self) -> None:
        self.prediction.validate()
        if self.activation.ndim != 2:
            raise ValueError("activation must have shape [batch, hidden]")
        if self.activation.shape[0] != self.prediction.positions.shape[0]:
            raise ValueError("activation batch must match prediction batch")
        if not torch.isfinite(self.activation).all():
            raise ValueError("activation must contain finite values")


@dataclass(frozen=True)
class BeliefPrediction:
    time_belief: torch.Tensor
    collapsed: torch.Tensor

    def validate(self) -> None:
        if self.time_belief.ndim != 4:
            raise ValueError("time_belief must have shape [batch, steps, height, width]")
        if self.collapsed.shape != (
            self.time_belief.shape[0],
            self.time_belief.shape[2],
            self.time_belief.shape[3],
        ):
            raise ValueError("collapsed belief shape does not match time belief")
        if not torch.isfinite(self.time_belief).all() or not torch.isfinite(self.collapsed).all():
            raise ValueError("belief tensors must be finite")
