"""Belief superpositions and second-order belief histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import RenderConfig
from .simulate import Trajectory


def _trajectory_mask(trajectory: Trajectory, render: RenderConfig) -> np.ndarray:
    mask = np.zeros((render.height, render.width), dtype=bool)
    for state in trajectory.states:
        if not (0.0 <= state.x <= 1.0 and 0.0 <= state.y <= 1.0):
            continue
        column = int(round(state.x * (render.width - 1)))
        row = int(round((1.0 - state.y) * (render.height - 1)))
        mask[row, column] = True
    return mask


def superpose(
    trajectories: Sequence[Trajectory], weights: np.ndarray, render: RenderConfig
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(trajectories) != weights.size:
        raise ValueError("weights must have one value per trajectory")
    if len(trajectories) == 0:
        return np.zeros((render.height, render.width), dtype=np.float32)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive mass")
    density = np.zeros((render.height, render.width), dtype=np.float64)
    normalized = weights / weights.sum()
    for trajectory, weight in zip(trajectories, normalized):
        density[_trajectory_mask(trajectory, render)] += weight
    return density.astype(np.float32)


@dataclass(frozen=True, eq=False)
class BeliefEvent:
    step: int
    pre: np.ndarray
    post: np.ndarray

    def __post_init__(self) -> None:
        pre = np.asarray(self.pre, dtype=np.float32)
        post = np.asarray(self.post, dtype=np.float32)
        if pre.ndim != 2 or post.ndim != 2:
            raise ValueError("belief images must be two-dimensional")
        if pre.shape != post.shape:
            raise ValueError("pre and post beliefs must have the same shape")
        if self.step < 0:
            raise ValueError("belief event step must be nonnegative")
        object.__setattr__(self, "pre", pre.copy())
        object.__setattr__(self, "post", post.copy())

    @property
    def difference(self) -> np.ndarray:
        return self.post - self.pre


@dataclass(frozen=True, eq=False)
class BeliefHistory:
    tensor: np.ndarray
    mask: np.ndarray
    steps: np.ndarray
    channel_names: tuple[str, str, str] = (
        "pre_belief",
        "post_belief",
        "belief_difference",
    )


def build_belief_history(
    events: Sequence[BeliefEvent], length: int = 4
) -> BeliefHistory:
    if length < 1:
        raise ValueError("history length must be positive")
    if not events:
        raise ValueError("at least one belief event is required to establish image shape")
    ordered = sorted(events, key=lambda event: event.step)[-length:]
    shape = ordered[0].pre.shape
    if any(event.pre.shape != shape for event in ordered):
        raise ValueError("all belief events must have the same shape")
    tensor = np.zeros((length, 3, *shape), dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    steps = np.full(length, -1, dtype=np.int64)
    offset = length - len(ordered)
    for index, event in enumerate(ordered, start=offset):
        tensor[index, 0] = event.pre
        tensor[index, 1] = event.post
        tensor[index, 2] = event.difference
        mask[index] = True
        steps[index] = event.step
    return BeliefHistory(tensor=tensor, mask=mask, steps=steps)

