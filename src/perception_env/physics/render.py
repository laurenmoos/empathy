"""First-order rendering and observation-only occlusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RenderConfig
from .simulate import BallState, Trajectory


@dataclass(frozen=True)
class Occluder:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("occluder bounds must lie in [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("occluder minimums must be below maximums")


@dataclass(frozen=True, eq=False)
class RenderedSequence:
    frames: np.ndarray
    ball_masks: np.ndarray
    visible_ball_masks: np.ndarray
    visible: np.ndarray
    fully_hidden: np.ndarray
    partially_occluded: np.ndarray
    latent_states: tuple[BallState, ...]
    occluders: tuple[Occluder, ...]
    config: RenderConfig
    seed: int
    rejection_reason: str | None


def _disk_mask(state: BallState, config: RenderConfig) -> np.ndarray:
    center_x = int(round(state.x * (config.width - 1)))
    center_y = int(round((1.0 - state.y) * (config.height - 1)))
    rows, columns = np.ogrid[: config.height, : config.width]
    return (columns - center_x) ** 2 + (rows - center_y) ** 2 <= config.ball_radius**2


def _occluder_mask(occluders: tuple[Occluder, ...], config: RenderConfig) -> np.ndarray:
    mask = np.zeros((config.height, config.width), dtype=bool)
    for occluder in occluders:
        x0 = int(np.floor(occluder.x_min * (config.width - 1)))
        x1 = int(np.ceil(occluder.x_max * (config.width - 1)))
        y0 = int(np.floor((1.0 - occluder.y_max) * (config.height - 1)))
        y1 = int(np.ceil((1.0 - occluder.y_min) * (config.height - 1)))
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _hidden_runs(hidden: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(hidden.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _rejection_reason(
    fully_hidden: np.ndarray,
    partially_occluded: np.ndarray,
    visible: np.ndarray,
    min_hidden_steps: int,
) -> str | None:
    runs = _hidden_runs(fully_hidden)
    if not runs:
        return "partial_only_occlusion" if partially_occluded.any() else "no_occlusion"
    qualified = [run for run in runs if run[1] - run[0] + 1 >= min_hidden_steps]
    if not qualified:
        return "insufficient_hidden_run"
    if not any(visible[end + 1 :].any() for _, end in qualified):
        return "no_visible_reappearance"
    return None


def render_sequence(
    trajectory: Trajectory,
    occluders: tuple[Occluder, ...],
    config: RenderConfig,
) -> RenderedSequence:
    occluder_mask = _occluder_mask(occluders, config)
    frames: list[np.ndarray] = []
    ball_masks: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []
    fully_hidden: list[bool] = []
    partially_occluded: list[bool] = []
    visible: list[bool] = []
    for state in trajectory.states:
        ball = _disk_mask(state, config)
        overlap = ball & occluder_mask
        visible_ball = ball & ~occluder_mask
        frame = np.full((config.height, config.width), 18, dtype=np.uint8)
        frame[occluder_mask] = 100
        frame[visible_ball] = 240
        frames.append(frame)
        ball_masks.append(ball)
        visible_masks.append(visible_ball)
        is_hidden = bool(ball.any() and overlap.sum() == ball.sum())
        is_partial = bool(overlap.any() and visible_ball.any())
        fully_hidden.append(is_hidden)
        partially_occluded.append(is_partial)
        visible.append(bool(visible_ball.any()))

    hidden_array = np.asarray(fully_hidden, dtype=bool)
    partial_array = np.asarray(partially_occluded, dtype=bool)
    visible_array = np.asarray(visible, dtype=bool)
    return RenderedSequence(
        frames=np.stack(frames),
        ball_masks=np.stack(ball_masks),
        visible_ball_masks=np.stack(visible_masks),
        visible=visible_array,
        fully_hidden=hidden_array,
        partially_occluded=partial_array,
        latent_states=trajectory.states,
        occluders=occluders,
        config=config,
        seed=trajectory.seed,
        rejection_reason=_rejection_reason(
            hidden_array, partial_array, visible_array, config.min_hidden_steps
        ),
    )

