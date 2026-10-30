"""Finite-state reachability oracle and binary possibility rasters."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import RenderConfig, SimulationConfig
from .simulate import BallState, Trajectory, rollout
from .world import WorldCode


def _axis_values(bounds: tuple[float, float], count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("grid resolution values must be positive")
    low, high = bounds
    if low > high:
        raise ValueError("grid lower bounds must not exceed upper bounds")
    if count == 1:
        return np.asarray([(low + high) / 2.0], dtype=np.float64)
    return np.linspace(low, high, count, dtype=np.float64)


@dataclass(frozen=True)
class EntryStateGrid:
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    vx_bounds: tuple[float, float]
    vy_bounds: tuple[float, float]
    resolution: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        for bounds in (self.x_bounds, self.y_bounds, self.vx_bounds, self.vy_bounds):
            _axis_values(bounds, 1)
        if len(self.resolution) != 4 or any(value < 1 for value in self.resolution):
            raise ValueError("resolution must contain four positive integers")

    def states(self) -> tuple[BallState, ...]:
        axes = [
            _axis_values(bounds, count)
            for bounds, count in zip(
                (self.x_bounds, self.y_bounds, self.vx_bounds, self.vy_bounds),
                self.resolution,
                strict=True,
            )
        ]
        return tuple(BallState(*(float(value) for value in values)) for values in product(*axes))

    def to_metadata(self) -> dict[str, list[float] | list[int]]:
        return {
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "vx_bounds": list(self.vx_bounds),
            "vy_bounds": list(self.vy_bounds),
            "resolution": list(self.resolution),
        }


@dataclass(frozen=True)
class Probe:
    context_id: str
    x: float
    y: float
    step: int
    tolerance: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("probe step must be nonnegative")
        if self.tolerance < 0:
            raise ValueError("probe tolerance must be nonnegative")


@dataclass(frozen=True)
class ReachabilityResult:
    context_id: str
    entry_grid: EntryStateGrid
    world: WorldCode
    query_steps: tuple[int, ...]
    trajectories: tuple[Trajectory, ...]
    endpoints: dict[int, tuple[BallState, ...]]
    config: SimulationConfig

    def label(self, probe: Probe) -> bool:
        if probe.context_id != self.context_id:
            raise ValueError("probe context does not match reachability result")
        if probe.step not in self.endpoints:
            raise ValueError(f"step {probe.step} was not queried")
        return any(
            np.hypot(state.x - probe.x, state.y - probe.y) <= probe.tolerance
            for state in self.endpoints[probe.step]
        )


def enumerate_reachable(
    entry_states: EntryStateGrid,
    world: WorldCode,
    query_steps: tuple[int, ...],
    config: SimulationConfig,
    *,
    context_id: str = "context",
) -> ReachabilityResult:
    if not query_steps:
        raise ValueError("query_steps must not be empty")
    if any(step < 0 or step > config.steps for step in query_steps):
        raise ValueError("query steps must lie within the configured rollout")
    ordered_steps = tuple(sorted(set(query_steps)))
    trajectories = tuple(rollout(state, world, config) for state in entry_states.states())
    endpoints: dict[int, tuple[BallState, ...]] = {}
    for query_step in ordered_steps:
        endpoints[query_step] = tuple(
            trajectory.states[query_step]
            for trajectory in trajectories
            if trajectory.valid and len(trajectory.states) > query_step
        )
    return ReachabilityResult(
        context_id=context_id,
        entry_grid=entry_states,
        world=world,
        query_steps=ordered_steps,
        trajectories=trajectories,
        endpoints=endpoints,
        config=config,
    )


def rasterize_endpoints(result: ReachabilityResult, render: RenderConfig) -> np.ndarray:
    occupancy = np.zeros(
        (len(result.query_steps), render.height, render.width), dtype=bool
    )
    for plane, step in enumerate(result.query_steps):
        for state in result.endpoints[step]:
            if not (0.0 <= state.x <= 1.0 and 0.0 <= state.y <= 1.0):
                continue
            column = int(round(state.x * (render.width - 1)))
            row = int(round((1.0 - state.y) * (render.height - 1)))
            occupancy[plane, row, column] = True
    return occupancy

