"""Deterministic latent simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import SimulationConfig
from .operators import acceleration, damping_factor, reflect_crossing
from .world import WorldCode, WorldSchema


@dataclass(frozen=True)
class BallState:
    x: float
    y: float
    vx: float
    vy: float


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass(frozen=True)
class CollisionEvent:
    step: int
    operator: str
    position: float


@dataclass(frozen=True)
class Trajectory:
    states: tuple[BallState, ...]
    collisions: tuple[CollisionEvent, ...]
    valid: bool
    seed: int
    termination_reason: TerminationReason


def _rounded(state: BallState) -> BallState:
    return BallState(*(round(value, 12) for value in (state.x, state.y, state.vx, state.vy)))


def _advance(
    state: BallState, world: WorldCode, config: SimulationConfig
) -> tuple[BallState, tuple[str, ...]]:
    WorldSchema.v1().validate(world)
    ax, ay = acceleration(world)
    vx = state.vx + ax * config.dt
    vy = state.vy + ay * config.dt
    decay = damping_factor(world, config.dt)
    vx *= decay
    vy *= decay
    x = state.x + vx * config.dt
    y = state.y + vy * config.dt
    collisions: list[str] = []

    vertical_wall = world.slot("vertical_wall")
    if vertical_wall.active:
        position, restitution = vertical_wall.parameters
        x, vx, collided = reflect_crossing(
            state.x, x, vx, position, restitution
        )
        if collided:
            collisions.append("vertical_wall")

    horizontal_wall = world.slot("horizontal_wall")
    if horizontal_wall.active:
        position, restitution = horizontal_wall.parameters
        y, vy, collided = reflect_crossing(
            state.y, y, vy, position, restitution
        )
        if collided:
            collisions.append("horizontal_wall")
    return _rounded(BallState(x, y, vx, vy)), tuple(collisions)


def step(state: BallState, world: WorldCode, config: SimulationConfig) -> BallState:
    next_state, _ = _advance(state, world, config)
    return next_state


def rollout(
    initial: BallState, world: WorldCode, config: SimulationConfig
) -> Trajectory:
    states = [_rounded(initial)]
    collisions: list[CollisionEvent] = []
    reason = TerminationReason.COMPLETED
    valid = True
    current = states[0]
    for step_index in range(1, config.steps + 1):
        current, operator_names = _advance(current, world, config)
        states.append(current)
        for operator_name in operator_names:
            position = world.slot(operator_name).parameters[0]
            collisions.append(CollisionEvent(step_index, operator_name, position))
        if not (0.0 <= current.x <= 1.0 and 0.0 <= current.y <= 1.0):
            valid = False
            reason = TerminationReason.OUT_OF_BOUNDS
            break
    return Trajectory(
        states=tuple(states),
        collisions=tuple(collisions),
        valid=valid,
        seed=config.seed,
        termination_reason=reason,
    )
