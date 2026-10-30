"""Pure motion-operator helpers."""

from __future__ import annotations

import math

from .world import WorldCode


def acceleration(world: WorldCode) -> tuple[float, float]:
    horizontal = world.slot("horizontal_acceleration")
    vertical = world.slot("vertical_acceleration")
    ax = horizontal.parameters[0] if horizontal.active else 0.0
    ay = vertical.parameters[0] if vertical.active else 0.0
    return ax, ay


def damping_factor(world: WorldCode, dt: float) -> float:
    damping = world.slot("damping")
    if not damping.active:
        return 1.0
    return math.exp(-damping.parameters[0] * dt)


def reflect_crossing(
    previous: float,
    proposed: float,
    velocity: float,
    position: float,
    restitution: float,
) -> tuple[float, float, bool]:
    crossed_positive = previous < position <= proposed
    crossed_negative = previous > position >= proposed
    if not (crossed_positive or crossed_negative):
        return proposed, velocity, False
    reflected_position = position - (proposed - position)
    return reflected_position, -velocity * restitution, True

