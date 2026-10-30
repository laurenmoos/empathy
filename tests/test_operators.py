import math

import pytest

from perception_env.config import SimulationConfig
from perception_env.simulate import BallState, step
from perception_env.world import WorldSchema


def test_constant_velocity_uses_semi_implicit_position_update() -> None:
    world = WorldSchema.v1().world()
    state = BallState(x=0.2, y=0.3, vx=0.4, vy=-0.1)

    result = step(state, world, SimulationConfig(dt=0.1, steps=1))

    assert result == BallState(x=0.24, y=0.29, vx=0.4, vy=-0.1)


def test_vertical_acceleration_updates_velocity_before_position() -> None:
    world = WorldSchema.v1().world(vertical_acceleration={"a_y": -0.5})
    state = BallState(x=0.2, y=0.3, vx=0.0, vy=0.2)

    result = step(state, world, SimulationConfig(dt=0.1, steps=1))

    assert result.vy == pytest.approx(0.15)
    assert result.y == pytest.approx(0.315)


def test_damping_follows_acceleration_and_precedes_position() -> None:
    world = WorldSchema.v1().world(damping={"lambda": 1.0})
    state = BallState(x=0.2, y=0.2, vx=1.0, vy=0.0)

    result = step(state, world, SimulationConfig(dt=0.1, steps=1))

    expected_velocity = math.exp(-0.1)
    assert result.vx == pytest.approx(expected_velocity)
    assert result.x == pytest.approx(0.2 + 0.1 * expected_velocity)


def test_vertical_wall_reflects_crossing_with_restitution() -> None:
    world = WorldSchema.v1().world(
        vertical_wall={"position": 0.5, "restitution": 0.5}
    )
    state = BallState(x=0.4, y=0.3, vx=0.3, vy=0.0)

    result = step(state, world, SimulationConfig(dt=0.5, steps=1))

    assert result.x == pytest.approx(0.45)
    assert result.vx == pytest.approx(-0.15)


def test_horizontal_wall_reflects_crossing_with_restitution() -> None:
    world = WorldSchema.v1().world(
        horizontal_wall={"position": 0.5, "restitution": 0.8}
    )
    state = BallState(x=0.3, y=0.6, vx=0.0, vy=-0.4)

    result = step(state, world, SimulationConfig(dt=0.5, steps=1))

    assert result.y == pytest.approx(0.6)
    assert result.vy == pytest.approx(0.32)


def test_acceleration_is_applied_before_wall_collision() -> None:
    world = WorldSchema.v1().world(
        horizontal_acceleration={"a_x": 0.2},
        vertical_wall={"position": 0.48, "restitution": 0.5},
    )
    state = BallState(x=0.45, y=0.3, vx=0.4, vy=0.0)

    result = step(state, world, SimulationConfig(dt=0.1, steps=1))

    assert result.x == pytest.approx(0.468)
    assert result.vx == pytest.approx(-0.21)

