from perception_env.physics.config import SimulationConfig
from perception_env.physics.simulate import BallState, TerminationReason, rollout
from perception_env.physics.world import WorldSchema


def test_rollout_is_deterministic_and_records_seed_and_states() -> None:
    world = WorldSchema.v1().world(vertical_acceleration={"a_y": -0.2})
    config = SimulationConfig(dt=0.1, steps=4, seed=17)
    initial = BallState(0.2, 0.7, 0.1, 0.0)

    first = rollout(initial, world, config)
    second = rollout(initial, world, config)

    assert first == second
    assert first.seed == 17
    assert len(first.states) == 5
    assert first.valid is True
    assert first.termination_reason is TerminationReason.COMPLETED


def test_rollout_records_collision_events() -> None:
    world = WorldSchema.v1().world(
        vertical_wall={"position": 0.5, "restitution": 1.0}
    )
    config = SimulationConfig(dt=0.5, steps=2)

    trajectory = rollout(BallState(0.4, 0.4, 0.3, 0.0), world, config)

    assert [(event.step, event.operator) for event in trajectory.collisions] == [
        (1, "vertical_wall")
    ]


def test_rollout_terminates_when_ball_leaves_domain_without_boundary() -> None:
    world = WorldSchema.v1().world()
    config = SimulationConfig(dt=0.5, steps=4)

    trajectory = rollout(BallState(0.95, 0.5, 0.2, 0.0), world, config)

    assert trajectory.valid is False
    assert trajectory.termination_reason is TerminationReason.OUT_OF_BOUNDS
    assert len(trajectory.states) == 2
    assert trajectory.states[-1].x == 1.05

