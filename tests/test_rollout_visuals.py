from empathy.physics.config import SimulationConfig
from empathy.physics.simulate import BallState, rollout
from empathy.physics.world import WorldSchema
from empathy.reasoning.visualize import render_trajectory_strip


def test_trajectory_strip_exposes_path_geometry_and_seed(tmp_path) -> None:
    schema = WorldSchema.v1()
    world = schema.world(
        vertical_acceleration={"a_y": -0.2},
        vertical_wall={"position": 0.8, "restitution": 0.9},
    )
    trajectory = rollout(
        BallState(0.2, 0.7, 0.25, 0.0),
        world,
        SimulationConfig(dt=0.1, steps=10, seed=23),
    )

    artifact = render_trajectory_strip(trajectory, world, schema, tmp_path / "rollout.png")

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["panels"] == ["latent trajectory", "operator summary"]
    assert artifact.metadata["axes"] == {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert artifact.metadata["seed"] == 23
    assert artifact.metadata["active_operators"] == [
        "vertical_acceleration",
        "vertical_wall",
    ]
