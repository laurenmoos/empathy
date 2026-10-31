from perception_env.physics.config import RenderConfig, SimulationConfig
from perception_env.physics.render import Occluder, render_sequence
from perception_env.physics.simulate import BallState, rollout
from perception_env.physics.world import WorldSchema
from perception_env.reasoning.visualize import render_observation_strip, render_operator_comparison


def test_observation_strip_shows_latent_visible_and_hidden_views(tmp_path) -> None:
    schema = WorldSchema.v1()
    world = schema.world()
    trajectory = rollout(
        BallState(0.15, 0.5, 0.2, 0.0),
        world,
        SimulationConfig(dt=0.2, steps=18, seed=31),
    )
    sequence = render_sequence(
        trajectory,
        (Occluder(0.35, 0.65, 0.35, 0.65),),
        RenderConfig(min_hidden_steps=3),
    )

    artifact = render_observation_strip(sequence, tmp_path / "observation.png")

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["panels"] == [
        "latent path",
        "visible frames",
        "visibility timeline",
    ]
    assert artifact.metadata["hidden_intervals"]
    assert artifact.metadata["reappearance_step"] is not None


def test_operator_comparison_holds_seed_and_initial_state_fixed(tmp_path) -> None:
    schema = WorldSchema.v1()
    config = SimulationConfig(dt=0.1, steps=8, seed=41)
    initial = BallState(0.2, 0.7, 0.2, 0.0)
    worlds = [
        schema.world(vertical_acceleration={"a_y": -0.2}),
        schema.world(vertical_acceleration={"a_y": -0.8}),
    ]
    trajectories = [rollout(initial, world, config) for world in worlds]

    artifact = render_operator_comparison(
        trajectories,
        worlds,
        schema,
        operator="vertical_acceleration",
        path=tmp_path / "comparison.png",
    )

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["same_seed"] is True
    assert artifact.metadata["same_initial_state"] is True
    assert artifact.metadata["operator"] == "vertical_acceleration"
    assert artifact.metadata["parameter_values"] == [-0.2, -0.8]
