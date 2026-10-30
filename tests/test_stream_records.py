import numpy as np

from perception_env.config import RenderConfig, SimulationConfig
from perception_env.render import Occluder
from perception_env.simulate import BallState
from perception_env.stream_records import load_stream, save_stream
from perception_env.streaming import generate_stream
from perception_env.world import WorldSchema


def test_persistent_world_stream_round_trips_without_rerunning(tmp_path) -> None:
    schema = WorldSchema.v1()
    world = schema.world(horizontal_acceleration={"a_x": 0.1})
    episodes = generate_stream(
        world,
        (BallState(0.12, 0.4, 0.25, 0), BallState(0.12, 0.7, 0.25, 0)),
        (Occluder(0.35, 0.55, 0.05, 0.95),),
        SimulationConfig(dt=0.1, steps=32, seed=9),
        RenderConfig(width=16, height=16, ball_radius=1, min_hidden_steps=3),
        "stream-a",
    )
    assert len(episodes) == 2
    assert [item.event_id for item in episodes] == ["stream-a-000", "stream-a-001"]
    assert all(
        item.evaluation["true_world_flat"] == world.to_flat(schema).tolist()
        for item in episodes
    )
    root = save_stream(tmp_path / "stream", episodes)
    restored = load_stream(root)
    assert [item.event_id for item in restored] == [item.event_id for item in episodes]
    assert np.array_equal(restored[1].frames, episodes[1].frames)
    assert restored[1].reappearance_xy == episodes[1].reappearance_xy


def test_generation_is_seed_reproducible() -> None:
    schema = WorldSchema.v1()
    args = (
        schema.world(),
        (BallState(0.12, 0.5, 0.25, 0),),
        (Occluder(0.35, 0.55, 0.05, 0.95),),
        SimulationConfig(dt=0.1, steps=32, seed=4),
        RenderConfig(width=16, height=16, ball_radius=1, min_hidden_steps=3),
        "same",
    )
    first = generate_stream(*args)
    second = generate_stream(*args)
    assert np.array_equal(first[0].frames, second[0].frames)
    assert first[0].evaluation == second[0].evaluation

