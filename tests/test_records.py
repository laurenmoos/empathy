import numpy as np
import pytest

from perception_env.beliefs import BeliefEvent
from perception_env.config import RenderConfig, SimulationConfig
from perception_env.reachability import EntryStateGrid, enumerate_reachable
from perception_env.records import (
    BenchmarkRecord,
    Model2InputRecord,
    belief_entropy,
    load_record,
    save_record,
)
from perception_env.render import Occluder, render_sequence
from perception_env.simulate import BallState, rollout
from perception_env.world import WorldSchema


def make_record(rejection_reason: str | None = None) -> BenchmarkRecord:
    schema = WorldSchema.v1()
    world = schema.world(vertical_acceleration={"a_y": -0.3})
    simulation = SimulationConfig(dt=0.1, steps=8, seed=19)
    render_config = RenderConfig(width=16, height=16, ball_radius=1)
    trajectory = rollout(BallState(0.2, 0.5, 0.25, 0.0), world, simulation)
    sequence = render_sequence(
        trajectory, (Occluder(0.35, 0.65, 0.3, 0.7),), render_config
    )
    grid = EntryStateGrid(
        (0.19, 0.21), (0.49, 0.51), (0.24, 0.26), (0.0, 0.0), (2, 2, 2, 1)
    )
    reachability = enumerate_reachable(
        grid, world, (4,), simulation, context_id="record-context"
    )
    pre = np.zeros((16, 16), dtype=np.float32)
    pre[8, 3:10] = 0.5
    post = np.zeros((16, 16), dtype=np.float32)
    post[8, 7:12] = 1.0
    return BenchmarkRecord.from_components(
        record_id="example-19",
        family_level="W1",
        schema=schema,
        world=world,
        trajectory=trajectory,
        sequence=sequence,
        reachability=reachability,
        belief=BeliefEvent(4, pre, post),
        rejection_reason=rejection_reason,
    )


def test_record_round_trip_preserves_schema_rejection_and_arrays(tmp_path) -> None:
    record = make_record(rejection_reason="insufficient_hidden_run")

    path = save_record(record, tmp_path)
    loaded = load_record(path)

    assert loaded.record_id == record.record_id
    assert loaded.schema_version == "v1"
    assert loaded.rejection_reason == "insufficient_hidden_run"
    assert loaded.family_level == "W1"
    assert loaded.metadata == record.metadata
    assert loaded.arrays.keys() == record.arrays.keys()
    for name in record.arrays:
        assert np.array_equal(loaded.arrays[name], record.arrays[name])


def test_loaded_record_reconstructs_visual_inputs_without_simulation(tmp_path) -> None:
    loaded = load_record(save_record(make_record(), tmp_path))
    schema = WorldSchema.v1()

    assert loaded.world(schema).schema_version == "v1"
    assert loaded.trajectory().states[0] == BallState(0.2, 0.5, 0.25, 0.0)
    assert np.array_equal(loaded.sequence().frames, loaded.arrays["frames"])
    assert np.array_equal(loaded.belief_event().pre, loaded.arrays["pre_belief"])
    assert loaded.reachability().query_steps == (4,)
    assert len(loaded.reachability().trajectories) == 8
    assert loaded.reachability().trajectories[0].states


@pytest.mark.parametrize(
    "forbidden",
    ["raw_frames", "positions", "visibility_masks", "detector_features", "event_marker"],
)
def test_model2_input_rejects_first_order_fields(forbidden: str) -> None:
    valid = {
        "pre_belief": np.zeros((4, 8, 8), dtype=np.float32),
        "post_belief": np.zeros((4, 8, 8), dtype=np.float32),
        "belief_difference": np.zeros((4, 8, 8), dtype=np.float32),
        "entropy": np.zeros(4, dtype=np.float32),
        "history_mask": np.ones(4, dtype=bool),
        "target_world": np.zeros(12, dtype=np.float32),
        forbidden: np.zeros(1),
    }

    with pytest.raises(ValueError, match="first-order|unsupported"):
        Model2InputRecord.from_mapping(valid)


def test_model2_input_accepts_belief_derived_fields_only() -> None:
    pre = np.zeros((4, 8, 8), dtype=np.float32)
    post = np.ones((4, 8, 8), dtype=np.float32)
    record = Model2InputRecord.from_mapping(
        {
            "pre_belief": pre,
            "post_belief": post,
            "belief_difference": post - pre,
            "entropy": belief_entropy(pre),
            "history_mask": np.ones(4, dtype=bool),
            "target_world": np.zeros(12, dtype=np.float32),
        }
    )

    assert record.channel_names == (
        "pre_belief",
        "post_belief",
        "belief_difference",
    )


@pytest.mark.parametrize(
    "mutation",
    ["one_dimensional", "wrong_difference", "bad_entropy", "bad_mask", "bad_target"],
)
def test_model2_input_enforces_tensor_contract(mutation: str) -> None:
    pre = np.zeros((4, 8, 8), dtype=np.float32)
    post = np.ones((4, 8, 8), dtype=np.float32)
    values = {
        "pre_belief": pre,
        "post_belief": post,
        "belief_difference": post - pre,
        "entropy": belief_entropy(pre),
        "history_mask": np.ones(4, dtype=bool),
        "target_world": np.zeros(12, dtype=np.float32),
    }
    if mutation == "one_dimensional":
        values["pre_belief"] = np.zeros(2)
        values["post_belief"] = np.zeros(2)
        values["belief_difference"] = np.zeros(2)
    elif mutation == "wrong_difference":
        values["belief_difference"] = np.zeros_like(pre)
    elif mutation == "bad_entropy":
        values["entropy"] = np.full(4, 999.0)
    elif mutation == "bad_mask":
        values["history_mask"] = np.ones(3, dtype=bool)
    else:
        values["target_world"] = np.zeros(11)

    with pytest.raises(ValueError):
        Model2InputRecord.from_mapping(values)

