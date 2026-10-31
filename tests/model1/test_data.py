from pathlib import Path

import numpy as np
import pytest

from perception_env.model1.config import load_model1_config
from perception_env.model1.data import (
    Model1Dataset,
    Model1ExampleRecord,
    collate_model1,
    generate_model1_dataset,
    load_example,
    make_model1_record,
    save_example,
)
from perception_env.physics.simulate import BallState
from perception_env.physics.world import WorldSchema


def _record() -> Model1ExampleRecord:
    frames = np.zeros((10, 64, 64), dtype=np.uint8)
    frames[:, 30:34, 20:24] = 240
    states = np.zeros((12, 4), dtype=np.float32)
    states[:, 0] = np.linspace(0.2, 0.8, 12)
    states[:, 1] = 0.5
    states[:, 2] = 0.2
    return Model1ExampleRecord(
        example_id="train-000000",
        split="train",
        family_level="W0",
        seed=7,
        world_id="world-7",
        world_flat=np.zeros(12, dtype=np.float32),
        frames=frames,
        hidden_start=4,
        hidden_end=6,
        reappearance_step=7,
        target_states=states,
        target_mask=np.ones(12, dtype=bool),
        oracle_pre_belief=np.zeros((64, 64), dtype=np.float32),
        actual_path_belief=np.zeros((64, 64), dtype=np.float32),
    )


def test_example_round_trip(tmp_path) -> None:
    original = _record()
    loaded = load_example(save_example(original, tmp_path))

    assert loaded.example_id == original.example_id
    assert loaded.family_level == "W0"
    assert loaded.hidden_start == 4
    for name in (
        "world_flat",
        "frames",
        "target_states",
        "target_mask",
        "oracle_pre_belief",
        "actual_path_belief",
    ):
        assert np.array_equal(getattr(loaded, name), getattr(original, name))


def test_collate_builds_pre_post_views_and_targets() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    batch = collate_model1([_record(), _record()], config)

    assert batch.pre_frames.shape == (2, 16, 1, 64, 64)
    assert batch.post_frames.shape == (2, 16, 1, 64, 64)
    assert batch.pre_mask.shape == (2, 16)
    assert batch.target_positions.shape == (2, 12, 2)
    assert batch.target_mask.shape == (2, 12)
    assert batch.entry_state.shape == (2, 4)
    assert batch.oracle_pre_belief.shape == (2, 64, 64)
    assert batch.pre_mask[0].sum().item() == 7
    assert batch.post_mask[0].sum().item() == 8


def test_quick_generation_is_deterministic_and_environment_backed(tmp_path) -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    first = generate_model1_dataset(
        config,
        Path("configs/world_families/v1.yaml"),
        tmp_path / "a",
        seed=7,
        quick=True,
    )
    second = generate_model1_dataset(
        config,
        Path("configs/world_families/v1.yaml"),
        tmp_path / "b",
        seed=7,
        quick=True,
    )

    assert first.split_ids == second.split_ids
    dataset = Model1Dataset(tmp_path / "a", "train")
    assert len(dataset) == config.dataset.quick_counts[0]
    sample = dataset[0]
    assert sample.hidden_end - sample.hidden_start + 1 >= 2
    assert sample.reappearance_step > sample.hidden_end
    assert sample.frames[sample.reappearance_step].max() == 240


def test_base_family_sampling_has_valid_support_across_registered_seed(tmp_path) -> None:
    from dataclasses import replace

    config = load_model1_config(Path("configs/model1/v1.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, quick_counts=(2, 1, 1), max_attempts=200),
    )

    manifest = generate_model1_dataset(
        config,
        Path("configs/world_families/v1.yaml"),
        tmp_path / "seed-23",
        seed=23,
        quick=True,
    )

    assert {key: len(value) for key, value in manifest.split_ids.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }


def test_generator_reports_exhausted_attempts(tmp_path) -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    from dataclasses import replace

    impossible_dataset = replace(config.dataset, max_attempts=0)
    config = replace(config, dataset=impossible_dataset)

    with pytest.raises(RuntimeError, match="attempt"):
        generate_model1_dataset(
            config,
            Path("configs/world_families/v1.yaml"),
            tmp_path / "bad",
            seed=7,
            quick=True,
        )


def test_supplied_world_generates_varied_episodes_with_shared_world_id() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    world = WorldSchema.v1().world()

    lower, lower_reason = make_model1_record(
        config,
        world,
        BallState(0.1, 0.44, 0.18, 0.0),
        example_id="stream-0-0",
        split="stream",
        family_level="W0",
        seed=11,
    )
    upper, upper_reason = make_model1_record(
        config,
        world,
        BallState(0.1, 0.56, 0.18, 0.0),
        example_id="stream-0-1",
        split="stream",
        family_level="W0",
        seed=12,
    )

    assert lower_reason is upper_reason is None
    assert lower is not None and upper is not None
    assert lower.world_id == upper.world_id
    assert not np.array_equal(lower.frames, upper.frames)


def test_supplied_invalid_trajectory_returns_rejection_reason() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))

    record, reason = make_model1_record(
        config,
        WorldSchema.v1().world(),
        BallState(0.1, 1.2, 0.18, 0.0),
        example_id="invalid",
        split="stream",
        family_level="W0",
        seed=13,
    )

    assert record is None
    assert reason
