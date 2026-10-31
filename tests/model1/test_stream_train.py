from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception_env.model1.config import load_model1_config
from perception_env.model1.data import Model1ExampleRecord
from perception_env.model1.stream_train import (
    Model1ReplayBuffer,
    StreamTrainingConfig,
    generate_environment_batches,
    load_stream_training_config,
)


def _stream_config(**overrides) -> StreamTrainingConfig:
    values = {
        "schema_version": "model1-stream-v1",
        "levels": ("W0",),
        "environment_count": 2,
        "episodes_per_environment": 3,
        "max_attempts_per_environment": 80,
        "replay_capacity": 8,
        "replay_batch_size": 2,
        "update_steps_per_environment": 1,
    }
    values.update(overrides)
    return StreamTrainingConfig(**values)


def _record(index: int) -> Model1ExampleRecord:
    return Model1ExampleRecord(
        example_id=f"event-{index}",
        split="stream",
        family_level="W0",
        seed=index,
        world_id="world",
        world_flat=np.zeros(12, np.float32),
        frames=np.zeros((8, 64, 64), np.uint8),
        hidden_start=2,
        hidden_end=4,
        reappearance_step=5,
        target_states=np.zeros((12, 4), np.float32),
        target_mask=np.ones(12, bool),
        oracle_pre_belief=np.zeros((64, 64), np.float32),
        actual_path_belief=np.zeros((64, 64), np.float32),
    )


def test_stream_configuration_rejects_nonpositive_and_oversized_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        _stream_config(environment_count=0)
    with pytest.raises(ValueError, match="replay"):
        _stream_config(replay_capacity=2, replay_batch_size=3)

    path = tmp_path / "stream.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "model1-stream-v1",
                "stream": {
                    "levels": ["W0"],
                    "environment_count": 2,
                    "episodes_per_environment": 3,
                    "max_attempts_per_environment": 80,
                    "replay_capacity": 8,
                    "replay_batch_size": 2,
                    "update_steps_per_environment": 1,
                },
            }
        )
    )
    assert load_stream_training_config(path) == _stream_config()


def test_environment_batches_are_deterministic_fixed_worlds_with_varied_episodes() -> None:
    model_config = load_model1_config(Path("configs/model1/v1.yaml"))
    stream_config = _stream_config()

    first = generate_environment_batches(
        model_config,
        stream_config,
        Path("configs/world_families/v1.yaml"),
        seed=17,
    )
    second = generate_environment_batches(
        model_config,
        stream_config,
        Path("configs/world_families/v1.yaml"),
        seed=17,
    )

    assert len(first) == 2
    assert [batch.world_id for batch in first] == [batch.world_id for batch in second]
    assert all(len(batch.records) == 3 for batch in first)
    assert all(len({record.world_id for record in batch.records}) == 1 for batch in first)
    assert any(
        not np.array_equal(first[0].records[0].frames, record.frames)
        for record in first[0].records[1:]
    )
    for left_batch, right_batch in zip(first, second, strict=True):
        for left, right in zip(left_batch.records, right_batch.records, strict=True):
            assert left.example_id == right.example_id
            assert np.array_equal(left.frames, right.frames)


def test_environment_batch_exhausts_attempt_budget_instead_of_shrinking() -> None:
    model_config = load_model1_config(Path("configs/model1/v1.yaml"))
    impossible = replace(
        model_config,
        dataset=replace(
            model_config.dataset,
            initial_state={**model_config.dataset.initial_state, "y": (1.2, 1.3)},
        ),
    )

    with pytest.raises(RuntimeError, match="environment 0.*attempt"):
        generate_environment_batches(
            impossible,
            _stream_config(environment_count=1, max_attempts_per_environment=2),
            Path("configs/world_families/v1.yaml"),
            seed=5,
        )


def test_replay_is_bounded_seeded_and_keeps_environment_provenance() -> None:
    left = Model1ReplayBuffer(3, seed=9)
    right = Model1ReplayBuffer(3, seed=9)
    for index in range(10):
        left.add(index // 5, _record(index))
        right.add(index // 5, _record(index))

    assert len(left.entries) == 3
    assert [entry.record.example_id for entry in left.entries] == [
        entry.record.example_id for entry in right.entries
    ]
    assert [entry.record.example_id for entry in left.sample(3)] == [
        entry.record.example_id for entry in right.sample(3)
    ]

    replay = Model1ReplayBuffer(4, seed=3)
    replay.add(0, _record(0))
    replay.add(0, _record(1))
    replay.add(1, _record(2))
    assert {entry.environment_index for entry in replay.sample(3)} == {0, 1}
