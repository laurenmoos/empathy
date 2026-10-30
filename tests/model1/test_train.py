from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from perception_env.model1.config import load_model1_config
from perception_env.model1.data import Model1ExampleRecord
from perception_env.model1.train import load_checkpoint, train_model1


def _record(index: int) -> Model1ExampleRecord:
    frames = np.zeros((8, 64, 64), dtype=np.uint8)
    for step in range(8):
        frames[step, 32, 8 + step * 4] = 240
    states = np.zeros((12, 4), dtype=np.float32)
    states[:, 0] = np.linspace(0.25, 0.7, 12)
    states[:, 1] = 0.5
    states[:, 2] = 0.2
    oracle = np.zeros((64, 64), dtype=np.float32)
    oracle[32, 16:46] = 1
    actual = np.zeros((64, 64), dtype=np.float32)
    actual[32, 16:46:3] = 1
    return Model1ExampleRecord(
        example_id=f"example-{index}",
        split="train",
        family_level="W0",
        seed=index,
        world_id="world",
        world_flat=np.zeros(12, dtype=np.float32),
        frames=frames,
        hidden_start=3,
        hidden_end=5,
        reappearance_step=6,
        target_states=states,
        target_mask=np.ones(12, dtype=bool),
        oracle_pre_belief=oracle,
        actual_path_belief=actual,
    )


def _config(epochs: int):
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    training = replace(config.training, epochs=epochs, batch_size=4, learning_rate=0.003)
    return replace(config, training=training)


def test_tiny_cpu_training_reduces_loss_and_saves_checkpoints(tmp_path) -> None:
    dataset = [_record(index) for index in range(4)]

    result = train_model1(
        _config(30), dataset, dataset, tmp_path / "run", device=torch.device("cpu"), seed=7
    )

    assert len(result.history) == 30
    assert result.history[-1]["train_total"] < 0.7 * result.history[0]["train_total"]
    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint.is_file()


def test_training_replays_under_same_seed(tmp_path) -> None:
    dataset = [_record(index) for index in range(4)]
    first = train_model1(
        _config(1), dataset, dataset, tmp_path / "a", device=torch.device("cpu"), seed=11
    )
    second = train_model1(
        _config(1), dataset, dataset, tmp_path / "b", device=torch.device("cpu"), seed=11
    )

    first_model, _ = load_checkpoint(first.last_checkpoint)
    second_model, _ = load_checkpoint(second.last_checkpoint)
    for first_value, second_value in zip(
        first_model.state_dict().values(), second_model.state_dict().values(), strict=True
    ):
        assert torch.equal(first_value, second_value)
    assert first.history == second.history


def test_checkpoint_round_trip_and_dimension_validation(tmp_path) -> None:
    dataset = [_record(index) for index in range(4)]
    config = _config(1)
    result = train_model1(
        config, dataset, dataset, tmp_path / "run", device=torch.device("cpu"), seed=13
    )
    model, metadata = load_checkpoint(result.best_checkpoint, expected_config=config)

    assert metadata.schema_version == "model1-v1"
    assert metadata.seed == 13
    assert metadata.epoch == 0
    assert metadata.best_validation_loss > 0
    assert model.training is False

    bad = replace(config, model=replace(config.model, modes=7))
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(result.best_checkpoint, expected_config=bad)
