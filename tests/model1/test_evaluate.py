from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from perception_env.model1.config import load_model1_config
from perception_env.model1.data import Model1ExampleRecord
from perception_env.model1.evaluate import (
    evaluate_model1,
    load_prediction_record,
)
from perception_env.model1.train import train_model1


def record(index: int, family: str) -> Model1ExampleRecord:
    frames = np.zeros((8, 64, 64), dtype=np.uint8)
    frames[:, 32, 8:40] = 80
    states = np.zeros((12, 4), dtype=np.float32)
    states[:, 0] = np.linspace(0.2, 0.7, 12)
    states[:, 1] = 0.5 + 0.02 * index
    states[:, 2] = 0.2
    target = np.zeros((64, 64), dtype=np.float32)
    target[32 - index, 13:46:3] = 1
    return Model1ExampleRecord(
        example_id=f"example-{index}",
        split="test",
        family_level=family,
        seed=index,
        world_id=f"world-{index}",
        world_flat=np.zeros(12, dtype=np.float32),
        frames=frames,
        hidden_start=3,
        hidden_end=5,
        reappearance_step=6,
        target_states=states,
        target_mask=np.ones(12, dtype=bool),
        oracle_pre_belief=target,
        actual_path_belief=target,
    )


def test_evaluation_saves_reconstructable_records_and_grouped_summary(tmp_path) -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    config = replace(config, training=replace(config.training, epochs=1, batch_size=2))
    examples = [record(0, "W0"), record(1, "W1")]
    training = train_model1(
        config, examples, examples, tmp_path / "train", device=torch.device("cpu"), seed=7
    )

    summary = evaluate_model1(
        training.best_checkpoint,
        examples,
        tmp_path / "evaluation",
        device=torch.device("cpu"),
        seed=7,
    )

    assert summary.count == 2
    assert summary.per_family["W0"]["count"] == 1
    assert summary.per_family["W1"]["count"] == 1
    assert "pre_min_ade" in summary.overall
    assert "min_ade_improvement" in summary.overall
    assert set(summary.calibration) == {"pre", "post"}
    assert summary.config_hash and summary.checkpoint_hash
    assert (tmp_path / "evaluation" / "summary.json").is_file()
    assert len(summary.record_paths) == 2

    loaded = load_prediction_record(summary.record_paths[0])
    assert loaded.example_id == "example-0"
    assert loaded.arrays["pre_frames"].shape == (16, 1, 64, 64)
    assert loaded.arrays["pre_positions"].shape == (8, 12, 2)
    assert loaded.arrays["pre_belief"].shape == (64, 64)
    assert "world_flat" not in loaded.arrays
