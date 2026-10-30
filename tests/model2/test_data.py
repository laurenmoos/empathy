import json
from pathlib import Path

import numpy as np
import pytest

from perception_env.model2.config import FixtureConfig, TrainingConfig, load_config
from perception_env.model2.contracts import BeliefInputs, WorldTargets
from perception_env.model2.data import BeliefDataset, collate_examples, save_dataset
from perception_env.world import WorldSchema


def test_dataset_roundtrip_only_opens_input_targets_manifest(tmp_path, monkeypatch):
    schema = WorldSchema.v1()
    x = BeliefInputs(np.zeros((3, 4, 3, 8, 8)), np.ones((3, 4), bool))
    y = WorldTargets(np.stack([schema.world().to_flat(schema)] * 3), np.zeros((3, 7), bool))
    root = tmp_path / "dataset"
    save_dataset(root, x, y, np.array([0, 1, 2]), {"world_ids": ["a", "b", "c"]})
    original = np.load

    def guarded(file, *args, **kwargs):
        assert Path(file).name in {"inputs.npz", "targets.npz"}
        return original(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    dataset = BeliefDataset(root, "train")
    assert len(dataset) == 1
    batch = collate_examples([dataset[0]])
    assert batch.beliefs.shape == (1, 4, 3, 8, 8)
    assert batch.world_flat.shape == (1, 12)
    assert dataset.manifest["world_ids"] == ["a", "b", "c"]
    (root / "manifest.json").write_text(json.dumps({"wrong": True}))
    with pytest.raises(ValueError):
        BeliefDataset(root, "train")


def test_config_defaults_and_unknown_keys(tmp_path):
    assert FixtureConfig().history_length == 4
    assert TrainingConfig().device == "cpu"
    path = tmp_path / "config.yaml"
    path.write_text("fixtures: {}\ntraining: {}\nevaluation: {}\n")
    assert load_config(path)[1].batch_size == 32
    path.write_text("fixtures: {secret_frames: true}\ntraining: {}\nevaluation: {}\n")
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)


@pytest.mark.parametrize("kwargs", [{"epochs": 1.5}, {"batch_size": True}, {"seed": -1}])
def test_training_config_rejects_nonintegral_counts_and_negative_seed(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"entry_resolution": (2, 3, 3, 3)}, {"world_count": 1.5}, {"steps": 2.5}]
)
def test_fixture_config_preserves_exact_grid_center_and_integer_counts(kwargs):
    with pytest.raises(ValueError):
        FixtureConfig(**kwargs)
