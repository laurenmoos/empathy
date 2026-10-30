import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from perception_env.model2.config import TrainingConfig
from perception_env.model2.contracts import BeliefInputs, WorldTargets
from perception_env.model2.data import save_dataset
from perception_env.model2.network import InverseWorldCNN
from perception_env.model2.train import load_model, predict, train_model
from perception_env.world import WorldSchema


def learning_dataset(root):
    """Deliberately repeated labels for an overfit check, not a scientific split."""
    schema = WorldSchema.v1()
    worlds = [schema.world(), schema.world(vertical_acceleration={"a_y": -0.4})]
    values = np.zeros((12, 4, 3, 16, 16), np.float32)
    targets = []
    for index in range(12):
        if index % 2:
            values[index, -1, 1, 2:8, 2:8] = 1
        else:
            values[index, -1, 1, 8:14, 8:14] = 1
        values[index, -1, 2] = values[index, -1, 1]
        targets.append(worlds[index % 2].to_flat(schema))
    mask = np.zeros((12, 4), bool)
    mask[:, -1] = True
    save_dataset(
        root,
        BeliefInputs(values, mask),
        WorldTargets(np.stack(targets), np.ones((12, 7), bool)),
        np.array([0] * 8 + [1] * 2 + [2] * 2),
        {"fixture": "overfit_only"},
    )
    return BeliefInputs(values, mask)


def test_real_cpu_training_overfits_and_roundtrips(tmp_path):
    root = tmp_path / "data"
    inputs = learning_dataset(root)
    config = TrainingConfig(epochs=80, batch_size=8, learning_rate=0.01, weight_decay=0.0)
    run = train_model(root, config, tmp_path / "run")
    metrics = json.loads(run.metrics_path.read_text())
    assert metrics["epochs"][-1]["train_total"] < 0.2 * metrics["initial_train_loss"]
    assert metrics["selected_epoch"] == min(
        range(len(metrics["epochs"])), key=lambda n: metrics["epochs"][n]["validation_total"]
    )
    model = load_model(run.best_checkpoint)
    first = predict(model, inputs)
    second = predict(load_model(run.best_checkpoint), inputs)
    assert torch.equal(first.activity_logits, second.activity_logits)
    assert not torch.equal(first.activity_logits[0], first.activity_logits[1])
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest["train_row_indices"] == list(range(8))
    assert manifest["validation_row_indices"] == [8, 9]
    assert not any(i in manifest["train_row_indices"] for i in [10, 11])
    assert manifest["parameter_count"] > 1000
    with pytest.raises(ValueError, match="hash"):
        load_model(run.best_checkpoint, expected_dataset_hash="wrong")


def test_two_cpu_runs_same_seed_same_weights_and_metrics(tmp_path):
    data = tmp_path / "data"
    learning_dataset(data)
    config = TrainingConfig(epochs=2, batch_size=4)
    one = train_model(data, config, tmp_path / "one")
    two = train_model(data, config, tmp_path / "two")
    a, b = load_model(one.best_checkpoint), load_model(two.best_checkpoint)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert json.loads(one.metrics_path.read_text()) == json.loads(two.metrics_path.read_text())
    with pytest.raises(FileExistsError):
        train_model(data, config, tmp_path / "one")
    with pytest.raises(ValueError):
        train_model(data, replace(config, device="cuda"), tmp_path / "gpu")
    assert not (tmp_path / "gpu").exists()


def test_changed_checkpoint_schema_rejected(tmp_path):
    data = tmp_path / "data"
    learning_dataset(data)
    run = train_model(data, TrainingConfig(epochs=1), tmp_path / "run")
    checkpoint = torch.load(run.best_checkpoint, weights_only=True)
    checkpoint["world_schema"] = "wrong"
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="schema"):
        load_model(broken)


def test_inference_only_accepts_input_type():
    with pytest.raises(TypeError):
        predict(InverseWorldCNN(), {"beliefs": np.zeros((1, 4, 3, 16, 16))})
