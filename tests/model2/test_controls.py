import json

import numpy as np
import pytest
import torch

from perception_env.model2.config import TrainingConfig
from perception_env.model2.contracts import BeliefInputs, WorldTargets
from perception_env.model2.controls import fit_constant_baseline, run_controls
from perception_env.model2.data import save_dataset
from perception_env.model2.train import load_model
from perception_env.world import WorldSchema


def test_constant_baseline_only_uses_active_identifiable_train_targets():
    schema = WorldSchema.v1()
    worlds = [schema.world(), schema.world(vertical_acceleration={"a_y": -0.4})]
    targets = WorldTargets(np.stack([w.to_flat(schema) for w in worlds]), np.ones((2, 7), bool))
    prediction = fit_constant_baseline(targets)
    assert prediction.activity_logits.shape == (1, 5)
    assert torch.sigmoid(prediction.activity_logits)[0, 0] == 0.5
    assert prediction.normalized_parameters[0, 0].item() == pytest.approx(0.3)
    assert torch.equal(prediction.normalized_parameters[0, 1:], torch.full((6,), 0.5))


def test_control_runs_use_same_split_seed_and_saved_transform(tmp_path):
    values = np.zeros((6, 4, 3, 8, 8), np.float32)
    mask = np.ones((6, 4), bool)
    data = tmp_path / "data"
    save_dataset(
        data,
        BeliefInputs(values, mask),
        WorldTargets(np.zeros((6, 12)), np.zeros((6, 7), bool)),
        np.array([0, 0, 1, 1, 2, 2]),
        {},
    )
    outputs = run_controls(data, TrainingConfig(epochs=1, batch_size=2), tmp_path / "controls")
    assert set(outputs) == {"constant", "single_belief", "shuffled_history"}
    for mode in ("single_belief", "shuffled_history"):
        assert load_model(outputs[mode]).input_mode == mode
    payload = json.loads(outputs["constant"].read_text())
    assert payload["fit_row_indices"] == [0, 1]
    assert payload["architecture"] == "constant-world-baseline-v1"
