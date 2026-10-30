import inspect

import numpy as np
import torch

from perception_env.model2.config import TrainingConfig
from perception_env.model2.contracts import BeliefInputs, WorldTargets
from perception_env.model2.data import save_dataset
from perception_env.model2.network import InverseWorldCNN
from perception_env.model2.train import load_model, predict, train_model
from perception_env.world import WorldSchema


def test_training_forward_arguments_exclude_supervision_and_first_order(tmp_path, monkeypatch):
    schema = WorldSchema.v1()
    values = np.zeros((6, 4, 3, 8, 8), np.float32)
    mask = np.ones((6, 4), bool)
    inputs = BeliefInputs(values, mask)
    targets = WorldTargets(np.stack([schema.world().to_flat(schema)] * 6), np.zeros((6, 7), bool))
    root = tmp_path / "data"
    save_dataset(root, inputs, targets, np.array([0, 0, 1, 1, 2, 2]), {})
    (root / "evaluation.json").write_text("deliberately invalid first-order sidecar")
    (root / "diagnostics").mkdir()
    (root / "diagnostics" / "frames.npz").write_bytes(b"not needed")
    original = InverseWorldCNN.forward
    observed = []

    def checked(self, beliefs, history_mask):
        observed.append((beliefs.shape, history_mask.shape))
        assert beliefs.ndim == 5 and beliefs.shape[1:3] == (4, 3)
        assert history_mask.dtype == torch.bool
        return original(self, beliefs, history_mask)

    monkeypatch.setattr(InverseWorldCNN, "forward", checked)
    run = train_model(root, TrainingConfig(epochs=1), tmp_path / "run")
    model = load_model(run.best_checkpoint)
    first = predict(model, inputs).activity_logits
    (root / "targets.npz").write_bytes(b"poisoned labels")
    (root / "evaluation.json").write_text('{"positions": [999,999]}')
    second = predict(model, inputs).activity_logits
    assert torch.equal(first, second)
    assert observed
    assert set(inspect.signature(original).parameters) == {"self", "beliefs", "history_mask"}
