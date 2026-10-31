import json
from pathlib import Path

import numpy as np
import torch

from empathy.model2.config import EvaluationConfig, TrainingConfig
from empathy.model2.contracts import WorldTargets
from empathy.model2.evaluate import evaluate_model, score_predictions
from empathy.model2.fixtures import generate_fixtures
from empathy.model2.network import InverseWorldCNN
from empathy.model2.train import train_model
from empathy.physics.world import WorldSchema

from .test_evaluate import prediction_for, tiny_config


def test_evaluation_opens_privileged_context_only_after_belief_forward(tmp_path, monkeypatch):
    data = generate_fixtures(tiny_config(), tmp_path / "data")
    run = train_model(data.root, TrainingConfig(epochs=1), tmp_path / "train")
    forward_done = False
    forward = InverseWorldCNN.forward
    read_text = Path.read_text

    def checked_forward(self, beliefs, history_mask):
        nonlocal forward_done
        assert beliefs.shape[1:3] == (4, 3)
        assert history_mask.dtype == torch.bool
        output = forward(self, beliefs, history_mask)
        forward_done = True
        return output

    def checked_read(path, *args, **kwargs):
        if path.name == "evaluation.json":
            assert forward_done
        assert path.name != "privileged_features.npz"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(InverseWorldCNN, "forward", checked_forward)
    monkeypatch.setattr(Path, "read_text", checked_read)
    result = evaluate_model(
        run.best_checkpoint,
        data.root,
        EvaluationConfig(search_seeds=(7,), budget=10),
        tmp_path / "eval",
    )
    assert forward_done
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metrics_hash"]
    assert manifest["shattering_world_ids"] == sorted(set(manifest["world_ids"]))


def test_empty_predicted_sets_and_duplicate_worlds_are_retained_in_macro_metrics():
    world = WorldSchema.v1().world()
    targets = WorldTargets(np.stack([world.to_flat(WorldSchema.v1())] * 3), np.zeros((3, 7), bool))
    metrics = score_predictions(
        prediction_for([world] * 3),
        targets,
        {
            "true_labels": np.ones((3, 1, 2), bool),
            "predicted_labels": np.array([[[1, 1]], [[0, 0]], [[0, 0]]], bool),
            "true_rasters": np.ones((3, 1, 8, 8), bool),
            "predicted_rasters": np.zeros((3, 1, 8, 8), bool),
            "world_ids": ["a", "a", "b"],
            "family_levels": ["W1"] * 3,
        },
    )
    assert metrics["world_macro_agreement"] == 0.25
    assert metrics["raster_iou"] == 0
    assert metrics["empty_predicted_contexts"] == 3
    assert len(metrics["per_example"]) == 3
