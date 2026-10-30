import json

import numpy as np
import pytest
import torch

from perception_env.model2.config import EvaluationConfig, FixtureConfig, TrainingConfig
from perception_env.model2.contracts import WorldPrediction, WorldTargets
from perception_env.model2.evaluate import evaluate_model, score_predictions
from perception_env.model2.fixtures import generate_fixtures
from perception_env.model2.train import train_model
from perception_env.world import WorldSchema


def tiny_config():
    return FixtureConfig(
        world_count=3,
        levels=("W1",),
        episodes_per_world=1,
        size=16,
        steps=60,
        max_context_attempts=4,
        world_attempt_multiplier=100,
        entry_resolution=(1, 1, 1, 1),
        audit_bank_size=2,
        audit_probe_count=4,
        heldout_probe_count=5,
        x_bounds=(0.15, 0.15),
        y_bounds=(0.65, 0.65),
        vx_bounds=(0.25, 0.25),
        vy_bounds=(0.0, 0.0),
    )


def prediction_for(worlds):
    schema = WorldSchema.v1()
    flat = np.stack([w.to_flat(schema) for w in worlds])
    return WorldPrediction(
        torch.tensor(np.where(flat[:, [0, 2, 4, 6, 9]], 100.0, -100.0)),
        torch.tensor(flat[:, [1, 3, 5, 7, 8, 10, 11]]),
    )


def test_exact_predictions_and_empty_rasters_have_explicit_denominators():
    schema = WorldSchema.v1()
    world = schema.world()
    target = WorldTargets(world.to_flat(schema)[None], np.zeros((1, 7), bool))
    result = score_predictions(
        prediction_for([world]),
        target,
        {
            "true_labels": np.zeros((1, 2, 3), bool),
            "predicted_labels": np.zeros((1, 2, 3), bool),
            "true_rasters": np.zeros((1, 2, 8, 8), bool),
            "predicted_rasters": np.zeros((1, 2, 8, 8), bool),
            "world_ids": ["a"],
            "family_levels": ["W0"],
        },
    )
    assert result["possibility_agreement"] == result["raster_iou"] == 1
    assert result["positive_recall"]["value"] is None
    assert result["positive_recall"]["denominator"] == 0
    assert result["negative_recall"]["value"] == 1
    assert result["empty_raster_pairs"] == 2
    assert result["parameter_mae"]["value"] is None
    assert result["activity_exact_match"] == 1


def test_equivalent_codes_and_unresolved_masks_are_not_exact_recovery():
    schema = WorldSchema.v1()
    true = schema.world(vertical_acceleration={"a_y": 0.2})
    predicted = schema.world(vertical_acceleration={"a_y": 0.4})
    target = WorldTargets(true.to_flat(schema)[None], np.ones((1, 7), bool))
    evidence = {
        "true_labels": np.ones((1, 1, 2), bool),
        "predicted_labels": np.ones((1, 1, 2), bool),
        "true_rasters": np.zeros((1, 1, 8, 8), bool),
        "predicted_rasters": np.ones((1, 1, 8, 8), bool),
        "world_ids": ["a"],
        "family_levels": ["W1"],
    }
    result = score_predictions(prediction_for([predicted]), target, evidence)
    assert result["exact_code_match"] == 0
    assert result["possibility_equivalent_rate"] == 1
    assert result["raster_iou"] == 0
    assert result["parameter_mae"]["denominator"] == 1
    assert result["parameter_mae"]["value"] == pytest.approx(0.1)
    assert result["physical_parameter_mae"]["value"] == pytest.approx(0.2)
    unresolved = score_predictions(
        prediction_for([predicted]), target, {**evidence, "audit_statuses": ["unresolved"]}
    )
    assert unresolved["parameter_mae"]["denominator"] == 0


def test_evaluation_regenerates_saved_labels_and_enforces_test_split(tmp_path):
    data = generate_fixtures(tiny_config(), tmp_path / "data")
    run = train_model(data.root, TrainingConfig(epochs=1), tmp_path / "train")
    output = evaluate_model(
        run.best_checkpoint,
        data.root,
        EvaluationConfig(search_seeds=(7,), budget=10),
        tmp_path / "eval",
    )
    assert output.prediction_path.exists()
    manifest = json.loads(output.manifest_path.read_text())
    checkpoint = torch.load(run.best_checkpoint, weights_only=True)
    assert set(manifest["row_indices"]).isdisjoint(checkpoint["provenance"]["train_row_indices"])
    assert set(manifest["row_indices"]).isdisjoint(
        checkpoint["provenance"]["validation_row_indices"]
    )
    with np.load(output.prediction_path) as arrays:
        assert arrays["true_labels"].shape == arrays["predicted_labels"].shape == (1, 4, 5)
        assert arrays["true_rasters"].shape == arrays["predicted_rasters"].shape
    assert output.metrics["shattering"][0]["metric_name"] == "context_conditioned"
    assert output.metrics["shattering"][0]["seeds"] == [7]
    assert output.metrics["heldout_scope"] in {"new_probes_same_contexts", "new_contexts"}
    # A tampered saved label cannot silently become oracle truth.
    with np.load(data.root / "evaluation.npz") as source:
        altered = {name: source[name] for name in source.files}
    altered["heldout_labels"] = ~altered["heldout_labels"]
    np.savez_compressed(data.root / "evaluation.npz", **altered)
    with pytest.raises(ValueError, match="hash"):
        evaluate_model(run.best_checkpoint, data.root, EvaluationConfig(), tmp_path / "bad")
