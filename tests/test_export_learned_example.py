import json
from pathlib import Path

import numpy as np
import torch

import perception_env.export_learned_example as exporter
from perception_env.model2.contracts import BeliefInputs, WorldPrediction


def _write_record(root: Path, example_id: str, reappearance_step: int) -> None:
    prediction = root / "model1" / "evaluation" / "records" / example_id
    truth = root / "model1" / "data" / "records" / example_id
    prediction.mkdir(parents=True)
    truth.mkdir(parents=True)
    belief = np.zeros((8, 8), dtype=np.float32)
    belief[3:5, 3:5] = 0.5
    np.savez(
        prediction / "arrays.npz",
        pre_belief=belief,
        post_belief=np.clip(belief + 0.25, 0, 1),
        target_positions=np.zeros((5, 2), dtype=np.float32),
        target_mask=np.ones(5, dtype=bool),
    )
    np.savez(
        truth / "arrays.npz",
        frames=np.zeros((5, 8, 8), dtype=np.uint8),
        world_flat=np.zeros(12, dtype=np.float32),
    )
    (truth / "metadata.json").write_text(
        json.dumps(
            {
                "example_id": example_id,
                "hidden_start": 1,
                "hidden_end": reappearance_step - 1,
                "reappearance_step": reappearance_step,
            }
        )
    )


def _fixture_runs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, list[BeliefInputs]]:
    _write_record(tmp_path, "test-000001", 3)
    _write_record(tmp_path, "test-000000", 4)
    model1 = tmp_path / "model1"
    model2 = tmp_path / "model2"
    checkpoint = model2 / "training" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint")
    captured: list[BeliefInputs] = []

    monkeypatch.setattr(exporter, "load_model", lambda _: object())

    def fake_predict(_, inputs: BeliefInputs) -> WorldPrediction:
        captured.append(inputs)
        return WorldPrediction(torch.full((1, 5), -10.0), torch.zeros((1, 7)))

    monkeypatch.setattr(exporter, "predict", fake_predict)
    return model1, model2, captured


def test_export_learned_examples_preserves_order_boundaries_and_reappearance(
    tmp_path, monkeypatch
) -> None:
    model1, model2, captured = _fixture_runs(tmp_path, monkeypatch)
    output = tmp_path / "learned-rollouts.json"

    exporter.export_learned_examples(
        model1,
        model2,
        output,
        ("test-000001", "test-000000"),
    )

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "learned-demo-v2"
    assert [rollout["example_id"] for rollout in payload["rollouts"]] == [
        "test-000001",
        "test-000000",
    ]
    assert [
        rollout["ground_truth"]["reappearance_step"] for rollout in payload["rollouts"]
    ] == [3, 4]
    assert [
        (
            rollout["ground_truth"]["hidden_start"],
            rollout["ground_truth"]["hidden_end"],
        )
        for rollout in payload["rollouts"]
    ] == [(1, 2), (1, 3)]
    for rollout in payload["rollouts"]:
        assert set(rollout) == {"example_id", "ground_truth", "model1", "model2", "provenance"}
        assert set(rollout["model2"]) == {
            "predicted_world_flat",
            "activity_probabilities",
            "normalized_parameters",
        }
        assert len(rollout["ground_truth"]["world_flat"]) == 12
        assert len(rollout["model2"]["predicted_world_flat"]) == 12
        assert rollout["provenance"]["model1_prediction_sha256"]
        assert rollout["provenance"]["model1_truth_sha256"]
        assert rollout["provenance"]["model2_checkpoint_sha256"]
    assert len(captured) == 2
    assert all(isinstance(inputs, BeliefInputs) for inputs in captured)
    assert all(inputs.beliefs.shape == (1, 4, 3, 8, 8) for inputs in captured)


def test_single_example_export_remains_a_v1_compatibility_wrapper(tmp_path, monkeypatch) -> None:
    model1, model2, _ = _fixture_runs(tmp_path, monkeypatch)
    output = tmp_path / "single.json"

    exporter.export_learned_example(model1, model2, output, example_id="test-000000")

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "learned-demo-v1"
    assert payload["example_id"] == "test-000000"
    assert payload["ground_truth"]["reappearance_step"] == 4
    assert "rollouts" not in payload
