import json
from pathlib import Path

from perception_env.export_learned_example import (
    export_learned_example,
    export_learned_examples,
)


ROOT = Path(__file__).parents[1]
MODEL1_RUN = ROOT / "artifacts" / "model1-learned-demo"
MODEL2_RUN = ROOT / "artifacts" / "model2-learned-demo"


def test_export_learned_examples_preserves_held_out_order_and_boundaries(tmp_path) -> None:
    output = tmp_path / "learned-rollouts.json"

    export_learned_examples(
        MODEL1_RUN,
        MODEL2_RUN,
        output,
        ("test-000001", "test-000000"),
    )

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "learned-demo-v2"
    assert [rollout["example_id"] for rollout in payload["rollouts"]] == [
        "test-000001",
        "test-000000",
    ]
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
        assert "belief" in rollout["provenance"]["note"].lower()


def test_single_example_export_remains_a_v1_compatibility_wrapper(tmp_path) -> None:
    output = tmp_path / "single.json"

    export_learned_example(MODEL1_RUN, MODEL2_RUN, output, example_id="test-000000")

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "learned-demo-v1"
    assert payload["example_id"] == "test-000000"
    assert "rollouts" not in payload
