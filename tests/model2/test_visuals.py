import json

import numpy as np
import pytest

from empathy.model2.data import array_hash
from empathy.model2.visualize import render_evaluation


def test_saved_visuals_never_call_oracle_or_inference(tmp_path, monkeypatch):
    import empathy.model2.evaluate as evaluation
    import empathy.model2.train as training

    def forbidden(*args, **kwargs):
        raise AssertionError("rendering must read saved results only")

    monkeypatch.setattr(evaluation, "_reachability", forbidden)
    monkeypatch.setattr(training, "predict", forbidden)
    root = tmp_path / "eval"
    root.mkdir()
    arrays = {
        "beliefs": np.zeros((1, 4, 3, 8, 8)),
        "history_mask": np.array([[0, 0, 0, 1]]),
        "decoded_worlds": np.zeros((1, 12)),
        "true_worlds": np.zeros((1, 12)),
        "true_rasters": np.zeros((1, 4, 8, 8), bool),
        "predicted_rasters": np.zeros((1, 4, 8, 8), bool),
        "normalized_parameters": np.zeros((1, 7)),
    }
    np.savez_compressed(root / "predictions.npz", **arrays)
    manifest = {
        "array_hash": array_hash(arrays),
        "dataset_hash": "dataset",
        "checkpoint_hash": "checkpoint",
        "seed": 7,
        "config": {},
        "world_ids": ["a"],
        "source": "privileged_oracle_bootstrap",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "per_example": [{"equivalent": True, "exact_code": True}],
                "per_operator": {
                    "operator": {"precision": {"value": None}, "recall": {"value": None}}
                },
                "parameter_errors": [{"value": None, "denominator": 0}] * 7,
            }
        )
    )
    artifacts = render_evaluation(tmp_path, root, tmp_path / "visuals")
    assert len(artifacts) >= 8
    assert (tmp_path / "visuals" / "index.html").exists()
    for artifact in artifacts:
        assert artifact.path.exists()
        metadata = json.loads(artifact.path.with_suffix(".json").read_text())
        assert metadata["dataset_hash"] == "dataset"
        assert metadata["checkpoint_hash"] == "checkpoint"
        assert metadata["source"] == "privileged_oracle_bootstrap"
    with pytest.raises(FileExistsError):
        render_evaluation(tmp_path, root, tmp_path / "visuals")


def test_exemplars_include_success_failure_empty_and_equivalent_when_available():
    from empathy.model2.visualize import select_exemplars

    metrics = {
        "per_example": [
            {"equivalent": True, "exact_code": True},
            {"equivalent": False, "exact_code": False},
            {"equivalent": True, "exact_code": False},
        ]
    }
    rasters = np.ones((3, 4, 8, 8), bool)
    rasters[1] = False
    selected = select_exemplars(["c", "b", "a"], metrics, rasters)
    assert selected == {"success": 2, "failure": 1, "empty": 1, "equivalent_distinct_code": 2}
