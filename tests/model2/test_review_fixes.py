import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from empathy.model2.config import EvaluationConfig, TrainingConfig, load_config


def test_torch_is_a_normal_dependency_and_lock_matches():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert any(item.startswith("torch") for item in project["dependencies"])
    assert "ml" not in project["optional-dependencies"]
    lock = tomllib.loads(Path("uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "empathy")
    assert any(item["name"] == "torch" for item in package["dependencies"])
    assert "ml" not in package["optional-dependencies"]


def test_base_package_import_does_not_require_torch():
    program = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, BlockTorch())
import empathy
from empathy.physics.world import WorldSchema
assert len(WorldSchema.v1().field_order) == 12
"""
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mode", ["full", "single_belief", "shuffled_history"])
def test_training_config_accepts_registered_control_modes(mode):
    assert TrainingConfig(input_mode=mode).input_mode == mode


@pytest.mark.parametrize(
    "values",
    [
        {"max_n": -1},
        {"max_n": 1.5},
        {"max_n": True},
        {"budget": 0},
        {"budget": -1},
        {"budget": float("nan")},
        {"budget": 1.5},
        {"search_seeds": ()},
        {"search_seeds": (-1,)},
        {"search_seeds": (True,)},
        {"search_seeds": (1.5,)},
        {"search_seeds": (7, 7)},
        {"search_seeds": (float("inf"),)},
    ],
)
def test_evaluation_config_rejects_invalid_search_settings(values):
    with pytest.raises(ValueError):
        EvaluationConfig(**values)


def test_evaluation_config_accepts_zero_dimension_and_yaml_controls(tmp_path):
    assert EvaluationConfig(max_n=0, budget=1, search_seeds=(0,)).max_n == 0
    path = tmp_path / "config.yaml"
    path.write_text(
        "fixtures: {}\ntraining: {input_mode: shuffled_history}\n"
        "evaluation: {max_n: 0, budget: 1, search_seeds: [0, 5]}\n"
    )
    _, training, evaluation = load_config(path)
    assert training.input_mode == "shuffled_history"
    assert evaluation.search_seeds == (0, 5)


def _control_inputs(count=6):
    from empathy.model2.contracts import BeliefInputs

    values = np.zeros((count, 4, 3, 8, 8), dtype=np.float32)
    mask = np.ones((count, 4), dtype=bool)
    for row in range(count):
        for event in range(4):
            values[row, event, 0, event : event + 2, row % 6 : row % 6 + 2] = 0.2
            values[row, event, 1, event : event + 2, row % 6 : row % 6 + 2] = 0.8
    values[:, :, 2] = values[:, :, 1] - values[:, :, 0]
    mask[0, :2] = False
    values[0, :2] = 0
    return BeliefInputs(values, mask)


def _control_dataset(root):
    from empathy.model2.contracts import WorldTargets
    from empathy.model2.data import save_dataset

    inputs = _control_inputs()
    save_dataset(
        root,
        inputs,
        WorldTargets(np.zeros((6, 12)), np.zeros((6, 7), bool)),
        np.array([0, 0, 1, 1, 2, 2]),
        {
            "world_ids": ["a", "a", "b", "b", "c", "c"],
            "rows": [{"episode_id": f"episode-{i}"} for i in range(6)],
        },
    )
    return inputs


def test_single_belief_control_is_latest_post_only():
    from empathy.model2.transforms import transform_control

    inputs = _control_inputs()
    result = transform_control(inputs, mode="single_belief", seed=7)
    assert np.array_equal(result.beliefs[:, -1, 0], inputs.beliefs[:, -1, 1])
    assert np.array_equal(result.beliefs[:, -1, 1], inputs.beliefs[:, -1, 1])
    assert not result.beliefs[:, :, 2].any()
    assert not result.beliefs[:, :-1].any()
    assert np.array_equal(result.history_mask.sum(axis=1), np.ones(6))


def test_shuffle_preserves_event_triples_and_replays_independent_of_batch_order():
    from empathy.model2.contracts import BeliefInputs
    from empathy.model2.transforms import transform_control

    inputs = _control_inputs()
    result = transform_control(inputs, mode="shuffled_history", seed=7)
    replay = transform_control(inputs, mode="shuffled_history", seed=7)
    assert np.array_equal(result.beliefs, replay.beliefs)
    assert not np.array_equal(result.beliefs, inputs.beliefs)
    for row in range(6):
        single = transform_control(
            BeliefInputs(inputs.beliefs[row : row + 1], inputs.history_mask[row : row + 1]),
            mode="shuffled_history",
            seed=7,
        )
        assert np.array_equal(single.beliefs[0], result.beliefs[row])
        valid = inputs.history_mask[row]
        assert sorted(x.tobytes() for x in inputs.beliefs[row, valid]) == sorted(
            x.tobytes() for x in result.beliefs[row, valid]
        )
    assert np.array_equal(result.history_mask, inputs.history_mask)


@pytest.mark.parametrize("mode", ["single_belief", "shuffled_history"])
def test_controls_train_and_checkpoint_replays_exactly_one_transform(tmp_path, mode):
    import torch

    from empathy.model2.network import InverseWorldCNN
    from empathy.model2.train import load_model, predict, train_model
    from empathy.model2.transforms import transform_control

    inputs = _control_dataset(tmp_path / "data")
    run = train_model(
        tmp_path / "data", TrainingConfig(epochs=1, batch_size=2, input_mode=mode), tmp_path / "run"
    )
    model = load_model(run.best_checkpoint)
    assert model.input_mode == mode
    assert model.control_seed == 7
    actual = predict(model, inputs).activity_logits
    manual = transform_control(inputs, mode=mode, seed=7)
    plain = InverseWorldCNN()
    plain.load_state_dict(model.state_dict())
    expected = predict(plain, manual).activity_logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, predict(model, inputs, batch_size=1).activity_logits)
    replay = load_model(run.best_checkpoint)
    assert torch.equal(actual, predict(replay, inputs).activity_logits)


def test_standalone_checkpoints_contain_split_runtime_and_selection_provenance(tmp_path):
    import json

    import torch

    from empathy.model2.train import train_model

    _control_dataset(tmp_path / "data")
    run = train_model(tmp_path / "data", TrainingConfig(epochs=2, batch_size=2), tmp_path / "run")
    metrics = json.loads(run.metrics_path.read_text())
    best_validation = min(row["validation_total"] for row in metrics["epochs"])
    for name in ("best.pt", "final.pt"):
        payload = torch.load(run.root / name, weights_only=True)
        assert payload["best_validation_loss"] == best_validation
        assert payload["selected_epoch"] == metrics["selected_epoch"]
        provenance = payload["provenance"]
        assert provenance["train_row_indices"] == [0, 1]
        assert provenance["validation_row_indices"] == [2, 3]
        assert provenance["train_row_ids"] == ["episode-0", "episode-1"]
        assert provenance["validation_row_ids"] == ["episode-2", "episode-3"]
        assert provenance["train_world_ids"] == ["a", "a"]
        assert provenance["validation_world_ids"] == ["b", "b"]
        assert provenance["dataset_hash"] == payload["dataset_hash"]
        assert len(provenance["input_hash"]) == len(provenance["target_hash"]) == 64
        assert provenance["runtime"]["device"] == "cpu"
        assert provenance["runtime"]["torch"] == str(torch.__version__)
        assert provenance["runtime"]["numpy"] == np.__version__
        assert provenance["runtime"]["python"]
        assert provenance["runtime"]["deterministic_algorithms"] is True
