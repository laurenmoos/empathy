import json
from pathlib import Path

import numpy as np
import pytest
import torch

from empathy.model2.config import FixtureConfig, TrainingConfig
from empathy.model2.first_order_ablation import train_first_order_control
from empathy.model2.fixtures import generate_fixtures
from empathy.model2.train import load_model


def test_privileged_control_is_isolated_and_cannot_load_as_production(tmp_path):
    config = FixtureConfig(
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
    fixture = generate_fixtures(config, tmp_path / "data")
    with np.load(fixture.root / "privileged_features.npz") as features:
        assert features["features"].shape == (3, 4, 9)
    run = train_first_order_control(fixture.root, TrainingConfig(epochs=2), tmp_path / "ablation")
    saved = torch.load(run.best_checkpoint, weights_only=True)
    assert saved["architecture"] == "privileged_first_order_ablation"
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest["initial_state_hash"] != manifest["final_state_hash"]
    with pytest.raises(ValueError, match="architecture"):
        load_model(run.best_checkpoint)
    for name in ("network", "train", "data", "evaluate", "adapters"):
        source = Path(f"src/empathy/model2/{name}.py").read_text()
        assert "first_order_ablation import" not in source
