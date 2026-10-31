from dataclasses import replace

import numpy as np
import pytest

from empathy.model2.config import FixtureConfig
from empathy.model2.data import BeliefDataset
from empathy.model2.fixtures import (
    build_fixture_event,
    generate_fixtures,
    split_world_groups,
)
from empathy.physics.simulate import BallState
from empathy.physics.world import WorldSchema


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


def test_split_groups_duplicate_codes():
    ids = ["a", "b", "c", "a", "d", "b"]
    split = split_world_groups(ids, (0.5, 0.25, 0.25), 7)
    assert set(split) == {0, 1, 2}
    assert split[0] == split[3] and split[1] == split[5]
    assert np.array_equal(split, split_world_groups(ids, (0.5, 0.25, 0.25), 7))
    with pytest.raises(ValueError):
        split_world_groups(["a"], (0.7, 0.15, 0.15), 7)


def test_event_assimilates_visible_reappearance_and_grid_contains_truth():
    schema = WorldSchema.v1()
    true = schema.world()
    prior = schema.world(vertical_acceleration={"a_y": -0.1})
    event, provenance = build_fixture_event(
        true, prior, BallState(0.15, 0.65, 0.25, 0), tiny_config(), 1, 17
    )
    assert event.step == 1
    assert provenance["reappearance_visible"]
    assert provenance["reappearance_step"] > provenance["hidden_start"]
    assert provenance["true_world"] != provenance["prior_world"]
    assert np.any(event.post)
    assert np.array_equal(event.difference, event.post - event.pre)
    assert provenance["entry_state"][0] == provenance["entry_grid"]["x_bounds"][0] + 0.005


def test_fixture_roundtrip_reproducible_history_and_split_isolation(tmp_path):
    first = generate_fixtures(tiny_config(), tmp_path / "one")
    second = generate_fixtures(tiny_config(), tmp_path / "two")
    assert first.dataset_hash == second.dataset_hash
    assert first.accepted_worlds == 3
    manifest = BeliefDataset(first.root, "train").manifest
    assert len(set(manifest["world_ids"])) == 3
    for row in manifest["rows"]:
        assert len(row["events"]) == 4
        assert [event["ordinal"] for event in row["events"]] == [0, 1, 2, 3]
        assert len({event["true_world"] for event in row["events"]}) == 1
    with np.load(first.root / "inputs.npz") as arrays:
        assert set(arrays.files) == {"beliefs", "history_mask"}
    with np.load(first.root / "evaluation.npz") as arrays:
        assert arrays["heldout_labels"].shape == (3, 4, 5)
    assert (first.root / "evaluation.json").exists()


def test_bounded_rejection_does_not_publish_partial_dataset(tmp_path):
    impossible = replace(tiny_config(), world_attempt_multiplier=1, max_context_attempts=1, steps=1)
    with pytest.raises(ValueError, match="incomplete"):
        generate_fixtures(impossible, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_input_manifest_does_not_contain_first_order_event_payload(tmp_path):
    import json

    generated = generate_fixtures(tiny_config(), tmp_path / "data")
    manifest = json.loads((generated.root / "manifest.json").read_text())
    for row in manifest["rows"]:
        for event in row["events"]:
            assert set(event) <= {"ordinal", "true_world", "prior_world", "source", "seed"}


def test_fixture_dataset_trains_real_cnn(tmp_path):
    import json

    from empathy.model2.config import TrainingConfig
    from empathy.model2.train import train_model

    generated = generate_fixtures(tiny_config(), tmp_path / "data")
    run = train_model(generated.root, TrainingConfig(epochs=2, batch_size=1), tmp_path / "run")
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest["initial_state_hash"] != manifest["final_state_hash"]
    assert manifest["dataset_hash"] == generated.dataset_hash


def test_shipped_smoke_configuration_completes_requested_world_bank(tmp_path):
    from pathlib import Path

    from empathy.model2.config import load_config

    config, _, _ = load_config(Path("configs/model2/smoke_v1.yaml"))
    result = generate_fixtures(config, tmp_path / "smoke")
    assert result.accepted_worlds == config.world_count == 12
