import json

import numpy as np

from perception_env.cli import build_demo, inspect_run


def test_tiny_demo_builds_visual_benchmark_and_belief_only_model2_input(tmp_path) -> None:
    config = "configs/world_families/v1.yaml"
    run = tmp_path / "demo"

    manifest = build_demo(config_path=config, seed=7, output=run, force=False, quick=True)

    required = {
        "world-family-contact-sheet.png",
        "trajectory-W0.png",
        "observation-W0.png",
        "belief-W0.png",
        "reachability-overlays.png",
        "shattering-matrix.png",
        "equivalence-gallery.png",
        "index.html",
        "model2-input.npz",
        "manifest.json",
    }
    assert required.issubset({path.name for path in run.iterdir()})
    assert manifest["command"] == "demo"
    assert manifest["seed"] == 7
    assert manifest["rejection_counts"]["accepted"] == 3
    assert manifest["shattering"]["metric_name"] == "context_conditioned"
    assert manifest["shattering"]["world_count"] == 3
    assert manifest["shattering"]["probe_count"] >= 3
    assert manifest["shattering"]["uncertainty_grid"]
    assert manifest["shattering"]["tolerance"] > 0
    assert manifest["shattering"]["algorithm"] == "exhaustive_or_seeded_subset_search"
    assert manifest["shattering"]["budget"] > 0
    assert manifest["shattering"]["seeds"] == [7]
    assert manifest["model2_history"]["world_level"] == "W2"
    assert manifest["model2_history"]["event_count"] == 4
    assert manifest["model2_history"]["all_events_observed"] is True

    with np.load(run / "model2-input.npz", allow_pickle=False) as model2:
        assert set(model2.files) == {
            "pre_belief",
            "post_belief",
            "belief_difference",
            "entropy",
            "history_mask",
            "target_world",
        }
        forbidden = {
            "raw_frames",
            "positions",
            "visibility_masks",
            "detector_features",
            "event_marker",
        }
        assert forbidden.isdisjoint(model2.files)

    inspection = tmp_path / "inspection"
    inspect_manifest = inspect_run(run=run, output=inspection, force=False)
    assert inspect_manifest["command"] == "inspect"
    assert (inspection / "world-family-contact-sheet.png").stat().st_size > 0
    assert (inspection / "index.html").stat().st_size > 0
    saved_hash = json.loads((run / "manifest.json").read_text())["deterministic_hash"]
    assert saved_hash == manifest["deterministic_hash"]
