import json
from pathlib import Path

import numpy as np

from perception_env.stream_records import load_streaming_run
from perception_env.streaming_demo import load_streaming_demo_config, run_quick_streaming_demo


def test_streaming_config_registers_the_small_continual_defaults() -> None:
    config = load_streaming_demo_config(Path("configs/streaming/v1.yaml"))

    assert config.history_length == 4
    assert config.event_count == 8
    assert config.max_feedback_iterations == 3
    assert config.replay_capacity == 64
    assert config.warmup == 4
    assert config.batch_size == 4
    assert config.enable_weight_updates is False


def test_quick_streaming_demo_is_complete_and_deterministic(tmp_path) -> None:
    first = run_quick_streaming_demo(tmp_path / "first", seed=17, event_count=3)
    second = run_quick_streaming_demo(tmp_path / "second", seed=17, event_count=3)

    assert first.deterministic_hash == second.deterministic_hash
    assert first.timeline_path.exists()
    assert first.index_path.exists()
    first_manifest = json.loads(first.manifest_path.read_text())
    second_manifest = json.loads(second.manifest_path.read_text())
    assert first_manifest["event_count"] == 3
    assert first_manifest["accepted_count"] == 3
    assert first_manifest["checkpoint_hashes"]["model1"].startswith("fixture-")
    assert first_manifest["checkpoint_hashes"]["model2"].startswith("fixture-")
    assert first_manifest["latent_update_count"] == 6
    assert first_manifest["weight_update_count"] == 0
    assert first_manifest["coverage_by_event"] == [True, True, True]
    assert set(first_manifest["output_content_hashes"]) == set(
        first_manifest["output_paths"]
    )
    assert all(
        len(value) == 64 for value in first_manifest["output_content_hashes"].values()
    )
    assert first_manifest["output_content_hashes"] == second_manifest["output_content_hashes"]

    first_run = load_streaming_run(first.run_path)
    second_run = load_streaming_run(second.run_path)
    assert [item.status for item in first_run.updates] == ["covered"] * 3
    assert [len(item.iterations) for item in first_run.updates] == [2, 2, 2]
    for left, right in zip(first_run.updates, second_run.updates, strict=True):
        assert np.array_equal(left.event.pre, right.event.pre)
        assert np.array_equal(left.event.post, right.event.post)
        assert np.array_equal(
            left.iterations[-1].prediction.flat_world,
            right.iterations[-1].prediction.flat_world,
        )


def test_quick_demo_refuses_to_replace_an_existing_output(tmp_path) -> None:
    output = tmp_path / "demo"
    run_quick_streaming_demo(output, seed=3, event_count=1)

    try:
        run_quick_streaming_demo(output, seed=3, event_count=1)
    except FileExistsError as error:
        assert error.args[0] == output
    else:
        raise AssertionError("existing output should not be replaced")
