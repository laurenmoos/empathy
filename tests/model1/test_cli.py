import json
from pathlib import Path

import yaml

from perception_env.model1.cli import main


def small_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(Path("configs/model1/v1.yaml").read_text())
    payload["dataset"]["counts"] = [4, 2, 4]
    payload["dataset"]["quick_counts"] = [4, 2, 4]
    payload["training"]["epochs"] = 1
    payload["training"]["batch_size"] = 2
    path = tmp_path / "model1.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def small_stream_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(Path("configs/model1/stream_v1.yaml").read_text())
    payload["stream"].update(
        {
            "levels": ["W0"],
            "environment_count": 2,
            "episodes_per_environment": 2,
            "max_attempts_per_environment": 80,
            "replay_capacity": 4,
            "replay_batch_size": 2,
            "update_steps_per_environment": 1,
        }
    )
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_package_local_commands_generate_train_evaluate_and_inspect(tmp_path) -> None:
    config = small_config(tmp_path)
    data = tmp_path / "data"
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    inspection = tmp_path / "inspection"

    assert main(
        [
            "generate",
            "--config",
            str(config),
            "--family-config",
            "configs/world_families/v1.yaml",
            "--seed",
            "7",
            "--output",
            str(data),
            "--quick",
        ]
    ) == 0
    assert main(
        ["train", "--config", str(config), "--data", str(data), "--output", str(training)]
    ) == 0
    assert main(
        [
            "evaluate",
            "--data",
            str(data),
            "--checkpoint",
            str(training / "best.pt"),
            "--output",
            str(evaluation),
        ]
    ) == 0
    assert (evaluation / "index.html").is_file()
    assert (evaluation / "calibration.png").stat().st_size > 0
    assert (evaluation / "failure-gallery.png").stat().st_size > 0
    data.rename(tmp_path / "data-moved")
    assert main(["inspect", "--run", str(evaluation), "--output", str(inspection)]) == 0
    assert (inspection / "index.html").is_file()
    assert json.loads((evaluation / "manifest.json").read_text())["deterministic_hash"]


def test_commands_refuse_overwrite_without_force(tmp_path) -> None:
    config = small_config(tmp_path)
    output = tmp_path / "data"
    arguments = [
        "generate",
        "--config",
        str(config),
        "--family-config",
        "configs/world_families/v1.yaml",
        "--output",
        str(output),
        "--quick",
    ]
    assert main(arguments) == 0
    assert main(arguments) == 2
    assert main([*arguments, "--force"]) == 0


def test_stream_train_command_writes_frozen_events_updates_and_checkpoint(tmp_path) -> None:
    output = tmp_path / "stream-run"
    arguments = [
        "stream-train",
        "--config",
        str(small_config(tmp_path)),
        "--stream-config",
        str(small_stream_config(tmp_path)),
        "--family-config",
        "configs/world_families/v1.yaml",
        "--seed",
        "31",
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    summary = json.loads((output / "stream-summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["environment_count"] == 2
    assert summary["event_count"] == 4
    assert [
        environment["recorded_weight_update_count"]
        for environment in summary["environments"]
    ] == [0, 1]
    assert len(list((output / "events").glob("*.npz"))) == 4
    assert (output / "history.jsonl").is_file()
    assert (output / "stream-last.pt").is_file()
    assert manifest["command"] == "stream-train"
    assert manifest["stream_summary"]["weight_update_count"] == 2
    assert main(arguments) == 2
