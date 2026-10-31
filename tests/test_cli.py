import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from perception_env.cli import _parser, inspect_run, main

ROOT = Path(__file__).parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
CONFIG = ROOT / "configs" / "world_families" / "v1.yaml"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-m", "perception_env.cli", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_all_research_commands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    commands = [
        "sample",
        "inspect",
        "estimate-shattering",
        "demo",
        "model1",
        "model2",
        "live-demo",
    ]
    assert all(command in result.stdout for command in commands)
    assert "stream-demo" not in result.stdout
    assert result.stderr == ""


def test_root_cli_delegates_model_help() -> None:
    assert run_cli("model1", "--help").returncode == 0
    assert run_cli("model2", "--help").returncode == 0


def test_live_demo_uses_one_tracked_checkpoint_by_default() -> None:
    args = _parser().parse_args(["live-demo"])

    assert args.model1_checkpoint == "checkpoints/model1.pt"
    assert (ROOT / args.model1_checkpoint).is_file()
    assert not hasattr(args, "model2_checkpoint")


def test_live_demo_delegates_only_the_model1_checkpoint(monkeypatch, tmp_path) -> None:
    from perception_env.live import server as live_server

    captured = {}

    def fake_run_live_server(*, host, port, root, model1_checkpoint) -> None:
        captured.update(
            host=host,
            port=port,
            root=root,
            model1_checkpoint=model1_checkpoint,
        )

    monkeypatch.setattr(live_server, "run_live_server", fake_run_live_server)

    result = main(
        [
            "live-demo",
            "--host",
            "127.0.0.2",
            "--port",
            "9001",
            "--root",
            str(tmp_path),
            "--model1-checkpoint",
            "custom.pt",
        ]
    )

    assert result == 0
    assert captured == {
        "host": "127.0.0.2",
        "port": 9001,
        "root": tmp_path,
        "model1_checkpoint": Path("custom.pt"),
    }


def test_missing_configuration_returns_clear_error(tmp_path) -> None:
    result = run_cli(
        "sample",
        "--config",
        str(tmp_path / "missing.yaml"),
        "--output",
        str(tmp_path / "run"),
    )

    assert result.returncode != 0
    assert "configuration not found" in result.stderr


def test_sample_manifests_have_deterministic_scientific_hash(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = ["sample", "--config", str(CONFIG), "--level", "W2", "--count", "4", "--seed", "9"]

    first_result = run_cli(*arguments, "--output", str(first))
    second_result = run_cli(*arguments, "--output", str(second))

    assert first_result.returncode == second_result.returncode == 0
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest["deterministic_hash"] == second_manifest["deterministic_hash"]
    assert first_manifest["seed"] == 9
    assert first_manifest["schema_versions"] == {"family": "v1", "world": "v1"}
    assert "wall_clock_duration_seconds" in first_manifest
    worlds_hash = hashlib.sha256((first / "worlds.json").read_bytes()).hexdigest()
    assert first_manifest["output_content_hashes"]["worlds.json"] == worlds_hash


def test_command_refuses_overwrite_without_force(tmp_path) -> None:
    output = tmp_path / "run"
    arguments = [
        "sample",
        "--config",
        str(CONFIG),
        "--level",
        "W1",
        "--count",
        "1",
        "--seed",
        "2",
        "--output",
        str(output),
    ]
    assert run_cli(*arguments).returncode == 0

    refused = run_cli(*arguments)
    replaced = run_cli(*arguments, "--force")

    assert refused.returncode != 0
    assert "already exists" in refused.stderr
    assert replaced.returncode == 0


def test_inspect_refuses_to_delete_its_source_run(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="overlap"):
        inspect_run(run=source, output=source, force=True)

    assert (source / "manifest.json").exists()

