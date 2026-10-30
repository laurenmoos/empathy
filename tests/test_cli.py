import json
import subprocess
from pathlib import Path

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
    commands = ["sample", "inspect", "estimate-shattering", "demo"]
    assert all(command in result.stdout for command in commands)
    assert result.stderr == ""


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

