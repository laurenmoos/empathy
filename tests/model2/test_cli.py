import json

from empathy.model2.cli import main


def test_cli_rejects_existing_output_and_bad_config(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("unknown: true")
    assert main(["fixtures", "--config", str(config), "--output", str(tmp_path / "out")]) == 2
    config.write_text("{}")
    assert main(["fixtures", "--config", str(config), "--output", str(tmp_path)]) == 2
    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path / "nested"),
            ]
        )
        == 2
    )


def test_demo_trains_evaluates_controls_and_saves_proof(tmp_path):
    config = tmp_path / "tiny.yaml"
    config.write_text("""fixtures:
  world_count: 3
  levels: [W1]
  episodes_per_world: 1
  size: 16
  steps: 60
  max_context_attempts: 4
  world_attempt_multiplier: 100
  entry_resolution: [1, 1, 1, 1]
  audit_bank_size: 2
  audit_probe_count: 4
  heldout_probe_count: 5
  x_bounds: [0.15, 0.15]
  y_bounds: [0.65, 0.65]
  vx_bounds: [0.25, 0.25]
  vy_bounds: [0.0, 0.0]
training:
  epochs: 1
evaluation:
  search_seeds: [7]
  budget: 10
""")
    root = tmp_path / "demo"
    assert main(["demo", "--config", str(config), "--output", str(root), "--seed", "19"]) == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["software_checks_passed"]
    assert manifest["research_gate_passed"] == "not_established"
    assert manifest["training_seed"] == 19
    assert (root / "visuals" / "index.html").exists()
    assert set(manifest["control_metrics"]) == {"constant", "single_belief", "shuffled_history"}
    assert main(["demo", "--config", str(config), "--output", str(root)]) == 2


def test_cli_reports_malformed_yaml_as_input_error(tmp_path):
    config = tmp_path / "broken.yaml"
    config.write_text("fixtures: [\ntraining: {")
    assert main(["fixtures", "--config", str(config), "--output", str(tmp_path / "output")]) == 2


def test_cli_missing_torch_has_install_hint(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; sys.modules["torch"] = None; '
            "from empathy.model2.cli import main; "
            "raise SystemExit(main(sys.argv[1:]))",
            "train",
            "--config",
            "configs/model2/smoke_v1.yaml",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "uv sync --extra ml" in result.stderr
