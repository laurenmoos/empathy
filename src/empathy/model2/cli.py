"""Package-local entry point: python -m empathy.model2.cli."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import yaml


def _safe_output(output: Path, sources: Sequence[Path] = ()) -> None:
    destination = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    for source in sources:
        source = source.resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("source/output overlap is not permitted")


def build_demo(config_path: Path, output: Path, *, seed: int | None = None) -> dict:
    from .config import load_config
    from .controls import run_controls
    from .evaluate import evaluate_model
    from .fixtures import generate_fixtures
    from .train import train_model
    from .visualize import render_evaluation

    output = Path(output)
    _safe_output(output, (Path(config_path),))
    fixtures, training, evaluation = load_config(config_path)
    if seed is not None:
        training = replace(training, seed=seed)
    output.mkdir(parents=True)
    data = generate_fixtures(fixtures, output / "data")
    run = train_model(data.root, training, output / "training")
    result = evaluate_model(run.best_checkpoint, data.root, evaluation, output / "evaluation")
    controls = run_controls(data.root, training, output / "controls")
    control_metrics = {}
    for name, checkpoint in controls.items():
        control_metrics[name] = evaluate_model(
            checkpoint, data.root, evaluation, output / f"evaluation-{name}"
        ).metrics
    (result.root / "controls.json").write_text(
        json.dumps({"full": result.metrics, **control_metrics}, indent=2, sort_keys=True)
    )
    artifacts = render_evaluation(data.root, result.root, output / "visuals")
    history = json.loads(run.metrics_path.read_text())
    train_manifest = json.loads(run.manifest_path.read_text())
    eval_manifest = json.loads(result.manifest_path.read_text())
    manifest = {
        "software_checks_passed": train_manifest["initial_state_hash"]
        != train_manifest["final_state_hash"],
        "research_gate_passed": "not_established",
        "research_gate_reason": "single training seed; three registered seeds and leakage evidence "
        "are required for the paired world-bootstrap gate",
        "training_seed": training.seed,
        "fixture_seed": fixtures.seed,
        "config": json.loads(
            json.dumps(
                {
                    "fixtures": data.dataset_hash,
                    "training": train_manifest["config"],
                    "evaluation": eval_manifest["config"],
                }
            )
        ),
        "dataset_hash": data.dataset_hash,
        "prediction_hash": eval_manifest["array_hash"],
        "initial_state_hash": train_manifest["initial_state_hash"],
        "final_state_hash": train_manifest["final_state_hash"],
        "initial_train_loss": history["initial_train_loss"],
        "final_train_loss": history["epochs"][-1]["train_total"],
        "metrics": result.metrics,
        "control_metrics": control_metrics,
        "artifact_hashes": {
            str(a.path.relative_to(output)): hashlib.sha256(a.path.read_bytes()).hexdigest()
            for a in artifacts
        },
        "evidence_paths": [
            "data/manifest.json",
            "training/manifest.json",
            "evaluation/manifest.json",
            "visuals/index.html",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Model 2 experiment</h1>'
        "<p>Oracle-bootstrap beliefs. Research gate: not established.</p>"
        '<a href="visuals/index.html">Saved visual proof</a><br>'
        '<a href="manifest.json">Metrics, controls, provenance and hashes</a>'
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="empathy model2")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fixtures", "train", "evaluate", "controls", "inspect", "demo"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name != "inspect":
            command.add_argument("--config", type=Path, required=True)
        if name in ("train", "evaluate", "controls", "inspect"):
            command.add_argument("--data", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
        if name == "inspect":
            command.add_argument("--evaluation", type=Path, required=True)
        if name in ("train", "controls", "demo"):
            command.add_argument("--seed", type=int)
        if name == "controls":
            command.add_argument("--include-first-order-ablation", action="store_true")
    args = parser.parse_args(argv)
    try:
        sources = [
            getattr(args, name)
            for name in ("data", "checkpoint", "evaluation", "config")
            if hasattr(args, name)
        ]
        _safe_output(args.output, sources)
        if args.command == "demo":
            build_demo(args.config, args.output, seed=args.seed)
            return 0
        if args.command == "inspect":
            from .visualize import render_evaluation

            render_evaluation(args.data, args.evaluation, args.output)
            return 0
        from .config import load_config

        fixtures, training, evaluation = load_config(args.config)
        if getattr(args, "seed", None) is not None:
            training = replace(training, seed=args.seed)
        if args.command == "fixtures":
            from .fixtures import generate_fixtures

            generate_fixtures(fixtures, args.output)
        elif args.command == "train":
            from .train import train_model

            train_model(args.data, training, args.output)
        elif args.command == "evaluate":
            from .evaluate import evaluate_model

            evaluate_model(args.checkpoint, args.data, evaluation, args.output)
        elif args.command == "controls":
            from .controls import run_controls

            run_controls(args.data, training, args.output)
            if args.include_first_order_ablation:
                from .first_order_ablation import train_first_order_control

                train_first_order_control(args.data, training, args.output / "privileged")
        return 0
    except ImportError as error:
        if error.name == "torch" or (error.name and error.name.startswith("torch.")):
            print("Model 2 training requires PyTorch: run uv sync --extra ml", file=sys.stderr)
            return 2
        raise
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"Model 2: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
