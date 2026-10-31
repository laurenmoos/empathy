"""Package-local reproducible command API for the Model 1 research demonstration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception_env.beliefs import BeliefEvent, build_belief_history
from perception_env.families import load_family
from perception_env.records import Model2InputRecord, belief_entropy

from .config import load_model1_config
from .data import Model1Dataset, generate_model1_dataset
from .evaluate import (
    evaluate_model1,
    load_evaluation_summary,
    load_prediction_record,
)
from .stream_train import load_stream_training_config, train_model1_stream
from .train import train_model1
from .visualize import (
    render_calibration_plot,
    render_failure_gallery,
    render_prediction_panel,
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    versions = {"torch": torch.__version__, "numpy": np.__version__}
    for name in ("perception-env", "matplotlib", "pillow", "pyyaml"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _prepare_output(path: Path, force: bool) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not force:
        raise FileExistsError(path)
    resolved = path.resolve()
    if resolved in {Path.cwd().resolve(), Path("/").resolve()}:
        raise ValueError("refusing to replace a broad output directory")
    shutil.rmtree(path)


def _validate_separate(source: Path, output: Path) -> None:
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output paths overlap")


def _write_manifest(
    output: Path,
    *,
    command: str,
    arguments: dict[str, Any],
    seed: int,
    stable: dict[str, Any],
    output_paths: list[str],
    started: float,
) -> dict[str, Any]:
    manifest = {
        "command": command,
        "arguments": arguments,
        "seed": seed,
        "schema_versions": {"model1": "model1-v1", "world": "v1"},
        "package_versions": _versions(),
        "output_paths": sorted(output_paths),
        "output_content_hashes": {
            name: _hash_file(output / name)
            for name in sorted(output_paths)
            if (output / name).is_file()
        },
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        **stable,
    }
    deterministic = {
        key: value
        for key, value in manifest.items()
        if key not in {"wall_clock_duration_seconds", "output_content_hashes"}
    }
    manifest["deterministic_hash"] = _semantic_hash(deterministic)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _write_index(output: Path, images: list[Path], title: str) -> Path:
    figures = "\n".join(
        f'<figure><img src="{image.name}" alt="{image.stem}">'
        f"<figcaption>{image.stem}</figcaption></figure>"
        for image in images
    )
    path = output / "index.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{title}</title><h1>{title}</h1>"
        "<p>Rendered only from saved Model 1 prediction records.</p>"
        f"<main>{figures}</main>"
    )
    return path


def _render_saved_evaluation(run: Path, output: Path) -> list[Path]:
    summary = load_evaluation_summary(run)
    records = [load_prediction_record(path) for path in summary.record_paths]
    images = []
    for record in records[: min(4, len(records))]:
        images.append(
            render_prediction_panel(record, output / f"prediction-{record.example_id}.png").path
        )
    images.append(render_calibration_plot(summary, output / "calibration.png").path)
    images.append(render_failure_gallery(records, output / "failure-gallery.png").path)
    _write_index(output, images, "Model 1 belief evaluation")
    return images


def generate_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    _prepare_output(output, args.force)
    config = load_model1_config(Path(args.config))
    generated = generate_model1_dataset(
        config,
        Path(args.family_config),
        output,
        seed=args.seed,
        quick=args.quick,
    )
    original = json.loads((output / "manifest.json").read_text())
    return _write_manifest(
        output,
        command="generate",
        arguments={"quick": args.quick},
        seed=args.seed,
        stable={
            "config_hash": _hash_file(Path(args.config)),
            "family_config_hash": _hash_file(Path(args.family_config)),
            "split_ids": {key: list(value) for key, value in generated.split_ids.items()},
            "rejection_counts": generated.rejection_counts,
            "dataset_manifest": original,
        },
        output_paths=["manifest.json"],
        started=started,
    )


def train_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    _validate_separate(Path(args.data), output)
    _prepare_output(output, args.force)
    config = load_model1_config(Path(args.config))
    config = replace(config, training=replace(config.training, seed=args.seed))
    result = train_model1(
        config,
        Model1Dataset(Path(args.data), "train"),
        Model1Dataset(Path(args.data), "validation"),
        output,
        device=torch.device(args.device),
        seed=args.seed,
    )
    return _write_manifest(
        output,
        command="train",
        arguments={"device": args.device},
        seed=args.seed,
        stable={
            "config_hash": _hash_file(Path(args.config)),
            "dataset_hash": _hash_file(Path(args.data) / "manifest.json"),
            "history_hash": _hash_file(output / "history.jsonl"),
        },
        output_paths=[result.best_checkpoint.name, result.last_checkpoint.name, "history.jsonl"],
        started=started,
    )


def stream_train_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    config = load_model1_config(Path(args.config))
    config = replace(config, training=replace(config.training, seed=args.seed))
    stream_config = load_stream_training_config(Path(args.stream_config))
    family_config = Path(args.family_config)
    for level in set(stream_config.levels):
        load_family(family_config, level)
    for source in (Path(args.config), Path(args.stream_config), family_config):
        _validate_separate(source, output)
    if output.exists() and not args.force:
        raise FileExistsError(output)
    if output.resolve() in {Path.cwd().resolve(), Path("/").resolve()}:
        raise ValueError("refusing to replace a broad output directory")
    if args.quick:
        episodes = min(stream_config.episodes_per_environment, 2)
        replay_batch = min(stream_config.replay_batch_size, episodes)
        stream_config = replace(
            stream_config,
            environment_count=min(stream_config.environment_count, 2),
            episodes_per_environment=episodes,
            replay_capacity=max(
                replay_batch, min(stream_config.replay_capacity, 8)
            ),
            replay_batch_size=replay_batch,
            update_steps_per_environment=1,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    staging.rmdir()
    try:
        result = train_model1_stream(
            config,
            stream_config,
            family_config,
            staging,
            device=torch.device(args.device),
            seed=args.seed,
        )
        event_paths = [str(event.path.relative_to(staging)) for event in result.events]
        manifest = _write_manifest(
            staging,
            command="stream-train",
            arguments={"device": args.device, "quick": args.quick},
            seed=args.seed,
            stable={
                "config_hash": _hash_file(Path(args.config)),
                "stream_config_hash": _hash_file(Path(args.stream_config)),
                "family_config_hash": _hash_file(family_config),
                "history_hash": _hash_file(staging / "history.jsonl"),
                "stream_summary": result.summary,
            },
            output_paths=[
                "stream-summary.json",
                "history.jsonl",
                result.checkpoint.name,
                *event_paths,
            ],
            started=started,
        )
        backup: Path | None = None
        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent)
            )
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def evaluate_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    _validate_separate(Path(args.data), output)
    _prepare_output(output, args.force)
    summary = evaluate_model1(
        Path(args.checkpoint),
        Model1Dataset(Path(args.data), "test"),
        output,
        device=torch.device(args.device),
        seed=args.seed,
    )
    images = _render_saved_evaluation(output, output)
    paths = ["summary.json", "index.html", *[image.name for image in images]]
    return _write_manifest(
        output,
        command="evaluate",
        arguments={"device": args.device},
        seed=args.seed,
        stable={
            "config_hash": summary.config_hash,
            "checkpoint_hash": summary.checkpoint_hash,
            "evaluation_metrics": summary.overall,
            "record_ids": [path.name for path in summary.record_paths],
        },
        output_paths=paths,
        started=started,
    )


def inspect_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    _validate_separate(Path(args.run), output)
    _prepare_output(output, args.force)
    output.mkdir(parents=True)
    images = _render_saved_evaluation(Path(args.run), output)
    return _write_manifest(
        output,
        command="inspect",
        arguments={},
        seed=load_evaluation_summary(Path(args.run)).seed,
        stable={"source_summary_hash": _hash_file(Path(args.run) / "summary.json")},
        output_paths=["index.html", *[image.name for image in images]],
        started=started,
    )


def _export_model2_fixture(data: Path, evaluation: Path, output: Path) -> str:
    summary = load_evaluation_summary(evaluation)
    prediction = load_prediction_record(summary.record_paths[0])
    events = [
        BeliefEvent(index, prediction.arrays["pre_belief"], prediction.arrays["post_belief"])
        for index in range(4)
    ]
    history = build_belief_history(events)
    source = Model1Dataset(data, "test")[0]
    arrays = {
        "pre_belief": history.tensor[:, 0],
        "post_belief": history.tensor[:, 1],
        "belief_difference": history.tensor[:, 2],
        "entropy": belief_entropy(history.tensor[:, 0]),
        "history_mask": history.mask,
        "target_world": source.world_flat,
    }
    checked = Model2InputRecord.from_mapping(arrays)
    arrays = {name: getattr(checked, name) for name in arrays}
    np.savez_compressed(output / "model2-input.npz", **arrays)
    return _array_hash(arrays)


def demo_run(args) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output)
    _prepare_output(output, args.force)
    output.mkdir(parents=True)
    config = load_model1_config(Path(args.config))
    if args.quick:
        config = replace(config, training=replace(config.training, epochs=2, seed=args.seed))
    data = output / "data"
    training = output / "training"
    evaluation = output / "evaluation"
    generated = generate_model1_dataset(
        config, Path(args.family_config), data, seed=args.seed, quick=args.quick
    )
    trained = train_model1(
        config,
        Model1Dataset(data, "train"),
        Model1Dataset(data, "validation"),
        training,
        device=torch.device(args.device),
        seed=args.seed,
    )
    summary = evaluate_model1(
        trained.best_checkpoint,
        Model1Dataset(data, "test"),
        evaluation,
        device=torch.device(args.device),
        seed=args.seed,
    )
    images = _render_saved_evaluation(evaluation, evaluation)
    model2_hash = _export_model2_fixture(data, evaluation, output)
    stable = {
        "config_hash": _hash_file(Path(args.config)),
        "family_config_hash": _hash_file(Path(args.family_config)),
        "split_ids": {key: list(value) for key, value in generated.split_ids.items()},
        "rejection_counts": generated.rejection_counts,
        "history": list(trained.history),
        "evaluation": {
            "overall": summary.overall,
            "per_family": summary.per_family,
            "calibration": summary.calibration,
            "config_hash": summary.config_hash,
        },
        "model2_input_hash": model2_hash,
    }
    return _write_manifest(
        output,
        command="demo",
        arguments={"quick": args.quick, "device": args.device},
        seed=args.seed,
        stable=stable,
        output_paths=[
            "model2-input.npz",
            "evaluation/index.html",
            *[f"evaluation/{image.name}" for image in images],
        ],
        started=started,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perception-model1")
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--family-config", required=True)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--output", required=True)
    generate.add_argument("--quick", action="store_true")
    generate.add_argument("--force", action="store_true")
    train = subcommands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", required=True)
    train.add_argument("--force", action="store_true")
    stream_train = subcommands.add_parser("stream-train")
    stream_train.add_argument("--config", required=True)
    stream_train.add_argument("--stream-config", required=True)
    stream_train.add_argument("--family-config", required=True)
    stream_train.add_argument("--seed", type=int, default=7)
    stream_train.add_argument("--device", default="cpu")
    stream_train.add_argument("--output", required=True)
    stream_train.add_argument("--quick", action="store_true")
    stream_train.add_argument("--force", action="store_true")
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--force", action="store_true")
    inspect = subcommands.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--force", action="store_true")
    demo = subcommands.add_parser("demo")
    demo.add_argument("--config", required=True)
    demo.add_argument("--family-config", required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--output", required=True)
    demo.add_argument("--quick", action="store_true")
    demo.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        {
            "generate": generate_run,
            "train": train_run,
            "stream-train": stream_train_run,
            "evaluate": evaluate_run,
            "inspect": inspect_run,
            "demo": demo_run,
        }[args.command](args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
